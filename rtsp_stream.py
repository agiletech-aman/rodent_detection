"""Threaded RTSP reader that always exposes only the latest frame.

Why a thread: cv2.VideoCapture buffers decoded frames internally. If the
consumer (face detector) is slower than the camera, frames pile up and the
picture lags seconds behind reality. A dedicated thread reads continuously,
which drains the buffer, and overwrites a single "latest frame" slot. The
consumer always gets the most recent frame; older ones are simply dropped.
"""

from __future__ import annotations

import enum
import logging
import os
import random
import threading
import time
from typing import Any

import cv2
import numpy as np

from utils import mask_url, parse_device

logger = logging.getLogger(__name__)


def configure_ffmpeg_transport(transport: str = "tcp") -> None:
    """Set FFMPEG capture options for RTSP. Must be called before any VideoCapture is opened.

    The option is process-global (read by OpenCV at open time), so it is set
    once from main instead of per stream.
    """
    transport = transport.lower()
    if transport not in ("tcp", "udp"):
        raise ValueError(f"Unsupported RTSP transport: {transport}")
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{transport}"


def open_capture(url: str, open_timeout_ms: int = 8000, read_timeout_ms: int = 8000) -> cv2.VideoCapture:
    """Open an RTSP/HTTP/file URL with FFMPEG, or a local camera ("device:0" / "/dev/video0") with V4L2."""
    dev = parse_device(url)
    if dev is not None:
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if cap.isOpened():
            # Most USB / laptop webcams only reach their full frame rate in MJPG mode.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return cap
    params = [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, open_timeout_ms,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC, read_timeout_ms,
    ]
    return cv2.VideoCapture(url, cv2.CAP_FFMPEG, params)


class StreamState(str, enum.Enum):
    INIT = "init"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"      # gave up after max_retries
    STOPPED = "stopped"


class RTSPStream:
    """Reads one RTSP camera in a background thread with auto-reconnect."""

    def __init__(
        self,
        camera_id: str,
        url: str,
        *,
        open_timeout_ms: int = 8000,
        read_timeout_ms: int = 8000,
        max_consecutive_failures: int = 5,
        max_retries: int = 0,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
    ) -> None:
        self.camera_id = camera_id
        self.url = url
        self.safe_url = mask_url(url)
        self.open_timeout_ms = open_timeout_ms
        self.read_timeout_ms = read_timeout_ms
        self.max_consecutive_failures = max(1, max_consecutive_failures)
        self.max_retries = max_retries  # 0 = infinite
        self.base_delay = base_delay
        self.max_delay = max_delay

        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._frame_id: int = 0          # increments on every new frame
        self._frame_time: float = 0.0    # time.time() when the frame was grabbed
        self._state = StreamState.INIT
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"rtsp-{camera_id}", daemon=True)

    # ------------------------------------------------------------------ API

    @classmethod
    def from_config(cls, camera: dict[str, Any], rtsp_cfg: dict[str, Any]) -> "RTSPStream":
        rc = rtsp_cfg.get("reconnect", {})
        return cls(
            camera_id=str(camera["id"]),
            url=str(camera["url"]),
            open_timeout_ms=int(rtsp_cfg.get("open_timeout_ms", 8000)),
            read_timeout_ms=int(rtsp_cfg.get("read_timeout_ms", 8000)),
            max_consecutive_failures=int(rtsp_cfg.get("max_consecutive_failures", 5)),
            max_retries=int(rc.get("max_retries", 0)),
            base_delay=float(rc.get("base_delay", 1.0)),
            max_delay=float(rc.get("max_delay", 30.0)),
        )

    def start(self) -> "RTSPStream":
        logger.info("[%s] Starting stream %s", self.camera_id, self.safe_url)
        self._thread.start()
        return self

    def stop(self, timeout: float = 10.0) -> None:
        """Signal the reader thread to exit and wait for it (capture is released inside the thread)."""
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout)
            if self._thread.is_alive():
                logger.warning("[%s] Reader thread did not exit within %.1fs", self.camera_id, timeout)

    def read(self) -> tuple[int, np.ndarray | None, float]:
        """Return (frame_id, frame, grab_timestamp). frame is None until the first frame arrives.

        The returned frame is a private copy, so callers can draw on it freely.
        """
        with self._lock:
            if self._frame is None:
                return self._frame_id, None, self._frame_time
            return self._frame_id, self._frame.copy(), self._frame_time

    @property
    def state(self) -> StreamState:
        return self._state

    @property
    def failed(self) -> bool:
        return self._state == StreamState.FAILED

    @property
    def is_running(self) -> bool:
        return self._thread.is_alive()

    # ------------------------------------------------------------- internal

    def _open(self) -> cv2.VideoCapture | None:
        try:
            cap = open_capture(self.url, self.open_timeout_ms, self.read_timeout_ms)
        except cv2.error as exc:
            logger.error("[%s] VideoCapture error: %s", self.camera_id, exc)
            return None
        if not cap.isOpened():
            cap.release()
            return None
        # Keep OpenCV's own queue as small as possible (not honoured by every backend).
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _backoff_delay(self, attempt: int) -> float:
        delay = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)
        return delay * random.uniform(0.8, 1.2)  # jitter so many cameras don't reconnect in lockstep

    def _run(self) -> None:
        attempt = 0  # consecutive failed connection attempts
        try:
            while not self._stop_event.is_set():
                self._state = StreamState.CONNECTING if attempt == 0 else StreamState.RECONNECTING
                cap = self._open()

                got_frames = False
                if cap is not None:
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    logger.info("[%s] Connected (%dx%d @ %.1f fps)", self.camera_id, w, h, fps)
                    self._state = StreamState.CONNECTED
                    try:
                        got_frames = self._read_loop(cap)
                    finally:
                        cap.release()

                if self._stop_event.is_set():
                    break

                # Only a session that actually delivered frames resets the backoff;
                # otherwise a camera that accepts and immediately drops would spin.
                attempt = 0 if got_frames else attempt + 1
                if got_frames:
                    # Short pause even after a good session so a flapping camera can't hot-loop.
                    logger.warning("[%s] Stream dropped, reconnecting in %.1fs...", self.camera_id, self.base_delay)
                    if self._stop_event.wait(self.base_delay):
                        break
                    continue

                if self.max_retries and attempt > self.max_retries:
                    logger.error("[%s] Giving up after %d failed reconnect attempts", self.camera_id, self.max_retries)
                    self._state = StreamState.FAILED
                    return

                delay = self._backoff_delay(attempt)
                limit = self.max_retries or "inf"
                logger.warning(
                    "[%s] Cannot open %s (attempt %d/%s), retrying in %.1fs",
                    self.camera_id, self.safe_url, attempt, limit, delay,
                )
                if self._stop_event.wait(delay):  # interruptible sleep
                    break
        except Exception:  # never let the thread die silently
            logger.exception("[%s] Unexpected error in reader thread", self.camera_id)
            self._state = StreamState.FAILED
            return

        self._state = StreamState.STOPPED
        logger.info("[%s] Reader stopped", self.camera_id)

    def _read_loop(self, cap: cv2.VideoCapture) -> bool:
        """Read until stop or the stream drops. Returns True if at least one frame was read."""
        failures = 0
        got_frames = False
        while not self._stop_event.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                failures += 1
                if failures >= self.max_consecutive_failures:
                    return got_frames
                continue
            failures = 0
            got_frames = True
            with self._lock:
                self._frame = frame
                self._frame_id += 1
                self._frame_time = time.time()
        return got_frames
