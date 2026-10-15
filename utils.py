"""Shared helpers: config, logging, drawing, saving, JSON log and REST API posting."""

from __future__ import annotations

import copy
import json
import logging
import os
import queue
import re
import threading
import time
from collections import deque
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np
import requests
import yaml

if TYPE_CHECKING:
    from detector import Detection

logger = logging.getLogger(__name__)

# ======================================================================
# Config
# ======================================================================

DEFAULT_CONFIG: dict[str, Any] = {
    "cameras": [],
    "rtsp": {
        "transport": "tcp",
        "frame_skip": 1,
        "open_timeout_ms": 8000,
        "read_timeout_ms": 8000,
        "max_consecutive_failures": 5,
        "reconnect": {"max_retries": 0, "base_delay": 1.0, "max_delay": 30.0},
        "display_max_width": 1280,
    },
    "detector": {
        "model_name": "buffalo_l",
        "model_root": "~/.insightface",
        "det_size": [640, 640],
        "conf_threshold": 0.5,
        "min_face_size": 0,
        "device": "auto",
    },
    "recognition": {"enabled": False, "known_faces_dir": "./known_faces", "similarity_threshold": 0.4},
    "database": {"path": "./data/app.db", "persons_dir": "./data/persons"},
    "retention": {"days": 30},
    "dedup": {"enabled": True, "cooldown_sec": 30, "similarity": 0.45, "best_window_sec": 3.0},
    "quality": {"enabled": True, "min_face_size": 80, "min_confidence": 0.7, "min_blur_score": 60.0,
                "max_yaw": 35.0, "max_pitch": 30.0, "require_embedding": True},
    "output": {
        "base_dir": "./output",
        "save_crops": True,
        "save_annotated": True,
        "crop_margin": 0.25,
        "json_log": "./output/detections.jsonl",
        "jpeg_quality": 92,
        "event_interval_sec": 1.0,
    },
    "image": {
        "input_path": "./images",
        "recursive": True,
        "extensions": [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"],
    },
    "api": {
        "enabled": False,
        "url": "",
        "timeout": 5,
        "max_retries": 3,
        "backoff_factor": 0.5,
        "queue_size": 1000,
        "auth_token": "",
    },
    "web": {
        "host": "127.0.0.1",
        "port": 8080,
        "max_fps": 15,
        "jpeg_quality": 75,
        "max_width": 1280,
        "max_upload_mb": 15,
        "username": "",
        "password": "",
    },
    "logging": {
        "level": "INFO",
        "file": "./logs/face_detection.log",
        "max_bytes": 10 * 1024 * 1024,
        "backup_count": 5,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _expand_env(value: Any) -> Any:
    """Recursively expand ${VAR} in strings so secrets can live in the environment."""
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        user_cfg = yaml.safe_load(fh) or {}
    return _expand_env(_deep_merge(DEFAULT_CONFIG, user_cfg))


# ======================================================================
# Logging
# ======================================================================

def setup_logging(level: str = "INFO", log_file: str | None = None,
                  max_bytes: int = 10 * 1024 * 1024, backup_count: int = 5) -> None:
    """Console + rotating file logging."""
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(threadName)-12s | %(name)s | %(message)s")
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)

    # Third-party libraries are noisy at INFO.
    for noisy in ("urllib3", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ======================================================================
# Small helpers
# ======================================================================

def mask_url(url: str) -> str:
    """rtsp://user:secret@host/... -> rtsp://user:***@host/... (never log passwords)."""
    try:
        parts = urlsplit(url)
        if parts.password:
            netloc = f"{parts.username}:***@{parts.hostname}"
            if parts.port:
                netloc += f":{parts.port}"
            return urlunsplit(parts._replace(netloc=netloc))
    except ValueError:
        pass
    return url


def parse_device(url: str) -> int | None:
    """'device:0' or '/dev/video0' -> 0 for a local USB / laptop webcam, else None."""
    m = re.fullmatch(r"(?:device:|/dev/video)(\d{1,3})", (url or "").strip())
    return int(m.group(1)) if m else None


def list_video_devices() -> list[dict[str, Any]]:
    """Local V4L2 capture devices (Linux). Metadata nodes (index != 0) are skipped."""
    out: list[dict[str, Any]] = []
    root = Path("/sys/class/video4linux")
    nodes = [p for p in root.glob("video*") if p.name[5:].isdigit()] if root.is_dir() else []
    for node in sorted(nodes, key=lambda p: int(p.name[5:])):
        try:
            if (node / "index").read_text().strip() != "0":
                continue
            name = (node / "name").read_text().strip()
        except OSError:
            name = node.name
        n = int(node.name[5:])
        out.append({"index": n, "url": f"device:{n}", "path": f"/dev/video{n}", "name": name})
    return out


def unmask_url(new_url: str, old_url: str) -> str:
    """If new_url still carries the '***' placeholder from mask_url, put the stored password back."""
    try:
        new, old = urlsplit(new_url), urlsplit(old_url)
        if new.password == "***" and old.password is not None:
            host = new.netloc.rsplit("@", 1)[1]
            return urlunsplit(new._replace(netloc=f"{new.username}:{old.password}@{host}"))
    except (ValueError, IndexError):
        pass
    return new_url


def safe_name(text: str) -> str:
    """Make a string safe for use as a file / directory name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "unknown"


def now_timestamps(ts: float | None = None) -> tuple[str, str]:
    """(ISO-8601 with timezone for JSON, filesystem-safe string for file names) for ts (default: now)."""
    now = (datetime.fromtimestamp(ts) if ts is not None else datetime.now()).astimezone()
    return now.isoformat(timespec="milliseconds"), now.strftime("%Y%m%d_%H%M%S_%f")


def has_display() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


class FPSCounter:
    """Frames-per-second over a sliding window of recent ticks."""

    def __init__(self, window: int = 30) -> None:
        self._times: deque[float] = deque(maxlen=window)

    def tick(self) -> None:
        self._times.append(time.perf_counter())

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / span if span > 0 else 0.0


# ======================================================================
# Drawing
# ======================================================================

LANDMARK_COLORS = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 255), (0, 255, 255)]
COLOR_KNOWN = (0, 200, 0)
COLOR_UNKNOWN = (0, 140, 255)
COLOR_DEFAULT = (0, 255, 0)
COLOR_REJECTED = (150, 150, 150)  # face seen, but not good enough to store (blur / small / pose / low_conf)


def _put_label(img: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int], scale: float = 0.5) -> None:
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    y = max(y, th + base + 2)
    cv2.rectangle(img, (x, y - th - base - 4), (x + tw + 4, y), color, -1)
    cv2.putText(img, text, (x + 2, y - base - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)


def draw_detections(img: np.ndarray, detections: list["Detection"], landmarks: bool = False) -> np.ndarray:
    """Draw boxes with confidence (and name) in place. The 5 landmarks are only drawn when
    landmarks=True; they are always kept in the JSON / database records."""
    thickness = max(1, round(min(img.shape[:2]) / 400))
    for d in detections:
        x1, y1, x2, y2 = d.bbox
        if d.reject:
            color, label = COLOR_REJECTED, f"{d.name if d.name and d.name != 'Unknown' else 'skip'}: {d.reject}"
        elif d.name is None:
            color, label = COLOR_DEFAULT, f"face {d.confidence:.2f}"
        elif d.name == "Unknown":
            color, label = COLOR_UNKNOWN, f"Unknown {d.confidence:.2f}"
        else:
            color, label = COLOR_KNOWN, f"{d.name} ({d.similarity:.2f}) {d.confidence:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness if d.reject else thickness + 1)
        _put_label(img, label, x1, y1, color)
        if landmarks:
            for i, (lx, ly) in enumerate(d.landmarks[:5]):
                cv2.circle(img, (int(lx), int(ly)), thickness + 1, LANDMARK_COLORS[i], -1, cv2.LINE_AA)
    return img


def draw_info(img: np.ndarray, lines: list[str]) -> np.ndarray:
    """Top-left info panel (FPS, camera id, etc.)."""
    y = 22
    for line in lines:
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24
    return img


def resize_for_display(img: np.ndarray, max_width: int = 1280, max_height: int = 0) -> np.ndarray:
    h, w = img.shape[:2]
    scale = 1.0
    if max_width and w > max_width:
        scale = max_width / w
    if max_height and h * scale > max_height:
        scale = max_height / h
    if scale >= 1.0:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


# ======================================================================
# Saving
# ======================================================================

def crop_face(img: np.ndarray, bbox: tuple[int, int, int, int], margin: float = 0.2) -> np.ndarray:
    """Crop a face with extra margin around it, clipped to the image."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    mx, my = int((x2 - x1) * margin), int((y2 - y1) * margin)
    x1, y1 = max(0, x1 - mx), max(0, y1 - my)
    x2, y2 = min(w, x2 + mx), min(h, y2 + my)
    return img[y1:y2, x1:x2]


def save_image(path: Path, img: np.ndarray, quality: int = 92) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            logger.error("cv2.imwrite failed for %s", path)
        return ok
    except Exception:
        logger.exception("Failed to save image %s", path)
        return False


class JsonlLogger:
    """Thread-safe append-only JSON Lines writer (one detection per line)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()


# ======================================================================
# REST API
# ======================================================================

def post_json_with_retry(
    url: str,
    payload: dict[str, Any],
    *,
    session: requests.Session | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 5.0,
    max_retries: int = 3,
    backoff_factor: float = 0.5,
    stop_event: threading.Event | None = None,
) -> bool:
    """POST JSON with exponential-backoff retries.

    Retries on connection errors, timeouts, 429 and 5xx. Other 4xx responses
    are treated as permanent errors (retrying a bad request won't help).
    """
    sess = session or requests.Session()
    for attempt in range(max_retries + 1):
        try:
            resp = sess.post(url, json=payload, headers=headers, timeout=timeout)
            if resp.status_code < 300:
                return True
            if resp.status_code != 429 and resp.status_code < 500:
                logger.error("API rejected payload: HTTP %d %s", resp.status_code, resp.text[:200])
                return False
            reason = f"HTTP {resp.status_code}"
        except requests.RequestException as exc:
            reason = type(exc).__name__
        if attempt < max_retries:
            delay = backoff_factor * (2 ** attempt)
            logger.warning("API POST failed (%s), retry %d/%d in %.1fs", reason, attempt + 1, max_retries, delay)
            if stop_event is not None:
                if stop_event.wait(delay):
                    break
            else:
                time.sleep(delay)
        else:
            logger.error("API POST failed after %d retries (%s)", max_retries, reason)
    return False


class ApiPoster:
    """Posts detection payloads from a background thread so the video loop never blocks on the network."""

    def __init__(self, url: str, auth_token: str = "", timeout: float = 5.0,
                 max_retries: int = 3, backoff_factor: float = 0.5, queue_size: int = 1000) -> None:
        self.url = url
        self.headers = {"Authorization": f"Bearer {auth_token}"} if auth_token and "${" not in auth_token else {}
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=queue_size)
        self._session = requests.Session()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, name="api-poster", daemon=True)
        self._thread.start()
        logger.info("API posting enabled -> %s", url)

    def submit(self, payload: dict[str, Any]) -> None:
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            try:  # drop the oldest payload to keep data fresh
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(payload)
            logger.warning("API queue full, dropped oldest payload")

    def _worker(self) -> None:
        while True:
            payload = self._queue.get()
            if payload is None:
                break
            post_json_with_retry(
                self.url, payload, session=self._session, headers=self.headers, timeout=self.timeout,
                max_retries=self.max_retries, backoff_factor=self.backoff_factor, stop_event=self._stop,
            )

    def close(self, timeout: float = 10.0) -> None:
        """Flush pending posts (up to timeout), then stop."""
        try:
            self._queue.put(None, timeout=1.0)
        except queue.Full:
            self._stop.set()
        self._thread.join(timeout)
        if self._thread.is_alive():
            self._stop.set()  # abort retry sleeps
            logger.warning("API poster still busy at shutdown; %d payload(s) not sent", self._queue.qsize())
        self._session.close()


# ======================================================================
# Output manager: crops + annotated + JSON log + API in one place
# ======================================================================

class OutputManager:
    """Handles everything that happens after a frame has detections.

    on_event(records, detections) is called for every stored event (used to write the detection
    history, including each face's embedding, into the database). Settings can be changed at runtime via apply_settings().
    """

    def __init__(self, output_cfg: dict[str, Any], api_cfg: dict[str, Any], save_enabled: bool = True,
                 on_event: Callable[[list[dict[str, Any]], list["Detection"]], None] | None = None) -> None:
        self.base_dir = Path(output_cfg["base_dir"])
        self.save_enabled = save_enabled
        self.save_crops = save_enabled and bool(output_cfg["save_crops"])
        self.save_annotated = save_enabled and bool(output_cfg["save_annotated"])
        self.crop_margin = float(output_cfg["crop_margin"])
        self.jpeg_quality = int(output_cfg["jpeg_quality"])
        self.event_interval = float(output_cfg["event_interval_sec"])
        self.json_logger = JsonlLogger(output_cfg["json_log"]) if output_cfg.get("json_log") else None
        self.on_event = on_event
        self._api_cfg = dict(api_cfg)
        self.api: ApiPoster | None = None
        self._api_key: tuple[str, str] | None = None
        self._last_event: dict[str, float] = {}
        self._lock = threading.Lock()
        self._configure_api(bool(api_cfg.get("enabled")), str(api_cfg.get("url") or ""),
                            str(api_cfg.get("auth_token") or ""))

    def _configure_api(self, enabled: bool, url: str, token: str) -> None:
        key = (url, token) if enabled and url else None
        if key == self._api_key:
            return
        old, self.api = self.api, None
        if key is not None:
            c = self._api_cfg
            self.api = ApiPoster(url, auth_token=token, timeout=float(c.get("timeout", 5)),
                                 max_retries=int(c.get("max_retries", 3)),
                                 backoff_factor=float(c.get("backoff_factor", 0.5)),
                                 queue_size=int(c.get("queue_size", 1000)))
        self._api_key = key
        if old is not None:
            threading.Thread(target=old.close, name="api-close", daemon=True).start()
            if key is None:
                logger.info("API posting disabled")

    def apply_settings(self, s: dict[str, Any]) -> None:
        """Apply runtime settings (keys as in database.SETTINGS_SCHEMA)."""
        with self._lock:
            self.save_crops = self.save_enabled and bool(s.get("output.save_crops", self.save_crops))
            self.save_annotated = self.save_enabled and bool(s.get("output.save_annotated", self.save_annotated))
            self.event_interval = float(s.get("output.event_interval_sec", self.event_interval))
            self.crop_margin = float(s.get("output.crop_margin", self.crop_margin))
            self._configure_api(bool(s.get("api.enabled")), str(s.get("api.url") or ""),
                                str(s.get("api.auth_token") or ""))

    def _event_due(self, camera_id: str) -> bool:
        """Per-camera rate limit for crops / annotated frames / API posts."""
        if self.event_interval <= 0:
            return True
        now = time.monotonic()
        with self._lock:
            if now - self._last_event.get(camera_id, -1e9) >= self.event_interval:
                self._last_event[camera_id] = now
                return True
        return False

    def handle(
        self,
        camera_id: str,
        frame: np.ndarray,
        detections: list["Detection"],
        *,
        frame_id: int | None = None,
        source: str | None = None,
        annotated: np.ndarray | None = None,
        rate_limit: bool = True,
        captured_at: float | None = None,
    ) -> list[dict[str, Any]]:
        """Save crops / annotated image, write JSON log and queue the API post. Returns the JSON records.

        captured_at: when the frame was grabbed (the tracker stores a track's best crop a few seconds later).
        """
        if not detections:
            return []
        iso_ts, file_ts = now_timestamps(captured_at)
        cam_dir = safe_name(camera_id)
        prefix = f"{safe_name(Path(source).stem)}_" if source else ""
        is_event = not rate_limit or self._event_due(camera_id)

        annotated_path: str | None = None
        if is_event and self.save_annotated:
            if annotated is None:
                annotated = draw_detections(frame.copy(), detections)
            path = self.base_dir / "annotated" / cam_dir / f"{prefix}{file_ts}.jpg"
            if save_image(path, annotated, self.jpeg_quality):
                annotated_path = str(path)

        records: list[dict[str, Any]] = []
        for i, det in enumerate(detections):
            crop_path: str | None = None
            if is_event and self.save_crops:
                crop = crop_face(frame, det.bbox, self.crop_margin)
                if crop.size:
                    path = self.base_dir / "faces" / cam_dir / f"{prefix}{file_ts}_f{i}.jpg"
                    if save_image(path, crop, self.jpeg_quality):
                        crop_path = str(path)
            record: dict[str, Any] = {
                "camera_id": camera_id,
                "timestamp": iso_ts,
                "face_index": i,
                **det.to_dict(),
            }
            if frame_id is not None:
                record["frame_id"] = frame_id
            if source:
                record["source"] = source
            if crop_path:
                record["crop_path"] = crop_path
            if annotated_path:
                record["annotated_path"] = annotated_path
            records.append(record)
            if self.json_logger:
                self.json_logger.write(record)

        if is_event and self.on_event is not None:
            try:
                self.on_event(records, detections)
            except Exception:
                logger.exception("on_event handler failed")

        api = self.api
        if is_event and api is not None:
            api.submit({
                "camera_id": camera_id,
                "timestamp": iso_ts,
                "frame_size": [int(frame.shape[1]), int(frame.shape[0])],
                "faces": records,
            })
        return records

    def close(self) -> None:
        if self.api is not None:
            self.api.close()
        if self.json_logger is not None:
            self.json_logger.close()
