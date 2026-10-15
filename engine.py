"""Detection engine: runs all cameras from the database and reacts to changes made in the web UI.

- LiveHub          latest annotated frame + status per camera (read by the web UI)
- DetectionEngine  starts / stops / restarts RTSP readers when cameras change, runs detection
- RetentionWorker  deletes old detection history and its images
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any

import cv2
import numpy as np

from autolearn import AutoLearner
from database import Database
from dedup import FaceTracker, IdentityCache
from quality import QualityConfig, assess, describe_reject
from detector import Detection, FaceDetector
from rodent import RodentDetection, RodentDetector, RodentWatcher, draw_rodents
from enrollment import remove_files
from rtsp_stream import RTSPStream
from utils import FPSCounter, OutputManager, draw_detections, draw_info, resize_for_display

logger = logging.getLogger(__name__)


# ======================================================================
# Live frames for the web UI
# ======================================================================

class LiveHub:
    """Latest annotated frame + status per camera. Frames are JPEG-encoded lazily (only when watched)."""

    def __init__(self, max_fps: float = 15.0, jpeg_quality: int = 75, max_width: int = 1280) -> None:
        self.frame_interval = 1.0 / max(1.0, max_fps)
        self.jpeg_quality = jpeg_quality
        self.max_width = max_width
        self._lock = threading.Lock()
        self._order: list[str] = []
        self._frames: dict[str, tuple[int, np.ndarray]] = {}
        self._jpeg_cache: dict[str, tuple[int, bytes]] = {}
        self._status: dict[str, dict[str, Any]] = {}
        self._stop = threading.Event()

    def register(self, camera_id: str, state: str = "init") -> None:
        with self._lock:
            if camera_id not in self._order:
                self._order.append(camera_id)
            self._status[camera_id] = {"state": state}

    def unregister(self, camera_id: str) -> None:
        with self._lock:
            if camera_id in self._order:
                self._order.remove(camera_id)
            self._frames.pop(camera_id, None)
            self._jpeg_cache.pop(camera_id, None)
            self._status.pop(camera_id, None)

    def update_frame(self, camera_id: str, frame: np.ndarray) -> None:
        with self._lock:
            if camera_id not in self._order:
                self._order.append(camera_id)
            seq = self._frames.get(camera_id, (0, None))[0] + 1
            self._frames[camera_id] = (seq, frame)

    def update_status(self, camera_id: str, **fields: Any) -> None:
        with self._lock:
            self._status.setdefault(camera_id, {}).update(fields, updated=time.time())

    def status(self) -> list[dict[str, Any]]:
        with self._lock:
            return [{"id": c, "has_frame": c in self._frames, **self._status.get(c, {})} for c in self._order]

    def status_of(self, camera_id: str) -> dict[str, Any]:
        with self._lock:
            return dict(self._status.get(camera_id, {}))

    def get_jpeg(self, camera_id: str) -> tuple[int, bytes | None]:
        with self._lock:
            seq, frame = self._frames.get(camera_id, (0, None))
            cached = self._jpeg_cache.get(camera_id)
        if frame is None:
            return 0, None
        if cached and cached[0] == seq:
            return cached
        ok, buf = cv2.imencode(".jpg", resize_for_display(frame, self.max_width),
                               [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            return 0, None
        with self._lock:
            self._jpeg_cache[camera_id] = (seq, buf.tobytes())
        return seq, buf.tobytes()

    def placeholder_jpeg(self, camera_id: str) -> bytes:
        state = self.status_of(camera_id).get("state", "waiting")
        img = np.zeros((360, 640, 3), np.uint8)
        cv2.putText(img, camera_id, (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2, cv2.LINE_AA)
        cv2.putText(img, f"{state}...", (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 170, 255), 2, cv2.LINE_AA)
        return cv2.imencode(".jpg", img)[1].tobytes()

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    def stop(self) -> None:
        self._stop.set()


# ======================================================================
# Runtime settings
# ======================================================================

def apply_runtime_settings(settings: dict[str, Any], detector: FaceDetector, output: OutputManager) -> None:
    """Push database settings (see database.SETTINGS_SCHEMA) into the detector and output manager."""
    detector.update_settings(
        conf_threshold=settings.get("detector.conf_threshold"),
        min_face_size=settings.get("detector.min_face_size"),
        similarity_threshold=settings.get("recognition.similarity_threshold"),
        recognition_enabled=settings.get("recognition.enabled"),
    )
    output.apply_settings(settings)


# ======================================================================
# Rodent worker
# ======================================================================

class RodentWorker(threading.Thread):
    """Runs the rodent model off the engine thread, so the live view and face detection never wait
    for it. Only the newest submitted frame per camera is kept; cameras are served round-robin."""

    def __init__(self, detector: RodentDetector, watcher: RodentWatcher) -> None:
        super().__init__(name="rodent", daemon=True)
        self.detector = detector
        self.watcher = watcher
        self._jobs: dict[str, tuple[int, np.ndarray, "CameraState"]] = {}
        self._cv = threading.Condition()
        self._stopping = False

    def submit(self, cid: str, frame_id: int, frame: np.ndarray, st: "CameraState") -> None:
        with self._cv:
            self._jobs[cid] = (frame_id, frame, st)  # replaces an older frame still waiting
            self._cv.notify()

    def drop(self, cid: str) -> None:
        with self._cv:
            self._jobs.pop(cid, None)

    def stop(self) -> None:
        with self._cv:
            self._stopping = True
            self._cv.notify()

    def run(self) -> None:
        while True:
            with self._cv:
                while not self._jobs and not self._stopping:
                    self._cv.wait()
                if self._stopping:
                    return
                cid = next(iter(self._jobs))
                frame_id, frame, st = self._jobs.pop(cid)
            try:
                t0 = time.perf_counter()
                dets = self.detector.detect(frame)
                st.rodent_ms = (time.perf_counter() - t0) * 1000
                st.rodents = dets
                st.rodent_confirmed, alert = self.watcher.update(cid, frame, dets, frame_id)
                if alert is not None:
                    st.rodent_alerts += 1
            except Exception:
                logger.exception("[%s] Rodent detection failed", cid)


# ======================================================================
# Engine
# ======================================================================

@dataclass
class CameraState:
    last_frame_id: int = 0
    frames_seen: int = 0
    detections: list[Detection] = field(default_factory=list)  # last result, reused on skipped frames
    infer_ms: float = 0.0
    display_fps: FPSCounter = field(default_factory=FPSCounter)
    det_fps: FPSCounter = field(default_factory=FPSCounter)
    saved: int = 0                                                   # crops stored since start
    rejects: dict[str, int] = field(default_factory=dict)            # reason -> count since start
    rodents: list[RodentDetection] = field(default_factory=list)     # last rodent result, reused on skipped frames
    rodent_confirmed: bool = False                                   # seen in enough recent runs (alert state)
    rodent_ms: float = 0.0
    rodent_alerts: int = 0                                           # alerts stored since start


class DetectionEngine:
    """Runs detection for all enabled cameras in the database.

    Call request_reload() after cameras or settings change; the loop picks the change up
    on its next iteration (streams are only touched from the engine thread).
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        db: Database,
        detector: FaceDetector,
        output: OutputManager,
        hub: LiveHub | None = None,
        *,
        show: bool = False,
        keep_alive: bool = False,
        camera_filter: list[str] | None = None,
        overrides: dict[str, Any] | None = None,
        rodent: RodentDetector | None = None,
    ) -> None:
        self.cfg = cfg
        self.db = db
        self.detector = detector
        self.output = output
        self.hub = hub
        self.show = show
        self.keep_alive = keep_alive          # web mode: keep running with zero / failed cameras
        self.camera_filter = set(camera_filter or [])
        self.overrides = overrides or {}      # CLI flags win over database settings (not persisted)
        self.display_width = int(cfg["rtsp"].get("display_max_width", 1280))
        self.settings: dict[str, Any] = {}
        self.tracker = FaceTracker(detector)
        self.quality = QualityConfig()
        self.identities = IdentityCache(detector)
        self.learner = AutoLearner(db, detector, cfg["database"]["persons_dir"])
        self.rodent = rodent                  # None = no rodent model (feature unavailable)
        self.rodent_watcher = RodentWatcher(db, cfg["output"]["base_dir"], int(cfg["output"].get("jpeg_quality", 92)))
        self.rodent_worker = RodentWorker(rodent, self.rodent_watcher) if rodent is not None else None

        self.streams: dict[str, RTSPStream] = {}
        self.cam_rows: dict[str, dict[str, Any]] = {}
        self.states: dict[str, CameraState] = {}
        self._reload = threading.Event()
        self._reload.set()  # load everything on the first iteration
        self.paused = False   # set from the web UI: streams keep running, no detection / saving
        self.raw_frames: dict[str, np.ndarray] = {}  # latest frame per camera without overlays (training capture)

    def request_reload(self) -> None:
        self._reload.set()

    # ------------------------------------------------------------ reloading

    def _apply_settings(self) -> None:
        self.settings = {**self.db.get_settings(), **self.overrides}
        apply_runtime_settings(self.settings, self.detector, self.output)
        self.quality = QualityConfig.from_settings(self.settings)
        self.learner.apply_settings(self.settings)
        self.tracker.update_settings(self.settings.get("dedup.cooldown_sec"), self.settings.get("dedup.similarity"),
                                     self.settings.get("dedup.best_window_sec"), self.quality.require_embedding)
        if not self.settings.get("dedup.enabled", True):
            self.tracker.reset()
        if self.rodent is not None:
            self.rodent.update_settings(conf_threshold=self.settings.get("rodent.conf_threshold"))
        self.rodent_watcher.apply_settings(self.settings)
        self.rodent_watcher.save_enabled = self.output.save_enabled

    def _sync_cameras(self) -> None:
        wanted = {c["id"]: c for c in self.db.list_cameras() if c["enabled"]}
        if self.camera_filter:
            wanted = {k: v for k, v in wanted.items() if k in self.camera_filter}

        for cid in list(self.streams):
            row = wanted.get(cid)
            if row is None:
                logger.info("[%s] Camera removed or disabled", cid)
                self._stop_stream(cid)
            elif row["url"] != self.cam_rows[cid]["url"]:
                logger.info("[%s] Camera URL changed, restarting stream", cid)
                self._stop_stream(cid)
            elif self.streams[cid].failed:
                logger.info("[%s] Retrying failed camera after update", cid)
                self._stop_stream(cid)

        for cid, row in wanted.items():
            self.cam_rows[cid] = row  # picks up name / frame_skip changes
            if cid not in self.streams:
                stream = RTSPStream.from_config({"id": cid, "url": row["url"]}, self.cfg["rtsp"])
                self.streams[cid] = stream.start()
                self.states[cid] = CameraState()
                if self.hub is not None:
                    self.hub.register(cid, stream.state.value)

    def _stop_stream(self, cid: str) -> None:
        stream = self.streams.pop(cid)
        self.tracker.reset(cid)
        self.identities.reset(cid)
        self.rodent_watcher.reset(cid)
        if self.rodent_worker is not None:
            self.rodent_worker.drop(cid)
        self.cam_rows.pop(cid, None)
        self.states.pop(cid, None)
        self.raw_frames.pop(cid, None)
        if self.hub is not None:
            self.hub.unregister(cid)
        # Joining can take up to the read timeout: don't block the detection loop.
        threading.Thread(target=stream.stop, name=f"stop-{cid}", daemon=True).start()
        if self.show:
            try:
                cv2.destroyWindow(f"Face Detection - {cid}")
            except cv2.error:
                pass

    # ----------------------------------------------------------------- loop

    def run(self, stop_event: threading.Event) -> None:
        logger.info("Detection engine started (keep_alive=%s, show=%s)", self.keep_alive, self.show)
        show = self.show
        if self.rodent_worker is not None:
            self.rodent_worker.start()
        try:
            while not stop_event.is_set():
                if self._reload.is_set():
                    self._reload.clear()
                    self._apply_settings()
                    self._sync_cameras()

                if not self.streams:
                    if not self.keep_alive:
                        logger.error("No enabled cameras (add them in the web UI or config.yaml)")
                        break
                    stop_event.wait(0.2)
                    continue

                if not self.keep_alive and all(s.failed for s in self.streams.values()):
                    logger.error("All camera streams failed; exiting")
                    break

                got_new_frame = False
                for cid, stream in list(self.streams.items()):
                    if self.hub is not None:
                        self.hub.update_status(cid, state=stream.state.value)
                    if stream.failed:
                        continue
                    frame_id, frame, _ = stream.read()
                    st = self.states[cid]
                    if frame is None or frame_id == st.last_frame_id:
                        continue
                    got_new_frame = True
                    show = self._process(cid, frame_id, frame, st, show)

                if show:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        logger.info("Quit requested from window")
                        stop_event.set()
                if not got_new_frame:
                    time.sleep(0.005)  # avoid a busy loop while waiting for frames
        finally:
            logger.info("Stopping %d stream(s)...", len(self.streams))
            if self.rodent_worker is not None:
                self.rodent_worker.stop()
            for s in self.streams.values():
                s.stop()
            if show:
                cv2.destroyAllWindows()
                cv2.waitKey(1)

    def _process(self, cid: str, frame_id: int, frame: np.ndarray, st: CameraState, show: bool) -> bool:
        st.last_frame_id = frame_id
        st.frames_seen += 1
        self.raw_frames[cid] = frame
        st.display_fps.tick()

        frame_skip = max(1, int(self.cam_rows[cid].get("frame_skip") or self.settings.get("rtsp.frame_skip") or 1))
        run_detection = not self.paused and (st.frames_seen - 1) % frame_skip == 0
        if self.paused:
            st.detections = []
        if run_detection:
            t0 = time.perf_counter()
            if self.detector.recognition_enabled:
                # Detect only, then recognise through the cache (ArcFace for new faces / re-checks).
                st.detections = self.detector.detect(frame, recognize=False)
                self.identities.apply(cid, frame, st.detections)
            else:
                st.detections = self.detector.detect(frame)
            row = self.cam_rows[cid]
            qcfg = replace(self.quality,  # per-camera overrides (small / far CCTV faces)
                           min_face_size=row.get("min_face_size") if row.get("min_face_size") is not None
                           else self.quality.min_face_size,
                           min_blur_score=row.get("min_blur_score") if row.get("min_blur_score") is not None
                           else self.quality.min_blur_score)
            for d in st.detections:
                assess(frame, d, qcfg)
                if d.reject:
                    for r in d.reject.split(","):
                        st.rejects[r] = st.rejects.get(r, 0) + 1
                    logger.debug("[%s] face %s not stored: %s", cid, d.bbox, describe_reject(d, qcfg))
            st.infer_ms = (time.perf_counter() - t0) * 1000
            st.det_fps.tick()

        rodent_on = self.rodent is not None and not self.paused and bool(self.settings.get("rodent.enabled", True))
        if not rodent_on:
            st.rodents, st.rodent_confirmed = [], False
        elif (st.frames_seen - 1) % max(1, int(self.settings.get("rodent.frame_skip") or 1)) == 0:
            self.rodent_worker.submit(cid, frame_id, frame, st)  # type: ignore[union-attr]  # result shows up later

        has_new_faces = run_detection and bool(st.detections)
        vis = None
        if show or self.hub is not None or (has_new_faces and self.output.save_annotated):
            vis = draw_detections(frame.copy(), st.detections)
            draw_rodents(vis, st.rodents, st.rodent_confirmed)
            draw_info(vis, [
                f"{self.cam_rows[cid].get('name') or cid} | FPS {st.display_fps.fps:.1f} | Det FPS {st.det_fps.fps:.1f}",
                "DETECTION PAUSED" if self.paused else
                f"Infer {st.infer_ms:.0f} ms | Faces {len(st.detections)} | {self.detector.device.upper()}",
                *([f"RODENT ALERT ({len(st.rodents)})" if st.rodent_confirmed and st.rodents else
                   f"Rodents {len(st.rodents)} | {st.rodent_ms:.0f} ms"] if rodent_on else []),
            ])
        if run_detection:
            if self.settings.get("dedup.enabled", True):
                # The tracker returns each face's best-quality crop once (also flushes faces that left),
                # so the per-camera rate limit is not needed here.
                for em in self.tracker.update(cid, frame, vis, st.detections, frame_id):
                    self.output.handle(cid, em.frame, [em.det], frame_id=em.frame_id, annotated=em.annotated,
                                       rate_limit=False, captured_at=em.captured_at)
                    st.saved += 1
                    self.learner.consider(cid, em.frame, em.det)
                    logger.debug("[%s] stored best crop of track %d (score %.2f, best of %d)", cid, em.track_id,
                                 (em.det.quality or {}).get("score", 0), em.candidates_seen)
            else:
                good = []
                for d in st.detections:
                    if d.reject is None and d.embedding is None:
                        d.embedding = self.detector.embed_detection(frame, d)
                    if d.reject is None and d.embedding is None and self.quality.require_embedding:
                        d.reject = "no_embedding"
                        logger.debug("[%s] face %s not stored: no_embedding", cid, d.bbox)
                    elif d.reject is None:
                        good.append(d)
                if good:
                    self.output.handle(cid, frame, good, frame_id=frame_id, annotated=vis)
                    st.saved += len(good)
                    for d in good:
                        self.learner.consider(cid, frame, d)

        if self.hub is not None and vis is not None:
            self.hub.update_frame(cid, vis)
            self.hub.update_status(cid, fps=st.display_fps.fps, det_fps=st.det_fps.fps,
                                   infer_ms=st.infer_ms, faces=len(st.detections),
                                   saved=st.saved, rejected=dict(st.rejects),
                                   names=[d.name for d in st.detections if d.name and d.name != "Unknown"],
                                   rodents=len(st.rodents), rodent_alert=st.rodent_confirmed and bool(st.rodents),
                                   rodent_alerts=st.rodent_alerts, rodent_ms=st.rodent_ms if rodent_on else None)
        if show and vis is not None:
            try:
                cv2.imshow(f"Face Detection - {cid}", resize_for_display(vis, self.display_width))
            except cv2.error as exc:
                logger.error("cv2.imshow failed (%s). Is opencv-python-headless installed? "
                             "See README 'Troubleshooting'. Continuing without window.", exc)
                return False
        return show


# ======================================================================
# Retention
# ======================================================================

class RetentionWorker(threading.Thread):
    """Periodically deletes detection history (and its images) older than retention.days."""

    def __init__(self, db: Database, stop_event: threading.Event, interval_sec: float = 3600) -> None:
        super().__init__(name="retention", daemon=True)
        self.db = db
        self.stop_event = stop_event
        self.interval = interval_sec

    def run_once(self) -> int:
        days = int(self.db.get_settings().get("retention.days") or 0)
        if days <= 0:
            return 0
        paths = self.db.delete_detections_before(time.time() - days * 86400)
        paths += self.db.delete_rodent_events_before(time.time() - days * 86400)
        remove_files(set(paths))
        if paths:
            logger.info("Retention: removed history older than %d day(s) (%d file(s))", days, len(set(paths)))
        return len(paths)

    def run(self) -> None:
        if self.stop_event.wait(60):  # let startup finish first
            return
        while True:
            try:
                self.run_once()
            except Exception:
                logger.exception("Retention cleanup failed")
            if self.stop_event.wait(self.interval):
                return
