"""Rodent (rat / mouse / shrew) detection on the camera streams.

RodentDetector  YOLO detector exported to ONNX (single class "rodent"), run with ONNX Runtime
                (no PyTorch needed at runtime). Trained by scripts/rodent/train.py.
RodentWatcher   per-camera confirmation + cooldown: an alert is raised only when a rodent is
                seen in at least `confirm_hits` of the last `confirm_window` detection runs,
                then not again on that camera for `cooldown_sec`. One stray box never alerts.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from detector import resolve_providers
from utils import crop_face, now_timestamps, save_image

logger = logging.getLogger(__name__)

COLOR_RODENT = (0, 0, 255)  # BGR red


@dataclass
class RodentDetection:
    bbox: tuple[int, int, int, int]   # x1, y1, x2, y2 (pixels, clipped to frame)
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {"bbox": [int(v) for v in self.bbox], "confidence": round(float(self.confidence), 4)}


class RodentDetector:
    """Thread-safe YOLO (ONNX) rodent detector. Settings can be changed while running."""

    def __init__(self, model_path: str | Path, device: str = "auto", conf_threshold: float = 0.5,
                 iou_threshold: float = 0.45, min_size: int = 8, threads: int = 2) -> None:
        import onnxruntime as ort

        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.min_size = min_size
        self._lock = threading.Lock()

        providers, _ = resolve_providers(device)
        opts = ort.SessionOptions()
        # Runs beside the face detector in its own thread: with all cores it starves face detection on CPU.
        opts.intra_op_num_threads = int(threads)
        self.session = ort.InferenceSession(str(self.model_path), opts, providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.imgsz = (int(inp.shape[2]), int(inp.shape[3]))  # (h, w); exported with a fixed size
        self.device = "cuda" if "CUDAExecutionProvider" in self.session.get_providers() else "cpu"
        logger.info("Rodent detector ready: model=%s input=%s device=%s", self.model_path.name, self.imgsz, self.device)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "RodentDetector | None":
        """None when the model file does not exist yet (rodent detection is then unavailable)."""
        rc = cfg.get("rodent") or {}
        path = Path(rc.get("model_path") or "models/rodent.onnx")
        if not path.is_file():
            logger.warning("Rodent model not found at %s; rodent detection disabled "
                           "(train it with scripts/rodent/train.py)", path)
            return None
        return cls(path, device=cfg["detector"].get("device", "auto"),
                   conf_threshold=float(rc.get("conf_threshold", 0.5)),
                   iou_threshold=float(rc.get("iou_threshold", 0.45)), threads=int(rc.get("threads", 2)))

    def update_settings(self, *, conf_threshold: float | None = None) -> None:
        if conf_threshold is not None:
            self.conf_threshold = float(conf_threshold)

    def _letterbox(self, img: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        ih, iw = self.imgsz
        h, w = img.shape[:2]
        r = min(ih / h, iw / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        top, left = (ih - nh) // 2, (iw - nw) // 2
        canvas = np.full((ih, iw, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return canvas, r, (left, top)

    def detect(self, frame: np.ndarray, conf_threshold: float | None = None) -> list[RodentDetection]:
        if frame is None or frame.size == 0:
            return []
        conf = self.conf_threshold if conf_threshold is None else conf_threshold
        h, w = frame.shape[:2]
        img, r, (dx, dy) = self._letterbox(frame)
        blob = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        with self._lock:
            out = self.session.run(None, {self.input_name: blob})[0]
        pred = out[0].T                                  # (N, 4 + classes): cx, cy, w, h, score...
        # Class 0 is the rodent. Extra classes (e.g. "person" in the interim YOLO-World model) are
        # distractors: a box only counts when class 0 is its best class.
        cls_scores = pred[:, 4:]
        scores = cls_scores[:, 0]
        keep = (scores >= conf) & (cls_scores.argmax(axis=1) == 0)
        if not keep.any():
            return []
        pred, scores = pred[keep], scores[keep]
        cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        boxes = np.stack([(cx - bw / 2 - dx) / r, (cy - bh / 2 - dy) / r, bw / r, bh / r], axis=1)
        idx = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf, self.iou_threshold)
        results = []
        for i in np.array(idx).reshape(-1):
            x, y, bw_, bh_ = boxes[i]
            x1, y1 = int(max(0, round(x))), int(max(0, round(y)))
            x2, y2 = int(min(w - 1, round(x + bw_))), int(min(h - 1, round(y + bh_)))
            if x2 - x1 < self.min_size or y2 - y1 < self.min_size:
                continue
            results.append(RodentDetection((x1, y1, x2, y2), float(scores[i])))
        results.sort(key=lambda d: d.confidence, reverse=True)
        return results


def draw_rodents(img: np.ndarray, dets: list[RodentDetection], confirmed: bool = True) -> np.ndarray:
    """Red boxes in place. Unconfirmed boxes (not yet an alert) are drawn thinner."""
    thickness = max(2, round(min(img.shape[:2]) / 300)) if confirmed else 1
    for d in dets:
        x1, y1, x2, y2 = d.bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_RODENT, thickness)
        label = f"RODENT {d.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        if y1 - th - 10 > 0:
            ty = y1 - 6                          # above the box
        elif y2 + th + 10 < img.shape[0]:
            ty = y2 + th + 6                     # below the box
        else:
            ty = y1 + th + 6                     # box fills the frame: inside, top-left
        cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 6, ty + 4), COLOR_RODENT, -1)
        cv2.putText(img, label, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return img


@dataclass
class _CamTrack:
    history: deque = field(default_factory=lambda: deque(maxlen=5))  # True/False per detection run
    last_alert: float = 0.0
    confirmed: bool = False


class RodentWatcher:
    """Confirms rodent sightings per camera and stores alerts (images + database row)."""

    def __init__(self, db: Any, output_dir: str | Path, jpeg_quality: int = 92) -> None:
        self.db = db
        self.output_dir = Path(output_dir) / "rodents"
        self.jpeg_quality = jpeg_quality
        self.confirm_hits = 3
        self.confirm_window = 5
        self.cooldown_sec = 60.0
        self.save_enabled = True
        self._cams: dict[str, _CamTrack] = {}

    def apply_settings(self, settings: dict[str, Any]) -> None:
        self.confirm_window = max(1, int(settings.get("rodent.confirm_window") or self.confirm_window))
        self.confirm_hits = min(self.confirm_window, max(1, int(settings.get("rodent.confirm_hits") or self.confirm_hits)))
        self.cooldown_sec = float(settings.get("rodent.cooldown_sec") or 0)
        for t in self._cams.values():
            t.history = deque(t.history, maxlen=self.confirm_window)

    def reset(self, camera_id: str | None = None) -> None:
        if camera_id is None:
            self._cams.clear()
        else:
            self._cams.pop(camera_id, None)

    def update(self, camera_id: str, frame: np.ndarray, dets: list[RodentDetection],
               frame_id: int | None = None) -> tuple[bool, dict[str, Any] | None]:
        """Feed one detection run. Returns (confirmed now, stored alert record or None)."""
        t = self._cams.get(camera_id)
        if t is None:
            t = self._cams[camera_id] = _CamTrack(history=deque(maxlen=self.confirm_window))
        t.history.append(bool(dets))
        t.confirmed = sum(t.history) >= self.confirm_hits
        if not (t.confirmed and dets):
            return t.confirmed, None
        now = time.time()
        if now - t.last_alert < self.cooldown_sec:
            return True, None
        t.last_alert = now
        return True, self._store(camera_id, frame, dets, frame_id, now)

    def _store(self, camera_id: str, frame: np.ndarray, dets: list[RodentDetection],
               frame_id: int | None, ts: float) -> dict[str, Any]:
        iso_ts, file_ts = now_timestamps(ts)
        best = dets[0]
        crop_path = annotated_path = None
        if self.save_enabled:
            folder = self.output_dir / camera_id
            crop = folder / f"{file_ts}_crop.jpg"
            if save_image(crop, crop_face(frame, best.bbox, 0.5), self.jpeg_quality):
                crop_path = str(crop)
            ann = folder / f"{file_ts}.jpg"
            if save_image(ann, draw_rodents(frame.copy(), dets), self.jpeg_quality):
                annotated_path = str(ann)
        record = {"camera_id": camera_id, "timestamp": iso_ts, "bbox": list(best.bbox),
                  "confidence": round(best.confidence, 4), "count": len(dets),
                  "boxes": [d.to_dict() for d in dets], "crop_path": crop_path,
                  "annotated_path": annotated_path, "frame_id": frame_id}
        try:
            record["id"] = self.db.insert_rodent_event(record)
        except Exception:
            logger.exception("[%s] Failed to store rodent alert", camera_id)
        logger.warning("[%s] RODENT detected: %d box(es), confidence %.2f", camera_id, len(dets), best.confidence)
        return record
