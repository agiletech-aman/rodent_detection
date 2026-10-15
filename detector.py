"""Face detection (+ optional recognition) on top of InsightFace.

Detection model : SCRFD (det_10g in buffalo_l / det_500m in buffalo_s)
Recognition     : ArcFace (w600k_r50 in buffalo_l), 512-d embeddings
Runtime         : ONNX Runtime -> CUDAExecutionProvider when available, else CPU

The models are called directly (instead of FaceAnalysis.get) so recognition can be
switched on/off at runtime without reloading anything.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Detection:
    """One detected face."""

    bbox: tuple[int, int, int, int]               # x1, y1, x2, y2 (pixels, clipped to frame)
    confidence: float
    landmarks: list[tuple[float, float]]          # 5 points: left eye, right eye, nose, mouth left, mouth right
    embedding: np.ndarray | None = field(default=None, repr=False)  # L2-normalised, only with recognition
    name: str | None = None                       # matched identity ("Unknown" if no match)
    person_id: int | None = None                  # database id of the matched person
    similarity: float | None = None
    quality: dict[str, Any] | None = None         # size / blur / yaw / pitch / score (quality.assess)
    reject: str | None = None                     # why the face is not good enough to store, e.g. "blur,pose"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "bbox": [int(v) for v in self.bbox],
            "confidence": round(float(self.confidence), 4),
            "landmarks": [[round(float(x), 1), round(float(y), 1)] for x, y in self.landmarks],
        }
        if self.name is not None:
            d["name"] = self.name
            d["person_id"] = self.person_id
            d["similarity"] = None if self.similarity is None else round(float(self.similarity), 4)
        if self.quality is not None:
            d["quality"] = self.quality
        return d


def _nvidia_driver_present() -> bool:
    import ctypes

    try:
        ctypes.CDLL("libcuda.so.1")
        return True
    except OSError:
        return False


def resolve_providers(device: str = "auto") -> tuple[list[str], int]:
    """Pick ONNX Runtime providers and InsightFace ctx_id (>=0 GPU, -1 CPU)."""
    import onnxruntime as ort

    # onnxruntime-gpu >= 1.21 can load CUDA/cuDNN from the pip nvidia-* wheels.
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception as exc:  # libraries missing -> CPU fallback below
            logger.debug("ort.preload_dlls() failed: %s", exc)

    available = ort.get_available_providers()
    device = device.lower()
    want_cuda = device in ("auto", "cuda", "gpu")
    if want_cuda and "CUDAExecutionProvider" in available and not _nvidia_driver_present():
        # onnxruntime-gpu lists CUDA even without a GPU and then logs loud load errors per model.
        logger.info("No NVIDIA driver (libcuda.so.1) found; using CPU")
        available = [p for p in available if p != "CUDAExecutionProvider"]
    if want_cuda and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"], 0
    if device in ("cuda", "gpu"):
        logger.warning("CUDA requested but not available in onnxruntime (%s); using CPU", available)
    return ["CPUExecutionProvider"], -1


class FaceDetector:
    """Thread-safe face detector + recognizer. Settings can be changed while running."""

    def __init__(
        self,
        model_name: str = "buffalo_l",
        model_root: str = "~/.insightface",
        det_size: tuple[int, int] = (640, 640),
        conf_threshold: float = 0.5,
        min_face_size: int = 0,
        device: str = "auto",
        recognition_enabled: bool = False,
        similarity_threshold: float = 0.4,
    ) -> None:
        from insightface.app import FaceAnalysis  # heavy import, keep it local

        self.model_name = model_name
        self.det_size = tuple(det_size)
        self.conf_threshold = conf_threshold
        self.min_face_size = min_face_size
        self.similarity_threshold = similarity_threshold
        self.recognition_enabled = recognition_enabled
        self._lock = threading.Lock()

        providers, ctx_id = resolve_providers(device)
        # The recognition model is always loaded (needed to enroll people from the web UI),
        # but it only runs per face when recognition is enabled.
        self.app = FaceAnalysis(
            name=model_name,
            root=str(Path(model_root).expanduser()),
            providers=providers,
            allowed_modules=["detection", "recognition"],
        )
        self.app.prepare(ctx_id=ctx_id, det_thresh=conf_threshold, det_size=self.det_size)
        self.det_model = self.app.det_model
        self.rec_model = self.app.models.get("recognition")
        if self.rec_model is None:
            logger.warning("Model pack %s has no recognition model; recognition disabled", model_name)

        # Report what ONNX Runtime really uses (CUDA can silently fall back if libs are missing).
        active = self.det_model.session.get_providers()
        self.device = "cuda" if "CUDAExecutionProvider" in active else "cpu"
        logger.info("Face detector ready: model=%s det_size=%s device=%s providers=%s",
                    model_name, self.det_size, self.device, active)

        self._gallery_ids: list[int] = []
        self._gallery_names: list[str] = []
        self._gallery: np.ndarray | None = None  # (N, 512), rows L2-normalised
        self.gallery_version = 0                  # bumped whenever persons change (invalidates caches)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "FaceDetector":
        det = cfg["detector"]
        return cls(
            model_name=det["model_name"],
            model_root=det["model_root"],
            det_size=tuple(det["det_size"]),
            conf_threshold=float(det["conf_threshold"]),
            min_face_size=int(det.get("min_face_size", 0)),
            device=det["device"],
            recognition_enabled=bool(cfg["recognition"]["enabled"]),
            similarity_threshold=float(cfg["recognition"]["similarity_threshold"]),
        )

    # ------------------------------------------------------------- settings

    def update_settings(self, *, conf_threshold: float | None = None, min_face_size: int | None = None,
                        similarity_threshold: float | None = None, recognition_enabled: bool | None = None) -> None:
        with self._lock:
            if conf_threshold is not None:
                self.conf_threshold = float(conf_threshold)
                self.det_model.det_thresh = float(conf_threshold)  # SCRFD's own first-pass filter
            if min_face_size is not None:
                self.min_face_size = int(min_face_size)
            if similarity_threshold is not None:
                self.similarity_threshold = float(similarity_threshold)
            if recognition_enabled is not None:
                self.recognition_enabled = bool(recognition_enabled) and self.rec_model is not None

    def set_gallery(self, entries: list[tuple[int, str, np.ndarray]]) -> None:
        """Replace the known-person gallery: [(person_id, name, normalised embedding)]."""
        with self._lock:
            self._gallery_ids = [e[0] for e in entries]
            self._gallery_names = [e[1] for e in entries]
            self._gallery = np.stack([e[2] for e in entries]).astype(np.float32) if entries else None
            self.gallery_version += 1
        logger.info("Recognition gallery loaded: %d person(s)", len(entries))

    # ------------------------------------------------------------ detection

    def _raw_faces(self, img: np.ndarray, with_embeddings: bool) -> list[Any]:
        """Run SCRFD (+ ArcFace). Caller must hold self._lock."""
        from insightface.app.common import Face

        bboxes, kpss = self.det_model.detect(img, max_num=0, metric="default")
        faces = []
        for i in range(bboxes.shape[0]):
            face = Face(bbox=bboxes[i, :4], kps=None if kpss is None else kpss[i], det_score=bboxes[i, 4])
            if with_embeddings and face.kps is not None:
                self.rec_model.get(img, face)  # sets face.embedding (aligned with the 5 landmarks)
            faces.append(face)
        return faces

    def match(self, embedding: np.ndarray) -> tuple[str, int | None, float | None]:
        """(name, person_id, similarity) of the closest known person, or ("Unknown", None, best_sim)."""
        with self._lock:
            gallery, ids, names, sim_thr = self._gallery, self._gallery_ids, self._gallery_names, self.similarity_threshold
        if gallery is None:
            return "Unknown", None, None
        sims = gallery @ embedding  # cosine similarity (both sides normalised)
        idx = int(np.argmax(sims))
        best = float(sims[idx])
        return (names[idx], ids[idx], best) if best >= sim_thr else ("Unknown", None, best)

    def rank(self, embedding: np.ndarray) -> list[tuple[int, float]]:
        """[(person_id, similarity)] for every known person, most similar first."""
        with self._lock:
            gallery, ids = self._gallery, self._gallery_ids
        if gallery is None:
            return []
        sims = gallery @ embedding
        return sorted(zip(ids, (float(s) for s in sims)), key=lambda x: x[1], reverse=True)

    def detect(self, frame: np.ndarray, recognize: bool | None = None) -> list[Detection]:
        """Detect faces in a BGR image. recognize=None follows the recognition setting;
        False skips ArcFace (the live engine recognises through IdentityCache instead)."""
        if frame is None or frame.size == 0:
            return []
        h, w = frame.shape[:2]
        with self._lock:
            recognise = (self.recognition_enabled if recognize is None else recognize) and self.rec_model is not None
            faces = self._raw_faces(frame, with_embeddings=recognise)
            conf, min_size = self.conf_threshold, self.min_face_size

        results: list[Detection] = []
        for f in faces:
            score = float(f.det_score)
            if score < conf:
                continue
            x1, y1, x2, y2 = f.bbox.astype(float)
            x1, y1 = int(max(0, round(x1))), int(max(0, round(y1)))
            x2, y2 = int(min(w - 1, round(x2))), int(min(h - 1, round(y2)))
            if x2 <= x1 or y2 <= y1 or min(x2 - x1, y2 - y1) < min_size:
                continue
            kps = [(float(x), float(y)) for x, y in f.kps] if f.kps is not None else []
            det = Detection(bbox=(x1, y1, x2, y2), confidence=score, landmarks=kps)

            if recognise and f.normed_embedding is not None:
                det.embedding = f.normed_embedding.astype(np.float32)
                det.name, det.person_id, det.similarity = self.match(det.embedding)
            results.append(det)

        results.sort(key=lambda d: d.confidence, reverse=True)
        return results

    def embed_detection(self, img: np.ndarray, det: Detection) -> np.ndarray | None:
        """ArcFace embedding for an already detected face (used by the duplicate filter)."""
        if self.rec_model is None or len(det.landmarks) != 5:
            return None
        from insightface.app.common import Face

        face = Face(bbox=np.array(det.bbox, np.float32), kps=np.array(det.landmarks, np.float32),
                    det_score=det.confidence)
        with self._lock:
            self.rec_model.get(img, face)
        emb = face.normed_embedding
        return None if emb is None else emb.astype(np.float32)

    # ----------------------------------------------------------- enrollment

    def embed_largest_face(self, img: np.ndarray) -> tuple[np.ndarray | None, int]:
        """Embedding of the largest face in an image (for enrolling a person).

        Returns (normalised embedding or None, number of faces found).
        """
        if self.rec_model is None:
            raise RuntimeError("Recognition model not available in this model pack")
        with self._lock:
            faces = self._raw_faces(img, with_embeddings=False)
            if not faces:
                return None, 0
            largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            if largest.kps is None:
                return None, len(faces)
            self.rec_model.get(img, largest)
        emb = largest.normed_embedding
        return (None if emb is None else emb.astype(np.float32)), len(faces)
