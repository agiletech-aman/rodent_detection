"""Face quality: size, confidence, sharpness (Laplacian variance) and head pose from the 5 landmarks.

Used to decide which crops are good enough to save / POST, and to pick the best crop of a track.
Pose is estimated geometrically from SCRFD's 5 points (no extra model per face), so it is an
approximation: good for rejecting side / looking-down faces, not for precise angles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from detector import Detection

# Nose position between the eye line (0) and the mouth line (1) for a frontal face
# (ArcFace 112x112 alignment template).
FRONTAL_NOSE_RATIO = 0.49
# How far the nose tip sticks out of the face plane, relative to half the eye distance (yaw)
# and to the eye-mouth distance (pitch). Projected offset = depth * tan(angle).
NOSE_DEPTH_YAW = 0.9
NOSE_DEPTH_PITCH = 0.6
BLUR_SIZE = 112  # faces are resized to this before measuring sharpness, so scores are comparable


@dataclass
class QualityConfig:
    enabled: bool = True
    min_face_size: int = 80        # px, shorter side of the box
    min_confidence: float = 0.7    # det_score
    min_blur_score: float = 60.0   # Laplacian variance on the 112x112 grey face
    max_yaw: float = 35.0          # degrees, left/right head turn
    max_pitch: float = 30.0        # degrees, up/down
    require_embedding: bool = True # a stored face must have an ArcFace embedding (needed for Assign person)

    @classmethod
    def from_settings(cls, s: dict[str, Any]) -> "QualityConfig":
        d = cls()
        return cls(
            enabled=bool(s.get("quality.enabled", d.enabled)),
            min_face_size=int(s.get("quality.min_face_size", d.min_face_size)),
            min_confidence=float(s.get("quality.min_confidence", d.min_confidence)),
            min_blur_score=float(s.get("quality.min_blur_score", d.min_blur_score)),
            max_yaw=float(s.get("quality.max_yaw", d.max_yaw)),
            max_pitch=float(s.get("quality.max_pitch", d.max_pitch)),
            require_embedding=bool(s.get("quality.require_embedding", d.require_embedding)),
        )


def estimate_pose(landmarks: list[tuple[float, float]]) -> tuple[float, float]:
    """(yaw, pitch) in degrees from 5 landmarks: left eye, right eye, nose, mouth left, mouth right.

    yaw > 0: nose towards the image right; pitch > 0: looking down. Roll is removed first.
    """
    if len(landmarks) != 5:
        return 0.0, 0.0
    pts = np.asarray(landmarks, dtype=np.float64)
    le, re = pts[0], pts[1]
    eye_mid = (le + re) / 2
    v = re - le
    ang = math.atan2(v[1], v[0])
    c, s = math.cos(-ang), math.sin(-ang)
    rot = np.array([[c, -s], [s, c]])
    p = (pts - eye_mid) @ rot.T  # eyes now on the x axis, centred at 0
    half_eye = (p[1, 0] - p[0, 0]) / 2
    if half_eye <= 1e-6:
        return 90.0, 0.0
    nose = p[2]
    yaw = math.degrees(math.atan(nose[0] / half_eye / NOSE_DEPTH_YAW))
    mouth_y = (p[3, 1] + p[4, 1]) / 2
    if mouth_y <= 1e-6:
        return yaw, 90.0
    t = nose[1] / mouth_y
    pitch = math.degrees(math.atan((t - FRONTAL_NOSE_RATIO) / NOSE_DEPTH_PITCH))
    return yaw, pitch


def blur_score(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    """Variance of the Laplacian of the face (higher = sharper)."""
    x1, y1, x2, y2 = bbox
    crop = frame[max(0, y1):y2, max(0, x1):x2]
    if crop.size == 0:
        return 0.0
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    grey = cv2.resize(grey, (BLUR_SIZE, BLUR_SIZE), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


def assess(frame: np.ndarray, det: Detection, cfg: QualityConfig) -> None:
    """Fill det.quality (metrics + score) and det.reject (comma-separated reasons, or None)."""
    x1, y1, x2, y2 = det.bbox
    size = min(x2 - x1, y2 - y1)
    blur = blur_score(frame, det.bbox)
    yaw, pitch = estimate_pose(det.landmarks)

    frontal = max(0.0, math.cos(math.radians(yaw)) * math.cos(math.radians(pitch)))
    size_f = min(1.0, size / 112.0)
    sharp_f = min(1.0, blur / max(4.0 * cfg.min_blur_score, 1.0))
    det.quality = {
        "score": round(size_f * sharp_f * frontal, 4),
        "size": int(size),
        "blur": round(blur, 1),
        "yaw": round(yaw, 1),
        "pitch": round(pitch, 1),
    }

    reasons = []
    if cfg.enabled:
        if size < cfg.min_face_size:
            reasons.append("small")
        if det.confidence < cfg.min_confidence:
            reasons.append("low_conf")
        if blur < cfg.min_blur_score:
            reasons.append("blur")
        if abs(yaw) > cfg.max_yaw or abs(pitch) > cfg.max_pitch:
            reasons.append("pose")
    det.reject = ",".join(reasons) or None


def describe_reject(det: Detection, cfg: QualityConfig) -> str:
    """Human-readable reject reason for logs, e.g. 'blur (21.3 < 40)'."""
    q = det.quality or {}
    parts = []
    for r in (det.reject or "").split(","):
        if r == "small":
            parts.append(f"small ({q.get('size')}px < {cfg.min_face_size})")
        elif r == "low_conf":
            parts.append(f"low_conf ({det.confidence:.2f} < {cfg.min_confidence})")
        elif r == "blur":
            parts.append(f"blur ({q.get('blur')} < {cfg.min_blur_score})")
        elif r == "no_embedding":
            parts.append("no_embedding (ArcFace could not embed the face)")
        elif r == "pose":
            parts.append(f"pose (yaw {q.get('yaw')}, pitch {q.get('pitch')}; max {cfg.max_yaw}/{cfg.max_pitch})")
    return ", ".join(parts)
