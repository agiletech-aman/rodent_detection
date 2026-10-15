"""Face tracking for live cameras: one best-quality crop per person, no duplicates.

A person in front of a camera is detected in every processed frame. FaceTracker keeps a
track per face (per camera) and, instead of storing every detection:

    - collects quality-passing detections of a track for `best_window_sec` and stores only the
      best one (score = size x sharpness x frontal-ness, see quality.py);
    - if the track disappears before the window ends, its best candidate is stored right away;
    - after storing, the same track / recognised person is not stored again on that camera
      for `cooldown_sec`; a face that was stored as Unknown and is later recognised may be
      stored once more with the name.

A detection is matched to an existing track by, in order:
    1. the same recognised person_id
    2. the same place: box overlap (IoU) / centre distance with a live track. CCTV faces are
       often tiny (20-40 px) and drop out for seconds when a seated person looks down, so
       position is more reliable than their embeddings.
    3. face-embedding similarity (same face that reappeared somewhere else); the embedding is
       only computed for faces that steps 1-2 could not match.

IdentityCache (below) makes live recognition cheap: ArcFace runs for new faces and
periodic re-checks only, instead of for every face in every frame.
"""

from __future__ import annotations

import itertools
import logging
import time
from dataclasses import dataclass

import numpy as np

from detector import Detection, FaceDetector

logger = logging.getLogger(__name__)

MAX_TRACKS_PER_CAMERA = 500
LOST_FLUSH_SEC = 1.5  # a track not seen for this long stores its pending best crop


@dataclass
class Candidate:
    det: Detection
    frame: np.ndarray
    annotated: np.ndarray | None
    frame_id: int | None
    captured_at: float
    score: float


@dataclass
class Emission:
    """A face to store: the best crop of a track."""
    camera_id: str
    det: Detection
    frame: np.ndarray
    annotated: np.ndarray | None
    frame_id: int | None
    captured_at: float
    track_id: int
    candidates_seen: int


@dataclass
class Track:
    track_id: int
    bbox: tuple[int, int, int, int]
    embedding: np.ndarray | None
    person_id: int | None
    first_seen: float
    last_seen: float
    name: str | None = None
    similarity: float | None = None
    best: Candidate | None = None
    window_start: float | None = None
    candidates_seen: int = 0
    last_emit: float | None = None
    emitted_person: int | None = None  # person_id stored with the last emission


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(area_a + area_b - inter)


def same_place_score(a: tuple[int, int, int, int], b: tuple[int, int, int, int], iou_threshold: float) -> float:
    """>0 when two boxes are at the same place: IoU above the threshold, or centres closer
    than half a face width (tiny boxes jitter a lot relative to their size)."""
    o = iou(a, b)
    if o >= iou_threshold:
        return o
    w = max(a[2] - a[0], b[2] - b[0], 1)
    dx = (a[0] + a[2] - b[0] - b[2]) / 2
    dy = (a[1] + a[3] - b[1] - b[3]) / 2
    return 0.01 if (dx * dx + dy * dy) ** 0.5 <= 0.5 * w else 0.0


class FaceTracker:
    """Per-camera face tracks; update() returns the best-quality crops that should be stored."""

    def __init__(self, detector: FaceDetector, cooldown_sec: float = 30.0, similarity: float = 0.45,
                 best_window_sec: float = 3.0, iou_threshold: float = 0.3) -> None:
        self.detector = detector
        self.cooldown = cooldown_sec
        self.similarity = similarity
        self.best_window = best_window_sec
        self.iou_threshold = iou_threshold
        self.require_embedding = True
        self._tracks: dict[str, list[Track]] = {}
        self._person_emit: dict[str, dict[int, float]] = {}  # camera -> person_id -> last stored
        self._ids = itertools.count(1)

    def update_settings(self, cooldown_sec: float | None = None, similarity: float | None = None,
                        best_window_sec: float | None = None, require_embedding: bool | None = None) -> None:
        if require_embedding is not None:
            self.require_embedding = bool(require_embedding)
        if cooldown_sec is not None:
            self.cooldown = float(cooldown_sec)
        if similarity is not None:
            self.similarity = float(similarity)
        if best_window_sec is not None:
            self.best_window = float(best_window_sec)

    def reset(self, camera_id: str | None = None) -> None:
        if camera_id is None:
            self._tracks.clear()
            self._person_emit.clear()
        else:
            self._tracks.pop(camera_id, None)
            self._person_emit.pop(camera_id, None)

    def active_tracks(self, camera_id: str) -> int:
        return len(self._tracks.get(camera_id, []))

    # ------------------------------------------------------------------

    def _in_cooldown(self, camera_id: str, t: Track, now: float) -> bool:
        """True while this face must not be stored again.

        - a recognised person stored on this camera (by any track) within the cooldown -> blocked
        - this track stored within the cooldown -> blocked, except once when it was stored as
          Unknown and has been recognised since (so the named crop is stored)
        """
        if t.person_id is not None:
            last = self._person_emit.get(camera_id, {}).get(t.person_id)
            if last is not None and now - last < self.cooldown:
                return True
        if t.last_emit is not None and now - t.last_emit < self.cooldown:
            upgraded_to_known = t.person_id is not None and t.emitted_person is None
            return not upgraded_to_known
        return False

    def _emit(self, camera_id: str, t: Track, now: float) -> Emission:
        c = t.best
        assert c is not None
        det = c.det
        if t.person_id is not None and det.person_id is None:  # recognised later in the track
            det.person_id, det.name, det.similarity = t.person_id, t.name, t.similarity
        em = Emission(camera_id, det, c.frame, c.annotated, c.frame_id, c.captured_at, t.track_id, t.candidates_seen)
        t.last_emit, t.emitted_person = now, t.person_id
        if t.person_id is not None:
            self._person_emit.setdefault(camera_id, {})[t.person_id] = now
        t.best, t.window_start, t.candidates_seen = None, None, 0
        return em

    def update(self, camera_id: str, frame: np.ndarray, annotated: np.ndarray | None,
               detections: list[Detection], frame_id: int | None = None,
               now: float | None = None) -> list[Emission]:
        """Feed one processed frame (also with zero faces, so pending crops get flushed).

        Detections must already carry .quality / .reject (quality.assess). Returns crops to store.
        """
        now = time.time() if now is None else now
        out: list[Emission] = []
        tracks = self._tracks.get(camera_id, [])

        # Tracks that left: store their pending best crop, drop them after the cooldown.
        keep = []
        for t in tracks:
            if t.best is not None and now - t.last_seen > LOST_FLUSH_SEC:
                out.append(self._emit(camera_id, t, now))
            if now - t.last_seen <= max(self.cooldown, self.best_window, LOST_FLUSH_SEC):
                keep.append(t)
        tracks = keep

        used: set[int] = set()
        for det in detections:  # already sorted by confidence
            track = self._match(det, tracks, used, frame, now)
            if track is None:
                if det.embedding is None:
                    det.embedding = self.detector.embed_detection(frame, det)
                track = Track(next(self._ids), det.bbox, det.embedding, det.person_id, now, now)
                tracks.append(track)
            else:
                track.bbox, track.last_seen = det.bbox, now
                if det.embedding is not None:
                    if track.embedding is None:
                        track.embedding = det.embedding
                    else:  # slow moving average keeps the track robust to pose changes
                        e = 0.8 * track.embedding + 0.2 * det.embedding
                        track.embedding = (e / (np.linalg.norm(e) + 1e-12)).astype(np.float32)
            if det.person_id is not None:
                track.person_id, track.name, track.similarity = det.person_id, det.name, det.similarity
            used.add(track.track_id)

            # Candidate for this track's best crop?
            if det.reject is None and not self._in_cooldown(camera_id, track, now):
                score = float((det.quality or {}).get("score", 0.0))
                if track.best is None or score > track.best.score:
                    # The stored face must carry its own embedding (Assign person uses it, no
                    # re-detection). Computed only for a new best, so ArcFace runs rarely.
                    if det.embedding is None:
                        det.embedding = self.detector.embed_detection(frame, det)
                    if det.embedding is None and self.require_embedding:
                        det.reject = "no_embedding"
                        logger.debug("[%s] face %s not stored: no_embedding", camera_id, det.bbox)
                        continue
                    track.best = Candidate(det, frame, annotated, frame_id, now, score)
                track.candidates_seen += 1
                if track.window_start is None:
                    track.window_start = now

        # Windows that are complete: store the best crop.
        for t in tracks:
            if t.best is not None and t.window_start is not None and now - t.window_start >= self.best_window:
                out.append(self._emit(camera_id, t, now))

        if len(tracks) > MAX_TRACKS_PER_CAMERA:  # crowded scene: drop the stalest tracks
            tracks.sort(key=lambda t: t.last_seen)
            tracks = tracks[-MAX_TRACKS_PER_CAMERA:]
        self._tracks[camera_id] = tracks
        return out

    def _match(self, det: Detection, tracks: list[Track], used: set[int],
               frame: np.ndarray, now: float) -> Track | None:
        def compatible(t: Track) -> bool:
            # never merge two different recognised people
            return t.track_id not in used and not (
                det.person_id is not None and t.person_id is not None and t.person_id != det.person_id)

        candidates = [t for t in tracks if compatible(t)]
        if not candidates:
            return None

        if det.person_id is not None:
            for t in candidates:
                if t.person_id == det.person_id:
                    return t

        place = [(same_place_score(det.bbox, t.bbox, self.iou_threshold), t) for t in candidates]
        place = [(sc, t) for sc, t in place if sc > 0]
        if place:  # best overlap; ties -> most recently seen
            return max(place, key=lambda x: (x[0], x[1].last_seen))[1]

        with_emb = [t for t in candidates if t.embedding is not None]
        if not with_emb:
            return None
        if det.embedding is None:
            det.embedding = self.detector.embed_detection(frame, det)
            if det.embedding is None:
                return None
        sims = np.array([float(t.embedding @ det.embedding) for t in with_emb])  # type: ignore[operator]
        idx = int(np.argmax(sims))
        return with_emb[idx] if sims[idx] >= self.similarity else None


@dataclass
class _Identity:
    bbox: tuple[int, int, int, int]
    last_seen: float
    checked_at: float
    gallery_version: int
    name: str
    person_id: int | None
    similarity: float | None
    embedding: np.ndarray | None


class IdentityCache:
    """Remembers who is where per camera, so ArcFace only runs for new faces and periodic re-checks.

    Unknown faces are re-checked more often (a person may turn towards the camera); known ones
    less often. A change to Persons (gallery_version) forces a re-check of everyone.
    """

    def __init__(self, detector: FaceDetector, recheck_unknown_sec: float = 3.0, recheck_known_sec: float = 10.0,
                 max_gap_sec: float = 5.0, iou_threshold: float = 0.3, max_rechecks_per_frame: int = 2) -> None:
        self.detector = detector
        self.max_rechecks_per_frame = max_rechecks_per_frame
        self.recheck_unknown = recheck_unknown_sec
        self.recheck_known = recheck_known_sec
        self.max_gap = max_gap_sec
        self.iou_threshold = iou_threshold
        self._tracks: dict[str, list[_Identity]] = {}

    def reset(self, camera_id: str | None = None) -> None:
        if camera_id is None:
            self._tracks.clear()
        else:
            self._tracks.pop(camera_id, None)

    def apply(self, camera_id: str, frame: np.ndarray, detections: list[Detection], now: float | None = None) -> int:
        """Fill name / person_id / similarity / embedding on detections. Returns how many ArcFace runs it did.

        New faces are always recognised immediately. Re-checks of tracked faces are spread over
        frames (at most max_rechecks_per_frame, oldest first) so CPU load stays smooth.
        """
        now = time.time() if now is None else now
        version = self.detector.gallery_version
        tracks = [t for t in self._tracks.get(camera_id, []) if now - t.last_seen <= self.max_gap]
        used: set[int] = set()
        pairs: list[tuple[Detection, _Identity | None]] = []
        for det in detections:
            best, best_score = None, 0.0
            for i, t in enumerate(tracks):
                if i not in used:
                    sc = same_place_score(det.bbox, t.bbox, self.iou_threshold)
                    if sc > best_score:
                        best, best_score = i, sc
            if best is not None:
                used.add(best)
            pairs.append((det, tracks[best] if best is not None else None))

        def due(t: _Identity) -> bool:
            limit = self.recheck_known if t.person_id is not None else self.recheck_unknown
            return t.gallery_version != version or now - t.checked_at >= limit

        rechecks = sorted((t for _, t in pairs if t is not None and due(t)), key=lambda t: t.checked_at)
        allowed = {id(t) for t in rechecks[:self.max_rechecks_per_frame]}

        runs = 0
        for det, t in pairs:
            if t is None or id(t) in allowed:
                emb = self.detector.embed_detection(frame, det)
                runs += 1
                name, pid, sim = self.detector.match(emb) if emb is not None else ("Unknown", None, None)
                if t is None:
                    t = _Identity(det.bbox, now, now, version, name, pid, sim, emb)
                    tracks.append(t)
                else:
                    t.checked_at, t.gallery_version = now, version
                    t.name, t.person_id, t.similarity, t.embedding = name, pid, sim, emb
            t.bbox, t.last_seen = det.bbox, now
            det.name, det.person_id, det.similarity, det.embedding = t.name, t.person_id, t.similarity, t.embedding
        self._tracks[camera_id] = tracks
        return runs
