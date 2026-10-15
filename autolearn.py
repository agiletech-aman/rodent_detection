"""Self-learning recognition: confidently recognised faces are added to the person.

Each stored face (already the best-quality crop of its track) that matches a known person
is a new view of them: another angle, distance, light or camera. Adding its embedding to the
person moves their gallery vector towards how they really look on these cameras, so the
next detections of them match with a higher similarity.

A wrong face added to a person would make things worse, so a face is only learned when:
  1. it matches the person with at least `min_similarity` (well above the recognition threshold),
  2. it is clearly closer to that person than to anyone else (MARGIN),
  3. it is not almost identical to a photo the person already has (REDUNDANT, nothing new to learn),
  4. the person did not learn anything in the last COOLDOWN_SEC (spreads views over time).
Auto-learned photos are marked 'auto' in Persons, where a wrong one can be removed. At most
`max_per_person` are kept; the oldest is replaced, manual photos are never touched.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

import numpy as np

from database import Database
from detector import Detection, FaceDetector
from enrollment import remove_files
from utils import crop_face, save_image

logger = logging.getLogger(__name__)

MARGIN = 0.10          # best person must beat the second best by this much
REDUNDANT = 0.85       # similarity to an existing photo above which a face adds nothing new
COOLDOWN_SEC = 300.0   # at most one learned face per person in this time
CROP_MARGIN = 0.25


class AutoLearner:
    def __init__(self, db: Database, detector: FaceDetector, persons_dir: str | Path) -> None:
        self.db = db
        self.detector = detector
        self.persons_dir = Path(persons_dir)
        self.enabled = True
        self.min_similarity = 0.55
        self.max_per_person = 30
        self._last_learned: dict[int, float] = {}

    def apply_settings(self, s: dict) -> None:
        self.enabled = bool(s.get("recognition.auto_learn", True))
        self.min_similarity = float(s.get("recognition.auto_learn_similarity", self.min_similarity))
        self.max_per_person = int(s.get("recognition.auto_learn_max", self.max_per_person))

    def consider(self, camera_id: str, frame: np.ndarray, det: Detection) -> bool:
        """Learn this stored face if it is a confident match. Returns True when it was added."""
        if not self.enabled or det.person_id is None or det.embedding is None:
            return False
        pid = det.person_id
        now = time.time()
        if now - self._last_learned.get(pid, 0.0) < COOLDOWN_SEC:
            return False

        ranked = self.detector.rank(det.embedding)
        if not ranked or ranked[0][0] != pid or ranked[0][1] < self.min_similarity:
            return False
        if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < MARGIN:
            logger.debug("[%s] not learning person #%d: too close to person #%d (%.2f vs %.2f)",
                         camera_id, pid, ranked[1][0], ranked[0][1], ranked[1][1])
            return False

        existing = self.db.person_embeddings(pid)
        if existing and max(float(e["embedding"] @ det.embedding) for e in existing) >= REDUNDANT:
            return False

        dest = self.persons_dir / str(pid) / f"auto_{uuid.uuid4().hex}.jpg"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not save_image(dest, crop_face(frame, det.bbox, CROP_MARGIN)):
            return False
        self.db.add_person_image(pid, str(dest), det.embedding, source="auto")
        self._last_learned[pid] = now

        autos = [e for e in existing if e["source"] == "auto"]
        for old in autos[:max(0, len(autos) + 1 - self.max_per_person)]:
            path = self.db.delete_person_image(pid, old["id"])
            if path:
                remove_files([path])

        self.detector.set_gallery(self.db.gallery())
        logger.info("[%s] Learned a new face of %s (#%d, similarity %.2f)", camera_id, det.name, pid, ranked[0][1])
        return True
