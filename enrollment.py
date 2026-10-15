"""Adding face photos to persons (used by the web UI and the first-run known_faces import)."""

from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path

import cv2
import numpy as np

from database import Database
from detector import IMAGE_EXTS, FaceDetector

logger = logging.getLogger(__name__)


class EnrollmentError(ValueError):
    """The image cannot be used to enroll a person (unreadable, no face...)."""


def decode_image(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
    if img is None:
        raise EnrollmentError("Not a valid image file")
    return img


def enroll_image(db: Database, detector: FaceDetector, persons_dir: Path,
                 person_id: int, img: np.ndarray) -> tuple[int, str | None]:
    """Store a face photo for a person. Returns (image_id, warning or None)."""
    embedding, n_faces = detector.embed_largest_face(img)
    if embedding is None:
        raise EnrollmentError("No face found in the image")
    warning = f"{n_faces} faces found, the largest one was used" if n_faces > 1 else None

    path = Path(persons_dir) / str(person_id) / f"{uuid.uuid4().hex}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    if max(img.shape[:2]) > 1600:  # keep the stored photos reasonably small
        scale = 1600 / max(img.shape[:2])
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if not cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise EnrollmentError("Could not save the image")
    image_id = db.add_person_image(person_id, str(path), embedding)
    return image_id, warning


def enroll_embedding(db: Database, persons_dir: Path, person_id: int, embedding: np.ndarray,
                     image_path: str | None) -> int:
    """Add an already computed embedding (e.g. from a stored detection) to a person.
    The detection's crop is copied as the person's photo. No face detection is run again."""
    dest = Path(persons_dir) / str(person_id) / f"{uuid.uuid4().hex}.jpg"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if image_path and Path(image_path).is_file():
        shutil.copyfile(image_path, dest)
    else:  # crop file gone: keep the embedding, use a placeholder image
        cv2.imwrite(str(dest), np.full((112, 112, 3), 60, np.uint8))
    return db.add_person_image(person_id, str(dest), embedding)


def remove_files(paths: list[str] | set[str]) -> None:
    for p in paths:
        try:
            Path(p).unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Could not delete %s: %s", p, exc)


def delete_person_dir(persons_dir: Path, person_id: int) -> None:
    shutil.rmtree(Path(persons_dir) / str(person_id), ignore_errors=True)


def import_known_faces_dir(db: Database, detector: FaceDetector, directory: str | Path, persons_dir: Path) -> int:
    """One-time import of the old folder layout into the database.

        known_faces/alice.jpg               -> person "alice"
        known_faces/bob/1.jpg, bob/2.jpg    -> person "bob"
    """
    root = Path(directory).expanduser()
    if not root.is_dir():
        return 0
    by_name: dict[str, int] = {p["name"]: p["id"] for p in db.list_persons()}
    added = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        name = path.parent.name if path.parent != root else path.stem
        img = cv2.imread(str(path))
        if img is None:
            logger.warning("Cannot read %s", path)
            continue
        if name not in by_name:
            by_name[name] = db.create_person(name, notes=f"Imported from {root}")["id"]
        try:
            enroll_image(db, detector, persons_dir, by_name[name], img)
            added += 1
        except EnrollmentError as exc:
            logger.warning("Skipping %s: %s", path, exc)
    logger.info("Imported %d photo(s) from %s", added, root)
    return added
