#!/usr/bin/env python3
"""Give old detection records (saved before migration 001) an embedding, without re-detecting.

Each old record still has its face crop, bbox and 5 landmarks. The landmarks are moved into the
crop's coordinates (the crop was cut with `--margin` padding around the bbox), and ArcFace is run
on the aligned face, the same as for live detections. Quality metrics are filled in as well, and if
the face matches someone in Persons, the record gets that name.

    python scripts/backfill_embeddings.py --dry-run
    python scripts/backfill_embeddings.py               # old crops were cut with 20 % padding
    python scripts/backfill_embeddings.py --margin 0.25

Safe to run while the app is running. Records whose crop file is gone stay without an embedding
(scripts/cleanup_no_embedding.py can delete those).
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from database import Database  # noqa: E402
from detector import Detection, FaceDetector  # noqa: E402
from migrations import run_all  # noqa: E402
from quality import QualityConfig, assess  # noqa: E402
from utils import load_config  # noqa: E402


def crop_origin(bbox: list[int], margin: float) -> tuple[int, int]:
    """Top-left of the crop that utils.crop_face cut for this bbox."""
    x1, y1, x2, y2 = bbox
    return max(0, x1 - int((x2 - x1) * margin)), max(0, y1 - int((y2 - y1) * margin))


def main() -> int:
    p = argparse.ArgumentParser(description="Compute embeddings for old detection records from their crops")
    p.add_argument("--config", default=str(ROOT / "config.yaml"))
    p.add_argument("--db", help="SQLite database (default: database.path from config)")
    p.add_argument("--margin", type=float, default=0.2, help="padding the old crops were cut with (default 0.2)")
    p.add_argument("--limit", type=int, default=0, help="process at most N records (0 = all)")
    p.add_argument("--dry-run", action="store_true", help="compute but don't write")
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    cfg = load_config(args.config)
    db_path = args.db or cfg["database"]["path"]
    if not Path(db_path).is_file():
        print(f"Database not found: {db_path}")
        return 1
    conn = sqlite3.connect(db_path, isolation_level=None)
    conn.execute("PRAGMA busy_timeout=10000")
    run_all(conn)

    rows = conn.execute("SELECT id, bbox, landmarks, confidence, crop_path, person_id FROM detections "
                        "WHERE embedding IS NULL ORDER BY id").fetchall()
    if args.limit:
        rows = rows[:args.limit]
    print(f"Old records without embedding: {len(rows)}")
    if not rows:
        return 0

    detector = FaceDetector.from_config(cfg)
    detector.set_gallery(Database(db_path).gallery())  # to name faces that match a person
    settings = {k: json.loads(v) for k, v in conn.execute("SELECT key, value FROM settings")}
    qcfg = QualityConfig.from_settings(settings)

    done = named = 0
    skipped: dict[str, int] = {}
    t0 = time.time()
    for i, (det_id, bbox_s, lm_s, conf, crop_path, person_id) in enumerate(rows, 1):
        reason = None
        crop = cv2.imread(crop_path) if crop_path else None
        if crop is None:
            reason = "crop file missing"
        elif not lm_s or len(json.loads(lm_s)) != 5:
            reason = "no landmarks"
        else:
            bbox = json.loads(bbox_s)
            ox, oy = crop_origin(bbox, args.margin)
            lms = [(x - ox, y - oy) for x, y in json.loads(lm_s)]
            h, w = crop.shape[:2]
            if not all(-2 <= x <= w + 2 and -2 <= y <= h + 2 for x, y in lms):
                reason = "landmarks outside crop (wrong --margin?)"
            else:
                d = Detection(bbox=(bbox[0] - ox, bbox[1] - oy, bbox[2] - ox, bbox[3] - oy),
                              confidence=float(conf), landmarks=lms)
                emb = detector.embed_detection(crop, d)
                if emb is None:
                    reason = "embedding failed"
                else:
                    assess(crop, d, qcfg)
                    name, pid, sim = detector.match(emb)
                    if not args.dry_run:
                        conn.execute("UPDATE detections SET embedding=?, quality=? WHERE id=?",
                                     (emb.astype(np.float32).tobytes(), json.dumps(d.quality), det_id))
                        if person_id is None and pid is not None:
                            conn.execute("UPDATE detections SET person_id=?, name=?, similarity=? WHERE id=?",
                                         (pid, name, sim, det_id))
                    named += int(person_id is None and pid is not None)
                    done += 1
        if reason:
            skipped[reason] = skipped.get(reason, 0) + 1
        if i % 100 == 0:
            print(f"  {i}/{len(rows)} ({time.time() - t0:.0f}s)")

    print(f"{'Would update' if args.dry_run else 'Updated'} {done} record(s); recognised as a known person: {named}")
    for r, n in skipped.items():
        print(f"  skipped {n}: {r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
