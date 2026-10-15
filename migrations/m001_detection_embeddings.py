"""001: store the face embedding + quality metrics with every detection.

- detections.embedding  BLOB  float32[512], L2-normalised (from the detection itself, so
                              "Assign person" no longer re-detects the crop)
- detections.quality    TEXT  JSON {score, size, blur, yaw, pitch}
- settings: crop padding 0.20 -> 0.25 and cooldown 60 -> 30 s, but only if they still have
  the old defaults (custom values are kept).

Old rows keep embedding = NULL; scripts/cleanup_no_embedding.py removes them.
"""

from __future__ import annotations

import json
import sqlite3

VERSION = 1
DESCRIPTION = "detections.embedding + detections.quality, new quality defaults"


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def migrate(conn: sqlite3.Connection) -> None:
    cols = _columns(conn, "detections")
    if "embedding" not in cols:
        conn.execute("ALTER TABLE detections ADD COLUMN embedding BLOB")
    if "quality" not in cols:
        conn.execute("ALTER TABLE detections ADD COLUMN quality TEXT")

    def bump(key: str, old: float, new: float) -> None:
        row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        if row is not None and float(json.loads(row[0])) == old:
            conn.execute("UPDATE settings SET value=? WHERE key=?", (json.dumps(new), key))

    bump("output.crop_margin", 0.2, 0.25)
    bump("dedup.cooldown_sec", 60, 30)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit("Run all migrations with: python -m migrations --db data/app.db")
