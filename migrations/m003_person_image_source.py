"""003: where a person's face photo came from.

- person_images.source  TEXT  'manual' (uploaded / assigned in the web UI) or 'auto'
  (learned by the engine from a confident recognition, see autolearn.py)

Auto-learned photos can be told apart in the UI and are the only ones the engine replaces.
"""

from __future__ import annotations

import sqlite3

VERSION = 3
DESCRIPTION = "person_images.source (manual / auto-learned face photos)"


def migrate(conn: sqlite3.Connection) -> None:
    cols = {r[1] for r in conn.execute("PRAGMA table_info(person_images)")}
    if "source" not in cols:
        conn.execute("ALTER TABLE person_images ADD COLUMN source TEXT NOT NULL DEFAULT 'manual'")
