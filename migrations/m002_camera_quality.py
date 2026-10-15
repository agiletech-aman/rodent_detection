"""002: per-camera face-quality overrides.

Small / far CCTV faces can never reach the global size limit; a camera can now use its own
values (NULL = use the global setting from Settings -> Face quality):

- cameras.min_face_size   INTEGER  px, shorter side of the face box
- cameras.min_blur_score  REAL     Laplacian variance of the face resized to 112x112
"""

from __future__ import annotations

import sqlite3

VERSION = 2
DESCRIPTION = "cameras.min_face_size + cameras.min_blur_score (per-camera quality overrides)"


def migrate(conn: sqlite3.Connection) -> None:
    cols = {r[1] for r in conn.execute("PRAGMA table_info(cameras)")}
    if "min_face_size" not in cols:
        conn.execute("ALTER TABLE cameras ADD COLUMN min_face_size INTEGER")
    if "min_blur_score" not in cols:
        conn.execute("ALTER TABLE cameras ADD COLUMN min_blur_score REAL")
