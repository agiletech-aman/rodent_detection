"""Database migrations. Each module has VERSION and migrate(conn); they run in order, once.

Runs automatically when the app opens the database (database.Database), or by hand:

    python -m migrations --db data/app.db
"""

from __future__ import annotations

import logging
import sqlite3

from migrations import (m001_detection_embeddings, m002_camera_quality, m003_person_image_source,
                        m004_rodent_events)

logger = logging.getLogger(__name__)

MIGRATIONS = [m001_detection_embeddings, m002_camera_quality, m003_person_image_source, m004_rodent_events]


def current_version(conn: sqlite3.Connection) -> int:
    conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    return int(row[0]) if row else 0


def run_all(conn: sqlite3.Connection) -> int:
    """Apply pending migrations. Returns the schema version afterwards."""
    version = current_version(conn)
    for m in MIGRATIONS:
        if m.VERSION <= version:
            continue
        logger.info("Applying DB migration %03d: %s", m.VERSION, m.DESCRIPTION)
        conn.execute("BEGIN")
        try:
            m.migrate(conn)
            conn.execute("INSERT INTO meta(key, value) VALUES('schema_version', ?) "
                         "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (str(m.VERSION),))
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        version = m.VERSION
    return version
