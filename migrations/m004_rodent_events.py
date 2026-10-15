"""004: rodent alerts.

- rodent_events  one row per confirmed rodent sighting (see rodent.RodentWatcher):
  best box + confidence, all boxes of that frame, crop / annotated image paths.
"""

from __future__ import annotations

import sqlite3

VERSION = 4
DESCRIPTION = "rodent_events table (rat / mouse detections)"


def migrate(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS rodent_events (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id       TEXT NOT NULL,
            ts              TEXT NOT NULL,            -- ISO-8601 with timezone
            ts_epoch        REAL NOT NULL,
            bbox            TEXT NOT NULL,            -- JSON [x1, y1, x2, y2] of the best box
            confidence      REAL NOT NULL,
            count           INTEGER NOT NULL DEFAULT 1,
            boxes           TEXT,                     -- JSON [{bbox, confidence}] all boxes in the frame
            crop_path       TEXT,
            annotated_path  TEXT,
            frame_id        INTEGER
        )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_rodent_ts ON rodent_events(ts_epoch)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_rodent_cam_ts ON rodent_events(camera_id, ts_epoch)")
