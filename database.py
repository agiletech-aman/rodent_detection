"""SQLite storage: cameras, persons (+ face embeddings), detections and runtime settings.

Everything the web UI manages lives here. config.yaml only bootstraps the first run
(see seed_from_config) and holds settings that need a restart (model, device, port).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from migrations import run_all as run_migrations

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS cameras (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    url         TEXT NOT NULL,
    enabled     INTEGER NOT NULL DEFAULT 1,
    frame_skip  INTEGER,                      -- NULL = use the global setting
    min_face_size  INTEGER,                   -- per-camera quality override (NULL = global)
    min_blur_score REAL,                      -- per-camera quality override (NULL = global)
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS persons (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    code        TEXT UNIQUE,                  -- optional employee / member id
    notes       TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS person_images (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id   INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    path        TEXT NOT NULL,
    embedding   BLOB NOT NULL,                -- float32[512], L2-normalised
    source      TEXT NOT NULL DEFAULT 'manual', -- 'manual' | 'auto' (learned from a confident recognition)
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_person_images_person ON person_images(person_id);
CREATE TABLE IF NOT EXISTS detections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id       TEXT NOT NULL,
    ts              TEXT NOT NULL,            -- ISO-8601 with timezone
    ts_epoch        REAL NOT NULL,            -- for range queries
    face_index      INTEGER NOT NULL DEFAULT 0,
    bbox            TEXT NOT NULL,            -- JSON [x1, y1, x2, y2]
    confidence      REAL NOT NULL,
    landmarks       TEXT,                     -- JSON [[x, y] * 5]
    person_id       INTEGER REFERENCES persons(id) ON DELETE SET NULL,
    name            TEXT,
    similarity      REAL,
    crop_path       TEXT,
    annotated_path  TEXT,
    frame_id        INTEGER,
    source          TEXT,
    embedding       BLOB,                     -- float32[512], L2-normalised, from the detection itself
    quality         TEXT                      -- JSON {score, size, blur, yaw, pitch}
);
CREATE INDEX IF NOT EXISTS idx_det_ts ON detections(ts_epoch);
CREATE INDEX IF NOT EXISTS idx_det_cam_ts ON detections(camera_id, ts_epoch);
CREATE INDEX IF NOT EXISTS idx_det_person_ts ON detections(person_id, ts_epoch);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

# Settings editable from the web UI at runtime. "cfg" is where the first-run value comes from.
SETTINGS_SCHEMA: list[dict[str, Any]] = [
    {"key": "detector.conf_threshold", "type": "float", "min": 0.05, "max": 0.99, "group": "Detection",
     "label": "Confidence threshold", "help": "Faces below this score are ignored."},
    {"key": "detector.min_face_size", "type": "int", "min": 0, "max": 2000, "group": "Detection",
     "label": "Minimum face size (px)", "help": "Ignore faces whose shorter side is smaller than this."},
    {"key": "rtsp.frame_skip", "type": "int", "min": 1, "max": 100, "group": "Detection",
     "label": "Default frame skip", "help": "Detect on every N-th frame. A camera's own value overrides this."},
    {"key": "recognition.enabled", "type": "bool", "group": "Recognition",
     "label": "Enable face recognition", "help": "Match faces against the people in Persons."},
    {"key": "recognition.similarity_threshold", "type": "float", "min": 0.1, "max": 0.95, "group": "Recognition",
     "label": "Similarity threshold", "help": "Higher = fewer false matches, more 'Unknown'. Typical 0.35-0.50."},
    {"key": "recognition.auto_learn", "type": "bool", "group": "Recognition",
     "label": "Learn from confident recognitions",
     "help": "When a stored face matches a person with high similarity, its embedding is added to that person "
             "(marked 'auto' in Persons), so later detections of them are recognised better."},
    {"key": "recognition.auto_learn_similarity", "type": "float", "min": 0.3, "max": 0.95, "group": "Recognition",
     "label": "Auto-learn min similarity",
     "help": "Only matches at least this sure are learned. Keep it well above the similarity threshold."},
    {"key": "recognition.auto_learn_max", "type": "int", "min": 1, "max": 500, "group": "Recognition",
     "label": "Auto-learned photos per person",
     "help": "When full, the oldest auto-learned photo is replaced. Manually added photos are never touched."},
    {"key": "quality.enabled", "type": "bool", "group": "Face quality",
     "label": "Store only good-quality faces",
     "help": "Faces failing any check below are still shown live (grey box) but not saved / posted."},
    {"key": "quality.min_face_size", "type": "int", "min": 0, "max": 1000, "group": "Face quality",
     "label": "Min face size (px)", "help": "Shorter side of the face box. Reliable recognition needs 80 px or more."},
    {"key": "quality.min_confidence", "type": "float", "min": 0.05, "max": 0.99, "group": "Face quality",
     "label": "Min confidence (det_score)"},
    {"key": "quality.min_blur_score", "type": "float", "min": 0, "max": 5000, "group": "Face quality",
     "label": "Min sharpness (blur score)",
     "help": "Laplacian variance of the face resized to 112x112. Higher = sharper. Check values in Image test."},
    {"key": "quality.max_yaw", "type": "float", "min": 0, "max": 90, "group": "Face quality",
     "label": "Max yaw (left/right turn, °)"},
    {"key": "quality.max_pitch", "type": "float", "min": 0, "max": 90, "group": "Face quality",
     "label": "Max pitch (up/down, °)", "help": "Estimated from the 5 landmarks (approximate)."},
    {"key": "quality.require_embedding", "type": "bool", "group": "Face quality",
     "label": "Require embedding",
     "help": "Store a face only if an ArcFace embedding could be computed (needed for Assign person / search)."},
    {"key": "dedup.enabled", "type": "bool", "group": "Tracking & duplicates",
     "label": "Track faces and store one best crop",
     "help": "Each face is tracked; only its best-quality crop is saved / posted, not every frame."},
    {"key": "dedup.best_window_sec", "type": "float", "min": 0, "max": 60, "group": "Tracking & duplicates",
     "label": "Best-crop window (s)",
     "help": "How long a new face is watched to pick its sharpest, most frontal crop before storing it."},
    {"key": "dedup.cooldown_sec", "type": "int", "min": 1, "max": 86400, "group": "Tracking & duplicates",
     "label": "Cooldown (seconds)",
     "help": "The same person / face is not stored again on the same camera within this time."},
    {"key": "dedup.similarity", "type": "float", "min": 0.2, "max": 0.95, "group": "Tracking & duplicates",
     "label": "Same-face similarity",
     "help": "Embedding similarity to treat a reappearing face as the same track. Lower = merges more."},
    {"key": "rodent.enabled", "type": "bool", "group": "Rodent detection",
     "label": "Detect rodents (rats / mice)",
     "help": "Runs the rodent model on every camera. Needs models/rodent.onnx (scripts/rodent/train.py)."},
    {"key": "rodent.conf_threshold", "type": "float", "min": 0.05, "max": 0.99, "group": "Rodent detection",
     "label": "Rodent confidence threshold", "help": "Boxes below this score are ignored. Raise it if you get false alerts."},
    {"key": "rodent.frame_skip", "type": "int", "min": 1, "max": 100, "group": "Rodent detection",
     "label": "Rodent frame skip", "help": "Run the rodent model on every N-th frame (separate from face detection)."},
    {"key": "rodent.confirm_hits", "type": "int", "min": 1, "max": 20, "group": "Rodent detection",
     "label": "Confirm: hits needed",
     "help": "An alert is raised only when a rodent is seen in this many of the last runs (see window)."},
    {"key": "rodent.confirm_window", "type": "int", "min": 1, "max": 30, "group": "Rodent detection",
     "label": "Confirm: window (runs)", "help": "Number of recent rodent-model runs the hits are counted over."},
    {"key": "rodent.cooldown_sec", "type": "int", "min": 0, "max": 86400, "group": "Rodent detection",
     "label": "Alert cooldown (s)", "help": "No new rodent alert on the same camera within this time."},
    {"key": "output.save_crops", "type": "bool", "group": "Storage", "label": "Save face crops"},
    {"key": "output.save_annotated", "type": "bool", "group": "Storage", "label": "Save annotated frames"},
    {"key": "output.event_interval_sec", "type": "float", "min": 0, "max": 3600, "group": "Storage",
     "label": "Event interval per camera (s)",
     "help": "Only used when tracking is off: at most one stored event per camera in this interval."},
    {"key": "output.crop_margin", "type": "float", "min": 0, "max": 1, "group": "Storage",
     "label": "Crop margin", "help": "Extra context around face crops (fraction of the box size)."},
    {"key": "retention.days", "type": "int", "min": 0, "max": 3650, "group": "Storage",
     "label": "Keep detections (days)", "help": "Older history rows and their images are deleted. 0 = keep forever."},
    {"key": "api.enabled", "type": "bool", "group": "REST API", "label": "POST detections to an API"},
    {"key": "api.url", "type": "str", "group": "REST API", "label": "API URL"},
    {"key": "api.auth_token", "type": "secret", "group": "REST API", "label": "Bearer token",
     "help": "Sent as 'Authorization: Bearer <token>'. Leave empty to keep the current value."},
]
SETTINGS_BY_KEY = {s["key"]: s for s in SETTINGS_SCHEMA}


def validate_settings(values: dict[str, Any]) -> dict[str, Any]:
    """Validate and coerce settings coming from the UI. Raises ValueError on bad input."""
    clean: dict[str, Any] = {}
    for key, value in values.items():
        spec = SETTINGS_BY_KEY.get(key)
        if spec is None:
            raise ValueError(f"Unknown setting: {key}")
        t = spec["type"]
        try:
            if t == "bool":
                v: Any = value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes", "on")
            elif t == "int":
                v = int(value)
            elif t == "float":
                v = float(value)
            else:
                v = str(value).strip()
        except (TypeError, ValueError):
            raise ValueError(f"{spec['label']}: invalid value {value!r}") from None
        if t in ("int", "float"):
            if v < spec.get("min", float("-inf")) or v > spec.get("max", float("inf")):
                raise ValueError(f"{spec['label']} must be between {spec['min']} and {spec['max']}")
        if key == "api.url" and v and not v.startswith(("http://", "https://")):
            raise ValueError("API URL must start with http:// or https://")
        clean[key] = v
    return clean


def iso_to_epoch(ts: str) -> float:
    try:
        return datetime.fromisoformat(ts).timestamp()
    except (TypeError, ValueError):
        return time.time()


class Database:
    """Thin thread-safe wrapper over one SQLite connection (WAL mode)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.executescript(SCHEMA)
            version = run_migrations(self._conn)
        logger.info("Database ready: %s (schema v%d)", self.path, version)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------ low level

    def _exec(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, tuple(params))

    def _all(self, sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, tuple(params)).fetchall()]

    def _one(self, sql: str, params: Iterable[Any] = ()) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(sql, tuple(params)).fetchone()
        return dict(row) if row else None

    # ----------------------------------------------------------------- meta

    def get_meta(self, key: str) -> str | None:
        row = self._one("SELECT value FROM meta WHERE key=?", (key,))
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self._exec("INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                   (key, value))

    # -------------------------------------------------------------- cameras

    def list_cameras(self) -> list[dict[str, Any]]:
        rows = self._all("SELECT * FROM cameras ORDER BY created_at, id")
        for r in rows:
            r["enabled"] = bool(r["enabled"])
        return rows

    def get_camera(self, camera_id: str) -> dict[str, Any] | None:
        row = self._one("SELECT * FROM cameras WHERE id=?", (camera_id,))
        if row:
            row["enabled"] = bool(row["enabled"])
        return row

    def create_camera(self, camera_id: str, name: str, url: str, enabled: bool = True,
                      frame_skip: int | None = None) -> dict[str, Any]:
        now = time.time()
        try:
            self._exec("INSERT INTO cameras(id, name, url, enabled, frame_skip, created_at, updated_at) "
                       "VALUES(?, ?, ?, ?, ?, ?, ?)", (camera_id, name, url, int(enabled), frame_skip, now, now))
        except sqlite3.IntegrityError:
            raise ValueError(f"Camera id '{camera_id}' already exists") from None
        return self.get_camera(camera_id)  # type: ignore[return-value]

    def update_camera(self, camera_id: str, **fields: Any) -> dict[str, Any] | None:
        allowed = {k: v for k, v in fields.items()
                   if k in ("name", "url", "enabled", "frame_skip", "min_face_size", "min_blur_score")}
        if "enabled" in allowed:
            allowed["enabled"] = int(bool(allowed["enabled"]))
        if allowed:
            cols = ", ".join(f"{k}=?" for k in allowed)
            self._exec(f"UPDATE cameras SET {cols}, updated_at=? WHERE id=?",
                       (*allowed.values(), time.time(), camera_id))
        return self.get_camera(camera_id)

    def delete_camera(self, camera_id: str) -> bool:
        return self._exec("DELETE FROM cameras WHERE id=?", (camera_id,)).rowcount > 0

    # -------------------------------------------------------------- persons

    _PERSON_SELECT = """
        SELECT p.*,
               (SELECT COUNT(*) FROM person_images pi WHERE pi.person_id = p.id) AS image_count,
               (SELECT path FROM person_images pi WHERE pi.person_id = p.id ORDER BY pi.id LIMIT 1) AS thumb_path,
               (SELECT MAX(ts_epoch) FROM detections d WHERE d.person_id = p.id) AS last_seen,
               (SELECT camera_id FROM detections d WHERE d.person_id = p.id
                 ORDER BY ts_epoch DESC LIMIT 1) AS last_camera
        FROM persons p
    """

    def list_persons(self, search: str | None = None) -> list[dict[str, Any]]:
        if search:
            like = f"%{search}%"
            return self._all(self._PERSON_SELECT + " WHERE p.name LIKE ? OR p.code LIKE ? "
                             "ORDER BY p.name COLLATE NOCASE", (like, like))
        return self._all(self._PERSON_SELECT + " ORDER BY p.name COLLATE NOCASE")

    def get_person(self, person_id: int) -> dict[str, Any] | None:
        return self._one(self._PERSON_SELECT + " WHERE p.id=?", (person_id,))

    def create_person(self, name: str, code: str | None = None, notes: str = "") -> dict[str, Any]:
        now = time.time()
        try:
            cur = self._exec("INSERT INTO persons(name, code, notes, created_at, updated_at) VALUES(?, ?, ?, ?, ?)",
                             (name, code or None, notes, now, now))
        except sqlite3.IntegrityError:
            raise ValueError(f"Person code '{code}' is already used") from None
        return self.get_person(int(cur.lastrowid))  # type: ignore[arg-type,return-value]

    def update_person(self, person_id: int, **fields: Any) -> dict[str, Any] | None:
        allowed = {k: v for k, v in fields.items() if k in ("name", "code", "notes")}
        if "code" in allowed:
            allowed["code"] = allowed["code"] or None
        if allowed:
            cols = ", ".join(f"{k}=?" for k in allowed)
            try:
                self._exec(f"UPDATE persons SET {cols}, updated_at=? WHERE id=?",
                           (*allowed.values(), time.time(), person_id))
            except sqlite3.IntegrityError:
                raise ValueError(f"Person code '{allowed.get('code')}' is already used") from None
            if "name" in allowed:  # keep history readable
                self._exec("UPDATE detections SET name=? WHERE person_id=?", (allowed["name"], person_id))
        return self.get_person(person_id)

    def delete_person(self, person_id: int) -> list[str]:
        """Delete a person and their images. Returns image paths for the caller to remove from disk."""
        paths = [r["path"] for r in self._all("SELECT path FROM person_images WHERE person_id=?", (person_id,))]
        self._exec("DELETE FROM persons WHERE id=?", (person_id,))
        return paths

    def list_person_images(self, person_id: int) -> list[dict[str, Any]]:
        return self._all("SELECT id, person_id, path, source, created_at FROM person_images WHERE person_id=? "
                         "ORDER BY id", (person_id,))

    def person_embeddings(self, person_id: int) -> list[dict[str, Any]]:
        """All face embeddings of one person, oldest first: [{id, source, embedding}]."""
        rows = self._all("SELECT id, source, embedding FROM person_images WHERE person_id=? ORDER BY id", (person_id,))
        return [{"id": r["id"], "source": r["source"], "embedding": np.frombuffer(r["embedding"], np.float32)}
                for r in rows]

    def add_person_image(self, person_id: int, path: str, embedding: np.ndarray, source: str = "manual") -> int:
        emb = np.asarray(embedding, dtype=np.float32).ravel()
        cur = self._exec("INSERT INTO person_images(person_id, path, embedding, source, created_at) "
                         "VALUES(?, ?, ?, ?, ?)", (person_id, path, emb.tobytes(), source, time.time()))
        self._exec("UPDATE persons SET updated_at=? WHERE id=?", (time.time(), person_id))
        return int(cur.lastrowid)  # type: ignore[arg-type]

    def delete_person_image(self, person_id: int, image_id: int) -> str | None:
        row = self._one("SELECT path FROM person_images WHERE id=? AND person_id=?", (image_id, person_id))
        if row is None:
            return None
        self._exec("DELETE FROM person_images WHERE id=?", (image_id,))
        return row["path"]

    def gallery(self) -> list[tuple[int, str, np.ndarray]]:
        """One averaged, re-normalised embedding per person: [(person_id, name, vector)]."""
        rows = self._all("SELECT pi.person_id, p.name, pi.embedding FROM person_images pi "
                         "JOIN persons p ON p.id = pi.person_id ORDER BY pi.person_id")
        grouped: dict[int, tuple[str, list[np.ndarray]]] = {}
        for r in rows:
            grouped.setdefault(r["person_id"], (r["name"], []))[1].append(np.frombuffer(r["embedding"], np.float32))
        out = []
        for pid, (name, embs) in grouped.items():
            mean = np.mean(embs, axis=0)
            out.append((pid, name, (mean / (np.linalg.norm(mean) + 1e-12)).astype(np.float32)))
        return out

    # ----------------------------------------------------------- detections

    def insert_detections(self, records: list[dict[str, Any]],
                          embeddings: list[np.ndarray | None] | None = None) -> None:
        embeddings = embeddings or [None] * len(records)
        rows = [(
            r["camera_id"], r["timestamp"], iso_to_epoch(r["timestamp"]), r.get("face_index", 0),
            json.dumps(r["bbox"]), r["confidence"], json.dumps(r.get("landmarks")),
            r.get("person_id"), r.get("name"), r.get("similarity"), r.get("crop_path"),
            r.get("annotated_path"), r.get("frame_id"), r.get("source"),
            None if e is None else np.asarray(e, np.float32).ravel().tobytes(),
            json.dumps(r["quality"]) if r.get("quality") else None,
        ) for r, e in zip(records, embeddings)]
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                self._conn.executemany(
                    "INSERT INTO detections(camera_id, ts, ts_epoch, face_index, bbox, confidence, landmarks, "
                    "person_id, name, similarity, crop_path, annotated_path, frame_id, source, embedding, quality) "
                    "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    @staticmethod
    def _det_filters(camera_id: str | None, person_id: int | None, known: bool | None,
                     since: float | None, until: float | None,
                     with_embedding: bool | None = True) -> tuple[str, list[Any]]:
        where, params = [], []
        if with_embedding is True:
            where.append("embedding IS NOT NULL")
        elif with_embedding is False:
            where.append("embedding IS NULL")
        if camera_id:
            where.append("camera_id=?")
            params.append(camera_id)
        if person_id is not None:
            where.append("person_id=?")
            params.append(person_id)
        if known is True:
            where.append("person_id IS NOT NULL")
        elif known is False:
            where.append("person_id IS NULL")
        if since is not None:
            where.append("ts_epoch>=?")
            params.append(since)
        if until is not None:
            where.append("ts_epoch<?")
            params.append(until)
        return (" WHERE " + " AND ".join(where)) if where else "", params

    def query_detections(self, *, camera_id: str | None = None, person_id: int | None = None,
                         known: bool | None = None, since: float | None = None, until: float | None = None,
                         limit: int = 50, offset: int = 0,
                         with_embedding: bool | None = True) -> tuple[list[dict[str, Any]], int]:
        """Newest first. with_embedding=True (default) hides old records saved without an embedding."""
        clause, params = self._det_filters(camera_id, person_id, known, since, until, with_embedding)
        total = self._one(f"SELECT COUNT(*) AS n FROM detections{clause}", params)["n"]  # type: ignore[index]
        rows = self._all(f"SELECT * FROM detections{clause} ORDER BY ts_epoch DESC, id DESC LIMIT ? OFFSET ?",
                         (*params, limit, offset))
        for r in rows:
            r["bbox"] = json.loads(r["bbox"])
            r["landmarks"] = json.loads(r["landmarks"]) if r["landmarks"] else None
            r["quality"] = json.loads(r["quality"]) if r.get("quality") else None
            r["has_embedding"] = r.pop("embedding", None) is not None  # never send the blob out
        return rows, total

    def get_detection(self, det_id: int) -> dict[str, Any] | None:
        """Full row; 'embedding' is returned as a numpy vector (or None for old records)."""
        row = self._one("SELECT * FROM detections WHERE id=?", (det_id,))
        if row is not None:
            row["embedding"] = np.frombuffer(row["embedding"], np.float32) if row.get("embedding") else None
        return row

    def label_detection(self, det_id: int, person_id: int, name: str) -> None:
        self._exec("UPDATE detections SET person_id=?, name=? WHERE id=?", (person_id, name, det_id))

    def delete_detections(self, ids: list[int]) -> tuple[int, list[str]]:
        """Delete detection records. Returns (rows deleted, image files no other record uses)."""
        if not ids:
            return 0, []
        q = ",".join("?" * len(ids))
        with self._lock:
            rows = self._all(f"SELECT crop_path, annotated_path FROM detections WHERE id IN ({q})", ids)
            deleted = self._exec(f"DELETE FROM detections WHERE id IN ({q})", ids).rowcount
            files = {p for r in rows for p in (r["crop_path"], r["annotated_path"]) if p}
            used: set[str] = set()
            if files:
                q2 = ",".join("?" * len(files))
                used = {r["p"] for r in self._all(
                    f"SELECT crop_path AS p FROM detections WHERE crop_path IN ({q2}) "
                    f"UNION SELECT annotated_path FROM detections WHERE annotated_path IN ({q2})",
                    [*files, *files])}
        return deleted, sorted(files - used)

    def delete_detections_before(self, epoch: float) -> list[str]:
        """Delete old history rows. Returns image paths that are no longer referenced."""
        with self._lock:
            rows = self._all("SELECT crop_path, annotated_path FROM detections WHERE ts_epoch<?", (epoch,))
            self._exec("DELETE FROM detections WHERE ts_epoch<?", (epoch,))
        return [p for r in rows for p in (r["crop_path"], r["annotated_path"]) if p]

    def stats(self, since: float) -> dict[str, Any]:
        """Counts for the dashboard: same records as the Detections page (only those with an embedding)."""
        row = self._one("SELECT COUNT(*) AS faces, SUM(person_id IS NOT NULL) AS known, "
                        "COUNT(DISTINCT person_id) AS people FROM detections "
                        "WHERE ts_epoch>=? AND embedding IS NOT NULL", (since,))
        per_cam = self._all("SELECT camera_id, COUNT(*) AS faces FROM detections WHERE ts_epoch>=? "
                            "AND embedding IS NOT NULL GROUP BY camera_id ORDER BY faces DESC", (since,))
        old = self._one("SELECT COUNT(*) AS n FROM detections WHERE embedding IS NULL")
        return {"faces": row["faces"] or 0, "known": row["known"] or 0, "people": row["people"] or 0,  # type: ignore[index]
                "per_camera": per_cam, "old_without_embedding": old["n"] if old else 0}

    # -------------------------------------------------------- rodent events

    def insert_rodent_event(self, r: dict[str, Any]) -> int:
        cur = self._exec(
            "INSERT INTO rodent_events(camera_id, ts, ts_epoch, bbox, confidence, count, boxes, crop_path, "
            "annotated_path, frame_id) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (r["camera_id"], r["timestamp"], iso_to_epoch(r["timestamp"]), json.dumps(r["bbox"]), r["confidence"],
             r.get("count", 1), json.dumps(r.get("boxes")), r.get("crop_path"), r.get("annotated_path"),
             r.get("frame_id")))
        return int(cur.lastrowid)

    def query_rodent_events(self, *, camera_id: str | None = None, since: float | None = None,
                            until: float | None = None, limit: int = 50,
                            offset: int = 0) -> tuple[list[dict[str, Any]], int]:
        """Newest first."""
        clause, params = self._det_filters(camera_id, None, None, since, until, with_embedding=None)
        total = self._one(f"SELECT COUNT(*) AS n FROM rodent_events{clause}", params)["n"]  # type: ignore[index]
        rows = self._all(f"SELECT * FROM rodent_events{clause} ORDER BY ts_epoch DESC, id DESC LIMIT ? OFFSET ?",
                         (*params, limit, offset))
        for r in rows:
            r["bbox"] = json.loads(r["bbox"])
            r["boxes"] = json.loads(r["boxes"]) if r.get("boxes") else None
        return rows, total

    def delete_rodent_events(self, ids: list[int]) -> tuple[int, list[str]]:
        """Delete rodent alerts. Returns (rows deleted, their image files)."""
        if not ids:
            return 0, []
        q = ",".join("?" * len(ids))
        with self._lock:
            rows = self._all(f"SELECT crop_path, annotated_path FROM rodent_events WHERE id IN ({q})", ids)
            deleted = self._exec(f"DELETE FROM rodent_events WHERE id IN ({q})", ids).rowcount
        return deleted, [p for r in rows for p in (r["crop_path"], r["annotated_path"]) if p]

    def delete_rodent_events_before(self, epoch: float) -> list[str]:
        with self._lock:
            rows = self._all("SELECT crop_path, annotated_path FROM rodent_events WHERE ts_epoch<?", (epoch,))
            self._exec("DELETE FROM rodent_events WHERE ts_epoch<?", (epoch,))
        return [p for r in rows for p in (r["crop_path"], r["annotated_path"]) if p]

    def rodent_stats(self, since: float) -> dict[str, Any]:
        row = self._one("SELECT COUNT(*) AS n, MAX(ts_epoch) AS last FROM rodent_events WHERE ts_epoch>=?", (since,))
        last = self._one("SELECT camera_id, ts_epoch FROM rodent_events ORDER BY ts_epoch DESC LIMIT 1")
        return {"alerts": (row or {}).get("n") or 0, "last": last}

    # ------------------------------------------------------------- settings

    def get_settings(self) -> dict[str, Any]:
        return {r["key"]: json.loads(r["value"]) for r in self._all("SELECT key, value FROM settings")}

    def set_settings(self, values: dict[str, Any]) -> None:
        with self._lock:
            for k, v in values.items():
                self._exec("INSERT INTO settings(key, value) VALUES(?, ?) "
                           "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (k, json.dumps(v)))


def _cfg_lookup(cfg: dict[str, Any], dotted: str) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def seed_from_config(db: Database, cfg: dict[str, Any]) -> None:
    """First-run import from config.yaml. Later runs use the database (the web UI is the source of truth)."""
    existing = db.get_settings()
    missing = {}
    for spec in SETTINGS_SCHEMA:
        if spec["key"] not in existing:
            value = _cfg_lookup(cfg, spec["key"])
            if value is not None and not (isinstance(value, str) and "${" in value):
                missing[spec["key"]] = value
            elif spec["type"] in ("str", "secret"):
                missing[spec["key"]] = ""
    if missing:
        db.set_settings(validate_settings(missing))
        logger.info("Seeded %d setting(s) from config.yaml", len(missing))

    if db.get_meta("seeded_cameras") is None:
        count = 0
        for cam in cfg.get("cameras") or []:
            try:
                db.create_camera(str(cam["id"]), str(cam.get("name") or cam["id"]), str(cam["url"]),
                                 bool(cam.get("enabled", True)), cam.get("frame_skip"))
                count += 1
            except (KeyError, ValueError) as exc:
                logger.warning("Skipping camera from config.yaml: %s", exc)
        db.set_meta("seeded_cameras", "1")
        logger.info("Imported %d camera(s) from config.yaml", count)
