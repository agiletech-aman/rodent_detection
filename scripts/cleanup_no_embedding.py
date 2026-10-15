#!/usr/bin/env python3
"""Delete detection records that have no saved embedding (created before migration 001),
together with their crop / annotated images that no remaining record uses.

    python scripts/cleanup_no_embedding.py --dry-run     # show what would be deleted
    python scripts/cleanup_no_embedding.py               # delete (asks for confirmation)
    python scripts/cleanup_no_embedding.py --yes         # delete without asking

Safe to run while the app is running (SQLite WAL). Persons and their photos are not touched.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from migrations import run_all  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Remove detection records without an embedding")
    p.add_argument("--db", default=str(ROOT / "data" / "app.db"), help="SQLite database (default: data/app.db)")
    p.add_argument("--dry-run", action="store_true", help="only report, change nothing")
    p.add_argument("--yes", action="store_true", help="don't ask for confirmation")
    args = p.parse_args()

    db = Path(args.db)
    if not db.is_file():
        print(f"Database not found: {db}")
        return 1
    conn = sqlite3.connect(str(db), isolation_level=None)
    conn.execute("PRAGMA busy_timeout=10000")
    run_all(conn)  # make sure the embedding column exists

    old = conn.execute("SELECT id, crop_path, annotated_path FROM detections WHERE embedding IS NULL").fetchall()
    kept = conn.execute("SELECT COUNT(*) FROM detections WHERE embedding IS NOT NULL").fetchone()[0]
    files = {f for _, c, a in old for f in (c, a) if f}
    still_used = {f for (f,) in conn.execute(
        "SELECT crop_path FROM detections WHERE embedding IS NOT NULL AND crop_path IS NOT NULL "
        "UNION SELECT annotated_path FROM detections WHERE embedding IS NOT NULL AND annotated_path IS NOT NULL")}
    to_delete = sorted(files - still_used)
    existing = [f for f in to_delete if Path(f).is_file()]

    print(f"Records without embedding : {len(old)}")
    print(f"Records kept (embedding)  : {kept}")
    print(f"Image files to delete     : {len(existing)} (of {len(to_delete)} referenced)")
    if args.dry_run or not old:
        print("Dry run: nothing changed." if args.dry_run else "Nothing to clean.")
        return 0
    if not args.yes and input("Delete them? [y/N] ").strip().lower() != "y":
        print("Aborted.")
        return 0

    conn.execute("DELETE FROM detections WHERE embedding IS NULL")
    removed = 0
    for f in existing:
        try:
            Path(f).unlink()
            removed += 1
        except OSError as exc:
            print(f"  could not delete {f}: {exc}")
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()
    print(f"Deleted {len(old)} records and {removed} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
