"""python -m migrations --db data/app.db   (safe to run more than once)"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path

from migrations import current_version, run_all


def main() -> int:
    p = argparse.ArgumentParser(description="Apply database migrations")
    p.add_argument("--db", default="data/app.db", help="path to the SQLite database")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not Path(args.db).is_file():
        print(f"Database not found: {args.db}")
        return 1
    conn = sqlite3.connect(args.db, isolation_level=None)
    try:
        before = current_version(conn)
        after = run_all(conn)
    finally:
        conn.close()
    print(f"schema version {before} -> {after}" if after != before else f"already up to date (version {after})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
