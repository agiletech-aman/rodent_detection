"""Grab background frames from the cameras as negative (no rodent) training images.

The rodent dataset only has close-up animal photos. Without examples of the real
camera scene, the detector tends to fire on floor clutter, shadows and bags.
Frames are saved without labels; YOLO treats an image with no label file as background.

Only run this while no rodent is in view.
  .venv/bin/python scripts/rodent/capture_negatives.py --minutes 10 --every 5
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from database import Database  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/app.db")
    ap.add_argument("--out", default="datasets/rodent_negatives")
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--every", type=float, default=5, help="seconds between saved frames")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cams = [c for c in Database(args.db).list_cameras() if c["enabled"]]
    caps = {}
    for c in cams:
        src = c["url"]
        cap = cv2.VideoCapture(int(src.split(":", 1)[1]) if src.startswith("device:") else src)
        if cap.isOpened():
            caps[c["id"]] = cap
            print(f"[{c['id']}] opened")
        else:
            print(f"[{c['id']}] could not open, skipped")
    if not caps:
        sys.exit("no camera could be opened")

    end, n, last = time.time() + args.minutes * 60, 0, 0.0
    while time.time() < end:
        frames = {cid: cap.read() for cid, cap in caps.items()}  # keep reading so buffers stay fresh
        if time.time() - last < args.every:
            continue
        last = time.time()
        for cid, (ok, frame) in frames.items():
            if ok:
                cv2.imwrite(str(out / f"{cid}_{int(last * 1000)}.jpg"), frame)
                n += 1
        print(f"saved {n} frames", end="\r", flush=True)
    print(f"\nsaved {n} frames to {out}")


if __name__ == "__main__":
    main()
