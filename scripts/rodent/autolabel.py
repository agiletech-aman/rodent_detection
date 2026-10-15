"""Auto-label the rodent classification dataset with bounding boxes.

The source dataset (archive/<Class>/*.jpg) only says WHICH animal is in an image,
not WHERE. A detector needs boxes, so an open-vocabulary detector (YOLO-World)
finds the animal in every image; the folder already tells us it is a rodent.
All classes are merged into a single class 0 = rodent.

Output (YOLO format):
  datasets/rodent/images/{train,val}/*.jpg
  datasets/rodent/labels/{train,val}/*.txt
  datasets/rodent/rejected.txt   images where no confident box was found

Run with the training venv:
  .venv-train/bin/python scripts/rodent/autolabel.py
"""
import argparse
import hashlib
import random
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLOWorld

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# "rat"/"mouse" score poorly in YOLO-World (CLIP confuses them, e.g. computer mouse), while
# "animal" finds them reliably; the folder already says it is a rodent. "person" and "hand"
# are distractors so people holding a rodent don't get boxed as the animal.
PROMPTS = ["animal", "person", "hand"]
KEEP_CLASS = 0


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def pick_boxes(boxes, min_conf, rel_conf):
    """Class-agnostic NMS, then keep boxes close to the best score of the image."""
    boxes = sorted(boxes, key=lambda b: -b[4])
    kept = []
    for b in boxes:
        if all(iou(b, k) < 0.5 for k in kept):
            kept.append(b)
    if not kept or kept[0][4] < min_conf:
        return []
    top = kept[0][4]
    return [b for b in kept if b[4] >= max(min_conf, rel_conf * top)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="archive")
    ap.add_argument("--out", default="datasets/rodent")
    ap.add_argument("--model", default="yolov8l-worldv2.pt")
    ap.add_argument("--min-conf", type=float, default=0.25)
    ap.add_argument("--rel-conf", type=float, default=0.4)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)

    # Deduplicate identical files so the same image can't land in both train and val.
    files, seen = [], set()
    for f in sorted(src.rglob("*")):
        if f.suffix.lower() not in EXTS:
            continue
        h = hashlib.md5(f.read_bytes()).hexdigest()
        if h not in seen:
            seen.add(h)
            files.append(f)
    print(f"{len(files)} unique images")

    random.seed(42)
    random.shuffle(files)
    n_val = int(len(files) * args.val_frac)

    model = YOLOWorld(args.model)
    model.set_classes(PROMPTS)

    rejected, n_boxes = [], 0
    for i, f in enumerate(files):
        img = cv2.imread(str(f))
        if img is None:
            rejected.append(f"{f}\tunreadable")
            continue
        h, w = img.shape[:2]
        r = model.predict(img, conf=0.05, imgsz=args.imgsz, verbose=False)[0]
        raw = [(*b.xyxy[0].tolist(), float(b.conf)) for b in r.boxes if int(b.cls) == KEEP_CLASS]
        boxes = pick_boxes(raw, args.min_conf, args.rel_conf)
        if not boxes:
            rejected.append(f"{f}\tno box (best {max((b[4] for b in raw), default=0):.2f})")
            continue

        split = "val" if i < n_val else "train"
        stem = f"{f.parent.name.lower()}_{i:05d}"
        cv2.imwrite(str(out / "images" / split / f"{stem}.jpg"), img)  # normalise png/jpeg -> jpg
        with open(out / "labels" / split / f"{stem}.txt", "w") as fh:
            for x1, y1, x2, y2, _ in boxes:
                cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                fh.write(f"0 {cx:.6f} {cy:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n")
        n_boxes += len(boxes)
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(files)}  kept={i + 1 - len(rejected)}  rejected={len(rejected)}", flush=True)

    (out / "rejected.txt").write_text("\n".join(rejected) + "\n")
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: rodent\n"
    )
    print(f"done: {len(files) - len(rejected)} labelled images, {n_boxes} boxes, {len(rejected)} rejected")


if __name__ == "__main__":
    main()
