"""Download public box-labelled images of rodents and computer mice (all saved as class 0 = rodent).

  Open Images V7  "Mouse" (/m/04rmv), "Hamster" (/m/03qrc), "Computer mouse" (/m/020lf)
  COCO 2017       "mouse" (computer mouse, many small ones on office desks)

Computer mice are included on purpose: the user wants mouse-like objects (computer mouse,
toy mouse) reported as rodents too.

Inputs (made by the commands in README "Training the model"):
  datasets/online_raw/oi_{train,validation,test}.csv   Open Images box rows for the 3 classes
  datasets/online_raw/annotations/instances_*2017.json COCO annotations
Output (YOLO layout, merged by train.py):
  datasets/rodent_online/images/*.jpg, labels/*.txt

  .venv/bin/python scripts/rodent/download_online.py
"""
import argparse
import csv
import json
import random
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

RAW = Path("datasets/online_raw")
OUT = Path("datasets/rodent_online")
OI_CLASSES = {"/m/04rmv", "/m/03qrc", "/m/020lf"}
MAX_SIDE = 1024


def open_images() -> dict[str, tuple[str, list]]:
    """{name: (url, [(x1, y1, x2, y2) normalised])}. Images with a group box are skipped:
    one box around several animals would teach wrong sizes."""
    boxes, grouped = defaultdict(list), set()
    for f in sorted(RAW.glob("oi_*.csv")):
        # split,ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,...
        for row in csv.reader(open(f)):
            split, image_id, label = row[0], row[1], row[3]
            if label not in OI_CLASSES:
                continue
            key = (split, image_id)
            if row[11] == "1":
                grouped.add(key)
            x1, x2, y1, y2 = map(float, row[5:9])
            boxes[key].append((x1, y1, x2, y2))
    return {f"oi_{iid}": (f"https://open-images-dataset.s3.amazonaws.com/{split}/{iid}.jpg", b)
            for (split, iid), b in boxes.items() if (split, iid) not in grouped}


def coco(limit: int) -> dict[str, tuple[str, list]]:
    out = {}
    for ann_file in sorted((RAW / "annotations").glob("instances_*2017.json")):
        data = json.load(open(ann_file))
        cat = next(c["id"] for c in data["categories"] if c["name"] == "mouse")
        imgs = {i["id"]: i for i in data["images"]}
        boxes, crowd = defaultdict(list), set()
        for a in data["annotations"]:
            if a["category_id"] != cat:
                continue
            if a.get("iscrowd"):
                crowd.add(a["image_id"])
            im = imgs[a["image_id"]]
            x, y, w, h = a["bbox"]
            boxes[a["image_id"]].append((x / im["width"], y / im["height"], (x + w) / im["width"], (y + h) / im["height"]))
        for iid, b in boxes.items():
            if iid not in crowd:
                out[f"coco_{iid}"] = (imgs[iid]["coco_url"], b)
    keys = sorted(out)
    random.seed(0)
    random.shuffle(keys)
    return {k: out[k] for k in keys[:limit]}


def fetch(item: tuple[str, tuple[str, list]]) -> str:
    name, (url, boxes) = item
    img_path, lbl_path = OUT / "images" / f"{name}.jpg", OUT / "labels" / f"{name}.txt"
    if img_path.is_file() and lbl_path.is_file():
        return "skip"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = r.read()
    except Exception:
        return "fail"
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return "fail"
    h, w = img.shape[:2]
    s = MAX_SIDE / max(h, w)
    if s < 1:  # smaller files; boxes are normalised so they stay valid
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    lines = []
    for x1, y1, x2, y2 in boxes:
        x1, y1, x2, y2 = (min(max(v, 0.0), 1.0) for v in (x1, y1, x2, y2))
        if x2 - x1 > 0.002 and y2 - y1 > 0.002:
            lines.append(f"0 {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
    if not lines:
        return "fail"
    cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    lbl_path.write_text("\n".join(lines) + "\n")
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-limit", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)

    items = {**open_images(), **coco(args.coco_limit)}
    print(f"{len(items)} images to fetch "
          f"({sum(k.startswith('oi_') for k in items)} Open Images, {sum(k.startswith('coco_') for k in items)} COCO)",
          flush=True)
    stats = defaultdict(int)
    with ThreadPoolExecutor(args.threads) as pool:
        for i, res in enumerate(pool.map(fetch, items.items()), 1):
            stats[res] += 1
            if i % 200 == 0:
                print(f"{i}/{len(items)} {dict(stats)}", flush=True)
    print(f"done: {dict(stats)}")


if __name__ == "__main__":
    main()
