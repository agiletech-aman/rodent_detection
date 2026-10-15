"""Train the rodent detector and export it to ONNX for the app.

  1. scripts/rodent/autolabel.py          archive/ -> datasets/rodent (boxes from YOLO-World)
     scripts/rodent/download_online.py    Open Images + COCO rodents / computer mice -> datasets/rodent_online
  2. scripts/rodent/capture_negatives.py  camera frames without rodents -> datasets/rodent_negatives
     + web UI Rodents -> Training frames   camera frames labelled by hand -> datasets/rodent_camera
  3. scripts/rodent/train.py              train YOLO11n, export models/rodent.onnx
  4. scripts/rodent/train.py --relabel runs/rodent/r1/weights/best.pt --name r2
     (optional second round: the trained model labels the photos YOLO-World could not,
      then trains again on the bigger set)

Run with the training venv (PyTorch + ultralytics); the app itself only needs onnxruntime:
  .venv-train/bin/python scripts/rodent/train.py --epochs 40
"""
import argparse
import random
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO

DATA = Path("datasets/rodent")


def add_negatives(neg_dir: Path, val_frac: float) -> int:
    """Copy background frames into the dataset without label files (= no objects)."""
    files = sorted(neg_dir.glob("*.jpg")) if neg_dir.is_dir() else []
    random.seed(1)
    random.shuffle(files)
    n_val = int(len(files) * val_frac)
    for split in ("train", "val"):
        for old in (DATA / "images" / split).glob("neg_*"):
            old.unlink()
    for i, f in enumerate(files):
        shutil.copy(f, DATA / "images" / ("val" if i < n_val else "train") / f"neg_{f.name}")
    return len(files)


def add_extra(src: Path, prefix: str, repeat: int, val_frac: float) -> tuple[int, int]:
    """Merge another YOLO-layout folder (images/, labels/) into the dataset.

    Camera frames labelled in the web UI are few but show the real scene, so each one is repeated
    in train (augmentation makes every copy look different) and some go to val so the reported
    accuracy is measured on this camera too. Images without a label file are skipped."""
    for split in ("train", "val"):
        for old in (DATA / "images" / split).glob(f"{prefix}_*"):
            old.unlink()
            (DATA / "labels" / split / f"{old.stem}.txt").unlink(missing_ok=True)
    pairs = [(img, src / "labels" / f"{img.stem}.txt") for img in sorted((src / "images").glob("*.jpg"))]
    pairs = [p for p in pairs if p[1].is_file()]
    random.seed(2)
    random.shuffle(pairs)
    n_val = int(len(pairs) * val_frac) if len(pairs) >= 10 else 0
    n_boxes = 0
    for i, (img, lbl) in enumerate(pairs):
        copies = [("val", 0)] if i < n_val else [("train", k) for k in range(repeat)]
        n_boxes += len(lbl.read_text().split())  // 5
        for split, k in copies:
            stem = f"{prefix}_{img.stem}_{k}"
            shutil.copy(img, DATA / "images" / split / f"{stem}.jpg")
            shutil.copy(lbl, DATA / "labels" / split / f"{stem}.txt")
    return len(pairs), n_boxes


def relabel(weights: str, conf: float) -> int:
    """Label the photos YOLO-World rejected using our own trained model (they are all rodents)."""
    model = YOLO(weights)
    added = 0
    rejected = DATA / "rejected.txt"
    for line in rejected.read_text().splitlines():
        path = Path(line.split("\t")[0])
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes = [b for b in model.predict(img, conf=conf, verbose=False)[0].boxes]
        if not boxes:
            continue
        stem = f"r2_{path.parent.name.lower()}_{added:05d}"
        cv2.imwrite(str(DATA / "images" / "train" / f"{stem}.jpg"), img)
        with open(DATA / "labels" / "train" / f"{stem}.txt", "w") as fh:
            for b in boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                fh.write(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n")
        added += 1
    print(f"relabel: added {added} of the rejected photos (conf >= {conf})")
    return added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.pt", help="start weights")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", default="r1")
    ap.add_argument("--negatives", default="datasets/rodent_negatives")
    ap.add_argument("--camera-frames", default="datasets/rodent_camera")
    ap.add_argument("--camera-repeat", type=int, default=5)
    ap.add_argument("--online", default="datasets/rodent_online", help="scripts/rodent/download_online.py output")
    ap.add_argument("--relabel", help="weights used to label the rejected photos first (round 2)")
    ap.add_argument("--relabel-conf", type=float, default=0.5)
    ap.add_argument("--export", default="models/rodent.onnx")
    args = ap.parse_args()

    for old in (DATA / "images" / "train").glob("r2_*"):  # make re-runs repeatable
        old.unlink()
        (DATA / "labels" / "train" / f"{old.stem}.txt").unlink(missing_ok=True)
    if args.relabel:
        relabel(args.relabel, args.relabel_conf)
    print(f"negatives: {add_negatives(Path(args.negatives), 0.15)} background frames")
    n_on, n_on_boxes = add_extra(Path(args.online), "web", 1, 0.1)
    print(f"online images: {n_on} ({n_on_boxes} boxes: rodents, hamsters, computer mice)")
    n_cam, n_cam_boxes = add_extra(Path(args.camera_frames), "cam", args.camera_repeat, 0.2)
    print(f"camera frames: {n_cam} labelled ({n_cam_boxes} rodent boxes), x{args.camera_repeat} in train")
    for c in DATA.glob("labels/*.cache"):
        c.unlink()

    model = YOLO(args.relabel or args.model)
    model.train(
        data=str(DATA / "data.yaml"), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        workers=args.workers, device="cpu", project="runs/rodent", name=args.name, exist_ok=True,
        patience=10, single_cls=True, plots=True,
        # CCTV: rodents are small and often in IR / greyscale -> zoom out more, vary colour more
        scale=0.7, hsv_s=0.8, degrees=5, fliplr=0.5, mosaic=1.0, close_mosaic=5,
    )
    best = Path("runs/rodent") / args.name / "weights" / "best.pt"
    metrics = YOLO(str(best)).val(data=str(DATA / "data.yaml"), imgsz=args.imgsz, device="cpu", plots=False)
    print(f"val: mAP50={metrics.box.map50:.3f} mAP50-95={metrics.box.map:.3f} "
          f"precision={metrics.box.mp:.3f} recall={metrics.box.mr:.3f}")

    onnx = Path(YOLO(str(best)).export(format="onnx", imgsz=args.imgsz, simplify=True, dynamic=False))
    out = Path(args.export)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(onnx, out)
    print(f"exported {out}")


if __name__ == "__main__":
    main()
