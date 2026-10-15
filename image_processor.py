"""Face detection on a single image or a folder of images."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from detector import FaceDetector
from utils import OutputManager, draw_detections, draw_info, resize_for_display

if TYPE_CHECKING:
    from engine import LiveHub

logger = logging.getLogger(__name__)


def read_image(path: Path) -> np.ndarray | None:
    """cv2.imread that also works with non-ASCII paths."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    except OSError:
        return None


class ImageProcessor:
    """Runs the detector over image files and hands results to the OutputManager."""

    CAMERA_ID = "images"  # crops go to output/faces/images/

    def __init__(
        self,
        detector: FaceDetector,
        output: OutputManager,
        *,
        extensions: list[str],
        recursive: bool = True,
        show: bool = False,
        stop_event: threading.Event | None = None,
        hub: LiveHub | None = None,
    ) -> None:
        self.detector = detector
        self.output = output
        self.hub = hub
        self.extensions = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
        self.recursive = recursive
        self.show = show
        self.stop_event = stop_event or threading.Event()

    def collect_images(self, path: str | Path) -> list[Path]:
        p = Path(path).expanduser()
        if p.is_file():
            return [p] if p.suffix.lower() in self.extensions else []
        if p.is_dir():
            pattern = p.rglob("*") if self.recursive else p.glob("*")
            return sorted(f for f in pattern if f.is_file() and f.suffix.lower() in self.extensions)
        raise FileNotFoundError(f"Image path not found: {p}")

    def process_image(self, path: Path) -> int:
        """Detect faces in one image. Returns the number of faces found (-1 on read error)."""
        img = read_image(path)
        if img is None:
            logger.error("Cannot read image: %s", path)
            return -1

        t0 = time.perf_counter()
        detections = self.detector.detect(img)
        ms = (time.perf_counter() - t0) * 1000
        logger.info("%s: %d face(s) in %.0f ms", path.name, len(detections), ms)

        annotated = draw_detections(img.copy(), detections)
        draw_info(annotated, [f"{path.name} | faces: {len(detections)} | {ms:.0f} ms | {self.detector.device}"])
        # Images are never rate-limited: every image with faces gets saved.
        self.output.handle(self.CAMERA_ID, img, detections, source=str(path),
                           annotated=annotated, rate_limit=False)

        if self.hub is not None:
            self.hub.update_frame(self.CAMERA_ID, annotated)
            self.hub.update_status(self.CAMERA_ID, infer_ms=ms, faces=len(detections),
                                   names=[d.name for d in detections if d.name and d.name != "Unknown"])

        if self.show:
            cv2.imshow("Face Detection - images", resize_for_display(annotated, 1280, 800))
            key = cv2.waitKey(0) & 0xFF  # any key -> next image
            if key in (ord("q"), 27):
                logger.info("Quit requested from image viewer")
                self.stop_event.set()
        return len(detections)

    def run(self, path: str | Path) -> dict[str, Any]:
        files = self.collect_images(path)
        logger.info("Found %d image(s) in %s", len(files), path)
        summary = {"images": 0, "faces": 0, "errors": 0, "no_faces": 0}
        start = time.perf_counter()
        if self.hub is not None:
            self.hub.register(self.CAMERA_ID)
            self.hub.update_status(self.CAMERA_ID, state="processing", total=len(files))
        try:
            for f in files:
                if self.stop_event.is_set():
                    logger.info("Image processing interrupted")
                    break
                try:
                    n = self.process_image(f)
                except Exception:
                    logger.exception("Failed to process %s", f)
                    n = -1
                if n < 0:
                    summary["errors"] += 1
                    continue
                summary["images"] += 1
                summary["faces"] += n
                summary["no_faces"] += int(n == 0)
        finally:
            if self.show:
                cv2.destroyAllWindows()
                cv2.waitKey(1)
        summary["seconds"] = round(time.perf_counter() - start, 2)
        if self.hub is not None:
            self.hub.update_status(self.CAMERA_ID, state="done", faces=summary["faces"])
        logger.info("Image summary: %s", summary)
        return summary
