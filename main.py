#!/usr/bin/env python3
"""Face Detection System - CLI entry point.

Examples:
    python main.py --mode web                     # everything from the browser (recommended)
    python main.py --mode rtsp --show             # cameras from the database, OpenCV window
    python main.py --mode image --path ./images
    python main.py --mode both
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from typing import Any

from database import Database, seed_from_config
from detector import FaceDetector
from engine import DetectionEngine, LiveHub, RetentionWorker, apply_runtime_settings
from rodent import RodentDetector
from enrollment import import_known_faces_dir
from image_processor import ImageProcessor
from rtsp_stream import configure_ffmpeg_transport
from utils import OutputManager, has_display, load_config, setup_logging

logger = logging.getLogger("main")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Face detection on RTSP streams and images (InsightFace SCRFD)")
    p.add_argument("--mode", choices=["web", "rtsp", "image", "both"], default="web",
                   help="web = web UI + all cameras (default); rtsp / image / both = CLI modes")
    p.add_argument("--config", default="config.yaml", help="path to config.yaml")
    p.add_argument("--path", help="image file or folder (image/both mode); default: image.input_path")
    p.add_argument("--show", action="store_true", help="show OpenCV windows with boxes and FPS")
    p.add_argument("--view", action="store_true", help="also start the web UI in rtsp/image/both mode")
    p.add_argument("--host", help="override web.host (0.0.0.0 = reachable from the LAN)")
    p.add_argument("--port", type=int, help="override web.port")
    p.add_argument("--cameras", nargs="+", metavar="ID", help="only run these camera ids")
    p.add_argument("--conf", type=float, help="override confidence threshold (not saved)")
    p.add_argument("--frame-skip", type=int, help="override default frame skip (not saved)")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], help="override detector.device")
    p.add_argument("--recognize", action="store_true", help="force face recognition on (not saved)")
    p.add_argument("--no-save", action="store_true", help="don't save crops / annotated images")
    p.add_argument("--log-level", help="override logging.level")
    return p.parse_args(argv)


def cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """CLI flags win over database settings for this run only."""
    o: dict[str, Any] = {}
    if args.conf is not None:
        o["detector.conf_threshold"] = args.conf
    if args.frame_skip is not None:
        o["rtsp.frame_skip"] = args.frame_skip
    if args.recognize:
        o["recognition.enabled"] = True
    return o


def install_signal_handlers(stop_event: threading.Event) -> None:
    def handler(signum: int, _frame: Any) -> None:
        if stop_event.is_set():
            logger.warning("Second signal received, forcing exit")
            sys.exit(130)
        logger.info("Received %s, shutting down gracefully (press Ctrl+C again to force)",
                    signal.Signals(signum).name)
        stop_event.set()

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cfg = load_config(args.config)
    except Exception as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2
    if args.device:
        cfg["detector"]["device"] = args.device
    if args.log_level:
        cfg["logging"]["level"] = args.log_level
    if args.host:
        cfg["web"]["host"] = args.host
    if args.port:
        cfg["web"]["port"] = args.port

    lc = cfg["logging"]
    setup_logging(lc["level"], lc["file"], int(lc["max_bytes"]), int(lc["backup_count"]))
    logger.info("Starting face detection: mode=%s config=%s", args.mode, args.config)
    configure_ffmpeg_transport(cfg["rtsp"].get("transport", "tcp"))

    show = args.show
    if show and not has_display():
        logger.warning("--show requested but no DISPLAY/WAYLAND_DISPLAY found; running headless")
        show = False
    use_web = args.mode == "web" or args.view

    stop_event = threading.Event()
    install_signal_handlers(stop_event)

    # ---- storage + models -------------------------------------------------
    db = Database(cfg["database"]["path"])
    seed_from_config(db, cfg)
    try:
        detector = FaceDetector.from_config(cfg)
    except Exception:
        logger.exception("Failed to initialise face detector")
        db.close()
        return 1
    persons_dir = cfg["database"]["persons_dir"]
    if db.get_meta("seeded_known_faces") is None:
        import_known_faces_dir(db, detector, cfg["recognition"]["known_faces_dir"], persons_dir)
        db.set_meta("seeded_known_faces", "1")
    detector.set_gallery(db.gallery())
    try:
        rodent = RodentDetector.from_config(cfg)
    except Exception:
        logger.exception("Failed to load the rodent model; rodent detection disabled")
        rodent = None

    output = OutputManager(cfg["output"], cfg["api"], save_enabled=not args.no_save,
                           on_event=lambda recs, dets: db.insert_detections(recs, [d.embedding for d in dets]))
    overrides = cli_overrides(args)
    apply_runtime_settings({**db.get_settings(), **overrides}, detector, output)

    w = cfg["web"]
    hub = LiveHub(float(w["max_fps"]), int(w["jpeg_quality"]), int(w["max_width"])) if use_web else None
    engine: DetectionEngine | None = None
    if args.mode in ("web", "rtsp", "both"):
        engine = DetectionEngine(cfg, db, detector, output, hub, show=show, keep_alive=use_web,
                                 camera_filter=args.cameras, overrides=overrides, rodent=rodent)

    # ---- web server ---------------------------------------------------------
    server = None
    if use_web:
        from web_app import AppContext, WebServer, create_app  # FastAPI only needed for the web UI

        ctx = AppContext(cfg=cfg, db=db, detector=detector, output=output, hub=hub, mode=args.mode, engine=engine,
                         rodent=rodent)  # type: ignore[arg-type]
        try:
            server = WebServer(create_app(ctx), str(w["host"]), int(w["port"])).start()
        except RuntimeError as exc:
            logger.error("%s", exc)
            output.close()
            db.close()
            return 1
        has_auth = bool(w.get("username") and w.get("password") and "${" not in str(w["password"]))
        if w["host"] not in ("127.0.0.1", "localhost") and not has_auth:
            logger.warning("Web UI is reachable from the network WITHOUT a password. "
                           "Set web.username / web.password in config.yaml.")

    retention = RetentionWorker(db, stop_event)
    retention.start()

    # ---- run ---------------------------------------------------------------
    image_path = args.path or cfg["image"]["input_path"]
    exit_code = 0
    try:
        processor = ImageProcessor(detector, output, extensions=cfg["image"]["extensions"],
                                   recursive=bool(cfg["image"]["recursive"]),
                                   show=show and args.mode == "image", stop_event=stop_event, hub=hub)
        if args.mode == "image":
            processor.run(image_path)
        elif args.mode == "both":
            # cv2.imshow must stay on the main thread, so the engine owns the windows and
            # images are processed headless in a worker thread (they still get saved/logged).
            img_thread = threading.Thread(target=processor.run, args=(image_path,), name="images", daemon=True)
            img_thread.start()
            engine.run(stop_event)  # type: ignore[union-attr]
            while img_thread.is_alive() and not stop_event.is_set():
                img_thread.join(0.5)
        else:  # web / rtsp
            engine.run(stop_event)  # type: ignore[union-attr]

        if server is not None and not stop_event.is_set():
            logger.info("Processing finished. Web UI still running at %s - press Ctrl+C to exit", server.url)
            while not stop_event.wait(0.5):  # short waits keep Ctrl+C responsive
                pass
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        exit_code = 2
    except Exception:
        logger.exception("Fatal error")
        exit_code = 1
    finally:
        stop_event.set()
        if hub is not None:
            hub.stop()
        if server is not None:
            server.stop()
        output.close()
        db.close()
        logger.info("Shutdown complete")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
