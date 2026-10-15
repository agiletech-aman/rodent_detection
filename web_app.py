"""Web UI + REST API (FastAPI). Everything is managed from here:

cameras (add / edit / test / enable / delete), persons + face photos, detection history,
image upload test, runtime settings and the live view.
"""

from __future__ import annotations

import asyncio
import base64
import csv
import hmac
import io
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import cv2
import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from database import SETTINGS_SCHEMA, Database, validate_settings
from detector import FaceDetector
from engine import DetectionEngine, LiveHub, apply_runtime_settings
from quality import QualityConfig, assess
from enrollment import (EnrollmentError, decode_image, delete_person_dir, enroll_embedding, enroll_image,
                        remove_files)
from rodent import RodentDetector, draw_rodents
from rtsp_stream import open_capture
from utils import (OutputManager, draw_detections, draw_info, list_video_devices, mask_url, parse_device,
                   resize_for_display, unmask_url)

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
ALLOWED_SCHEMES = ("rtsp", "rtsps", "rtmp", "http", "https", "file", "device")
CAMERA_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"


@dataclass
class AppContext:
    cfg: dict[str, Any]
    db: Database
    detector: FaceDetector
    output: OutputManager
    hub: LiveHub
    mode: str
    engine: DetectionEngine | None = None
    rodent: RodentDetector | None = None
    started: float = field(default_factory=time.time)

    @property
    def output_dir(self) -> Path:
        return Path(self.cfg["output"]["base_dir"]).resolve()

    @property
    def persons_dir(self) -> Path:
        return Path(self.cfg["database"]["persons_dir"]).resolve()

    @property
    def rodent_frames_dir(self) -> Path:
        """Camera frames labelled in the web UI for training the rodent model (YOLO layout)."""
        return Path((self.cfg.get("rodent") or {}).get("camera_dataset") or "datasets/rodent_camera").resolve()

    def reload_gallery(self) -> None:
        self.detector.set_gallery(self.db.gallery())

    def settings_changed(self) -> None:
        if self.engine is not None:
            self.engine.request_reload()
        else:  # no engine (image mode): apply directly
            apply_runtime_settings(self.db.get_settings(), self.detector, self.output)

    def media_url(self, path: str | None) -> str | None:
        if not path:
            return None
        p = Path(path).resolve()
        for root, base in (("output", self.output_dir), ("persons", self.persons_dir),
                           ("rodent-frames", self.rodent_frames_dir)):
            try:
                return f"/media/{root}/" + p.relative_to(base).as_posix()
            except ValueError:
                continue
        return None


# ======================================================================
# Request models
# ======================================================================

class CameraIn(BaseModel):
    id: str = Field(pattern=CAMERA_ID_PATTERN)
    name: str = Field(min_length=1, max_length=100)
    url: str = Field(max_length=2000)
    enabled: bool = True
    frame_skip: int | None = Field(default=None, ge=1, le=100)
    min_face_size: int | None = Field(default=None, ge=0, le=1000)
    min_blur_score: float | None = Field(default=None, ge=0, le=5000)


class CameraUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    url: str | None = Field(default=None, max_length=2000)
    enabled: bool | None = None
    frame_skip: int | None = Field(default=None, ge=1, le=100)
    clear_frame_skip: bool = False
    min_face_size: int | None = Field(default=None, ge=0, le=1000)
    min_blur_score: float | None = Field(default=None, ge=0, le=5000)
    clear_quality: list[str] = Field(default_factory=list)  # fields reset to the global setting


class CameraTest(BaseModel):
    url: str = Field(max_length=2000)
    camera_id: str | None = None  # lets a masked URL ("***") reuse the stored password


class DetectionState(BaseModel):
    paused: bool


class DetectionIds(BaseModel):
    ids: list[int] = Field(min_length=1, max_length=1000)


class FrameCapture(BaseModel):
    camera_id: str = Field(pattern=CAMERA_ID_PATTERN)


class FrameLabels(BaseModel):
    # [[x1, y1, x2, y2], ...] normalised to 0..1; an empty list marks the frame as "no rodent" (negative)
    boxes: list[list[float]] = Field(max_length=50)


class PersonIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    code: str | None = Field(default=None, max_length=64)
    notes: str = Field(default="", max_length=2000)


class PersonUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    code: str | None = Field(default=None, max_length=64)
    notes: str | None = Field(default=None, max_length=2000)


# ======================================================================
# Helpers
# ======================================================================

def _check_url(url: str) -> str:
    url = url.strip()
    if not url:
        raise HTTPException(422, "Enter the camera URL (e.g. rtsp://user:pass@192.168.1.10:554/stream) or pick a device camera")
    dev = parse_device(url)
    if dev is not None:
        return f"device:{dev}"  # normalise /dev/video0 -> device:0
    if url.startswith("device:"):
        raise HTTPException(422, "Device camera must look like device:0")
    scheme = urlsplit(url).scheme.lower()
    if scheme not in ALLOWED_SCHEMES:
        raise HTTPException(422, f"URL must start with one of: {', '.join(s + '://' for s in ALLOWED_SCHEMES)}")
    return url


def _jpeg_data_url(img: Any, max_width: int = 1280, quality: int = 80) -> str:
    ok, buf = cv2.imencode(".jpg", resize_for_display(img, max_width), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""


def _records_filter(records: str) -> bool | None:
    """'new' = with embedding (default), 'old' = saved before migration 001, 'all'."""
    if records not in ("new", "old", "all"):
        raise HTTPException(422, "records must be new, old or all")
    return {"new": True, "old": False, "all": None}[records]


def _parse_date(value: str | None, end: bool = False) -> float | None:
    """'2026-10-13' or '2026-10-13T14:30' (local time) -> epoch. end=True makes a date inclusive."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        raise HTTPException(422, f"Invalid date: {value}") from None
    ts = dt.timestamp()
    if end and len(value) <= 10:
        ts += 86400
    return ts


def probe_stream(url: str, timeout_ms: int = 8000) -> dict[str, Any]:
    """Open a stream once and grab a frame (camera 'Test' button)."""
    t0 = time.perf_counter()
    cap = open_capture(url, timeout_ms, timeout_ms)
    try:
        if not cap.isOpened():
            if parse_device(url) is not None:
                return {"ok": False, "error": "Cannot open the device camera (unplugged, or used by another app?)"}
            return {"ok": False, "error": "Cannot open stream (check IP, port, path, user/password)"}
        frame = None
        for _ in range(25):  # first frames of some cameras are empty until a key frame arrives
            ok, frame = cap.read()
            if ok and frame is not None:
                break
        if frame is None:
            return {"ok": False, "error": "Connected but no frame received"}
        return {
            "ok": True,
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
            "fps": round(float(cap.get(cv2.CAP_PROP_FPS) or 0), 1),
            "latency_ms": round((time.perf_counter() - t0) * 1000),
            "snapshot": _jpeg_data_url(frame, 960),
        }
    finally:
        cap.release()


# ======================================================================
# App
# ======================================================================

def create_app(ctx: AppContext) -> FastAPI:
    web = ctx.cfg["web"]
    auth = (str(web.get("username") or ""), str(web.get("password") or ""))
    auth_enabled = bool(auth[0] and auth[1] and "${" not in auth[1])
    max_upload = int(web.get("max_upload_mb", 15)) * 1024 * 1024
    security = HTTPBasic(auto_error=False)

    def require_auth(credentials: HTTPBasicCredentials | None = Depends(security)) -> None:
        if not auth_enabled:
            return
        if credentials is None or not (
            hmac.compare_digest(credentials.username.encode(), auth[0].encode())
            and hmac.compare_digest(credentials.password.encode(), auth[1].encode())
        ):
            raise HTTPException(401, "Authentication required", headers={"WWW-Authenticate": 'Basic realm="Face Detection"'})

    app = FastAPI(title="Face Detection", version="2.0", dependencies=[Depends(require_auth)],
                  docs_url="/api/docs", redoc_url=None, openapi_url="/api/openapi.json")

    @app.middleware("http")
    async def csrf_guard(request: Request, call_next: Any) -> Any:
        # Browsers resend Basic-Auth credentials automatically, so state-changing requests must
        # carry a header that a cross-site form cannot set.
        if request.method in ("POST", "PUT", "PATCH", "DELETE") and request.headers.get("x-requested-with") != "fetch":
            return JSONResponse({"detail": "Missing X-Requested-With: fetch header"}, status_code=403)
        return await call_next(request)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> Response:
        # app.js / style.css get a ?v=<mtime> suffix so browsers fetch them again after an update
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        for name in ("app.js", "style.css"):
            html = html.replace(f"/static/{name}\"", f"/static/{name}?v={int((STATIC_DIR / name).stat().st_mtime)}\"")
        return Response(html, media_type="text/html", headers={"Cache-Control": "no-store"})

    # ------------------------------------------------------------- system

    @app.get("/api/system")
    def system() -> dict[str, Any]:
        midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        d = ctx.cfg["detector"]
        return {
            "mode": ctx.mode,
            "device": ctx.detector.device,
            "model": d["model_name"],
            "det_size": list(d["det_size"]),
            "uptime_sec": round(time.time() - ctx.started),
            "engine_running": ctx.engine is not None,
            "detection_paused": ctx.engine is not None and ctx.engine.paused,
            "cameras": len(ctx.db.list_cameras()),
            "persons": len(ctx.db.list_persons()),
            "today": ctx.db.stats(midnight),
            "rodent_model": ctx.rodent is not None,
            "rodent_enabled": ctx.rodent is not None and bool(ctx.db.get_settings().get("rodent.enabled", True)),
            "rodents_today": ctx.db.rodent_stats(midnight),
        }

    @app.put("/api/detection")
    def set_detection(body: DetectionState) -> dict[str, Any]:
        """Pause / resume face detection on all cameras (live video keeps running). Not persisted."""
        if ctx.engine is None:
            raise HTTPException(409, "Detection engine is not running")
        ctx.engine.paused = body.paused
        logger.info("Detection %s from web", "paused" if body.paused else "resumed")
        return {"paused": ctx.engine.paused}

    @app.get("/api/status")
    def status() -> list[dict[str, Any]]:
        names = {c["id"]: c["name"] for c in ctx.db.list_cameras()}
        return [{**s, "name": names.get(s["id"], s["id"])} for s in ctx.hub.status()]

    # ------------------------------------------------------------- live

    @app.get("/api/stream/{camera_id}")
    async def stream(camera_id: str, request: Request) -> StreamingResponse:
        async def frames() -> Any:
            last_seq, last_sent = -1, 0.0
            while not ctx.hub.stopping:
                if await request.is_disconnected():
                    break
                seq, jpeg = await run_in_threadpool(ctx.hub.get_jpeg, camera_id)
                now = time.monotonic()
                if jpeg is None and now - last_sent > 1.0:
                    jpeg, seq = await run_in_threadpool(ctx.hub.placeholder_jpeg, camera_id), -int(now)
                if jpeg is not None and seq != last_seq:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                           + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                    last_seq, last_sent = seq, now
                await asyncio.sleep(ctx.hub.frame_interval)

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame",
                                 headers={"Cache-Control": "no-store"})

    @app.get("/api/snapshot/{camera_id}")
    def snapshot(camera_id: str) -> Response:
        _, jpeg = ctx.hub.get_jpeg(camera_id)
        return Response(jpeg or ctx.hub.placeholder_jpeg(camera_id), media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    # ------------------------------------------------------------- cameras

    def camera_out(row: dict[str, Any]) -> dict[str, Any]:
        live = ctx.hub.status_of(row["id"])
        return {
            "id": row["id"], "name": row["name"], "url": mask_url(row["url"]),
            "enabled": row["enabled"], "frame_skip": row["frame_skip"],
            "min_face_size": row.get("min_face_size"), "min_blur_score": row.get("min_blur_score"),
            "state": (live.get("state") or "starting") if row["enabled"] else "disabled",
            "fps": live.get("fps"), "faces": live.get("faces"),
        }

    def device_in_use(url: str, exclude_id: str | None = None) -> dict[str, Any] | None:
        dev = parse_device(url)
        if dev is None:
            return None
        return next((c for c in ctx.db.list_cameras()
                     if c["id"] != exclude_id and parse_device(c["url"]) == dev), None)

    @app.get("/api/devices")
    def list_devices() -> list[dict[str, Any]]:
        """Local cameras (laptop webcam / USB) on the machine running this server."""
        out = []
        for d in list_video_devices():
            used = device_in_use(d["url"])
            out.append({**d, "used_by": used["name"] if used else None, "used_by_id": used["id"] if used else None})
        return out

    @app.get("/api/cameras")
    def list_cameras() -> list[dict[str, Any]]:
        return [camera_out(c) for c in ctx.db.list_cameras()]

    @app.post("/api/cameras", status_code=201)
    def create_camera(body: CameraIn) -> dict[str, Any]:
        url = _check_url(body.url)
        if (other := device_in_use(url)) is not None:
            raise HTTPException(409, f"This device camera is already added as '{other['name']}'")
        try:
            row = ctx.db.create_camera(body.id, body.name.strip(), url, body.enabled, body.frame_skip)
            if body.min_face_size is not None or body.min_blur_score is not None:
                row = ctx.db.update_camera(body.id, min_face_size=body.min_face_size, min_blur_score=body.min_blur_score)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None
        logger.info("Camera added from web: %s (%s)", body.id, mask_url(body.url))
        ctx.settings_changed()
        return camera_out(row)

    @app.put("/api/cameras/{camera_id}")
    def update_camera(camera_id: str, body: CameraUpdate) -> dict[str, Any]:
        row = ctx.db.get_camera(camera_id)
        if row is None:
            raise HTTPException(404, "Camera not found")
        fields = body.model_dump(exclude_none=True, exclude={"clear_frame_skip", "clear_quality"})
        for f in body.clear_quality:
            if f in ("min_face_size", "min_blur_score"):
                fields[f] = None
        if "url" in fields:
            fields["url"] = unmask_url(_check_url(fields["url"]), row["url"])
            if (other := device_in_use(fields["url"], exclude_id=camera_id)) is not None:
                raise HTTPException(409, f"This device camera is already added as '{other['name']}'")
        if body.clear_frame_skip:
            fields["frame_skip"] = None
        row = ctx.db.update_camera(camera_id, **fields)
        logger.info("Camera updated from web: %s", camera_id)
        ctx.settings_changed()
        return camera_out(row)  # type: ignore[arg-type]

    @app.delete("/api/cameras/{camera_id}", status_code=204)
    def delete_camera(camera_id: str) -> Response:
        if not ctx.db.delete_camera(camera_id):
            raise HTTPException(404, "Camera not found")
        logger.info("Camera deleted from web: %s", camera_id)
        ctx.settings_changed()
        return Response(status_code=204)

    @app.post("/api/cameras/test")
    def test_camera(body: CameraTest) -> dict[str, Any]:
        url = _check_url(body.url)
        if body.camera_id:
            row = ctx.db.get_camera(body.camera_id)
            if row:
                url = unmask_url(url, row["url"])
        # A webcam can only be opened by one reader: if the engine already streams it, show that stream.
        used = device_in_use(url)
        if used is not None and used["enabled"]:
            _, jpeg = ctx.hub.get_jpeg(used["id"])
            if jpeg:
                img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                return {"ok": True, "width": int(img.shape[1]), "height": int(img.shape[0]), "fps": None,
                        "latency_ms": 0, "snapshot": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode(),
                        "note": f"Device is running as camera '{used['name']}'; showing its live frame"}
        return probe_stream(url, int(ctx.cfg["rtsp"].get("open_timeout_ms", 8000)))

    # ------------------------------------------------------------- persons

    def person_out(p: dict[str, Any]) -> dict[str, Any]:
        return {**{k: p[k] for k in ("id", "name", "code", "notes", "image_count", "last_seen",
                                     "last_camera", "created_at")},
                "thumb_url": ctx.media_url(p.get("thumb_path"))}

    def get_person_or_404(person_id: int) -> dict[str, Any]:
        p = ctx.db.get_person(person_id)
        if p is None:
            raise HTTPException(404, "Person not found")
        return p

    @app.get("/api/persons")
    def list_persons(q: str | None = None) -> list[dict[str, Any]]:
        return [person_out(p) for p in ctx.db.list_persons(q)]

    @app.post("/api/persons", status_code=201)
    def create_person(body: PersonIn) -> dict[str, Any]:
        try:
            p = ctx.db.create_person(body.name.strip(), (body.code or "").strip() or None, body.notes)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None
        logger.info("Person added from web: %s (#%d)", p["name"], p["id"])
        return person_out(p)

    @app.get("/api/persons/{person_id}")
    def get_person(person_id: int) -> dict[str, Any]:
        p = person_out(get_person_or_404(person_id))
        p["images"] = [{"id": i["id"], "url": ctx.media_url(i["path"]), "source": i["source"],
                        "created_at": i["created_at"]}
                       for i in ctx.db.list_person_images(person_id)]
        return p

    @app.put("/api/persons/{person_id}")
    def update_person(person_id: int, body: PersonUpdate) -> dict[str, Any]:
        get_person_or_404(person_id)
        fields = body.model_dump(exclude_none=True)
        if "name" in fields:
            fields["name"] = fields["name"].strip()
        if "code" in fields:
            fields["code"] = fields["code"].strip() or None
        try:
            p = ctx.db.update_person(person_id, **fields)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None
        if "name" in fields:
            ctx.reload_gallery()
        return person_out(p)  # type: ignore[arg-type]

    @app.delete("/api/persons/{person_id}", status_code=204)
    def delete_person(person_id: int) -> Response:
        p = get_person_or_404(person_id)
        remove_files(ctx.db.delete_person(person_id))
        delete_person_dir(ctx.persons_dir, person_id)
        ctx.reload_gallery()
        logger.info("Person deleted from web: %s (#%d)", p["name"], person_id)
        return Response(status_code=204)

    @app.post("/api/persons/{person_id}/images")
    def add_person_images(person_id: int, files: list[UploadFile] = File(...)) -> dict[str, Any]:
        get_person_or_404(person_id)
        results = []
        for f in files:
            data = f.file.read(max_upload + 1)
            item: dict[str, Any] = {"file": f.filename}
            try:
                if len(data) > max_upload:
                    raise EnrollmentError(f"File larger than {max_upload // (1024 * 1024)} MB")
                image_id, warning = enroll_image(ctx.db, ctx.detector, ctx.persons_dir, person_id, decode_image(data))
                item.update(ok=True, image_id=image_id, warning=warning)
            except EnrollmentError as exc:
                item.update(ok=False, error=str(exc))
            results.append(item)
        if any(r["ok"] for r in results):
            ctx.reload_gallery()
        return {"results": results, "person": get_person(person_id)}

    @app.post("/api/faces/check")
    def check_face(file: UploadFile = File(...)) -> dict[str, Any]:
        """Quick quality check for a photo before enrolling (nothing is saved)."""
        data = file.file.read(max_upload + 1)
        if len(data) > max_upload:
            raise HTTPException(413, "File too large")
        try:
            img = decode_image(data)
        except EnrollmentError as exc:
            raise HTTPException(422, str(exc)) from None
        dets = ctx.detector.detect(img, recognize=False)
        if not dets:
            return {"ok": False, "faces": 0, "message": "No face found"}
        largest = max(dets, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))
        width = largest.bbox[2] - largest.bbox[0]
        qcfg = QualityConfig.from_settings(ctx.db.get_settings())
        assess(img, largest, qcfg)
        q = largest.quality or {}
        if width < 60:
            ok, msg = False, "Face too small: move closer"
        elif q.get("blur", 0) < qcfg.min_blur_score:
            ok, msg = False, "Blurry: hold still / more light"
        elif abs(q.get("yaw", 0)) > qcfg.max_yaw or abs(q.get("pitch", 0)) > qcfg.max_pitch:
            ok, msg = False, "Face turned too far: look more towards the camera"
        elif len(dets) > 1:
            ok, msg = True, f"{len(dets)} faces: the largest is used"
        else:
            ok, msg = True, "Good"
        return {"ok": ok, "faces": len(dets), "face_width": int(width),
                "confidence": round(largest.confidence, 2), "quality": q, "message": msg}

    @app.delete("/api/persons/{person_id}/images/{image_id}", status_code=204)
    def delete_person_image(person_id: int, image_id: int) -> Response:
        path = ctx.db.delete_person_image(person_id, image_id)
        if path is None:
            raise HTTPException(404, "Image not found")
        remove_files([path])
        ctx.reload_gallery()
        return Response(status_code=204)

    @app.post("/api/persons/{person_id}/from-detection/{det_id}")
    def enroll_from_detection(person_id: int, det_id: int) -> dict[str, Any]:
        """Teach the system who this face is, using the embedding saved with the detection
        (the crop is NOT re-detected, so tight / small crops work too). Also labels that detection."""
        p = get_person_or_404(person_id)
        det = ctx.db.get_detection(det_id)
        if det is None:
            raise HTTPException(404, "Detection not found")
        if det.get("embedding") is None:
            raise HTTPException(422, "This is an old record saved without an embedding. "
                                     "Pick a newer detection (run scripts/cleanup_no_embedding.py to remove old ones).")
        image_id = enroll_embedding(ctx.db, ctx.persons_dir, person_id, det["embedding"], det.get("crop_path"))
        ctx.db.label_detection(det_id, person_id, p["name"])
        ctx.reload_gallery()
        return {"ok": True, "image_id": image_id, "warning": None}

    @app.post("/api/persons/{person_id}/from-detections")
    def enroll_from_detections(person_id: int, body: DetectionIds) -> dict[str, Any]:
        """Bulk version of from-detection: the selected faces are all the same person.
        Old records without an embedding are skipped (and reported)."""
        p = get_person_or_404(person_id)
        added, skipped = 0, []
        for det_id in dict.fromkeys(body.ids):
            det = ctx.db.get_detection(det_id)
            if det is None or det.get("embedding") is None:
                skipped.append(det_id)
                continue
            enroll_embedding(ctx.db, ctx.persons_dir, person_id, det["embedding"], det.get("crop_path"))
            ctx.db.label_detection(det_id, person_id, p["name"])
            added += 1
        if added:
            ctx.reload_gallery()
        logger.info("Assigned %d detection(s) to person #%d from web (%d skipped)", added, person_id, len(skipped))
        return {"added": added, "skipped": skipped}

    # ------------------------------------------------------------- detections

    def det_out(r: dict[str, Any]) -> dict[str, Any]:
        r = dict(r)
        r["crop_url"] = ctx.media_url(r.pop("crop_path", None))
        r["annotated_url"] = ctx.media_url(r.pop("annotated_path", None))
        return r

    @app.get("/api/detections")
    def list_detections(
        camera_id: str | None = None,
        person_id: int | None = None,
        known: bool | None = None,
        date_from: str | None = Query(default=None, alias="from"),
        date_to: str | None = Query(default=None, alias="to"),
        limit: int = Query(default=50, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
        records: str = "new",
    ) -> dict[str, Any]:
        rows, total = ctx.db.query_detections(camera_id=camera_id, person_id=person_id, known=known,
                                              since=_parse_date(date_from), until=_parse_date(date_to, end=True),
                                              limit=limit, offset=offset, with_embedding=_records_filter(records))
        return {"total": total, "items": [det_out(r) for r in rows]}

    @app.post("/api/detections/delete")
    def delete_detections(body: DetectionIds) -> dict[str, Any]:
        """Delete selected detection records and their images (if no other record uses them)."""
        deleted, files = ctx.db.delete_detections(body.ids)
        remove_files(files)
        logger.info("Deleted %d detection record(s) and %d file(s) from web", deleted, len(files))
        return {"deleted": deleted, "files_removed": len(files)}

    @app.delete("/api/detections/{det_id}", status_code=204)
    def delete_detection(det_id: int) -> Response:
        deleted, files = ctx.db.delete_detections([det_id])
        if not deleted:
            raise HTTPException(404, "Detection not found")
        remove_files(files)
        return Response(status_code=204)

    @app.get("/api/detections.csv")
    def export_detections(
        camera_id: str | None = None,
        person_id: int | None = None,
        known: bool | None = None,
        date_from: str | None = Query(default=None, alias="from"),
        date_to: str | None = Query(default=None, alias="to"),
        records: str = "new",
    ) -> StreamingResponse:
        rows, _ = ctx.db.query_detections(camera_id=camera_id, person_id=person_id, known=known,
                                          since=_parse_date(date_from), until=_parse_date(date_to, end=True),
                                          limit=100000, offset=0, with_embedding=_records_filter(records))
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["id", "timestamp", "camera_id", "person_id", "name", "similarity", "confidence",
                    "x1", "y1", "x2", "y2", "crop_path", "source"])
        for r in rows:
            w.writerow([r["id"], r["ts"], r["camera_id"], r["person_id"], r["name"], r["similarity"],
                        r["confidence"], *r["bbox"], r["crop_path"], r["source"]])
        name = f"detections_{datetime.now():%Y%m%d_%H%M%S}.csv"
        return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                                 headers={"Content-Disposition": f'attachment; filename="{name}"'})

    # ------------------------------------------------------------- rodents

    @app.get("/api/rodents")
    def list_rodents(
        camera_id: str | None = None,
        date_from: str | None = Query(default=None, alias="from"),
        date_to: str | None = Query(default=None, alias="to"),
        limit: int = Query(default=50, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        rows, total = ctx.db.query_rodent_events(camera_id=camera_id, since=_parse_date(date_from),
                                                 until=_parse_date(date_to, end=True), limit=limit, offset=offset)
        return {"total": total, "items": [det_out(r) for r in rows]}

    @app.post("/api/rodents/delete")
    def delete_rodents(body: DetectionIds) -> dict[str, Any]:
        deleted, files = ctx.db.delete_rodent_events(body.ids)
        remove_files(files)
        logger.info("Deleted %d rodent alert(s) from web", deleted)
        return {"deleted": deleted, "files_removed": len(files)}

    # ------------------------------------------------------------- rodent training frames

    frame_name = re.compile(r"^[A-Za-z0-9_-]{1,100}\.jpg$")

    def frame_paths(name: str) -> tuple[Path, Path]:
        if not frame_name.match(name):
            raise HTTPException(404, "Frame not found")
        base = ctx.rodent_frames_dir
        return base / "images" / name, base / "labels" / (name[:-4] + ".txt")

    def frame_out(img: Path) -> dict[str, Any]:
        lbl = ctx.rodent_frames_dir / "labels" / (img.stem + ".txt")
        boxes = None
        if lbl.is_file():
            boxes = []
            for line in lbl.read_text().split("\n"):
                parts = line.split()
                if len(parts) == 5:
                    _, cx, cy, bw, bh = map(float, parts)
                    boxes.append([round(v, 5) for v in (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)])
        return {"name": img.name, "camera_id": img.stem.rsplit("_", 3)[0], "url": ctx.media_url(str(img)),
                "mtime": img.stat().st_mtime, "boxes": boxes,
                "status": "pending" if boxes is None else ("rodent" if boxes else "empty")}

    @app.get("/api/rodent-frames")
    def list_rodent_frames() -> dict[str, Any]:
        imgs = sorted((ctx.rodent_frames_dir / "images").glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
        items = [frame_out(p) for p in imgs]
        counts = {k: sum(1 for i in items if i["status"] == k) for k in ("pending", "rodent", "empty")}
        return {"items": items, "counts": counts}

    @app.post("/api/rodent-frames", status_code=201)
    def capture_rodent_frame(body: FrameCapture) -> dict[str, Any]:
        """Save the camera's current frame (without overlays) for labelling."""
        frame = ctx.engine.raw_frames.get(body.camera_id) if ctx.engine is not None else None
        if frame is None:
            raise HTTPException(409, "No frame from this camera yet")
        name = f"{body.camera_id}_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
        img, _ = frame_paths(name)
        img.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(img), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise HTTPException(500, "Could not save the frame")
        return frame_out(img)

    @app.put("/api/rodent-frames/{name}")
    def label_rodent_frame(name: str, body: FrameLabels) -> dict[str, Any]:
        img, lbl = frame_paths(name)
        if not img.is_file():
            raise HTTPException(404, "Frame not found")
        lines = []
        for b in body.boxes:
            if len(b) != 4:
                raise HTTPException(422, "Each box needs [x1, y1, x2, y2]")
            x1, x2 = sorted(min(max(v, 0.0), 1.0) for v in (b[0], b[2]))
            y1, y2 = sorted(min(max(v, 0.0), 1.0) for v in (b[1], b[3]))
            if x2 - x1 < 0.002 or y2 - y1 < 0.002:
                continue
            lines.append(f"0 {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
        lbl.parent.mkdir(parents=True, exist_ok=True)
        lbl.write_text("\n".join(lines) + ("\n" if lines else ""))
        return frame_out(img)

    @app.delete("/api/rodent-frames/{name}", status_code=204)
    def delete_rodent_frame(name: str) -> Response:
        img, lbl = frame_paths(name)
        if not img.is_file():
            raise HTTPException(404, "Frame not found")
        img.unlink()
        lbl.unlink(missing_ok=True)
        return Response(status_code=204)

    # ------------------------------------------------------------- image test

    @app.post("/api/images/detect")
    def detect_images(files: list[UploadFile] = File(...)) -> dict[str, Any]:
        results = []
        for f in files:
            data = f.file.read(max_upload + 1)
            if len(data) > max_upload:
                results.append({"file": f.filename, "ok": False, "error": "File too large"})
                continue
            try:
                img = decode_image(data)
            except EnrollmentError as exc:
                results.append({"file": f.filename, "ok": False, "error": str(exc)})
                continue
            t0 = time.perf_counter()
            dets = ctx.detector.detect(img)
            ms = (time.perf_counter() - t0) * 1000
            qcfg = QualityConfig.from_settings(ctx.db.get_settings())
            for d in dets:  # shown for tuning the thresholds; uploads are not filtered
                assess(img, d, qcfg)
                if d.embedding is None:  # stored with the record so it can be assigned to a person
                    d.embedding = ctx.detector.embed_detection(img, d)
            rodents = []
            if ctx.rodent is not None:  # always run on uploads (the setting only switches the cameras)
                t1 = time.perf_counter()
                rodents = ctx.rodent.detect(img, conf_threshold=ctx.db.get_settings().get("rodent.conf_threshold"))
                rodent_ms = (time.perf_counter() - t1) * 1000
            annotated = draw_detections(img.copy(), dets)
            draw_rodents(annotated, rodents)
            draw_info(annotated, [f"faces: {len(dets)} | {ms:.0f} ms | {ctx.detector.device}",
                                  *([f"rodents: {len(rodents)} | {rodent_ms:.0f} ms"] if ctx.rodent is not None else [])])
            ctx.output.handle("upload", img, dets, source=f.filename or "upload.jpg",
                              annotated=annotated, rate_limit=False)
            results.append({
                "file": f.filename, "ok": True, "infer_ms": round(ms),
                "width": int(img.shape[1]), "height": int(img.shape[0]),
                "faces": [{**d.to_dict(), "reject": d.reject} for d in dets],
                "rodents": [r.to_dict() for r in rodents] if ctx.rodent is not None else None,
                "annotated": _jpeg_data_url(annotated),
            })
        return {"results": results}

    # ------------------------------------------------------------- settings

    @app.get("/api/settings")
    def get_settings() -> dict[str, Any]:
        values = ctx.db.get_settings()
        for spec in SETTINGS_SCHEMA:  # never send secrets back to the browser
            if spec["type"] == "secret":
                values[spec["key"]] = "" if not values.get(spec["key"]) else "__set__"
        return {"schema": SETTINGS_SCHEMA, "values": values}

    @app.put("/api/settings")
    def put_settings(body: dict[str, Any]) -> dict[str, Any]:
        body = {k: v for k, v in body.items() if not (k.startswith("api.auth_token") and v in ("", "__set__"))}
        try:
            clean = validate_settings(body)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        ctx.db.set_settings(clean)
        logger.info("Settings updated from web: %s", ", ".join(sorted(clean)))
        ctx.settings_changed()
        return get_settings()

    # ------------------------------------------------------------- media

    @app.get("/media/{root}/{path:path}", include_in_schema=False)
    def media(root: str, path: str) -> FileResponse:
        base = {"output": ctx.output_dir, "persons": ctx.persons_dir, "rodent-frames": ctx.rodent_frames_dir}.get(root)
        if base is None:
            raise HTTPException(404)
        p = (base / path).resolve()
        try:
            p.relative_to(base)  # blocks ../ traversal
        except ValueError:
            raise HTTPException(404) from None
        if not p.is_file():
            raise HTTPException(404)
        return FileResponse(p, headers={"Cache-Control": "private, max-age=86400"})

    return app


class WebServer:
    """Runs uvicorn in a background thread so the main thread stays free for the engine / cv2.imshow."""

    def __init__(self, app: FastAPI, host: str, port: int) -> None:
        self.host, self.port = host, port
        config = uvicorn.Config(app, host=host, port=port, log_config=None, access_log=False,
                                timeout_graceful_shutdown=3, lifespan="off")
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, name="web", daemon=True)

    @property
    def url(self) -> str:
        host = "localhost" if self.host in ("0.0.0.0", "") else self.host
        return f"http://{host}:{self.port}/"

    def start(self, timeout: float = 15.0) -> "WebServer":
        self.thread.start()
        deadline = time.monotonic() + timeout
        while not self.server.started:
            if not self.thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError(f"Web server failed to start on {self.host}:{self.port} (port in use?)")
            time.sleep(0.05)
        logger.info("Web UI running at %s (listening on %s:%d)", self.url, self.host, self.port)
        return self

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(6)
        logger.info("Web server stopped")
