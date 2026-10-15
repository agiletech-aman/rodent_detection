# Face Detection System (RTSP + Images)

Real-time face detection on RTSP IP cameras and static images, built with
**InsightFace SCRFD** (detection), **ArcFace** (optional recognition), OpenCV and ONNX Runtime.
The whole system is managed from a **web UI**: cameras, people, detection history and settings.

## Features

- **Web UI** (`python main.py`): run everything from the browser
  - **Live**: all camera streams with boxes, landmarks and FPS, plus today's stats and a recent-detections feed
  - **Cameras**: add, edit, enable/disable and delete RTSP cameras, with a *Test connection* button that shows a snapshot. Changes apply immediately, with no restart.
  - **Persons**: store people (name, code / employee ID, notes) with face photos. Recognition starts using new photos immediately.
  - **Detections**: searchable history (camera / person / known-unknown / date), face thumbnails, CSV export,
    and *Assign person* to teach the system an unknown face
  - **Image test**: upload photos and see detection/recognition results
  - **Settings**: thresholds, frame skip, recognition, storage, retention and the REST API, applied live
- All data lives in SQLite (`data/app.db`) and face photos are stored in `data/persons/`
- CLI modes still work: `--mode rtsp | image | both`
- One background thread per camera that keeps **only the latest frame**, so there's no buffer lag
- Auto-reconnect with exponential backoff and jitter; `max_retries` is configurable (0 = retry forever)
- Multiple cameras, frame skip, per-camera FPS / inference-time overlay
- Bounding box, confidence score and 5 landmarks for each face
- Face recognition (ArcFace) against the persons stored in the database
- Self-learning: confidently recognised faces are added to the person (marked *auto* in Persons), so the next detections of them match better
- Face crops, annotated frames, a JSON Lines detection log and an async REST API POST with retry
- Uses the GPU (CUDA) when available and falls back to CPU
- Detection history retention (auto-delete after N days)
- Rotating file + console logging, and graceful shutdown on Ctrl+C / SIGTERM

## Architecture

```
  Browser ── web_app.py (FastAPI) ──────────── SQLite (data/app.db)
     │  cameras / persons / settings CRUD        cameras, persons, face embeddings,
     │  live MJPEG, history, uploads             detections, settings
     │                │ request_reload()                 ▲
     │                ▼                                  │ insert events
     │        DetectionEngine (main thread) ───────── OutputManager ── crops / annotated / JSONL / REST API
     │          │ starts/stops streams on change         ▲
     │          ▼                                        │
     │   RTSPStream thread per camera ── latest frame ──► FaceDetector (SCRFD + ArcFace, ONNX Runtime CUDA/CPU)
     │                                                    ▲ gallery reloaded when persons change
     └──── LiveHub (latest annotated frame + status per camera)
```

| File | Responsibility |
|---|---|
| `main.py` | CLI, startup wiring, signal handling |
| `web_app.py` | FastAPI app: REST API, live MJPEG, uploads, media files, auth + CSRF guard |
| `static/` | Web UI (plain HTML/CSS/JS, no build step) |
| `database.py` | SQLite schema and queries, editable settings schema, first-run import from config.yaml |
| `dedup.py` | Per-camera face tracker (best crop + cooldown) and recognition identity cache |
| `quality.py` | Face quality: size, confidence, blur, head pose (yaw/pitch) |
| `migrations/`, `scripts/` | Database migrations; cleanup of old records without embeddings |
| `engine.py` | `DetectionEngine` (reacts to camera/settings changes), `LiveHub`, retention cleanup |
| `enrollment.py` | Adding face photos to persons, importing the old `known_faces/` folder |
| `rtsp_stream.py` | Threaded reader, latest-frame slot, reconnect with backoff |
| `detector.py` | `FaceDetector` (InsightFace) and the `Detection` dataclass, runtime settings, recognition gallery |
| `rodent.py` | `RodentDetector` (YOLO ONNX), `RodentWatcher` (confirmation + cooldown, stores alerts) |
| `scripts/rodent/` | Dataset auto-labelling, negative-frame capture and training of the rodent model |
| `image_processor.py` | Single image / folder processing (CLI) |
| `utils.py` | Config loading, logging, drawing, saving, JSONL writer, API POST with retry |

### Why InsightFace (SCRFD) and not MediaPipe?

| | InsightFace SCRFD | MediaPipe Face Detection |
|---|---|---|
| Accuracy (WIDER FACE hard) | ~83% (10G) / ~68% (500M) | Built for selfie/near range; weak on small and far faces |
| CCTV use (small faces, angles, crowds) | Strong | Misses many faces beyond ~2–5 m |
| Landmarks | 5 points (eyes, nose, mouth corners), as required | 6 keypoints (different set) |
| Recognition | ArcFace embeddings in the same package | None |
| GPU | ONNX Runtime CUDA | Python package is CPU-only on Linux |

CCTV/RTSP footage usually has small, off-angle faces, so SCRFD is the better fit.
`buffalo_l` (SCRFD-10G) runs comfortably in real time on a GPU. On a CPU-only machine,
switch to `buffalo_s` (SCRFD-500M) and/or lower `det_size` to `[480, 480]`.

## Setup (Ubuntu 20.04+/22.04/24.04, Python 3.10+)

```bash
# System packages: compiler for insightface, FFmpeg libs, GUI libs for imshow
sudo apt update
sudo apt install -y python3-venv python3-dev build-essential ffmpeg libgl1 libglib2.0-0

cd Face_detection
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt

# insightface pulls in opencv-python-headless (via albumentations), which breaks cv2.imshow.
# Keep only the GUI build:
pip uninstall -y opencv-python-headless opencv-python
pip install opencv-python
```

Models (~280 MB for `buffalo_l`) download automatically to `~/.insightface/models/` on the first run.

### GPU check

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# should include 'CUDAExecutionProvider'
```

At startup the log prints the providers that are **actually** in use, for example
`device=cuda providers=['CUDAExecutionProvider', 'CPUExecutionProvider']`.
If it reports `cpu` although you have a GPU, check that `nvidia-smi` works (NVIDIA driver ≥ 525 for CUDA 12).
On a CPU-only machine you can replace `onnxruntime-gpu[...]` with `onnxruntime` in `requirements.txt`.

## Quick start (web UI)

```bash
source .venv/bin/activate
python main.py                      # same as: python main.py --mode web
# open http://localhost:8080/
```

1. **Cameras → + Add camera**: choose the source, click *Test connection* to check it, then *Save*. The stream starts right away.
   - **🌐 IP / RTSP camera**: enter the RTSP URL.
   - **💻 Device camera**: pick a webcam or USB camera connected to the computer that runs the server, from the list.
     It is stored as `device:0` (`/dev/video0`). A device can be added only once, because only one reader can open it.
2. **Persons → + Add person**: enter the name / code, then add 3–5 face photos:
   - **📷 Use camera**: live preview from this device's webcam or phone camera. Use *Capture*, or *Auto 5 photos*,
     which prompts you to look straight / left / right / up / down. Each photo is checked right away
     (✓ face / *No face found* / *Face too small*), and photos that fail the check are not saved.
   - **🖼 Choose files**: upload existing photos.

   Browsers only allow the live camera on **https** or **localhost**. If you open the UI from a phone over
   `http://<lan-ip>:8080`, the button becomes *📷 Take photo*, which opens the phone's camera app.
   For a live preview on phones, put the server behind HTTPS (nginx/Caddy).
3. **Settings → Recognition**: turn on *Enable face recognition* and click *Save*.
4. **Live** now shows names on the boxes, and **Detections** shows the full history.
   For an *Unknown* face, click **Assign person**. That crop is added to the person, so the face is recognised next time.

To open the UI from other PCs on the LAN, set a password first, then start the server on all interfaces:

```yaml
# config.yaml
web:
  username: admin
  password: "${WEB_PASS}"           # export WEB_PASS='a-strong-password'
```
```bash
python main.py --host 0.0.0.0       # open http://<server-ip>:8080/
sudo ufw allow 8080/tcp             # if the firewall is on
```

The login is HTTP Basic Auth over plain HTTP. For access over the internet, put the server behind a reverse proxy
with HTTPS (nginx/Caddy) or use a VPN. State-changing API calls require the `X-Requested-With: fetch` header
(CSRF protection), which the UI sends automatically. Interactive API docs are at `/api/docs`.

### Where the data is

| What | Where |
|---|---|
| Cameras, persons, face embeddings, detection history, settings | `data/app.db` (SQLite) |
| Uploaded face photos | `data/persons/<person_id>/` |
| Face crops / annotated frames | `output/faces/<camera_id>/`, `output/annotated/<camera_id>/` |
| Every detection (raw log) | `output/detections.jsonl` |

Back up `data/` and `output/` together. To start fresh, stop the app and delete `data/app.db`.
On the next start, cameras and settings are imported from `config.yaml` again.

## Configuration

`config.yaml` holds the things that need a restart (model, device, web host/port/password, paths).
The cameras and the settings marked `[web]` are **imported once**, when the database is empty. After that, the
web UI and the database are the source of truth, so editing those keys in `config.yaml` has no effect.

```yaml
database: {path: ./data/app.db, persons_dir: ./data/persons}
detector:
  model_name: buffalo_l    # restart to change
  device: auto             # auto | cuda | cpu
web:
  host: 127.0.0.1
  port: 8080
```

- `${VAR}` placeholders are filled from environment variables, so passwords stay out of the file.
- Special characters in a camera password **inside the URL** must be URL-encoded (`@` → `%40`, `#` → `%23`).
- Camera passwords are masked as `***` in logs and in the UI. When editing a camera, leave `***` in place to keep the saved password.
- Common URL patterns: Hikvision `/Streaming/Channels/101` (main) or `102` (sub);
  Dahua `/cam/realmonitor?channel=1&subtype=0`. A **sub-stream** (lower resolution)
  is usually plenty for detection and saves a lot of CPU and bandwidth.

## CLI modes

All modes use the same database (cameras, persons, settings).

```bash
python main.py --mode web --show                     # web UI + OpenCV windows
python main.py --mode rtsp --show                    # cameras from the database, no web UI (q / Esc to quit)
python main.py --mode rtsp --view                    # same, plus the web UI
python main.py --mode image --path ./images          # single image or folder (recursive)
python main.py --mode image --path ./images --show   # any key = next image, q = quit
python main.py --mode both --path ./images           # cameras + image folder
```

Useful options:

```bash
python main.py --cameras gate lobby                   # only run these camera ids
python main.py --frame-skip 5 --conf 0.6              # override settings for this run (not saved)
python main.py --device cpu                           # force CPU
python main.py --recognize                            # force recognition on for this run
python main.py --mode image --path ./images --no-save # detect + log only, no image files
python main.py --host 0.0.0.0 --port 9000 --log-level DEBUG
```

Run headless on a server by leaving out `--show`. If there is no `DISPLAY`, `--show` is ignored with a warning.

## Rodent detection

Rats, mice and shrews are detected on every camera by a YOLO11n model (one class, `rodent`)
that runs through ONNX Runtime next to the face detector. Alerts appear on the **Rodents** page,
on the Live page (red boxes, a red camera card and a *Rodent alerts today* counter),
and on Image test uploads.

A box only becomes an alert when it is seen in `confirm_hits` of the last `confirm_window`
model runs (default 3 of 5). After that, the camera gets no new alert for `cooldown_sec`.
This filters out one-frame false positives. Each alert stores the full frame with boxes
and a crop under `output/rodents/<camera_id>/`, plus a row in the `rodent_events` table.
Thresholds are under **Settings → Rodent detection**.

### Training the model

`archive/` (Mice / Rats / Shrew) is a *classification* dataset: it has photos and no boxes.
The scripts add the boxes themselves. Training runs in its own virtualenv, because
the app only needs `onnxruntime`:

```bash
python3 -m venv .venv-train
.venv-train/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv-train/bin/pip install ultralytics onnx onnxslim onnxruntime

# 1. boxes from YOLO-World ("animal" prompt); photos without a confident box go to rejected.txt
.venv-train/bin/python scripts/rodent/autolabel.py
# 2. background frames from your cameras, so the model learns your scene (run while no rodent is visible)
.venv/bin/python scripts/rodent/capture_negatives.py --minutes 10 --every 5
# 3. train YOLO11n and export models/rodent.onnx
.venv-train/bin/python scripts/rodent/train.py --epochs 40
# 4. optional round 2: the trained model labels the rejected photos, then trains again
.venv-train/bin/python scripts/rodent/train.py --relabel runs/rodent/r1/weights/best.pt --name r2
```

Restart `main.py` to load a new model.

**Training on your own camera (most important for accuracy).** The archive photos are close-ups
from the internet. A rodent seen from a ceiling camera is small, seen from above and often in IR.
Open **Rodents → Training frames** to collect real examples:

1. Choose the camera and click *Capture frame*, or use *Auto capture* (for example every 3 s for 2 min)
   while you move a rodent or a toy rodent to different spots.
2. Click a frame and drag a box around each rodent. Use *Zoom* for small ones, then *Save boxes*.
   Mark frames without a rodent as *No rodent*.
3. Run step 3 again. Frames are saved in `datasets/rodent_camera/` and `train.py` adds them
   automatically: 5 copies each in train, and about 20% in val.

## Face recognition

Add people in **Persons** (2–5 photos each: front-facing plus slight angles, good light), then enable recognition in **Settings**.
Each person's photo embeddings are averaged. A face matches the closest person when the cosine similarity
is at least *Similarity threshold* (0.40 by default; raise it to reduce false matches). Unmatched faces are labelled `Unknown`.

An existing `known_faces/` folder (`alice.jpg` or `bob/1.jpg, bob/2.jpg`) is imported into Persons
automatically on the first start.

## Output

```
output/
├── faces/<camera_id>/<YYYYmmdd_HHMMSS_micro>_f<idx>.jpg   # face crops (images: <file-stem>_<ts>_f<idx>.jpg)
├── annotated/<camera_id>/<timestamp>.jpg                  # full frame with boxes
└── detections.jsonl                                       # one JSON object per detected face
logs/face_detection.log                                    # rotating log
```

Example JSON line:

```json
{"camera_id": "cam1", "timestamp": "2026-10-13T16:49:15.819+05:30", "face_index": 0,
 "bbox": [192, 144, 384, 336], "confidence": 0.93,
 "landmarks": [[256.0, 192.0], [320.0, 192.0], [288.0, 240.0], [256.0, 288.0], [320.0, 288.0]],
 "frame_id": 2400, "crop_path": "output/faces/cam1/20261013_164915_819000_f0.jpg"}
```

`bbox` is `[x1, y1, x2, y2]` in pixels. The landmarks are, in order: left eye, right eye, nose, left mouth corner, right mouth corner.

**Face quality + tracking (what gets saved / posted):** every face is checked before it is stored.
The thresholds live in `config.yaml` (`quality:`, `dedup:`) and can be changed live in **Settings**:

| Check | Default | Reject reason |
|---|---|---|
| detection score | `min_confidence: 0.7` | `low_conf` |
| face size (shorter side) | `min_face_size: 80` px | `small` |
| sharpness (Laplacian variance of the face resized to 112×112) | `min_blur_score: 60` | `blur` |
| head pose from the 5 landmarks | `max_yaw: 35`°, `max_pitch: 30`° | `pose` |
| ArcFace embedding available | `require_embedding: true` | `no_embedding` |

Rejected faces are still shown in the live view (grey box with the reason) and counted under the stream
("Skipped: pose 52, small 10"). The reason is written to the debug log, for example
`[cam1] face (905, 60, 1018, 205) not stored: pose (yaw -46.7, pitch 13.8; max 35.0/30.0)`
(run with `--log-level DEBUG`). Use **Image test** to see each face's quality values while you tune the thresholds.

A simple per-camera tracker (IoU / position + embedding similarity) follows each face. For each track it watches
`best_window_sec` (3 s) and stores only the **best** crop (score = size × sharpness × frontal-ness). If the face
leaves earlier, its best crop is stored then. After that, the same face / person is not stored again on that camera
for `cooldown_sec` (30 s). Crops have 25 % padding (`output.crop_margin: 0.25`).

Every stored detection keeps its own **embedding**, bbox and det_score in the database. **Assign person** uses that
saved embedding and does not re-detect the crop, so tight or small crops can be assigned too.
In **Detections** you can select records with the checkboxes and click **Delete selected**, or use **Delete** on a single row.
The record's images are removed too, unless another record still uses them.

`detections.jsonl` gets the same stored events. When tracking is off, every face that passes the quality checks is stored,
rate-limited to one event per camera every `event_interval_sec`. Uploads (Image test / CLI images) are not filtered.

### Database migration and cleanup

The database is upgraded automatically on start. To run the migration by hand (safe to repeat):

```bash
python -m migrations --db data/app.db
```

Migration 001 adds `detections.embedding` and `detections.quality`. It also changes the crop padding from 0.20 to 0.25
and the cooldown from 60 s to 30 s, but only if those settings were still at their old defaults.
Records saved before this migration have no embedding and are hidden from the lists. To delete them, with their images:

```bash
python scripts/cleanup_no_embedding.py --dry-run   # show counts only
python scripts/cleanup_no_embedding.py             # asks before deleting
```

### REST API

Enable it in **Settings → REST API** (URL + optional Bearer token). Each event is POSTed as:

```json
{"camera_id": "cam1", "timestamp": "...", "frame_size": [1920, 1080], "faces": [ ...records as above... ]}
```

Posting runs in a background thread, so a slow server never stalls video processing.
Connection errors, timeouts, 429 and 5xx responses are retried with exponential backoff
(`backoff_factor * 2^attempt`). Other 4xx responses are not retried. You can also call
`utils.post_json_with_retry(url, payload)` directly from your own code.

## Performance notes

- Each camera has its own reader thread. Detection runs in the main loop, round-robin across cameras,
  and always on the newest frame. If detection is slower than the camera, the stale frames are dropped
  instead of queued, so latency stays low.
- Throughput depends on the number of cameras × camera FPS ÷ `frame_skip`. For many cameras (8+),
  run several processes with different `--cameras` lists, or use a GPU.
- Lower `det_size` (e.g. 480 or 320) gives much faster inference but misses small faces.
- Live-view frames are JPEG-encoded only while a browser is watching, and one encoded frame is shared by all viewers.
- `rtsp.transport: tcp` avoids the smeared/grey frames caused by UDP packet loss.

## Troubleshooting

| Problem | Fix |
|---|---|
| `cv2.imshow ... The function is not implemented` | `pip uninstall -y opencv-python-headless opencv-python && pip install opencv-python` |
| `x86_64-linux-gnu-g++: No such file` while installing insightface | `sudo apt install build-essential python3-dev` |
| `numpy.dtype size changed` / binary incompatibility | `pip install "numpy<2" --force-reinstall` then reinstall insightface: `pip install --no-cache-dir --force-reinstall --no-deps insightface==0.7.3` |
| Stream connects but frames are grey/corrupt | Use `transport: tcp`, or switch to the sub-stream |
| `Cannot open rtsp://...` | Test with `ffplay -rtsp_transport tcp "rtsp://user:pass@ip:554/..."`; check the URL path, credentials and URL-encoding |
| `Web server failed to start ... (port in use?)` | Another process is using the port; pick another with `--port 8081` |
| Web UI not reachable from another PC | Use `--host 0.0.0.0` and open the port: `sudo ufw allow 8080/tcp` |
| Person upload says "No face found" | Use a clearer, larger, front-facing photo; the face should be at least ~80 px wide |
| Known person shows as Unknown | Add more photos (different angles / lighting), or use *Assign person* on their Unknown detections; lower the similarity threshold slightly (e.g. 0.35) |
| Edited `cameras:` in config.yaml but nothing changed | After the first run, cameras are managed in the web UI (database). Delete `data/app.db` to re-import |
| GPU not used | Check the startup log line `device=...`; run `nvidia-smi`; reinstall `onnxruntime-gpu[cuda,cudnn]` and make sure plain `onnxruntime` is **not** also installed (`pip uninstall onnxruntime`) |
# rodent_detection
