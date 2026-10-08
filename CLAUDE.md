# DLC Labeler

A web app for DeepLabCut keypoint labeling on stereo video, for clinical
researchers with no programming background. FastAPI + vanilla JS, no build
step. A stripped-down fork of Movement Tracker: it keeps the labeling,
MediaPipe and DeepLabCut paths and drops everything else (skeleton fitting,
HRnet, event detection, deidentification, group analysis, remote execution).

## Quick start

```bash
./setup.sh                 # base install (labeling + MediaPipe), then run
./setup.sh --with-dlc      # also install DeepLabCut
./setup.sh --reinstall     # rebuild .venv from scratch

run.bat                    # Windows equivalents
run.bat --with-dlc
```

Opens at http://localhost:8080. `.env` holds local config
(`DLC_DATA_DIR`, `DLC_PORT`).

## Architecture

**Backend**: FastAPI, SQLite, OpenCV, MediaPipe, scikit-learn (k-means
frame selection), ffmpeg (via imageio-ffmpeg)
**Frontend**: vanilla JS, HTML5 Canvas for video and labeling, three.js
(vendored, no CDN) for the 3D view
**No build step** — static files are served directly.

```
dlc_labeler/
├── app.py              # routes, static mount, startup
├── config.py           # Settings singleton, DATA_DIR resolution
├── db.py               # schema + migrations
├── models.py           # pydantic models, STAGES
├── routers/
│   ├── labeling.py     # the Label page's entire API
│   ├── subjects.py     # CRUD + filesystem discovery
│   ├── queue.py        # the Jobs page's API
│   ├── jobs.py         # job detail, log tail, SSE, cancel
│   ├── packages.py     # list packages, import returned labels
│   └── settings.py     # settings, status, data directory
├── services/
│   ├── video.py              # trial maps, frame extraction, stereo split
│   ├── mediapipe_prelabel.py # hand passes + best-per-frame fusion
│   ├── labels.py             # commit to DLC, corrections CSVs
│   ├── frameselect.py        # DLC's k-means, on MediaPipe crops
│   ├── labelpack.py          # export/read/import labeling packages
│   ├── dlc_predictions.py    # read DLC CSVs back as layers
│   ├── dlc_pipeline.py       # STANDALONE: train/refine/analyze subprocess
│   ├── dlc_wrapper.py        # config.yaml repair
│   ├── calibration.py        # stereo calibration + triangulation
│   ├── queue_manager.py      # CPU and GPU lanes
│   ├── local_executor.py     # launches jobs as subprocesses
│   ├── worker.py             # STANDALONE: MediaPipe subprocess entry point
│   ├── jobs.py               # subprocess registry, progress, cancel
│   ├── job_history.py        # append-only lifetime log
│   ├── discovery.py          # infer stage from the filesystem
│   ├── trace.py              # distance-trace spike filter
│   └── ffmpeg.py             # find ffmpeg (PATH → imageio-ffmpeg)
└── static/
    ├── label.html / js/label.js       # the one labeling page (~2.4k lines)
    ├── jobs.html  / js/jobs.js
    ├── index.html / js/subjects.js
    ├── settings.html / js/settings.js
    ├── js/nav.js                      # shared header, built in JS
    └── vendor/                        # three.js + OrbitControls, vendored
```

### Data layout (outside the repo, at `DLC_DATA_DIR`)

See README.md. The database filename and schema match Movement Tracker's
on purpose: pointing both apps at one data directory is supported.

## Key concepts

### Subjects and trials
- Trials are discovered from video filenames: `{Subject}_{Trial}.mp4`.
  The filename is the record — `/api/subjects/sync` rebuilds the subject
  list from the videos and DLC projects on disk.
- Camera modes: `stereo` (one side-by-side file), `multicam` (one file
  per camera), `single`. Per subject, defaulting from settings.
- Camera names default to OS (left half) and OD (right half).

### Frame coordinates
- Stereo frames are split at the midline when served; all stored
  coordinates are in per-camera-half pixel space.
- Frame numbers are global across a subject's trials; `build_trial_map`
  maps global → (video, local frame).
- `frame_offset` handles videos whose container exposes negative-PTS
  pre-roll frames to OpenCV that browsers skip.

### MediaPipe passes
Per trial, under `dlc/{subject}/{trial}/`:
`mediapipe.npz` (forward), `mediapipe_reverse.npz`,
`mediapipe_static.npz` (frame-by-frame), `mediapipe_cropped.npz`
(forward with a crop box), `mediapipe_combined.npz` (best per frame).
Arrays are trial-local `(n, 21, 2)` with a `start_frame` offset. Each has
a `.params.json` sidecar recording how it was produced.

`build_combined_mp_npz_for_trial` rebuilds the fusion automatically
whenever a source pass is written and two or more exist.

### Labeling packages
A package is a folder under `DATA_DIR/packages/` holding frames to be
labeled elsewhere: `package.json` (manifest), `frames.csv` (one row per
image, with its crop offset and source frame), `frames/<Trial>_<Camera>/
img0042.png`, and `labels/labels.csv`.

`services/frameselect.py` picks the frames with DeepLabCut's own
`KmeansbasedFrameselectioncv2` — transcribed from it, not approximated —
applied to per-frame crops around the MediaPipe hand. At DLC's 30px
working width a whole camera half leaves the hand about two pixels, so
cropping first is what makes the clusters be postures rather than arm
positions.

A package is read back as an ordinary subject whose trials have
`kind == "frames"`: `build_trial_map` returns them when a subject has no
videos, `extract_frame` reads the image file, and `/video` 404s so the
page falls back to its JPEG path. Labels save through to
`labels/labels.csv` on every write, because the folder is what travels.

`import_package_labels` maps each image back by **trial name and local
frame**, not by the global frame number the package recorded — global
numbering shifts if a trial is added. Crop offsets turn the crop's
coordinates back into camera-half coordinates.

### Layers
Everything the Label page draws is a layer with one shape. Bodypart
layers (`dlc`, `refine`, `corrections`, `committed`) are subject-wide
`{camera: {bodypart: [[x,y]|null, ...]}}`; MediaPipe layers are per trial
with all 21 joints, shipped as base64 float32 (`_f32` / `decodeF32`)
because JSON for a trial's 2D+3D arrays is megabytes of text.

### The DeepLabCut loop
`initial` session → commit → `labeled-data/round{N}/` → **train** →
`labels_v1/` → `corrections` session → `corrections/` → `refine` session
(pick frames) → commit → `labeled-data/round{N+1}/` → **refine** →
`labels_v2/`.

`services/dlc_pipeline.py` runs train/refine/analyze as one subprocess and
has **no `dlc_labeler` imports on purpose** — the interpreter with
DeepLabCut may be a different venv or conda env (Settings → Python
executable). Same for `services/worker.py`, which puts PROJECT_DIR on
`sys.path` itself so absolute-path invocation works under Windows
portable Python.

## Patterns

### Adding an API endpoint
1. Add to a router in `routers/`, or create one and `include_router` it
   in `app.py`.
2. Use `get_db_ctx()` for database access.
3. Return dicts; FastAPI serializes them. Cast numpy scalars with
   `int()` / `float()` first — it cannot serialize `numpy.int64`.

### Adding a page
1. `static/newpage.html` with an empty `<div class="header"></div>` —
   `nav.js` builds the header, so there is no nav markup to copy.
2. `static/js/newpage.js` as an IIFE.
3. Add the route and the `PAGES` entry in `app.py`.
4. Add the link to `NAV_LINKS` in `nav.js`.
5. Bump `?v=N` on the script tag.

### Database migrations
`init_db()` runs reshaping migrations first, then
`CREATE TABLE IF NOT EXISTS`. Add columns with `_add_missing_columns`;
anything that rewrites a table needs its own function, run before the
schema pass (which would otherwise be a silent no-op).

### Canvas conventions (label.js)
- `scale` / `offsetX` / `offsetY` are the view transform;
  `screenToImage` / `imageToScreen` convert.
- `hasUserZoom` suppresses auto-zoom once the user has panned or zoomed.
- `render()` redraws from the cached JPEG; `drawVideoToCanvas()` redraws
  from the `<video>` element. `videoFrameMode` says which is live.
- Playback uses `requestVideoFrameCallback` so overlays can never be
  drawn for a frame that is not on screen.

## Gotchas

- **MediaPipe version**: pinned `<0.10.19`; newer releases removed the
  `mediapipe.solutions` API this uses. That also caps Python at 3.12.
- **Frame seeking**: OpenCV's `POS_FRAMES` is off by one on many H.264
  files. The browser's decoder is the ground truth, so the page draws
  from `<video>` whenever it can and falls back to server JPEGs.
- **ON CONFLICT targets** must match a real UNIQUE constraint.
  `mp_crop_boxes` is unique on `(subject_id, trial_idx, camera_name,
  model_name)` — all four.
- **One crop box model name**: `BBOX_MODEL = "run-mediapipe"` in
  `routers/labeling.py`. The box you drag and the box MediaPipe crops to
  must be the same row.
- **DATA_DIR is resolved at import time**, so changing it re-execs the
  process (exit code 42; the launchers loop on it).
- **three.js is vendored** under `static/vendor/`. Do not switch it back
  to a CDN — these machines are often offline or firewalled.
- **Cache busting**: bump `?v=N` in the HTML when changing a JS file.
- **macOS Gatekeeper**: a downloaded zip needs "Open Anyway" once under
  System Settings → Privacy & Security.
- **Shared data directory**: pointing this app and the full Movement
  Tracker at one data directory is supported, so anything touching the
  shared tables must stay in its own lane. The queue manager filters
  every read by `RESOURCE_MAP`'s job types — add a job type there and to
  `_run_job` together (and to `STEP_DEFINITIONS` and `worker.py`'s
  `JOB_DISPATCH` to make it reachable), or it will be queued and never
  drained. Settings
  keys this app does not model are preserved on save (`Settings._foreign`).
- **Port and browser**: both apps default to 8080. `scripts/launcher.py`
  answers both launch questions — `free <port>` (by binding, not by
  parsing lsof/netstat) and `open <port>` (which waits for the server to
  answer before opening a browser). Do not reintroduce an unconditional
  kill, a netstat parse, or a fixed sleep before opening the URL: a sleep
  sends the browser to whatever else holds the port when startup fails,
  which on these machines is Movement Tracker.

## Testing

No formal test suite. Verify with:

```bash
python3 -c "import ast; ast.parse(open('dlc_labeler/FILE.py').read()); print('OK')"
node --check dlc_labeler/static/js/FILE.js
DLC_DATA_DIR=/tmp/t python3 -m uvicorn dlc_labeler.app:app --port 8099
curl -s http://localhost:8099/api/subjects
```

## Owner

Dillan Newbold (newboldd) — neurology research, movement disorder analysis.
Built for NYU medical students running DeepLabCut on the rest of the
subject set.
