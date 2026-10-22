# DLC Labeler

A web app for DeepLabCut keypoint labeling on stereo video, for clinical
researchers with no programming background. FastAPI + vanilla JS, no build
step.

**It labels frames. It runs nothing.** No MediaPipe, no DeepLabCut, no job
queue. It reads the MediaPipe landmarks Movement Tracker left on disk, uses
them to aim the labeling, and writes a DeepLabCut training set. Training,
inference, corrections and refinement happen elsewhere.

## Quick start

```bash
./setup.sh                 # install and run
./setup.sh --reinstall     # rebuild .venv from scratch

run.bat                    # Windows equivalent
```

Opens at http://localhost:8080. `.env` holds local config
(`DLC_DATA_DIR`, `DLC_PORT`).

## Architecture

**Backend**: FastAPI, SQLite, OpenCV, scikit-learn (k-means frame
selection), ffmpeg (via imageio-ffmpeg). Note what is *not* here:
`mediapipe` is not a dependency, which is also why Python is not capped
at 3.12.
**Frontend**: vanilla JS, HTML5 Canvas for video and labeling
**No build step** — static files are served directly.

```
dlc_labeler/
├── app.py              # routes, static mount, startup
├── config.py           # Settings singleton, DATA_DIR resolution
├── db.py               # schema + migrations
├── models.py           # pydantic models, STAGES
├── routers/
│   ├── labeling.py     # the Label page's API: sessions, frames, labels,
│   │                   #   frame selection, commit to DLC
│   ├── subjects.py     # CRUD + filesystem discovery
│   ├── packages.py     # labeling packages: list, export, import
│   └── settings.py     # settings, status, data directory, self-update
├── services/
│   ├── video.py           # trial maps, frame extraction, stereo split
│   ├── mediapipe_data.py  # READ the npz Movement Tracker wrote
│   ├── frameselect.py     # DLC's k-means on MediaPipe crops
│   ├── labelpack.py       # export / read / import labeling packages
│   ├── labels.py          # write the DLC training set
│   ├── calibration.py     # stereo calibration + triangulation
│   ├── discovery.py       # infer stage from the filesystem
│   └── ffmpeg.py          # find ffmpeg (PATH → imageio-ffmpeg)
└── static/
    ├── label.html / js/label.js       # the labeling page
    ├── index.html / js/subjects.js
    ├── settings.html / js/settings.js
    └── js/nav.js                      # shared header, built in JS
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

### MediaPipe landmarks
Read, never written. `dlc/{subject}/{trial}/mediapipe.npz` holds
trial-local `(n, 21, 2)` arrays per camera with a `start_frame` offset;
`services/mediapipe_data.py` is the only thing that reads them and it
imports nothing from the `mediapipe` package. A subject with no npz
still labels — it is simply not pre-aimed.

### Guided labeling
The whole Label page. There is no other mode: `select_training_frames`
picks N frames per *subject* (one pick is one frame, labeled on every
camera, so the clustering feature is both cameras' MediaPipe crops side
by side), saved to `dlc/{subject}/selection.json`.

The page walks them frame-major, then bodypart, then camera (thumb OS,
thumb OD, index OS, index OD) — `guidedRows()` defines that order and
the grid rows reuse it. Only the point being asked for can be placed:
every labeled frame is a k-means pick, so a label on a neighbouring
frame would never be saved but would still look placed. `guidedBusy`
drops clicks while the view is moving. `bodypart_joints` in session info
maps each bodypart to the landmark the zoom aims at, matched by name
server-side. Auto-advance is off by default, remembered in
`localStorage`.

`POST /{id}/selection` runs on a thread and reports into
`_selection_progress`, polled at `/selection/progress` — in memory
because it is live state of this process, meaningless after a restart.
Progress has named phases; MiniBatchKMeans offers no callback, so the
clustering band cannot tick and naming it is what stops it reading as a
hang. scikit-learn's import costs more than the fit on anything but a
huge matrix, so `_warm_kmeans` loads it during the decode.

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

### What the page draws
Two things: this session's labels, always, and the MediaPipe landmarks
behind one checkbox. The landmarks arrive per trial as base64 float32
(`_f32` / `decodeF32`) because JSON for a trial's arrays is megabytes of
text. The trace under the video plots the MediaPipe aperture always —
it is how you tell a frame worth labeling from one where the hand is
mid-blur, which you want before deciding to look at the landmarks.

### The DeepLabCut output
`initial` session → **Save labels** → `labeled-data/round{N}/` with a
CollectedData CSV, plus a `config.yaml` so the directory is a valid DLC
project. That is where this app stops. Training, inference, corrections
and refinement are somebody else's job — run DeepLabCut against the
project it writes.

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

- **Frame seeking**: OpenCV's `POS_FRAMES` is off by one on many H.264
  files. The browser's decoder is the ground truth, so the page draws
  from `<video>` whenever it can and falls back to server JPEGs.
- **DATA_DIR is resolved at import time**, so changing it re-execs the
  process (exit code 42; the launchers loop on it).
- **No CDNs.** These machines are often offline or firewalled, so every
  asset the pages need is served from `static/`. Vendor anything new.
- **Cache busting**: bump `?v=N` in the HTML when changing a JS or CSS
  file — and in *every* page that links it, for `main.css`.
- **Camera-switch alignment must measure both sides the same way.**
  `cameraShift` uses labels only for bodyparts present on *both* cameras,
  and MediaPipe for both otherwise. Mixing them — one thumb label on OS
  against a 21-landmark centroid on OD — compares a fingertip with the
  middle of a palm, and the view jumps the moment you label one camera.
- **Stereo camera switching must not reload the video.** Both halves come
  from one file, so only `multicam` changes file when the camera changes.
  Reloading on every switch re-buffers and re-seeks, which guided
  labeling does on every other point.
- **macOS Gatekeeper**: a downloaded zip needs "Open Anyway" once under
  System Settings → Privacy & Security.
- **Restarting the process** (`_schedule_restart` in `routers/settings.py`,
  used by the data-directory change and the Update button): prefer exit
  code 42, which the launchers loop on — they know how the app was
  started. Re-exec is the fallback, and it must rebuild the command from
  `__main__.__spec__`, not from `sys.argv`: under `python -m uvicorn`,
  `sys.argv[0]` is uvicorn's `__main__.py`, and re-execing that as a
  script puts uvicorn's package directory on `sys.path`, so the next
  `import logging` finds `uvicorn/logging.py` and the process dies.
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
