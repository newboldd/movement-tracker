# Movement Tracker

A web app for video-based finger-tapping analysis in movement disorder research. Built with FastAPI + vanilla JS, designed for clinical researchers with zero programming experience.

## Quick Start

```bash
# Mac
./setup.sh

# Windows
run.bat

# With external data directory
MT_DATA_DIR=~/data/movement-tracker ./setup.sh
```

Opens at http://localhost:8080. Uses `.env` for persistent config (e.g., `MT_DATA_DIR`).

## Architecture

**Backend**: FastAPI (Python), SQLite, OpenCV, MediaPipe, ffmpeg (via imageio-ffmpeg)
**Frontend**: Vanilla JS (no framework), HTML5 Canvas for video/labeling, Chart.js-style custom plotting
**No build step** — static files served directly.

### Directory Layout

```
movement_tracker/
├── app.py              # FastAPI app, routes, startup hooks
├── config.py           # Settings, paths (DATA_DIR, DB_PATH, etc.)
├── db.py               # SQLite schema, migrations, get_db_ctx()
├── routers/            # API endpoints (one file per feature)
│   ├── labeling.py     # Frame labeling, MediaPipe crop boxes
│   ├── deidentify.py   # Face blur, hand protection, rendering
│   ├── results.py      # Distance traces, movements, group analysis
│   ├── subjects.py     # Subject CRUD, filesystem sync
│   ├── video_tools.py  # Probe, trim, onboarding pipeline
│   └── ...
├── services/           # Business logic
│   ├── mediapipe_prelabel.py  # Hand/pose detection, npz storage
│   ├── deidentify.py          # Blur mask building, ffmpeg pipe rendering
│   ├── video.py               # Frame extraction, trial maps, stereo crop
│   ├── ffmpeg.py              # Find ffmpeg (PATH → imageio-ffmpeg fallback)
│   └── ...
└── static/
    ├── js/
    │   ├── labeler.js      # 213KB — main labeling canvas (also MediaPipe/DLC)
    │   ├── deidentify.js   # 102KB — blur editor, timeline, hand protection
    │   ├── results.js      # Analysis plots
    │   ├── dashboard.js    # Subject overview
    │   └── ...
    └── *.html              # One page per feature, nav.js shared
```

### Data Layout (external, via MT_DATA_DIR)

```
~/data/movement-tracker/
├── dlc_app.db          # SQLite database
├── settings.json       # Runtime settings
├── sample_data/        # Downloaded sample videos
├── videos/             # Trimmed trial videos (symlink OK)
│   └── deidentified/   # Rendered blur videos
├── dlc/                # Per-subject DLC projects
│   └── {subject}/
│       ├── config.yaml
│       ├── mediapipe_prelabels.npz  # Hand landmarks
│       ├── pose_prelabels.npz       # Pose landmarks
│       ├── labeled-data/
│       └── ...
└── calibration/        # Stereo camera calibrations
```

## Key Concepts

### Subjects & Trials
- Each **subject** has one or more **trials** (L1, L2, R1, R2 = left/right hand, trial 1/2)
- Trials are discovered from video filenames: `{Subject}_{Trial}.mp4`
- Camera modes: `stereo` (side-by-side 3840x1080), `multicam` (separate files per camera), `single`
- Camera names: OS (left eye/camera) and OD (right eye/camera)

### Frame Coordinate System
- Stereo videos: frames are 3840x1080, left half = OS, right half = OD
- The backend splits stereo frames when serving per-camera views
- All landmark coordinates are stored in per-camera-half pixel space (0–1920 for stereo)

### MediaPipe Data (npz files)
- `mediapipe_prelabels.npz`: OS_landmarks, OD_landmarks (N, 21, 2), distances, confidence, run history
- `pose_prelabels.npz`: OS_pose, OD_pose (N, 33, 2), pose_confidence
- Landmarks are in image pixel coords for the camera half
- Run history: `distances_run_N` / `crop_run_N` keys track previous MediaPipe runs

### Deidentification Pipeline
- Face detection → automatic blur spots per face cluster
- Custom blur spots (oval or rectangle, draggable on canvas)
- Hand protection: morphological close on hand keypoints + forearm triangle
- Timeline: visual face/custom/hand rows with drag-to-edit
- Render: pipe frames to ffmpeg, ROI-only blur for efficiency
- Preview: server-side single-frame blur using same pipeline as render

### Database
- SQLite with WAL mode, dict row factory
- Auto-migrations in `init_db()` — new columns/tables added on startup
- Key tables: subjects, jobs, label_sessions, frame_labels, blur_specs, face_detections, blur_hand_settings, mp_crop_boxes

## Patterns

### Adding a New Page
1. Create `static/newpage.html` with nav (copy from any existing page)
2. Create `static/js/newpage.js` as an IIFE: `const page = (() => { ... return { publicFn }; })();`
3. Add route in `app.py`: `@app.get("/newpage") def newpage(): return FileResponse(...)`
4. Add nav link to ALL HTML files (grep for `DLC</a>` to find the nav block)
5. Add `?v=N` cache buster to script tag

### Adding an API Endpoint
1. Add to existing router in `routers/` or create new one
2. If new router: import and `app.include_router()` in `app.py`
3. Use `get_db_ctx()` context manager for DB access
4. Return dicts — FastAPI auto-serializes. Watch for numpy types (use `int()`, `float()`)

### Database Migrations
1. Add `CREATE TABLE IF NOT EXISTS` to `SCHEMA` string in `db.py`
2. Add `_migrate_add_X(conn)` function
3. Call it from `init_db()` before `conn.executescript(SCHEMA)`

### Canvas Video Pages
- labeler.js, deidentify.js, mano.js all follow similar patterns
- `loadFrame()` fetches JPEG from API, draws on canvas
- `render()` redraws canvas with overlays
- `scale`/`offsetX`/`offsetY` for zoom/pan, `hasUserZoom` prevents auto-fit
- Keyboard: arrow keys for frames, space for play, e for camera toggle, r for reset zoom

### Cross-Page Subject Persistence
- `sessionStorage.setItem('lastSubjectId', id)` when viewing a subject
- Other pages read it on load to pre-select the same subject

## Common Gotchas

- **MediaPipe version**: pinned `<0.10.19` because newer versions removed `mediapipe.solutions` API
- **numpy serialization**: FastAPI can't serialize `numpy.int64` etc. — cast to `int()`/`float()` in responses
- **Stereo frame offsets**: some videos have negative-PTS pre-roll frames that OpenCV sees but browsers skip. `_compute_frame_offset()` handles this
- **Windows compatibility**: `run.bat` handles portable Python install, conda detection, App Execution Aliases
- **ffmpeg**: bundled via `imageio-ffmpeg` pip package, `get_ffmpeg_path()` finds it
- **Cache busting**: bump `?v=N` in HTML script tags when changing JS files
- **macOS Gatekeeper**: `.command` file needs "Open Anyway" in System Settings on first run from zip download

## Testing

No formal test suite. Verify with:
```bash
python3 -c "import ast; ast.parse(open('movement_tracker/FILE.py').read()); print('OK')"
python3 -c "code=open('movement_tracker/static/js/FILE.js').read(); print(f'{code.count(chr(123))}/{code.count(chr(125))}')"
curl -s http://localhost:8080/api/subjects  # quick API check
```

## Owner
Dillan Newbold (newboldd) — neurology research, movement disorder analysis
