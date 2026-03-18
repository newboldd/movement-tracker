# WIP: Multi-camera mode support

## Goal
Make the app compatible with three camera modes:
1. **Single camera** — one video per trial, no stereo cropping
2. **Stereo** — one wide video per trial, left/right halves cropped (current behavior)
3. **Multi-camera** — multiple separate video files per trial, not necessarily synchronized

For stereo/multicam, users select a previously calibrated camera setup or are directed to a calibration workflow (checkerboard-based, from video or paired images).

## Completed

### 1. Config: `camera_mode` setting
- **`config.py`** — Added `self.camera_mode: str = "stereo"` (values: `single`, `stereo`, `multicam`)
- Added to `_apply_dict()`, `to_dict()`, and string key list for persistence
- **`routers/settings.py`** — Added `camera_mode` to `SettingsUpdate` model

### 2. DB: `camera_setups` table
- **`db.py`** — Added `camera_setups` table to schema (id, name, mode, camera_count, camera_names JSON, calibration_path, checkerboard_rows/cols, created_at)
- Added `_migrate_add_camera_setups()` migration function, called in `init_db()`

### 3. Camera setups API router
- **`routers/camera_setups.py`** — New file, fully written:
  - CRUD: `GET/POST/DELETE /api/camera-setups`, `GET /api/camera-setups/{id}`
  - `POST /api/camera-setups/calibrate-from-video` — extracts checkerboard frames from stereo video, runs `cv2.stereoCalibrate`, saves OpenCV YAML
  - `POST /api/camera-setups/calibrate-from-images` — same from paired image directory (left_01.png + right_01.png)
  - Background threaded workers with job progress tracking
  - Helper: `_find_image_pairs()`, `_save_calibration_yaml()`

### 4. App registration
- **`app.py`** — Imported `camera_setups` router, registered with `app.include_router()`, added `/calibration` page route and to no-cache middleware list

### 5. Video service: single camera support
- **`services/video.py`** — Modified `_extract_frame_cached()` and `extract_frame_raw()` to skip cropping when `camera_mode == "single"` (return full frame)
- Modified `get_subject_videos()` to accept optional `camera_name` param for multicam filtering (videos named `{Subject}_{Trial}_{CameraName}.mp4`)

### 6. Calibration page HTML
- **`static/calibration.html`** — New page with:
  - List of existing camera setups
  - Create form (name, mode, camera count, camera names)
  - Calibration section: source selection (video vs image pairs), checkerboard params, file browser, video trimmer, progress bar

### 7. Calibration page JavaScript
- **`static/js/calibration.js`** — Complete with:
  - `loadSetups()` / `selectSetup()` / `deleteSetup()` — CRUD list with status dots
  - `showCreateForm()` / `createSetup()` — create form with mode toggle, camera names
  - `loadFileBrowser()` / `browseTo()` — reusable file browser (same `/api/files` as onboarding)
  - `selectCalibVideo()` / `setIn()` / `setOut()` — video selection + trim controls (I/O keyboard shortcuts)
  - `selectImageDir()` — directory selection for image pair calibration
  - `runCalibration()` — calls calibrate-from-video or calibrate-from-images, streams job progress via `API.streamJob()`
  - `setSource()` — toggle between video and image pair calibration modes

### 8. Onboarding flow: camera setup selection
- **`static/onboarding.html`** — Added Step 1b (camera setup dropdown + "New Setup" link)
- **`static/js/onboarding.js`** — Added:
  - Camera mode detection (fetches settings, skips step1b in `single` mode)
  - `_loadCameraSetups()` — populates dropdown from `/api/camera-setups`
  - `confirmCameraSetup()` — stores selected setup and advances to file browser
  - `_updateStepNumbers()` — dynamically adjusts step numbers based on mode

### 9. Settings page: camera mode selector
- **`static/settings.html`** — Camera Mode dropdown (single/stereo/multicam) in Cameras section
- **`static/js/settings.js`** — Already loads and saves `camera_mode`

## Remaining (future work)

### 10. Multicam video selection in onboarding — NOT STARTED
- **`static/js/onboarding.js`** — In `multicam` mode:
  - Allow selecting multiple videos (one per camera per trial)
  - Show camera name labels for each video slot
- **`routers/video_tools.py`** — Update `ProcessSubjectRequest` and `_do_process_subject` to handle multicam naming (`{Subject}_{Trial}_{CameraName}.mp4`)

### 11. Multicam video handling in labeler — NOT STARTED
- **`services/video.py`** — `build_trial_map()` and frame extraction need multicam awareness (show frames from each camera file side by side or in tabs)
- **`routers/labeling.py`** — Frame serving for multicam subjects
- **`static/js/labeler.js`** — Display frames from multiple camera files

### 12. Discovery for multicam — NOT STARTED
- **`services/discovery.py`** — `_find_videos()` should understand multicam naming conventions

## Architecture Notes
- Camera setups are stored in DB (`camera_setups` table) and reference calibration YAML files on disk
- Subjects link to setups via `subjects.camera_name` field (existing field, now maps to setup name)
- Settings `calibrations` dict is auto-updated when calibrations complete
- Multicam video naming convention: `{Subject}_{Trial}_{CameraName}.mp4`
- Single camera mode simply skips all stereo cropping logic
