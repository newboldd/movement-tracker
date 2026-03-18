# WIP: Multi-camera mode — COMPLETED

All three remaining tasks have been implemented.

## 1. Multicam video selection in onboarding — DONE
- **`static/js/onboarding.js`** — In multicam mode:
  - Fetches full camera setup with camera names on `confirmCameraSetup()`
  - Tracks `cameraNames[]` and `currentCameraIdx` for per-camera video selection
  - `_updateCameraLabel()` shows which camera is being selected for
  - `addSegment()` tags segments with `camera_name` in multicam mode
  - `renderSegments()` shows camera name badges on segments
  - `startProcessing()` sends `camera_name` per segment to API
  - Review display shows multicam output filenames (`Subject_Trial_Camera.mp4`)
- **`static/onboarding.html`** — Added `#cameraLabel` div for camera indicator
- **`routers/video_tools.py`** — `SegmentDef` now has optional `camera_name` field;
  `_do_process_subject` outputs `{Subject}_{Trial}_{Camera}.mp4` when set

## 2. Multicam video handling in labeler — DONE
- **`services/video.py`**:
  - `_group_multicam_videos()` — groups per-camera files by trial stem
  - `build_trial_map()` — returns `cameras` list per trial, `trial_stem` field
  - `_resolve_frame()` — now returns trial dict (for camera lookup)
  - `_resolve_camera_path()` — finds camera-specific video path for a given side
  - `extract_frame()` — resolves multicam camera file before extraction
  - `_extract_frame_cached()` / `extract_frame_raw()` — skip stereo cropping in multicam mode
- **`routers/labeling.py`**:
  - `get_session_info()` — returns `camera_mode` and per-trial `cameras` list
  - `get_frame()` — relaxed side validation in multicam mode
- **`static/js/labeler.js`**:
  - Added `cameraMode` state, set from session info
  - `_getActiveCameraNames()` — returns per-trial cameras in multicam mode
  - `_currentTrial()` — helper to find trial for current frame
  - `toggleSide()` — uses active camera names, skips viewport shift in multicam
  - Three video draw locations updated to use full-frame (no crop) in multicam/single
  - `recomputeCameraShift()` — skips in multicam/single mode
  - `loadSession()` — defaults `currentSide` to first multicam camera name

## 3. Discovery for multicam — DONE
- **`services/discovery.py`** — `_find_videos()` groups multicam files by trial stem
  so dashboard `video_count` reflects trials, not individual camera files

## 4. Video viewer multicam — DONE (from Codespace)
- **`routers/mano.py`**:
  - `get_trial_video()` — accepts `camera` query param for multicam file selection
  - `get_video_list()` — returns `cameras` list per trial for multicam
- **`static/js/videos.js`** — Already has `loadCameraFile()`, `updateCameraToggle()`,
  multicam-aware `toggleSide()` from Codespace work

## Architecture Notes
- Camera setups stored in DB (`camera_setups` table) with calibration YAML files on disk
- Multicam video naming: `{Subject}_{Trial}_{CameraName}.mp4`
- `_group_multicam_videos()` infers grouping from filenames (no DB lookup needed)
- Single camera mode skips all stereo cropping logic
- Multicam mode: each camera is a separate full-frame video file, no cropping
- Stereo mode: single side-by-side video, cropped to left/right halves
