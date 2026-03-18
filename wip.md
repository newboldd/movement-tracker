# WIP: Multi-camera mode — remaining work

## 1. Multicam video selection in onboarding — NOT STARTED
- **`static/js/onboarding.js`** — In `multicam` mode:
  - Allow selecting multiple videos (one per camera per trial)
  - Show camera name labels for each video slot
- **`routers/video_tools.py`** — Update `ProcessSubjectRequest` and `_do_process_subject` to handle multicam naming (`{Subject}_{Trial}_{CameraName}.mp4`)

## 2. Multicam video handling in labeler — NOT STARTED
- **`services/video.py`** — `build_trial_map()` and frame extraction need multicam awareness (show frames from each camera file side by side or in tabs)
- **`routers/labeling.py`** — Frame serving for multicam subjects
- **`static/js/labeler.js`** — Display frames from multiple camera files

## 3. Discovery for multicam — NOT STARTED
- **`services/discovery.py`** — `_find_videos()` should understand multicam naming conventions

## Architecture Notes
- Camera setups are stored in DB (`camera_setups` table) and reference calibration YAML files on disk
- Subjects link to setups via `subjects.camera_name` field (existing field, now maps to setup name)
- Settings `calibrations` dict is auto-updated when calibrations complete
- Multicam video naming convention: `{Subject}_{Trial}_{CameraName}.mp4`
- Single camera mode simply skips all stereo cropping logic
