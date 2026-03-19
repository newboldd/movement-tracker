# Movement Tracker

A self-hosted web app for video-based fine motor assessment. Designed for clinical research on conditions such as Parkinson's disease, MSA, and essential tremor — specifically tasks like finger tapping, alternating hand movements, and fist opening/closing that are standard in neurological exams.

The app manages the full pipeline: importing and de-identifying subject videos, labeling keypoints with DeepLabCut, running MediaPipe hand tracking, fitting 3D hand models with MANO, and computing movement metrics for analysis.

---

## Features

- **Subject management** — import videos, trim trials, organise subjects by diagnosis group
- **Face de-identification** — automatic face blurring before storage or sharing
- **Video viewer** — scrub, zoom, and switch camera views; MediaPipe-guided auto-crop
- **DLC labeling** — frame-by-frame keypoint annotation with keyboard shortcuts, auto-detection of movement events (open/peak/close/pause)
- **MediaPipe prelabeling** — hand landmark tracking used to guide crop and camera selection; re-run with a custom bounding box per camera for improved detection
- **MANO viewer** — 3D hand pose overlaid on stereo video with distance traces
- **Results dashboard** — per-trial movement metrics with configurable sequence and group comparisons
- **Remote processing** — run DLC inference and MediaPipe on a GPU server over SSH; results download automatically
- **Configurable** — camera names, bodyparts, diagnosis groups, event types all set via the Settings page or environment variables

---

## Quick start

**Requirements:** Python 3.9+, macOS or Linux (Windows via WSL or the included `run.bat`)

```bash
git clone https://github.com/newboldd/movement-tracker
cd movement-tracker
./setup.sh
```

`setup.sh` will:
1. Create a virtual environment and install Python dependencies
2. Download the sample video (~24 MB) from Zenodo on first run
3. Start the app and open `http://localhost:8080` in your browser

> **Shortcut:** The repo includes `Movement Tracker.command` — on macOS you can drag it to your Dock or Desktop to launch the app with a double-click.

On first launch, open **Settings** and point the app at your video directory and DeepLabCut project directory. To explore with the included sample data, set the video directory to `sample_data/` and click **Sync from disk** on the dashboard.

> **Workshop?** See [`WORKSHOP.md`](WORKSHOP.md) for a guided 45-minute hands-on walkthrough — covers installation, MediaPipe, labeling, bounding box cropping, and results. Ideal for lab meetings or onboarding new team members.

### Windows

```bat
run.bat
```

Or, using WSL, follow the same steps as macOS/Linux above.

---

## Typical workflow

```
1. Add Subject    (/onboarding)
   Browse to a source video → trim trial segments → set labels (L1, R1, ...) → process
   Face blurring happens here; trimmed trials are saved to your video directory.

2. DLC Labeling   (/labeling-select)
   Open a subject → label keypoints frame by frame → use auto-detect to mark events.

3. MediaPipe      (runs via Processing tab or automatically during import)
   Hand landmarks extracted per trial; used for auto-crop in the video viewer.
   Draw a bounding box in the labeler to re-run on a cropped region per camera.

4. MANO           (/mano)
   3D hand pose fitted to stereo video; distance traces computed per trial.

5. Results        (/results)
   Per-subject and group-level movement metrics; export to CSV.
```

---

## Sample data

A short finger-tapping example video (Con01_R1.mp4, ~24 MB, stereo) is available on Zenodo:

> Newbold, D. (2025). *Finger tapping example* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.19099855

This file is downloaded automatically by `setup.sh`. To fetch it manually:

```bash
python scripts/download_sample.py
```

---

## Configuration

Settings are stored in `movement_tracker/settings.json` (created on first run, gitignored). All settings can also be provided as environment variables — useful for deployment or CI.

| Setting | Env var | Description |
|---|---|---|
| Video directory | `DLC_APP_VIDEO_DIR` | Root folder containing subject video subfolders |
| DLC directory | `DLC_APP_DLC_DIR` | DeepLabCut project directory |
| Camera names | `DLC_APP_CAMERA_NAMES` | Comma-separated, e.g. `OS,OD` |
| Bodyparts | `DLC_APP_BODYPARTS` | Comma-separated, e.g. `thumb,index` |
| Port | `DLC_APP_PORT` | Default: `8080` |
| Remote host | `DLC_APP_REMOTE_HOST` | SSH target, e.g. `user@gpu-server` |
| Remote Python | `DLC_APP_REMOTE_PYTHON` | Python executable on remote, e.g. `/home/user/envs/dlc/bin/python` |
| Remote work dir | `DLC_APP_REMOTE_WORK_DIR` | Scratch directory on remote |
| Remote SSH key | `DLC_APP_REMOTE_SSH_KEY` | Optional path to private key |
| Remote SSH port | `DLC_APP_REMOTE_SSH_PORT` | Default: `22` |

### Remote GPU processing

The **Processing** tab lets you run DLC inference and MediaPipe on a remote machine over SSH without installing GPU dependencies locally. Videos are uploaded to the remote host, processed, and results are downloaded when the job completes. Configure the connection in **Settings → Remote Processing**.

### Calibration (stereo only)

For 3D reconstruction, place camera calibration YAML files in `calibration/` and set the calibration directory in Settings. Default calibration files for two camera configurations are included.

---

## Project layout

```
movement_tracker/       Python/FastAPI package
  app.py                Route registration and startup
  config.py             Settings singleton (JSON + env var overrides)
  routers/              API endpoints (subjects, labeling, MANO, results, ...)
  services/             Business logic (DLC, MediaPipe, MANO, jobs, remote SSH, ...)
  static/
    css/main.css        Single stylesheet (dark theme, CSS variables)
    js/                 One JS module per page — no build step, no framework
    *.html              One HTML file per page
calibration/            Camera calibration YAML files
scripts/                Utility scripts
  download_sample.py    Fetch sample data from Zenodo
setup.sh                One-command setup + launch (macOS/Linux)
run.bat                 Launch script (Windows)
requirements.txt
CITATION.cff
```

---

## Dependencies

Core:
- [FastAPI](https://fastapi.tiangolo.com/) + [uvicorn](https://www.uvicorn.org/) — web server
- [DeepLabCut](https://deeplabcut.github.io/DeepLabCut/) — pose estimation
- [MediaPipe](https://mediapipe.dev/) — hand landmark tracking
- [OpenCV](https://opencv.org/) — video processing and face detection
- [NumPy](https://numpy.org/) / [SciPy](https://scipy.org/) — numerical computation

All dependencies are listed in `requirements.txt` and installed automatically by `setup.sh`.

---

## Citation

If you use this software in your research, please cite it:

```bibtex
@software{newbold_movement_tracker,
  author  = {Newbold, David},
  title   = {Movement Tracker},
  url     = {https://github.com/newboldd/movement-tracker},
  license = {CC-BY-NC-4.0}
}
```

A [CITATION.cff](CITATION.cff) file is included; GitHub shows a "Cite this repository" button in the sidebar automatically.

---

## License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — free for academic and non-commercial use.
If you intend to use this in a commercial product, please get in touch.
