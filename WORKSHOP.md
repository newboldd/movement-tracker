# Movement Tracker — Hands-On Workshop

A guided walkthrough for new users. By the end you will have imported a sample video, run MediaPipe hand tracking, labeled keypoints, and explored movement metrics — all on your own laptop.

**Time:** ~45 minutes
**Prerequisites:** A laptop running macOS, Linux, or Windows (WSL). No prior experience with the app is needed.

---

## 0. Install prerequisites

You need **git** and **Python 3.9+**. The setup script will install Python and ffmpeg automatically if they are missing (via Homebrew on macOS or apt/dnf on Linux), but git must be installed beforehand.

### macOS

```bash
# If you don't have Homebrew yet:
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

brew install git
```

### Linux (Debian/Ubuntu)

```bash
sudo apt update && sudo apt install -y git
```

### Windows

Install [Git for Windows](https://git-scm.com/download/win), then open **Git Bash** for the remaining steps. Alternatively, install [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) and follow the Linux instructions.

---

## 1. Clone and launch

```bash
git clone https://github.com/newboldd/movement-tracker
cd movement-tracker
./setup.sh
```

This will:
- Create a Python virtual environment
- Install all dependencies (~2-5 min on first run)
- Download a sample finger-tapping video from Zenodo (~24 MB)
- Start the app at **http://localhost:8080** and open your browser

> **Windows (without WSL):** Double-click `run.bat` instead. You will need to install Python 3.9+ and ffmpeg manually first.

Wait until you see `Started server process` in the terminal, then proceed.

---

## 2. First look — the Dashboard

The browser should open to the **Dashboard** page. It will be empty because no subjects have been synced yet.

### Sync the sample data

1. Click **Settings** in the sidebar
2. Set **Video directory** to `sample_data` (relative path is fine — the app resolves it from the project root)
3. Leave **DLC directory** as the default (`dlc`)
4. Click **Save**
5. Return to the **Dashboard** and click **Sync from disk**

You should see **Con01** appear — this is the sample subject with one finger-tapping trial recorded in stereo.

---

## 3. Run MediaPipe hand tracking

MediaPipe detects 21 hand landmarks per frame and is used to guide auto-crop and provide prelabels in the labeler.

1. On the Dashboard, click the **Con01** row to expand the detail panel
2. Click **Run MediaPipe**
3. A progress bar appears — this processes all frames in the video (~30-60 seconds)
4. When complete, the stage updates to **prelabeled**

> You can also monitor the job on the **Processing** page (sidebar). MediaPipe jobs appear in the CPU lane.

---

## 4. Open the DLC Labeler

1. Click **Label** in the sidebar (or go to `/labeling-select`)
2. Select **Con01** → click **Open**
3. The labeler loads the video with MediaPipe ghost markers overlaid

### Navigate the video

| Action | Control |
|--------|---------|
| Next/previous frame | Arrow keys (← →) |
| Jump 10 frames | Shift + Arrow |
| Play/pause | Space |
| Zoom in/out | Mouse wheel |
| Pan | Click + drag on canvas background |
| Switch camera (OS ↔ OD) | `S` key or click the camera badge |
| Fit to screen | `F` key |

### Place a label

1. Make sure you are in **Label** mode (top of sidebar)
2. The active bodypart is shown in the sidebar — click a bodypart name to select it
3. Click on the video to place the label at that position
4. Press the right arrow to advance to the next frame
5. MediaPipe ghost markers (translucent dots) show where the hand landmarks were detected — use these as a guide

### Keyboard shortcuts for labeling

| Key | Action |
|-----|--------|
| `1`-`9` | Select bodypart by number |
| `D` | Delete selected label on current frame |
| `Z` | Undo last action |

---

## 5. MediaPipe bounding box (new feature)

Sometimes MediaPipe detects the wrong hand or misses entirely because the hand is small in frame. You can define a crop region to improve detection.

1. In the labeler sidebar (Label mode), find the **MediaPipe Region** section
2. Check **Edit bounding box** — a green dashed rectangle appears over the full frame
3. **Resize** the box by dragging corners or edges; **move** it by dragging the interior
4. Adjust the box so it tightly surrounds the hand you want to track
5. Click **Save**
6. Switch cameras with `S` — each camera gets its own independent bounding box
7. Click **Re-run MediaPipe (this trial)** — the job appears in the dashboard/processing queue
8. Watch the progress percentage update; when complete, the ghost markers reload with improved positions

> **Tip:** The bounding box persists as you scrub through frames or play the video, so you can verify it covers the hand across the full trial before re-running.

---

## 6. Auto-detect events

Movement events (open, peak, close, pause) can be detected automatically from the distance trace.

1. Switch to **Events** mode in the sidebar
2. Click **Auto-detect** — the algorithm marks events along the timeline
3. Review and adjust: click an event marker on the timeline to select it, then drag or delete
4. Switch to **Final** mode when you are satisfied

---

## 7. Explore the Results page

1. Click **Results** in the sidebar
2. Select **Con01** and expand the trial
3. View movement metrics: amplitude, duration, frequency, and velocity per tap
4. Use **Export CSV** to download the data

---

## 8. (Optional) MANO 3D hand viewer

If the subject has stereo calibration and MediaPipe data, you can view 3D hand poses:

1. Click **MANO** in the sidebar
2. Select **Con01**
3. The 3D hand model is overlaid on the stereo video with distance traces below

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `setup.sh` fails with "Python not found" | Install Python 3.9+ manually from [python.org](https://www.python.org/downloads/) and re-run |
| `setup.sh` fails with "ffmpeg not found" | Install ffmpeg: `brew install ffmpeg` (macOS) or `sudo apt install ffmpeg` (Linux) |
| Port 8080 already in use | The script kills existing processes on that port. If it persists, run `lsof -ti :8080 \| xargs kill` manually |
| Browser doesn't open | Navigate to http://localhost:8080 manually |
| MediaPipe is slow | This is normal on CPU — a single trial takes 30-60s. GPU is not required. |
| Labels don't appear | Make sure you are in **Label** mode, not View mode |
| Sample video missing | Run `python scripts/download_sample.py` manually |

---

## What's next?

- **Add your own videos:** Create a subfolder in the video directory with the subject name, place trial videos inside, then **Sync from disk**
- **Remote GPU processing:** Configure an SSH server in Settings to offload DLC training and inference
- **Batch processing:** Use the Processing page to queue MediaPipe/blur/train jobs across multiple subjects

---

## Quick reference — project URLs

| Page | URL | Description |
|------|-----|-------------|
| Dashboard | `/` | Subject overview, run pipeline steps |
| Settings | `/settings` | Configure directories, cameras, remote |
| Label Select | `/labeling-select` | Choose a subject to label |
| Labeler | `/labeling?session=N` | Frame-by-frame annotation |
| Processing | `/remote` | Job queue (CPU/GPU lanes) |
| Results | `/results` | Movement metrics and export |
| MANO | `/mano` | 3D hand pose viewer |
