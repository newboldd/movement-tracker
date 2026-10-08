# DLC Labeler

A web app for labeling stereo video for DeepLabCut, and for running the
standard DeepLabCut loop over it: label → train → analyze → correct →
refine. One page to do the labeling on, one page to run the jobs from.

It is a stripped-down fork of
[Movement Tracker](https://github.com/newboldd/movement-tracker), keeping
the parts that serve DeepLabCut work and dropping the rest. No automatic
editing, no model fitting, no event detection — the goal is field-standard
DeepLabCut analysis, made fast, precise and reproducible.

## Install

**macOS / Linux**

```bash
./setup.sh                 # labeling, MediaPipe, viewing predictions
./setup.sh --with-dlc      # also install DeepLabCut, for training
```

**Windows**

```bat
run.bat
run.bat --with-dlc
```

Either one installs what it needs on first run and opens
<http://localhost:8080>. On macOS you can also double-click
**DLC Labeler.command** or **DLC Labeler.app**.

The install is two-tier on purpose. The base tier is small and fast, and
covers labeling, MediaPipe and reviewing predictions. DeepLabCut and
torch are several gigabytes and only matter on a machine that will train,
so they are opt-in — with `--with-dlc`, or the **Install DeepLabCut**
button on the Jobs page, whenever a machine first needs them.

### Where the data goes

Videos, DLC projects, the database and the job history live in a data
directory outside the code, so updating the app never touches the data.
It defaults to `data/` next to the app; point it somewhere else with a
`.env` file:

```
DLC_DATA_DIR=~/data/dlc-labeler
```

or from **Settings → Data directory**. An existing `MT_DATA_DIR` is read
too, so a machine already set up for Movement Tracker finds the same
data with no reconfiguration — the two apps share the layout and the
`dlc_app.db` schema deliberately.

```
~/data/dlc-labeler/
├── dlc_app.db              # subjects, sessions, labels, job queue
├── settings.json
├── job_history.jsonl       # every job ever run, with the app version
├── videos/                 # {Subject}_{Trial}.mp4
│   └── deidentified/       # face-blurred copies, used when present
├── calibration/            # stereo calibration YAML + camera_assignments
└── dlc/
    └── {Subject}/
        ├── config.yaml
        ├── labeled-data/round1/     # training images + CollectedData
        ├── labels_v1/               # predictions from the first model
        ├── labels_v2/               # predictions after refinement
        ├── corrections/             # reviewed labels, as DLC CSVs
        └── {Subject}_{Trial}/       # MediaPipe passes, one npz per pass
```

## Getting a subject onto the screen

Name the trial videos `{Subject}_{Trial}.mp4` — `Con01_R1.mp4`,
`MSA03_L2.mp4` — put them in the video directory, and press **Sync from
disk** on the Subjects page. Nothing else registers a subject; the
filename is the record.

Stereo recordings are one side-by-side file per trial, split at the
midline into the two cameras (OS first, OD second, configurable in
Settings). One file per camera (`multicam`) and single-camera recordings
also work; set the layout per subject.

## The loop

1. **MediaPipe** on the Jobs page. It finds the hand in all 21 landmarks
   per camera. You do not have to run it, but it is what frames the
   default zoom while you label, so labeling without it is slower.
2. **Label** on the Label page. Click to place each bodypart, drag to
   adjust, right-click to remove. Label a spread of frames across every
   trial, then **Save & commit** — that extracts the frames as training
   images and writes the DeepLabCut `CollectedData` files.
3. **Train** on the Jobs page. It creates the training dataset, trains,
   crops each trial into per-camera video, and analyzes it all into
   `labels_v1/`.
4. **Correct** on the Label page. Predictions appear as ghosts; click one
   to take it over and drag it right. Saving writes `corrections/`.
5. **Refine**. The page lists the frames whose correction actually moved
   the model's prediction, largest first — those are the frames worth
   retraining on. Tick them, commit, and run **Refine** from Jobs.

Repeat 4–5 until the predictions are good enough. `labels_v2/` holds the
refined model's output.

### MediaPipe passes

MediaPipe can be run three ways, and each wins on different frames:

- **Forward** — the normal pass, with the between-frame tracker.
- **Backwards** — frames fed in descending order, so the tracker enters a
  hard frame already locked on instead of cold.
- **Frame-by-frame** — the full palm detector on every frame, no tracker.
  Slower, but recovers poses the tracker loses entirely.

They are stored side by side rather than overwriting each other. As soon
as two exist for a trial, a **Best per frame** layer is built: for each
frame and each camera it picks the pass whose pairing triangulates to the
most plausible thumb-index aperture, with a stereo Y-disparity check to
reject pairings where the tips line up but the rest of the hand does not.

### The crop box

Each trial has one crop box per camera. It does two jobs: it frames the
default zoom when a frame loads, and it is the region MediaPipe crops to.
**Edit crop box** on the Label page starts from the landmark extent,
saves to the trial (optionally seeding every trial that has no box of its
own), and **Re-detect this trial** re-runs MediaPipe inside it — adjust,
re-detect, see whether the frames that were failing come back.

### 3D

With a stereo calibration for the subject, every layer is triangulated:
the MediaPipe hands as full skeletons, predictions and corrections as
their labeled points, and this session's manual labels live as you drag
them. Distances are reported in millimetres. Without a calibration the
app works normally in 2D and reports distances in pixels.

Point Settings at a calibration YAML, and map subjects to cameras with a
`camera_assignments.yaml` in the data directory's `calibration/` folder.

## Jobs

Two lanes run at once — CPU for MediaPipe, GPU for DeepLabCut — and jobs
inside a lane run one at a time. Every job runs as a subprocess with its
PID recorded, so closing the browser, or the app restarting, does not
kill a training run in progress; it is picked back up on the next start.

Every run is also appended to `job_history.jsonl` with the app's git
version and its stage timings. The jobs table is a working view and gets
pruned; that file is the record of what was actually run, and it survives
the database being deleted.

## Keyboard

| | |
|---|---|
| `←` `→` | previous / next frame (`Shift` for 10) |
| `↑` `↓` | previous / next labeled frame |
| `Space` | play / pause |
| `E` | switch camera |
| `D` | show / hide 3D |
| `R` | reset zoom |
| `1`–`9` | jump to a trial |
| `T` | (Refine) tick this frame for training |
| `Ctrl/⌘ Z` | undo — jumps to the frame it changed |

## Requirements

Python 3.9–3.12. MediaPipe publishes no wheels for 3.13+, and the version
pinned here (`<0.10.19`, the last with the `mediapipe.solutions` API this
uses) stops at 3.12. Everything except hand detection works on newer
Pythons.

A CUDA GPU makes training practical but is not required; without one DLC
runs on the CPU.

## License

BUSL-1.1 — see [LICENSE](LICENSE).
