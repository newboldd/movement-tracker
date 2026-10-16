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
├── packages/               # folders of frames, sent out and received back
└── dlc/
    └── {Subject}/
        ├── config.yaml
        ├── labeled-data/round1/     # training images + CollectedData
        ├── labels_v1/               # predictions from the first model
        ├── labels_v2/               # predictions after refinement
        ├── corrections/             # reviewed labels, as DLC CSVs
        └── {Subject}_{Trial}/       # MediaPipe passes, one npz per pass
```

## Running alongside Movement Tracker

The two apps are built to coexist. Install this one in its own folder —
a separate clone or unzip — and it gets its own `.venv`. The Python
package is `dlc_labeler`, not `movement_tracker`, so nothing collides
even if you do put them in one environment.

That leaves two things to decide.

**Which data directory.** By default this app reuses Movement Tracker's:
it reads `MT_DATA_DIR` and `~/.movement_tracker/data_dir` as fallbacks,
so it opens the same subjects, videos and DLC projects with no setup.
That is usually what you want — label here, and the work shows up there.
To keep them apart instead, put a `.env` next to `setup.sh`:

```
DLC_DATA_DIR=~/data/dlc-labeler
```

or set it from **Settings → Data directory**, which writes
`~/.dlc_labeler/data_dir` and takes precedence over Movement Tracker's.

**Whether they run at the same time.** They can, on separate data
directories. On a *shared* data directory, run one at a time. SQLite
handles the concurrent reads, and each app ignores the other's jobs — but
two servers writing the same database while both drain a job queue is
more coordination than either was built for, and `settings.json` is
shared, so changing camera names or bodyparts in one changes them in the
other.

Specifically, when the data directory is shared:

- **Jobs stay separate.** This app only ever picks up the job types it
  owns (`mediapipe`, `export-frames`, `train`, `refine`, `analyze`,
  `install-dlc`). It
  will not claim, run, show or fail a Movement Tracker job — HRnet,
  skeleton fits, deidentify and preproc are invisible to it, and left
  alone.
- **Labels are shared, deliberately.** Both apps read the same
  `frame_labels`, `labeled-data/` and `corrections/`. A frame labeled
  here is a frame labeled there.
- **The database schema is shared.** This app adds one column Movement
  Tracker does not have (`subjects.notes`) and creates no tables of its
  own beyond the ones they both use. Opening the same database with
  either app is safe in both directions.

**The port.** Both default to 8080. Neither launcher will kill the other:
if the port is held by anything that is not this app, it moves up to the
next free one, says so, and opens the browser on the port it actually
started on. Pin a port with `DLC_PORT=8081` in `.env` if you would rather
it be predictable.

If Movement Tracker's interface appears when you launch this, you are
looking at Movement Tracker on port 8080, not at this app — check the
port in the address bar against the `Starting DLC Labeler at …` line the
launcher prints.

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
   trial, then **Save labels** — that extracts the frames as training
   images and writes the DeepLabCut `CollectedData` files.

   For the first training set, use **Guided labeling** instead: see below.
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

### Guided labeling

The first training set is 30 frames x 2 cameras x 2 bodyparts = 120
points, and the slow part is not placing them — it is finding them:
scrolling to the right frame, switching camera, zooming in. **Guided
labeling** (the button above the timeline, or `G`) does that part.

Pressing it the first time picks the frames. That runs DeepLabCut's own
k-means frame selector — MiniBatchKMeans over mean-centred 30px
greyscale frames, one random frame per cluster — but without DeepLabCut
installed, over the whole subject rather than one video at a time, and
on the MediaPipe hand crop from each camera rather than the whole frame.
That last one matters: squeezed to 30px wide, a 1920px camera half
leaves the hand about two pixels, so whole-frame clusters sort by arm
position and lighting. Cropping to the hand first makes them sort by
hand posture, which is what a labeling set should span. Where MediaPipe
has nothing, the whole frame stands in. The result is saved to
`dlc/{Subject}/selection.json`, so it only runs once.

Then `N` takes you to the next point nobody has placed, on the right
frame and camera, zoomed on MediaPipe's guess at that exact point —
close enough that placing it is one click. Place it and it moves on by
itself. The grid under the video is one row per point and one column per
selected frame, so what is left is one glance; click any cell to jump
there.

You can still move around the whole video and label anything else while
the mode is on. What the **Save labels** button writes follows what is
on screen:

| Showing | Button | Writes |
|---|---|---|
| The grid | **Save 30 frames** | only the selected frames, to `labeled-data/selected{N}/` |
| The timeline | **Save all frames** | every labeled frame, to `labeled-data/round{N}/` |

Saving one clears the other for that round, because
`create_training_dataset` merges every `labeled-data` subdirectory it
finds and training on both would weight the shared frames twice. Both
are rebuilt from the labels in one click, so switching is cheap.

### The crop box

Each trial has one crop box per camera. It does two jobs: it frames the
default zoom when a frame loads, and it is the region MediaPipe crops to.
**Edit crop box** on the Label page starts from the landmark extent,
saves to the trial (optionally seeding every trial that has no box of its
own), and **Re-detect this trial** re-runs MediaPipe inside it — adjust,
re-detect, see whether the frames that were failing come back.

### Distances

With a stereo calibration for the subject, the thumb-index aperture under
the video is triangulated and reported in millimetres. Without one the app
works normally and reports the distance in pixels.

Point Settings at a calibration YAML, and map subjects to cameras with a
`camera_assignments.yaml` in the data directory's `calibration/` folder.

## Sending frames out to be labeled

Labeling is the slow part, and it parallelises across people. **Export
frames to label** on the Jobs page picks a spread of frames and writes
them as a self-contained folder — a *package* — that someone else can
label without the videos, without DeepLabCut, and without access to this
data directory.

Frames are picked with DeepLabCut's own k-means frame selector, but
applied to per-frame crops around the MediaPipe hand rather than to whole
frames. That matters more than it sounds: the selector squeezes each
frame to 30 pixels wide before clustering, so on a 1920-pixel camera half
the hand is about two pixels and the clusters end up sorting by arm
position, lighting and background. Cropping to the hand first means the
clusters are hand postures, which is what you want a spread of.

The crop box is a constant size per trial (the 95th percentile of the
hand's extent, times 1.6, so one bad detection cannot inflate it) and
recentres on every frame. Frames MediaPipe lost borrow the nearest good
centre and are flagged in `frames.csv` — those are often the frames most
worth labeling, so dropping them would bias the training set towards the
easy ones. If a face-blurred render exists, the frames are cut from that.

A package looks like this:

```
Con01_label_20260408/
├── package.json        what this is, which subject, how it was selected
├── README.txt          instructions for whoever receives it
├── frames.csv          one row per image, with its crop and source frame
├── frames/
│   └── Con01_R1_OS/
│       └── img0042.png
└── labels/             filled in as they label
```

**To label one**, install this app, drop the folder into the data
directory under `packages/`, and press **Sync from disk** — it appears
as a subject and opens on the Label page like any other. Labels save
into `labels/labels.csv` inside the folder as they go, so sending the
whole folder back is all there is to it.

**To bring one home**, drop the returned folder into `packages/`, press
Sync, and press **Import labels** on its row. Each image's labels are
mapped back through its crop offset onto the original video frame and
merged into the subject's labeling session, ready to review and commit.
It always runs a dry pass first and tells you how many frames will land,
how many images never came back, and which frames are already labeled
differently — those are left alone unless you say otherwise. Importing
the same package twice does nothing the second time.

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
| `G` | guided labeling on / off |
| `N` | (Guided) next unlabeled point |
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
