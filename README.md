# DLC Labeler

A web app for labeling stereo video for DeepLabCut.

**It labels frames and nothing else.** It does not run MediaPipe and it
does not run DeepLabCut. It reads the MediaPipe landmarks
[Movement Tracker](https://github.com/newboldd/movement-tracker) already
wrote, uses them to put every point you need to place under your cursor,
and writes a DeepLabCut training set. Training, inference, correcting
predictions and refining happen elsewhere, against the project it
produces.

That narrowness is the point: the install is small, there is one page to
learn, and the slow part of making a training set — finding the frames
and the points — is what the app is for.

## Install

**macOS / Linux**

```bash
./setup.sh
```

**Windows**

```bat
run.bat
```

Either one installs what it needs on first run and opens
<http://localhost:8080>. On macOS you can also double-click
**DLC Labeler.command** or **DLC Labeler.app**.

### Where the data goes

Videos, DLC projects, the database and labeling packages live in a data
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
├── dlc_app.db              # subjects, sessions, labels
├── settings.json
├── videos/                 # {Subject}_{Trial}.mp4
│   └── deidentified/       # face-blurred copies, used when present
├── calibration/            # stereo calibration YAML + camera_assignments
├── packages/               # folders of frames, sent out and received back
└── dlc/
    └── {Subject}/
        ├── config.yaml
        ├── selection.json          # the frames k-means chose
        ├── labeled-data/round1/    # training images + CollectedData
        └── {Subject}_{Trial}/      # MediaPipe landmarks, read not written
```

## Running alongside Movement Tracker

The two apps are built to coexist, and this one depends on the other
having run: MediaPipe landmarks come from Movement Tracker.

Install this one in its own folder — a separate clone or unzip — and it
gets its own `.venv`. The Python package is `dlc_labeler`, not
`movement_tracker`, so nothing collides.

**Which data directory.** By default this app reuses Movement Tracker's:
it reads `MT_DATA_DIR` and `~/.movement_tracker/data_dir` as fallbacks,
so it opens the same subjects, videos and MediaPipe output with no
setup. That is usually what you want. To keep them apart, put a `.env`
next to `setup.sh`, or set it from **Settings → Data directory**.

**Whether they run at the same time.** They can. This app has no job
queue and runs nothing in the background except its own frame selection,
so it will never pick up or interfere with Movement Tracker's work. It
shares `frame_labels`, `labeled-data/` and `settings.json`, so a frame
labeled here is a frame labeled there, and changing camera names or
bodyparts in one changes them in the other.

**The port.** Both default to 8080. Neither launcher will kill the
other: if the port is held by anything that is not this app, it moves up
to the next free one, says so, and opens the browser on the port it
actually started on. Pin a port with `DLC_PORT=8081` in `.env`.

## Getting a subject onto the screen

Name the trial videos `{Subject}_{Trial}.mp4` — `Con01_R1.mp4`,
`MSA03_L2.mp4` — put them in the video directory, and press **Sync from
disk** on the Subjects page. Nothing else registers a subject; the
filename is the record.

Stereo recordings are one side-by-side file per trial, split at the
midline into the two cameras (OS first, OD second, configurable in
Settings). One file per camera (`multicam`) and single-camera recordings
also work; set the layout per subject.

## Labeling

Open a subject on the Label page. The first time, it picks the frames.

**Choosing the frames.** This runs DeepLabCut's own k-means frame
selector — MiniBatchKMeans over mean-centred 30px greyscale frames, one
random frame per cluster — but without DeepLabCut installed, over the
whole subject rather than one video at a time, and on the MediaPipe hand
crop from each camera rather than the whole frame. That last one
matters: squeezed to 30px wide, a 1920px camera half leaves the hand
about two pixels, so whole-frame clusters sort by arm position and
lighting. Cropping to the hand first makes them sort by hand posture,
which is what a labeling set should span. Where MediaPipe has nothing,
the whole frame stands in. The result is saved to
`dlc/{Subject}/selection.json`, so it only runs once.

**Placing the points.** `N` takes you to the next point nobody has
placed, on the right frame and camera, zoomed on MediaPipe's guess at
that exact point — close enough that placing it is one click. Points go
thumb then index, each on both cameras in turn, so the pairing you can
check by eye comes back to back.

Placing a point does not move you on, so you can look at what you just
placed. Tick **advance automatically** if you would rather it did; the
choice is remembered.

Only the point being asked for can be placed. Every labeled frame is one
of the k-means picks, so a label dropped on a neighbouring frame would
never be saved but would still look placed — refusing it is clearer.
Stepping to neighbouring frames to see how the point moves is still
expected: the view gets an orange border and a **Back to target** button
(`B`) whenever you are not on the frame being asked about, and going
back keeps your zoom and pan so the comparison survives the trip.

The grid under the video is one row per point and one column per
selected frame, so what is left is one glance; click any cell to jump
there. The thumb-index trace stays visible, because whether a frame is
worth labeling often shows up there first. **MediaPipe landmarks** in
the sidebar draws the hand over the frame; the zoom uses the landmarks
either way.

**Save labels** writes every labeled frame into
`labeled-data/round{N}/` with a CollectedData CSV and a `config.yaml`,
which is a valid DeepLabCut project. Point DeepLabCut at it to train.

### Distances

With a stereo calibration for the subject, the thumb-index aperture
under the video is triangulated and reported in millimetres. Without one
the app reports the distance in pixels. The camera you have not labeled
yet falls back to MediaPipe, so the number means something before a
frame is finished.

Point Settings at a calibration YAML, and map subjects to cameras with a
`camera_assignments.yaml` in the data directory's `calibration/` folder.

## Sending frames out to be labeled

Labeling parallelises across people. **Export for labeling** writes the
frames this subject already selected as a self-contained folder — a
*package* — that someone else can label without the videos and without
MediaPipe.

It exports the same frames the page would have shown you, read back from
`selection.json`, so what a collaborator labels is exactly the set you
would have labeled and their labels drop straight back onto the same
frames. If a face-blurred render exists, the frames are cut from that.

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
merged into the subject's labeling session. It always runs a dry pass
first and tells you how many frames will land, how many images never
came back, and which frames are already labeled differently — those are
left alone unless you say otherwise. Importing the same package twice
does nothing the second time.

## Updating

The **Update** button in the header pulls the latest version and
restarts the app if the code changed. It only appears on a git checkout
with a remote — if you installed by downloading a zip, download it again
to update.

It fast-forwards and nothing else: it never merges, never rewrites
history, and never discards local work. Uncommitted changes to the app's
own files stop it, with the files named. A copy with commits the remote
does not have is reported rather than merged.

If the update only touches pages or styles the browser just reloads. If
it touches Python, the server restarts.

## Keyboard

| | |
|---|---|
| `N` | next unlabeled point |
| `B` | back to the target frame |
| `←` `→` | previous / next frame (`Shift` for 10) |
| `Space` | play / pause |
| `E` | switch camera |
| `R` | reset zoom |
| `1`–`9` | jump to a trial |
| `Ctrl/⌘ Z` | undo — jumps to the frame it changed |

## Requirements

Python 3.9 or newer. Nothing here caps the upper end: this app does not
import `mediapipe`, which is what used to hold it at 3.12.

No GPU, and no DeepLabCut.

## License

BUSL-1.1 — see [LICENSE](LICENSE).
