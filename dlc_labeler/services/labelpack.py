"""Labeling packages: a folder of frames to hand to someone else.

A package is self-contained on purpose.  It holds the images, the
geometry needed to put the resulting labels back where they came from,
and enough metadata that it can be opened months later without the
project that produced it.  Someone who receives one needs this app and
nothing else — no videos, no DeepLabCut, no access to the original data
directory.

    <name>/
      package.json                     what this is, and how it was made
      README.txt                       what to do with it
      frames.csv                       one row per image, with its crop
      frames/
        <Subject>_<Trial>_<Camera>/    one directory per trial and camera
          img0042.png                  named by frame index, DLC-style

The directory layout and image naming follow DeepLabCut's
``labeled-data/<video stem>/img<frame>.png`` so a package can also be
dropped into a DLC project by hand if that is ever easier.
"""
from __future__ import annotations

import csv
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

import cv2

from ..config import get_settings
from .frameselect import (
    CROP_PADDING, CROP_SIDE_PERCENTILE, DLC_RESIZE_WIDTH, FrameChoice,
    select_for_subject,
)
from .job_history import git_version
from .video import (
    _deidentified_path, _get_no_face_videos, build_trial_map,
)

logger = logging.getLogger(__name__)

PACKAGE_SCHEMA = "dlc-labeler-package/1"
MANIFEST_NAME = "package.json"
FRAMES_CSV = "frames.csv"
FRAMES_DIR = "frames"

README = """\
DLC Labeler — frames to label
=============================

This folder holds video frames cropped around the hand, selected for
labeling.  Everything needed is here; you do not need the original
videos or DeepLabCut.

To label them
-------------
1. Install DLC Labeler:  https://github.com/newboldd/movement-tracker
   (clone the branch named in package.json, then run ./setup.sh, or
   run.bat on Windows)
2. Put this whole folder inside the app's data directory, under
   `packages/`.
3. Open the app, press "Sync from disk" on the Subjects page, and this
   package appears as a subject.
4. Open it on the Label page and label every frame: click to place each
   point in order, drag to adjust, right-click to remove.
   The arrow keys move between frames; the up arrow jumps to the next
   frame you have not labeled yet.
5. Labels save as you go. When you are finished, send the whole folder
   back — your labels travel with it, in `labels/`.

If a frame should not be labeled
--------------------------------
Leave it empty and move on.  Frames are cropped individually, so
skipping one costs only that frame.  If a frame shows a face or anything
else identifying, delete the image file as well as leaving it unlabeled,
and say so when you send the folder back.

What the numbers in the filenames mean
--------------------------------------
`img0042.png` is frame 42 of that trial's video.  `frames.csv` records
where in the original video each image was cropped from, which is how
your labels get mapped back.  Do not rename the files.
"""


def _source_video(subject_name: str, trial: dict) -> str:
    """The file to cut frames from, preferring a face-blurred render.

    Exporting the blurred version when one exists is the cheapest
    protection there is: the frames leave the building, so they should
    leave with the faces already gone.  Blurred renders are frame-aligned
    with their originals, so the MediaPipe coordinates still apply.
    """
    path = trial["video_path"]
    settings = get_settings()
    if not settings.prefer_deidentified:
        return path
    if Path(path).stem in _get_no_face_videos(subject_name):
        return path
    return _deidentified_path(path) or path


def _frame_dir_name(subject_name: str, trial_name: str, camera: str) -> str:
    # trial_name already carries the subject prefix (Con01_R1), so the
    # camera is all that needs adding to make it unique.
    return f"{trial_name}_{camera}"


def _index_width(frame_count: int) -> int:
    """DeepLabCut's zero-padding width: ceil(log10(frames in the video))."""
    return max(int(math.ceil(math.log10(max(frame_count, 2)))), 1)


def _write_images(subject_name: str, trial: dict, choices: list[FrameChoice],
                  camera_index_of: dict, camera_mode: str,
                  frames_root: Path) -> list[dict]:
    """Cut and write the chosen frames of one trial. Returns CSV rows."""
    from .frameselect import _camera_half

    video_path = _source_video(subject_name, trial)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("Could not open %s for export", video_path)
        return []

    width = _index_width(int(trial["frame_count"]))
    rows = []

    # Only a handful of frames per trial are wanted, so seek to each rather
    # than decoding the whole file a second time.
    for choice in sorted(choices, key=lambda c: c.local_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, choice.local_frame)
        ok, frame = cap.read()
        if not ok or frame is None:
            logger.warning("Frame %s missing from %s",
                           choice.local_frame, video_path)
            continue

        half = _camera_half(frame, camera_index_of[choice.camera], camera_mode)
        patch = half[choice.y0:choice.y0 + choice.side,
                     choice.x0:choice.x0 + choice.side]
        if patch.size == 0:
            logger.warning("Empty crop for frame %s", choice.local_frame)
            continue

        dir_name = _frame_dir_name(subject_name, choice.trial_name,
                                   choice.camera)
        out_dir = frames_root / dir_name
        out_dir.mkdir(parents=True, exist_ok=True)
        img_name = f"img{str(choice.local_frame).zfill(width)}.png"
        cv2.imwrite(str(out_dir / img_name), patch)

        rows.append({
            "image": f"{FRAMES_DIR}/{dir_name}/{img_name}",
            "subject": subject_name,
            "trial_name": choice.trial_name,
            "trial_idx": choice.trial_idx,
            "camera": choice.camera,
            "global_frame": choice.global_frame,
            "local_frame": choice.local_frame,
            "crop_x0": choice.x0,
            "crop_y0": choice.y0,
            "crop_side": choice.side,
            "mp_available": int(choice.mp_available),
            "source_video": Path(video_path).name,
        })

    cap.release()
    return rows


def export_package(subject_name: str, dest_dir: str | Path, *,
                   n_per_trial: int = 20,
                   cameras: list[str] | None = None,
                   step: int = 1,
                   seed: int | None = None,
                   package_name: str | None = None,
                   progress=None) -> dict:
    """Select frames for a subject and write them as a labeling package.

    Returns a summary dict: where it went, how many images, and any
    trials that produced nothing and why.
    """
    settings = get_settings()
    cam_names = cameras or settings.camera_names
    camera_index_of = {name: i for i, name in enumerate(settings.camera_names)}
    for name in cam_names:
        if name not in camera_index_of:
            raise ValueError(f"Unknown camera: {name}")

    from ..db import get_db_ctx
    with get_db_ctx() as db:
        subj = db.execute("SELECT camera_mode FROM subjects WHERE name = ?",
                          (subject_name,)).fetchone()
    camera_mode = (subj or {}).get("camera_mode") or settings.default_camera_mode

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    name = package_name or f"{subject_name}_label_{stamp}"
    root = Path(dest_dir).expanduser() / name
    frames_root = root / FRAMES_DIR
    frames_root.mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(exist_ok=True)

    # Selection decodes each trial once; writing seeks to the chosen
    # frames.  Report the first 80% of progress against selection, which
    # is where nearly all the time goes.
    def select_progress(pct):
        if progress:
            progress(min(pct * 0.8, 80.0))

    choices, notes = select_for_subject(
        subject_name, n_per_trial, cameras=cam_names,
        camera_mode=camera_mode, step=step, seed=seed,
        progress=select_progress)

    if not choices:
        raise RuntimeError(
            "No frames could be selected. "
            + (notes[0] if notes else "No MediaPipe output for this subject?"))

    trials = build_trial_map(subject_name, camera_mode=camera_mode)
    by_trial: dict[int, list[FrameChoice]] = {}
    for choice in choices:
        by_trial.setdefault(choice.trial_idx, []).append(choice)

    rows: list[dict] = []
    for i, (trial_idx, trial_choices) in enumerate(sorted(by_trial.items())):
        if trial_idx >= len(trials):
            continue
        rows.extend(_write_images(subject_name, trials[trial_idx],
                                  trial_choices, camera_index_of, camera_mode,
                                  frames_root))
        if progress:
            progress(80.0 + 18.0 * (i + 1) / max(len(by_trial), 1))

    if not rows:
        raise RuntimeError("Frames were selected but none could be written — "
                           "check that the trial videos are readable.")

    fieldnames = list(rows[0].keys())
    with open(root / FRAMES_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "schema": PACKAGE_SCHEMA,
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "app_version": git_version(),
        "subject": subject_name,
        "camera_mode": camera_mode,
        "cameras": cam_names,
        "bodyparts": settings.bodyparts,
        "image_count": len(rows),
        "directories": sorted({r["image"].split("/")[1] for r in rows}),
        "selection": {
            # Recorded in full so a selection can be explained, repeated or
            # argued with later.
            "method": "deeplabcut-kmeans-on-mediapipe-crop",
            "description": (
                "DeepLabCut's KmeansbasedFrameselectioncv2 (MiniBatchKMeans, "
                "tol=1e-3, batch_size=100, max_iter=50, mean-centred "
                "greyscale, one random frame per cluster), applied to "
                "per-frame crops around the MediaPipe hand rather than to "
                "whole frames."
            ),
            "frames_per_trial_per_camera": n_per_trial,
            "step": step,
            "seed": seed,
            "resize_width": DLC_RESIZE_WIDTH,
            "crop_padding": CROP_PADDING,
            "crop_side_percentile": CROP_SIDE_PERCENTILE,
        },
        "notes": notes,
    }
    (root / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    (root / "README.txt").write_text(README)

    if progress:
        progress(100.0)

    logger.info("Wrote %d images to %s", len(rows), root)
    return {
        "package_dir": str(root),
        "image_count": len(rows),
        "directories": manifest["directories"],
        "notes": notes,
    }


def read_manifest(package_dir: str | Path) -> dict | None:
    """Load a package manifest, or None when the folder is not a package."""
    path = Path(package_dir) / MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        manifest = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        logger.warning("Could not read %s: %s", path, e)
        return None
    if not str(manifest.get("schema", "")).startswith("dlc-labeler-package/"):
        return None
    return manifest


def read_frames_csv(package_dir: str | Path) -> list[dict]:
    """The per-image rows of a package, with numbers already parsed."""
    path = Path(package_dir) / FRAMES_CSV
    if not path.is_file():
        return []
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for key in ("trial_idx", "global_frame", "local_frame",
                        "crop_x0", "crop_y0", "crop_side", "mp_available"):
                if row.get(key) not in (None, ""):
                    row[key] = int(row[key])
            out.append(row)
    return out
