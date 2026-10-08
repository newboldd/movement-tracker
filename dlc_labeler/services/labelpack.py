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

README_TEMPLATE = """\
DLC Labeler — frames to label
=============================

This folder holds video frames cropped around the hand, selected for
labeling.  Everything needed is here; you do not need the original
videos or DeepLabCut.

Label {n_points} point{plural} on every frame, in this order:
{points}

To label them
-------------
1. Install DLC Labeler:  https://github.com/newboldd/movement-tracker
   Clone it, then run ./setup.sh (macOS, Linux) or run.bat (Windows).
   package.json records the exact version this folder was made with, if
   you ever need to match it.
2. Put this whole folder inside the app's data directory, under
   `packages/`.
3. Open the app, press "Sync from disk" on the Subjects page, and this
   package appears as a subject named `{name}`.
4. Open it on the Label page: click to place each point in the order
   above, drag to adjust, right-click to remove.
   The arrow keys move between frames; the up arrow jumps to the next
   frame you have not labeled yet.
5. Labels save as you go, into `labels/labels.csv` inside this folder.
   When you are finished, send the whole folder back — your labels
   travel with it, and there is nothing to export.

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


def _readme(name: str, bodyparts: list[str]) -> str:
    """The instructions that ship inside a package.

    The bodyparts go in by name and in order: a package labeled with the
    points in the wrong order is worse than one not labeled at all, and
    the person opening it has no other way to know what was wanted.
    """
    return README_TEMPLATE.format(
        name=name,
        n_points=len(bodyparts),
        plural="" if len(bodyparts) == 1 else "s",
        points="\n".join(f"  {i + 1}. {bp}" for i, bp in enumerate(bodyparts)),
    )


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
    (root / "README.txt").write_text(_readme(name, settings.bodyparts))

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


# ── Opening a package as a subject ─────────────────────────────────────
#
# A received package has no videos, so the app has to be able to label a
# directory of images directly.  Rather than inventing a second kind of
# subject, a package becomes an ordinary subject whose trials are backed
# by image files instead of video files: one trial per frames directory,
# single-camera, frames in file order.  Everything downstream — the
# timeline, the label store, commit — then works unchanged.

# Image sizes are read once per directory; a package's images are all
# the same size and never change under us.
_dims_cache: dict[str, tuple[int, int]] = {}

# What a frames trial reports as its frame rate.  A package is a sparse
# set of frames with no timebase, so there is nothing true to report;
# this exists only to keep "frames per second" arithmetic from dividing
# by zero.
NOMINAL_FPS = 30.0

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def package_dir(subject_name: str) -> Path | None:
    """The package directory for this subject name, if there is one."""
    root = get_settings().packages_path / subject_name
    if not root.is_dir():
        return None
    return root if read_manifest(root) else None


def list_packages() -> list[dict]:
    """Every package in the packages directory, newest first.

    Returns ``[{name, dir, manifest}]``.  A folder without a readable
    manifest is skipped rather than guessed at — being strict here is
    what keeps a stray folder of holiday photos from becoming a subject.
    """
    root = get_settings().packages_path
    if not root.is_dir():
        return []
    found = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        manifest = read_manifest(entry)
        if manifest:
            found.append({"name": entry.name, "dir": str(entry),
                          "manifest": manifest})
    found.sort(key=lambda p: p["manifest"].get("created_at") or "", reverse=True)
    return found


def _image_dims(path: Path) -> tuple[int, int]:
    """(width, height) of an image, cached per directory."""
    key = str(path.parent)
    if key not in _dims_cache:
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Could not read image {path}")
        _dims_cache[key] = (int(img.shape[1]), int(img.shape[0]))
    return _dims_cache[key]


def package_trials(subject_name: str) -> list[dict]:
    """Trial dicts for a package, in the shape ``build_trial_map`` returns.

    ``frames.csv`` is the source of order and of each image's crop, so
    labels can be mapped back to the original video later.  When it is
    missing — a package someone rearranged by hand — the directory
    listing stands in, and the crop geometry is simply unknown.
    """
    root = package_dir(subject_name)
    if root is None:
        return []

    rows_by_dir: dict[str, list[dict]] = {}
    for row in read_frames_csv(root):
        image = (row.get("image") or "").replace("\\", "/")
        parts = image.split("/")
        if len(parts) < 2:
            continue
        rows_by_dir.setdefault(parts[-2], []).append(row)

    frames_root = root / FRAMES_DIR
    dir_names = sorted(d.name for d in frames_root.iterdir() if d.is_dir()) \
        if frames_root.is_dir() else sorted(rows_by_dir)

    trials = []
    for dir_name in dir_names:
        directory = frames_root / dir_name
        on_disk = {p.name: p for p in sorted(directory.iterdir())
                   if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES} \
            if directory.is_dir() else {}

        ordered: list[tuple[Path, dict | None]] = []
        seen = set()
        # CSV order first, so a package's frames stay in the order it
        # recorded them even if a filename sorts oddly.
        for row in sorted(rows_by_dir.get(dir_name, []),
                          key=lambda r: r.get("local_frame") or 0):
            fname = (row.get("image") or "").replace("\\", "/").split("/")[-1]
            if fname in on_disk:
                ordered.append((on_disk[fname], row))
                seen.add(fname)
        # Images the CSV does not mention — including ones a collaborator
        # added — come after, so nothing on disk is silently unlabelable.
        for fname, path in on_disk.items():
            if fname not in seen:
                ordered.append((path, None))

        if not ordered:
            continue
        try:
            width, height = _image_dims(ordered[0][0])
        except ValueError as e:
            logger.warning("Skipping %s: %s", directory, e)
            continue

        trials.append({
            "kind": "frames",
            # The directory stands in for the video file: it is what
            # identifies the trial on disk, and Path(...).stem still works.
            "video_path": str(directory),
            "trial_name": dir_name,
            "trial_stem": dir_name,
            "frames": [str(p) for p, _ in ordered],
            "frame_rows": [r for _, r in ordered],
            "frame_count": len(ordered),
            "fps": NOMINAL_FPS,
            "width": width,
            "height": height,
            "frame_offset": 0,
            "cameras": [{"name": "default", "path": str(directory), "idx": 0}],
        })
    return trials


LABELS_DIR = "labels"
LABELS_CSV = "labels.csv"


def package_image_index(subject_name: str) -> dict[int, dict]:
    """Map each global frame number to the package image it draws.

    The global frame number is what the label store keys on, and the
    image path is what a returned package keys on, so this is the join
    between the two — used to write labels out and to read them back.
    """
    from .video import build_trial_map

    index: dict[int, dict] = {}
    for trial in build_trial_map(subject_name, camera_mode="single"):
        if trial.get("kind") != "frames":
            continue
        rows = trial.get("frame_rows") or []
        for local, path in enumerate(trial.get("frames") or []):
            index[trial["start_frame"] + local] = {
                "image": f"{FRAMES_DIR}/{trial['trial_name']}/{Path(path).name}",
                "trial_name": trial["trial_name"],
                "local_frame": local,
                "row": rows[local] if local < len(rows) else None,
            }
    return index


def write_labels_into_package(subject_name: str) -> dict | None:
    """Write this package's labels into the package folder itself.

    The whole point of a package is that it travels as one folder.  If
    the labels only lived in the database, sending the folder back would
    send back the frames and nothing else — so every save is mirrored
    into ``labels/labels.csv``, keyed by image name, and the folder is
    complete at all times without anybody having to remember to export.

    Returns a summary, or None when this subject is not a package.
    """
    root = package_dir(subject_name)
    if root is None:
        return None

    from ..db import get_db_ctx

    with get_db_ctx() as db:
        subj = db.execute("SELECT id FROM subjects WHERE name = ?",
                          (subject_name,)).fetchone()
        if not subj:
            return None
        # Newest write per frame wins: a package is labeled in one pass,
        # but re-opening it makes a second session and both are the same
        # person's work on the same images.
        rows = db.execute(
            """SELECT fl.frame_num, fl.keypoints, fl.updated_at
                 FROM frame_labels fl
                 JOIN label_sessions ls ON fl.session_id = ls.id
                WHERE ls.subject_id = ?
                ORDER BY fl.updated_at""",
            (subj["id"],)).fetchall()

    manifest = read_manifest(root) or {}
    bodyparts = manifest.get("bodyparts") or get_settings().bodyparts
    index = package_image_index(subject_name)

    best: dict[int, dict] = {}
    for row in rows:
        kp = row["keypoints"]
        kp = json.loads(kp) if isinstance(kp, str) else (kp or {})
        entry = index.get(row["frame_num"])
        if entry is None:
            continue
        if not any(kp.get(bp) for bp in bodyparts):
            # An emptied frame is a deliberate "not this one" — drop it
            # from the file rather than writing a row of blanks.
            best.pop(row["frame_num"], None)
            continue
        best[row["frame_num"]] = {
            "entry": entry, "kp": kp, "updated_at": row["updated_at"]}

    out_dir = root / LABELS_DIR
    out_dir.mkdir(exist_ok=True)
    fieldnames = ["image", "labeled_at"]
    for bp in bodyparts:
        fieldnames += [f"{bp}_x", f"{bp}_y"]

    with open(out_dir / LABELS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(best):
            item = best[key]
            out = {"image": item["entry"]["image"],
                   "labeled_at": item["updated_at"]}
            for bp in bodyparts:
                point = item["kp"].get(bp)
                out[f"{bp}_x"] = round(float(point[0]), 2) if point else ""
                out[f"{bp}_y"] = round(float(point[1]), 2) if point else ""
            writer.writerow(out)

    return {"labeled": len(best), "path": str(out_dir / LABELS_CSV)}


# ── Bringing a labeled package home ────────────────────────────────────


def read_package_labels(package_dir: str | Path) -> dict[str, dict]:
    """``labels/labels.csv`` as ``{image: {bodypart: (x, y)}}``.

    Coordinates are in the cropped image's own pixel space — turning
    them back into video coordinates is :func:`import_package_labels`'s
    job, because only the project knows which video they belong to.
    """
    path = Path(package_dir) / LABELS_DIR / LABELS_CSV
    if not path.is_file():
        return {}

    out: dict[str, dict] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            image = (row.get("image") or "").replace("\\", "/")
            if not image:
                continue
            points = {}
            for key, value in row.items():
                if not key.endswith("_x") or value in (None, ""):
                    continue
                bodypart = key[:-2]
                y_value = row.get(f"{bodypart}_y")
                if y_value in (None, ""):
                    continue
                try:
                    points[bodypart] = (float(value), float(y_value))
                except ValueError:
                    continue
            if points:
                out[image] = points
    return out


def import_package_labels(package_name: str, *, target_subject: str | None = None,
                          overwrite: bool = False,
                          dry_run: bool = False) -> dict:
    """Merge a returned package's labels into the subject they came from.

    Each image is mapped back by trial name and local frame rather than
    by the global frame number the package recorded: global numbering
    depends on which trials exist, and a trial added since the export
    would silently shift every label by a whole video.  The crop offset
    in ``frames.csv`` turns the crop's coordinates back into the camera
    half's, which is the space the label store uses.

    Nothing is overwritten unless asked.  A frame already labeled with
    different coordinates is reported as a conflict and left alone —
    somebody has done that frame twice, and which of them is right is
    not a decision to make silently.
    """
    from ..db import get_db_ctx
    from .video import build_trial_map

    root = get_settings().packages_path / package_name
    manifest = read_manifest(root)
    if manifest is None:
        raise ValueError(f"{package_name} is not a labeling package")

    subject_name = target_subject or manifest.get("subject")
    if not subject_name:
        raise ValueError("The package does not say which subject it came from")

    with get_db_ctx() as db:
        subj = db.execute("SELECT id, camera_mode FROM subjects WHERE name = ?",
                          (subject_name,)).fetchone()
    if not subj:
        raise ValueError(f"No subject named {subject_name} on this machine")

    camera_mode = subj.get("camera_mode") or get_settings().default_camera_mode
    trials = build_trial_map(subject_name, camera_mode=camera_mode)
    if not trials or trials[0].get("kind") == "frames":
        raise ValueError(f"{subject_name} has no trial videos to import into")
    trial_by_name = {t["trial_name"]: (i, t) for i, t in enumerate(trials)}

    frames = {r["image"].replace("\\", "/"): r for r in read_frames_csv(root)}
    labels = read_package_labels(root)

    report = {
        "package": package_name,
        "subject": subject_name,
        "labeled_images": len(labels),
        "imported": 0,
        "conflicts": [],
        "unknown_trial": [],
        "out_of_range": [],
        "missing_images": [],
        "unlabeled": 0,
        "unknown_bodyparts": [],
        "dry_run": dry_run,
    }

    # Images the package listed but that are no longer in the folder: a
    # collaborator deleting a frame that showed a face is expected, and
    # worth reporting rather than passing over in silence.
    for image in frames:
        if not (root / image).is_file():
            report["missing_images"].append(image)
    report["unlabeled"] = max(
        len(frames) - len(report["missing_images"]) - len(labels), 0)

    known_bodyparts = set(get_settings().bodyparts)
    pending: list[tuple[int, int, str, dict]] = []

    for image, points in sorted(labels.items()):
        row = frames.get(image)
        if row is None:
            report["unknown_trial"].append(image)
            continue
        entry = trial_by_name.get(row.get("trial_name"))
        if entry is None:
            report["unknown_trial"].append(image)
            continue
        trial_idx, trial = entry
        local_frame = int(row.get("local_frame") or 0)
        if not 0 <= local_frame < trial["frame_count"]:
            report["out_of_range"].append(image)
            continue

        x0 = int(row.get("crop_x0") or 0)
        y0 = int(row.get("crop_y0") or 0)
        keypoints = {}
        for bodypart, (x, y) in points.items():
            if bodypart not in known_bodyparts:
                if bodypart not in report["unknown_bodyparts"]:
                    report["unknown_bodyparts"].append(bodypart)
            keypoints[bodypart] = [round(x0 + x, 2), round(y0 + y, 2)]

        pending.append((trial["start_frame"] + local_frame, trial_idx,
                        row.get("camera") or "", keypoints))

    if not pending:
        return report

    with get_db_ctx() as db:
        session = db.execute(
            """SELECT ls.id, COUNT(fl.id) AS label_count
                 FROM label_sessions ls
                 LEFT JOIN frame_labels fl ON fl.session_id = ls.id
                WHERE ls.subject_id = ? AND ls.session_type = 'initial'
                  AND ls.status = 'active'
                GROUP BY ls.id
                ORDER BY label_count DESC, ls.id DESC LIMIT 1""",
            (subj["id"],)).fetchone()
        session_id = session["id"] if session else None

        if session_id is None:
            if dry_run:
                report["session_id"] = None
            else:
                iteration = db.execute(
                    "SELECT iteration FROM subjects WHERE id = ?",
                    (subj["id"],)).fetchone()["iteration"]
                session_id = db.execute(
                    """INSERT INTO label_sessions
                           (subject_id, iteration, session_type)
                       VALUES (?, ?, 'initial')""",
                    (subj["id"], iteration)).lastrowid

        for frame_num, trial_idx, side, keypoints in pending:
            existing = None
            if session_id is not None:
                existing = db.execute(
                    "SELECT keypoints FROM frame_labels "
                    "WHERE session_id = ? AND frame_num = ? AND side = ?",
                    (session_id, frame_num, side)).fetchone()
            if existing:
                old = existing["keypoints"]
                old = json.loads(old) if isinstance(old, str) else (old or {})
                if old == keypoints:
                    continue  # Already imported; importing twice is a no-op.
                if not overwrite:
                    report["conflicts"].append(
                        {"frame": frame_num, "camera": side,
                         "existing": old, "incoming": keypoints})
                    continue

            report["imported"] += 1
            if dry_run:
                continue
            db.execute(
                """INSERT INTO frame_labels
                       (session_id, frame_num, trial_idx, side, keypoints,
                        updated_at)
                   VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                   ON CONFLICT(session_id, frame_num, trial_idx, side)
                   DO UPDATE SET keypoints = excluded.keypoints,
                                 updated_at = CURRENT_TIMESTAMP""",
                (session_id, frame_num, trial_idx, side,
                 json.dumps(keypoints)))

    report["session_id"] = session_id
    logger.info("Imported %d labels from %s into %s (%d conflicts)",
                report["imported"], package_name, subject_name,
                len(report["conflicts"]))
    return report
