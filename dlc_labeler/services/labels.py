"""CollectedData CSV read/write and H5 conversion for DLC."""
from __future__ import annotations

import csv
import json
import logging
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from ..config import get_settings
from .video import extract_frame_raw, build_trial_map, _resolve_frame

logger = logging.getLogger(__name__)


def write_collected_data_csv(
    labels: list[dict],
    output_dir: Path,
    training_name: str = "round1",
):
    """Write CollectedData_labels.csv in DLC multi-header format.

    Args:
        labels: list of dicts with keys: keypoints (dict), img_filename
        output_dir: directory to write CSV into
        training_name: name for the labeled-data subdirectory
    """
    settings = get_settings()
    bodyparts = settings.bodyparts
    scorer = settings.dlc_scorer

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "CollectedData_labels.csv"

    # DLC multi-header format:
    # Row 0: scorer,,,[scorer],[scorer],[scorer],[scorer]
    # Row 1: bodyparts,,,[bp1],[bp1],[bp2],[bp2]
    # Row 2: coords,,,x,y,x,y
    # Data rows: labeled-data,[training_name],[imgfile],x1,y1,x2,y2

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        n_bp = len(bodyparts)
        scorer_row = ["scorer", "", ""] + [scorer] * (n_bp * 2)
        bp_row = ["bodyparts", "", ""]
        for bp in bodyparts:
            bp_row.extend([bp, bp])
        coords_row = ["coords", "", ""] + ["x", "y"] * n_bp

        writer.writerow(scorer_row)
        writer.writerow(bp_row)
        writer.writerow(coords_row)

        # Data rows
        for label in sorted(labels, key=lambda x: x["img_filename"]):
            kp = label.get("keypoints", {})
            row = ["labeled-data", training_name, label["img_filename"]]
            for bp in bodyparts:
                coords = kp.get(bp)
                if coords and len(coords) >= 2 and coords[0] is not None:
                    row.extend([coords[0], coords[1]])
                else:
                    row.extend(["", ""])
            writer.writerow(row)

    return csv_path


def convert_csv_to_h5(config_path: str):
    """Call deeplabcut.convertcsv2h5 via subprocess."""
    settings = get_settings()
    python_exe = settings.python_executable

    script = (
        f"import deeplabcut; "
        f"deeplabcut.convertcsv2h5(r'{config_path}', userfeedback=False)"
    )
    result = subprocess.run(
        [python_exe, "-c", script],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        # Just the last line: the common failure is "No module named
        # 'deeplabcut'", and a full traceback in the app log for an
        # expected, non-fatal condition only buries real problems.
        tail = (result.stderr or "").strip().splitlines()
        raise RuntimeError(f"convertcsv2h5 failed: "
                           f"{tail[-1] if tail else 'no output'}")
    return result.stdout


def backfill_label_metadata(subject_name: str):
    """Backfill video_file and local_frame into existing label_metadata.json.

    Old metadata only has global frame_num. This adds video_file (basename)
    and local_frame so the remote training script can extract PNGs from the
    correct source video.
    """
    settings = get_settings()
    dlc_path = settings.dlc_path / subject_name
    labeled_data_root = dlc_path / "labeled-data"
    if not labeled_data_root.is_dir():
        return

    trials = build_trial_map(subject_name)
    if not trials:
        return

    for subdir in labeled_data_root.iterdir():
        meta_path = subdir / "label_metadata.json"
        if not meta_path.is_file():
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        changed = False
        for img_name, info in meta.items():
            if "video_file" in info and "local_frame" in info:
                continue  # Already has new fields
            try:
                video_path, local_frame, _trial = _resolve_frame(trials, info["frame_num"])
                info["video_file"] = Path(video_path).name
                info["local_frame"] = local_frame
                changed = True
            except (ValueError, KeyError):
                pass

        if changed:
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)


# Distinct colors per bodypart for QC overlay images
_BP_COLORS = [
    (0, 0, 255),    # red (BGR)
    (0, 255, 0),    # green
    (255, 0, 0),    # blue
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
    (255, 255, 0),  # cyan
]


def _draw_label_overlay(frame: np.ndarray, keypoints: dict, bodyparts: list[str]) -> np.ndarray:
    """Draw scatter crosses on a frame copy for each labeled keypoint.

    Returns a new image with crosses and bodypart labels drawn on it.
    """
    img = frame.copy()
    arm = 12  # cross arm length in pixels
    thickness = 2

    for i, bp in enumerate(bodyparts):
        coords = keypoints.get(bp)
        if not coords or len(coords) < 2 or coords[0] is None:
            continue
        x, y = int(round(coords[0])), int(round(coords[1]))
        color = _BP_COLORS[i % len(_BP_COLORS)]
        # Draw cross
        cv2.line(img, (x - arm, y - arm), (x + arm, y + arm), color, thickness)
        cv2.line(img, (x - arm, y + arm), (x + arm, y - arm), color, thickness)
        # Label
        cv2.putText(img, bp, (x + arm + 4, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img


def commit_labels_to_dlc(
    subject_name: str,
    session_labels: list[dict],
    iteration: int = 1,
    camera_mode: str | None = None,
    training_name: str | None = None,
    supersedes: list[str] | None = None,
):
    """Commit labeled frames: extract PNGs, write CSV/H5, create DLC structure.

    Args:
        subject_name: Subject identifier
        session_labels: list of dicts from frame_labels table (with keypoints JSON)
        iteration: labeling iteration (1 = initial, 2+ = refinement)
        camera_mode: Per-subject camera mode (falls back to global default)
        training_name: labeled-data subdirectory; defaults to round{iteration}
        supersedes: sibling directories to clear out, because DeepLabCut's
            create_training_dataset merges every labeled-data subdirectory
            that has a CollectedData CSV.  Two sets covering the same
            frames would otherwise both be trained on, silently
            double-weighting whatever they share.

    Returns:
        dict with paths and counts
    """
    settings = get_settings()
    dlc_path = settings.dlc_path / subject_name
    training_name = training_name or f"round{iteration}"
    labeled_data_dir = dlc_path / "labeled-data" / training_name

    for sibling in (supersedes or []):
        sib_dir = dlc_path / "labeled-data" / sibling
        if sib_dir.is_dir() and sib_dir != labeled_data_dir:
            shutil.rmtree(sib_dir)
            logger.info("Removed superseded training set %s", sib_dir)

    labeled_data_dir.mkdir(parents=True, exist_ok=True)
    qc_dir = labeled_data_dir / "qc"
    qc_dir.mkdir(exist_ok=True)

    # Clean up old PNGs from previous commits
    for old_png in labeled_data_dir.glob("img*.png"):
        old_png.unlink()
    for old_png in qc_dir.glob("img*.png"):
        old_png.unlink()

    # Build trial map for frame resolution
    trials = build_trial_map(subject_name, camera_mode=camera_mode)
    bodyparts = settings.bodyparts

    # Group labels by (frame_num, trial_idx, side) to handle both camera views
    # For DLC, each labeled image is one camera half at one frame
    extracted = []
    metadata = {}  # img_filename -> {frame_num, trial_idx, side}
    img_idx = 0

    for label in sorted(session_labels, key=lambda x: (x["trial_idx"], x["frame_num"], x["side"])):
        # Parse keypoints from JSON string if needed
        kp = label.get("keypoints", {})
        if isinstance(kp, str):
            kp = json.loads(kp)

        # Skip labels without any coordinates
        has_any = any(
            coords and len(coords) >= 2 and coords[0] is not None
            for coords in kp.values()
        )
        if not has_any:
            continue

        # Resolve to video path and local frame
        video_path, local_frame, trial = _resolve_frame(trials, label["frame_num"])

        # For multicam subjects, resolve to the camera-specific video file
        if camera_mode == "multicam" and label.get("side"):
            for cam in trial.get("cameras", []):
                if cam["name"] == label["side"]:
                    video_path = cam["path"]
                    break

        # Extract frame as PNG
        img_filename = f"img{img_idx:04d}.png"
        frame_array = extract_frame_raw(video_path, local_frame, label["side"],
                                        camera_mode=camera_mode, trial=trial)
        png_path = labeled_data_dir / img_filename
        cv2.imwrite(str(png_path), frame_array)

        # Save QC overlay with label crosses
        qc_img = _draw_label_overlay(frame_array, kp, bodyparts)
        cv2.imwrite(str(qc_dir / img_filename), qc_img)

        extracted.append({
            "img_filename": img_filename,
            "keypoints": kp,
        })
        metadata[img_filename] = {
            "frame_num": label["frame_num"],
            "local_frame": local_frame,
            "video_file": Path(video_path).name,
            "trial_idx": label["trial_idx"],
            "side": label["side"],
        }
        img_idx += 1

    # Write label metadata sidecar for future recovery
    meta_path = labeled_data_dir / "label_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Write CollectedData CSV
    csv_path = write_collected_data_csv(extracted, labeled_data_dir, training_name)

    # Create DLC config.yaml if it doesn't exist
    config_path = dlc_path / "config.yaml"
    if not config_path.exists():
        _create_dlc_config(dlc_path, subject_name, training_name)

    # Convert CSV to H5
    try:
        convert_csv_to_h5(str(config_path))
    except Exception as e:
        # Not fatal: the CSV is the primary record, and the training
        # pipeline regenerates the H5 inside the DeepLabCut environment
        # before it builds the training dataset.  Labeling a subject
        # before DeepLabCut is installed is a normal thing to do.
        logger.info("CSV-to-H5 conversion skipped (%s); the training job "
                    "will do it.", e)

    return {
        "dlc_dir": str(dlc_path),
        "labeled_data_dir": str(labeled_data_dir),
        "csv_path": str(csv_path),
        "frame_count": len(extracted),
    }


def _create_dlc_config(dlc_path: Path, subject_name: str, training_name: str):
    """Create a minimal DLC config.yaml."""
    settings = get_settings()
    config_path = dlc_path / "config.yaml"
    dlc_path.mkdir(parents=True, exist_ok=True)

    # Build bodyparts YAML list
    bp_lines = "\n".join(f"- {bp}" for bp in settings.bodyparts)

    # Build skeleton (chain all bodyparts)
    skeleton_lines = ""
    if len(settings.bodyparts) >= 2:
        pairs = []
        for i in range(len(settings.bodyparts) - 1):
            pairs.append(f"- - {settings.bodyparts[i]}\n  - {settings.bodyparts[i+1]}")
        skeleton_lines = "\n".join(pairs)

    config_content = f"""# Project definitions (do not edit)
Task: PD-PSP-MSA_finger_tapping
scorer: {settings.dlc_scorer}
date: {settings.dlc_date}
multianimalproject: false
identity:


# Project path (change when moving around)
project_path: {dlc_path}


# Default DeepLabCut engine to use for shuffle creation (either pytorch or tensorflow)
engine: pytorch


# Annotation data set configuration (and individual video cropping parameters)
video_sets:
  {dlc_path / 'videos' / f'{training_name}.mp4'}:
    crop: 0, 1920, 0, 1080


bodyparts:
{bp_lines}


# Fraction of video to start/stop when extracting frames for labeling/refinement
start: 0
stop: 1
numframes2pick: 20


# Plotting configuration
skeleton:
{skeleton_lines}
skeleton_color: white
pcutoff: 0.6
dotsize: 8
alphavalue: 0.7
colormap: rainbow


# Training
TrainingFraction:
- 0.95
iteration: 0
default_net_type: {settings.dlc_net_type}
default_augmenter: imgaug
snapshotindex: -1
batch_size: 8
"""
    config_path.write_text(config_content)
    return str(config_path)
