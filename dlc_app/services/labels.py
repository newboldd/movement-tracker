"""CollectedData CSV read/write and H5 conversion for DLC."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2

from ..config import get_settings
from .video import extract_frame_raw, build_trial_map, _resolve_frame


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


def convert_csv_to_h5(csv_path: Path):
    """Convert CollectedData CSV to H5 in DLC multi-index format.

    Reads the 3-row header (scorer, bodyparts, coords) and data rows,
    builds a MultiIndex DataFrame, and saves as HDF5.
    No DLC dependency required.
    """
    import pandas as pd

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        scorer_row = next(reader)    # scorer, "", "", scorer, scorer, ...
        bp_row = next(reader)        # bodyparts, "", "", bp1, bp1, bp2, bp2, ...
        coords_row = next(reader)    # coords, "", "", x, y, x, y, ...
        data_rows = list(reader)

    # Build column MultiIndex from header rows (skip first 3 index columns)
    n_prefix = 3
    tuples = list(zip(scorer_row[n_prefix:], bp_row[n_prefix:], coords_row[n_prefix:]))
    columns = pd.MultiIndex.from_tuples(tuples, names=["scorer", "bodyparts", "coords"])

    # Build row index from first 3 columns (labeled-data, round, imgfile)
    index = pd.MultiIndex.from_tuples(
        [(r[0], r[1], r[2]) for r in data_rows]
    )

    # Build data — convert to float, empty strings become NaN
    values = []
    for r in data_rows:
        row = []
        for v in r[n_prefix:]:
            try:
                row.append(float(v))
            except (ValueError, TypeError):
                row.append(float("nan"))
        values.append(row)

    df = pd.DataFrame(values, index=index, columns=columns)

    h5_path = csv_path.with_suffix(".h5")
    df.to_hdf(h5_path, key="df_with_missing", mode="w")
    return h5_path


def commit_labels_to_dlc(
    subject_name: str,
    session_labels: list[dict],
    iteration: int = 1,
):
    """Commit labeled frames: extract PNGs, write CSV/H5, create DLC structure.

    Args:
        subject_name: Subject identifier
        session_labels: list of dicts from frame_labels table (with keypoints JSON)
        iteration: labeling iteration (1 = initial, 2+ = refinement)

    Returns:
        dict with paths and counts
    """
    settings = get_settings()
    dlc_path = settings.dlc_path / subject_name
    training_name = f"round{iteration}"
    labeled_data_dir = dlc_path / "labeled-data" / training_name

    labeled_data_dir.mkdir(parents=True, exist_ok=True)

    # Build trial map for frame resolution
    trials = build_trial_map(subject_name)

    # Group labels by (frame_num, trial_idx, side) to handle both camera views
    # For DLC, each labeled image is one camera half at one frame
    extracted = []
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
        video_path, local_frame = _resolve_frame(trials, label["frame_num"])

        # Extract frame as PNG
        img_filename = f"img{img_idx:04d}.png"
        frame_array = extract_frame_raw(video_path, local_frame, label["side"])
        png_path = labeled_data_dir / img_filename
        cv2.imwrite(str(png_path), frame_array)

        extracted.append({
            "img_filename": img_filename,
            "keypoints": kp,
        })
        img_idx += 1

    # Write CollectedData CSV
    csv_path = write_collected_data_csv(extracted, labeled_data_dir, training_name)

    # Create DLC config.yaml if it doesn't exist
    config_path = dlc_path / "config.yaml"
    if not config_path.exists():
        _create_dlc_config(dlc_path, subject_name, training_name)

    # Convert CSV to H5
    try:
        convert_csv_to_h5(csv_path)
    except Exception as e:
        # Non-fatal — CSV is the primary format
        print(f"Warning: CSV to H5 conversion failed: {e}")

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
