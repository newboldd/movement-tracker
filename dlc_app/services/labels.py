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
    """Write CollectedData_labels.csv in DLC format.

    DLC expects: single index column with relative image path,
    3-row multi-level header (scorer, bodyparts, coords).

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

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        n_bp = len(bodyparts)

        # Header rows: first cell empty (index column header)
        scorer_row = ["scorer"] + [scorer] * (n_bp * 2)
        bp_row = ["bodyparts"]
        for bp in bodyparts:
            bp_row.extend([bp, bp])
        coords_row = ["coords"] + ["x", "y"] * n_bp

        writer.writerow(scorer_row)
        writer.writerow(bp_row)
        writer.writerow(coords_row)

        # Data rows: index = relative path to image
        for label in sorted(labels, key=lambda x: x["img_filename"]):
            kp = label.get("keypoints", {})
            img_path = f"labeled-data/{training_name}/{label['img_filename']}"
            row = [img_path]
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

    # Use pandas to read in DLC's expected format
    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)

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

    # Create training dataset so subject is ready to train immediately
    try:
        td_result = create_training_dataset(dlc_path)
    except Exception as e:
        print(f"Warning: Training dataset creation failed: {e}")
        td_result = None

    result = {
        "dlc_dir": str(dlc_path),
        "labeled_data_dir": str(labeled_data_dir),
        "csv_path": str(csv_path),
        "frame_count": len(extracted),
    }
    if td_result:
        result["training_dataset_dir"] = td_result
    return result


def create_training_dataset(dlc_path: Path) -> str:
    """Create DLC training dataset natively (no DLC dependency).

    Reads labeled-data CSVs, builds a train/test split, and writes
    the training-datasets directory structure that DLC expects.

    Args:
        dlc_path: Path to the subject's DLC project directory

    Returns:
        str path to the training dataset directory
    """
    import random
    import yaml

    settings = get_settings()

    labeled_dir = dlc_path / "labeled-data"
    if not labeled_dir.exists():
        raise FileNotFoundError(f"No labeled data at {labeled_dir}")

    # Read config.yaml for training fraction
    config = {}
    config_file = dlc_path / "config.yaml"
    if config_file.exists():
        config = yaml.safe_load(config_file.read_text()) or {}

    train_fraction = 0.95
    if "TrainingFraction" in config and config["TrainingFraction"]:
        train_fraction = config["TrainingFraction"][0]

    # Gather all image paths from labeled-data subdirectories
    all_images = []
    for subdir in sorted(labeled_dir.iterdir()):
        if not subdir.is_dir():
            continue
        csv_file = subdir / "CollectedData_labels.csv"
        if csv_file.exists():
            for img in sorted(subdir.glob("img*.png")):
                all_images.append(str(img.relative_to(dlc_path)))

    if not all_images:
        raise FileNotFoundError("No labeled images found in labeled-data/")

    # Train/test split
    random.seed(42)
    shuffled = all_images[:]
    random.shuffle(shuffled)
    n_train = max(1, int(len(shuffled) * train_fraction))
    train_set = sorted(shuffled[:n_train])
    test_set = sorted(shuffled[n_train:]) if n_train < len(shuffled) else []

    # Write training-datasets directory
    iteration = config.get("iteration", 0)
    dataset_name = f"iteration-{iteration}/UnaugmentedDataSet_{config.get('Task', 'project')}{config.get('date', '')}"
    dataset_dir = dlc_path / "training-datasets" / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Write train/test index files
    (dataset_dir / "CollectedData_train.csv").write_text("\n".join(train_set))
    if test_set:
        (dataset_dir / "CollectedData_test.csv").write_text("\n".join(test_set))

    return str(dataset_dir)


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
