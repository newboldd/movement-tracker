"""CollectedData CSV read/write and H5 conversion for DLC."""
from __future__ import annotations

import csv
import json
import subprocess
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
        raise RuntimeError(f"convertcsv2h5 failed: {result.stderr}")
    return result.stdout


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

    # Clean up old PNGs from previous commits
    for old_png in labeled_data_dir.glob("img*.png"):
        old_png.unlink()

    # Build trial map for frame resolution
    trials = build_trial_map(subject_name)

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
        metadata[img_filename] = {
            "frame_num": label["frame_num"],
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
        # Non-fatal — CSV is the primary format; H5 needs DeepLabCut
        print(f"Warning: CSV to H5 conversion skipped ({e}). Install DeepLabCut for H5 support.")

    # Update crop params in config.yaml from MP + manual corrections
    try:
        _update_crop_params(subject_name, config_path)
    except Exception as e:
        print(f"Warning: Could not update crop params ({e})")

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


def _update_crop_params(subject_name: str, config_path: Path):
    """Update DLC config.yaml video_sets crop params from MP + manual corrections.

    Computes bounding box of all detected thumb/index positions per camera,
    adds margin, and writes crop: x1, x2, y1, y2 into config.yaml video_sets.
    """
    from .mediapipe_prelabel import compute_optimal_crop

    crops = compute_optimal_crop(subject_name)

    if not any(v is not None for v in crops.values()):
        return

    if not config_path.exists():
        return

    # Read existing config
    text = config_path.read_text()
    lines = text.splitlines(keepends=True)

    # Find and update crop lines in video_sets section
    # The crop format in DLC config is: crop: x1, x2, y1, y2
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("crop:"):
            # Use first camera's crop (since DLC trains on cropped single-camera images)
            settings = get_settings()
            cam = settings.camera_names[0] if settings.camera_names else None
            crop = crops.get(cam) if cam else None
            if crop:
                x1, x2, y1, y2 = crop
                indent = line[:len(line) - len(line.lstrip())]
                new_lines.append(f"{indent}crop: {x1}, {x2}, {y1}, {y2}\n")
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    config_path.write_text("".join(new_lines))


def save_corrections_to_csv(
    subject_name: str,
    session_labels: list[dict],
) -> dict:
    """Write correction labels as DLC-format CSVs in corrections/ directory.

    Creates one CSV per (trial, camera) pair, matching the naming pattern
    recognized by _match_csv_to_trial(): {Subject}_{Trial}_DLC_{Camera}.csv

    Args:
        subject_name: Subject identifier
        session_labels: list of dicts from frame_labels table

    Returns:
        dict with paths and counts
    """
    settings = get_settings()
    corrections_dir = settings.dlc_path / subject_name / "corrections"
    corrections_dir.mkdir(parents=True, exist_ok=True)

    trials = build_trial_map(subject_name)
    if not trials:
        return {"corrections_dir": str(corrections_dir), "csv_count": 0, "frame_count": 0}

    bodyparts = settings.bodyparts
    scorer = settings.dlc_scorer

    # Group labels by (trial_idx, side)
    groups: dict[tuple[int, str], list[dict]] = {}
    for label in session_labels:
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

        key = (label["trial_idx"], label["side"])
        if key not in groups:
            groups[key] = []
        groups[key].append({
            "frame_num": label["frame_num"],
            "keypoints": kp,
        })

    csv_count = 0
    total_frames = 0

    for (trial_idx, side), frame_labels in groups.items():
        if trial_idx >= len(trials):
            continue

        trial = trials[trial_idx]
        trial_name = trial["trial_name"]
        # Strip subject prefix to get trial part: "MSA01_L1" -> "L1"
        trial_part = trial_name
        if trial_name.startswith(f"{subject_name}_"):
            trial_part = trial_name[len(f"{subject_name}_"):]

        # Build full-trial-length arrays (one row per frame in the trial)
        n_frames = trial["frame_count"]
        start_frame = trial["start_frame"]

        # Initialize all frames as empty
        frame_data = [None] * n_frames

        for fl in frame_labels:
            local_frame = fl["frame_num"] - start_frame
            if 0 <= local_frame < n_frames:
                frame_data[local_frame] = fl["keypoints"]

        # Write CSV: {Subject}_{TrialPart}_DLC_{Camera}.csv
        csv_name = f"{subject_name}_{trial_part}_DLC_{side}.csv"
        csv_path = corrections_dir / csv_name

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            n_bp = len(bodyparts)

            # DLC multi-header rows
            scorer_row = ["scorer"] + [scorer] * (n_bp * 3)
            bp_row = ["bodyparts"]
            for bp in bodyparts:
                bp_row.extend([bp, bp, bp])
            coords_row = ["coords"] + ["x", "y", "likelihood"] * n_bp

            writer.writerow(scorer_row)
            writer.writerow(bp_row)
            writer.writerow(coords_row)

            # Data rows — one per frame in the trial
            for local_frame in range(n_frames):
                row = [local_frame]
                kp = frame_data[local_frame]
                for bp in bodyparts:
                    if kp:
                        coords = kp.get(bp)
                        if coords and len(coords) >= 2 and coords[0] is not None:
                            row.extend([coords[0], coords[1], 1.0])
                        else:
                            row.extend(["", "", 0.0])
                    else:
                        row.extend(["", "", 0.0])
                writer.writerow(row)

        csv_count += 1
        total_frames += len(frame_labels)

    return {
        "corrections_dir": str(corrections_dir),
        "csv_count": csv_count,
        "frame_count": total_frames,
    }
