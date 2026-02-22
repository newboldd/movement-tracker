"""Parse DLC analysis CSV outputs and serve predictions for the labeler."""
from __future__ import annotations

import csv
import logging
import re
from pathlib import Path

from ..config import get_settings
from .video import build_trial_map, get_subject_videos

logger = logging.getLogger(__name__)


def _parse_dlc_csv(csv_path: Path, likelihood_threshold: float = 0.6) -> dict:
    """Parse a DLC multi-header CSV into per-bodypart coordinate arrays.

    DLC CSVs have 3 header rows:
        Row 0: scorer name
        Row 1: bodypart names (repeated for x, y, likelihood)
        Row 2: 'x', 'y', 'likelihood' columns

    Returns:
        dict mapping bodypart -> list of [x, y] or None per frame.
    """
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) < 4:
        return {}

    # Parse headers
    bodypart_row = rows[1]
    coord_row = rows[2]

    # Build column map: {bodypart: {x: col_idx, y: col_idx, likelihood: col_idx}}
    col_map = {}
    for col_idx in range(1, len(bodypart_row)):
        bp = bodypart_row[col_idx]
        coord = coord_row[col_idx]
        if bp not in col_map:
            col_map[bp] = {}
        col_map[bp][coord] = col_idx

    # Parse data rows
    result = {}
    for bp, cols in col_map.items():
        if "x" not in cols or "y" not in cols:
            continue
        coords = []
        for row in rows[3:]:
            try:
                x = float(row[cols["x"]])
                y = float(row[cols["y"]])
                likelihood = float(row[cols["likelihood"]]) if "likelihood" in cols else 1.0
                if likelihood >= likelihood_threshold:
                    coords.append([x, y])
                else:
                    coords.append(None)
            except (IndexError, ValueError):
                coords.append(None)
        result[bp] = coords

    return result


def _match_csv_to_trial(csv_name: str, subject_name: str, cam_names: list[str]) -> tuple[str | None, str | None]:
    """Match a DLC CSV filename to (camera_name, trial_name).

    DLC CSV names follow pattern:
        {Subject}_{Trial}_{Camera}DLC_*.csv
    e.g. Con07_L1_OSDLC_resnet50_labelsSep16shuffle1_100000.csv
    """
    stem = csv_name.replace(".csv", "")

    for cam in cam_names:
        # Look for camera name followed by 'DLC'
        pattern = f"_{cam}DLC"
        if pattern in stem:
            # Extract trial: everything between subject_ and _camDLC
            prefix = f"{subject_name}_"
            if stem.startswith(prefix):
                trial_part = stem[len(prefix):stem.index(pattern)]
                trial_name = f"{subject_name}_{trial_part}"
                return cam, trial_name
    return None, None


def get_dlc_predictions_for_session(subject_name: str) -> dict | None:
    """Load DLC analysis predictions for a subject.

    Looks in the subject's labels_v1/ directory for DLC CSV outputs.
    Returns dict in same format as mediapipe prelabels:
        {camera: {bodypart: [[x,y]|null, ...]}}

    Coordinates are in the cropped single-camera frame space (same as
    what DLC analyzed). The labeler already works with per-camera coords.
    """
    settings = get_settings()
    dlc_dir = settings.dlc_path / subject_name
    cam_names = settings.camera_names

    # Find labels_v1 directory
    labels_dir = None
    for name in ["labels_v1", "labels_v1.0", "labels_v0.1"]:
        candidate = dlc_dir / name
        if candidate.exists() and candidate.is_dir():
            labels_dir = candidate
            break

    if not labels_dir:
        return None

    # Find DLC CSV files
    csv_files = sorted(labels_dir.glob("*DLC*.csv"))
    if not csv_files:
        return None

    # Build trial map to know frame offsets
    trials = build_trial_map(subject_name)
    if not trials:
        return None

    total_frames = trials[-1]["end_frame"] + 1

    # Initialize result structure
    result = {}
    for cam in cam_names:
        result[cam] = {}
        for bp in settings.bodyparts:
            result[cam][bp] = [None] * total_frames

    # Parse each CSV and place predictions into the correct global frame range
    for csv_path in csv_files:
        cam, trial_name = _match_csv_to_trial(csv_path.name, subject_name, cam_names)
        if not cam or not trial_name:
            continue

        # Find matching trial in trial map
        matching_trial = None
        for t in trials:
            if t["trial_name"] == trial_name:
                matching_trial = t
                break

        if not matching_trial:
            # Try without subject prefix
            trial_short = trial_name.replace(f"{subject_name}_", "")
            for t in trials:
                if t["trial_name"].endswith(trial_short):
                    matching_trial = t
                    break

        if not matching_trial:
            logger.debug(f"No trial match for CSV {csv_path.name}")
            continue

        # Parse CSV
        parsed = _parse_dlc_csv(csv_path)
        if not parsed:
            continue

        start_frame = matching_trial["start_frame"]

        # Map bodypart names (DLC may use different names)
        for bp in settings.bodyparts:
            dlc_coords = parsed.get(bp)
            if not dlc_coords:
                continue

            for local_frame, coord in enumerate(dlc_coords):
                global_frame = start_frame + local_frame
                if global_frame < total_frames and coord is not None:
                    result[cam][bp][global_frame] = coord

    # Check if we actually have any predictions
    has_data = any(
        any(c is not None for c in result[cam][bp])
        for cam in cam_names
        for bp in settings.bodyparts
    )

    return result if has_data else None
