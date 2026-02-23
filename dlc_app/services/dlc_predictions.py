"""Parse DLC analysis CSV outputs and serve predictions for the labeler."""
from __future__ import annotations

import csv
import logging
import re
from pathlib import Path

import numpy as np

from ..config import get_settings
from .calibration import get_calibration_for_subject, triangulate_points
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

    Handles two naming conventions:
        1. {Subject}_{Trial}_{Camera}DLC_*.csv
           e.g. Con07_L1_OSDLC_resnet50_labelsSep16shuffle1_100000.csv
        2. {Subject}_{Trial}_DLC_{Camera}.csv
           e.g. Con01_R1_DLC_OS.csv
    """
    stem = csv_name.replace(".csv", "")
    prefix = f"{subject_name}_"
    if not stem.startswith(prefix):
        return None, None

    for cam in cam_names:
        # Pattern 1: {Subject}_{Trial}_{Camera}DLC_*
        pattern1 = f"_{cam}DLC"
        if pattern1 in stem:
            trial_part = stem[len(prefix):stem.index(pattern1)]
            trial_name = f"{subject_name}_{trial_part}"
            return cam, trial_name

        # Pattern 2: {Subject}_{Trial}_DLC_{Camera}
        pattern2 = f"_DLC_{cam}"
        if stem.endswith(pattern2) or f"{pattern2}_" in stem:
            trial_part = stem[len(prefix):stem.index(pattern2)]
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

    if not has_data:
        return None

    # Compute 3D distances from DLC thumb/index via stereo triangulation
    dlc_distances = _compute_dlc_distances(result, cam_names, subject_name, total_frames)
    if dlc_distances is not None:
        result["distances"] = dlc_distances

    return result


def _compute_dlc_distances(predictions: dict, cam_names: list[str],
                           subject_name: str, total_frames: int) -> list | None:
    """Compute 3D thumb-index distances from DLC predictions via stereo triangulation.

    Returns list of float|None per frame, or None if calibration unavailable.
    """
    if len(cam_names) < 2:
        return None

    calib = get_calibration_for_subject(subject_name)
    if calib is None:
        return None

    os_thumb = predictions.get(cam_names[0], {}).get("thumb", [])
    os_index = predictions.get(cam_names[0], {}).get("index", [])
    od_thumb = predictions.get(cam_names[1], {}).get("thumb", [])
    od_index = predictions.get(cam_names[1], {}).get("index", [])

    if not os_thumb:
        return None

    distances = []
    valid_count = 0
    for i in range(total_frames):
        if (os_thumb[i] is None or os_index[i] is None or
                od_thumb[i] is None or od_index[i] is None):
            distances.append(None)
            continue

        pts_L = np.array([os_thumb[i], os_index[i]], dtype=np.float64)
        pts_R = np.array([od_thumb[i], od_index[i]], dtype=np.float64)
        pts_3d = triangulate_points(pts_L, pts_R, calib)

        if np.any(np.isnan(pts_3d)):
            distances.append(None)
        else:
            distances.append(round(float(np.linalg.norm(pts_3d[0] - pts_3d[1])), 2))
            valid_count += 1

    if valid_count == 0:
        return None

    logger.info(f"Computed DLC distances for {subject_name}: {valid_count}/{total_frames} frames")
    return distances
