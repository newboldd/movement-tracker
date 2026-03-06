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


def _parse_simple_corrections_csv(csv_path: Path) -> dict:
    """Parse a simple corrections CSV (header: thumb_x,thumb_y,index_x,index_y,...).

    Returns:
        dict mapping bodypart -> list of [x, y] or None per frame,
        or {} if the file isn't in this format.
    """
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) < 2:
        return {}

    header = rows[0]

    # Detect simple format: header cells are like "thumb_x", "thumb_y", "index_x", ...
    col_map = {}  # {bodypart: {"x": col_idx, "y": col_idx}}
    for col_idx, col_name in enumerate(header):
        if "_x" in col_name:
            bp = col_name.rsplit("_x", 1)[0]
            col_map.setdefault(bp, {})["x"] = col_idx
        elif "_y" in col_name:
            bp = col_name.rsplit("_y", 1)[0]
            col_map.setdefault(bp, {})["y"] = col_idx

    if not col_map or not any("x" in v and "y" in v for v in col_map.values()):
        return {}

    result = {}
    for bp, cols in col_map.items():
        if "x" not in cols or "y" not in cols:
            continue
        coords = []
        for row in rows[1:]:
            try:
                x = float(row[cols["x"]])
                y = float(row[cols["y"]])
                if x != x or y != y:  # NaN check
                    coords.append(None)
                else:
                    coords.append([x, y])
            except (IndexError, ValueError):
                coords.append(None)
        result[bp] = coords

    return result


def _parse_dlc_csv(csv_path: Path, likelihood_threshold: float = 0.0) -> dict:
    """Parse a DLC multi-header CSV into per-bodypart coordinate arrays.

    DLC CSVs have 3 header rows:
        Row 0: scorer name
        Row 1: bodypart names (repeated for x, y, likelihood)
        Row 2: 'x', 'y', 'likelihood' columns

    Also handles simple corrections CSVs (thumb_x,thumb_y,...) as a fallback.

    Returns:
        dict mapping bodypart -> list of [x, y] or None per frame.
    """
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) < 2:
        return {}

    # Detect simple corrections format (header like "thumb_x,thumb_y,...")
    if len(rows[0]) >= 2 and "_x" in rows[0][0]:
        return _parse_simple_corrections_csv(csv_path)

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
                coords.append([x, y])
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

        # Pattern 3: {Subject}_{Trial}_{Camera}_* (generic, no DLC in name)
        remainder = stem[len(prefix):]
        cam_sep = f"_{cam}_"
        idx = remainder.find(cam_sep)
        if idx >= 0:
            trial_part = remainder[:idx]
            if trial_part:
                return cam, f"{subject_name}_{trial_part}"
        if remainder.endswith(f"_{cam}"):
            trial_part = remainder[:-(len(cam) + 1)]
            if trial_part:
                return cam, f"{subject_name}_{trial_part}"

    return None, None


def _find_label_dir(dlc_dir: Path, dir_names: list[str]) -> tuple[Path | None, list[Path]]:
    """Find the first label directory with DLC CSVs.

    Args:
        dlc_dir: Subject's DLC directory.
        dir_names: Directory names to search in priority order.

    Returns:
        (labels_dir, csv_files) or (None, []) if not found.
    """
    for name in dir_names:
        candidate = dlc_dir / name
        if candidate.exists() and candidate.is_dir():
            found = sorted(candidate.glob("*DLC*.csv"))
            if not found:
                # Fallback: any CSV (e.g. manual corrections without DLC in name)
                found = sorted(candidate.glob("*.csv"))
            if found:
                return candidate, found
    return None, []


def _load_from_label_dir(
    subject_name: str, labels_dir: Path, csv_files: list[Path],
) -> dict | None:
    """Load DLC predictions from a specific label directory.

    Parses CSVs, maps to global frame indices via trial map, computes 3D distances.

    Returns dict: {camera: {bodypart: [[x,y]|null, ...]}, distances: [...]}
    or None if no usable data.
    """
    settings = get_settings()
    cam_names = settings.camera_names

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

        parsed = _parse_dlc_csv(csv_path)
        if not parsed:
            continue

        start_frame = matching_trial["start_frame"]

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

    # Compute 3D distances
    dlc_distances = _compute_dlc_distances(result, cam_names, subject_name, total_frames)
    if dlc_distances is not None:
        result["distances"] = dlc_distances

    return result


def get_dlc_predictions_for_session(subject_name: str) -> dict | None:
    """Load DLC analysis predictions for a subject.

    Searches for the best available label directory in priority order:
    corrections/ > labels_v2/ > labels_v1/ (and variants).

    Returns dict in same format as mediapipe prelabels:
        {camera: {bodypart: [[x,y]|null, ...]}}

    Coordinates are in the cropped single-camera frame space (same as
    what DLC analyzed). The labeler already works with per-camera coords.
    """
    settings = get_settings()
    dlc_dir = settings.dlc_path / subject_name

    labels_dir, csv_files = _find_label_dir(
        dlc_dir, ["corrections", "labels_v2", "labels_v1", "labels_v1.0", "labels_v0.1"]
    )
    if not labels_dir:
        return None

    logger.info(f"Using label source '{labels_dir.name}' for {subject_name}")
    return _load_from_label_dir(subject_name, labels_dir, csv_files)


# Stage name → directory names mapping for per-stage loading
STAGE_DIR_MAP = {
    "dlc": ["labels_v1", "labels_v1.0", "labels_v0.1"],
    "refine": ["labels_v2"],
    "corrections": ["corrections"],
}


def get_dlc_predictions_for_stage(subject_name: str, stage: str) -> dict | None:
    """Load DLC predictions from a specific processing stage only.

    Args:
        subject_name: Subject identifier.
        stage: One of 'dlc', 'refine', 'corrections'.

    Returns:
        dict in labeler format, or None if no data for this stage.
    """
    dir_names = STAGE_DIR_MAP.get(stage)
    if not dir_names:
        return None

    settings = get_settings()
    dlc_dir = settings.dlc_path / subject_name

    labels_dir, csv_files = _find_label_dir(dlc_dir, dir_names)
    if not labels_dir:
        return None

    logger.info(f"Loading stage '{stage}' from '{labels_dir.name}' for {subject_name}")
    return _load_from_label_dir(subject_name, labels_dir, csv_files)


def has_stage_data(subject_name: str, stage: str) -> bool:
    """Check whether a specific processing stage has data for a subject."""
    dir_names = STAGE_DIR_MAP.get(stage)
    if not dir_names:
        return False

    settings = get_settings()
    dlc_dir = settings.dlc_path / subject_name
    labels_dir, _ = _find_label_dir(dlc_dir, dir_names)
    return labels_dir is not None


def get_stage_csv_files(subject_name: str, stage: str) -> list[str]:
    """Return CSV filenames for a specific processing stage."""
    dir_names = STAGE_DIR_MAP.get(stage)
    if not dir_names:
        return []

    settings = get_settings()
    dlc_dir = settings.dlc_path / subject_name
    labels_dir, csv_files = _find_label_dir(dlc_dir, dir_names)
    if not labels_dir:
        return []
    return [f.name for f in csv_files]


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
