"""MANO 3D hand model data loading, projection, and serving.

Loads mano_fit_v2.npz, mediapipe.pkl, heatmaps, and calibration for each trial.
Projects 3D MANO joints to 2D camera coordinates.  Computes distance traces.
"""
from __future__ import annotations

import json
import logging
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..config import get_settings
from .calibration import get_calibration_for_subject, load_calibration, triangulate_points
from .video import build_trial_map

logger = logging.getLogger(__name__)

# ── Hand skeleton constants ────────────────────────────────────────────

HAND_SKELETON = [
    (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),   # wrist → finger bases
    (1, 2), (2, 3), (3, 4),                       # thumb
    (5, 6), (6, 7), (7, 8),                       # index
    (9, 10), (10, 11), (11, 12),                   # middle
    (13, 14), (14, 15), (15, 16),                  # ring
    (17, 18), (18, 19), (19, 20),                  # pinky
]

FINGER_GROUPS = {
    "wrist": [0],
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}

JOINT_NAMES = [
    "Wrist",
    "T_CMC", "T_MCP", "T_IP", "T_Tip",
    "I_MCP", "I_PIP", "I_DIP", "I_Tip",
    "M_MCP", "M_PIP", "M_DIP", "M_Tip",
    "R_MCP", "R_PIP", "R_DIP", "R_Tip",
    "P_MCP", "P_PIP", "P_DIP", "P_Tip",
]

DISTANCE_OPTIONS: dict[str, tuple[int, int]] = {
    "Thumb-Index Aperture": (4, 8),
    "Thumb-Middle Aperture": (4, 12),
    "Index-Middle Aperture": (8, 12),
    "Thumb: CMC-MCP (1-2)": (1, 2),
    "Thumb: MCP-IP (2-3)": (2, 3),
    "Thumb: IP-Tip (3-4)": (3, 4),
    "Index: MCP-PIP (5-6)": (5, 6),
    "Index: PIP-DIP (6-7)": (6, 7),
    "Index: DIP-Tip (7-8)": (7, 8),
    "Middle: MCP-PIP (9-10)": (9, 10),
    "Middle: PIP-DIP (10-11)": (10, 11),
    "Middle: DIP-Tip (11-12)": (11, 12),
    "Wrist-Thumb base (0-1)": (0, 1),
    "Wrist-Index base (0-5)": (0, 5),
    "Wrist-Middle base (0-9)": (0, 9),
}


# ── Helpers ────────────────────────────────────────────────────────────

def _nan_to_none(val) -> float | None:
    """Convert NaN to None for JSON serialisation."""
    if val is None:
        return None
    v = float(val)
    if v != v:  # NaN check (faster than math.isnan)
        return None
    return v


def _points_to_list(arr: np.ndarray) -> list:
    """Convert (N, J, D) numpy array to nested list, replacing NaN with None.

    Returns list of frames, each frame is list of joints, each joint is [x,y]
    or None if NaN.
    """
    result = []
    for t in range(arr.shape[0]):
        frame = []
        for j in range(arr.shape[1]):
            if np.isnan(arr[t, j, 0]):
                frame.append(None)
            else:
                frame.append([round(float(arr[t, j, d]), 2) for d in range(arr.shape[2])])
        result.append(frame)
    return result


def _array_to_list(arr: np.ndarray) -> list:
    """Convert 1D numpy array to list, NaN → None."""
    return [_nan_to_none(v) for v in arr]


def _compute_distances(joints_3d: np.ndarray) -> dict[str, list]:
    """Compute distance traces for all metrics from (N,21,3) joints."""
    distances = {}
    for name, (ja, jb) in DISTANCE_OPTIONS.items():
        n = joints_3d.shape[0]
        dist = np.full(n, np.nan)
        valid = ~np.isnan(joints_3d[:, ja, 0]) & ~np.isnan(joints_3d[:, jb, 0])
        if valid.any():
            dist[valid] = np.linalg.norm(
                joints_3d[valid, ja, :] - joints_3d[valid, jb, :], axis=1
            )
        distances[name] = _array_to_list(dist)
    return distances


def _project_to_2d(joints_3d: np.ndarray, K, dist, R, T) -> np.ndarray:
    """Project (N,21,3) world-frame joints to (N,21,2) image coordinates.

    For the left camera (camera 1), R=identity, T=zero.
    For the right camera (camera 2), use the stereo R, T.
    """
    N = joints_3d.shape[0]
    proj = np.full((N, 21, 2), np.nan)

    rvec = cv2.Rodrigues(R)[0] if R.shape != (3, 1) else R
    tvec = T.reshape(3, 1) if T.shape != (3, 1) else T

    for t in range(N):
        valid = ~np.isnan(joints_3d[t, :, 0])
        if not valid.any():
            continue
        pts = joints_3d[t, valid].reshape(-1, 1, 3).astype(np.float64)
        pts_2d, _ = cv2.projectPoints(pts, rvec, tvec, K, dist)
        proj[t, valid] = pts_2d.reshape(-1, 2)

    return proj


# ── Trial discovery ────────────────────────────────────────────────────

def _mano_dir(subject_name: str) -> Path:
    """Return path to mano data directory for a subject."""
    settings = get_settings()
    return settings.dlc_path / subject_name / "mano"


def list_mano_trials(subject_name: str) -> list[dict]:
    """List trials that have MANO fit results.

    Cross-references against the video trial map so trial indices match
    the existing video service.
    """
    mano_root = _mano_dir(subject_name)
    if not mano_root.is_dir():
        return []

    # Build video trial map for index alignment
    try:
        video_trials = build_trial_map(subject_name)
    except Exception:
        video_trials = []

    trial_stem_to_idx = {t["trial_name"]: i for i, t in enumerate(video_trials)}
    results = []

    for d in sorted(mano_root.iterdir()):
        if not d.is_dir():
            continue
        if not (d / "mano_fit_v2.npz").exists():
            continue
        trial_stem = d.name
        trial_idx = trial_stem_to_idx.get(trial_stem)

        # Get frame count and fps from video trial map if available
        n_frames = 0
        fps = 30.0
        if trial_idx is not None and trial_idx < len(video_trials):
            vt = video_trials[trial_idx]
            n_frames = vt["frame_count"]
            fps = vt["fps"]
        else:
            # Fall back to mano data frame count
            try:
                data = np.load(str(d / "mano_fit_v2.npz"), allow_pickle=True)
                n_frames = data["joints_3d"].shape[0]
            except Exception:
                pass

        has_heatmaps = (d / "hrnet_w18_heatmaps.npz").exists()

        results.append({
            "trial_idx": trial_idx,
            "trial_stem": trial_stem,
            "n_frames": n_frames,
            "has_heatmaps": has_heatmaps,
            "fps": fps,
        })

    return results


# ── Data loading ───────────────────────────────────────────────────────

@lru_cache(maxsize=8)
def _load_mano_npz(npz_path: str) -> dict:
    """Load and cache MANO fit npz.  Returns dict of numpy arrays."""
    data = np.load(npz_path, allow_pickle=True)
    return dict(data)


@lru_cache(maxsize=8)
def _load_mediapipe_pkl(pkl_path: str) -> dict:
    """Load and cache mediapipe pkl."""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _load_trial_calibration(subject_name: str, trial_stem: str) -> dict | None:
    """Load calibration for a trial.

    Priority:
      1. Per-trial calibration.yaml in mano data dir
      2. Subject-level calibration from the app's calibration service
    """
    mano_trial_dir = _mano_dir(subject_name) / trial_stem
    local_calib = mano_trial_dir / "calibration.yaml"
    if local_calib.exists():
        try:
            return load_calibration(str(local_calib))
        except Exception as e:
            logger.warning(f"Failed to load trial calibration {local_calib}: {e}")

    return get_calibration_for_subject(subject_name)


def load_mano_trial_data(subject_name: str, trial_stem: str) -> dict[str, Any]:
    """Load all MANO viewer data for a single trial.

    Returns a dict ready for JSON serialisation (numpy arrays converted to lists).
    """
    mano_trial_dir = _mano_dir(subject_name) / trial_stem
    npz_path = mano_trial_dir / "mano_fit_v2.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"No mano_fit_v2.npz for {subject_name}/{trial_stem}")

    # ── Load MANO fit ──────────────────────────────────────────
    mano = _load_mano_npz(str(npz_path))
    joints_3d = mano["joints_3d"]  # (N, 21, 3)
    N = joints_3d.shape[0]

    fit_error_L = mano.get("fit_error_L", np.full(N, np.nan))
    fit_error_R = mano.get("fit_error_R", np.full(N, np.nan))
    residuals_L = mano.get("residuals_L")  # (N, 21) or None
    residuals_R = mano.get("residuals_R")
    mp_weights_L = mano.get("mp_weights_L")  # (N, 21) or None
    mp_weights_R = mano.get("mp_weights_R")

    # ── Load calibration ───────────────────────────────────────
    calib = _load_trial_calibration(subject_name, trial_stem)
    if calib is None:
        raise FileNotFoundError(f"No calibration for {subject_name}")

    K1, K2 = calib["K1"], calib["K2"]
    dist1, dist2 = calib["dist1"], calib["dist2"]
    R, T = calib["R"], calib["T"]

    # ── Project 3D→2D ──────────────────────────────────────────
    # Left camera: identity rotation and translation
    R_eye = np.eye(3, dtype=np.float64)
    T_zero = np.zeros((3, 1), dtype=np.float64)

    mano_proj_L = _project_to_2d(joints_3d, K1, dist1, R_eye, T_zero)
    mano_proj_R = _project_to_2d(joints_3d, K2, dist2, R, T)

    # ── Load MediaPipe ─────────────────────────────────────────
    mp_path = mano_trial_dir / "mediapipe.pkl"
    mp_tracked_L = np.full((N, 21, 2), np.nan)
    mp_tracked_R = np.full((N, 21, 2), np.nan)

    if mp_path.exists():
        mp_data = _load_mediapipe_pkl(str(mp_path))
        mp_L = mp_data.get("tracked_L")
        mp_R = mp_data.get("tracked_R")
        if mp_L is not None:
            n = min(N, mp_L.shape[0])
            mp_tracked_L[:n] = mp_L[:n]
        if mp_R is not None:
            n = min(N, mp_R.shape[0])
            mp_tracked_R[:n] = mp_R[:n]

    # ── Triangulate MP to 3D ───────────────────────────────────
    mp_joints_3d = np.full((N, 21, 3), np.nan)
    for j in range(21):
        pts_L = mp_tracked_L[:, j, :]  # (N, 2)
        pts_R = mp_tracked_R[:, j, :]  # (N, 2)
        mp_joints_3d[:, j, :] = triangulate_points(pts_L, pts_R, calib)

    # ── Load DLC predictions (thumb+index only) ────────────────
    dlc_thumb_OS = np.full((N, 2), np.nan)
    dlc_thumb_OD = np.full((N, 2), np.nan)
    dlc_index_OS = np.full((N, 2), np.nan)
    dlc_index_OD = np.full((N, 2), np.nan)

    try:
        from .dlc_predictions import get_dlc_predictions_for_trial
        # This may not exist — gracefully skip
        pass
    except (ImportError, Exception):
        pass

    # Use existing corrections CSV if available
    settings = get_settings()
    corr_dir = settings.dlc_path / subject_name / "corrections"
    labels_v2_dir = settings.dlc_path / subject_name / "labels_v2"

    # Try to load from labels_v2 CSVs (DLC analyze output)
    for labels_dir in [corr_dir, labels_v2_dir]:
        if labels_dir and labels_dir.is_dir():
            import csv as _csv
            for csv_file in sorted(labels_dir.glob(f"*{trial_stem}*.csv")):
                try:
                    _load_dlc_csv(
                        csv_file, N,
                        dlc_thumb_OS, dlc_thumb_OD,
                        dlc_index_OS, dlc_index_OD,
                    )
                    break
                except Exception:
                    continue
            else:
                continue
            break

    # ── Compute distances ──────────────────────────────────────
    distances_mano = _compute_distances(joints_3d)
    distances_mp = _compute_distances(mp_joints_3d)

    # ── Get video trial info for fps ───────────────────────────
    fps = 30.0
    try:
        trials = build_trial_map(subject_name)
        for t in trials:
            if t["trial_name"] == trial_stem:
                fps = t["fps"]
                break
    except Exception:
        pass

    # ── Assemble result ────────────────────────────────────────
    result: dict[str, Any] = {
        "n_frames": N,
        "fps": fps,
        "skeleton": HAND_SKELETON,
        "finger_groups": FINGER_GROUPS,
        "distance_options": {k: list(v) for k, v in DISTANCE_OPTIONS.items()},
        "joint_names": JOINT_NAMES,
        # Camera calibration for snap-to-camera 3D overlay
        "calib": {
            "K_L": K1.tolist(),
            "K_R": K2.tolist(),
            "R": R.tolist(),
            "T": T.ravel().tolist(),
        },
        # 2D projections
        "mano_proj_L": _points_to_list(mano_proj_L),
        "mano_proj_R": _points_to_list(mano_proj_R),
        # 3D joints
        "mano_joints_3d": _points_to_list(joints_3d),
        # MediaPipe
        "mp_tracked_L": _points_to_list(mp_tracked_L),
        "mp_tracked_R": _points_to_list(mp_tracked_R),
        "mp_joints_3d": _points_to_list(mp_joints_3d),
        # DLC (thumb=joint4, index=joint8)
        "dlc_thumb_OS": [_nan_to_none_pair(dlc_thumb_OS[t]) for t in range(N)],
        "dlc_thumb_OD": [_nan_to_none_pair(dlc_thumb_OD[t]) for t in range(N)],
        "dlc_index_OS": [_nan_to_none_pair(dlc_index_OS[t]) for t in range(N)],
        "dlc_index_OD": [_nan_to_none_pair(dlc_index_OD[t]) for t in range(N)],
        # Distances
        "distances_mano": distances_mano,
        "distances_mp": distances_mp,
        # Fit quality
        "fit_error_L": _array_to_list(fit_error_L),
        "fit_error_R": _array_to_list(fit_error_R),
    }

    # Optional: weights and residuals
    if mp_weights_L is not None:
        result["mp_weights_L"] = [[_nan_to_none(round(float(mp_weights_L[t, j]), 3))
                                    for j in range(21)] for t in range(N)]
    if mp_weights_R is not None:
        result["mp_weights_R"] = [[_nan_to_none(round(float(mp_weights_R[t, j]), 3))
                                    for j in range(21)] for t in range(N)]

    return result


def _nan_to_none_pair(arr: np.ndarray) -> list | None:
    """Convert a (2,) array to [x,y] or None if NaN."""
    if np.isnan(arr[0]):
        return None
    return [round(float(arr[0]), 2), round(float(arr[1]), 2)]


def _load_dlc_csv(csv_path: Path, n_frames: int,
                   thumb_OS: np.ndarray, thumb_OD: np.ndarray,
                   index_OS: np.ndarray, index_OD: np.ndarray):
    """Load DLC predictions from a CSV file into the output arrays."""
    import csv
    with open(csv_path) as f:
        reader = csv.reader(f)
        # Skip header rows (typically 3 header rows in DLC CSVs)
        headers = []
        for _ in range(3):
            try:
                headers.append(next(reader))
            except StopIteration:
                return

        # Parse column mapping from headers
        if len(headers) < 3:
            return

        # Header row 2: bodyparts, Header row 3: coords
        bodyparts_row = headers[1]
        coords_row = headers[2]

        col_map = {}
        for i, (bp, coord) in enumerate(zip(bodyparts_row, coords_row)):
            if bp and coord:
                col_map[(bp.strip(), coord.strip())] = i

        for row in reader:
            if not row:
                continue
            try:
                frame = int(row[0])
            except (ValueError, IndexError):
                continue
            if frame >= n_frames:
                break

            for bp, target_OS, target_OD, jname in [
                ("thumb", thumb_OS, thumb_OD, "thumb"),
                ("index", index_OS, index_OD, "index"),
            ]:
                for side, target in [("OS", target_OS), ("OD", target_OD)]:
                    x_key = (f"{bp}_{side}" if (f"{bp}_{side}", "x") in col_map
                             else (bp, "x"))
                    y_key = (f"{bp}_{side}" if (f"{bp}_{side}", "y") in col_map
                             else (bp, "y"))
                    # Try various column naming conventions
                    for xk, yk in [(x_key, y_key)]:
                        xi = col_map.get((xk[0] if isinstance(xk, tuple) else xk, "x"))
                        yi = col_map.get((xk[0] if isinstance(xk, tuple) else xk, "y"))
                        if xi is not None and yi is not None:
                            try:
                                target[frame] = [float(row[xi]), float(row[yi])]
                            except (ValueError, IndexError):
                                pass


# ── Heatmap serving ────────────────────────────────────────────────────

# Keep mmap reference alive to avoid re-opening
_heatmap_cache: dict[str, np.ndarray] = {}


def get_heatmap(subject_name: str, trial_stem: str,
                frame: int, joint: int, side: str) -> dict | None:
    """Load a single 64x64 heatmap for one joint at one frame.

    Uses memory-mapped access to avoid loading the entire ~350MB file.
    Returns dict with 'heatmap' (64x64 list), 'bbox' [x1,y1,x2,y2], 'max_val'.
    """
    mano_trial_dir = _mano_dir(subject_name) / trial_stem
    hm_path = mano_trial_dir / "hrnet_w18_heatmaps.npz"
    crop_path = mano_trial_dir / "hand_crop.json"

    if not hm_path.exists() or not crop_path.exists():
        return None

    # Load crop info
    with open(crop_path) as f:
        crop_info = json.load(f)

    if side == "OS":
        bbox = crop_info.get("crop_L")
        hm_key = "heatmaps_L"
    else:
        bbox = crop_info.get("crop_R")
        hm_key = "heatmaps_R"

    if bbox is None:
        return None

    # Memory-map the heatmap file
    cache_key = str(hm_path)
    if cache_key not in _heatmap_cache:
        try:
            _heatmap_cache[cache_key] = np.load(str(hm_path), mmap_mode="r")
        except Exception as e:
            logger.error(f"Failed to load heatmaps: {e}")
            return None

    hm_data = _heatmap_cache[cache_key]

    # Try the side-specific key, fall back to generic 'heatmaps'
    if hm_key in hm_data:
        hm_arr = hm_data[hm_key]
    elif "heatmaps" in hm_data:
        hm_arr = hm_data["heatmaps"]
    else:
        # List available keys for debugging
        logger.warning(f"No heatmap arrays found in {hm_path}. Keys: {list(hm_data.keys())}")
        return None

    if frame >= hm_arr.shape[0] or joint >= hm_arr.shape[1]:
        return None

    hm = hm_arr[frame, joint].astype(np.float32)
    max_val = float(hm.max())

    # Normalise to [0, 1]
    if max_val > 0:
        hm = hm / max_val

    return {
        "heatmap": [[round(float(hm[r, c]), 4) for c in range(hm.shape[1])]
                     for r in range(hm.shape[0])],
        "bbox": bbox,
        "max_val": round(max_val, 4),
    }
