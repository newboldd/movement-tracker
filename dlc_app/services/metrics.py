"""Video-based metric computation and Strategy G event detection.

Ported from scratch/evaluate_peak_detection.py — computes reversal and SSD
motion metrics from video crops around finger tips, and uses them to refine
distance-based event detection.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from ..config import get_settings
from .dlc_predictions import get_corrections_with_dlc_fallback, get_dlc_predictions_for_stage

logger = logging.getLogger(__name__)

CROP_HALF = 24  # 48x48 px crop around finger tips
BODYPARTS = ["index", "thumb"]


# ── Helpers ──────────────────────────────────────────────────────────────────


def msd(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared difference between two arrays."""
    if a.shape != b.shape:
        return float("inf")
    diff = a.astype(np.float64) - b.astype(np.float64)
    return float(np.sum(diff * diff) / diff.size)


def gaussian_smooth(arr: np.ndarray, sigma: float) -> np.ndarray:
    """1D Gaussian smoothing."""
    radius = int(3 * sigma)
    if radius < 1:
        return arr.copy()
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(arr, kernel, mode="same")


# ── Metric computation ───────────────────────────────────────────────────────


def compute_trial_metrics(subject_name: str, trial: dict, cam_names: list[str]) -> dict:
    """Compute distance, reversal, and SSD motion metrics for one trial.

    Args:
        subject_name: Subject identifier.
        trial: Single trial dict from build_trial_map() with video_path, start_frame, etc.
        cam_names: Camera names (e.g. ["OS", "OD"]).

    Returns:
        {distance, reversal, motion_ssd, per_cam_ssd, n_frames} with values as lists.
    """
    sf = trial["start_frame"]
    ef = trial["end_frame"]
    n_frames = trial["frame_count"]
    vpath = trial["video_path"]

    # Load coordinates for crop extraction
    stage_data = get_corrections_with_dlc_fallback(subject_name)
    if stage_data is None:
        raise RuntimeError("No coordinate data found")

    bp_coords: dict[str, dict[str, list | None]] = {}
    for bp in BODYPARTS:
        bp_coords[bp] = {}
        for cam in cam_names:
            if cam in stage_data and bp in stage_data[cam]:
                bp_coords[bp][cam] = stage_data[cam][bp]
            else:
                bp_coords[bp][cam] = None

    # Load distance trace
    dlc_data = None
    for stage in ("corrections", "refine", "dlc"):
        dlc_data = get_dlc_predictions_for_stage(subject_name, stage)
        if dlc_data and dlc_data.get("distances"):
            break
    if not dlc_data or not dlc_data.get("distances"):
        raise RuntimeError("No distance data found")

    dist_raw = dlc_data["distances"][sf : ef + 1]
    distance = [float(d) if d is not None else 0.0 for d in dist_raw]

    # Extract crops from video
    logger.info("Extracting crops from %s (%d bodyparts)…", trial["trial_name"], len(BODYPARTS))
    cap = cv2.VideoCapture(vpath)
    crops: dict[tuple[int, str, str], np.ndarray | None] = {}

    for local_idx in range(n_frames):
        global_idx = sf + local_idx
        ret, frame_bgr = cap.read()
        if not ret:
            break

        h, w = frame_bgr.shape[:2]
        midline = w // 2

        for cam_idx, cam in enumerate(cam_names):
            cam_frame = frame_bgr[:, :midline] if cam_idx == 0 else frame_bgr[:, midline:]

            for bp in BODYPARTS:
                coords_list = bp_coords[bp][cam]
                if coords_list is None or global_idx >= len(coords_list):
                    crops[(local_idx, cam, bp)] = None
                    continue

                pos = coords_list[global_idx]
                if pos is None:
                    crops[(local_idx, cam, bp)] = None
                    continue

                x, y = int(round(pos[0])), int(round(pos[1]))
                x_min = max(0, x - CROP_HALF)
                y_min = max(0, y - CROP_HALF)
                x_max = min(cam_frame.shape[1], x + CROP_HALF)
                y_max = min(cam_frame.shape[0], y + CROP_HALF)

                if x_max <= x_min or y_max <= y_min:
                    crops[(local_idx, cam, bp)] = None
                else:
                    crops[(local_idx, cam, bp)] = (
                        cv2.cvtColor(cam_frame[y_min:y_max, x_min:x_max], cv2.COLOR_BGR2GRAY).astype(
                            np.float32
                        )
                    )

    cap.release()

    def _compute_reversal(cam_filter=None):
        """Reversal metric: msd(frame[i-1], frame[i+1])."""
        arr = np.zeros(n_frames, dtype=np.float64)
        cams = [cam_filter] if cam_filter else cam_names
        for local_idx in range(1, n_frames - 1):
            metric_sum = 0.0
            n_valid = 0
            for cam in cams:
                for bp in BODYPARTS:
                    prev = crops.get((local_idx - 1, cam, bp))
                    nxt = crops.get((local_idx + 1, cam, bp))
                    if prev is not None and nxt is not None and prev.shape == nxt.shape:
                        metric_sum += msd(prev, nxt)
                        n_valid += 1
            if n_valid > 0:
                arr[local_idx] = metric_sum / n_valid
        return arr

    def _compute_motion_ssd(cam_filter=None):
        """SSD motion metric: msd(frame[i], frame[i-1])."""
        arr = np.zeros(n_frames, dtype=np.float64)
        cams = [cam_filter] if cam_filter else cam_names
        for local_idx in range(1, n_frames):
            metric_sum = 0.0
            n_valid = 0
            for cam in cams:
                for bp in BODYPARTS:
                    curr = crops.get((local_idx, cam, bp))
                    prev = crops.get((local_idx - 1, cam, bp))
                    if curr is not None and prev is not None and curr.shape == prev.shape:
                        metric_sum += msd(curr, prev)
                        n_valid += 1
            if n_valid > 0:
                arr[local_idx] = metric_sum / n_valid
        return arr

    # Compute combined metrics
    reversal = _compute_reversal()
    motion_ssd = _compute_motion_ssd()

    # Compute per-camera SSD
    per_cam_ssd = {}
    for cam in cam_names:
        per_cam_ssd[cam] = _compute_motion_ssd(cam_filter=cam)

    return {
        "distance": distance,
        "reversal": reversal.tolist(),
        "motion_ssd": motion_ssd.tolist(),
        "per_cam_ssd": {cam: arr.tolist() for cam, arr in per_cam_ssd.items()},
        "n_frames": n_frames,
    }


# ── Detection helpers ────────────────────────────────────────────────────────


def _clean_and_prep_distance(
    dist: list,
    min_event_gap: int = 10,
    min_peak_height: float = 15.0,
    valley_thresh: float = 25.0,
) -> tuple[list[float], list[float], list[bool], list[int]]:
    """Shared distance preprocessing: clean, interpolate, find peaks.

    Returns (d, ddist1, nan_mask, merged_peaks).
    """
    n = len(dist)
    max_valid_dist = 200.0
    dist_clean: list[float | None] = []
    for d_val in dist:
        if d_val is None or not (0 <= float(d_val) <= max_valid_dist):
            dist_clean.append(None)
        else:
            dist_clean.append(float(d_val))

    nan_mask = [v is None for v in dist_clean]

    # Linear interpolation over None runs
    for i in range(n):
        if dist_clean[i] is not None:
            continue
        left_v = left_i = None
        for j in range(i - 1, -1, -1):
            if dist_clean[j] is not None:
                left_v, left_i = dist_clean[j], j
                break
        right_v = right_i = None
        for j in range(i + 1, n):
            if dist_clean[j] is not None:
                right_v, right_i = dist_clean[j], j
                break
        if left_v is not None and right_v is not None:
            t = (i - left_i) / (right_i - left_i)
            dist_clean[i] = left_v + t * (right_v - left_v)
        elif left_v is not None:
            dist_clean[i] = left_v
        elif right_v is not None:
            dist_clean[i] = right_v
        else:
            dist_clean[i] = 0.0

    d = [v if v is not None else 0.0 for v in dist_clean]
    ddist1 = [0.0] * n
    for i in range(1, n):
        ddist1[i] = d[i] - d[i - 1]

    # Find peaks
    half_win = max(min_event_gap // 2, 10)
    peak_candidates: list[int] = []
    for i in range(half_win, n - half_win):
        if d[i] < min_peak_height:
            continue
        if any(nan_mask[i - half_win : i + half_win + 1]):
            continue
        if d[i] >= max(d[i - half_win : i + half_win + 1]):
            peak_candidates.append(i)

    # Merge nearby candidates
    merged_peaks: list[int] = []
    if peak_candidates:
        group = [peak_candidates[0]]
        for k in range(1, len(peak_candidates)):
            if peak_candidates[k] - peak_candidates[k - 1] < min_event_gap:
                group.append(peak_candidates[k])
            else:
                merged_peaks.append(max(group, key=lambda f: d[f]))
                group = [peak_candidates[k]]
        merged_peaks.append(max(group, key=lambda f: d[f]))

    # Valley filter
    filtered: list[int] = []
    for pk in merged_peaks:
        if not filtered:
            filtered.append(pk)
            continue
        valley_min = min(d[filtered[-1] : pk + 1])
        if valley_min < valley_thresh:
            filtered.append(pk)
        elif d[pk] > d[filtered[-1]]:
            filtered[-1] = pk
    merged_peaks = filtered

    return d, ddist1, nan_mask, merged_peaks


def _ssd_refine_open(
    sm_ssd: np.ndarray,
    d: list[float],
    open_frame_dist: int,
    search_start: int,
    pk: int,
    ssd_search_radius: int,
    open_bias: int,
) -> tuple[int, float]:
    """Find SSD-refined open frame. Returns (frame, distance_at_frame)."""
    n = len(d)
    ssd_lo = max(search_start, open_frame_dist - ssd_search_radius)
    ssd_hi = min(pk - 1, open_frame_dist + ssd_search_radius)
    if ssd_hi > ssd_lo:
        ssd_window = sm_ssd[ssd_lo : ssd_hi + 1]
        if len(ssd_window) > 1:
            ssd_diff = np.diff(ssd_window)
            onset_idx = int(np.argmax(ssd_diff))
            frame = max(0, min(n - 1, ssd_lo + onset_idx + open_bias))
            return frame, d[frame]
    return open_frame_dist, d[open_frame_dist]


def _ssd_refine_close(
    sm_ssd: np.ndarray,
    d: list[float],
    close_frame_dist: int,
    pk: int,
    next_pk_or_n: int,
    ssd_search_radius: int,
    close_bias: int,
) -> tuple[int, float]:
    """Find SSD-refined close frame. Returns (frame, distance_at_frame)."""
    n = len(d)
    ssd_lo = max(pk + 1, close_frame_dist - ssd_search_radius)
    ssd_hi = min(next_pk_or_n - 1, close_frame_dist + ssd_search_radius)
    if ssd_hi > ssd_lo:
        ssd_window = sm_ssd[ssd_lo : ssd_hi + 1]
        if len(ssd_window) > 1:
            ssd_diff = np.diff(ssd_window)
            cessation_idx = int(np.argmin(ssd_diff))
            frame = max(0, min(n - 1, ssd_lo + cessation_idx + close_bias))
            return frame, d[frame]
    return close_frame_dist, d[close_frame_dist]


# ── Strategy G detection ─────────────────────────────────────────────────────


def detect_strategy_g(
    dist: list[float],
    reversal: list[float] | None,
    motion_ssd: list[float] | None,
    per_cam_ssd: dict[str, list[float]] | None,
    cam_names: list[str] | None,
    start_frame: int,
    *,
    steps: dict,
    params: dict,
) -> dict[str, list[int]]:
    """Configurable Strategy G: multi-metric event detection.

    steps: {use_reversal, use_ssd, use_dist_guard, use_peak_guard}
    params: {min_peak_height, min_event_gap, open_start_thresh, valley_thresh, nback,
             reversal_search_radius, ssd_search_radius, open_bias, close_bias,
             dist_guard_factor, peak_guard_factor}

    Returns {open, peak, close} as GLOBAL frame numbers.
    """
    min_peak_height = float(params.get("min_peak_height", 15.0))
    min_event_gap = int(params.get("min_event_gap", 10))
    open_start_thresh = float(params.get("open_start_thresh", 5.0))
    valley_thresh = float(params.get("valley_thresh", 25.0))
    nback = int(params.get("nback", 60))
    reversal_search_radius = int(params.get("reversal_search_radius", 3))
    ssd_search_radius = int(params.get("ssd_search_radius", 5))
    open_bias = int(params.get("open_bias", -1))
    close_bias = int(params.get("close_bias", 1))
    dist_guard_factor = float(params.get("dist_guard_factor", 0.25))
    peak_guard_factor = float(params.get("peak_guard_factor", 0.1))

    use_reversal = steps.get("use_reversal", True) and reversal is not None
    use_ssd = steps.get("use_ssd", True) and motion_ssd is not None
    use_dist_guard = steps.get("use_dist_guard", True) and use_ssd
    use_peak_guard = steps.get("use_peak_guard", True) and use_reversal

    n = len(dist)
    d, ddist1, nan_mask, merged_peaks = _clean_and_prep_distance(
        dist, min_event_gap, min_peak_height, valley_thresh
    )

    # Refine peaks with reversal metric
    refined_peaks = []
    if use_reversal:
        rev = np.array(reversal, dtype=np.float64)
        for pk in merged_peaks:
            search_lo = max(0, pk - reversal_search_radius)
            search_hi = min(n - 1, pk + reversal_search_radius)
            candidates = list(range(search_lo, search_hi + 1))
            valid = [f for f in candidates if rev[f] > 0]
            if valid:
                refined = min(valid, key=lambda f: float(rev[f]))
                if use_peak_guard and peak_guard_factor > 0 and d[refined] < d[pk] * (1 - peak_guard_factor):
                    refined_peaks.append(pk)
                else:
                    refined_peaks.append(refined)
            else:
                refined_peaks.append(pk)
    else:
        refined_peaks = list(merged_peaks)

    # Prepare SSD arrays
    sm_ssd = None
    sm_ssd_per_cam: dict[str, np.ndarray] = {}
    if use_ssd:
        sm_ssd = gaussian_smooth(np.array(motion_ssd, dtype=np.float64), 2.0)
        if per_cam_ssd and cam_names:
            for cam in cam_names:
                if cam in per_cam_ssd:
                    sm_ssd_per_cam[cam] = gaussian_smooth(np.array(per_cam_ssd[cam], dtype=np.float64), 2.0)

    opens = []
    closes = []
    for idx, pk in enumerate(refined_peaks):
        search_start = closes[-1] if closes else max(0, pk - nback)
        search_start = min(search_start, pk)

        # Distance-based open/close (fallback or primary)
        valley = min(range(search_start, pk + 1), key=lambda f: d[f])
        open_frame_dist = valley
        for k in range(valley, pk):
            if ddist1[k] > open_start_thresh:
                open_frame_dist = k - 1
                break
        open_frame_dist = max(search_start, open_frame_dist)

        next_pk = refined_peaks[idx + 1] if idx + 1 < len(refined_peaks) else n
        close_frame_dist = min(range(pk, min(next_pk, n)), key=lambda f: d[f])

        # SSD-refined open
        if use_ssd and sm_ssd is not None:
            open_frame, open_dist = _ssd_refine_open(
                sm_ssd, d, open_frame_dist, search_start, pk, ssd_search_radius, open_bias
            )
            # Distance guard
            if use_dist_guard and dist_guard_factor > 0:
                threshold = d[open_frame_dist] + dist_guard_factor * d[pk]
                if open_dist > threshold:
                    if sm_ssd_per_cam and cam_names:
                        best_cam_frame = open_frame_dist
                        best_cam_dist = d[open_frame_dist]
                        for cam in cam_names:
                            if cam in sm_ssd_per_cam:
                                cf, cd = _ssd_refine_open(
                                    sm_ssd_per_cam[cam], d, open_frame_dist, search_start, pk,
                                    ssd_search_radius, open_bias,
                                )
                                if cd < best_cam_dist:
                                    best_cam_frame = cf
                                    best_cam_dist = cd
                        if best_cam_dist <= threshold:
                            open_frame = best_cam_frame
                        else:
                            open_frame = open_frame_dist
                    else:
                        open_frame = open_frame_dist
        else:
            open_frame = open_frame_dist

        opens.append(max(0, min(n - 1, open_frame)))

        # SSD-refined close
        if use_ssd and sm_ssd is not None:
            close_frame, close_dist = _ssd_refine_close(
                sm_ssd, d, close_frame_dist, pk, min(next_pk, n), ssd_search_radius, close_bias
            )
            # Distance guard
            if use_dist_guard and dist_guard_factor > 0:
                threshold = d[close_frame_dist] + dist_guard_factor * d[pk]
                if close_dist > threshold:
                    if sm_ssd_per_cam and cam_names:
                        best_cam_frame = close_frame_dist
                        best_cam_dist = d[close_frame_dist]
                        for cam in cam_names:
                            if cam in sm_ssd_per_cam:
                                cf, cd = _ssd_refine_close(
                                    sm_ssd_per_cam[cam], d, close_frame_dist, pk,
                                    min(next_pk, n), ssd_search_radius, close_bias,
                                )
                                if cd < best_cam_dist:
                                    best_cam_frame = cf
                                    best_cam_dist = cd
                        if best_cam_dist <= threshold:
                            close_frame = best_cam_frame
                        else:
                            close_frame = close_frame_dist
                    else:
                        close_frame = close_frame_dist
        else:
            close_frame = close_frame_dist

        closes.append(max(0, min(n - 1, close_frame)))

    return {
        "open": [start_frame + f for f in opens],
        "peak": [start_frame + f for f in refined_peaks],
        "close": [start_frame + f for f in closes],
    }
