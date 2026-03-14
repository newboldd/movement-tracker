"""Evaluate event detection strategies against saved events for MSA02_L1.

Compares:
  A) Distance-only (baseline, replicates existing algorithm)
  B) Distance + reversal metric refinement for peaks
  C) Distance + reversal + SSD motion metric for all events

Reports per-event frame errors and generates comparison plots.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dlc_app.config import get_settings
from dlc_app.services.video import build_trial_map
from dlc_app.services.dlc_predictions import (
    get_corrections_with_dlc_fallback,
    get_dlc_predictions_for_stage,
)

# ── Config ──────────────────────────────────────────────────────────────────

SUBJECT = "MSA02"
TRIAL_NAME = "MSA02_L1"
CROP_HALF = 24  # 48×48 px crop around finger tips
BODYPARTS = ["index", "thumb"]  # both fingers for more robust metrics
OUTPUT_DIR = Path("scratch/peak_metric_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_events(subject_name: str) -> dict[str, list[int]]:
    settings = get_settings()
    events_path = Path(settings.dlc_path) / subject_name / "events.csv"
    events: dict[str, list[int]] = {"open": [], "peak": [], "close": []}
    if not events_path.exists():
        return events
    with open(events_path, newline="") as f:
        for row in csv.DictReader(f):
            etype = row["event_type"].strip()
            if etype in events:
                events[etype].append(int(row["frame_num"]))
    for v in events.values():
        v.sort()
    return events


def msd(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float('inf')
    diff = a - b
    return float(np.sum(diff * diff) / diff.size)


def gaussian_smooth(arr: np.ndarray, sigma: float) -> np.ndarray:
    radius = int(3 * sigma)
    if radius < 1:
        return arr.copy()
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(arr, kernel, mode="same")


# ── Metric computation ───────────────────────────────────────────────────────

def compute_all_metrics(subject_name: str, trial: dict) -> dict:
    """Compute distance trace, reversal metric, and SSD motion metric.

    Extracts crops for both index and thumb bodyparts, computes per-camera
    and per-bodypart metrics, then returns combined (averaged) and per-camera
    variants so strategies can choose the best signal.
    """
    settings = get_settings()
    cam_names = settings.camera_names

    sf = trial["start_frame"]
    ef = trial["end_frame"]
    n_frames = trial["frame_count"]
    vpath = trial["video_path"]

    # Load coordinates for crop extraction (both index and thumb)
    stage_data = get_corrections_with_dlc_fallback(subject_name)
    if stage_data is None:
        raise RuntimeError("No coordinate data found")

    bp_coords: dict[str, dict[str, list | None]] = {}  # bp_coords[bp][cam]
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

    dist_raw = dlc_data["distances"][sf:ef + 1]
    distance = np.array([d if d is not None else 0.0 for d in dist_raw], dtype=np.float64)

    # Extract crops from video — keyed by (local_idx, cam, bp)
    print(f"  Extracting crops from {trial['trial_name']} ({len(BODYPARTS)} bodyparts)…")
    cap = cv2.VideoCapture(vpath)
    # crops[(local_idx, cam, bp)] = grayscale crop or None
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
                        cv2.cvtColor(cam_frame[y_min:y_max, x_min:x_max], cv2.COLOR_BGR2GRAY)
                        .astype(np.float32)
                    )

    cap.release()

    def _compute_reversal(cam_filter=None, bp_filter=None):
        """Reversal metric (offset=1): msd(frame[i-1], frame[i+1]).

        cam_filter: None=all cameras, or a specific cam name.
        bp_filter: None=all bodyparts, or a specific bodypart name.
        """
        arr = np.zeros(n_frames, dtype=np.float64)
        cams = [cam_filter] if cam_filter else cam_names
        bps = [bp_filter] if bp_filter else BODYPARTS
        for local_idx in range(1, n_frames - 1):
            metric_sum = 0.0
            n_valid = 0
            for cam in cams:
                for bp in bps:
                    prev = crops.get((local_idx - 1, cam, bp))
                    nxt = crops.get((local_idx + 1, cam, bp))
                    if prev is not None and nxt is not None and prev.shape == nxt.shape:
                        metric_sum += msd(prev, nxt)
                        n_valid += 1
            if n_valid > 0:
                arr[local_idx] = metric_sum / n_valid
        return arr

    def _compute_motion_ssd(cam_filter=None, bp_filter=None):
        """SSD motion metric: msd(frame[i], frame[i-1])."""
        arr = np.zeros(n_frames, dtype=np.float64)
        cams = [cam_filter] if cam_filter else cam_names
        bps = [bp_filter] if bp_filter else BODYPARTS
        for local_idx in range(1, n_frames):
            metric_sum = 0.0
            n_valid = 0
            for cam in cams:
                for bp in bps:
                    curr = crops.get((local_idx, cam, bp))
                    prev = crops.get((local_idx - 1, cam, bp))
                    if curr is not None and prev is not None and curr.shape == prev.shape:
                        metric_sum += msd(curr, prev)
                        n_valid += 1
            if n_valid > 0:
                arr[local_idx] = metric_sum / n_valid
        return arr

    # Compute combined metrics (all cameras, all bodyparts)
    print(f"  Computing reversal metric (all cams, all bodyparts)…")
    reversal = _compute_reversal()
    print(f"  Computing SSD motion metric (all cams, all bodyparts)…")
    motion_ssd = _compute_motion_ssd()

    # Compute per-camera metrics (all bodyparts, single camera)
    per_cam_reversal = {}
    per_cam_ssd = {}
    for cam in cam_names:
        print(f"  Computing per-camera metrics for {cam}…")
        per_cam_reversal[cam] = _compute_reversal(cam_filter=cam)
        per_cam_ssd[cam] = _compute_motion_ssd(cam_filter=cam)

    return {
        'distance': distance,
        'reversal': reversal,
        'motion_ssd': motion_ssd,
        'per_cam_reversal': per_cam_reversal,
        'per_cam_ssd': per_cam_ssd,
        'cam_names': cam_names,
        'n_frames': n_frames,
        'sf': sf,
        'ef': ef,
    }


# ── Detection strategies ─────────────────────────────────────────────────────

def detect_distance_only(dist: np.ndarray, sf: int,
                         min_peak_height: float = 15.0,
                         min_event_gap: int = 10,
                         open_start_thresh: float = 5.0,
                         valley_thresh: float = 25.0,
                         nback: int = 60) -> dict[str, list[int]]:
    """Strategy A: Distance-only (replicates existing algorithm)."""
    n = len(dist)

    # Clean and interpolate
    max_valid_dist = 200.0
    dist_clean = []
    for d in dist:
        if d is None or not (0 <= float(d) <= max_valid_dist):
            dist_clean.append(None)
        else:
            dist_clean.append(float(d))

    nan_mask = [v is None for v in dist_clean]

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

    # 1-frame velocity
    ddist1 = [0.0] * n
    for i in range(1, n):
        ddist1[i] = d[i] - d[i - 1]

    # Find peaks
    half_win = max(min_event_gap // 2, 10)
    peak_candidates = []
    for i in range(half_win, n - half_win):
        if d[i] < min_peak_height:
            continue
        if any(nan_mask[i - half_win: i + half_win + 1]):
            continue
        if d[i] >= max(d[i - half_win: i + half_win + 1]):
            peak_candidates.append(i)

    # Merge nearby
    merged_peaks = []
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
    filtered = []
    for pk in merged_peaks:
        if not filtered:
            filtered.append(pk)
            continue
        valley_min = min(d[filtered[-1]: pk + 1])
        if valley_min < valley_thresh:
            filtered.append(pk)
        elif d[pk] > d[filtered[-1]]:
            filtered[-1] = pk
    merged_peaks = filtered

    # Derive opens and closes
    opens = []
    closes = []
    for idx, pk in enumerate(merged_peaks):
        search_start = closes[-1] if closes else max(0, pk - nback)
        valley = min(range(search_start, pk + 1), key=lambda f: d[f])
        open_frame = valley
        for k in range(valley, pk):
            if ddist1[k] > open_start_thresh:
                open_frame = k - 1
                break
        opens.append(max(search_start, open_frame))

        next_pk = merged_peaks[idx + 1] if idx + 1 < len(merged_peaks) else n
        closes.append(min(range(pk, min(next_pk, n)), key=lambda f: d[f]))

    return {
        "open": [sf + f for f in opens],
        "peak": [sf + f for f in merged_peaks],
        "close": [sf + f for f in closes],
    }


def detect_with_reversal(dist: np.ndarray, reversal: np.ndarray, sf: int,
                         min_peak_height: float = 15.0,
                         min_event_gap: int = 10,
                         open_start_thresh: float = 5.0,
                         valley_thresh: float = 25.0,
                         nback: int = 60,
                         reversal_search_radius: int = 3) -> dict[str, list[int]]:
    """Strategy B: Distance peaks refined by reversal metric minimum."""
    # First run distance-only to get coarse peaks
    n = len(dist)

    max_valid_dist = 200.0
    dist_clean = []
    for d_val in dist:
        if d_val is None or not (0 <= float(d_val) <= max_valid_dist):
            dist_clean.append(None)
        else:
            dist_clean.append(float(d_val))

    nan_mask = [v is None for v in dist_clean]

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

    half_win = max(min_event_gap // 2, 10)
    peak_candidates = []
    for i in range(half_win, n - half_win):
        if d[i] < min_peak_height:
            continue
        if any(nan_mask[i - half_win: i + half_win + 1]):
            continue
        if d[i] >= max(d[i - half_win: i + half_win + 1]):
            peak_candidates.append(i)

    merged_peaks = []
    if peak_candidates:
        group = [peak_candidates[0]]
        for k in range(1, len(peak_candidates)):
            if peak_candidates[k] - peak_candidates[k - 1] < min_event_gap:
                group.append(peak_candidates[k])
            else:
                merged_peaks.append(max(group, key=lambda f: d[f]))
                group = [peak_candidates[k]]
        merged_peaks.append(max(group, key=lambda f: d[f]))

    filtered = []
    for pk in merged_peaks:
        if not filtered:
            filtered.append(pk)
            continue
        valley_min = min(d[filtered[-1]: pk + 1])
        if valley_min < valley_thresh:
            filtered.append(pk)
        elif d[pk] > d[filtered[-1]]:
            filtered[-1] = pk
    merged_peaks = filtered

    # REFINEMENT: For each coarse peak, find the local minimum in the reversal
    # metric within ±reversal_search_radius frames. The reversal minimum is
    # where frame[i-1] and frame[i+1] are most similar = exact reversal point.
    refined_peaks = []
    for pk in merged_peaks:
        search_lo = max(0, pk - reversal_search_radius)
        search_hi = min(n - 1, pk + reversal_search_radius)
        # Only refine if reversal data exists in the window
        candidates = list(range(search_lo, search_hi + 1))
        valid = [f for f in candidates if reversal[f] > 0]
        if valid:
            refined = min(valid, key=lambda f: reversal[f])
            refined_peaks.append(refined)
        else:
            refined_peaks.append(pk)

    # Derive opens and closes using refined peaks
    opens = []
    closes = []
    for idx, pk in enumerate(refined_peaks):
        search_start = closes[-1] if closes else max(0, pk - nback)
        valley = min(range(search_start, pk + 1), key=lambda f: d[f])
        open_frame = valley
        for k in range(valley, pk):
            if ddist1[k] > open_start_thresh:
                open_frame = k - 1
                break
        opens.append(max(search_start, open_frame))

        next_pk = refined_peaks[idx + 1] if idx + 1 < len(refined_peaks) else n
        closes.append(min(range(pk, min(next_pk, n)), key=lambda f: d[f]))

    return {
        "open": [sf + f for f in opens],
        "peak": [sf + f for f in refined_peaks],
        "close": [sf + f for f in closes],
    }


def _clean_and_prep_distance(dist, min_event_gap=10, min_peak_height=15.0, valley_thresh=25.0):
    """Shared distance preprocessing: clean, interpolate, find peaks."""
    n = len(dist)
    max_valid_dist = 200.0
    dist_clean = []
    for d_val in dist:
        if d_val is None or not (0 <= float(d_val) <= max_valid_dist):
            dist_clean.append(None)
        else:
            dist_clean.append(float(d_val))

    nan_mask = [v is None for v in dist_clean]
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

    half_win = max(min_event_gap // 2, 10)
    peak_candidates = []
    for i in range(half_win, n - half_win):
        if d[i] < min_peak_height:
            continue
        if any(nan_mask[i - half_win: i + half_win + 1]):
            continue
        if d[i] >= max(d[i - half_win: i + half_win + 1]):
            peak_candidates.append(i)

    merged_peaks = []
    if peak_candidates:
        group = [peak_candidates[0]]
        for k in range(1, len(peak_candidates)):
            if peak_candidates[k] - peak_candidates[k - 1] < min_event_gap:
                group.append(peak_candidates[k])
            else:
                merged_peaks.append(max(group, key=lambda f: d[f]))
                group = [peak_candidates[k]]
        merged_peaks.append(max(group, key=lambda f: d[f]))

    filtered = []
    for pk in merged_peaks:
        if not filtered:
            filtered.append(pk)
            continue
        valley_min = min(d[filtered[-1]: pk + 1])
        if valley_min < valley_thresh:
            filtered.append(pk)
        elif d[pk] > d[filtered[-1]]:
            filtered[-1] = pk
    merged_peaks = filtered

    return d, ddist1, nan_mask, merged_peaks


def _ssd_refine_open(sm_ssd, d, open_frame_dist, search_start, pk, ssd_search_radius, open_bias):
    """Find SSD-refined open frame. Returns (frame, distance_at_frame)."""
    n = len(d)
    ssd_lo = max(search_start, open_frame_dist - ssd_search_radius)
    ssd_hi = min(pk - 1, open_frame_dist + ssd_search_radius)
    if ssd_hi > ssd_lo:
        ssd_window = sm_ssd[ssd_lo:ssd_hi + 1]
        if len(ssd_window) > 1:
            ssd_diff = np.diff(ssd_window)
            onset_idx = np.argmax(ssd_diff)
            frame = max(0, min(n - 1, ssd_lo + onset_idx + open_bias))
            return frame, d[frame]
    return open_frame_dist, d[open_frame_dist]


def _ssd_refine_close(sm_ssd, d, close_frame_dist, pk, next_pk_or_n, ssd_search_radius, close_bias):
    """Find SSD-refined close frame. Returns (frame, distance_at_frame)."""
    n = len(d)
    ssd_lo = max(pk + 1, close_frame_dist - ssd_search_radius)
    ssd_hi = min(next_pk_or_n - 1, close_frame_dist + ssd_search_radius)
    if ssd_hi > ssd_lo:
        ssd_window = sm_ssd[ssd_lo:ssd_hi + 1]
        if len(ssd_window) > 1:
            ssd_diff = np.diff(ssd_window)
            cessation_idx = np.argmin(ssd_diff)
            frame = max(0, min(n - 1, ssd_lo + cessation_idx + close_bias))
            return frame, d[frame]
    return close_frame_dist, d[close_frame_dist]


def detect_with_all_metrics(dist: np.ndarray, reversal: np.ndarray,
                            motion_ssd: np.ndarray, sf: int,
                            per_cam_ssd: dict | None = None,
                            cam_names: list[str] | None = None,
                            min_peak_height: float = 15.0,
                            min_event_gap: int = 10,
                            open_start_thresh: float = 5.0,
                            valley_thresh: float = 25.0,
                            nback: int = 60,
                            reversal_search_radius: int = 3,
                            ssd_search_radius: int = 5,
                            open_bias: int = 0,
                            close_bias: int = 0,
                            dist_guard_factor: float = 0.0,
                            peak_guard_factor: float = 0.0,
                            ) -> dict[str, list[int]]:
    """Strategy C/D: Distance + reversal for peaks, SSD for open/close refinement.

    Open: steepest SSD increase near the distance-based open (motion onset).
    Close: steepest SSD decrease near the distance-based close (motion cessation).

    dist_guard_factor: if > 0, reject SSD refinement when the distance at the
    refined frame exceeds the distance-based frame by more than this factor
    times the peak distance.  When per_cam_ssd is provided, falls back to
    trying each camera individually and picking the result with the lowest
    distance.

    peak_guard_factor: if > 0, reject reversal refinement for peaks when the
    distance at the refined frame drops below the original peak distance by
    more than this fraction.
    """
    n = len(dist)
    d, ddist1, nan_mask, merged_peaks = _clean_and_prep_distance(
        dist, min_event_gap, min_peak_height, valley_thresh)

    # Refine peaks with reversal metric (with distance guard)
    refined_peaks = []
    for pk in merged_peaks:
        search_lo = max(0, pk - reversal_search_radius)
        search_hi = min(n - 1, pk + reversal_search_radius)
        candidates = list(range(search_lo, search_hi + 1))
        valid = [f for f in candidates if reversal[f] > 0]
        if valid:
            refined = min(valid, key=lambda f: reversal[f])
            # Guard: reject if refined peak's distance drops too much
            if peak_guard_factor > 0 and d[refined] < d[pk] * (1 - peak_guard_factor):
                refined_peaks.append(pk)
            else:
                refined_peaks.append(refined)
        else:
            refined_peaks.append(pk)

    # Smooth SSD for onset/cessation detection — combined + per-camera
    sm_ssd = gaussian_smooth(motion_ssd, 2.0)
    sm_ssd_per_cam = {}
    if per_cam_ssd and cam_names:
        for cam in cam_names:
            sm_ssd_per_cam[cam] = gaussian_smooth(per_cam_ssd[cam], 2.0)

    opens = []
    closes = []
    for idx, pk in enumerate(refined_peaks):
        search_start = closes[-1] if closes else max(0, pk - nback)
        search_start = min(search_start, pk)

        # --- Distance-based open/close (fallback) ---
        valley = min(range(search_start, pk + 1), key=lambda f: d[f])
        open_frame_dist = valley
        for k in range(valley, pk):
            if ddist1[k] > open_start_thresh:
                open_frame_dist = k - 1
                break
        open_frame_dist = max(search_start, open_frame_dist)

        next_pk = refined_peaks[idx + 1] if idx + 1 < len(refined_peaks) else n
        close_frame_dist = min(range(pk, min(next_pk, n)), key=lambda f: d[f])

        # --- SSD-refined open ---
        open_frame, open_dist = _ssd_refine_open(
            sm_ssd, d, open_frame_dist, search_start, pk, ssd_search_radius, open_bias)

        # Distance guard: reject if SSD pushes open to high distance
        if dist_guard_factor > 0:
            threshold = d[open_frame_dist] + dist_guard_factor * d[pk]
            if open_dist > threshold:
                # Try per-camera fallback
                if sm_ssd_per_cam:
                    best_cam_frame = open_frame_dist
                    best_cam_dist = d[open_frame_dist]
                    for cam in cam_names:
                        cf, cd = _ssd_refine_open(
                            sm_ssd_per_cam[cam], d, open_frame_dist, search_start, pk,
                            ssd_search_radius, open_bias)
                        if cd < best_cam_dist:
                            best_cam_frame = cf
                            best_cam_dist = cd
                    if best_cam_dist <= threshold:
                        open_frame = best_cam_frame
                    else:
                        open_frame = open_frame_dist
                else:
                    open_frame = open_frame_dist
        opens.append(max(0, min(n - 1, open_frame)))

        # --- SSD-refined close ---
        close_frame, close_dist = _ssd_refine_close(
            sm_ssd, d, close_frame_dist, pk, min(next_pk, n), ssd_search_radius, close_bias)

        # Distance guard for close
        if dist_guard_factor > 0:
            threshold = d[close_frame_dist] + dist_guard_factor * d[pk]
            if close_dist > threshold:
                if sm_ssd_per_cam:
                    best_cam_frame = close_frame_dist
                    best_cam_dist = d[close_frame_dist]
                    for cam in cam_names:
                        cf, cd = _ssd_refine_close(
                            sm_ssd_per_cam[cam], d, close_frame_dist, pk,
                            min(next_pk, n), ssd_search_radius, close_bias)
                        if cd < best_cam_dist:
                            best_cam_frame = cf
                            best_cam_dist = cd
                    if best_cam_dist <= threshold:
                        close_frame = best_cam_frame
                    else:
                        close_frame = close_frame_dist
                else:
                    close_frame = close_frame_dist
        closes.append(max(0, min(n - 1, close_frame)))

    return {
        "open": [sf + f for f in opens],
        "peak": [sf + f for f in refined_peaks],
        "close": [sf + f for f in closes],
    }


def detect_with_all_metrics_v2(dist: np.ndarray, reversal: np.ndarray,
                                motion_ssd: np.ndarray, sf: int,
                                per_cam_reversal: dict | None = None,
                                per_cam_ssd: dict | None = None,
                                cam_names: list[str] | None = None,
                                min_peak_height: float = 15.0,
                                min_event_gap: int = 10,
                                open_start_thresh: float = 5.0,
                                valley_thresh: float = 25.0,
                                nback: int = 60,
                                reversal_search_radius: int = 3,
                                ssd_search_radius: int = 5,
                                open_bias: int = 0,
                                close_bias: int = 0,
                                use_reversal_for_close: bool = False,
                                close_reversal_radius: int = 3,
                                use_best_camera: bool = False,
                                ) -> dict[str, list[int]]:
    """Strategy E: All metrics + reversal for closes + per-camera best signal.

    New vs detect_with_all_metrics:
    - use_reversal_for_close: after SSD-based close, refine with reversal minimum
    - use_best_camera: pick the camera with highest variance (clearest signal)
      for each metric instead of averaging both cameras.
    """
    n = len(dist)
    d, ddist1, nan_mask, merged_peaks = _clean_and_prep_distance(
        dist, min_event_gap, min_peak_height, valley_thresh)

    # Optionally select best camera's metrics
    eff_reversal = reversal
    eff_ssd = motion_ssd
    if use_best_camera and per_cam_reversal and per_cam_ssd and cam_names:
        # Pick camera with highest variance for each metric
        rev_vars = {cam: np.var(per_cam_reversal[cam]) for cam in cam_names}
        best_rev_cam = max(rev_vars, key=rev_vars.get)
        eff_reversal = per_cam_reversal[best_rev_cam]

        ssd_vars = {cam: np.var(per_cam_ssd[cam]) for cam in cam_names}
        best_ssd_cam = max(ssd_vars, key=ssd_vars.get)
        eff_ssd = per_cam_ssd[best_ssd_cam]

    # Refine peaks with reversal metric
    refined_peaks = []
    for pk in merged_peaks:
        search_lo = max(0, pk - reversal_search_radius)
        search_hi = min(n - 1, pk + reversal_search_radius)
        candidates = list(range(search_lo, search_hi + 1))
        valid = [f for f in candidates if eff_reversal[f] > 0]
        if valid:
            refined = min(valid, key=lambda f: eff_reversal[f])
            refined_peaks.append(refined)
        else:
            refined_peaks.append(pk)

    # Smooth SSD for onset/cessation detection
    sm_ssd = gaussian_smooth(eff_ssd, 2.0)

    opens = []
    closes = []
    for idx, pk in enumerate(refined_peaks):
        search_start = closes[-1] if closes else max(0, pk - nback)
        search_start = min(search_start, pk)

        # --- Open detection: SSD onset ---
        valley = min(range(search_start, pk + 1), key=lambda f: d[f])
        open_frame_dist = valley
        for k in range(valley, pk):
            if ddist1[k] > open_start_thresh:
                open_frame_dist = k - 1
                break
        open_frame_dist = max(search_start, open_frame_dist)

        ssd_lo = max(search_start, open_frame_dist - ssd_search_radius)
        ssd_hi = min(pk - 1, open_frame_dist + ssd_search_radius)
        if ssd_hi > ssd_lo:
            ssd_window = sm_ssd[ssd_lo:ssd_hi + 1]
            if len(ssd_window) > 1:
                ssd_diff = np.diff(ssd_window)
                onset_idx = np.argmax(ssd_diff)
                open_frame = ssd_lo + onset_idx + open_bias
            else:
                open_frame = open_frame_dist
        else:
            open_frame = open_frame_dist
        opens.append(max(0, min(n - 1, open_frame)))

        # --- Close detection: SSD cessation (steepest decrease) ---
        next_pk = refined_peaks[idx + 1] if idx + 1 < len(refined_peaks) else n
        close_frame_dist = min(range(pk, min(next_pk, n)), key=lambda f: d[f])

        ssd_lo = max(pk + 1, close_frame_dist - ssd_search_radius)
        ssd_hi = min(min(next_pk, n) - 1, close_frame_dist + ssd_search_radius)
        if ssd_hi > ssd_lo:
            ssd_window = sm_ssd[ssd_lo:ssd_hi + 1]
            if len(ssd_window) > 1:
                ssd_diff = np.diff(ssd_window)
                cessation_idx = np.argmin(ssd_diff)
                close_frame = ssd_lo + cessation_idx + close_bias
            else:
                close_frame = close_frame_dist
        else:
            close_frame = close_frame_dist

        # --- Close reversal refinement (improvement #1) ---
        if use_reversal_for_close:
            r_lo = max(0, close_frame - close_reversal_radius)
            r_hi = min(n - 1, close_frame + close_reversal_radius)
            r_candidates = list(range(r_lo, r_hi + 1))
            r_valid = [f for f in r_candidates if eff_reversal[f] > 0]
            if r_valid:
                close_frame = min(r_valid, key=lambda f: eff_reversal[f])

        closes.append(max(0, min(n - 1, close_frame)))

    return {
        "open": [sf + f for f in opens],
        "peak": [sf + f for f in refined_peaks],
        "close": [sf + f for f in closes],
    }


# ── Evaluation ───────────────────────────────────────────────────────────────

def match_events(detected: list[int], saved: list[int],
                 max_match_dist: int = 15) -> list[tuple[int, int, int]]:
    """Match detected events to saved events greedily.

    Returns list of (saved_frame, detected_frame, error) tuples.
    Unmatched saved events get detected_frame=None.
    """
    det_used = set()
    matches = []

    for s in saved:
        best_d = None
        best_err = max_match_dist + 1
        for d in detected:
            if d in det_used:
                continue
            err = abs(d - s)
            if err < best_err:
                best_err = err
                best_d = d
        if best_d is not None and best_err <= max_match_dist:
            matches.append((s, best_d, best_d - s))
            det_used.add(best_d)
        else:
            matches.append((s, None, None))

    # Count false positives (detected but not matched)
    fp = len(detected) - len(det_used)
    return matches, fp


def print_eval(name: str, detected: dict, saved: dict, sf: int, ef: int):
    """Print evaluation summary for a detection strategy."""
    print(f"\n{'=' * 60}")
    print(f"  Strategy: {name}")
    print(f"{'=' * 60}")

    for etype in ["open", "peak", "close"]:
        saved_in_trial = [f for f in saved.get(etype, []) if sf <= f <= ef]
        det_events = detected.get(etype, [])

        matches, fp = match_events(det_events, saved_in_trial)

        matched = [(s, d, e) for s, d, e in matches if d is not None]
        missed = [(s, d, e) for s, d, e in matches if d is None]

        errors = [e for _, _, e in matched]

        print(f"\n  {etype.upper()}: saved={len(saved_in_trial)}, detected={len(det_events)}, "
              f"matched={len(matched)}, missed={len(missed)}, false_pos={fp}")

        if errors:
            abs_errors = [abs(e) for e in errors]
            print(f"    Error: mean={np.mean(errors):.1f}, "
                  f"abs_mean={np.mean(abs_errors):.1f}, "
                  f"median={np.median(errors):.0f}, "
                  f"abs_median={np.median(abs_errors):.0f}, "
                  f"max={max(abs_errors)}")
            # Distribution
            within_1 = sum(1 for e in abs_errors if e <= 1)
            within_2 = sum(1 for e in abs_errors if e <= 2)
            within_3 = sum(1 for e in abs_errors if e <= 3)
            within_5 = sum(1 for e in abs_errors if e <= 5)
            n_matched = len(errors)
            print(f"    Within ±1: {within_1}/{n_matched} ({100*within_1/n_matched:.0f}%), "
                  f"±2: {within_2}/{n_matched} ({100*within_2/n_matched:.0f}%), "
                  f"±3: {within_3}/{n_matched} ({100*within_3/n_matched:.0f}%), "
                  f"±5: {within_5}/{n_matched} ({100*within_5/n_matched:.0f}%)")

    return detected


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_comparison(metrics: dict, saved: dict, strategies: dict[str, dict],
                    out_dir: Path):
    """Plot distance trace with saved vs detected events for all strategies."""
    sf = metrics['sf']
    ef = metrics['ef']
    distance = metrics['distance']
    reversal = metrics['reversal']
    motion_ssd = metrics['motion_ssd']

    # Zoom to ~200 frames in the middle of the trial
    mid = len(distance) // 2
    z_lo = max(0, mid - 100)
    z_hi = min(len(distance) - 1, mid + 100)
    z_frames = np.arange(sf + z_lo, sf + z_hi + 1)

    EVENT_COLORS = {"open": "#2196F3", "peak": "#FF9800", "close": "#E91E63"}
    STRAT_STYLES = {}  # filled dynamically
    markers = [("v", 8, 0.7), ("^", 8, 0.7), ("D", 7, 0.8), ("s", 7, 0.8), ("o", 7, 0.8), ("P", 7, 0.8)]
    for i, name in enumerate(strategies.keys()):
        STRAT_STYLES[name] = markers[i % len(markers)]

    n_strats = len(strategies)
    fig, axes = plt.subplots(1 + n_strats, 1, figsize=(20, 3.5 * (1 + n_strats)), sharex=True)

    # Row 1: Distance trace with saved events
    ax = axes[0]
    ax.plot(z_frames, distance[z_lo:z_hi + 1], color="#1565C0", linewidth=1.5)
    ax.set_ylabel("3D Distance (mm)", fontsize=10)
    ax.set_title(f"{SUBJECT} – {TRIAL_NAME} – Event Detection Comparison (zoom)", fontsize=11, fontweight="bold")

    for etype, color in EVENT_COLORS.items():
        saved_in_range = [f for f in saved.get(etype, []) if sf + z_lo <= f <= sf + z_hi]
        if saved_in_range:
            ax.vlines(saved_in_range, 0, ax.get_ylim()[1] or 120, color=color, linewidth=1.2, alpha=0.5,
                      label=f"Saved {etype}")
    ax.legend(fontsize=8, loc='upper right', ncol=3)

    # Row 2: Distance + detected events for each strategy
    for strat_idx, (strat_name, strat_events) in enumerate(strategies.items()):
        ax = axes[1 + strat_idx]
        ax.plot(z_frames, distance[z_lo:z_hi + 1], color="#1565C0", linewidth=1.0, alpha=0.5)
        ax.set_ylabel("Distance", fontsize=9)
        ax.set_title(strat_name, fontsize=10, loc='left')

        for etype, color in EVENT_COLORS.items():
            # Saved events as vertical lines
            saved_in_range = [f for f in saved.get(etype, []) if sf + z_lo <= f <= sf + z_hi]
            if saved_in_range:
                ax.vlines(saved_in_range, 0, ax.get_ylim()[1] or 120,
                          color=color, linewidth=0.8, alpha=0.3, linestyle='--')

            # Detected events as markers
            det_in_range = [f for f in strat_events.get(etype, []) if sf + z_lo <= f <= sf + z_hi]
            if det_in_range:
                marker, size, alpha = STRAT_STYLES[strat_name]
                y_vals = [distance[f - sf] for f in det_in_range if 0 <= f - sf < len(distance)]
                valid_x = [f for f in det_in_range if 0 <= f - sf < len(distance)]
                ax.scatter(valid_x, y_vals, color=color, marker=marker, s=size**2,
                           alpha=alpha, zorder=5, label=f"Detected {etype}")

        ax.legend(fontsize=7, loc='upper right', ncol=3)

    axes[-1].set_xlabel("Global frame index", fontsize=10)
    plt.tight_layout()
    out_path = out_dir / f"{SUBJECT}_{TRIAL_NAME}_detection_comparison.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\n  → saved {out_path}")


def plot_error_histograms(saved: dict, strategies: dict[str, dict],
                          sf: int, ef: int, out_dir: Path):
    """Plot error histograms for each event type and strategy."""
    EVENT_COLORS = {"open": "#2196F3", "peak": "#FF9800", "close": "#E91E63"}

    n_strats = len(strategies)
    fig, axes = plt.subplots(n_strats, 3, figsize=(16, 3.5 * n_strats))

    for col_idx, etype in enumerate(["open", "peak", "close"]):
        saved_in_trial = [f for f in saved.get(etype, []) if sf <= f <= ef]

        for row_idx, (strat_name, strat_events) in enumerate(strategies.items()):
            ax = axes[row_idx, col_idx]
            det_events = strat_events.get(etype, [])
            matches, fp = match_events(det_events, saved_in_trial)
            errors = [e for _, _, e in matches if e is not None]

            if errors:
                bins = range(min(errors) - 1, max(errors) + 2)
                ax.hist(errors, bins=bins, color=EVENT_COLORS[etype], alpha=0.7, edgecolor='white')
                ax.axvline(0, color='black', linewidth=1, linestyle='--', alpha=0.5)
                abs_errors = [abs(e) for e in errors]
                ax.set_title(f"{etype.upper()} | abs_mean={np.mean(abs_errors):.1f}", fontsize=9)
            else:
                ax.set_title(f"{etype.upper()} | no matches", fontsize=9)

            if col_idx == 0:
                parts = strat_name.split(':')
                lbl = parts[0] + (":" + parts[1][:15] if len(parts) > 1 else "")
                ax.set_ylabel(lbl, fontsize=8)
            if row_idx == n_strats - 1:
                ax.set_xlabel("Frame error (detected - saved)", fontsize=8)

    plt.suptitle(f"{SUBJECT} – {TRIAL_NAME} – Error Distributions", fontsize=11, fontweight="bold")
    plt.tight_layout()
    out_path = out_dir / f"{SUBJECT}_{TRIAL_NAME}_error_histograms.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  → saved {out_path}")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"=== Event Detection Evaluation: {SUBJECT} / {TRIAL_NAME} ===\n")

    events = load_events(SUBJECT)
    print(f"Events loaded: {', '.join(f'{k}:{len(v)}' for k,v in events.items())}")

    trials = build_trial_map(SUBJECT)
    trial = next((t for t in trials if t["trial_name"] == TRIAL_NAME), None)
    if not trial:
        print(f"Trial {TRIAL_NAME} not found!")
        sys.exit(1)

    sf = trial["start_frame"]
    ef = trial["end_frame"]
    print(f"Trial {TRIAL_NAME}: frames {sf}–{ef}")

    # Count saved events in this trial
    for etype in ["open", "peak", "close"]:
        in_trial = [f for f in events[etype] if sf <= f <= ef]
        print(f"  Saved {etype} events in trial: {len(in_trial)}")

    print(f"\nComputing metrics…")
    metrics = compute_all_metrics(SUBJECT, trial)

    print(f"\n--- Running detection strategies ---")

    strat_a = detect_distance_only(metrics['distance'], sf)
    print_eval("A: Distance only", strat_a, events, sf, ef)

    strat_b = detect_with_reversal(metrics['distance'], metrics['reversal'], sf)
    print_eval("B: Distance + Reversal", strat_b, events, sf, ef)

    strat_c = detect_with_all_metrics(
        metrics['distance'], metrics['reversal'], metrics['motion_ssd'], sf)
    print_eval("C: Dist + Rev + SSD (no bias)", strat_c, events, sf, ef)

    # --- Sweep bias corrections to maximize exact matches (error=0) ---
    print(f"\n--- Sweeping bias corrections (optimizing for mode=0) ---")
    best_open_bias = 0
    best_close_bias = 0
    best_exact = -1

    saved_opens = [f for f in events["open"] if sf <= f <= ef]
    saved_peaks = [f for f in events["peak"] if sf <= f <= ef]
    saved_closes = [f for f in events["close"] if sf <= f <= ef]

    for ob in range(-5, 6):
        for cb in range(-5, 6):
            result = detect_with_all_metrics(
                metrics['distance'], metrics['reversal'], metrics['motion_ssd'], sf,
                open_bias=ob, close_bias=cb)

            # Count exact matches (error == 0) across all event types
            exact = 0
            for etype, saved_list in [("open", saved_opens), ("peak", saved_peaks), ("close", saved_closes)]:
                matches, _ = match_events(result[etype], saved_list)
                exact += sum(1 for _, _, e in matches if e is not None and e == 0)

            if exact > best_exact:
                best_exact = exact
                best_open_bias = ob
                best_close_bias = cb

    total_saved = len(saved_opens) + len(saved_peaks) + len(saved_closes)
    print(f"  Best bias: open={best_open_bias:+d}, close={best_close_bias:+d}")
    print(f"  Exact matches: {best_exact}/{total_saved} ({100*best_exact/total_saved:.0f}%)")

    strat_d = detect_with_all_metrics(
        metrics['distance'], metrics['reversal'], metrics['motion_ssd'], sf,
        open_bias=best_open_bias, close_bias=best_close_bias)
    print_eval(f"D: +SSD (ob={best_open_bias:+d},cb={best_close_bias:+d})",
               strat_d, events, sf, ef)

    # ── Strategy G: D + distance guards for peaks AND opens/closes ──
    print(f"\n--- Sweeping guards + bias ---")
    best_ob_g = 0
    best_cb_g = 0
    best_gf = 0.0
    best_pgf = 0.0
    best_exact_g = -1

    guard_factors = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    peak_guard_factors = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3]

    for gf in guard_factors:
        for pgf in peak_guard_factors:
            for ob in range(-5, 6):
                for cb in range(-5, 6):
                    result = detect_with_all_metrics(
                        metrics['distance'], metrics['reversal'], metrics['motion_ssd'], sf,
                        per_cam_ssd=metrics.get('per_cam_ssd'),
                        cam_names=metrics.get('cam_names'),
                        open_bias=ob, close_bias=cb,
                        dist_guard_factor=gf,
                        peak_guard_factor=pgf)
                    # Score: exact matches minus penalty for large outliers
                    exact = 0
                    max_err = 0
                    for etype, saved_list in [("open", saved_opens), ("peak", saved_peaks), ("close", saved_closes)]:
                        matches, _ = match_events(result[etype], saved_list)
                        for _, _, e in matches:
                            if e is not None:
                                if e == 0:
                                    exact += 1
                                max_err = max(max_err, abs(e))
                    # Penalize: subtract 1 exact-match-equivalent per frame of max error above 5
                    score = exact - max(0, max_err - 5)
                    if score > best_exact_g:
                        best_exact_g = score
                        best_ob_g = ob
                        best_cb_g = cb
                        best_gf = gf
                        best_pgf = pgf

    print(f"  Best: dist_guard={best_gf}, peak_guard={best_pgf}, ob={best_ob_g:+d}, cb={best_cb_g:+d}")
    print(f"  Score (exact - outlier penalty): {best_exact_g}")

    strat_g = detect_with_all_metrics(
        metrics['distance'], metrics['reversal'], metrics['motion_ssd'], sf,
        per_cam_ssd=metrics.get('per_cam_ssd'),
        cam_names=metrics.get('cam_names'),
        open_bias=best_ob_g, close_bias=best_cb_g,
        dist_guard_factor=best_gf,
        peak_guard_factor=best_pgf)
    print_eval(f"G: guards (dg={best_gf},pg={best_pgf},ob={best_ob_g:+d},cb={best_cb_g:+d})",
               strat_g, events, sf, ef)

    # ── Diagnose worst close errors in G ──
    print(f"\n--- Close outlier diagnosis (strategy G) ---")
    saved_closes_trial = [f for f in events["close"] if sf <= f <= ef]
    close_matches, _ = match_events(strat_g["close"], saved_closes_trial)
    close_errors = [(s, d, e) for s, d, e in close_matches if e is not None]
    close_errors.sort(key=lambda x: abs(x[2]), reverse=True)
    dist_arr = metrics['distance']
    rev_arr = metrics['reversal']
    ssd_arr = metrics['motion_ssd']
    for s, det, err in close_errors[:5]:
        s_local = s - sf
        d_local = det - sf
        print(f"  saved={s} det={det} err={err:+d}  "
              f"dist[saved]={dist_arr[s_local]:.1f} dist[det]={dist_arr[d_local]:.1f}  "
              f"rev[saved]={rev_arr[s_local]:.1f} rev[det]={rev_arr[d_local]:.1f}  "
              f"ssd[saved]={ssd_arr[s_local]:.1f} ssd[det]={ssd_arr[d_local]:.1f}")
    # Also show which peak each outlier close belongs to
    for s, det, err in close_errors[:5]:
        # Find the peak just before this close
        peak_before = [p for p in strat_g["peak"] if p <= det]
        peak_after = [p for p in strat_g["peak"] if p > det]
        pk = peak_before[-1] if peak_before else None
        npk = peak_after[0] if peak_after else None
        print(f"    close {det}: prev_peak={pk}, next_peak={npk}")

    print(f"\nGenerating plots…")
    strategies = {
        "A: Distance only": strat_a,
        f"D: +SSD (ob={best_open_bias:+d},cb={best_close_bias:+d})": strat_d,
        f"G: guards (dg={best_gf},pg={best_pgf},ob={best_ob_g:+d},cb={best_cb_g:+d})": strat_g,
    }
    plot_comparison(metrics, events, strategies, OUTPUT_DIR)
    plot_error_histograms(events, strategies, sf, ef, OUTPUT_DIR)

    print(f"\nDone.")
