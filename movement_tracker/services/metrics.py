"""Video-based metric computation and Strategy G event detection.

Ported from scratch/evaluate_peak_detection.py — computes reversal and SSD
motion metrics from video crops around finger tips, and uses them to refine
distance-based event detection.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np

from ..config import get_settings
from .dlc_predictions import get_corrections_with_dlc_fallback, get_dlc_predictions_for_stage

logger = logging.getLogger(__name__)

CROP_HALF = 24  # 48x48 px crop around finger tips


# ── Distance-trace cleanup helpers ────────────────────────────────
#
# Two complementary sources of integer-frame error feed into the
# auto-detection.  First, the combined-MP merge occasionally picks an
# (OS, OD) source pair that triangulates to an aperture much smaller
# or larger than its temporal neighbours -- a 1- to 2-frame spike that
# the local-max scan mistakes for a real peak.  Second, even on clean
# stretches the raw distance trace has ±0.5 mm noise, so the integer
# location of argmax / argmin jitters by 1 frame between adjacent
# values.  Both inflate the residual offset between the auto picks and
# manually-saved events.  ``clean_distance_trace`` does a robust spike
# removal (MAD z-score + velocity gate); an optional Gaussian smooth
# is available but defaults off because even a small sigma shifts
# the integer location of argmax / argmin on asymmetric peaks and
# ends up hurting more exact matches than it helps.

def _interp_nans_1d(arr: np.ndarray) -> np.ndarray:
    """Replace NaN values with linear interpolation between the nearest
    finite samples.  Edge NaNs extend the nearest finite value."""
    out = np.asarray(arr, dtype=float).copy()
    mask = np.isnan(out)
    if mask.all() or not mask.any():
        return out
    idx = np.arange(out.size)
    out[mask] = np.interp(idx[mask], idx[~mask], out[~mask])
    return out


def _gaussian1d(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Tiny Gaussian smoothing.  ``sigma`` in samples; radius = ceil(3σ)."""
    if sigma <= 0:
        return np.asarray(arr, dtype=float)
    radius = max(1, int(np.ceil(3 * sigma)))
    t = np.arange(-radius, radius + 1)
    k = np.exp(-(t ** 2) / (2 * sigma * sigma))
    k = k / k.sum()
    return np.convolve(np.asarray(arr, dtype=float), k, mode="same")


def clean_distance_trace(
    dist,
    max_valid_dist: float = 200.0,
    spike_window: int = 3,
    spike_z: float = 3.5,
    spike_velocity: float = 25.0,
    smooth_sigma: float = 0.0,
) -> np.ndarray:
    """Return a cleaned, smoothed copy of a 1-D distance trace.

    Pipeline:
      1. Mark <0 / >max_valid_dist / NaN samples as NaN.
      2. MAD-based spike filter: for each sample t compare it to a
         ``±spike_window`` neighbourhood; if |z| > ``spike_z`` AND
         either single-frame jump exceeds ``spike_velocity``, mark NaN.
      3. Linear interpolation across NaN runs.
      4. Light Gaussian smoothing with ``smooth_sigma``.

    The combination removes the 1- to 2-frame outliers that combined-MP
    occasionally produces without flattening genuine fast motion.
    """
    arr = np.asarray(dist, dtype=float).copy()
    n = arr.size
    if n == 0:
        return arr
    # 1. Range filter.
    arr[~np.isfinite(arr)] = np.nan
    arr[(arr < 0) | (arr > max_valid_dist)] = np.nan
    # 2. MAD spike filter.  Compare each finite sample to its small
    # neighbourhood; a |z| > spike_z _and_ a >spike_velocity jump on
    # either side flags a real spike (vs noise on a stable stretch).
    finite0 = np.isfinite(arr)
    base = _interp_nans_1d(arr)   # interp once so window stats use neighbours
    for t in range(n):
        if not finite0[t]:
            continue
        lo, hi = max(0, t - spike_window), min(n, t + spike_window + 1)
        win = base[lo:hi]
        if win.size < 3:
            continue
        med = float(np.median(win))
        mad = 1.4826 * float(np.median(np.abs(win - med)))
        if mad < 1.0:
            mad = 1.0   # floor — flat stretches shouldn't be hyper-sensitive
        z = abs(base[t] - med) / mad
        if z <= spike_z:
            continue
        dl = abs(base[t] - base[t - 1]) if t > 0 else 0.0
        dr = abs(base[t + 1] - base[t]) if t < n - 1 else 0.0
        if max(dl, dr) > spike_velocity:
            arr[t] = np.nan
    # 3. Re-interpolate NaNs.
    out = _interp_nans_1d(arr)
    # 4. Gaussian smooth.
    return _gaussian1d(out, smooth_sigma)


# ── Auto-detect calibration offsets ───────────────────────────────
#
# A small JSON file at ``<MT_DATA_DIR>/auto_detect_calibration.json``
# stores per-event integer offsets to add to each auto-detected event
# frame.  ``scripts/calibrate_auto_detect.py`` measures the median
# (auto − saved) offset across every subject's saved events and writes
# the file; ``auto_detect_from_distance`` reads it on each call.  Use
# ``MT_DISABLE_AUTO_CAL=1`` to ignore the file (handy for re-measuring).

_CALIBRATION_CACHE: dict | None = None
_CALIBRATION_CACHE_MTIME: float = 0.0


def _calibration_path() -> Path:
    from ..config import DATA_DIR
    return DATA_DIR / "auto_detect_calibration.json"


def _normalize_cal_entry(v) -> dict:
    """Accept either a bare int (legacy) or a dict {shift, veto}
    and return the dict form ``{"shift": int, "veto": dict|None}``."""
    if isinstance(v, dict):
        try:
            shift = int(v.get("shift", 0))
        except (TypeError, ValueError):
            shift = 0
        veto = v.get("veto")
        if isinstance(veto, dict):
            try:
                side = str(veto.get("side", "next"))
                thresh = float(veto.get("thresh", 0.6))
                return {"shift": shift, "veto": {"side": side, "thresh": thresh}}
            except (TypeError, ValueError):
                pass
        return {"shift": shift, "veto": None}
    try:
        return {"shift": int(v), "veto": None}
    except (TypeError, ValueError):
        return {"shift": 0, "veto": None}


def _load_calibration_offsets() -> dict:
    """Return ``{event: {shift: int, veto: {side, thresh} | None}}`` from
    disk (cached).  Accepts the legacy bare-int format
    (``{"open": -2, ...}``) for backwards compatibility."""
    global _CALIBRATION_CACHE, _CALIBRATION_CACHE_MTIME
    default = {k: {"shift": 0, "veto": None} for k in ("open", "peak", "close")}
    if os.environ.get("MT_DISABLE_AUTO_CAL"):
        return default
    p = _calibration_path()
    try:
        st = p.stat()
    except OSError:
        _CALIBRATION_CACHE = default
        _CALIBRATION_CACHE_MTIME = 0.0
        return _CALIBRATION_CACHE
    if _CALIBRATION_CACHE is not None and st.st_mtime == _CALIBRATION_CACHE_MTIME:
        return _CALIBRATION_CACHE
    try:
        data = json.loads(p.read_text())
        out = {k: _normalize_cal_entry(data.get(k, 0)) for k in ("open", "peak", "close")}
    except (OSError, ValueError, TypeError):
        out = default
    _CALIBRATION_CACHE = out
    _CALIBRATION_CACHE_MTIME = st.st_mtime
    return out


# ── Metrics disk cache ────────────────────────────────────────────────────────


def _metrics_cache_path(subject_name: str) -> Path:
    """Path to the metrics cache JSON for a subject."""
    return get_settings().dlc_path / subject_name / "metrics_cache.json"


def _source_mtime(subject_name: str) -> float:
    """Get the latest mtime across all prediction/correction CSVs for a subject.

    Used as a staleness check — if any source data is newer than the cache,
    the cache is stale and metrics should be recomputed.
    """
    settings = get_settings()
    dlc_dir = settings.dlc_path / subject_name
    latest = 0.0
    for subdir in ("corrections", "labels_v2", "labels_v1", "labels_v1.0", "labels_v0.1"):
        d = dlc_dir / subdir
        if d.is_dir():
            for f in d.iterdir():
                if f.suffix == ".csv":
                    try:
                        latest = max(latest, f.stat().st_mtime)
                    except OSError:
                        pass
    return latest


def _load_metrics_cache(subject_name: str) -> dict | None:
    """Load the entire metrics cache for a subject, or None if stale/missing."""
    path = _metrics_cache_path(subject_name)
    if not path.exists():
        return None
    try:
        cache = json.loads(path.read_text())
        # Check staleness: if source CSVs are newer than cache, invalidate
        cache_mtime = cache.get("_mtime", 0)
        if _source_mtime(subject_name) > cache_mtime:
            logger.info("Metrics cache stale for %s, will recompute", subject_name)
            return None
        return cache
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Could not read metrics cache for %s: %s", subject_name, e)
        return None


def _save_metrics_cache(subject_name: str, cache: dict):
    """Persist the metrics cache for a subject."""
    path = _metrics_cache_path(subject_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    import time as _time
    cache["_mtime"] = _time.time()
    path.write_text(json.dumps(cache))


def get_cached_trial_metrics(subject_name: str, trial_name: str) -> dict | None:
    """Return cached metrics for a single trial, or None if not cached/stale.

    Entries from before ``motion_flow`` was added are treated as stale so
    they get recomputed on next access.
    """
    cache = _load_metrics_cache(subject_name)
    if cache is None:
        return None
    entry = cache.get(trial_name)
    if entry is None:
        return None
    # Schema check: require motion_flow (added when we switched from
    # frame-MSD to phase-correlation flow for opens/closes detection).
    if "motion_flow" not in entry:
        return None
    return entry


def save_trial_metrics_to_cache(subject_name: str, trial_name: str, metrics: dict):
    """Save metrics for a single trial into the subject's cache file."""
    cache = _load_metrics_cache(subject_name) or {}
    cache[trial_name] = metrics
    _save_metrics_cache(subject_name, cache)
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

    def _compute_motion_flow(cam_filter=None):
        """Phase-correlation flow magnitude: |translation(frame[i-1] → frame[i])|.

        Empirically (see scripts/calibrate_strategy_g.py and the SSD vs
        flow A/B in commit notes) flow tracks tip motion onset/offset
        better than raw MSD, which gives ~20% relative improvement on
        exact-open matches and is at least as good on closes.
        """
        arr = np.zeros(n_frames, dtype=np.float64)
        cams = [cam_filter] if cam_filter else cam_names
        for local_idx in range(1, n_frames):
            metric_sum = 0.0
            n_valid = 0
            for cam in cams:
                for bp in BODYPARTS:
                    curr = crops.get((local_idx, cam, bp))
                    prev = crops.get((local_idx - 1, cam, bp))
                    if curr is None or prev is None or curr.shape != prev.shape:
                        continue
                    try:
                        (dx, dy), _ = cv2.phaseCorrelate(prev, curr)
                    except cv2.error:
                        continue
                    metric_sum += float(np.hypot(dx, dy))
                    n_valid += 1
            if n_valid > 0:
                arr[local_idx] = metric_sum / n_valid
        return arr

    # Compute combined metrics
    reversal = _compute_reversal()
    motion_ssd = _compute_motion_ssd()
    motion_flow = _compute_motion_flow()

    # Per-camera variants (used by detect_strategy_g's per-camera
    # tip-search pass).  We keep both so old callers / debug views
    # that still want SSD don't lose it.
    per_cam_ssd = {}
    per_cam_flow = {}
    for cam in cam_names:
        per_cam_ssd[cam] = _compute_motion_ssd(cam_filter=cam)
        per_cam_flow[cam] = _compute_motion_flow(cam_filter=cam)

    return {
        "distance": distance,
        "reversal": reversal.tolist(),
        "motion_ssd": motion_ssd.tolist(),
        "motion_flow": motion_flow.tolist(),
        "per_cam_ssd": {cam: arr.tolist() for cam, arr in per_cam_ssd.items()},
        "per_cam_flow": {cam: arr.tolist() for cam, arr in per_cam_flow.items()},
        "n_frames": n_frames,
    }


# ── Detection helpers ────────────────────────────────────────────────────────


def _clean_and_prep_distance(
    dist: list,
    min_event_gap: int = 10,
    min_peak_height: float = 15.0,
    valley_thresh: float = 25.0,
    max_valid_dist: float = 200.0,
) -> tuple[list[float], list[float], list[bool], list[int]]:
    """Shared distance preprocessing: clean, interpolate, find peaks.

    Returns (d, ddist1, nan_mask, merged_peaks).
    """
    n = len(dist)
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
    half_win = max(min_event_gap // 2, 5)
    peak_candidates: list[int] = []
    for i in range(half_win, n - half_win):
        if d[i] < min_peak_height:
            continue
        if any(nan_mask[i - half_win : i + half_win + 1]):
            continue
        if d[i] >= max(d[i - half_win : i + half_win + 1]):
            peak_candidates.append(i)

    # Merge nearby candidates (valley-aware: split when a deep valley separates them)
    merged_peaks: list[int] = []
    if peak_candidates:
        group = [peak_candidates[0]]
        for k in range(1, len(peak_candidates)):
            if peak_candidates[k] - peak_candidates[k - 1] < min_event_gap:
                # Check if a deep valley separates these nearby candidates
                valley_min = min(d[peak_candidates[k - 1] : peak_candidates[k] + 1])
                if valley_min < valley_thresh:
                    # Deep valley → treat as separate peaks
                    merged_peaks.append(max(group, key=lambda f: d[f]))
                    group = [peak_candidates[k]]
                else:
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
    peaks_only: bool = False,
    existing_peaks: list[int] | None = None,
) -> dict[str, list[int]]:
    """Configurable Strategy G: multi-metric event detection.

    steps: {use_reversal, use_ssd, use_dist_guard, use_peak_guard}
    params: {min_peak_height, min_event_gap, open_start_thresh, valley_thresh, nback,
             reversal_search_radius, ssd_search_radius, open_bias, close_bias,
             dist_guard_factor, peak_guard_factor, gaussian_sigma}

    Returns {open, peak, close} as GLOBAL frame numbers.
    """
    min_peak_height = float(params.get("min_peak_height", 15.0))
    min_event_gap = int(params.get("min_event_gap", 10))
    open_start_thresh = float(params.get("open_start_thresh", 5.0))
    valley_thresh = float(params.get("valley_thresh", 25.0))
    nback = int(params.get("nback", 60))
    reversal_search_radius = int(params.get("reversal_search_radius", 3))
    ssd_search_radius = int(params.get("ssd_search_radius", 5))
    open_bias = int(params.get("open_bias", 0))
    close_bias = int(params.get("close_bias", 0))
    dist_guard_factor = float(params.get("dist_guard_factor", 0.25))
    peak_guard_factor = float(params.get("peak_guard_factor", 0.1))
    gaussian_sigma = float(params.get("gaussian_sigma", 2.0))
    max_valid_dist = float(params.get("max_valid_dist", 200.0))
    edge_min_peak = float(params.get("edge_min_peak", 15.0))

    use_reversal = steps.get("use_reversal", True) and reversal is not None
    use_ssd = steps.get("use_ssd", True) and motion_ssd is not None
    use_dist_guard = steps.get("use_dist_guard", True) and use_ssd
    use_peak_guard = steps.get("use_peak_guard", True) and use_reversal

    n = len(dist)
    d, ddist1, nan_mask, merged_peaks = _clean_and_prep_distance(
        dist, min_event_gap, min_peak_height, valley_thresh, max_valid_dist
    )

    # ``existing_peaks`` lets the caller hand us the peak set to use
    # for open/close detection without re-running peak finding.  This
    # is what the Events page's "Detect Opens/Closes" button sends:
    # the user's current (possibly hand-edited, possibly unsaved)
    # peaks become the anchors, and downstream open/close detection
    # runs around them.  Peaks are in LOCAL (trial-relative) indices,
    # already clipped to [0, n).
    if existing_peaks is not None:
        refined_peaks = sorted(set(
            int(p) for p in existing_peaks if 0 <= int(p) < n
        ))
    else:
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

    # Peaks-only mode: skip open/close detection entirely
    if peaks_only:
        return {
            "open": [],
            "peak": [start_frame + f for f in refined_peaks],
            "close": [],
        }

    # Prepare SSD arrays
    sm_ssd = None
    sm_ssd_per_cam: dict[str, np.ndarray] = {}
    if use_ssd:
        sm_ssd = gaussian_smooth(np.array(motion_ssd, dtype=np.float64), gaussian_sigma)
        if per_cam_ssd and cam_names:
            for cam in cam_names:
                if cam in per_cam_ssd:
                    sm_ssd_per_cam[cam] = gaussian_smooth(np.array(per_cam_ssd[cam], dtype=np.float64), gaussian_sigma)

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


# ── Distance-based event detection (no video/session needed) ─────────────────


def auto_detect_from_distance(
    dist_raw: list,
    trials: list[dict],
    *,
    min_peak_height: float = 15.0,
    open_start_thresh: float = 5.0,
    nback: int = 60,
    min_event_gap: int = 10,
    max_valid_dist: float = 200.0,
    valley_thresh: float = 25.0,
    edge_margin: int = 2,
    edge_min_peak: float = 15.0,
    pre_clean: bool = True,
    smooth_sigma: float = 0.0,
    calibration_offsets: dict | None = None,
) -> dict:
    """Distance-based event detection without session dependency.

    Extracts the core detection algorithm from the labeling endpoint so it can
    be reused by the results page (auto-detect fallback) and other callers.

    When ``pre_clean`` is True (default), the input trace is first run
    through ``clean_distance_trace`` (MAD spike filter + light Gaussian)
    so 1-frame combined-MP outliers don't masquerade as peaks and the
    integer location of argmax / argmin stops jittering on noise.  Pass
    ``pre_clean=False`` when the caller has already cleaned the trace
    (e.g. distances_clean from the combined npz).

    ``calibration_offsets`` is ``{open: int, peak: int, close: int}``
    integer frame shifts applied AFTER detection.  Defaults to the
    on-disk calibration file (or zeros).

    Returns ``{open: [frames], peak: [frames], close: [frames]}``.
    """
    # Pre-clean: smooth + spike-filter so downstream extremum finders
    # land on the same integer frame the user sees on the video.
    if pre_clean:
        cleaned = clean_distance_trace(dist_raw, max_valid_dist=max_valid_dist,
                                         smooth_sigma=smooth_sigma)
        dist = cleaned.tolist()
        nan_mask = [False] * len(dist)
    else:
        # Caller supplied an already-cleaned trace; still range-filter
        # and interpolate any residual NaNs so the rest of the code can
        # assume a finite sequence.
        arr = np.asarray(dist_raw, dtype=float).copy()
        arr[~np.isfinite(arr)] = np.nan
        arr[(arr < 0) | (arr > max_valid_dist)] = np.nan
        nan_mask = [bool(m) for m in np.isnan(arr)]
        arr = _interp_nans_1d(arr)
        dist = arr.tolist()

    n = len(dist)

    # 4-frame and 1-frame velocity
    ddist1 = [0.0] * n
    for i in range(1, n):
        ddist1[i] = dist[i] - dist[i - 1]

    # Find peaks (local maxima above min_peak_height)
    half_win = max(min_event_gap // 2, 5)
    peak_candidates: list[int] = []
    for i in range(half_win, n - half_win):
        if dist[i] < min_peak_height:
            continue
        if any(nan_mask[i - half_win: i + half_win + 1]):
            continue
        if dist[i] >= max(dist[i - half_win: i + half_win + 1]):
            peak_candidates.append(i)

    # Merge nearby candidates (valley-aware: split when a deep valley separates them)
    merged_peaks: list[int] = []
    if peak_candidates:
        group = [peak_candidates[0]]
        for k in range(1, len(peak_candidates)):
            if peak_candidates[k] - peak_candidates[k - 1] < min_event_gap:
                valley_min = min(dist[peak_candidates[k - 1] : peak_candidates[k] + 1])
                if valley_min < valley_thresh:
                    merged_peaks.append(max(group, key=lambda f: dist[f]))
                    group = [peak_candidates[k]]
                else:
                    group.append(peak_candidates[k])
            else:
                merged_peaks.append(max(group, key=lambda f: dist[f]))
                group = [peak_candidates[k]]
        merged_peaks.append(max(group, key=lambda f: dist[f]))

    # Valley filter
    filtered: list[int] = []
    for pk in merged_peaks:
        if not filtered:
            filtered.append(pk)
            continue
        valley_min = min(dist[filtered[-1]: pk + 1])
        if valley_min < valley_thresh:
            filtered.append(pk)
        elif dist[pk] > dist[filtered[-1]]:
            filtered[-1] = pk
    merged_peaks = filtered

    # For each peak, find open onset and close.
    #
    # Open gate: instead of "first k where ddist1[k] > open_start_thresh"
    # (which trips on a single-frame noise tick before motion really
    # starts and biases auto-opens 1–3 frames early), require positive
    # velocity for 3 consecutive frames AND a real cumulative gain
    # (≈1.5x the single-frame threshold) before declaring the onset.
    SUSTAIN = 3
    PER_FRAME_MIN = max(1.0, 0.5 * open_start_thresh)
    CUM_MIN = 1.5 * open_start_thresh
    opens_raw: list[int] = []
    closes_raw: list[int] = []
    for idx, pk in enumerate(merged_peaks):
        search_start = closes_raw[-1] if closes_raw else max(0, pk - nback)
        valley = min(range(search_start, pk + 1), key=lambda f: dist[f])
        open_frame = valley
        for k in range(valley, pk):
            if k + SUSTAIN >= n:
                break
            if all(ddist1[k + j] > PER_FRAME_MIN for j in range(SUSTAIN)) \
                    and (dist[k + SUSTAIN] - dist[k]) > CUM_MIN:
                open_frame = k
                break
        opens_raw.append(max(search_start, open_frame))

        next_pk = merged_peaks[idx + 1] if idx + 1 < len(merged_peaks) else n
        closes_raw.append(min(range(pk, min(next_pk, n)), key=lambda f: dist[f]))

    # De-duplicate opens with min gap
    all_open = sorted(opens_raw)
    opens: list[int] = []
    last_f = -min_event_gap
    for f in all_open:
        if f - last_f >= min_event_gap:
            opens.append(f)
            last_f = f

    # Per-trial peak/close detection using opens as anchors
    frame_to_trial: dict[int, int] = {}
    for ti, t in enumerate(trials):
        for f in range(t["start_frame"], t["end_frame"] + 1):
            frame_to_trial[f] = ti

    anchor_by_trial: dict[int, list[int]] = {}
    for f in opens:
        ti = frame_to_trial.get(f, 0)
        anchor_by_trial.setdefault(ti, []).append(f)

    peaks: list[int] = []
    closes: list[int] = []

    for ti, t in enumerate(trials):
        vid_anchors = sorted(anchor_by_trial.get(ti, []))
        if not vid_anchors:
            continue
        vid_start = t["start_frame"]
        vid_end = t["end_frame"] + 1

        # Before first anchor
        if vid_anchors[0] - vid_start >= 5:
            pk_f = max(range(vid_start, vid_anchors[0]), key=lambda f: dist[f])
            if dist[pk_f] >= edge_min_peak:
                peaks.append(pk_f)
                closes.append(min(range(pk_f, vid_anchors[0]), key=lambda f: dist[f]))

        # Between consecutive anchors
        for ai in range(len(vid_anchors) - 1):
            t0, t1 = vid_anchors[ai], vid_anchors[ai + 1]
            if t1 - t0 < 5:
                continue
            peak_frame = max(range(t0, t1), key=lambda f: dist[f])
            peaks.append(peak_frame)
            closes.append(min(range(peak_frame, min(t1, n)), key=lambda f: dist[f]))

        # After last anchor
        if vid_end - vid_anchors[-1] >= 5:
            pk_f = max(range(vid_anchors[-1], vid_end), key=lambda f: dist[f])
            if dist[pk_f] >= edge_min_peak:
                peaks.append(pk_f)
                closes.append(min(range(pk_f, vid_end), key=lambda f: dist[f]))

    # Filter events at video edges
    valid_frames = set()
    for t in trials:
        for f in range(t["start_frame"] + edge_margin, t["end_frame"] - edge_margin + 1):
            valid_frames.add(f)

    opens = [f for f in opens if f in valid_frames]
    peaks = [f for f in peaks if f in valid_frames]
    closes = [f for f in closes if f in valid_frames]

    # Validate sequence: open → peak → close → repeat
    all_events = (
        [(f, "open") for f in opens]
        + [(f, "peak") for f in peaks]
        + [(f, "close") for f in closes]
    )
    all_events.sort()

    valid_events: dict[str, list[int]] = {"open": [], "peak": [], "close": []}
    expected_next = "open"

    for frame, etype in all_events:
        if expected_next == "open":
            if etype == "open":
                valid_events["open"].append(frame)
                expected_next = "peak"
        elif expected_next == "peak":
            if etype == "peak":
                valid_events["peak"].append(frame)
                expected_next = "close"
            elif etype == "close":
                valid_events["close"].append(frame)
                expected_next = "open"
        elif expected_next == "close":
            if etype == "close":
                valid_events["close"].append(frame)
                expected_next = "open"

    # Apply residual-offset calibration so the auto picks line up with
    # the user's manual selections.  Offsets are integer frame shifts
    # (and optionally an adaptive veto rule — see detect_events_from_peaks)
    # measured by scripts/calibrate_auto_detect.py and stored at
    # ``<MT_DATA_DIR>/auto_detect_calibration.json``.  Here in the
    # distance-only path we honour just the shift component.
    offsets = (calibration_offsets if calibration_offsets is not None
               else _load_calibration_offsets())
    def _entry_shift(v):
        if isinstance(v, dict): return int(v.get("shift", 0))
        try: return int(v)
        except (TypeError, ValueError): return 0
    max_frame = max((t.get("end_frame", 0) for t in trials), default=n - 1)
    def _shift(seq, k):
        if not k:
            return list(seq)
        out = []
        for f in seq:
            nf = f + k
            if 0 <= nf <= max_frame:
                out.append(nf)
        return sorted(set(out))

    return {
        "open":  _shift(valid_events["open"],  _entry_shift(offsets.get("open", 0))),
        "peak":  _shift(valid_events["peak"],  _entry_shift(offsets.get("peak", 0))),
        "close": _shift(valid_events["close"], _entry_shift(offsets.get("close", 0))),
    }


# ── Velocity-based detection (anchored on saved peaks) ────────────
#
# When the user has already labelled peaks, the auto-detection's hardest
# job (peak picking + the search-window anchoring) is solved.  All
# that's left is the opens and closes.  This path uses:
#
#   * Index-tip 3D velocity onset for OPENS — walk backward from each
#     peak and find the first frame where the smoothed |v_index| drops
#     below 30 % of its local-window maximum.  On the same dataset this
#     more than doubles the exact-match rate vs the distance-derivative
#     sustained-velocity gate (35 % vs 16 %).
#   * d/dt(distance) zero-crossing (negative → positive) for CLOSES.
#     The moment the thumb-index distance stops decreasing, which the
#     eye picks more cleanly than the literal argmin in a noisy valley.
#   * Calibration shifts from the JSON file.
#   * Optional adaptive veto for closes — if the post-shift candidate
#     frame still has high velocity at the NEXT frame (motion still
#     under way), revert the shift for that single event.  This lets a
#     larger global shift (e.g. close = −3) win on the majority of
#     events while preserving the small fraction where the unshifted
#     pick was already correct.

INDEX_TIP = 8


def _triangulate_one(os_pt, od_pt, calib):
    """Triangulate one 2-D landmark pair to 3-D, returning None on
    NaN or cv2 failure."""
    if calib is None or os_pt is None or od_pt is None:
        return None
    if np.any(np.isnan(os_pt)) or np.any(np.isnan(od_pt)):
        return None
    try:
        from .calibration import triangulate_points
        pts = triangulate_points(
            np.array([os_pt], dtype=np.float64),
            np.array([od_pt], dtype=np.float64),
            calib,
        )
    except Exception:
        return None
    if pts is None or np.any(np.isnan(pts)):
        return None
    return pts[0]


def build_joint_3d_trace(OS_landmarks, OD_landmarks, joint: int, calib):
    """Return an (n, 3) array of triangulated joint positions; NaN
    where either camera was missing the landmark."""
    n = OS_landmarks.shape[0]
    out = np.full((n, 3), np.nan)
    for f in range(n):
        p = _triangulate_one(OS_landmarks[f, joint], OD_landmarks[f, joint], calib)
        if p is not None:
            out[f] = p
    return out


def velocity_magnitude_3d(pos3d: np.ndarray, smooth_sigma: float = 1.0) -> np.ndarray:
    """Centred-difference velocity magnitude per frame, NaN-aware,
    with an optional NaN-aware Gaussian smooth (σ ≈ 1 default — the
    raw centred-difference is noisier than position)."""
    n = pos3d.shape[0]
    v = np.full(n, np.nan)
    for f in range(1, n - 1):
        a, b = pos3d[f - 1], pos3d[f + 1]
        if np.all(np.isfinite(a)) and np.all(np.isfinite(b)):
            v[f] = float(np.linalg.norm(b - a) / 2.0)
    if smooth_sigma > 0:
        r = int(np.ceil(3 * smooth_sigma))
        tt = np.arange(-r, r + 1)
        k = np.exp(-(tt ** 2) / (2 * smooth_sigma ** 2))
        k = k / k.sum()
        valid = np.isfinite(v).astype(float)
        v0 = np.where(np.isnan(v), 0.0, v)
        num = np.convolve(v0, k, mode="same")
        den = np.convolve(valid, k, mode="same")
        with np.errstate(invalid="ignore", divide="ignore"):
            v = np.where(den > 0.3, num / den, np.nan)
    return v


def _index_velocity_onset_back(v_index, peak_frame, search_back=60, ratio=0.3):
    """Walk backward from ``peak_frame`` and find the motion-onset frame.

    Velocity profile from peak going back: at the apex velocity is
    near zero, then HIGH through the rising phase, then drops back to
    near zero as we cross the open onset.  Two-phase search:

      1. Walk back until we find the first frame with v > thresh (we
         are now inside the rising-motion zone).
      2. Continue walking back until v first drops below thresh again
         — that's the frame just before motion started, i.e. the open.
    """
    n = len(v_index)
    if peak_frame <= 0 or peak_frame >= n:
        return None
    lo = max(0, peak_frame - search_back)
    win = v_index[lo:peak_frame + 1]
    if not np.any(np.isfinite(win)):
        return None
    local_max = float(np.nanmax(win))
    if not np.isfinite(local_max) or local_max < 1.0:
        return None
    thresh = ratio * local_max
    # Phase 1: locate the rising-motion zone behind the peak.
    above = None
    for f in range(peak_frame, lo - 1, -1):
        v = v_index[f]
        if np.isfinite(v) and v > thresh:
            above = f
            break
    if above is None:
        return None
    # Phase 2: continue walking back; first below-threshold frame is
    # the motion-onset (open) candidate.
    for f in range(above - 1, lo - 1, -1):
        v = v_index[f]
        if np.isfinite(v) and v < thresh:
            return f
    return None


def _dv_zero_crossing_after(distances, peak_frame, next_peak_frame, search_forward=60):
    """First frame after ``peak_frame`` where d/dt(distance) crosses
    from negative to positive (motion stops decreasing → close)."""
    n = len(distances)
    end = min(next_peak_frame, n - 1, peak_frame + search_forward)
    if peak_frame + 2 >= end:
        return None
    for f in range(peak_frame + 1, end):
        try:
            d_prev = float(distances[f - 1])
            d_cur = float(distances[f])
            d_next = float(distances[f + 1])
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(d_prev) and np.isfinite(d_cur) and np.isfinite(d_next)):
            continue
        dv_prev = d_cur - d_prev
        dv_cur = d_next - d_cur
        if dv_prev < 0 and dv_cur >= 0:
            return f
    return None


def _veto_shift(v_index, shifted_frame, unshifted_frame, veto):
    """Adaptive veto.  ``veto`` = {"side": "prior"|"next", "thresh": float}.
    Look up v_index[shifted ± 1] and compare to the local max in a
    ±5 window.  If v_at_neighbour > thresh * local_max we've shifted
    INTO ongoing motion — revert to unshifted_frame."""
    if not veto or v_index is None:
        return shifted_frame
    n = len(v_index)
    if shifted_frame < 1 or shifted_frame >= n - 1:
        return shifted_frame
    side = veto.get("side", "next")
    thresh = float(veto.get("thresh", 0.6))
    fq = shifted_frame - 1 if side == "prior" else shifted_frame + 1
    if fq < 0 or fq >= n:
        return shifted_frame
    lo = max(0, shifted_frame - 5)
    hi = min(n, shifted_frame + 6)
    local_max = np.nanmax(v_index[lo:hi]) if hi > lo else np.nan
    if not np.isfinite(local_max) or local_max < 1.0:
        return shifted_frame
    v_q = v_index[fq]
    if np.isfinite(v_q) and v_q > thresh * local_max:
        return unshifted_frame
    return shifted_frame


def detect_events_from_peaks(
    distances,
    trials: list[dict],
    saved_peaks: list[int],
    *,
    OS_landmarks=None,
    OD_landmarks=None,
    calib=None,
    calibration: dict | None = None,
    open_ratio: float = 0.3,
    search_back: int = 60,
    search_forward: int = 60,
) -> dict:
    """Anchor on user-corrected peaks; derive opens and closes.

    Primary path (when ``OS_landmarks``, ``OD_landmarks``, ``calib`` are
    all supplied): index-tip velocity onset for opens, d/dt(distance)
    zero-crossing for closes, adaptive close-veto driven by the velocity
    signature of the candidate frame.  Falls back to distance-only
    heuristics for any peak the velocity path can't resolve.

    ``calibration`` is the rich form
    ``{"open": {"shift": int, "veto": ...}, "peak": ..., "close": ...}``;
    when ``None`` the on-disk file is loaded (with the
    ``MT_DISABLE_AUTO_CAL`` escape hatch).

    Returns ``{"open": [...], "peak": <saved_peaks>, "close": [...]}``.
    """
    n = len(distances)
    if n == 0:
        return {"open": [], "peak": [], "close": []}
    peaks = sorted(int(p) for p in saved_peaks if 0 <= int(p) < n)
    if not peaks:
        return {"open": [], "peak": [], "close": []}

    cal = calibration if calibration is not None else _load_calibration_offsets()
    open_cal = _normalize_cal_entry(cal.get("open", 0))
    close_cal = _normalize_cal_entry(cal.get("close", 0))

    # Clean the distance trace once (spike filter + interpolation).
    dist_arr = clean_distance_trace(np.asarray(distances, dtype=float))

    # Try to build index-tip velocity.  Any failure → distance fallback.
    v_index = None
    if OS_landmarks is not None and OD_landmarks is not None and calib is not None:
        try:
            idx3d = build_joint_3d_trace(OS_landmarks, OD_landmarks, INDEX_TIP, calib)
            v = velocity_magnitude_3d(idx3d, smooth_sigma=1.0)
            if np.any(np.isfinite(v)):
                v_index = v
        except Exception as e:
            logger.warning(f"detect_events_from_peaks: velocity build failed ({e})")

    max_frame = max((int(t.get("end_frame", 0)) for t in trials), default=n - 1) if trials else (n - 1)

    def _clamp(f):
        if f is None: return None
        return max(0, min(max_frame, int(f)))

    opens: list[int] = []
    closes: list[int] = []
    for i, pk in enumerate(peaks):
        next_pk = peaks[i + 1] if i + 1 < len(peaks) else max_frame + 1

        # OPEN — velocity-onset back-walk from peak.
        op_raw = None
        if v_index is not None:
            op_raw = _index_velocity_onset_back(
                v_index, pk, search_back=search_back, ratio=open_ratio,
            )
        if op_raw is None:
            # Fallback: distance-trace argmin (valley) before the peak
            # then refine forward via the existing sustained gate.
            start = max(0, pk - search_back)
            try:
                valley = int(np.argmin(dist_arr[start:pk + 1])) + start
            except ValueError:
                valley = None
            if valley is not None:
                op_raw = valley
                # Sustained-velocity refine — first k where 3 consecutive
                # ddist1 > 1 and cumulative gain > 7.5 (matches the
                # existing distance-only auto-detect's defaults).
                for k in range(valley, pk - 3):
                    d0 = dist_arr[k]; d1 = dist_arr[k + 1]
                    d2 = dist_arr[k + 2]; d3 = dist_arr[k + 3]
                    if all(np.isfinite(x) for x in (d0, d1, d2, d3)):
                        if (d1 - d0) > 1 and (d2 - d1) > 1 and (d3 - d2) > 1 \
                                and (d3 - d0) > 7.5:
                            op_raw = k
                            break
        if op_raw is not None:
            op_shifted = _clamp(op_raw + open_cal["shift"])
            if op_cal_veto := open_cal["veto"]:
                op_shifted = _veto_shift(v_index, op_shifted, _clamp(op_raw), op_cal_veto)
            opens.append(op_shifted)

        # CLOSE — d/dt zero-crossing forward from peak.
        cl_raw = _dv_zero_crossing_after(
            dist_arr, pk, next_pk, search_forward=search_forward,
        )
        if cl_raw is None:
            # Fallback: argmin between peaks.
            end = min(next_pk, n)
            try:
                cl_raw = int(np.argmin(dist_arr[pk:end])) + pk
            except ValueError:
                cl_raw = None
        if cl_raw is not None:
            cl_shifted = _clamp(cl_raw + close_cal["shift"])
            cl_veto = close_cal["veto"]
            if cl_veto and v_index is not None:
                cl_shifted = _veto_shift(v_index, cl_shifted, _clamp(cl_raw), cl_veto)
            closes.append(cl_shifted)

    return {
        "open":  sorted(set(opens)),
        "peak":  list(peaks),
        "close": sorted(set(closes)),
    }
