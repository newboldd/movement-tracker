"""Analysis results: distance traces, movement parameters, group comparison."""

from __future__ import annotations

import csv as _csv
import logging
import math
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from ..config import get_settings
from ..db import get_db_ctx
from ..services.dlc_predictions import get_dlc_predictions_for_session
from ..services.mediapipe_prelabel import get_mediapipe_for_session
from ..services.metrics import auto_detect_from_distance
from ..services.video import build_trial_map

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/results", tags=["results"])

AUTO_DETECT_TYPES = ("open", "peak", "close")  # used by auto-detection


# ── helpers ─────────────────────────────────────────────────────────────

def _get_event_type_names() -> list[str]:
    """Get configured event type names from settings."""
    return [et["name"] for et in get_settings().event_types]


def _get_subject(subject_id: int) -> dict:
    """Look up a subject by ID; raise 404 if missing."""
    with get_db_ctx() as db:
        subj = db.execute(
            "SELECT * FROM subjects WHERE id = ?", (subject_id,)
        ).fetchone()
    if not subj:
        raise HTTPException(404, "Subject not found")
    return dict(subj)


def _read_events_csv(subject_name: str) -> dict:
    """Read events.csv → {event_type: [frame_nums]}."""
    settings = get_settings()
    path = settings.dlc_path / subject_name / "events.csv"
    configured = _get_event_type_names()
    result = {t: [] for t in configured}
    if not path.exists():
        return result
    with open(path, newline="") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            et = row.get("event_type", "").strip()
            fn = row.get("frame_num", "").strip()
            if fn.isdigit():
                if et not in result:
                    result[et] = []
                result[et].append(int(fn))
    return result


def _load_events(
    subject_name: str, distances: list, trials: list[dict],
) -> tuple[dict, str]:
    """Load events, falling back to auto-detection if none saved.

    Returns ``(events_dict, source)`` where *source* is ``"saved"`` or ``"auto"``.
    """
    events = _read_events_csv(subject_name)
    has_events = any(len(v) > 0 for v in events.values())
    if has_events:
        return events, "saved"
    if not distances or not trials:
        return events, "saved"
    auto = auto_detect_from_distance(distances, trials)
    return auto, "auto"


def _load_distances_and_trials(subject_name: str) -> tuple[list, list[dict], str]:
    """Return (distances, trials, source) for a subject.

    distances: list[float|None] across all frames.
    trials: list of {name, fps, start_frame, end_frame, frame_count}.
    source: 'dlc', 'mediapipe', or 'none'.
    """
    trials = build_trial_map(subject_name)
    if not trials:
        return [], [], "none"

    # Try DLC predictions first (corrections > v2 > v1)
    preds = get_dlc_predictions_for_session(subject_name)
    distances = preds.get("distances") if preds else None
    if distances:
        return distances, trials, "dlc"

    # Fall back to MediaPipe prelabels
    mp = get_mediapipe_for_session(subject_name)
    distances = mp.get("distances") if mp else None
    if distances:
        return distances, trials, "mediapipe"

    return [], trials, "none"


def _compute_velocity(dist: list, fps: float, half_win: int = 2) -> list:
    """Compute velocity via symmetric finite difference.

    vel[i] = (dist[i+hw] - dist[i-hw]) / (2*hw) * fps   (mm/s)
    """
    n = len(dist)
    vel = [None] * n
    for i in range(n):
        i0 = max(0, i - half_win)
        i1 = min(n - 1, i + half_win)
        d0, d1 = dist[i0], dist[i1]
        if d0 is not None and d1 is not None and (i1 - i0) > 0:
            vel[i] = round((d1 - d0) / (i1 - i0) * fps, 2)
    return vel


def _build_movement_params(
    distances: list,
    events: dict,
    trials: list[dict],
) -> tuple[list[dict], list[str]]:
    """Compute per-movement parameters from events + distance trace.

    Returns (movements_list, trial_names).
    """
    opens = sorted(events.get("open", []))
    peaks = sorted(events.get("peak", []))
    closes = sorted(events.get("close", []))

    if not peaks or not distances:
        return [], [t["trial_name"] for t in trials]

    # Match events into triplets: for each peak find nearest open before + close after
    movements = []
    trial_names = [t["trial_name"] for t in trials]

    # Determine fps per frame (for multi-trial with potentially different fps)
    frame_fps = [60.0] * len(distances)
    for t in trials:
        fps = t.get("fps", 60)
        for f in range(t["start_frame"], min(t["end_frame"] + 1, len(distances))):
            frame_fps[f] = fps

    # Determine which trial each frame belongs to
    frame_trial = [0] * len(distances)
    for ti, t in enumerate(trials):
        for f in range(t["start_frame"], min(t["end_frame"] + 1, len(distances))):
            frame_trial[f] = ti

    # Compute velocity for the entire trace (use median fps)
    med_fps = trials[0].get("fps", 60) if trials else 60
    vel = _compute_velocity(distances, med_fps)

    # Global minimum distance (ignoring None)
    valid_dists = [d for d in distances if d is not None]
    global_min = min(valid_dists) if valid_dists else 0

    first_peak = peaks[0]

    # Track previous amplitude per trial for relative amplitude
    prev_amp_by_trial: dict[int, float | None] = {}

    for idx, pk in enumerate(peaks):
        fps = frame_fps[pk] if pk < len(frame_fps) else 60
        ti = frame_trial[pk] if pk < len(frame_trial) else 0

        # Find matching open: largest open frame < pk
        open_f = None
        for o in reversed(opens):
            if o < pk:
                open_f = o
                break

        # Find matching close: smallest close frame > pk
        close_f = None
        for c in closes:
            if c > pk:
                close_f = c
                break

        # Peak time relative to first peak (seconds)
        peak_time = round((pk - first_peak) / fps, 4)

        # IMI — only within the same trial (cross-video IMI is meaningless)
        imi = None
        if idx > 0:
            prev_pk = peaks[idx - 1]
            prev_ti = frame_trial[prev_pk] if prev_pk < len(frame_trial) else 0
            if prev_ti == ti:
                imi = round((pk - prev_pk) / fps, 4)

        # Amplitude
        pk_dist = distances[pk] if pk < len(distances) else None
        amplitude = round(pk_dist - global_min, 2) if pk_dist is not None else None

        # Relative amplitude (ratio to previous movement's amplitude)
        rel_amplitude = None
        if amplitude is not None:
            prev_amp = prev_amp_by_trial.get(ti)
            if prev_amp is not None and prev_amp > 0:
                rel_amplitude = round(amplitude / prev_amp, 4)
            prev_amp_by_trial[ti] = amplitude

        # Velocities
        peak_open_vel = None
        mean_open_vel = None
        if open_f is not None and pk_dist is not None:
            # Peak opening velocity: max velocity between open and peak
            v_slice = vel[open_f:pk + 1]
            valid_v = [v for v in v_slice if v is not None]
            if valid_v:
                peak_open_vel = round(max(valid_v), 2)

            # Mean opening velocity: amplitude / duration
            dur_open = (pk - open_f) / fps
            if dur_open > 0 and amplitude is not None:
                mean_open_vel = round(amplitude / dur_open, 2)

        peak_close_vel = None
        mean_close_vel = None
        if close_f is not None and pk_dist is not None:
            # Peak closing velocity: min velocity between peak and close (negative)
            v_slice = vel[pk:close_f + 1]
            valid_v = [v for v in v_slice if v is not None]
            if valid_v:
                peak_close_vel = round(min(valid_v), 2)

            # Mean closing velocity: amplitude / duration (negative)
            dur_close = (close_f - pk) / fps
            if dur_close > 0 and amplitude is not None:
                mean_close_vel = round(-amplitude / dur_close, 2)

        # Power: velocity × amplitude (fast-small ≈ slow-large)
        power = None
        if peak_open_vel is not None and amplitude is not None:
            power = round(peak_open_vel * amplitude, 2)

        movements.append({
            "peak_frame": pk,
            "peak_dist": round(pk_dist, 2) if pk_dist is not None else None,
            "peak_time": peak_time,
            "imi": imi,
            "amplitude": amplitude,
            "rel_amplitude": rel_amplitude,
            "peak_open_vel": peak_open_vel,
            "peak_close_vel": peak_close_vel,
            "mean_open_vel": mean_open_vel,
            "mean_close_vel": mean_close_vel,
            "power": power,
            "trial_idx": ti,
        })

    return movements, trial_names


def _linreg_slope(x: list[float], y: list[float]) -> float | None:
    """Simple OLS slope. Returns None if < 2 points."""
    pairs = [(xi, yi) for xi, yi in zip(x, y)
             if xi is not None and yi is not None
             and math.isfinite(xi) and math.isfinite(yi)]
    if len(pairs) < 2:
        return None
    xa = np.array([p[0] for p in pairs])
    ya = np.array([p[1] for p in pairs])
    xa_mean = xa.mean()
    denom = ((xa - xa_mean) ** 2).sum()
    if denom == 0:
        return None
    slope = float(((xa - xa_mean) * (ya - ya.mean())).sum() / denom)
    return round(slope, 6)


# ── API endpoints ───────────────────────────────────────────────────────

def _preview_cache_path():
    """Path to the preview distances JSON cache file."""
    from .subjects import get_settings
    settings = get_settings()
    return settings.dlc_path.parent / ".preview_distances.json"


def _compute_preview_for_subject(subj_name: str, subj_id: int) -> dict | None:
    """Compute 10-second distance preview for one subject."""
    try:
        distances, trials, source = _load_distances_and_trials(subj_name)
        if not distances or not trials:
            return None

        trial = trials[-1]
        fps = trial.get("fps", 60)
        start = trial["start_frame"]
        end = trial["end_frame"]
        window_frames = int(10 * fps)

        win_start = max(start, end + 1 - window_frames)
        win_end = min(end + 1, len(distances))
        segment = distances[win_start:win_end]

        if len(segment) > 200:
            step = len(segment) // 200
            segment = segment[::step]

        values = [round(d, 1) if d is not None else None for d in segment]

        return {
            "values": values,
            "fps": fps,
            "source": source,
            "trial_name": trial["trial_name"],
        }
    except Exception:
        return None


@router.get("/preview-distances")
def get_preview_distances() -> dict:
    """Return 10-second distance preview for all subjects.

    Loads instantly from a JSON cache file. If no cache exists, computes
    all previews and writes the file. Returns the cached data immediately.
    """
    import json as _json

    cache_path = _preview_cache_path()

    # Try loading from file cache (instant)
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                return _json.load(f)
        except Exception:
            pass

    # No cache — compute all and write file
    return _rebuild_preview_cache()


@router.post("/preview-distances/refresh")
def refresh_preview_distances() -> dict:
    """Recompute preview distances and update the cache file.

    Called in the background by the dashboard after loading the cached version.
    Checks each subject's npz mtime against the cached value to skip unchanged ones.
    """
    import json as _json, os

    cache_path = _preview_cache_path()

    # Load existing cache to compare mtimes
    old_cache = {}
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                old_data = _json.load(f)
                old_cache = old_data.get("previews", {})
        except Exception:
            pass

    with get_db_ctx() as db:
        subjects = db.execute("SELECT id, name FROM subjects ORDER BY name").fetchall()

    from ..config import get_settings
    settings = get_settings()
    previews = {}
    updated_count = 0

    for subj in subjects:
        sid = str(subj["id"])
        name = subj["name"]

        # Check npz mtime
        npz_path = settings.dlc_path / name / "mediapipe_prelabels.npz"
        current_mtime = os.path.getmtime(str(npz_path)) if npz_path.exists() else 0

        # If mtime matches cached value, reuse cached preview
        old_entry = old_cache.get(sid)
        if old_entry and old_entry.get("_mtime") == current_mtime:
            previews[sid] = old_entry
            continue

        # Recompute
        preview = _compute_preview_for_subject(name, subj["id"])
        if preview:
            preview["_mtime"] = current_mtime
            previews[sid] = preview
            updated_count += 1

    result = {"previews": previews}

    # Write cache file
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(cache_path), "w") as f:
            _json.dump(result, f)
    except Exception:
        pass

    return {"status": "ok", "total": len(previews), "updated": updated_count}


def _rebuild_preview_cache() -> dict:
    """Full rebuild of preview cache — used on first load when no cache file exists."""
    import json as _json, os
    from ..config import get_settings

    settings = get_settings()
    cache_path = _preview_cache_path()

    with get_db_ctx() as db:
        subjects = db.execute("SELECT id, name FROM subjects ORDER BY name").fetchall()

    previews = {}
    for subj in subjects:
        npz_path = settings.dlc_path / subj["name"] / "mediapipe_prelabels.npz"
        mtime = os.path.getmtime(str(npz_path)) if npz_path.exists() else 0

        preview = _compute_preview_for_subject(subj["name"], subj["id"])
        if preview:
            preview["_mtime"] = mtime
            previews[str(subj["id"])] = preview

    result = {"previews": previews}

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(cache_path), "w") as f:
            _json.dump(result, f)
    except Exception:
        pass

    return result


def invalidate_preview_cache():
    """Delete cache file to force full rebuild on next dashboard load."""
    try:
        path = _preview_cache_path()
        if path.exists():
            path.unlink()
    except Exception:
        pass


@router.get("/{subject_id}/traces")
def get_traces(subject_id: int) -> dict:
    """Distance and velocity traces split by trial for the Distances tab."""
    subj = _get_subject(subject_id)
    subject_name = subj["name"]

    distances, trials, data_source = _load_distances_and_trials(subject_name)
    if not distances:
        return {"trials": [], "subject": subject_name, "data_source": "none"}

    result_trials = []
    all_dists: list[float] = []
    all_vels: list[float] = []

    for t in trials:
        s, e = t["start_frame"], t["end_frame"] + 1
        trial_dist = distances[s:e]
        fps = t.get("fps", 60)
        trial_vel = _compute_velocity(trial_dist, fps)

        result_trials.append({
            "name": t["trial_name"],
            "fps": fps,
            "distances": trial_dist,
            "velocities": trial_vel,
        })

        all_dists.extend(d for d in trial_dist if d is not None)
        all_vels.extend(v for v in trial_vel if v is not None)

    # Global y-ranges for matched scaling across trials
    y_range = None
    if all_dists:
        y_range = {
            "dist_min": round(min(all_dists), 2),
            "dist_max": round(max(all_dists), 2),
            "vel_min": round(min(all_vels), 2) if all_vels else 0,
            "vel_max": round(max(all_vels), 2) if all_vels else 0,
        }

    return {"trials": result_trials, "subject": subject_name, "y_range": y_range, "data_source": data_source}


@router.get("/{subject_id}/movements")
def get_movements(subject_id: int) -> dict:
    """Per-movement parameters for the Movements tab."""
    subj = _get_subject(subject_id)
    subject_name = subj["name"]

    distances, trials, data_source = _load_distances_and_trials(subject_name)
    events, event_source = _load_events(subject_name, distances, trials)

    movements, trial_names = _build_movement_params(distances, events, trials)

    return {
        "movements": movements,
        "trial_names": trial_names,
        "subject": subject_name,
        "event_source": event_source,
        "data_source": data_source,
    }


@router.get("/group")
def get_group_comparison(include_auto: bool = Query(False)) -> dict:
    """Aggregated per-subject statistics grouped by diagnosis."""
    with get_db_ctx() as db:
        subjects = db.execute(
            "SELECT * FROM subjects ORDER BY group_label, name"
        ).fetchall()

    PARAM_KEYS = [
        "imi", "amplitude", "rel_amplitude", "power",
        "peak_open_vel", "peak_close_vel",
        "mean_open_vel", "mean_close_vel",
    ]

    results = []
    for subj in subjects:
        subject_name = subj["name"]
        diagnosis = subj.get("group_label") or subj.get("diagnosis") or "Control"

        try:
            distances, trials, _src = _load_distances_and_trials(subject_name)
            if include_auto:
                events, event_source = _load_events(subject_name, distances, trials)
            else:
                events = _read_events_csv(subject_name)
                has = any(len(v) > 0 for v in events.values())
                event_source = "saved" if has else "none"
            movements, _ = _build_movement_params(distances, events, trials)
        except Exception as exc:
            logger.warning(f"Skipping {subject_name}: {exc}")
            continue

        if len(movements) < 2:
            continue

        # Aggregate each parameter
        entry: dict = {
            "name": subject_name,
            "diagnosis": diagnosis,
            "event_source": event_source,
        }

        for key in PARAM_KEYS:
            vals = [m[key] for m in movements if m[key] is not None]
            if vals:
                mean_v = np.mean(vals)
                std_v = np.std(vals)
                entry[f"mean_{key}"] = round(float(mean_v), 4)
                entry[f"cv_{key}"] = round(float(std_v / abs(mean_v)), 4) if mean_v != 0 else None

                # Sequence effect: slope of parameter vs movement index
                # Negate closing velocities (which are negative) so the
                # sequence-effect slope reflects changes in magnitude.
                seq_vals = [-v for v in vals] if key in ("peak_close_vel", "mean_close_vel") else vals
                indices = [float(i) for i, m in enumerate(movements) if m[key] is not None]
                slope = _linreg_slope(indices, seq_vals)
                entry[f"seq_{key}"] = slope
            else:
                entry[f"mean_{key}"] = None
                entry[f"cv_{key}"] = None
                entry[f"seq_{key}"] = None

        # Frequency = 1 / mean IMI
        mean_imi = entry.get("mean_imi")
        entry["frequency"] = round(1.0 / mean_imi, 4) if mean_imi and mean_imi > 0 else None

        results.append(entry)

    # Collect unique groups
    groups = sorted(set(r["diagnosis"] for r in results))

    return {"subjects": results, "groups": groups}
