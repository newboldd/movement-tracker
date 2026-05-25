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

import os as _os
import json as _json

# Static-export movement cache: when MT_EXPORT_CACHE is set, the group
# endpoint caches each (subject, source, include_auto) movement build so
# the many seq_mode/hand/trial combos don't recompute it.  Off (empty)
# in the live app, so behavior there is unchanged.
_EXPORT_CACHE_ON = bool(_os.environ.get("MT_EXPORT_CACHE"))
_EXPORT_MOVE_CACHE: dict = {}


def _mp_newest_mtime(settings, subject_name: str) -> float:
    """Return the newest mtime across this subject's MediaPipe outputs.

    Walks ``<dlc>/<subject>/<trial_stem>/mediapipe.npz`` (per-trial layout)
    and includes the legacy ``mediapipe_prelabels.npz`` if still present.
    Returns 0.0 when nothing exists.  Used as a cache-invalidation proxy
    by the distance / metric cache freshness checks.
    """
    import os
    subj_dir = settings.dlc_path / subject_name
    newest = 0.0
    if not subj_dir.exists():
        return newest
    try:
        for trial_dir in subj_dir.iterdir():
            if not trial_dir.is_dir():
                continue
            for fname in ("mediapipe.npz", "mediapipe_reverse.npz"):
                p = trial_dir / fname
                if p.exists():
                    newest = max(newest, os.path.getmtime(str(p)))
    except OSError:
        pass
    legacy = subj_dir / "mediapipe_prelabels.npz"
    if legacy.exists():
        try:
            newest = max(newest, os.path.getmtime(str(legacy)))
        except OSError:
            pass
    return newest

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


_DOSE_HM_RE = __import__("re").compile(
    r"(?:(\d+(?:\.\d+)?)\s*h(?:ours?|rs?)?)?\s*(?:(\d+(?:\.\d+)?)\s*m(?:in)?)?",
    flags=__import__("re").IGNORECASE,
)


def _parse_last_dose_minutes(raw) -> float | None:
    """Parse a free-text ``last_dose`` field into minutes since dose.

    Accepts ``"5h"``, ``"30m"``, ``"5h 30m"``, ``"1h30m"``, ``"2.5h"``,
    case-insensitive.  Bare numbers are treated as minutes.  ``"-"``,
    empty strings, or anything that doesn't parse to a positive value
    returns ``None`` so the subject can be excluded from time-since-
    dose plots without polluting the axis.
    """
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if not s or s in ("-", "n/a", "na", "none", "?"):
        return None
    # Bare numeric → minutes.
    try:
        v = float(s)
        return v if v > 0 else None
    except ValueError:
        pass
    m = _DOSE_HM_RE.fullmatch(s.replace(" ", ""))
    if not m:
        # Try a permissive search (handles "5h 30m extra notes"
        # mostly by ignoring the trailing junk).  Only accept when at
        # least one of the h/m groups landed; otherwise bail.
        m = _DOSE_HM_RE.search(s)
        if not m or (m.group(1) is None and m.group(2) is None):
            return None
    h_part = m.group(1)
    m_part = m.group(2)
    total = 0.0
    if h_part:
        total += float(h_part) * 60.0
    if m_part:
        total += float(m_part)
    return total if total > 0 else None


def _is_levodopa_off(raw) -> bool:
    """True when a subject is explicitly NOT taking levodopa.

    The clinical ``levodopa`` field holds the regimen; a value of
    ``"0"`` (or none/off/no) marks an off-levodopa subject.  An empty
    field is treated as unknown (not off) so we don't lump
    unrecorded subjects into the Off cluster.
    """
    if raw is None:
        return False
    s = str(raw).strip().lower()
    return s in ("0", "none", "no", "off", "nil", "n")


def _laterality_more_side(lat) -> str:
    """More-affected side ('L'/'R') from a clinical laterality value.

    Handles 'L', 'R', 'left', 'right', and 'L>R' / 'R>L' (more-affected
    side listed first).  Bilateral / none / blank / unrecognized →
    default 'R' (per project convention: if no clinical laterality is
    recorded the right hand is treated as more-affected).
    """
    if lat is None:
        return "R"
    s = str(lat).strip().lower()
    if not s or s in ("none", "b/l", "bilateral", "n/a", "na"):
        return "R"
    if ">" in s:
        s = s.split(">")[0].strip()
    if s.startswith("l"):
        return "L"
    if s.startswith("r"):
        return "R"
    return "R"


def _hand_of_trial(trial_name: str) -> str | None:
    """'L'/'R' from a trial name like 'Con03_L1'."""
    suffix = str(trial_name).split("_")[-1]
    c = suffix[:1].upper()
    return c if c in ("L", "R") else None


def _se_chosen_hand(trials, all_movements, seq_mode: str,
                     prefer_larger: bool) -> str | None:
    """Pick L or R based on the seq-effect slope of each hand's last
    trial (using amplitude under the requested ``seq_mode``).
    ``prefer_larger`` = True → return the hand with the MORE NEGATIVE
    slope (larger magnitude decrement); False → the less negative one.
    Returns None when no slopes are computable for either hand."""
    if seq_mode == "none":
        return None
    by_hand: dict[str, list[int]] = {"L": [], "R": []}
    for i, t in enumerate(trials):
        h = _hand_of_trial(t.get("trial_name", ""))
        if h:
            by_hand[h].append(i)
    slopes: dict[str, float] = {}
    for h, idxs in by_hand.items():
        if not idxs:
            continue
        last_idx = idxs[-1]
        movs = [m for m in all_movements if m.get("trial_idx") == last_idx]
        se = _sequence_effect(movs, "amplitude", seq_mode)
        if se is not None and se.get("slope") is not None:
            slopes[h] = se["slope"]
    if not slopes:
        return None
    # Larger SE = more negative slope (treat as descending decrement).
    return (min if prefer_larger else max)(slopes, key=lambda h: slopes[h])


def _select_trial_indices(trials: list[dict], hand: str, trial_sel: str,
                          laterality) -> set[int]:
    """Which trial indices to include for the chosen hand + trial mode.

    hand ∈ {more, less, L, R, average, larger_se, smaller_se};
    trial_sel ∈ {first, last, average}.  Returns an empty set when the
    selection can't be resolved (e.g. SE-based selection requested but
    no slope data available).  For larger_se / smaller_se the caller
    must use the ``hand=`` wrapper in ``get_group_comparison`` which
    converts to a concrete 'L'/'R' first.
    """
    by_hand: dict[str, list[int]] = {"L": [], "R": []}
    for i, t in enumerate(trials):
        h = _hand_of_trial(t.get("trial_name", ""))
        if h:
            by_hand[h].append(i)

    more = _laterality_more_side(laterality)
    if hand == "more":
        hands = [more]
    elif hand == "less":
        hands = ["R" if more == "L" else "L"]
    elif hand in ("L", "R"):
        hands = [hand]
    else:  # average → both hands
        hands = ["L", "R"]

    sel: set[int] = set()
    for h in hands:
        idxs = by_hand.get(h, [])
        if not idxs:
            continue
        if trial_sel == "first":
            sel.add(idxs[0])
        elif trial_sel == "last":
            sel.add(idxs[-1])
        else:  # average → all trials of this hand
            sel.update(idxs)
    return sel


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


def _load_distances_and_trials(subject_name: str, source: str | None = None) -> tuple[list, list[dict], str]:
    """Return (distances, trials, source) for a subject.

    distances: list[float|None] across all frames.
    trials: list of {name, fps, start_frame, end_frame, frame_count}.
    source: 'dlc', 'mediapipe', 'vision', 'skeleton_v1', 'skeleton_v2',
    'skeleton_v3', 'corrections', or 'none'.
    """
    trials = build_trial_map(subject_name)
    if not trials:
        return [], [], "none"

    settings = get_settings()

    def _try_source(src):
        if src == "corrections":
            from ..services.dlc_predictions import get_dlc_predictions_for_stage
            data = get_dlc_predictions_for_stage(subject_name, "corrections")
            return data.get("distances") if data else None
        elif src == "dlc":
            preds = get_dlc_predictions_for_session(subject_name)
            return preds.get("distances") if preds else None
        elif src == "mp_combined":
            mp = get_mediapipe_for_session(subject_name, prefer_combined=True)
            return mp.get("distances") if mp else None
        elif src in ("mediapipe", "mp_forward"):
            mp = get_mediapipe_for_session(subject_name)
            return mp.get("distances") if mp else None
        elif src == "vision":
            vision_path = settings.dlc_path / subject_name / "vision_prelabels.npz"
            if not vision_path.exists():
                return None
            try:
                v_data = np.load(str(vision_path))
                v_os = v_data.get("OS_landmarks")
                v_od = v_data.get("OD_landmarks")
                from ..services.calibration import get_calibration_for_subject, triangulate_points
                calib = get_calibration_for_subject(subject_name)
                if calib is not None and v_os is not None and v_od is not None:
                    # Triangulate thumb-index distance
                    thumb_3d = triangulate_points(v_os[:, 4, :], v_od[:, 4, :], calib)
                    index_3d = triangulate_points(v_os[:, 8, :], v_od[:, 8, :], calib)
                    dist = np.linalg.norm(thumb_3d - index_3d, axis=1)
                    return [round(float(d), 2) if not np.isnan(d) else None for d in dist]
                elif v_os is not None:
                    # 2D fallback
                    dist = np.linalg.norm(v_os[:, 4, :] - v_os[:, 8, :], axis=1)
                    return [round(float(d), 2) if not np.isnan(d) else None for d in dist]
            except Exception:
                return None
            return None
        elif src in ("skeleton_v1", "skeleton_v2", "skeleton_v3"):
            # v1 = original stage-1 fit (skeleton_v1.npz),
            # v2 = frozen legacy smoothing (skeleton_v2.npz),
            # v3 = corrections pipeline (skeleton_v3.npz).
            _NPZ_MAP = {"skeleton_v1": "skeleton_v1.npz",
                        "skeleton_v2": "skeleton_v2.npz",
                        "skeleton_v3": "skeleton_v3.npz"}
            from ..services.skeleton_data import _skeleton_dir
            import numpy as np
            skeleton_root = _skeleton_dir(subject_name)
            all_dists = [None] * sum(t["frame_count"] for t in trials)
            for t in trials:
                npz_path = skeleton_root / t["trial_name"] / _NPZ_MAP[src]
                if not npz_path.exists():
                    continue
                data = np.load(str(npz_path), allow_pickle=True)
                j3d = data.get("joints_3d")
                if j3d is None:
                    continue
                for i in range(j3d.shape[0]):
                    gi = t["start_frame"] + i
                    if gi < len(all_dists) and not np.isnan(j3d[i, 4, 0]) and not np.isnan(j3d[i, 8, 0]):
                        all_dists[gi] = round(float(np.linalg.norm(j3d[i, 4] - j3d[i, 8])), 2)
            return all_dists if any(d is not None for d in all_dists) else None
        return None

    # If a specific source is requested, try only that
    if source and source != "auto":
        distances = _try_source(source)
        if distances:
            return distances, trials, source
        return [], trials, "none"

    # Auto: DLC corrected > MP combined > MP forward.  Older fallback
    # sources (raw DLC, skeleton variants, vision) were removed from
    # the priority list — the results page only ever picks from the
    # three the dropdown exposes.
    for src in ("corrections", "mp_combined", "mp_forward"):
        distances = _try_source(src)
        if distances:
            return distances, trials, src

    return [], trials, "none"


def _try_source_quick(subject_name: str, src: str) -> bool:
    """Quick check if a distance source has data for a subject (no full load)."""
    settings = get_settings()
    try:
        if src == "corrections":
            return (settings.dlc_path / subject_name / "corrections").is_dir()
        elif src == "dlc":
            for d in ("corrections", "labels_v2", "labels_v1", "labels_v1.0"):
                if (settings.dlc_path / subject_name / d).is_dir():
                    return True
            return False
        elif src in ("mediapipe", "mp_forward"):
            from ..services.mediapipe_prelabel import has_mediapipe_data
            return has_mediapipe_data(subject_name)
        elif src == "mp_combined":
            # Combined MP exists when at least one per-trial combined
            # npz has been written by the fusion step.
            for d in settings.dlc_path.glob(f"{subject_name}/*/mediapipe_combined.npz"):
                if d.exists():
                    return True
            return False
        elif src == "vision":
            return (settings.dlc_path / subject_name / "vision_prelabels.npz").exists()
        elif src in ("skeleton_v1", "skeleton_v2", "skeleton_v3"):
            _QNPZ = {"skeleton_v1": "skeleton_v1.npz",
                     "skeleton_v2": "skeleton_v2.npz",
                     "skeleton_v3": "skeleton_v3.npz"}
            skel_dir = settings.dlc_path / subject_name / "skeleton"
            if not skel_dir.is_dir():
                return False
            want = _QNPZ[src]
            for d in skel_dir.iterdir():
                if d.is_dir() and (d / want).exists():
                    return True
            return False
    except Exception:
        return False
    return False


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

        # Velocities.  Track the FRAME where each peak velocity occurs
        # (not the distance-peak frame) so the velocity-plot markers and
        # the velocity sequence-effect fits use the correct time axis.
        peak_open_vel = None
        peak_open_vel_frame = None
        mean_open_vel = None
        if open_f is not None and pk_dist is not None:
            # Peak opening velocity: max velocity between open and peak.
            best_v, best_f = None, None
            for fi in range(open_f, min(pk + 1, len(vel))):
                v = vel[fi]
                if v is None:
                    continue
                if best_v is None or v > best_v:
                    best_v, best_f = v, fi
            if best_v is not None:
                peak_open_vel = round(best_v, 2)
                peak_open_vel_frame = best_f

            # Mean opening velocity: amplitude / duration
            dur_open = (pk - open_f) / fps
            if dur_open > 0 and amplitude is not None:
                mean_open_vel = round(amplitude / dur_open, 2)

        peak_close_vel = None
        peak_close_vel_frame = None
        mean_close_vel = None
        if close_f is not None and pk_dist is not None:
            # Peak closing velocity: min velocity between peak and close (negative).
            best_v, best_f = None, None
            for fi in range(pk, min(close_f + 1, len(vel))):
                v = vel[fi]
                if v is None:
                    continue
                if best_v is None or v < best_v:
                    best_v, best_f = v, fi
            if best_v is not None:
                peak_close_vel = round(best_v, 2)
                peak_close_vel_frame = best_f

            # Mean closing velocity: amplitude / duration (negative)
            dur_close = (close_f - pk) / fps
            if dur_close > 0 and amplitude is not None:
                mean_close_vel = round(-amplitude / dur_close, 2)

        # Times of the velocity peaks, relative to the first distance
        # peak (same reference as ``peak_time``) so the sequence-effect
        # fits stay on a common axis.
        peak_open_vel_time = (round((peak_open_vel_frame - first_peak) / fps, 4)
                              if peak_open_vel_frame is not None else None)
        peak_close_vel_time = (round((peak_close_vel_frame - first_peak) / fps, 4)
                               if peak_close_vel_frame is not None else None)

        # Power: velocity × amplitude (fast-small ≈ slow-large)
        power = None
        if peak_open_vel is not None and amplitude is not None:
            power = round(peak_open_vel * amplitude, 2)

        # Per-trial-local frames for the opening and closing event, so
        # the client can slice each trial's distance trace directly
        # without re-computing the global → local offset.
        trial_start = trials[ti]["start_frame"] if ti < len(trials) else 0
        open_frame_local = (open_f - trial_start) if open_f is not None else None
        close_frame_local = (close_f - trial_start) if close_f is not None else None
        peak_frame_local = pk - trial_start

        movements.append({
            "peak_frame": pk,
            "peak_dist": round(pk_dist, 2) if pk_dist is not None else None,
            "peak_time": peak_time,
            "open_frame": open_f,
            "close_frame": close_f,
            "open_frame_local": open_frame_local,
            "close_frame_local": close_frame_local,
            "peak_frame_local": peak_frame_local,
            "imi": imi,
            "amplitude": amplitude,
            "rel_amplitude": rel_amplitude,
            "peak_open_vel": peak_open_vel,
            "peak_close_vel": peak_close_vel,
            "peak_open_vel_frame": peak_open_vel_frame,
            "peak_close_vel_frame": peak_close_vel_frame,
            "peak_open_vel_time": peak_open_vel_time,
            "peak_close_vel_time": peak_close_vel_time,
            "mean_open_vel": mean_open_vel,
            "mean_close_vel": mean_close_vel,
            "power": power,
            "trial_idx": ti,
        })

    return movements, trial_names


# ── Movement-similarity helpers (parallels the JS shape-overlay code) ──
def _resample_seg_to_grid(xs, ys, x_lo: float, x_hi: float, n_pts: int):
    """Linear-interp resample to an evenly-spaced grid; NaN outside the
    segment's own [xs[0], xs[-1]] range."""
    out = np.full(n_pts, np.nan)
    xs_arr = np.asarray(xs, dtype=float)
    ys_arr = np.asarray(ys, dtype=float)
    if xs_arr.size < 2:
        return out
    grid = np.linspace(x_lo, x_hi, n_pts)
    in_range = (grid >= xs_arr[0]) & (grid <= xs_arr[-1])
    out[in_range] = np.interp(grid[in_range], xs_arr, ys_arr)
    return out


def _upper_triangle_pearson_mean(arrs) -> float | None:
    """nanmean of pairwise Pearson r across the upper triangle of the
    correlation matrix between the resampled segments."""
    vals = []
    n = len(arrs)
    for i in range(n):
        a = arrs[i]
        for j in range(i + 1, n):
            b = arrs[j]
            mask = np.isfinite(a) & np.isfinite(b)
            cnt = int(mask.sum())
            if cnt < 5:
                continue
            aa = a[mask]
            bb = b[mask]
            ma = aa.mean(); mb = bb.mean()
            ax = aa - ma; bx = bb - mb
            den = math.sqrt(max(1e-12, (ax * ax).sum() * (bx * bx).sum()))
            r = float((ax * bx).sum() / den) if den > 0 else None
            if r is not None and math.isfinite(r):
                vals.append(r)
    if not vals:
        return None
    return float(np.mean(vals))


def _movement_corr_shifts(segs, x_max: float, dt: float = 1.0 / 240) -> list:
    """Peak-aligned 2-pass cross-correlation refinement.  Returns the
    shift applied to each segment's open-aligned xs.  Mirror of the
    JavaScript `_computeCorrShifts` in static/js/results.js."""
    n = len(segs)
    if n == 0:
        return []
    x_lo_ref, x_hi_ref = -x_max, x_max
    n_grid = int(round((x_hi_ref - x_lo_ref) / dt)) + 1
    # Peak-aligned resamples.
    peak_aligned = []
    peak_ts = [s["peakT"] for s in segs]
    for s, pt in zip(segs, peak_ts):
        xs_shift = (np.asarray(s["xs"], dtype=float) - pt).tolist()
        peak_aligned.append(_resample_seg_to_grid(
            xs_shift, s["ys"], x_lo_ref, x_hi_ref, n_grid))

    min_count = max(2, math.ceil(n * 0.25))

    def build_mean(extra_lags):
        sums = np.zeros(n_grid)
        counts = np.zeros(n_grid)
        for i, arr in enumerate(peak_aligned):
            k = int(round(extra_lags[i])) if extra_lags is not None else 0
            if k == 0:
                m = np.isfinite(arr)
                sums[m] += arr[m]
                counts[m] += 1
            else:
                ii_lo = max(0, k)
                ii_hi = min(n_grid, n_grid + k)
                jj_lo = ii_lo - k
                jj_hi = ii_hi - k
                if ii_hi <= ii_lo:
                    continue
                seg_slice = arr[jj_lo:jj_hi]
                m = np.isfinite(seg_slice)
                sums[ii_lo:ii_hi][m] += seg_slice[m]
                counts[ii_lo:ii_hi][m] += 1
        ref = np.where(counts >= min_count, sums / np.maximum(counts, 1), np.nan)
        idxs = np.where(np.isfinite(ref))[0]
        if idxs.size == 0:
            return ref, -math.inf, math.inf
        first, last = int(idxs[0]), int(idxs[-1])
        return ref, x_lo_ref + first * dt, x_lo_ref + last * dt

    max_lag = int(round(x_max / 2 / dt))

    def one_pass(ref, mean_xmin, mean_xmax):
        lags = np.zeros(n)
        for si in range(n):
            seg = peak_aligned[si]
            k_lo = max(-max_lag, math.ceil(-mean_xmax / dt))
            k_hi = min( max_lag, math.floor(-mean_xmin / dt))
            if k_lo > k_hi:
                lags[si] = 0
                continue
            corr = np.full(2 * max_lag + 1, -math.inf)
            best_lag = 0
            best_c = -math.inf
            for k in range(k_lo, k_hi + 1):
                ii_lo = max(0, k)
                ii_hi = min(n_grid, n_grid + k)
                if ii_hi <= ii_lo:
                    continue
                a = seg[ii_lo:ii_hi]
                b = ref[ii_lo - k:ii_hi - k]
                m = np.isfinite(a) & np.isfinite(b)
                if m.sum() < 5:
                    continue
                aa = a[m]; bb = b[m]
                ma = aa.mean(); mb = bb.mean()
                ax = aa - ma; bx = bb - mb
                den = math.sqrt(max(1e-12, (ax * ax).sum() * (bx * bx).sum()))
                c = float((ax * bx).sum() / den) if den > 0 else -math.inf
                corr[k + max_lag] = c
                if c > best_c:
                    best_c = c
                    best_lag = k
            refined = float(best_lag)
            if k_lo < best_lag < k_hi:
                c0 = corr[best_lag - 1 + max_lag]
                c1 = corr[best_lag     + max_lag]
                c2 = corr[best_lag + 1 + max_lag]
                denom = c0 - 2 * c1 + c2
                if math.isfinite(c0) and math.isfinite(c2) and denom < 0:
                    refined = best_lag + 0.5 * (c0 - c2) / denom
                    refined = max(float(k_lo), min(float(k_hi), refined))
            lags[si] = refined
        return lags

    ref1, mx_min1, mx_max1 = build_mean(None)
    lags1 = one_pass(ref1, mx_min1, mx_max1)
    ref2, mx_min2, mx_max2 = build_mean(lags1.tolist())
    lags2 = one_pass(ref2, mx_min2, mx_max2)
    # Final guard: clamp so the shifted peak lands inside the
    # rebuilt-from-pass2 mean extent.
    _, fxmin, fxmax = build_mean(lags2.tolist())
    if math.isfinite(fxmin) and math.isfinite(fxmax):
        for i in range(n):
            peak_out = -lags2[i] * dt
            if peak_out < fxmin:
                lags2[i] = -fxmin / dt
            elif peak_out > fxmax:
                lags2[i] = -fxmax / dt
    return [-peak_ts[i] - float(lags2[i]) * dt for i in range(n)]


def _movement_is_valid(m: dict, distances: list, phase: str = "whole") -> bool:
    """Validity test for the movement-similarity analysis.

    phase='whole' (default): require open<peak<close events and a
       finite distance at every frame between open and close.
    phase='open':  require open<peak; finite distances open→peak.
    phase='close': require peak<close; finite distances peak→close.
    """
    o = m.get("open_frame")
    p = m.get("peak_frame")
    c = m.get("close_frame")
    if phase == "open":
        if o is None or p is None or o >= p:
            return False
        lo, hi = o, p
    elif phase == "close":
        if p is None or c is None or p >= c:
            return False
        lo, hi = p, c
    else:
        if o is None or p is None or c is None or not (o < p < c):
            return False
        lo, hi = o, c
    n = len(distances)
    if lo < 0 or hi >= n:
        return False
    for f in range(lo, hi + 1):
        v = distances[f]
        if v is None:
            return False
        try:
            if not math.isfinite(float(v)):
                return False
        except (TypeError, ValueError):
            return False
    return True


def _movement_similarity_per_trial(distances: list, trials: list[dict],
                                     all_movements: list[dict],
                                     grid: int = 120) -> dict:
    """Compute the per-trial "movement similarity" for each alignment
    mode.  Returns {trial_idx: {'corr':r, 'open':r, 'peak':r, 'close':r}}
    where r is the nanmean of the upper triangle of the Pearson
    correlation matrix between all movements in that trial under the
    given alignment.  Only movements with open<peak<close in order and
    finite distances at every frame from open to close are included."""
    movs_by_trial: dict[int, list] = {}
    for m in all_movements:
        if (m.get("open_frame_local") is None
                or m.get("close_frame_local") is None):
            continue
        if not _movement_is_valid(m, distances, "whole"):
            continue
        movs_by_trial.setdefault(int(m["trial_idx"]), []).append(m)

    result: dict[int, dict] = {}
    n_dist = len(distances)
    for ti, trial in enumerate(trials):
        movs = movs_by_trial.get(ti, [])
        if len(movs) < 2:
            result[ti] = {a: None for a in ("corr", "open", "peak", "close")}
            continue
        fps = float(trial.get("fps", 60) or 60)
        s_start = int(trial["start_frame"])
        s_end = int(trial["end_frame"])

        segs = []
        global_max_t = 0.0
        for m in movs:
            o_loc = max(0, int(m["open_frame_local"]))
            c_loc = int(m["close_frame_local"])
            o_g = s_start + o_loc
            c_g = min(s_end, s_start + c_loc)
            if c_g <= o_g:
                continue
            xs = []; ys = []
            for f in range(o_g, c_g + 1):
                if f < 0 or f >= n_dist:
                    continue
                v = distances[f]
                if v is None:
                    continue
                xs.append((f - o_g) / fps)
                ys.append(float(v))
            if len(xs) < 2:
                continue
            p_loc = m.get("peak_frame_local")
            peak_g = (s_start + int(p_loc)) if p_loc is not None else o_g
            peak_g = max(o_g, min(c_g, peak_g))
            segs.append({
                "xs": xs, "ys": ys,
                "peakT": (peak_g - o_g) / fps,
                "closeT": (c_g  - o_g) / fps,
            })
            if xs[-1] > global_max_t:
                global_max_t = xs[-1]

        if len(segs) < 2 or global_max_t <= 0:
            result[ti] = {a: None for a in ("corr", "open", "peak", "close")}
            continue

        x_max = max(0.5, math.ceil(global_max_t * 1.05 * 10) / 10)

        # Open: t=0 at opening event.
        arrs = [_resample_seg_to_grid(s["xs"], s["ys"], 0, x_max, grid)
                for s in segs]
        sim_open = _upper_triangle_pearson_mean(arrs)
        # Peak: t=0 at peak.
        arrs = [_resample_seg_to_grid(
                    [x - s["peakT"] for x in s["xs"]],
                    s["ys"], -x_max / 2, x_max / 2, grid) for s in segs]
        sim_peak = _upper_triangle_pearson_mean(arrs)
        # Close: t=0 at closing event.
        arrs = [_resample_seg_to_grid(
                    [x - s["closeT"] for x in s["xs"]],
                    s["ys"], -x_max, 0, grid) for s in segs]
        sim_close = _upper_triangle_pearson_mean(arrs)
        # Corr: peak-seeded 2-pass cross-correlation.
        shifts = _movement_corr_shifts(segs, x_max)
        arrs = [_resample_seg_to_grid(
                    [x + shifts[i] for x in segs[i]["xs"]],
                    segs[i]["ys"], -x_max / 2, x_max / 2, grid)
                for i in range(len(segs))]
        sim_corr = _upper_triangle_pearson_mean(arrs)

        result[ti] = {
            "corr": sim_corr, "open": sim_open,
            "peak": sim_peak, "close": sim_close,
        }
    return result


def _linreg_fit(x: list[float], y: list[float]) -> tuple[float, float] | None:
    """OLS fit of y on x → (slope, R²).  None if < 2 valid points."""
    pairs = [(xi, yi) for xi, yi in zip(x, y)
             if xi is not None and yi is not None
             and math.isfinite(xi) and math.isfinite(yi)]
    if len(pairs) < 2:
        return None
    xa = np.array([p[0] for p in pairs])
    ya = np.array([p[1] for p in pairs])
    xm, ym = xa.mean(), ya.mean()
    sxx = ((xa - xm) ** 2).sum()
    if sxx == 0:
        return None
    slope = float(((xa - xm) * (ya - ym)).sum() / sxx)
    sstot = ((ya - ym) ** 2).sum()
    r2 = float((slope * slope * sxx) / sstot) if sstot > 0 else 0.0
    return slope, r2


def _exp_fit(x: list[float], y: list[float]) -> tuple[float, float] | None:
    """Exponential fit y = a·exp(b·x) → (rate b, R² on original scale)."""
    pairs = [(xi, yi) for xi, yi in zip(x, y)
             if xi is not None and yi is not None
             and math.isfinite(xi) and math.isfinite(yi) and yi > 0]
    if len(pairs) < 2:
        return None
    xa = np.array([p[0] for p in pairs])
    ya = np.array([p[1] for p in pairs])
    la = np.log(ya)
    xm = xa.mean()
    denom = ((xa - xm) ** 2).sum()
    if denom == 0:
        return None
    b = float(((xa - xm) * (la - la.mean())).sum() / denom)
    a = np.exp(la.mean() - b * xm)
    pred = a * np.exp(b * xa)
    sstot = ((ya - ya.mean()) ** 2).sum()
    r2 = float(1.0 - ((ya - pred) ** 2).sum() / sstot) if sstot > 0 else 0.0
    return b, r2


def _optimize_sequences(amps: list[float], min_moves: int = 5,
                        min_r2: float = 0.3) -> list[tuple[int, int]]:
    """Port of the client ``optimizeSequences``: DP-optimal set of
    non-overlapping decreasing windows (slope<0, R²≥min_r2, length≥
    min_moves) on the amplitude series, maximizing total explained SS.

    Returns a list of (start, end-exclusive) index windows.
    """
    n = len(amps)
    if n < min_moves:
        return []
    am = np.asarray(amps, dtype=float)
    if not np.all(np.isfinite(am)):
        # Replace non-finite with interpolation-free guard: bail if any.
        if np.isnan(am).any():
            return []
    total_ss = float(((am - am.mean()) ** 2).sum())
    if total_ss == 0:
        return []

    windows = []  # (start, end, ss_reg)
    for i in range(0, n - min_moves + 1):
        for j in range(i + min_moves, n + 1):
            seg = am[i:j]
            m = j - i
            xs = np.arange(m, dtype=float)
            xm = xs.mean()
            sxx = ((xs - xm) ** 2).sum()
            if sxx == 0:
                continue
            slope = ((xs - xm) * (seg - seg.mean())).sum() / sxx
            if slope >= 0:
                continue
            sstot = ((seg - seg.mean()) ** 2).sum()
            ssreg = slope * slope * sxx
            r2 = ssreg / sstot if sstot > 0 else 0
            if r2 < min_r2:
                continue
            windows.append((i, j, ssreg))
    if not windows:
        return []
    windows.sort(key=lambda w: w[1])

    dp = [0.0] * (n + 1)
    choice: list[list] = [[] for _ in range(n + 1)]
    wi = 0
    for pos in range(1, n + 1):
        dp[pos] = dp[pos - 1]
        choice[pos] = choice[pos - 1]
        while wi < len(windows) and windows[wi][1] == pos:
            w = windows[wi]
            val = dp[w[0]] + w[2]
            if val > dp[pos]:
                dp[pos] = val
                choice[pos] = choice[w[0]] + [w]
            wi += 1
    return [(w[0], w[1]) for w in choice[n]]


def _sequence_effect(movements: list[dict], key: str, mode: str) -> dict | None:
    """Per-subject "sequence effect" for one parameter under the
    selected ``mode`` (matches the individual-page Sequence Calculation
    options).  Returns ``{"r2": …, "slope": …}`` where slope is the
    linear slope (linear modes) or exponential rate (exp modes); multi
    modes aggregate both across amplitude-detected sequences
    (length-weighted mean).  ``none`` / no data → None.
    """
    if mode == "none":
        return None
    flip = key in ("peak_close_vel", "mean_close_vel")

    def _series(rng=None):
        xs, ys = [], []
        idxs = range(len(movements)) if rng is None else rng
        for i in idxs:
            v = movements[i].get(key)
            if v is None:
                continue
            xs.append(float(i))
            ys.append(-v if flip else v)
        return xs, ys

    is_exp = mode.startswith("exp_")
    fit = _exp_fit if is_exp else _linreg_fit  # → (slope_or_rate, r2)

    if mode.endswith("_multi"):
        amps = [m.get("amplitude") for m in movements]
        if any(a is None for a in amps):
            amps = [a if a is not None else float("nan") for a in amps]
        wins = _optimize_sequences(amps)
        if not wins:
            return None
        s_acc = r_acc = w_acc = 0.0
        for a, b in wins:
            xs, ys = _series(range(a, b))
            if len(xs) < 2:
                continue
            res = fit(xs, ys)
            if res is None:
                continue
            s, r = res
            w = len(xs)
            s_acc += s * w
            r_acc += r * w
            w_acc += w
        if w_acc == 0:
            return None
        return {"slope": round(s_acc / w_acc, 6), "r2": round(r_acc / w_acc, 6)}

    xs, ys = _series()
    if mode.endswith("_first10"):
        xs, ys = xs[:10], ys[:10]
    res = fit(xs, ys)
    if res is None:
        return None
    s, r = res
    return {"slope": round(s, 6), "r2": round(r, 6)}


# ── API endpoints ───────────────────────────────────────────────────────

def _preview_cache_path():
    """Path to the preview distances JSON cache file."""
    from .subjects import get_settings
    settings = get_settings()
    return settings.dlc_path.parent / ".preview_distances.json"


def _compute_preview_for_subject(subj_name: str, subj_id: int) -> dict | None:
    """Compute 10-second distance preview for one subject.

    Strategy: take the last 10 seconds of the last trial.  If that
    window is entirely null (the chosen auto-source has no values in
    that segment — common for ``corrections`` / ``skeleton_v1`` sources
    where a subject's last trial was never corrected or fit), walk
    backwards trial-by-trial until we find a trial whose last 10s has
    any valid value.  As a final fallback, sample the 10s ending at
    the *last frame with valid data* across the entire trace.
    """
    try:
        distances, trials, source = _load_distances_and_trials(subj_name)
        if not distances or not trials:
            return None

        def _segment_for_trial(t):
            fps = t.get("fps", 60)
            start = t["start_frame"]
            end = t["end_frame"]
            window_frames = int(10 * fps)
            win_start = max(start, end + 1 - window_frames)
            win_end = min(end + 1, len(distances))
            return fps, distances[win_start:win_end]

        chosen_trial = None
        segment = None
        fps = 60

        # 1. Last trial's last 10s (the historical default).
        # 2. Walk backwards until a trial has any valid value in its
        #    tail window.
        for t in reversed(trials):
            fps, seg = _segment_for_trial(t)
            if seg and any(d is not None for d in seg):
                chosen_trial = t
                segment = seg
                break

        # 3. Fallback: locate the last frame with valid data anywhere
        #    in the trace and sample 10s ending there.  Handles subjects
        #    whose tail of every trial is empty (sparse mid-trial only).
        if segment is None:
            last_valid = None
            for i in range(len(distances) - 1, -1, -1):
                if distances[i] is not None:
                    last_valid = i
                    break
            if last_valid is None:
                return None
            # Identify the trial containing that frame for fps + name.
            containing = None
            for t in trials:
                if t["start_frame"] <= last_valid <= t["end_frame"]:
                    containing = t
                    break
            if containing is None:
                containing = trials[-1]
            chosen_trial = containing
            fps = containing.get("fps", 60)
            window_frames = int(10 * fps)
            win_start = max(0, last_valid + 1 - window_frames)
            segment = distances[win_start:last_valid + 1]

        if len(segment) > 200:
            step = max(1, len(segment) // 200)
            segment = segment[::step]

        values = [round(d, 1) if d is not None else None for d in segment]

        return {
            "values": values,
            "fps": fps,
            "source": source,
            "trial_name": chosen_trial.get("trial_name", ""),
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

        # Newest MediaPipe-output mtime across per-trial files; falls back
        # to the legacy combined file when present.
        current_mtime = _mp_newest_mtime(settings, name)

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
        mtime = _mp_newest_mtime(settings, subj["name"])

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
def get_traces(subject_id: int, source: str = Query("auto")) -> dict:
    """Distance and velocity traces split by trial for the Distances tab."""
    subj = _get_subject(subject_id)
    subject_name = subj["name"]

    distances, trials, data_source = _load_distances_and_trials(subject_name, source)
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

    # Check which sources are available (only the three the
    # dropdown exposes — corrections, mp_combined, mp_forward).
    available_sources = []
    for src in ("corrections", "mp_combined", "mp_forward"):
        if _try_source_quick(subject_name, src):
            available_sources.append(src)

    return {"trials": result_trials, "subject": subject_name, "y_range": y_range, "data_source": data_source, "available_sources": available_sources}


@router.get("/{subject_id}/movements")
def get_movements(subject_id: int, source: str = Query("auto")) -> dict:
    """Per-movement parameters for the Movements tab."""
    subj = _get_subject(subject_id)
    subject_name = subj["name"]

    distances, trials, data_source = _load_distances_and_trials(subject_name, source)
    events, event_source = _load_events(subject_name, distances, trials)

    movements, trial_names = _build_movement_params(distances, events, trials)

    return {
        "movements": movements,
        "trial_names": trial_names,
        "subject": subject_name,
        "event_source": event_source,
        "data_source": data_source,
    }


@router.post("/group/regenerate")
def regenerate_group_cache(include_auto: bool = Query(False),
                             source: str = Query("auto"),
                             seq_mode: str = Query("linear_full"),
                             hand: str = Query("more"),
                             trial: str = Query("last")) -> dict:
    """Force a fresh recompute of the group-comparison data for the
    given query combo and overwrite the on-disk cache JSON.  Useful
    after editing a subject's events — drop in a new copy of the
    cached file so subsequent visits to the page (here or on the
    static site) reflect the change."""
    cache = _group_cache_path(include_auto, source, seq_mode, hand, trial)
    if cache is not None and cache.is_file():
        try:
            cache.unlink()
        except OSError:
            pass
    # Drop the in-memory per-subject movement cache too (if the
    # export-cache flag happens to be on) so the recompute starts from
    # disk events.
    try:
        _EXPORT_MOVE_CACHE.clear()
    except Exception:
        pass
    data = get_group_comparison(
        include_auto=include_auto, source=source,
        seq_mode=seq_mode, hand=hand, trial=trial,
    )
    if cache is not None:
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(_json.dumps(data))
        except OSError:
            pass
    return data


@router.get("/group")
def get_group_comparison(include_auto: bool = Query(False),
                          source: str = Query("auto"),
                          seq_mode: str = Query("linear_full"),
                          hand: str = Query("more"),
                          trial: str = Query("last")) -> dict:
    """Aggregated per-subject statistics grouped by diagnosis.

    ``seq_mode`` controls the per-subject sequence-effect value (same
    options as the individual Sequence Calculation dropdown):
      none, linear_full, linear_first10, linear_multi,
      exp_full, exp_first10, exp_multi.
    The value is the R² (goodness of fit) of the decrement model;
    multi modes aggregate the per-sequence R² (length-weighted mean).

    ``hand`` ∈ {more, less, L, R, average} selects which hand's trials
    contribute; ``trial`` ∈ {first, last, average} selects which
    trial(s) within each chosen hand.
    """
    # Cache fast path: every combination is exported to site/data/.
    # Saves the multi-second aggregation pass when the user is just
    # browsing the Group Comparison page.  Falls through if the cache
    # is missing, unreadable, or stale (predates newer fields).
    cache = _group_cache_path(include_auto, source, seq_mode, hand, trial)
    if cache is not None and cache.is_file():
        try:
            data = _json.loads(cache.read_text())
            subjs = data.get("subjects") or []
            if subjs and "variance_amplitude" not in subjs[0]:
                # variance (σ) = |cv * mean| — derive instead of recompute.
                for s in subjs:
                    for k in list(s.keys()):
                        if not k.startswith("cv_"):
                            continue
                        base = k[3:]
                        cv_v = s.get(k)
                        m_v = s.get(f"mean_{base}")
                        if cv_v is None or m_v is None:
                            s[f"variance_{base}"] = None
                        else:
                            s[f"variance_{base}"] = round(abs(float(cv_v) * float(m_v)), 4)
            # Movement-similarity keys can't be derived; fall through
            # to live compute if any cached subject is missing them.
            if subjs and "movement_similarity" not in subjs[0]:
                raise RuntimeError("cache missing movement_similarity")
            return data
        except Exception:
            pass

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
            saved_events = _read_events_csv(subject_name)
            has_saved = any(len(v) > 0 for v in saved_events.values())
            # "Complete" = at least one open, peak, AND close saved.
            has_complete = all(len(saved_events.get(t, [])) > 0
                               for t in ("open", "peak", "close"))
            # Per-(subject, source, include_auto) movements are
            # independent of seq_mode/hand/trial; cache them during a
            # static export so the many combos don't rebuild every time.
            # Gated by MT_EXPORT_CACHE so the live app is unaffected.
            ckey = (subject_name, source, bool(include_auto))
            cached = _EXPORT_MOVE_CACHE.get(ckey) if _EXPORT_CACHE_ON else None
            if cached is not None and len(cached) >= 4:
                trials, all_movements, event_source, similarities = cached
            else:
                distances, trials, _src = _load_distances_and_trials(subject_name, source)
                if include_auto:
                    events, event_source = _load_events(subject_name, distances, trials)
                else:
                    events = saved_events
                    event_source = "saved" if has_saved else "none"
                all_movements, _ = _build_movement_params(distances, events, trials)
                # Per-trial movement-similarity (4 alignments).  Computed
                # once per (subject, source, include_auto); aggregation
                # over hand+trial selection happens below.
                similarities = _movement_similarity_per_trial(
                    distances, trials, all_movements)
                if _EXPORT_CACHE_ON:
                    _EXPORT_MOVE_CACHE[ckey] = (trials, all_movements,
                                                  event_source, similarities)
            # Larger / Smaller SE: pick which hand based on each hand's
            # last-trial sequence-effect slope (amplitude key, current
            # seq_mode) before applying the standard trial selector.
            if hand in ("larger_se", "smaller_se"):
                chosen = _se_chosen_hand(trials, all_movements, seq_mode,
                                          prefer_larger=(hand == "larger_se"))
                effective_hand = chosen if chosen else hand   # unresolved → empty set
            else:
                effective_hand = hand
            # Restrict to the selected hand + trial(s).  An empty
            # selection (no trials for the chosen hand, or SE unresolved)
            # excludes the subject via the len<2 check below.
            sel_idx = _select_trial_indices(trials, effective_hand, trial,
                                            subj.get("laterality"))
            movements = [m for m in all_movements if m.get("trial_idx") in sel_idx]
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
            "has_saved_events": has_saved,
            "has_complete_events": has_complete,
            "last_dose_raw": subj.get("last_dose") or None,
            "time_since_dose_min": _parse_last_dose_minutes(subj.get("last_dose")),
            "levodopa_off": _is_levodopa_off(subj.get("levodopa")),
            "laterality_raw": subj.get("laterality") or None,
            "laterality_side": _laterality_more_side(subj.get("laterality")),
        }

        # Movement-similarity (nanmean upper-triangle r) averaged
        # across the trials matching the hand+trial selection.
        _SIM_KEYS = (
            ("movement_similarity",       "corr"),
            ("movement_similarity_open",  "open"),
            ("movement_similarity_peak",  "peak"),
            ("movement_similarity_close", "close"),
        )
        for out_key, align_key in _SIM_KEYS:
            vals = []
            for ti in sel_idx:
                v = (similarities.get(ti) or {}).get(align_key)
                if v is not None and math.isfinite(v):
                    vals.append(float(v))
            entry[out_key] = round(float(np.mean(vals)), 4) if vals else None

        for key in PARAM_KEYS:
            vals = [m[key] for m in movements if m[key] is not None]
            if vals:
                mean_v = np.mean(vals)
                std_v = np.std(vals)
                entry[f"mean_{key}"] = round(float(mean_v), 4)
                entry[f"variance_{key}"] = round(float(std_v), 4)
                entry[f"cv_{key}"] = round(float(std_v / abs(mean_v)), 4) if mean_v != 0 else None

                # Sequence effect under the selected mode (closing
                # velocities are negated inside the helper so the effect
                # reflects changes in magnitude).  Expose both R² (the
                # group "Sequence Effect" row) and slope (explore page).
                se = _sequence_effect(movements, key, seq_mode)
                entry[f"seq_{key}"] = se["r2"] if se else None
                entry[f"seqslope_{key}"] = se["slope"] if se else None
            else:
                entry[f"mean_{key}"] = None
                entry[f"variance_{key}"] = None
                entry[f"cv_{key}"] = None
                entry[f"seq_{key}"] = None
                entry[f"seqslope_{key}"] = None

        # Frequency = 1 / mean IMI
        mean_imi = entry.get("mean_imi")
        entry["frequency"] = round(1.0 / mean_imi, 4) if mean_imi and mean_imi > 0 else None

        results.append(entry)

    # Collect unique groups
    groups = sorted(set(r["diagnosis"] for r in results))

    return {"subjects": results, "groups": groups}


# Numeric clinical variables exposed by the explore tool.
_CLINICAL_VARS = [
    ("age", "Age (years)"),
    ("disease_duration", "Disease Duration (years)"),
    ("time_since_dose_min", "Time Since Dose (min)"),
    ("hand_size_left", "Hand Size — Left (mm)"),
    ("hand_size_right", "Hand Size — Right (mm)"),
]

# Per-movement parameter base labels (units in parentheses).
_MOVE_PARAM_LABELS = {
    "imi": "IMI (s)",
    "amplitude": "Amplitude (mm)",
    "rel_amplitude": "Rel. Amplitude",
    "power": "Power (mm²/s)",
    "peak_open_vel": "Peak Open Vel (mm/s)",
    "peak_close_vel": "Peak Close Vel (mm/s)",
    "mean_open_vel": "Mean Open Vel (mm/s)",
    "mean_close_vel": "Mean Close Vel (mm/s)",
}


def _explore_variable_catalog() -> list[dict]:
    """Catalog of plottable variables, one entry per base variable.

    Movement parameters are listed once with ``aggregatable: True``;
    the UI adds Mean/CV/Sequence-effect radios and forms the data key
    as ``{agg}_{key}``.  Clinical fields and frequency are scalar.
    """
    out = [{"key": k, "label": lbl, "category": "clinical", "aggregatable": False}
           for k, lbl in _CLINICAL_VARS]
    for k, lbl in _MOVE_PARAM_LABELS.items():
        out.append({"key": k, "label": lbl, "category": "movement",
                    "aggregatable": True})
    out.append({"key": "frequency", "label": "Frequency (Hz)",
                "category": "movement", "aggregatable": False})
    for k, lbl in (
        ("movement_similarity",       "Movement Similarity"),
        ("movement_similarity_open",  "Movement Similarity (open)"),
        ("movement_similarity_peak",  "Movement Similarity (peak)"),
        ("movement_similarity_close", "Movement Similarity (close)"),
    ):
        out.append({"key": k, "label": lbl, "category": "movement",
                    "aggregatable": False})
    return out


def _explore_value_keys() -> list[str]:
    """Underlying per-subject value keys the UI looks up."""
    keys = [k for k, _ in _CLINICAL_VARS] + ["frequency"]
    for k in _MOVE_PARAM_LABELS:
        keys += [f"mean_{k}", f"variance_{k}", f"cv_{k}", f"seq_{k}", f"seqslope_{k}"]
    keys += ["movement_similarity", "movement_similarity_open",
             "movement_similarity_peak", "movement_similarity_close"]
    return keys


# Combinations exported to the static site (see scripts/export_results_static.py).
# When the live request matches one of these, the endpoint returns the cached
# JSON instead of recomputing — same trade-off the static site makes.
_SITE_DATA_DIR = Path(__file__).resolve().parents[2] / "site" / "data"

# Group page: every combination is exported.
_GROUP_STATIC_SOURCES   = {"auto", "corrections", "mp_combined", "mp_forward"}
_GROUP_STATIC_SEQ_MODES = {"none", "linear_full", "linear_first10", "linear_multi",
                            "exp_full",    "exp_first10",    "exp_multi"}
_GROUP_STATIC_HANDS     = {"more", "less", "L", "R", "average"}
_GROUP_STATIC_TRIALS    = {"first", "last", "average"}

# Explore page: smaller subset (no source≠auto, no seq=none, no L/R hand).
_EXPLORE_STATIC_SOURCES   = {"auto"}
_EXPLORE_STATIC_SEQ_MODES = {"linear_full", "linear_first10", "linear_multi",
                              "exp_full",    "exp_first10",    "exp_multi"}
_EXPLORE_STATIC_HANDS     = {"more", "less", "average"}
_EXPLORE_STATIC_TRIALS    = {"first", "last", "average"}


def _static_cache_path(endpoint: str, include_auto: bool,
                        source: str, seq_mode: str, hand: str, trial: str,
                        srcs: set, sms: set, hands: set, trials: set):
    """Return Path to a cached JSON for the given combo, or None if the
    combination wasn't exported (no cache to read)."""
    if not include_auto: return None
    if source not in srcs: return None
    if seq_mode not in sms: return None
    if hand not in hands: return None
    if trial not in trials: return None
    name = (f"api_results_{endpoint}_include_auto_true_source_{source}_"
            f"seq_mode_{seq_mode}_hand_{hand}_trial_{trial}.json")
    return _SITE_DATA_DIR / name


def _explore_cache_path(include_auto, source, seq_mode, hand, trial):
    return _static_cache_path("explore", include_auto, source, seq_mode, hand, trial,
                              _EXPLORE_STATIC_SOURCES, _EXPLORE_STATIC_SEQ_MODES,
                              _EXPLORE_STATIC_HANDS, _EXPLORE_STATIC_TRIALS)


def _group_cache_path(include_auto, source, seq_mode, hand, trial):
    return _static_cache_path("group", include_auto, source, seq_mode, hand, trial,
                              _GROUP_STATIC_SOURCES, _GROUP_STATIC_SEQ_MODES,
                              _GROUP_STATIC_HANDS, _GROUP_STATIC_TRIALS)


@router.get("/explore")
def get_explore_variables(include_auto: bool = Query(False),
                           source: str = Query("auto"),
                           seq_mode: str = Query("linear_full"),
                           hand: str = Query("more"),
                           trial: str = Query("last")) -> dict:
    """Per-subject clinical + movement variables for the explore tool.

    Reuses the group aggregation for movement variables (honoring the
    same hand/trial selection) and merges in numeric clinical fields,
    returning a flat ``vars`` dict per subject plus a variable catalog
    the UI builds its dropdowns from.
    """
    # Cache check: for the small subset of (source × seq_mode × hand ×
    # trial) combinations that the static-site exporter writes out, just
    # return the pre-computed JSON.  Everything else (or a stale cache
    # missing newer fields) falls through to live computation below.
    cache = _explore_cache_path(include_auto, source, seq_mode, hand, trial)
    if cache is not None and cache.is_file():
        try:
            data = _json.loads(cache.read_text())
            # Older caches predate the `variance_*` keys.  Variance (σ) is
            # just |cv * mean|, so derive it on the fly instead of forcing
            # a full re-export.
            subjs = data.get("subjects") or []
            if subjs and "variance_amplitude" not in (subjs[0].get("vars") or {}):
                for s in subjs:
                    vars_ = s.get("vars") or {}
                    for k in list(vars_.keys()):
                        if not k.startswith("cv_"):
                            continue
                        base = k[3:]
                        cv_v = vars_.get(k)
                        m_v = vars_.get(f"mean_{base}")
                        if cv_v is None or m_v is None:
                            vars_[f"variance_{base}"] = None
                        else:
                            vars_[f"variance_{base}"] = round(abs(float(cv_v) * float(m_v)), 4)
                    s["vars"] = vars_
            # Ensure the catalog advertises the variance keys.
            cat = data.get("catalog") or {}
            if "variance" not in (cat.get("aggregators") or []):
                cat.setdefault("aggregators", []).append("variance")
                data["catalog"] = cat
            # Movement-similarity keys can't be derived from cached
            # values; fall through to live compute when missing.
            if subjs and "movement_similarity" not in (subjs[0].get("vars") or {}):
                raise RuntimeError("cache missing movement_similarity")
            return data
        except Exception:
            pass    # corrupt cache → recompute

    grp = get_group_comparison(include_auto=include_auto, source=source,
                               seq_mode=seq_mode, hand=hand, trial=trial)
    catalog = _explore_variable_catalog()
    var_keys = _explore_value_keys()

    # Numeric clinical fields by subject name (incl. median hand sizes).
    _DB_CLINICAL = ("age", "disease_duration", "hand_size_left", "hand_size_right")
    with get_db_ctx() as db:
        rows = db.execute(
            "SELECT name, age, disease_duration, hand_size_left, hand_size_right "
            "FROM subjects"
        ).fetchall()
    clinical = {}
    for r in rows:
        def _num(x):
            try:
                return float(x) if x is not None and str(x).strip() != "" else None
            except (ValueError, TypeError):
                return None
        clinical[r["name"]] = {c: _num(r[c]) for c in _DB_CLINICAL}

    subjects = []
    for s in grp["subjects"]:
        cl = clinical.get(s["name"], {})
        vars_ = {}
        for k in var_keys:
            if k in _DB_CLINICAL:
                vars_[k] = cl.get(k)
            else:
                v = s.get(k)
                vars_[k] = v if isinstance(v, (int, float)) else None
        subjects.append({
            "name": s["name"],
            "group": s.get("diagnosis") or "Control",
            "has_saved_events": s.get("has_saved_events", False),
            "has_complete_events": s.get("has_complete_events", False),
            "laterality_side": s.get("laterality_side"),
            "vars": vars_,
        })

    return {"groups": grp["groups"], "subjects": subjects, "variables": catalog}
