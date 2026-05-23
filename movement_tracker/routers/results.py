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


def _laterality_more_side(lat) -> str | None:
    """More-affected side ('L'/'R') from a clinical laterality value.

    Handles 'L', 'R', 'left', 'right', and 'L>R' / 'R>L' (more-affected
    side listed first).  Bilateral / none / blank → None (unknown).
    """
    if lat is None:
        return None
    s = str(lat).strip().lower()
    if not s or s in ("none", "b/l", "bilateral", "n/a", "na"):
        return None
    if ">" in s:
        s = s.split(">")[0].strip()
    if s.startswith("l"):
        return "L"
    if s.startswith("r"):
        return "R"
    return None


def _hand_of_trial(trial_name: str) -> str | None:
    """'L'/'R' from a trial name like 'Con03_L1'."""
    suffix = str(trial_name).split("_")[-1]
    c = suffix[:1].upper()
    return c if c in ("L", "R") else None


def _select_trial_indices(trials: list[dict], hand: str, trial_sel: str,
                          laterality) -> set[int]:
    """Which trial indices to include for the chosen hand + trial mode.

    hand ∈ {more, less, L, R, average};
    trial_sel ∈ {first, last, average}.  Returns an empty set when the
    selection can't be resolved (e.g. unknown more-affected side).
    """
    by_hand: dict[str, list[int]] = {"L": [], "R": []}
    for i, t in enumerate(trials):
        h = _hand_of_trial(t.get("trial_name", ""))
        if h:
            by_hand[h].append(i)

    more = _laterality_more_side(laterality)
    if hand == "more":
        hands = [more] if more else []
    elif hand == "less":
        hands = [("R" if more == "L" else "L")] if more else []
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

        movements.append({
            "peak_frame": pk,
            "peak_dist": round(pk_dist, 2) if pk_dist is not None else None,
            "peak_time": peak_time,
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
            if cached is not None:
                trials, all_movements, event_source = cached
            else:
                distances, trials, _src = _load_distances_and_trials(subject_name, source)
                if include_auto:
                    events, event_source = _load_events(subject_name, distances, trials)
                else:
                    events = saved_events
                    event_source = "saved" if has_saved else "none"
                all_movements, _ = _build_movement_params(distances, events, trials)
                if _EXPORT_CACHE_ON:
                    _EXPORT_MOVE_CACHE[ckey] = (trials, all_movements, event_source)
            # Restrict to the selected hand + trial(s).  An empty
            # selection (unknown affected side, or no trials for the
            # chosen hand) excludes the subject via the len<2 check.
            sel_idx = _select_trial_indices(trials, hand, trial,
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

        for key in PARAM_KEYS:
            vals = [m[key] for m in movements if m[key] is not None]
            if vals:
                mean_v = np.mean(vals)
                std_v = np.std(vals)
                entry[f"mean_{key}"] = round(float(mean_v), 4)
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
    return out


def _explore_value_keys() -> list[str]:
    """Underlying per-subject value keys the UI looks up."""
    keys = [k for k, _ in _CLINICAL_VARS] + ["frequency"]
    for k in _MOVE_PARAM_LABELS:
        keys += [f"mean_{k}", f"cv_{k}", f"seq_{k}", f"seqslope_{k}"]
    return keys


# Combinations exported to the static site (see scripts/export_results_static.py).
# When the live request matches one of these, the endpoint returns the cached
# JSON instead of recomputing — same trade-off the static site makes.
_EXPLORE_STATIC_SOURCES   = {"auto"}
_EXPLORE_STATIC_SEQ_MODES = {"linear_full", "linear_first10", "linear_multi",
                              "exp_full",    "exp_first10",    "exp_multi"}
_EXPLORE_STATIC_HANDS     = {"more", "less", "average"}
_EXPLORE_STATIC_TRIALS    = {"first", "last", "average"}
_SITE_DATA_DIR = Path(__file__).resolve().parents[2] / "site" / "data"


def _explore_cache_path(include_auto: bool, source: str, seq_mode: str,
                         hand: str, trial: str) -> "Path | None":
    if not include_auto:
        return None
    if source not in _EXPLORE_STATIC_SOURCES: return None
    if seq_mode not in _EXPLORE_STATIC_SEQ_MODES: return None
    if hand not in _EXPLORE_STATIC_HANDS: return None
    if trial not in _EXPLORE_STATIC_TRIALS: return None
    name = (f"api_results_explore_include_auto_true_source_{source}_"
            f"seq_mode_{seq_mode}_hand_{hand}_trial_{trial}.json")
    return _SITE_DATA_DIR / name


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
    # return the pre-computed JSON.  Everything else falls through to
    # live computation below.
    cache = _explore_cache_path(include_auto, source, seq_mode, hand, trial)
    if cache is not None and cache.is_file():
        try:
            return _json.loads(cache.read_text())
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
