"""MediaPipe error detection: scoring + detection + camera attribution.

Pipeline
--------
1.  **Detection scores** (per joint, per frame) identify *suspicious* 3D labels.
    Factors: z-jump, y-disparity, bone-length, bone-length-agreement
    (pairwise-distance consistency), joint-angle violation, MP confidence.
2.  **Attribution scores** (per joint, per frame, per camera) decide *which*
    camera is to blame when a joint is flagged.  Factors: reprojection
    residual, 2D jump, MP confidence.
3.  Each factor produces a percentile-normalised score σ ∈ [0, 1] (1 = most
    anomalous).  A slider value s ∈ [0, 1] is the flagging threshold:
    ``flag = (σ > 1 - s)``.  Factors are combined with OR.
4.  Final error matrix: (N_frames, 21, 2)  — True iff joint is mislabeled
    on that camera.

Edge cases / normalisation
--------------------------
- Percentile ranks computed globally across the whole (frame × joint)
  population, ignoring NaN entries.
- Wrist (j=0) is excluded from bone-length factors because there's no parent.
- Factors that depend on 3D require calibration; skipped when missing.
- Factors that require confidence return ``None`` when the per-camera
  confidence arrays aren't available.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .calibration import triangulate_points
from .mano_data import HAND_SKELETON, _mano_dir

logger = logging.getLogger(__name__)

BONES = HAND_SKELETON  # 20 (parent, child) tuples

DETECTION_FACTORS = ("z_jump", "z_outlier", "y_disp", "bone_length",
                     "bone_agreement", "angle", "reproj", "confidence",
                     "hrnet_mismatch", "stereo_dist")
ATTRIBUTION_FACTORS = ("jump_2d", "confidence", "hrnet")

# Distal descendants for each joint along the finger chains.  A joint-angle
# violation at joint j is attributed to its descendants — the downstream
# joints whose 3D position depends on j's angle — not to j itself.
JOINT_DESCENDANTS = {
    0:  [],           # wrist root — don't propagate globally
    1:  [2, 3, 4],    # thumb CMC
    2:  [3, 4],       # thumb MCP
    3:  [4],          # thumb IP
    4:  [],
    5:  [6, 7, 8],    # index MCP
    6:  [7, 8],       # index PIP
    7:  [8],          # index DIP
    8:  [],
    9:  [10, 11, 12], # middle MCP
    10: [11, 12],
    11: [12],
    12: [],
    13: [14, 15, 16], # ring MCP
    14: [15, 16],
    15: [16],
    16: [],
    17: [18, 19, 20], # pinky MCP
    18: [19, 20],
    19: [20],
    20: [],
}


# ─── Percentile normalisation ──────────────────────────────────────────────

def _percentile_rank(arr: np.ndarray) -> np.ndarray:
    """Rank-normalise ``arr`` to [0, 1]; NaN entries stay NaN.

    Shape is preserved.  Ranks are averaged for ties.  "1.0" means the most
    anomalous entry in the whole array.
    """
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    flat = arr.reshape(-1).astype(np.float64)
    valid = ~np.isnan(flat)
    vals = flat[valid]
    n = len(vals)
    if n < 2:
        return out
    order = np.argsort(vals, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n)
    ranks /= max(n - 1, 1)
    out_flat = out.reshape(-1)
    out_flat[valid] = ranks.astype(np.float32)
    return out


def _percentile_rank_per_joint(arr: np.ndarray) -> np.ndarray:
    """Like ``_percentile_rank`` but ranks each joint's column independently.

    For a (N, 21) array, each joint's frames are ranked against ONLY that
    joint's own distribution.  This means a stable joint with a small
    cluster of unusual frames produces high percentiles for those frames,
    even if a different joint has many larger scores in absolute terms.
    """
    if arr.ndim != 2:
        return _percentile_rank(arr)
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    for j in range(arr.shape[1]):
        col = arr[:, j].astype(np.float64)
        valid = ~np.isnan(col)
        vals = col[valid]
        n = len(vals)
        if n < 2: continue
        order = np.argsort(vals, kind="stable")
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(n)
        ranks /= max(n - 1, 1)
        col_out = out[:, j]
        col_out[valid] = ranks.astype(np.float32)
        out[:, j] = col_out
    return out


# ─── Robust temporal helpers ───────────────────────────────────────────────

TEMPORAL_WINDOW = 15    # ±frames around current frame
MIN_SIGMA_3D = 2.0      # mm
MIN_SIGMA_2D = 3.0      # pixels
TRUST_PCT = 0.85        # frame-local scores above this percentile mark the
                        # frame as untrusted (excluded from temporal baselines)


def _build_trust_mask(frame_local_scores: list) -> np.ndarray:
    """OR-combine frame-local score percentile flags into an (N, 21) bool mask.

    ``frame_local_scores`` is a list of (N, 21) raw score arrays.  A joint-frame
    in the TRUST_PCT-tail of ANY factor is marked untrusted.  Used to exclude
    contaminated frames from temporal baselines.
    """
    if not frame_local_scores:
        shape = (0, 21)
        for s in frame_local_scores:
            if s is not None:
                shape = s.shape
                break
        return np.zeros(shape, dtype=bool)
    out = None
    for s in frame_local_scores:
        if s is None:
            continue
        flat = s.reshape(-1)
        valid = ~np.isnan(flat)
        if valid.sum() < 2:
            continue
        thresh = np.quantile(flat[valid], TRUST_PCT)
        flag = (s > thresh) & ~np.isnan(s)
        out = flag if out is None else (out | flag)
    if out is None:
        # no factors available → nobody is untrusted
        shape = frame_local_scores[0].shape if frame_local_scores[0] is not None else (0, 21)
        return np.zeros(shape, dtype=bool)
    return out


def _robust_deviation_1d(series: np.ndarray,
                         untrusted: np.ndarray,
                         min_sigma: float) -> np.ndarray:
    """|x[f] − predicted(f)| / MAD(fit residuals), per frame.

    Fits a low-order polynomial in time to trusted neighbours (excluding
    the current frame), predicts the value at f, and divides the residual
    by the MAD of the fit residuals against the neighbours themselves.
    Falls back to median+MAD if too few neighbours to fit.

    Why polynomial-fit: during smooth motion, the median of ±15 frames is
    the midpoint of the arc; a correct current position looks "offset from
    the median" and a large MAD has to absorb it.  A polynomial fit captures
    the expected trajectory, so sigma genuinely measures noise around the
    trajectory rather than the range of motion.
    """
    N = len(series)
    out = np.zeros(N, dtype=np.float32)
    if N == 0:
        return out
    W = TEMPORAL_WINDOW
    for f in range(N):
        lo, hi = max(0, f - W), min(N, f + W + 1)
        window = series[lo:hi]
        mask = ~untrusted[lo:hi] & ~np.isnan(window)
        idx_f = f - lo
        if 0 <= idx_f < len(mask):
            mask[idx_f] = False
        vals = window[mask]
        if len(vals) < 3:
            out[f] = 0.0
            continue
        cur = series[f]
        if np.isnan(cur):
            out[f] = 0.0
            continue
        ts = np.where(mask)[0].astype(np.float32)
        # Degree 2 if enough points, degree 1 otherwise
        deg = 2 if len(vals) >= 5 else 1
        try:
            coef = np.polyfit(ts, vals, deg)
            pred_cur = float(np.polyval(coef, idx_f))
            fit = np.polyval(coef, ts)
            fit_err = np.abs(fit - vals)
        except (np.linalg.LinAlgError, ValueError):
            # Fall back to median+MAD
            med = np.median(vals)
            out[f] = abs(cur - med) / max(np.median(np.abs(vals - med)) * 1.4826, min_sigma)
            continue
        sigma = max(float(np.median(fit_err)) * 1.4826, min_sigma)
        out[f] = abs(cur - pred_cur) / sigma
    return out


def _adjacent_jump_1d(series: np.ndarray,
                       untrusted: np.ndarray,
                       min_sigma: float,
                       skip_mask: np.ndarray | None = None) -> np.ndarray:
    """Frame-to-frame jump magnitude, scaled by robust MAD of all deltas.

    Guarantees that a sudden discontinuity is caught even if a polynomial
    fit smooths through it.  Attributes the same score to both sides of
    each adjacent pair so both f and f-1 get flagged.

    ``untrusted`` is kept for interface parity but NOT used to skip deltas
    here (we want the safety-net property, especially at trial boundaries).

    ``skip_mask`` (optional): if provided, deltas where EITHER side is in
    the mask are skipped AND not scored on either endpoint.  Use this to
    exclude known-bad frames (e.g. already-detected Z outliers) so their
    deltas don't contaminate neighbouring frames with spurious jump scores.
    """
    N = len(series)
    out = np.zeros(N, dtype=np.float32)
    if N < 2:
        return out
    d = np.full(N - 1, np.nan, dtype=np.float32)
    for i in range(N - 1):
        a, b = series[i], series[i + 1]
        if np.isnan(a) or np.isnan(b):
            continue
        if skip_mask is not None and (skip_mask[i] or skip_mask[i + 1]):
            continue
        d[i] = abs(b - a)
    valid = ~np.isnan(d)
    if valid.sum() < 3:
        return out
    med = float(np.median(d[valid]))
    mad = float(np.median(np.abs(d[valid] - med)))
    sigma = max(mad * 1.4826, min_sigma)
    scores = np.where(np.isnan(d), 0.0, d / sigma)
    for i in range(N - 1):
        s = scores[i]
        if s > out[i]:     out[i] = s
        if s > out[i + 1]: out[i + 1] = s
    return out


def _adjacent_jump_2d(series: np.ndarray,
                       untrusted: np.ndarray,
                       min_sigma: float) -> np.ndarray:
    """2D analogue of _adjacent_jump_1d using Euclidean deltas.
    Does not skip untrusted frames (see _adjacent_jump_1d rationale)."""
    N = series.shape[0]
    out = np.zeros(N, dtype=np.float32)
    if N < 2:
        return out
    d = np.full(N - 1, np.nan, dtype=np.float32)
    for i in range(N - 1):
        a, b = series[i], series[i + 1]
        if not (np.isnan(a[0]) or np.isnan(b[0])):
            d[i] = float(np.hypot(b[0] - a[0], b[1] - a[1]))
    valid = ~np.isnan(d)
    if valid.sum() < 3:
        return out
    med = float(np.median(d[valid]))
    mad = float(np.median(np.abs(d[valid] - med)))
    sigma = max(mad * 1.4826, min_sigma)
    scores = np.where(np.isnan(d), 0.0, d / sigma)
    for i in range(N - 1):
        s = scores[i]
        if s > out[i]:     out[i] = s
        if s > out[i + 1]: out[i + 1] = s
    return out


def _robust_deviation_2d(series: np.ndarray,
                         untrusted: np.ndarray,
                         min_sigma: float) -> np.ndarray:
    """Polynomial-fit residual for (N, 2) 2D positions.

    Fits x(t) and y(t) separately as quadratics over trusted neighbours,
    predicts (x, y) at current frame, returns Euclidean residual scaled
    by MAD of the 2D fit residuals.
    """
    N = series.shape[0]
    out = np.zeros(N, dtype=np.float32)
    if N == 0:
        return out
    W = TEMPORAL_WINDOW
    for f in range(N):
        lo, hi = max(0, f - W), min(N, f + W + 1)
        window = series[lo:hi]
        mask = ~untrusted[lo:hi] & ~np.isnan(window[:, 0])
        idx_f = f - lo
        if 0 <= idx_f < len(mask):
            mask[idx_f] = False
        pts = window[mask]
        if len(pts) < 3:
            out[f] = 0.0
            continue
        cur = series[f]
        if np.isnan(cur[0]):
            out[f] = 0.0
            continue
        ts = np.where(mask)[0].astype(np.float32)
        deg = 2 if len(pts) >= 5 else 1
        try:
            cx = np.polyfit(ts, pts[:, 0], deg)
            cy = np.polyfit(ts, pts[:, 1], deg)
            pred_x = float(np.polyval(cx, idx_f))
            pred_y = float(np.polyval(cy, idx_f))
            fx = np.polyval(cx, ts)
            fy = np.polyval(cy, ts)
            fit_err = np.sqrt((fx - pts[:, 0]) ** 2 + (fy - pts[:, 1]) ** 2)
        except (np.linalg.LinAlgError, ValueError):
            med = np.median(pts, axis=0)
            out[f] = float(np.linalg.norm(cur - med)) / max(
                float(np.median(np.linalg.norm(pts - med, axis=1))) * 1.4826, min_sigma)
            continue
        sigma = max(float(np.median(fit_err)) * 1.4826, min_sigma)
        out[f] = float(np.hypot(cur[0] - pred_x, cur[1] - pred_y)) / sigma
    return out


# ─── Detection factors (→ [N, 21] raw scores) ──────────────────────────────

def _z_outlier_score(mp_3d: np.ndarray,
                      exclude_mask: np.ndarray | None = None) -> np.ndarray:
    """Per-joint deviation of current Z from the joint's robust video-wide
    Z baseline (median), normalised by MAD.

    ``exclude_mask`` (optional, (N, J) bool): frames that should NOT
    contribute to the median/MAD baseline.  Scores are still computed for
    those frames against the cleaner baseline.  Pass y-disparity-bad frames
    here so corrupted Z values don't pull the baseline.
    """
    N, J, _ = mp_3d.shape
    out = np.zeros((N, J), dtype=np.float32)
    for j in range(J):
        z = mp_3d[:, j, 2]
        valid = ~np.isnan(z)
        if exclude_mask is not None:
            valid = valid & ~exclude_mask[:, j]
        if valid.sum() < 10:
            # Fall back to including excluded frames if we'd otherwise have nothing
            valid = ~np.isnan(z)
            if valid.sum() < 10:
                continue
        med = float(np.median(z[valid]))
        mad = float(np.median(np.abs(z[valid] - med)))
        sigma = max(mad * 1.4826, MIN_SIGMA_3D)
        out[:, j] = np.where(np.isnan(z), 0.0, np.abs(z - med) / sigma)
    return out


def _poly_residual_1d(series: np.ndarray) -> np.ndarray:
    """Per-frame |Z[f] − poly_predict(f)|, raw mm, no sigma scaling.

    Polynomial is degree-2 (degree-1 if too few points), fit to ALL valid
    neighbours in a ±15-frame window EXCLUDING the current frame.  Returns
    a (N,) float array; 0.0 where the residual can't be computed.
    """
    N = len(series)
    out = np.zeros(N, dtype=np.float32)
    if N == 0: return out
    W = TEMPORAL_WINDOW
    for f in range(N):
        cur = series[f]
        if np.isnan(cur):
            continue
        lo, hi = max(0, f - W), min(N, f + W + 1)
        idxs = np.arange(lo, hi)
        mask = ~np.isnan(series[lo:hi]) & (idxs != f)
        if mask.sum() < 3:
            continue
        ts = idxs[mask].astype(np.float64)
        vals = series[lo:hi][mask].astype(np.float64)
        deg = 2 if len(vals) >= 5 else 1
        try:
            coef = np.polyfit(ts, vals, deg)
            pred = float(np.polyval(coef, f))
        except (np.linalg.LinAlgError, ValueError):
            continue
        out[f] = abs(float(cur) - pred)
    return out


def _z_jump_score(mp_3d: np.ndarray,
                   untrusted: np.ndarray,
                   outlier_mask: np.ndarray | None = None) -> np.ndarray:
    """Z-jump score = raw polynomial-fit residual in mm.

    For each (frame, joint), fit a quadratic to all valid Z values in a
    ±15-frame window (excluding the current frame), evaluate at f, and
    return |Z[f] − pred(f)|.  No sigma scaling, no adjacent-frame Δ, no
    trust mask — purely the raw residual.  Per-joint percentile ranking
    in ``compute_scores`` then turns these into 0–1 scores.
    """
    N, J, _ = mp_3d.shape
    out = np.zeros((N, J), dtype=np.float32)
    for j in range(J):
        out[:, j] = _poly_residual_1d(mp_3d[:, j, 2])
    return out


def _y_disparity_score(mp_L: np.ndarray, mp_R: np.ndarray,
                        calib: dict | None = None,
                        mp_3d: np.ndarray | None = None) -> np.ndarray:
    """Y-disparity ERROR per joint per frame.

    Score is ``|(Y_L − Y_R) − ref[j]|`` where ``ref[j]`` is the joint's
    EMPIRICAL median y-disparity across all frames.  The median is robust
    to outliers (typically the wrongly-labelled frames we're trying to
    flag), and using it instead of a calibration-predicted disparity
    sidesteps the small Ty miscalibration that is otherwise indistinguishable
    from a true labeling error to the score.

    ``calib`` and ``mp_3d`` are accepted for API compatibility but no
    longer used — empirical disparity is the right reference because
    triangulation already absorbs Ty bias and so does the corrector's
    empirical Y target.
    """
    obs = (mp_L[:, :, 1] - mp_R[:, :, 1]).astype(np.float32)
    N, J = obs.shape
    ref = np.zeros(J, dtype=np.float32)
    for j in range(J):
        col = obs[:, j]
        m = ~np.isnan(col)
        if m.any():
            ref[j] = float(np.median(col[m]))
    return np.abs(obs - ref[None, :])


def _per_endpoint_split(dev: np.ndarray,
                        mp_3d_p: np.ndarray, mp_3d_c: np.ndarray,
                        reproj_p: np.ndarray | None, reproj_c: np.ndarray | None,
                        untrusted_p: np.ndarray, untrusted_c: np.ndarray) -> tuple:
    """Split bone deviation between endpoints.

    Primary: per-endpoint combined reprojection residual at frame f.
    Fallback: per-endpoint robust temporal instability (Z residual) when
    reprojections are too similar to decide.

    ``dev``: (N,) per-frame bone-length deviation.
    Returns (share_p, share_c) arrays (N,) summing to 1.
    """
    N = len(dev)
    share_p = np.full(N, 0.5, dtype=np.float32)

    # Stage 1: reprojection-based split
    if reproj_p is not None and reproj_c is not None:
        # Combined across cameras, ignoring NaN
        r_p = np.nansum(reproj_p, axis=1) if reproj_p.ndim == 2 else reproj_p
        r_c = np.nansum(reproj_c, axis=1) if reproj_c.ndim == 2 else reproj_c
        total = r_p + r_c
        FLOOR_PX = 3.0  # below combined 3 px, residuals are uninformative
        decisive = total > FLOOR_PX
        with np.errstate(all="ignore"):
            share_p[decisive] = r_p[decisive] / total[decisive]

        # Stage 2: temporal instability tiebreaker where Stage 1 was inconclusive
        if np.any(~decisive):
            inst_p = _robust_deviation_1d(mp_3d_p[:, 2], untrusted_p, MIN_SIGMA_3D)
            inst_c = _robust_deviation_1d(mp_3d_c[:, 2], untrusted_c, MIN_SIGMA_3D)
            tot_inst = inst_p + inst_c
            ok_inst = (~decisive) & (tot_inst > 0.1)
            share_p[ok_inst] = inst_p[ok_inst] / tot_inst[ok_inst]
    else:
        # No reproj (no calibration) — fall back to temporal tiebreaker only
        inst_p = _robust_deviation_1d(mp_3d_p[:, 2], untrusted_p, MIN_SIGMA_3D)
        inst_c = _robust_deviation_1d(mp_3d_c[:, 2], untrusted_c, MIN_SIGMA_3D)
        tot_inst = inst_p + inst_c
        ok_inst = tot_inst > 0.1
        share_p[ok_inst] = inst_p[ok_inst] / tot_inst[ok_inst]

    share_p = np.clip(share_p, 0.0, 1.0)
    share_c = 1.0 - share_p
    return share_p, share_c


def _bone_length_score(mp_3d: np.ndarray,
                        reproj: np.ndarray | None,
                        untrusted: np.ndarray) -> np.ndarray:
    """Per-joint bone-length deviation with per-endpoint attribution.

    Robust median of bone length uses only trusted frames.  Each bone's
    per-frame deviation is split between its endpoints according to which
    endpoint looks more wrong (by reprojection residual, with temporal
    instability as a tiebreaker)."""
    N, J, _ = mp_3d.shape
    score = np.zeros((N, 21), dtype=np.float32)
    for (p, c) in BONES:
        diff = mp_3d[:, c] - mp_3d[:, p]
        length = np.linalg.norm(diff, axis=1)
        # Robust median ignoring untrusted endpoints / NaN
        bone_bad = untrusted[:, p] | untrusted[:, c] | np.isnan(length)
        trusted_lengths = length[~bone_bad]
        if len(trusted_lengths) < 3:
            continue
        med = float(np.median(trusted_lengths))
        dev = np.abs(length - med)
        dev[np.isnan(dev)] = 0.0

        rp = reproj[:, p] if reproj is not None else None
        rc = reproj[:, c] if reproj is not None else None
        share_p, share_c = _per_endpoint_split(
            dev, mp_3d[:, p], mp_3d[:, c], rp, rc,
            untrusted[:, p], untrusted[:, c])
        score_p = dev * share_p
        score_c = dev * share_c
        score[:, p] = np.maximum(score[:, p], score_p)
        score[:, c] = np.maximum(score[:, c], score_c)
    return score


def _bone_agreement_score(mp_3d: np.ndarray,
                           reproj: np.ndarray | None,
                           untrusted: np.ndarray) -> np.ndarray:
    """Frame-to-frame change in bone lengths, with per-endpoint attribution.

    The change is attributed to whichever endpoint moved more.  We pick the
    attribution per frame pair (f, f−1) and assign the resulting score to
    both f and f−1 (we don't know which frame caused the jitter)."""
    N, J, _ = mp_3d.shape
    score = np.zeros((N, 21), dtype=np.float32)
    for (p, c) in BONES:
        diff = mp_3d[:, c] - mp_3d[:, p]
        length = np.linalg.norm(diff, axis=1)
        dlen = np.zeros(N, dtype=np.float32)
        dlen[1:] = np.abs(length[1:] - length[:-1])
        dlen[np.isnan(dlen)] = 0.0

        rp = reproj[:, p] if reproj is not None else None
        rc = reproj[:, c] if reproj is not None else None
        share_p, share_c = _per_endpoint_split(
            dlen, mp_3d[:, p], mp_3d[:, c], rp, rc,
            untrusted[:, p], untrusted[:, c])
        score_p = dlen * share_p
        score_c = dlen * share_c
        # Distribute to both f and f−1 (jitter could originate at either)
        for (j, sj) in ((p, score_p), (c, score_c)):
            cur = score[:, j]
            score[:, j] = np.maximum(cur, sj)
            score[:-1, j] = np.maximum(score[:-1, j], sj[1:])
    return score


def _angle_score(mp_3d: np.ndarray, angle_priors: list) -> np.ndarray:
    """Joint-angle violation magnitude (degrees out of allowed range)."""
    from .mano_data import _compute_angles, FLEX_ANGLE_OPTIONS
    N = mp_3d.shape[0]
    score = np.zeros((N, 21), dtype=np.float32)
    if N == 0:
        return score
    angles = _compute_angles(mp_3d)

    # Build name → joint index map from FLEX_ANGLE_OPTIONS
    # opt: (name, parent, joint, child) where name is "Flex: <short>"
    opt_joint = {}
    for (name, _p, j, _c) in FLEX_ANGLE_OPTIONS:
        short = name.replace("Flex:", "").strip()
        opt_joint[short] = j

    for prior in (angle_priors or []):
        pname = prior.get("name", "").strip()
        j = opt_joint.get(pname)
        if j is None:
            continue
        for axis_key, trace_prefix in [("flex", "Flex:"), ("abd", "Abd:")]:
            trace_name = f"{trace_prefix} {pname}"
            trace = angles.get(trace_name)
            if trace is None:
                continue
            mn = prior.get(f"{axis_key}_min")
            mx = prior.get(f"{axis_key}_max")
            if mn is None or mx is None:
                continue
            t = np.array([np.nan if v is None else v for v in trace], dtype=np.float32)
            viol = np.maximum(np.maximum(mn - t, 0), np.maximum(t - mx, 0))
            viol = np.nan_to_num(viol, nan=0.0)
            # Attribute violation to the joint's DESCENDANTS, not the joint
            # itself — a bad angle at joint j displaces all distal joints,
            # but j's own 3D coordinate can still be correct.
            for d in JOINT_DESCENDANTS.get(j, []):
                score[:, d] = np.maximum(score[:, d], viol)
    return score


def _confidence_score(conf_L: np.ndarray | None,
                      conf_R: np.ndarray | None,
                      N: int) -> np.ndarray | None:
    """Detection-level confidence score: 1 − min(conf_L, conf_R), broadcast
    to all 21 joints."""
    if conf_L is None and conf_R is None:
        return None
    if conf_L is None:
        cL = np.full(N, np.nan, dtype=np.float32)
    else:
        cL = np.array(conf_L, dtype=np.float32)[:N]
        if len(cL) < N:
            cL = np.concatenate([cL, np.full(N - len(cL), np.nan, np.float32)])
    if conf_R is None:
        cR = np.full(N, np.nan, dtype=np.float32)
    else:
        cR = np.array(conf_R, dtype=np.float32)[:N]
        if len(cR) < N:
            cR = np.concatenate([cR, np.full(N - len(cR), np.nan, np.float32)])
    # Min across cameras, nan-tolerant.  Silence the All-NaN-slice
    # RuntimeWarning for frames where both cameras are NaN (normal for
    # out-of-range frames / subjects without MP data yet).
    stack = np.stack([cL, cR], axis=1)  # (N, 2)
    import warnings as _warnings
    with np.errstate(all="ignore"), _warnings.catch_warnings():
        _warnings.simplefilter("ignore", category=RuntimeWarning)
        min_conf = np.nanmin(stack, axis=1)  # (N,)
    score = 1.0 - np.nan_to_num(min_conf, nan=1.0)
    score[np.isnan(min_conf)] = np.nan
    # Broadcast to (N, 21)
    return np.tile(score[:, None], (1, 21)).astype(np.float32)


# ─── Attribution factors (→ [N, 21, 2] raw scores) ────────────────────────

def _reprojection_per_camera(mp_L, mp_R, mp_3d, calib) -> np.ndarray:
    """Raw per-camera reprojection residuals, (N, 21, 2).

    Kept as a per-camera array because the bone-length endpoint splitter
    sums across cameras per joint.  For the detection factor we collapse
    to (N, 21) via sum across cameras (see _reprojection_score_det).
    """
    from .mano_data import _project_to_2d
    rp_L = _project_to_2d(mp_3d, calib["K1"], calib["dist1"],
                           np.eye(3, dtype=np.float64),
                           np.zeros((3, 1), dtype=np.float64))
    rp_R = _project_to_2d(mp_3d, calib["K2"], calib["dist2"],
                           calib["R"], calib["T"])
    res_L = np.linalg.norm(rp_L - mp_L, axis=2).astype(np.float32)
    res_R = np.linalg.norm(rp_R - mp_R, axis=2).astype(np.float32)
    return np.stack([res_L, res_R], axis=2)  # (N, 21, 2)


def _reprojection_score_det(reproj_per_cam: np.ndarray) -> np.ndarray:
    """Detection version: sum residuals across cameras → (N, 21).

    Triangulation distributes error between cameras, so per-camera residual
    is a weak per-camera signal, but the summed per-joint residual is a
    strong per-joint detection signal (big for joints where something is
    wrong on either or both cameras)."""
    with np.errstate(all="ignore"):
        return np.nansum(reproj_per_cam, axis=2).astype(np.float32)


def _jump_2d_score(mp_L: np.ndarray, mp_R: np.ndarray,
                    untrusted_per_cam: np.ndarray) -> np.ndarray:
    """Per-camera 2D jump = max(polynomial-fit residual, adjacent-frame Δ).

    Ensures large 2D jumps are flagged even if a polynomial fit smooths
    through them (see _z_jump_score for the analogous reasoning).
    """
    N = mp_L.shape[0]
    out = np.zeros((N, 21, 2), dtype=np.float32)
    for ci, mp in enumerate([mp_L, mp_R]):
        for j in range(21):
            u = untrusted_per_cam[:, j, ci]
            poly = _robust_deviation_2d(mp[:, j, :], u, MIN_SIGMA_2D)
            jump = _adjacent_jump_2d(mp[:, j, :], u, MIN_SIGMA_2D)
            out[:, j, ci] = np.maximum(poly, jump)
    return out


def _conf_per_camera(conf_L, conf_R, N: int) -> np.ndarray | None:
    if conf_L is None and conf_R is None:
        return None
    out = np.full((N, 21, 2), np.nan, dtype=np.float32)
    if conf_L is not None:
        c = np.array(conf_L, dtype=np.float32)[:N]
        if len(c) < N:
            c = np.concatenate([c, np.full(N - len(c), np.nan, np.float32)])
        out[:, :, 0] = np.tile((1.0 - c)[:, None], (1, 21))
    if conf_R is not None:
        c = np.array(conf_R, dtype=np.float32)[:N]
        if len(c) < N:
            c = np.concatenate([c, np.full(N - len(c), np.nan, np.float32)])
        out[:, :, 1] = np.tile((1.0 - c)[:, None], (1, 21))
    return out


# ─── Top-level ──────────────────────────────────────────────────────────────

def compute_scores(mp_L, mp_R, conf_L, conf_R, calib, angle_priors,
                   subject_name: str | None = None, trial_stem: str | None = None):
    """Compute all normalised detection + attribution scores for a trial.

    Two-pass pipeline:
      1. Compute frame-local raw scores (y-disparity, angle, confidence,
         reprojection).  These don't need clean neighbouring frames.
      2. Build a trust mask from frame-local signals (top TRUST_PCT=15% of
         each factor marked untrusted).
      3. Compute temporal raw scores (z-jump, bone-length, bone-agreement,
         2D-jump) with the trust mask as an exclusion list for baselines
         — breaks burst-error contamination.
      4. Percentile-normalise each raw score for slider-threshold semantics.

    Returns (detection, attribution) — dicts of factor → ndarray.  Detection
    arrays are (N, 21); attribution arrays are (N, 21, 2).  Values are
    percentile ranks in [0, 1] (or None if factor unavailable)."""
    N = mp_L.shape[0]
    mp_3d = np.full((N, 21, 3), np.nan, dtype=np.float32)
    if calib is not None:
        for j in range(21):
            mp_3d[:, j] = triangulate_points(mp_L[:, j], mp_R[:, j], calib).astype(np.float32)

    # ─── 1. Frame-local raw scores ─────────────────────────────────────
    y_disp_raw       = _y_disparity_score(mp_L, mp_R, calib=calib, mp_3d=mp_3d)
    angle_raw        = _angle_score(mp_3d, angle_priors) if calib is not None else None
    conf_det_raw     = _confidence_score(conf_L, conf_R, N)
    reproj_per_cam   = _reprojection_per_camera(mp_L, mp_R, mp_3d, calib) if calib is not None else None
    reproj_det_raw   = _reprojection_score_det(reproj_per_cam) if reproj_per_cam is not None else None
    conf_attr_raw    = _conf_per_camera(conf_L, conf_R, N)

    # ─── 1b. Dedicated Y-disparity exclusion mask ─────────────────────
    # Any frame whose |Y_L − Y_R| sits in the top ~15% is considered to
    # have corrupted 3D (stereo mismatch → bad triangulation → bad Z).
    # Exclude these from Z-based baselines so their corruption doesn't
    # propagate into the median/MAD used by z_outlier and z_jump.
    y_disp_mask = None
    if y_disp_raw is not None:
        flat = y_disp_raw.reshape(-1)
        valid = ~np.isnan(flat)
        if valid.sum() > 10:
            thresh_y = float(np.quantile(flat[valid], 0.85))
            y_disp_mask = (y_disp_raw > thresh_y) & ~np.isnan(y_disp_raw)

    # ─── 1c. z_outlier with Y-disp exclusion from its baseline ──────
    z_outlier_raw = _z_outlier_score(mp_3d, exclude_mask=y_disp_mask) if calib is not None else None

    # ─── 2. Trust masks (fixed-percentile thresholds on frame-local) ──
    # (a) Joint-level trust: per (frame, joint).  Union of top TRUST_PCT of
    #     all frame-local detection factors.
    trust_untrusted = _build_trust_mask(
        [y_disp_raw, angle_raw, conf_det_raw, reproj_det_raw, z_outlier_raw])

    # (b) Per-camera trust: per (frame, joint, camera).  Uses per-camera
    #     reprojection + per-camera confidence.
    per_cam_frame_local = []
    if reproj_per_cam is not None:
        per_cam_frame_local.append(reproj_per_cam)
    if conf_attr_raw is not None:
        per_cam_frame_local.append(conf_attr_raw)
    if per_cam_frame_local:
        trust_untrusted_per_cam = _build_trust_mask(per_cam_frame_local)
    else:
        trust_untrusted_per_cam = np.zeros((N, 21, 2), dtype=bool)

    # ─── 3. Temporal raw scores (use trust masks) ─────────────────────
    # Build a dedicated skip mask for z_jump's adjacent-Δ that combines:
    #   - z_outlier_mask: |Z−median|/MAD > 2.0 (known Z-bad frames)
    #   - y_disp_mask:    top 15% Y-disparity (triangulation-corrupted)
    # Any frame in this mask is skipped when scoring adjacent deltas so
    # its corrupted Z can't spuriously flag its correct neighbours.
    z_outlier_mask = None
    if z_outlier_raw is not None:
        z_outlier_mask = (z_outlier_raw > 2.0) & ~np.isnan(z_outlier_raw)
    z_jump_skip = None
    if z_outlier_mask is not None and y_disp_mask is not None:
        z_jump_skip = z_outlier_mask | y_disp_mask
    elif z_outlier_mask is not None:
        z_jump_skip = z_outlier_mask
    elif y_disp_mask is not None:
        z_jump_skip = y_disp_mask
    z_jump_raw = _z_jump_score(mp_3d, trust_untrusted, z_jump_skip) if calib is not None else None
    bone_length_raw = _bone_length_score(mp_3d, reproj_per_cam, trust_untrusted) if calib is not None else None
    bone_agreement_raw = _bone_agreement_score(mp_3d, reproj_per_cam, trust_untrusted) if calib is not None else None
    jump_2d_raw = _jump_2d_score(mp_L, mp_R, trust_untrusted_per_cam)

    # ─── 4. Percentile-normalise ──────────────────────────────────────
    det_raw = {
        "z_jump":         z_jump_raw,
        "z_outlier":      z_outlier_raw,
        "y_disp":         y_disp_raw,
        "bone_length":    bone_length_raw,
        "bone_agreement": bone_agreement_raw,
        "angle":          angle_raw,
        "reproj":         reproj_det_raw,
        "confidence":     conf_det_raw,
    }
    # Optional HRnet attribution: heatmap response at each camera's MP label.
    # Higher score = lower heatmap value = camera the heatmap thinks is wrong.
    hrnet_attr_raw = None
    if subject_name is not None and trial_stem is not None:
        hrnet_attr_raw = _hrnet_attr_score(mp_L, mp_R, subject_name, trial_stem)

    attr_raw = {
        "jump_2d":    jump_2d_raw,
        "confidence": conf_attr_raw,
        "hrnet":      hrnet_attr_raw,
    }
    # Rank z_jump and z_outlier per-joint so stable joints (e.g. MCPs) can
    # surface their own outliers without being drowned out by noisier
    # joints (wrist, fingertips) whose absolute scores dominate a global
    # ranking.  All other factors keep the global ranking so cross-joint
    # comparisons (bone-length, angles, reproj) work as before.
    PER_JOINT_RANK = {"z_jump", "z_outlier"}
    def _rank(name, v):
        if v is None: return None
        return _percentile_rank_per_joint(v) if name in PER_JOINT_RANK else _percentile_rank(v)
    detection   = {k: _rank(k, v) for k, v in det_raw.items()}
    attribution = {k: _rank(k, v) for k, v in attr_raw.items()}
    return detection, attribution


def compute_scores_hrnet(peaks_L, peaks_R, auc_L, auc_R, calib, angle_priors,
                          factors: set | None = None,
                          conf_attr_override: np.ndarray | None = None):
    """HRnet-flavoured compute_scores: same factors as compute_scores but
    drops the HRnet-mismatch attribution term and uses per-(frame, joint,
    camera) cluster AUC as the per-camera "confidence" signal in place of
    the MP per-frame confidence.

    ``factors`` (optional): set of detection factor names to actually
    compute.  Unlisted factors return None.  Use this to avoid expensive
    polynomial fits / Hungarian-style assignments for factors the caller
    won't threshold on.  HRnet Correct passes ``{"y_disp", "z_outlier",
    "z_jump"}`` (plus ``jump_2d`` and ``confidence`` are always free).
    Default = full set (back-compat).

    ``auc_L`` / ``auc_R``: ``(N, 21)`` non-negative cluster sums.  Higher
    = stronger heatmap evidence at this peak.
    """
    N = peaks_L.shape[0]
    J = peaks_L.shape[1]
    if factors is None:
        factors = {"z_jump", "z_outlier", "y_disp", "bone_length",
                   "bone_agreement", "angle", "reproj"}
    mp_3d = np.full((N, J, 3), np.nan, dtype=np.float32)
    if calib is not None:
        for j in range(J):
            mp_3d[:, j] = triangulate_points(peaks_L[:, j], peaks_R[:, j], calib).astype(np.float32)

    # Per-camera AUC normalised to [0,1] per joint, treated as a "trust"
    # signal: low AUC → low confidence in this camera's peak.  Always
    # computed — it's cheap and anchors the bad-camera attribution.
    def _norm_auc(arr):
        a = np.asarray(arr, dtype=np.float32)
        out = np.zeros_like(a)
        for j in range(a.shape[1]):
            col = a[:, j]
            vmax = float(np.nanmax(col)) if np.isfinite(np.nanmax(col)) else 0.0
            if vmax > 0:
                out[:, j] = np.clip(col / vmax, 0.0, 1.0)
        return out

    aL = _norm_auc(auc_L)
    aR = _norm_auc(auc_R)
    min_auc = np.minimum(aL, aR)              # (N, J)
    conf_det_raw = (1.0 - min_auc).astype(np.float32)
    if conf_attr_override is not None:
        # Caller-supplied per-camera attribution (HRnet Correct uses an
        # AUC-delta-after-correction signal: higher Δ → that camera
        # benefits more from being moved → that camera is the bad one).
        conf_attr_raw = np.asarray(conf_attr_override, dtype=np.float32)
    else:
        conf_attr_raw = np.stack([1.0 - aL, 1.0 - aR], axis=2).astype(np.float32)  # (N, J, 2)

    y_disp_raw = _y_disparity_score(peaks_L, peaks_R, calib=calib, mp_3d=mp_3d) \
        if "y_disp" in factors else None
    angle_raw  = (_angle_score(mp_3d, angle_priors)
                  if "angle" in factors and calib is not None else None)
    need_reproj = ("reproj" in factors or "bone_length" in factors
                   or "bone_agreement" in factors)
    reproj_per_cam = (_reprojection_per_camera(peaks_L, peaks_R, mp_3d, calib)
                      if need_reproj and calib is not None else None)
    reproj_det_raw = (_reprojection_score_det(reproj_per_cam)
                      if "reproj" in factors and reproj_per_cam is not None else None)

    y_disp_mask = None
    if y_disp_raw is not None:
        flat = y_disp_raw.reshape(-1)
        valid = ~np.isnan(flat)
        if valid.sum() > 10:
            thresh_y = float(np.quantile(flat[valid], 0.85))
            y_disp_mask = (y_disp_raw > thresh_y) & ~np.isnan(y_disp_raw)
    z_outlier_raw = (_z_outlier_score(mp_3d, exclude_mask=y_disp_mask)
                     if "z_outlier" in factors and calib is not None else None)

    trust_untrusted = _build_trust_mask(
        [y_disp_raw, angle_raw, conf_det_raw, reproj_det_raw, z_outlier_raw])
    per_cam_frame_local = []
    if reproj_per_cam is not None:
        per_cam_frame_local.append(reproj_per_cam)
    per_cam_frame_local.append(conf_attr_raw)
    trust_untrusted_per_cam = _build_trust_mask(per_cam_frame_local)

    z_outlier_mask = None
    if z_outlier_raw is not None:
        z_outlier_mask = (z_outlier_raw > 2.0) & ~np.isnan(z_outlier_raw)
    z_jump_skip = None
    if z_outlier_mask is not None and y_disp_mask is not None:
        z_jump_skip = z_outlier_mask | y_disp_mask
    elif z_outlier_mask is not None:
        z_jump_skip = z_outlier_mask
    elif y_disp_mask is not None:
        z_jump_skip = y_disp_mask
    z_jump_raw = (_z_jump_score(mp_3d, trust_untrusted, z_jump_skip)
                  if "z_jump" in factors and calib is not None else None)
    bone_length_raw = (_bone_length_score(mp_3d, reproj_per_cam, trust_untrusted)
                       if "bone_length" in factors and calib is not None else None)
    bone_agreement_raw = (_bone_agreement_score(mp_3d, reproj_per_cam, trust_untrusted)
                          if "bone_agreement" in factors and calib is not None else None)
    # jump_2d is the camera-attribution workhorse — always computed.
    jump_2d_raw = _jump_2d_score(peaks_L, peaks_R, trust_untrusted_per_cam)

    det_raw = {
        "z_jump":         z_jump_raw,
        "z_outlier":      z_outlier_raw,
        "y_disp":         y_disp_raw,
        "bone_length":    bone_length_raw,
        "bone_agreement": bone_agreement_raw,
        "angle":          angle_raw,
        "reproj":         reproj_det_raw,
        "confidence":     conf_det_raw,
    }
    # NB: HRnet correction does *not* use the heatmap-mismatch attribution
    # term — that would create a feedback loop with the peaks we're
    # correcting.  Only jump_2d and (cluster-AUC) confidence drive
    # camera-blame attribution.
    attr_raw = {
        "jump_2d":    jump_2d_raw,
        "confidence": conf_attr_raw,
    }
    PER_JOINT_RANK = {"z_jump", "z_outlier"}
    def _rank(name, v):
        if v is None: return None
        return _percentile_rank_per_joint(v) if name in PER_JOINT_RANK else _percentile_rank(v)
    detection   = {k: _rank(k, v) for k, v in det_raw.items()}
    attribution = {k: _rank(k, v) for k, v in attr_raw.items()}
    return detection, attribution


def _flag_with_conf(factors_weights, conf_weight: float, conf_score):
    """OR-combine per-factor flags, with MP confidence acting as a MODULATOR.

    Rules:
    - If at least one joint-specific factor is active AND the confidence
      slider is > 0, confidence does NOT flag standalone.  Instead it
      boosts each joint-specific score additively by (w_conf * σ_conf),
      which effectively lowers the flagging threshold in low-confidence
      frames — a low-confidence frame becomes "more likely" to surface
      errors where another factor already has mid-range signal, without
      blanket-flagging every joint.
    - If confidence is the ONLY active signal, fall back to standalone
      flagging (whole frame) so the slider still does something useful.

    ``factors_weights`` is a list of ``(score, weight)`` for joint-specific
    factors (both can be None / 0).  Returns an OR-combined mask, or None
    if no factor is active.
    """
    active = [(s, w) for (s, w) in factors_weights if s is not None and w > 0]
    if active:
        boost = None
        if conf_weight > 0 and conf_score is not None:
            boost = conf_weight * np.nan_to_num(conf_score, nan=0.0)
        flag_mask = None
        for score, w in active:
            boosted = score if boost is None else np.minimum(score + boost, 1.0)
            flag = (boosted > (1.0 - w)) & ~np.isnan(boosted)
            flag_mask = flag if flag_mask is None else (flag_mask | flag)
        return flag_mask

    # Only confidence active → standalone whole-frame flagging
    if conf_weight > 0 and conf_score is not None:
        return (conf_score > (1.0 - conf_weight)) & ~np.isnan(conf_score)
    return None


def apply_thresholds(detection: dict, attribution: dict,
                     det_weights: dict, attr_weights: dict,
                     winner_take_all: bool = False) -> np.ndarray:
    """Produce error matrix [N, 21, 2].

    Detection: per-factor threshold ``score > (1 - slider)``, OR-combined.
    Confidence modulates joint-specific factors when any of them are active
    (it does not flag standalone in that case).

    Attribution: same rule per camera — confidence boosts the per-camera
    reprojection / 2D-jump scores rather than flagging entire frames.
    """
    # ─── Detection ────────────────────────────────────────────────────────
    conf_w = float(det_weights.get("confidence", 0.0))
    conf_score = detection.get("confidence")
    joint_specific = [(detection.get(k), float(det_weights.get(k, 0.0)))
                       for k in DETECTION_FACTORS if k != "confidence"]
    susp = _flag_with_conf(joint_specific, conf_w, conf_score)

    if susp is None:
        for v in attribution.values():
            if v is not None:
                return np.zeros((v.shape[0], 21, 2), dtype=bool)
        return np.zeros((0, 21, 2), dtype=bool)

    N, J = susp.shape

    # ─── Attribution ──────────────────────────────────────────────────────
    conf_w_a = float(attr_weights.get("confidence", 0.0))
    conf_score_a = attribution.get("confidence")  # (N, 21, 2)
    joint_specific_a = [(attribution.get(k), float(attr_weights.get(k, 0.0)))
                         for k in ATTRIBUTION_FACTORS if k != "confidence"]
    any_attr_active = any(w > 0 and s is not None for (s, w) in joint_specific_a) \
                      or (conf_w_a > 0 and conf_score_a is not None)

    if not any_attr_active:
        cam_flag = np.zeros((N, J, 2), dtype=bool)
        if winner_take_all:
            # WTA stage with no user-weighted attribution: fall back to
            # whichever per-camera signal is available.  Priority:
            # hrnet (heatmap-derived) → jump_2d (always available).  For
            # the wrist (0) and thumb CMC (1), HRnet is never used —
            # those joints fall back directly to jump_2d.
            hr_score   = attribution.get("hrnet")
            jump_score = attribution.get("jump_2d")
            if hr_score is not None:
                fallback = np.nan_to_num(hr_score, nan=0.0).astype(np.float32)
                if jump_score is not None:
                    jr = np.nan_to_num(jump_score, nan=0.0).astype(np.float32)
                    for jx in _HRNET_EXEMPT_JOINTS:
                        fallback[:, jx, :] = jr[:, jx, :]
                else:
                    for jx in _HRNET_EXEMPT_JOINTS:
                        fallback[:, jx, :] = 0.0
            else:
                fallback = jump_score
            if fallback is not None:
                pooled = _pool_finger_attr(np.nan_to_num(fallback, nan=0.0))
                diff = pooled[:, :, 0] - pooled[:, :, 1]
                cam_flag[:, :, 0] = susp & (diff >= 0)
                cam_flag[:, :, 1] = susp & (diff < 0)
            else:
                cam_flag[:, :, 0] = susp
            return cam_flag
        # Non-WTA: blame both cameras for every suspicious joint.
        cam_flag[:, :, 0] = susp
        cam_flag[:, :, 1] = susp
        return cam_flag

    cam_flag = _flag_with_conf(joint_specific_a, conf_w_a, conf_score_a)
    if cam_flag is None:
        cam_flag = np.zeros((N, J, 2), dtype=bool)
    # Zero out cameras for non-suspicious (frame, joint) pairs
    cam_flag = cam_flag.copy()
    cam_flag[~susp] = False

    # Combined per-camera attribution (sum of weighted factor ranks),
    # then pooled across each finger's PIP/DIP/tip so all three joints
    # in a finger share their bad-camera decision.  Wrist (0) and thumb
    # CMC (1) are exempt from HRnet and fall back to jump_2d when no
    # other factor is active for them.
    per_cam_total = np.zeros((N, J, 2), dtype=np.float32)
    for name in ATTRIBUTION_FACTORS:
        s_k = attribution.get(name)
        w_k = float(attr_weights.get(name, 0.0))
        if s_k is None or w_k <= 0:
            continue
        contrib = np.nan_to_num(s_k, nan=0.0) * w_k
        if name == "hrnet":
            for jx in _HRNET_EXEMPT_JOINTS:
                contrib[:, jx, :] = 0.0
        per_cam_total += contrib
    per_cam_total = _pool_finger_attr(per_cam_total)
    jump_attr = attribution.get("jump_2d")
    if jump_attr is not None:
        for jx in _HRNET_EXEMPT_JOINTS:
            zero_mask = (per_cam_total[:, jx, 0] == 0) & (per_cam_total[:, jx, 1] == 0)
            if zero_mask.any():
                jr = np.nan_to_num(jump_attr[:, jx, :], nan=0.0).astype(np.float32)
                per_cam_total[zero_mask, jx, :] = jr[zero_mask]

    if winner_take_all:
        # Stages whose corrections only ever move one camera per frame:
        # collapse cam_flag onto whichever camera has the higher total
        # (or camera 0 on a perfect tie) so the user never sees the same
        # joint flagged red in both cameras simultaneously.
        diff = per_cam_total[:, :, 0] - per_cam_total[:, :, 1]
        cam0_only = susp & (diff >= 0)
        cam1_only = susp & (diff < 0)
        cam_flag = np.zeros((N, J, 2), dtype=bool)
        cam_flag[:, :, 0] = cam0_only
        cam_flag[:, :, 1] = cam1_only
        return cam_flag

    # Fallback path (non-WTA): for suspicious joints where neither camera
    # crossed the threshold, fall back to a winner-take-all on the combined
    # score; ties → blame both.  This is the historical detection behaviour.
    neither = susp & ~cam_flag[:, :, 0] & ~cam_flag[:, :, 1]
    if bool(neither.any()):
        diff = per_cam_total[:, :, 0] - per_cam_total[:, :, 1]
        both_zero = neither & (per_cam_total[:, :, 0] <= 0) & (per_cam_total[:, :, 1] <= 0)
        cam0_wins = neither & (diff > 0) & ~both_zero
        cam1_wins = neither & (diff < 0) & ~both_zero
        both = neither & (~cam0_wins) & (~cam1_wins)
        cam_flag[:, :, 0] = cam_flag[:, :, 0] | cam0_wins | both
        cam_flag[:, :, 1] = cam_flag[:, :, 1] | cam1_wins | both
    return cam_flag


# ─── Cache + trial-level entry point ───────────────────────────────────────

# Scores keyed by (subject, trial, mp_data_mtime).  Small LRU.
_score_cache: dict = {}
_CACHE_MAX = 8


def _get_scores(subject_name: str, trial_stem: str,
                mp_L: np.ndarray, mp_R: np.ndarray,
                conf_L, conf_R, calib, angle_priors, cache_key=None):
    if cache_key is not None and cache_key in _score_cache:
        return _score_cache[cache_key]
    d, a = compute_scores(mp_L, mp_R, conf_L, conf_R, calib, angle_priors,
                          subject_name=subject_name, trial_stem=trial_stem)
    if cache_key is not None:
        if len(_score_cache) >= _CACHE_MAX:
            # Drop oldest (insertion order)
            _score_cache.pop(next(iter(_score_cache)))
        _score_cache[cache_key] = (d, a)
    return d, a


def correct_y_from_errors(mp_L: np.ndarray, mp_R: np.ndarray,
                          errors: np.ndarray,
                          jump_2d_scores: np.ndarray | None = None,
                          calib: dict | None = None) -> tuple:
    """Y-disparity correction.

    For each (frame, joint, camera) flagged in ``errors``:
      1. Pick the camera to move.  Single-camera flag → that camera.
         Both flagged → winner-take-all by 2D-jump score.
      2. Compute a target Y for the loser camera:
         - If calibration is available: anchor on the WINNER camera's
           current Y and apply the calibration-predicted disparity for
           this joint at this frame (from triangulating the joint and
           projecting both cameras through the calibration).  This
           preserves the geometrically-correct epipolar Y offset.
         - Otherwise: linear interpolation between the loser camera's
           own nearest non-flagged neighbours.
    Returns ``(corrected_mp_L, corrected_mp_R, n_applied)``.
    X coordinates are left unchanged.
    """
    mp_L_c = mp_L.copy()
    mp_R_c = mp_R.copy()
    N, J, _ = mp_L.shape
    WIN = 15

    flagged_any = errors[:, :, 0] | errors[:, :, 1]

    # Pre-compute per-(frame, joint) calibration-predicted Y disparity
    # (Y_L_pred − Y_R_pred) by triangulating MP into 3D and reprojecting
    # both cameras.  Only used when calib is provided.
    pred_disp = None
    if calib is not None:
        import cv2 as _cv2
        pred_disp = np.full((N, J), np.nan, dtype=np.float32)
        rvec_I = np.zeros((3, 1)); tvec_0 = np.zeros((3, 1))
        rvec_R = _cv2.Rodrigues(calib['R'])[0]; tvec_R = calib['T'].reshape(3, 1)
        for j in range(J):
            p3d = triangulate_points(mp_L[:, j], mp_R[:, j], calib).astype(np.float32)
            valid = ~np.any(np.isnan(p3d), axis=1)
            if not valid.any(): continue
            pts = p3d[valid].reshape(-1, 1, 3).astype(np.float64)
            pL_p, _ = _cv2.projectPoints(pts, rvec_I, tvec_0, calib['K1'], calib['dist1'])
            pR_p, _ = _cv2.projectPoints(pts, rvec_R, tvec_R, calib['K2'], calib['dist2'])
            pred_disp[valid, j] = (pL_p.reshape(-1, 2)[:, 1] - pR_p.reshape(-1, 2)[:, 1]).astype(np.float32)

    def _poly_predict(series: np.ndarray, untrust: np.ndarray, f: int) -> float:
        """Linear interpolation between the nearest clean neighbours on
        either side of ``f``.  Doesn't extrapolate — when only one side
        has a clean neighbour, returns that neighbour's value verbatim
        (no slope estimation).  Falls back to including flagged
        neighbours if no clean ones exist within ±WIN.
        """
        # Find nearest clean neighbour before f
        prev_idx = -1
        for k in range(f - 1, max(-1, f - WIN - 1), -1):
            v = series[k]
            if not np.isnan(v) and not untrust[k]:
                prev_idx = k; break
        # Find nearest clean neighbour after f
        next_idx = -1
        for k in range(f + 1, min(N, f + WIN + 1)):
            v = series[k]
            if not np.isnan(v) and not untrust[k]:
                next_idx = k; break
        # Fall back to flagged neighbours if no clean ones available
        if prev_idx < 0:
            for k in range(f - 1, max(-1, f - WIN - 1), -1):
                v = series[k]
                if not np.isnan(v):
                    prev_idx = k; break
        if next_idx < 0:
            for k in range(f + 1, min(N, f + WIN + 1)):
                v = series[k]
                if not np.isnan(v):
                    next_idx = k; break
        if prev_idx < 0 and next_idx < 0:
            return float("nan")
        if prev_idx < 0:
            return float(series[next_idx])
        if next_idx < 0:
            return float(series[prev_idx])
        # Linear interpolation between neighbours
        x0, y0 = float(prev_idx), float(series[prev_idx])
        x1, y1 = float(next_idx), float(series[next_idx])
        if x1 == x0:
            return y0
        return y0 + (y1 - y0) * (f - x0) / (x1 - x0)

    count = 0
    for j in range(J):
        y_L = mp_L[:, j, 1].copy()
        y_R = mp_R[:, j, 1].copy()
        unt = flagged_any[:, j]
        for f in range(N):
            e0 = bool(errors[f, j, 0])
            e1 = bool(errors[f, j, 1])
            if not e0 and not e1:
                continue
            if np.isnan(y_L[f]) and np.isnan(y_R[f]):
                continue
            # Decide which camera to move
            if e0 and not e1:
                cam = 0
            elif e1 and not e0:
                cam = 1
            else:
                if jump_2d_scores is not None:
                    j0 = jump_2d_scores[f, j, 0]
                    j1 = jump_2d_scores[f, j, 1]
                    j0 = 0.0 if np.isnan(j0) else float(j0)
                    j1 = 0.0 if np.isnan(j1) else float(j1)
                else:
                    j0 = j1 = 0.0
                cam = 0 if j0 >= j1 else 1
            # Calibration-anchored target preferred when available:
            # winner_Y ± predicted_disparity for this joint+frame.
            target = float("nan")
            if pred_disp is not None and not np.isnan(pred_disp[f, j]):
                if cam == 0 and not np.isnan(y_R[f]):
                    # Move OS: Y_L_pred = Y_R + predicted (Y_L − Y_R)
                    target = float(y_R[f]) + float(pred_disp[f, j])
                elif cam == 1 and not np.isnan(y_L[f]):
                    target = float(y_L[f]) - float(pred_disp[f, j])
            if np.isnan(target):
                # Fallback: temporal linear-interp on the chosen camera's
                # own clean neighbours.
                series = y_L if cam == 0 else y_R
                target = _poly_predict(series, unt, f)
            if np.isnan(target):
                continue
            if cam == 0:
                mp_L_c[f, j, 1] = target
            else:
                mp_R_c[f, j, 1] = target
            count += 1
    return mp_L_c, mp_R_c, count


# Finger-tip joint groups for pooled camera attribution.  Each tuple lists
# joints that share a camera-attribution decision (sum of weighted attribution
# scores across the group → single bad-camera pick applied to every joint in
# the group).  Wrist (0), thumb CMC (1), and the four finger MCPs
# (5/9/13/17) stay attributed independently.
FINGER_ATTR_GROUPS: tuple[tuple[int, ...], ...] = (
    (2, 3, 4),       # thumb MCP, IP, tip
    (6, 7, 8),       # index PIP, DIP, tip
    (10, 11, 12),    # middle PIP, DIP, tip
    (14, 15, 16),    # ring PIP, DIP, tip
    (18, 19, 20),    # pinky PIP, DIP, tip
)


def _pool_finger_attr(per_cam_total: np.ndarray) -> np.ndarray:
    """Pool per-camera attribution scores across each finger's PIP/DIP/tip
    so all three joints share the same bad-camera decision.  Other joints
    are passed through unchanged.

    ``per_cam_total`` is (N, 21, 2).  Returns a copy with entries for the
    grouped joints replaced by the per-(frame, camera) sum over the group.
    """
    if per_cam_total is None:
        return None
    out = per_cam_total.copy()
    for grp in FINGER_ATTR_GROUPS:
        idxs = list(grp)
        group_sum = per_cam_total[:, idxs, :].sum(axis=1)  # (N, 2)
        for j in idxs:
            out[:, j, :] = group_sum
    return out


_HRNET_EXEMPT_JOINTS = (0, 1)  # wrist + thumb CMC


def _combined_attr_per_cam(attribution: dict,
                            attr_weights: dict) -> np.ndarray | None:
    """Linear-combine the active attribution factors into a single per-camera
    'badness' score (N, 21, 2).  Higher score = camera the corrector should
    blame.  Per-finger pooling (PIP/DIP/tip share their bad-camera pick) is
    applied at the end.  Wrist (joint 0) and thumb CMC (joint 1) are
    EXEMPT from the HRnet factor — its contribution is masked to zero
    there — and fall back to jump_2d when no other signal is active for
    them.  Returns ``None`` if no factor with non-zero weight has a score."""
    out = None
    for name in ATTRIBUTION_FACTORS:
        s = attribution.get(name)
        w = float(attr_weights.get(name, 0.0))
        if s is None or w <= 0:
            continue
        contrib = np.nan_to_num(s, nan=0.0).astype(np.float32) * w
        if name == "hrnet":
            for jx in _HRNET_EXEMPT_JOINTS:
                contrib[:, jx, :] = 0.0
        out = contrib if out is None else (out + contrib)
    if out is None:
        return None
    out = _pool_finger_attr(out)
    # Wrist + thumb CMC fallback: when no non-HRnet signal weighed in
    # (sum is zero on both cameras), substitute the jump_2d percentile
    # rank so we always have *some* attribution signal for those joints.
    jump = attribution.get("jump_2d")
    if jump is not None:
        for jx in _HRNET_EXEMPT_JOINTS:
            zero_mask = (out[:, jx, 0] == 0) & (out[:, jx, 1] == 0)
            if zero_mask.any():
                jr = np.nan_to_num(jump[:, jx, :], nan=0.0).astype(np.float32)
                out[zero_mask, jx, :] = jr[zero_mask]
    return out


def correct_yz_from_errors(mp_L: np.ndarray, mp_R: np.ndarray,
                           y_errors: np.ndarray | None,
                           z_errors: np.ndarray | None,
                           calib,
                           jump_2d_scores: np.ndarray | None = None,
                           attr_per_cam: np.ndarray | None = None,
                           z_outlier_slider: float = 0.0) -> tuple:
    """Combined Y-disparity + Z-outlier correction, anchored to the good camera.

    Each per-(frame, joint) decision is processed in two phases per joint:
      Phase 1 — every frame flagged by Y-disparity (``y_errors``)
      Phase 2 — every frame flagged ONLY by Z-outlier (``z_errors``)
    Within each phase, frames are sorted by descending |z deviation|.  Y-disp
    frames going first lets the corrector clean up triangulation-corrupted
    depths before z-outlier scoring needs to fit polynomials over them.

    Steps per flagged (f, j):
      1. Pick the bad camera via combined per-camera attribution
         (``attr_per_cam``, falling back to ``jump_2d_scores``).
      2. Interpolate target Z from clean neighbour frames (frames not flagged
         by y_disp or z_outlier).
      3. Lift the GOOD camera's pixel onto the world-Z = Z_target plane.
      4. Project that 3D point into the BAD camera; replace x and y of the
         bad camera's label.

    Returns ``(corrected_mp_L, corrected_mp_R, n_applied)``.
    """
    import cv2 as _cv2

    mp_L_c = mp_L.copy()
    mp_R_c = mp_R.copy()
    N, J, _ = mp_L.shape
    W = 15

    # Build per-(frame, joint) and per-camera flag arrays.  ``errors_union``
    # is what the bad-camera attribution path reads; ``y_flag_jt`` and
    # ``z_flag_jt`` drive the two-phase ordering below.
    if y_errors is None:
        y_errors = np.zeros((N, J, 2), dtype=bool)
    if z_errors is None:
        z_errors = np.zeros((N, J, 2), dtype=bool)
    errors_union = y_errors | z_errors
    y_flag_jt = y_errors[:, :, 0] | y_errors[:, :, 1]
    z_flag_jt = z_errors[:, :, 0] | z_errors[:, :, 1]

    # Triangulate current MP for current Z trajectory.
    mp_3d = np.full((N, J, 3), np.nan, dtype=np.float32)
    for j in range(J):
        mp_3d[:, j] = triangulate_points(
            mp_L[:, j], mp_R[:, j], calib).astype(np.float32)

    K1 = np.asarray(calib["K1"], dtype=np.float64)
    K2 = np.asarray(calib["K2"], dtype=np.float64)
    dist1 = np.asarray(calib["dist1"], dtype=np.float64)
    dist2 = np.asarray(calib["dist2"], dtype=np.float64)
    R = np.asarray(calib["R"], dtype=np.float64)
    T = np.asarray(calib["T"], dtype=np.float64).reshape(3)
    Rt = R.T
    rvec_I = np.zeros((3, 1)); tvec_0 = np.zeros((3, 1))
    rvec_R = _cv2.Rodrigues(R)[0]; tvec_R = T.reshape(3, 1)

    flagged_any = errors_union[:, :, 0] | errors_union[:, :, 1]

    # Pool the bad-camera attribution score across each finger's PIP/DIP/tip
    # so all three joints in a finger always pick the same bad camera.
    # ``attr_per_cam`` is already pooled by ``_combined_attr_per_cam``;
    # we additionally pool the ``jump_2d_scores`` fallback here so it
    # behaves the same when no user-weighted attribution is supplied.
    if attr_per_cam is None and jump_2d_scores is not None:
        jump_2d_scores = _pool_finger_attr(
            np.nan_to_num(jump_2d_scores, nan=0.0).astype(np.float32))

    # ── Empirical per-joint y-disparity from clean frames ───────────
    # The stereo calibration's predicted y-disparity systematically
    # disagrees with the actual MP-tracked y-disparity by a constant
    # offset (a small Ty error in extrinsics — invisible to triangulation
    # but fatal to one-way reprojection).  We anchor the corrected y
    # on whatever the clean frames actually show, so corrected pairs
    # sit on the same (y_L − y_R) distribution as the rest of the trial.
    empirical_y_disp = np.full(J, np.nan, dtype=np.float64)
    for j in range(J):
        m = (~flagged_any[:, j]) \
            & ~np.isnan(mp_L[:, j, 1]) & ~np.isnan(mp_R[:, j, 1])
        if m.sum() >= 5:
            empirical_y_disp[j] = float(
                np.median(mp_L[m, j, 1] - mp_R[m, j, 1]))

    # ── Diagnostic baseline: y-disparity distribution on CLEAN frames ──
    # If the corrector is unbiased, corrected (y_L − y_R) per (frame,joint)
    # should sit within the calibration-predicted band, which the clean
    # frames already obey (modulo their own MP noise).
    diag_clean = []          # raw observed y_L − y_R on clean frames
    diag_clean_pred = []     # calibration-predicted y_L − y_R on clean frames
    for j in range(J):
        m = (~flagged_any[:, j]) & ~np.isnan(mp_L[:, j, 1]) & ~np.isnan(mp_R[:, j, 1])
        if not m.any(): continue
        diag_clean.extend((mp_L[m, j, 1] - mp_R[m, j, 1]).tolist())
        # Calibration-predicted disparity from triangulated 3D
        p3d = mp_3d[m, j]
        if len(p3d):
            pts = p3d.reshape(-1, 1, 3).astype(np.float64)
            pL_p, _ = _cv2.projectPoints(pts, np.zeros((3, 1)), np.zeros((3, 1)),
                                          calib['K1'], calib['dist1'])
            pR_p, _ = _cv2.projectPoints(pts, _cv2.Rodrigues(R)[0], T.reshape(3, 1),
                                          calib['K2'], calib['dist2'])
            diag_clean_pred.extend(
                (pL_p.reshape(-1, 2)[:, 1] - pR_p.reshape(-1, 2)[:, 1]).tolist())
    diag_corr_post = []      # observed y_L − y_R AFTER correction (per fix)
    diag_corr_pred = []      # predicted y_L − y_R for the corrected 3D
    diag_residual_OS = []    # corrected_y_OS − pre-correction y_OS (when OS is bad)
    diag_residual_OD = []    # corrected_y_OD − pre-correction y_OD (when OD is bad)

    def _undistort_norm(p2d, K, dist):
        p = np.asarray(p2d, dtype=np.float64).reshape(1, 1, 2)
        return _cv2.undistortPoints(p, K, dist).reshape(2)

    def _lift_good(good_2d, good_cam, z_target):
        """World-frame 3D point on the good cam's ray with world Z = z_target."""
        if good_cam == 0:
            u_n, v_n = _undistort_norm(good_2d, K1, dist1)
            return np.array([u_n * z_target, v_n * z_target, float(z_target)])
        u_n, v_n = _undistort_norm(good_2d, K2, dist2)
        ray = np.array([u_n, v_n, 1.0])
        a = (Rt @ ray)[2]
        b = (Rt @ T)[2]
        if abs(a) < 1e-9:
            return np.full(3, np.nan)
        lam = (float(z_target) + b) / a
        X_cam2 = lam * ray
        return Rt @ (X_cam2 - T)

    def _project_world(X, cam):
        if X is None or np.any(np.isnan(X)):
            return None
        pts = X.reshape(1, 1, 3).astype(np.float64)
        if cam == 0:
            p, _ = _cv2.projectPoints(pts, rvec_I, tvec_0, K1, dist1)
        else:
            p, _ = _cv2.projectPoints(pts, rvec_R, tvec_R, K2, dist2)
        return p.reshape(2)

    count = 0
    s = float(z_outlier_slider) if z_outlier_slider is not None else 0.0
    s = max(0.0, min(1.0, s))
    for j in range(J):
        z = mp_3d[:, j, 2].copy()
        valid = ~np.isnan(z)
        clean = (~flagged_any[:, j]) & valid
        if clean.sum() < 5:
            continue
        clean_z = z[clean]
        med = float(np.median(clean_z))
        # Per-joint clip range matched to the z_outlier detection's threshold.
        # The detector computes raw_score = |z − med| / sigma where sigma =
        # max(MAD * 1.4826, MIN_SIGMA_3D), then percentile-ranks per joint
        # and flags scores above (1 − slider).  The clip uses the same
        # construction so any depth that *would* be flagged as an outlier
        # by the detector is also rejected by the corrector here.
        mad = float(np.median(np.abs(clean_z - med)))
        sigma = max(mad * 1.4826, MIN_SIGMA_3D)
        if s > 0 and valid.sum() >= 5:
            raw_scores = np.abs(z[valid] - med) / sigma
            score_thresh = float(np.quantile(raw_scores, max(0.0, 1.0 - s)))
        else:
            # No detection threshold set → use a generous fallback.
            score_thresh = 6.0
        z_lo = med - score_thresh * sigma
        z_hi = med + score_thresh * sigma
        # Two-phase ordering for this joint:
        #   1. Y-disparity-flagged frames first (sorted by |z deviation|).
        #   2. Then Z-outlier-only-flagged frames (sorted similarly).
        # Once Y-disp errors are corrected, the joint's Z trajectory becomes
        # well-behaved triangulation, so z-outlier corrections that follow
        # are scoring against a cleaner baseline.
        y_frames = np.where(y_flag_jt[:, j] & valid)[0]
        z_only_frames = np.where(z_flag_jt[:, j] & ~y_flag_jt[:, j] & valid)[0]
        if len(y_frames) == 0 and len(z_only_frames) == 0:
            continue
        y_devs = np.abs(z[y_frames] - med) if len(y_frames) else np.array([])
        z_devs = np.abs(z[z_only_frames] - med) if len(z_only_frames) else np.array([])
        order = np.concatenate([
            y_frames[np.argsort(-y_devs)] if len(y_frames) else np.array([], dtype=np.int64),
            z_only_frames[np.argsort(-z_devs)] if len(z_only_frames) else np.array([], dtype=np.int64),
        ])

        for f in order:
            f = int(f)
            e0 = bool(errors_union[f, j, 0])
            e1 = bool(errors_union[f, j, 1])
            if not e0 and not e1:
                continue
            # Bad camera attribution
            if e0 and not e1:
                bad_cam = 0
            elif e1 and not e0:
                bad_cam = 1
            else:
                src = attr_per_cam if attr_per_cam is not None else jump_2d_scores
                if src is not None:
                    s0 = src[f, j, 0]; s1 = src[f, j, 1]
                    s0 = 0.0 if np.isnan(s0) else float(s0)
                    s1 = 0.0 if np.isnan(s1) else float(s1)
                else:
                    s0 = s1 = 0.0
                bad_cam = 0 if s0 >= s1 else 1
            good_cam = 1 - bad_cam

            # Interpolate target Z from clean neighbours.  Requirements:
            # 1) at least one clean anchor on EACH side of f within ±W —
            #    one-sided clean data extrapolates wildly across long
            #    bursts of flagged frames (e.g. index DIP frames 18-34).
            # 2) clip the predicted z to the joint's robust range so a
            #    quadratic that overshoots can't crash the projection.
            lo, hi = max(0, f - W), min(N, f + W + 1)
            window_idxs = np.arange(lo, hi)
            local_clean = clean[lo:hi]
            left_ok  = bool(local_clean[: f - lo].any())
            right_ok = bool(local_clean[f - lo + 1:].any())
            if left_ok and right_ok and local_clean.sum() >= 3:
                z_target = _poly_predict_Z(
                    window_idxs[local_clean], z[lo:hi][local_clean], f)
            else:
                # One-sided window or too few anchors → don't extrapolate.
                # Use a robust local mean (mean of nearest few clean frames
                # on either available side) — safer than a polynomial that
                # can swing wildly outside the data.
                anchors = z[lo:hi][local_clean]
                if len(anchors) >= 2:
                    z_target = float(np.median(anchors))
                else:
                    # Try widening to all clean frames in the trial.
                    if clean.sum() >= 5:
                        z_target = med  # global per-joint median fallback
                    else:
                        continue
            if np.isnan(z_target) or z_target <= 0:
                continue
            # Clip to robust per-joint range so the lift can't produce
            # an absurd 3D point even if the fit slightly overshoots.
            z_target = float(np.clip(z_target, z_lo, z_hi))

            good_2d = (mp_L_c[f, j] if good_cam == 0 else mp_R_c[f, j])
            if np.any(np.isnan(good_2d)):
                continue
            X = _lift_good(good_2d, good_cam, z_target)
            new_2d = _project_world(X, bad_cam)
            if new_2d is None or np.any(np.isnan(new_2d)):
                continue

            # Override the y component with the empirical-disparity result.
            # The calibration's predicted y carries a systematic Ty bias
            # vs. observed clean labels; matching the empirical distribution
            # makes corrected frames blend with clean frames cleanly.
            disp_j = empirical_y_disp[j]
            if not np.isnan(disp_j):
                if bad_cam == 0:
                    good_y_R = float(mp_R_c[f, j, 1])
                    if not np.isnan(good_y_R):
                        new_2d = np.array([new_2d[0], good_y_R + disp_j])
                else:
                    good_y_L = float(mp_L_c[f, j, 1])
                    if not np.isnan(good_y_L):
                        new_2d = np.array([new_2d[0], good_y_L - disp_j])

            # Diagnostics (collected before the assignment so we can compare)
            pre_yL = float(mp_L_c[f, j, 1])
            pre_yR = float(mp_R_c[f, j, 1])

            if bad_cam == 0:
                mp_L_c[f, j] = new_2d
                if not np.isnan(pre_yL):
                    diag_residual_OS.append(float(new_2d[1]) - pre_yL)
            else:
                mp_R_c[f, j] = new_2d
                if not np.isnan(pre_yR):
                    diag_residual_OD.append(float(new_2d[1]) - pre_yR)

            # Observed disparity after correction (uses both cameras' final values)
            post_L = float(mp_L_c[f, j, 1])
            post_R = float(mp_R_c[f, j, 1])
            if not (np.isnan(post_L) or np.isnan(post_R)):
                diag_corr_post.append(post_L - post_R)
            # Calibration-predicted disparity for the lifted 3D
            if X is not None and not np.any(np.isnan(X)):
                pts = X.reshape(1, 1, 3).astype(np.float64)
                pL_p, _ = _cv2.projectPoints(
                    pts, np.zeros((3, 1)), np.zeros((3, 1)),
                    calib['K1'], calib['dist1'])
                pR_p, _ = _cv2.projectPoints(
                    pts, _cv2.Rodrigues(R)[0], T.reshape(3, 1),
                    calib['K2'], calib['dist2'])
                diag_corr_pred.append(
                    float(pL_p.reshape(2)[1] - pR_p.reshape(2)[1]))

            z[f] = float(z_target)
            clean[f] = True
            count += 1

    # ── Summarise the diagnostics ────────────────────────────────────
    def _stats(name, arr):
        if not arr:
            return f"  {name:24s}: (no samples)"
        a = np.asarray(arr, dtype=np.float64)
        return (f"  {name:24s}: n={len(a):5d}  mean={a.mean():+7.3f}  "
                f"median={np.median(a):+7.3f}  std={a.std():6.3f}  "
                f"p10={np.percentile(a, 10):+7.3f}  p90={np.percentile(a, 90):+7.3f}")
    logger.info("[yz-correct] y-disparity diagnostics (px):")
    logger.info(_stats("clean observed (y_L−y_R)", diag_clean))
    logger.info(_stats("clean predicted (y_L−y_R)", diag_clean_pred))
    logger.info(_stats("corrected observed", diag_corr_post))
    logger.info(_stats("corrected predicted", diag_corr_pred))
    logger.info(_stats("OS bad: Δy_OS (post−pre)", diag_residual_OS))
    logger.info(_stats("OD bad: Δy_OD (post−pre)", diag_residual_OD))
    if diag_corr_post and diag_clean:
        bias = float(np.median(diag_corr_post)) - float(np.median(diag_clean))
        logger.info(f"  median bias: corrected − clean = {bias:+.3f} px")

    return mp_L_c, mp_R_c, count


def _poly_predict_Z(frames: np.ndarray, z_vals: np.ndarray, target_f: int) -> float:
    """Local polynomial regression: fit degree-2 (or 1 if ≤4 pts) in time to
    ``z_vals`` at ``frames`` indices, evaluate at ``target_f``.  Returns NaN
    if too few points or fit fails."""
    if len(frames) < 3:
        return float('nan')
    deg = 2 if len(frames) >= 5 else 1
    try:
        coef = np.polyfit(frames.astype(np.float64), z_vals.astype(np.float64), deg)
        return float(np.polyval(coef, target_f))
    except (np.linalg.LinAlgError, ValueError):
        return float('nan')


def correct_z_from_errors(mp_L: np.ndarray, mp_R: np.ndarray,
                          errors: np.ndarray, calib,
                          jump_2d_scores: np.ndarray | None,
                          window: int = 15) -> tuple:
    """Z-outlier correction: shift labels laterally so Z matches the target
    predicted by polynomial regression over clean neighbouring frames.

    Algorithm:
    1. Triangulate → current Z per (frame, joint).
    2. Per joint, collect clean frames (not flagged by Z error).
    3. Sort flagged frames by deviation magnitude, LARGEST FIRST.
    4. For each flagged frame, fit a local quadratic polynomial to the
       clean frames in a ±15-frame window and evaluate at the flagged
       frame to get target Z.
    5. Convert target Z → target X-disparity via ``disp_target =
       disp_cur × (Z_cur / Z_target)`` (parallel-stereo approx.).
    6. Distribute the Δdisparity between cameras by 2D-jump share —
       higher jump absorbs more of the shift.
    7. Mark the corrected frame as "clean" so SMALLER errors nearby
       can use it in their polynomial fits.  This keeps bursts of
       outliers from mutually sabotaging each other.

    Returns ``(corrected_mp_L, corrected_mp_R, n_applied)``.
    """
    mp_L_c = mp_L.copy()
    mp_R_c = mp_R.copy()
    N, J, _ = mp_L.shape
    W = max(1, int(window))   # window half-width for local polynomial fit

    # Current 3D
    mp_3d = np.full((N, J, 3), np.nan, dtype=np.float32)
    for j in range(J):
        mp_3d[:, j] = triangulate_points(
            mp_L[:, j], mp_R[:, j], calib).astype(np.float32)

    flagged_any = errors[:, :, 0] | errors[:, :, 1]
    count = 0

    for j in range(J):
        z = mp_3d[:, j, 2]
        valid = ~np.isnan(z)
        if valid.sum() < 5:
            continue
        # Clean mask: not flagged and not NaN.  Updated as we correct.
        clean = (~flagged_any[:, j]) & valid
        # Global baseline for ordering the corrections by deviation magnitude.
        # Use robust median + MAD of clean frames so the ordering is stable.
        clean_z = z[clean]
        if len(clean_z) < 5:
            continue
        med = float(np.median(clean_z))
        flagged_frames = np.where(flagged_any[:, j] & valid)[0]
        if len(flagged_frames) == 0:
            continue
        # Sort flagged frames by descending deviation from clean median
        devs = np.abs(z[flagged_frames] - med)
        order = flagged_frames[np.argsort(-devs)]

        for f in order:
            f = int(f)
            z_cur = float(z[f])
            if np.isnan(z_cur) or z_cur <= 0:
                continue
            # Local polynomial fit on CURRENT clean frames in window ±W
            lo, hi = max(0, f - W), min(N, f + W + 1)
            idxs = np.arange(lo, hi)
            local_clean = clean[lo:hi]
            if local_clean.sum() < 3:
                # Fallback: use all clean frames globally
                if clean.sum() < 3:
                    continue
                z_target = _poly_predict_Z(
                    np.where(clean)[0], z[clean], f)
            else:
                z_target = _poly_predict_Z(
                    idxs[local_clean], z[lo:hi][local_clean], f)
            if np.isnan(z_target) or z_target <= 0:
                continue
            if abs(z_target - z_cur) < 0.5:
                continue

            x_l = float(mp_L_c[f, j, 0])
            x_r = float(mp_R_c[f, j, 0])
            if np.isnan(x_l) or np.isnan(x_r):
                continue
            disp_cur = x_l - x_r
            disp_target = disp_cur * (z_cur / z_target)
            delta_disp = disp_target - disp_cur

            # Winner-take-all camera attribution: move only the noisier
            # camera's X.  (Per request: don't split the shift across both
            # — apply it entirely to whichever camera has the higher 2D
            # jump for this joint at this frame.)
            if jump_2d_scores is not None:
                j0 = jump_2d_scores[f, j, 0]
                j1 = jump_2d_scores[f, j, 1]
                j0 = 0.0 if np.isnan(j0) else float(j0)
                j1 = 0.0 if np.isnan(j1) else float(j1)
            else:
                j0 = j1 = 0.0

            # disp = X_L − X_R → to change disp by delta, shift one cam:
            #   move OS by +delta if OS is the loser (no need to negate),
            #   move OD by −delta if OD is the loser.
            if j0 >= j1:
                mp_L_c[f, j, 0] = x_l + delta_disp
            else:
                mp_R_c[f, j, 0] = x_r - delta_disp

            # Update z at this frame so the next correction sees the new value
            z[f] = z_target
            # This frame is now corrected — treat as clean for subsequent
            # polynomial fits on smaller-deviation flagged frames.
            clean[f] = True
            count += 1

    return mp_L_c, mp_R_c, count


WRIST_BONES_LIST = [(0, 1), (0, 5), (0, 9), (0, 13), (0, 17)]
FINGER_BONES_LIST = [
    [(1, 2), (2, 3), (3, 4)],
    [(5, 6), (6, 7), (7, 8)],
    [(9, 10), (10, 11), (11, 12)],
    [(13, 14), (14, 15), (15, 16)],
    [(17, 18), (18, 19), (19, 20)],
]


def _project_point_stereo(pt_3d: np.ndarray, calib: dict) -> tuple:
    """Project a single 3D point (3,) into both cameras.  Returns
    ``((xL, yL), (xR, yR))`` in pixel coordinates, or ``(None, None)``
    if the point is invalid."""
    if pt_3d is None or np.any(np.isnan(pt_3d)):
        return (None, None)
    pts = pt_3d.reshape(1, 1, 3).astype(np.float64)
    # Left: identity pose
    rvec_I = np.zeros((3, 1))
    tvec_0 = np.zeros((3, 1))
    pL, _ = cv2.projectPoints(pts, rvec_I, tvec_0, calib['K1'], calib['dist1'])
    # Right: stereo R, T
    R = calib.get('R')
    T = calib.get('T')
    rvec_R = cv2.Rodrigues(R)[0]
    tvec_R = T.reshape(3, 1)
    pR, _ = cv2.projectPoints(pts, rvec_R, tvec_R, calib['K2'], calib['dist2'])
    return (pL.reshape(2), pR.reshape(2))


def _local_poly_fit_3d(frames: np.ndarray, pts: np.ndarray, target_f: int) -> np.ndarray:
    """Fit a local quadratic to each 3D component independently, evaluate
    at ``target_f``.  Returns (3,) target position or [nan]*3 on failure."""
    if len(frames) < 3:
        return np.full(3, np.nan)
    out = np.empty(3)
    deg = 2 if len(frames) >= 3 else 1
    for k in range(3):
        try:
            c = np.polyfit(frames, pts[:, k], min(deg, len(frames) - 1))
            out[k] = np.polyval(c, target_f)
        except Exception:
            return np.full(3, np.nan)
    return out


def _load_mp_labels(subject_name: str, trial_stem: str, start: int, N: int):
    """Load MP labels using the SAME precedence as mano_data.load_mano_trial_data
    so the correction pipeline and the viewer's 'MediaPipe' stage read from
    the same source.  Returns (mp_L, mp_R, conf_L, conf_R), with the
    legacy per-trial ``mediapipe.pkl`` preferred over the app's
    ``mediapipe_prelabels.npz``."""
    import pickle

    mp_dir = _mano_dir(subject_name) / trial_stem
    pkl_path = mp_dir / "mediapipe.pkl"
    if pkl_path.exists():
        try:
            with open(pkl_path, "rb") as f:
                mp_data = pickle.load(f)
            mp_L_pkl = mp_data.get("tracked_L")
            mp_R_pkl = mp_data.get("tracked_R")
            if mp_L_pkl is not None and mp_R_pkl is not None:
                mp_L = np.full((N, 21, 2), np.nan, dtype=np.float32)
                mp_R = np.full((N, 21, 2), np.nan, dtype=np.float32)
                n = min(N, int(mp_L_pkl.shape[0]))
                mp_L[:n] = mp_L_pkl[:n].astype(np.float32)
                mp_R[:n] = mp_R_pkl[:n].astype(np.float32)
                # Legacy pkl has no per-frame confidence → leave as None
                return mp_L, mp_R, None, None
        except Exception:
            pass  # fall back to npz on any load failure

    from .mediapipe_prelabel import load_mediapipe_prelabels
    pre = load_mediapipe_prelabels(subject_name)
    if pre is None:
        # No MediaPipe data yet for this subject — return all-NaN arrays so
        # the error/detection pipeline can short-circuit gracefully instead
        # of raising a 500 into the UI.
        mp_L = np.full((N, 21, 2), np.nan, dtype=np.float32)
        mp_R = np.full((N, 21, 2), np.nan, dtype=np.float32)
        return mp_L, mp_R, None, None
    mp_L = pre["OS_landmarks"][start:start+N].copy().astype(np.float32)
    mp_R = pre["OD_landmarks"][start:start+N].copy().astype(np.float32)
    conf_L = pre.get("confidence_OS")
    conf_R = pre.get("confidence_OD")
    if conf_L is not None:
        conf_L = np.array(conf_L)[start:start+N]
    if conf_R is not None:
        conf_R = np.array(conf_R)[start:start+N]
    return mp_L, mp_R, conf_L, conf_R


# Joint → (child_joint) edges for "child-bone direction" used by the
# HRnet-snap systematic-offset correction.  Fingertips have no child, so
# we extrapolate along the parent bone (parent → tip direction).
HRNET_CHILD_MAP = {
    # Thumb, index, middle, ring, pinky — MCP → PIP → DIP → (tip uses parent)
    1: 2, 2: 3, 3: 4,
    5: 6, 6: 7, 7: 8,
    9: 10, 10: 11, 11: 12,
    13: 14, 14: 15, 15: 16,
    17: 18, 18: 19, 19: 20,
}
HRNET_TIPS = {4: 3, 8: 7, 12: 11, 16: 15, 20: 19}  # tip → parent

# Parent joint for each non-wrist joint.  Used by the 3D HRnet-offset
# basis so the frame at each joint matches the flex/abd joint-angle
# constraints (b_in = parent→joint, dorsal = flex axis, cross = abd axis).
HRNET_PARENT_MAP = {
    1: 0, 2: 1, 3: 2, 4: 3,
    5: 0, 6: 5, 7: 6, 8: 7,
    9: 0, 10: 9, 11: 10, 12: 11,
    13: 0, 14: 13, 15: 14, 16: 15,
    17: 0, 18: 17, 19: 18, 20: 19,
}


def _hrnet_attr_score(mp_L: np.ndarray, mp_R: np.ndarray,
                       subject_name: str, trial_stem: str) -> np.ndarray | None:
    """Per-camera attribution score from HRnet.

    Primary signal: heatmap value at the MP label position (higher value
    → better placement).  Score = ``1 − h_sampled`` so worse placement is
    a higher score.

    Fallback when both cameras' sampled values are weak (≤ 0.3): switch
    to a distance-based score equal to the pixel distance from the MP
    label to the HRnet refined peak (the "Peak select" stage's peak).
    The camera whose label is farther from the peak gets a higher score.
    This rescues frames where neither label landed on the peak — the
    heatmap-value signal is uninformative there but spatial proximity
    to the peak is.

    Returns ``(N, 21, 2)`` float32 or ``None`` when heatmaps / crop info
    are missing for this trial."""
    import json as _json
    from .mano_data import _mano_dir
    mano_trial_dir = _mano_dir(subject_name) / trial_stem
    hm_path = mano_trial_dir / "hrnet_w18_heatmaps.npz"
    crop_path = mano_trial_dir / "hand_crop.json"
    if not hm_path.exists() or not crop_path.exists():
        return None
    try:
        crop_info = _json.loads(crop_path.read_text())
    except Exception:
        return None
    try:
        hm = np.load(str(hm_path), mmap_mode="r")
    except Exception:
        return None

    N, J, _ = mp_L.shape
    # Per-frame bboxes — falls back to broadcasting the legacy union
    # bbox when an older ``hand_crop.json`` only has ``crop_L``/``crop_R``.
    from .hrnet_bbox import read_per_frame_bboxes
    bboxes_L_pf = read_per_frame_bboxes(crop_info, "crop_L", N)
    bboxes_R_pf = read_per_frame_bboxes(crop_info, "crop_R", N)
    if bboxes_L_pf is None and bboxes_R_pf is None:
        return None

    # Sampled heatmap values per (frame, joint, camera).  NaN where the
    # MP label falls outside the per-frame crop bbox or the heatmap is
    # missing.
    h_val = np.full((N, J, 2), np.nan, dtype=np.float32)

    def _sample(heatmaps, mp, bboxes_pf, col):
        if bboxes_pf is None or heatmaps is None: return
        Hf, Wf = heatmaps.shape[2], heatmaps.shape[3]
        for f in range(N):
            x1, y1, x2, y2 = float(bboxes_pf[f, 0]), float(bboxes_pf[f, 1]), \
                             float(bboxes_pf[f, 2]), float(bboxes_pf[f, 3])
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0: continue
            for j in range(J):
                x = mp[f, j, 0]; y = mp[f, j, 1]
                if np.isnan(x) or np.isnan(y): continue
                hx = (x - x1) / bw * Wf - 0.5
                hy = (y - y1) / bh * Hf - 0.5
                xi = int(np.floor(hx)); yi = int(np.floor(hy))
                if not (0 <= xi and xi + 1 < Wf and 0 <= yi and yi + 1 < Hf):
                    continue
                tx = hx - xi; ty = hy - yi
                hm_f = heatmaps[f, j].astype(np.float32)
                v = ((1 - tx) * (1 - ty) * hm_f[yi, xi]
                     + tx * (1 - ty) * hm_f[yi, xi + 1]
                     + (1 - tx) * ty * hm_f[yi + 1, xi]
                     + tx * ty * hm_f[yi + 1, xi + 1])
                h_val[f, j, col] = float(v)

    hL = hm["heatmaps_L"] if "heatmaps_L" in hm.files else None
    hR = hm["heatmaps_R"] if "heatmaps_R" in hm.files else None
    _sample(hL, mp_L, bboxes_L_pf, 0)
    _sample(hR, mp_R, bboxes_R_pf, 1)

    # Heatmap-value-based score: higher value → lower score.
    out = np.full((N, J, 2), np.nan, dtype=np.float32)
    have_h = ~np.isnan(h_val)
    out[have_h] = 1.0 - h_val[have_h]

    # Distance-based fallback: load the refined HRnet peaks ("Peak select"
    # stage) and compute |MP − peak| per camera.  Activated for
    # (frame, joint) cells where BOTH cameras' sampled values are ≤ 0.3.
    peaks_L, peaks_R = _load_hrnet_peaks_arr(subject_name, trial_stem, 0, N)
    if peaks_L is not None and peaks_R is not None:
        with np.errstate(invalid="ignore"):
            both_weak = (np.nan_to_num(h_val[:, :, 0], nan=0.0) <= 0.3) & \
                        (np.nan_to_num(h_val[:, :, 1], nan=0.0) <= 0.3)
        # |MP − peak| per camera.
        d_L = np.linalg.norm(mp_L[:, :, :2] - peaks_L[:, :, :2], axis=-1)
        d_R = np.linalg.norm(mp_R[:, :, :2] - peaks_R[:, :, :2], axis=-1)
        # Where both are weak, override the heatmap-based score with the
        # raw pixel distance.  Percentile-ranking downstream normalises
        # the scale so mixing units is harmless within a joint's column.
        out[both_weak, 0] = np.where(np.isnan(d_L[both_weak]), np.nan, d_L[both_weak]).astype(np.float32)
        out[both_weak, 1] = np.where(np.isnan(d_R[both_weak]), np.nan, d_R[both_weak]).astype(np.float32)
    return out


def _load_hrnet_peaks_arr(subject_name: str, trial_stem: str,
                           start: int, N: int,
                           source: str = "auto") -> tuple:
    """Load HRnet peaks from ``hrnet_peak_assignments.json`` into (N, 21, 2)
    arrays.  Returns (peaks_L, peaks_R) or (None, None).

    ``source`` selects which peak set:
      * ``"auto"``      — prefer hungarian → centroid → legacy refined → raw.
      * ``"hungarian"`` — HRnet Fit Stage 2 (joint stereo Hungarian).
      * ``"centroid"``  — HRnet Fit Stage 1 (cluster centroid).
      * ``"refined"``   — legacy MP-Hungarian Peak-Select.
      * ``"raw"``       — legacy raw argmax.
    """
    import json as _json
    from .mano_data import _mano_dir, JOINT_NAMES
    p = _mano_dir(subject_name) / trial_stem / "hrnet_peak_assignments.json"
    if not p.exists():
        return None, None
    try:
        d = _json.loads(p.read_text())
    except Exception:
        return None, None

    def _get(field_L, field_R):
        return d.get(field_L) or {}, d.get(field_R) or {}

    if source == "hungarian":
        pL_in, pR_in = _get("peaks_hungarian_L", "peaks_hungarian_R")
    elif source == "centroid":
        pL_in, pR_in = _get("peaks_centroid_L",  "peaks_centroid_R")
    elif source == "refined":
        pL_in, pR_in = _get("peaks_L", "peaks_R")
    elif source == "raw":
        pL_in, pR_in = _get("peaks_L_raw", "peaks_R_raw")
    else:  # "auto" — fall through preference order
        for fl, fr in (
            ("peaks_hungarian_L", "peaks_hungarian_R"),
            ("peaks_centroid_L",  "peaks_centroid_R"),
            ("peaks_L",           "peaks_R"),
            ("peaks_L_raw",       "peaks_R_raw"),
        ):
            pL_in, pR_in = _get(fl, fr)
            if pL_in or pR_in:
                break
    if not pL_in and not pR_in:
        return None, None

    def _pack(src):
        out = np.full((N, 21, 2), np.nan, dtype=np.float32)
        for j in range(21):
            frames = src.get(JOINT_NAMES[j])
            if not frames: continue
            for f in range(N):
                fg = f  # peak json indexes are already trial-local
                if fg < len(frames) and frames[fg] is not None:
                    out[f, j] = frames[fg]
        return out
    return _pack(pL_in), _pack(pR_in)


def _palm_basis_3d(joints_3d_f: np.ndarray) -> np.ndarray | None:
    """Unit palm-normal for a single frame's (21, 3) joints, computed from
    the wrist + 4 MCPs (indices 5/9/13/17).  Returns (3,) or None if the
    4 MCPs don't define a plane."""
    try:
        p5  = joints_3d_f[5]
        p13 = joints_3d_f[13]
        p0  = joints_3d_f[0]
        if np.any(np.isnan(p0)) or np.any(np.isnan(p5)) or np.any(np.isnan(p13)):
            return None
        v1 = p5 - p0; v2 = p13 - p0
        n = np.cross(v1, v2)
        L = float(np.linalg.norm(n))
        if L < 1e-6: return None
        return (n / L).astype(np.float32)
    except Exception:
        return None


def _joint_basis_3d(joints_3d_f: np.ndarray, j: int):
    """Per-frame orthonormal basis at joint ``j``, matching the flex/abd
    axes used by the joint-angle constraints.

      e_along = unit(joint − parent)   (b_in)
      e_flex  = dorsal axis (palm normal projected ⊥ e_along)
      e_abd   = cross(e_flex, e_along)  (in the plane perpendicular to b_in)

    Returns (e_along, e_flex, e_abd) each (3,) or None if degenerate.
    """
    parent = HRNET_PARENT_MAP.get(j)
    if parent is None:
        return None
    p = joints_3d_f[parent]; q = joints_3d_f[j]
    if np.any(np.isnan(p)) or np.any(np.isnan(q)):
        return None
    b_in = q - p
    L = float(np.linalg.norm(b_in))
    if L < 1e-6: return None
    e_along = (b_in / L).astype(np.float32)
    palm = _palm_basis_3d(joints_3d_f)
    if palm is None:
        return None
    # Dorsal axis = palm normal projected perpendicular to b_in.
    e_flex = palm - float(np.dot(palm, e_along)) * e_along
    fl = float(np.linalg.norm(e_flex))
    if fl < 1e-6: return None
    e_flex = (e_flex / fl).astype(np.float32)
    e_abd = np.cross(e_flex, e_along).astype(np.float32)
    return e_along, e_flex, e_abd


def _hrnet_snap_compute_offsets_3d(mp_3d: np.ndarray, hr_3d: np.ndarray) -> tuple:
    """3D per-joint systematic-offset estimation.

    Builds a per-frame basis at each joint that matches the flex/abd
    axes used by the joint-angle constraints:

      e_along = b_in        (parent → joint)
      e_flex  = dorsal axis (flex / extension direction)
      e_abd   = cross(flex, along)  (abduction / adduction direction)

    Decomposes (HR_3d − MP_3d) onto (along, flex, abd), then medians each
    scalar over the trial → three systematic-offset constants per joint.

    Returns (along_med, flex_med, abd_med, child_of).
    """
    N = mp_3d.shape[0]
    along_med = np.zeros(21, dtype=np.float32)
    flex_med  = np.zeros(21, dtype=np.float32)
    abd_med   = np.zeros(21, dtype=np.float32)
    child_of = np.full(21, -1, dtype=np.int32)
    for j in range(21):
        if j in HRNET_CHILD_MAP:
            child_of[j] = HRNET_CHILD_MAP[j]
        elif j in HRNET_TIPS:
            child_of[j] = j
        else:
            continue
        av = []; fv = []; bv = []
        for f in range(N):
            mp = mp_3d[f, j]; hr = hr_3d[f, j]
            if np.any(np.isnan(mp)) or np.any(np.isnan(hr)): continue
            basis = _joint_basis_3d(mp_3d[f], j)
            if basis is None: continue
            e_along, e_flex, e_abd = basis
            delta = hr - mp
            av.append(float(np.dot(delta, e_along)))
            fv.append(float(np.dot(delta, e_flex)))
            bv.append(float(np.dot(delta, e_abd)))
        if len(av) >= 5:
            along_med[j] = float(np.median(av))
            flex_med[j]  = float(np.median(fv))
            abd_med[j]   = float(np.median(bv))
    return along_med, flex_med, abd_med, child_of


def _hrnet_snap_compute_offsets(mp_2d: np.ndarray, hr_2d: np.ndarray) -> tuple:
    """Per-joint median systematic offset between HRnet peak and MP label,
    decomposed into along-bone and perpendicular-to-bone components.

    mp_2d, hr_2d: (N, 21, 2) single-camera arrays.
    Returns (along_med, perp_med, child_pairs): two (21,) arrays of scalar
    medians and the child-joint used for each (for snap-time direction).
    """
    N = mp_2d.shape[0]
    along_med = np.zeros(21, dtype=np.float32)
    perp_med  = np.zeros(21, dtype=np.float32)
    child_of = np.full(21, -1, dtype=np.int32)
    for j in range(21):
        if j in HRNET_CHILD_MAP:
            c = HRNET_CHILD_MAP[j]; parent = None
        elif j in HRNET_TIPS:
            c = j; parent = HRNET_TIPS[j]
        else:
            continue
        child_of[j] = c
        along_vals = []; perp_vals = []
        for f in range(N):
            mp = mp_2d[f, j]; hr = hr_2d[f, j]
            if np.any(np.isnan(mp)) or np.any(np.isnan(hr)): continue
            # Bone direction for this frame: j→child, or for tips parent→tip.
            if parent is None:
                other = mp_2d[f, c]
                if np.any(np.isnan(other)): continue
                bd = other - mp
            else:
                other = mp_2d[f, parent]
                if np.any(np.isnan(other)): continue
                bd = mp - other  # parent → tip, extend same direction past tip
            L = float(np.linalg.norm(bd))
            if L < 1e-6: continue
            bx, by = bd[0] / L, bd[1] / L
            # Perpendicular (90° CW in image coords)
            px, py = by, -bx
            dx, dy = float(hr[0] - mp[0]), float(hr[1] - mp[1])
            along_vals.append(dx * bx + dy * by)
            perp_vals.append(dx * px + dy * py)
        if len(along_vals) >= 5:
            along_med[j] = float(np.median(along_vals))
            perp_med[j]  = float(np.median(perp_vals))
    return along_med, perp_med, child_of


def hrnet_snap_labels(mp_L: np.ndarray, mp_R: np.ndarray,
                      hr_L: np.ndarray, hr_R: np.ndarray,
                      mismatch_threshold_px: float = 0.0,
                      calib: dict | None = None) -> tuple:
    """Snap MP labels for joints that mismatch the HRnet Peak-Select.

    Detection (per joint, per frame):
      - Build joint basis (e_along, e_flex, e_abd) from the Z-smooth 3D
        pose.  Estimate per-joint medians of (HR_3d − MP_3d) projected
        onto this basis (the "anatomical offset").
      - Predict the HRnet peak 3D = MP_3d + offset.  Project into both
        cameras and compare to the actual HRnet Peak-Select 2D peaks.
      - Flag the joint if EITHER camera's distance exceeds the threshold.

    Correction for flagged joints:
      - Take the raw HRnet peak 3D and SUBTRACT the same anatomical
        offset (so the corrected position lands at the MP-style joint
        centre, not the HR-peak centre — this keeps corrected joints
        consistent with the unflagged MP labels around them).
      - Project that target 3D into both cameras → snapped 2D for OS+OD.
      - Replace BOTH cameras' MP labels with the snapped values.

    Wrist (0) and thumb CMC (1) are exempt — their HRnet heatmaps are
    too unreliable to drive corrections.

    The downstream pipeline re-runs Y/Z-correct and Z-jump on the snapped
    labels to clean up any residual inconsistency.

    Returns (mp_L_snapped, mp_R_snapped, offsets_dict).
    """
    mp_L_c = mp_L.copy(); mp_R_c = mp_R.copy()
    if calib is None or hr_L is None or hr_R is None:
        return _hrnet_snap_labels_2d(mp_L, mp_R, hr_L, hr_R, mismatch_threshold_px)
    N, J, _ = mp_L.shape

    # Per-(frame, joint) bool mask of cells the snap actually replaced;
    # returned to the pipeline so the post-snap Y/Z re-clean can skip
    # those cells (otherwise it would re-flag them as y_disp/z_outlier
    # outliers vs. the empirical baseline and pull them off-peak).
    snap_mask = np.zeros((N, 21), dtype=bool)

    mp_3d = np.full((N, 21, 3), np.nan, dtype=np.float32)
    hr_3d = np.full((N, 21, 3), np.nan, dtype=np.float32)
    for j in range(21):
        mp_3d[:, j] = triangulate_points(mp_L[:, j], mp_R[:, j], calib).astype(np.float32)
        hr_3d[:, j] = triangulate_points(hr_L[:, j], hr_R[:, j], calib).astype(np.float32)
    along_med, flex_med, abd_med, child_of = _hrnet_snap_compute_offsets_3d(mp_3d, hr_3d)

    # Per-joint EMPIRICAL y-disparity from the input (Z-smooth-stage)
    # MP labels.  Used to re-write the snapped pair's y so it matches
    # the convention of the rest of the trial — otherwise snapped pixels
    # land at the calibration-predicted disparity (~17 px off from
    # empirical for trials with a Ty miscalibration), giving a visible
    # ladder offset between snapped and unsnapped joints.
    empirical_y_disp = np.full(J, np.nan, dtype=np.float64)
    for j in range(J):
        m = ~np.isnan(mp_L[:, j, 1]) & ~np.isnan(mp_R[:, j, 1])
        if m.sum() >= 5:
            empirical_y_disp[j] = float(np.median(mp_L[m, j, 1] - mp_R[m, j, 1]))

    thr = float(mismatch_threshold_px) if mismatch_threshold_px > 0 else 0.0
    for j in range(21):
        if j in _HRNET_EXEMPT_JOINTS: continue  # wrist + thumb CMC ignored
        if child_of[j] < 0: continue
        for f in range(N):
            mp_pt = mp_3d[f, j]; hr_pt = hr_3d[f, j]
            if np.any(np.isnan(mp_pt)) or np.any(np.isnan(hr_pt)): continue
            basis = _joint_basis_3d(mp_3d[f], j)
            if basis is None: continue
            e_along, e_flex, e_abd = basis
            offset_vec = (float(along_med[j]) * e_along
                          + float(flex_med[j])  * e_flex
                          + float(abd_med[j])   * e_abd)

            # Detection: predicted HRnet 2D = project(MP_3d + offset).
            pred_hr_3d = mp_pt + offset_vec
            pL_pred, pR_pred = _project_point_stereo(
                pred_hr_3d.astype(np.float32), calib)
            if pL_pred is None or pR_pred is None: continue
            hrL2d = hr_L[f, j]; hrR2d = hr_R[f, j]
            if np.any(np.isnan(hrL2d)) or np.any(np.isnan(hrR2d)): continue
            d_L = float(np.hypot(pL_pred[0] - hrL2d[0], pL_pred[1] - hrL2d[1]))
            d_R = float(np.hypot(pR_pred[0] - hrR2d[0], pR_pred[1] - hrR2d[1]))
            if thr > 0 and max(d_L, d_R) <= thr:
                continue

            # Correction: anchor on the EMPIRICAL HR-peak 2D label (the
            # convention every un-snapped joint already uses) and apply
            # the calibration-derived projection of -offset_vec as a
            # delta.  This sidesteps the ~9 px y-residual that
            # triangulation+reprojection round-trips would introduce.
            #
            #   snap_OS = HR_OS + (project(target_3d) − project(HR_3d))_OS
            #   snap_OD = HR_OD + (project(target_3d) − project(HR_3d))_OD
            #
            # The Ty calibration bias affects project(target_3d) and
            # project(HR_3d) equally and cancels in the delta, so the
            # snap output sits at "HR peak − projected anatomical offset"
            # in the empirical 2D coordinate system.
            target_3d = hr_pt - offset_vec
            pL_tgt, pR_tgt = _project_point_stereo(
                target_3d.astype(np.float32), calib)
            pL_hr,  pR_hr  = _project_point_stereo(
                hr_pt.astype(np.float32), calib)
            if pL_tgt is None or pR_tgt is None: continue
            if pL_hr  is None or pR_hr  is None: continue
            xL = float(hrL2d[0]) + float(pL_tgt[0]) - float(pL_hr[0])
            yL = float(hrL2d[1]) + float(pL_tgt[1]) - float(pL_hr[1])
            xR = float(hrR2d[0]) + float(pR_tgt[0]) - float(pR_hr[0])
            yR = float(hrR2d[1]) + float(pR_tgt[1]) - float(pR_hr[1])

            mp_L_c[f, j, 0] = xL; mp_L_c[f, j, 1] = yL
            mp_R_c[f, j, 0] = xR; mp_R_c[f, j, 1] = yR
            snap_mask[f, j] = True

    return mp_L_c, mp_R_c, {
        "along_3d": along_med, "flex_3d": flex_med, "abd_3d": abd_med,
        "child_3d": child_of,
        "snap_mask": snap_mask,
    }


def _hrnet_snap_labels_2d(mp_L, mp_R, hr_L, hr_R, mismatch_threshold_px):
    """Legacy per-camera 2D fallback, used when calibration is missing."""
    mp_L_c = mp_L.copy(); mp_R_c = mp_R.copy()
    N = mp_L.shape[0]
    def _snap_one(mp_c, hr):
        if hr is None: return mp_c, None, None, None
        along_med, perp_med, child_of = _hrnet_snap_compute_offsets(mp_c, hr)
        out = mp_c.copy()
        for j in range(21):
            if child_of[j] < 0: continue
            c = int(child_of[j])
            parent = HRNET_TIPS.get(j)
            for f in range(N):
                mp = mp_c[f, j]; hrp = hr[f, j]
                if np.any(np.isnan(mp)) or np.any(np.isnan(hrp)): continue
                other = mp_c[f, parent] if parent is not None else mp_c[f, c]
                if np.any(np.isnan(other)): continue
                bd = (other - mp) if parent is None else (mp - other)
                L = float(np.linalg.norm(bd))
                if L < 1e-6: continue
                bx, by = bd[0] / L, bd[1] / L; px, py = by, -bx
                tgt_x = float(hrp[0]) - (along_med[j] * bx + perp_med[j] * px)
                tgt_y = float(hrp[1]) - (along_med[j] * by + perp_med[j] * py)
                if mismatch_threshold_px > 0:
                    dx = tgt_x - float(mp[0]); dy = tgt_y - float(mp[1])
                    if (dx*dx + dy*dy) <= mismatch_threshold_px ** 2: continue
                out[f, j, 0] = tgt_x; out[f, j, 1] = tgt_y
        return out, along_med, perp_med, child_of
    mp_L_c, aL, pL_, cL = _snap_one(mp_L_c, hr_L)
    mp_R_c, aR, pR_, cR = _snap_one(mp_R_c, hr_R)
    return mp_L_c, mp_R_c, {
        "along_L": aL, "perp_L": pL_, "child_L": cL,
        "along_R": aR, "perp_R": pR_, "child_R": cR,
    }


def reconstruct_wrist_from_mcps(mp_L: np.ndarray, mp_R: np.ndarray,
                                 calib: dict) -> tuple:
    """Replace the MediaPipe wrist (joint 0) with a geometrically
    reconstructed position from the 5 MCP joints and the trial-wide
    median MCP→wrist bone lengths.

    Rationale: MediaPipe's wrist landmark is frequently unreliable — it
    drifts up the forearm and its Y can be way off even when the MCPs
    are tracked cleanly.  With 5 MCPs and 5 known bone lengths we have
    an overdetermined multilateration problem; the least-squares
    solution is typically much more stable than MP's single label.

    Pipeline per frame:
      1. Triangulate MCPs (joints 1, 5, 9, 13, 17) from mp_L, mp_R.
      2. Solve for wrist 3D minimising Σ(||MCP_i − w|| − med_i)².
      3. Apply temporal smoothing across the whole trajectory.
      4. Project the final wrist 3D to both cameras and overwrite the
         wrist label in mp_L/mp_R.
    """
    from scipy.optimize import minimize

    mp_L_c = mp_L.copy()
    mp_R_c = mp_R.copy()
    N = mp_L.shape[0]
    MCP_IDS = [1, 5, 9, 13, 17]

    # Step 1: triangulate everything (need wrist and MCPs in 3D)
    pts_3d = np.full((N, 21, 3), np.nan, dtype=np.float32)
    for j in [0] + MCP_IDS:
        pts_3d[:, j] = triangulate_points(
            mp_L[:, j], mp_R[:, j], calib).astype(np.float32)

    # Step 2: trial-wide median for each wrist→MCP bone length, using the
    # MP-initial wrist only for bone-length estimation (median is robust).
    medians = np.full(5, np.nan, dtype=np.float32)
    for i, m in enumerate(MCP_IDS):
        d = np.linalg.norm(pts_3d[:, m] - pts_3d[:, 0], axis=1)
        good = ~np.isnan(d)
        if good.sum() >= 5:
            medians[i] = float(np.median(d[good]))
    if np.any(np.isnan(medians)):
        return mp_L_c, mp_R_c, 0  # not enough data to reconstruct

    # Step 3a: PASS 1 — unconstrained per-frame multilateration seeded
    # from the MP wrist (fallback: MCP-centroid-along-forearm ray).  This
    # gives us a noisy initial trajectory to smooth.
    initial = np.full((N, 3), np.nan, dtype=np.float32)
    for f in range(N):
        mcps = [pts_3d[f, m] for m in MCP_IDS]
        if any(np.isnan(p).any() for p in mcps):
            continue
        w0 = pts_3d[f, 0]
        if np.any(np.isnan(w0)):
            centroid = np.mean(mcps, axis=0)
            dir_ = centroid - pts_3d[f, 9]
            n = np.linalg.norm(dir_)
            w0 = centroid + (dir_ / n) * float(medians[2]) if n > 1e-6 else centroid
        def cost(w):
            s = 0.0
            for i, m in enumerate(MCP_IDS):
                s += (np.linalg.norm(mcps[i] - w) - float(medians[i])) ** 2
            return s
        try:
            res = minimize(cost, w0, method='Nelder-Mead',
                           options={'xatol': 0.2, 'fatol': 0.05, 'maxiter': 150})
            initial[f] = (res.x if (res.success or res.fun < cost(w0)) else w0).astype(np.float32)
        except Exception:
            initial[f] = w0

    # Step 3b: PASS 2 — re-optimise each frame with a STRONG temporal
    # anchor from a local quadratic fit of the initial trajectory over a
    # wide window.  This smooths the trajectory hard while still respecting
    # bone lengths.  Large LAMBDA pulls the wrist toward the smooth path;
    # only meaningful bone-length residuals let it deviate.
    LAMBDA = 5.0
    WIN = 15
    def _poly_target(arr, f):
        lo, hi = max(0, f - WIN), min(N, f + WIN + 1)
        idxs = np.arange(lo, hi)
        msk = ~np.isnan(arr[lo:hi, 0]) & (idxs != f)
        if msk.sum() < 3:
            return arr[f]
        out = np.empty(3, dtype=np.float32)
        for k in range(3):
            try:
                c = np.polyfit(idxs[msk], arr[lo:hi, k][msk], 2)
                out[k] = np.polyval(c, f)
            except Exception:
                return arr[f]
        return out

    new_wrist = np.full((N, 3), np.nan, dtype=np.float32)
    for f in range(N):
        mcps = [pts_3d[f, m] for m in MCP_IDS]
        if any(np.isnan(p).any() for p in mcps):
            continue
        target = _poly_target(initial, f)
        if np.any(np.isnan(target)):
            new_wrist[f] = initial[f]
            continue
        def cost2(w):
            s = 0.0
            for i, m in enumerate(MCP_IDS):
                s += (np.linalg.norm(mcps[i] - w) - float(medians[i])) ** 2
            s += LAMBDA * float(np.sum((w - target) ** 2))
            return s
        try:
            res = minimize(cost2, target, method='Nelder-Mead',
                           options={'xatol': 0.2, 'fatol': 0.05, 'maxiter': 150})
            w_new = res.x.astype(np.float32) if (res.success or res.fun < cost2(target)) else target
        except Exception:
            w_new = target
        # Reject excursions: cap movement from the smooth target at 15 mm.
        delta = float(np.linalg.norm(w_new - target))
        if delta > 15.0:
            w_new = target + (w_new - target) * (15.0 / delta)
        new_wrist[f] = w_new.astype(np.float32)

    # Step 4: final light smoothing — wide quadratic window irons out any
    # remaining per-frame jitter.
    smoothed = new_wrist.copy()
    W = 7
    for f in range(N):
        lo, hi = max(0, f - W), min(N, f + W + 1)
        window = new_wrist[lo:hi]
        valid = ~np.isnan(window[:, 0])
        if valid.sum() < 3:
            continue
        idx = np.arange(lo, hi)[valid]
        pts = window[valid]
        for k in range(3):
            try:
                c = np.polyfit(idx, pts[:, k], 2)
                smoothed[f, k] = np.polyval(c, f)
            except Exception:
                pass

    # Step 5: project back to 2D, overwrite wrist labels
    count = 0
    for f in range(N):
        if np.any(np.isnan(smoothed[f])):
            continue
        pL, pR = _project_point_stereo(smoothed[f], calib)
        if pL is None or pR is None:
            continue
        mp_L_c[f, 0, 0] = float(pL[0]); mp_L_c[f, 0, 1] = float(pL[1])
        mp_R_c[f, 0, 0] = float(pR[0]); mp_R_c[f, 0, 1] = float(pR[1])
        count += 1
    return mp_L_c, mp_R_c, count


def correct_bone_length_from_errors(
        mp_L: np.ndarray, mp_R: np.ndarray, calib: dict,
        bone_length_weight: float,
        jump_2d_scores: np.ndarray | None,
        progress_callback=None) -> tuple:
    """Bone-length correction: wrist pass + finger proximal→distal passes.

    Strategy:
    - Compute per-bone robust median and a pooled (1-W) percentile threshold
      across all bone-frame deviations (same semantic as the UI threshold).
    - **Wrist pass**: move joint 0 to minimise Σ(|MCP_i − wrist| − median_i)²
      across all 5 wrist bones (iterate until mean signed deviation is small).
      Process frames in order of decreasing |mean deviation|.
    - **Finger passes**: for each finger, proximal→distal; for each bone,
      fix frames where |length − median| > threshold by moving the distal
      endpoint to a new 3D position that restores median length, favouring
      temporal stability and keeping the low-2D-jump camera's label as close
      to its original as possible.  Leave descendants untouched.
    - **Cleanup pass**: repeat wrist + finger passes once more.
    - Projections of corrected 3D preserve Y-disparity automatically.
      Frames whose new Z would create a large jump/outlier are skipped.
    """
    from scipy.optimize import minimize

    W = float(bone_length_weight or 0.0)
    if W <= 0:
        return mp_L.copy(), mp_R.copy(), 0

    mp_L_c = mp_L.copy()
    mp_R_c = mp_R.copy()
    N, J, _ = mp_L.shape

    def _tri():
        out = np.full((N, J, 3), np.nan, dtype=np.float32)
        for j in range(J):
            out[:, j] = triangulate_points(
                mp_L_c[:, j], mp_R_c[:, j], calib).astype(np.float32)
        return out

    # Camera-stability weights from 2D-jump.  Lower jump → stable camera →
    # preserve its original 2D more strongly.
    def _cam_weights(f, j):
        if jump_2d_scores is None:
            return 1.0, 1.0
        j0 = jump_2d_scores[f, j, 0]
        j1 = jump_2d_scores[f, j, 1]
        j0 = 0.0 if np.isnan(j0) else float(j0)
        j1 = 0.0 if np.isnan(j1) else float(j1)
        # Weight = 1/(jump + eps) — the stable (low-jump) camera gets the
        # bigger penalty for moving its 2D.
        return 1.0 / (j0 + 0.25), 1.0 / (j1 + 0.25)

    # Z jump / outlier tolerances derived from the current trial.  We
    # disallow a correction that would push |Z − local_poly| or
    # |Z − global_median| past these bounds.
    mp_3d = _tri()
    z_global_mad = np.zeros(J)
    z_global_med = np.zeros(J)
    for j in range(J):
        z = mp_3d[:, j, 2]
        valid = ~np.isnan(z)
        if valid.sum() < 3:
            continue
        z_global_med[j] = float(np.median(z[valid]))
        z_global_mad[j] = float(np.median(np.abs(z[valid] - z_global_med[j]))) or 1.0
    Z_JUMP_TOL_K = 6.0    # multiples of local MAD for Z-jump tolerance
    Z_OUT_TOL_K  = 10.0   # multiples of global MAD for Z-outlier tolerance

    def _z_ok(f, j, new_z, z_all):
        # Local poly target from ±10 clean neighbours (non-NaN).
        WIN = 10
        lo, hi = max(0, f - WIN), min(N, f + WIN + 1)
        idxs = np.arange(lo, hi)
        msk = ~np.isnan(z_all[lo:hi]) & (idxs != f)
        if msk.sum() >= 3:
            target = _poly_predict_Z(idxs[msk], z_all[lo:hi][msk], f)
            local_mad = float(np.median(np.abs(z_all[lo:hi][msk] - np.median(z_all[lo:hi][msk])))) or 1.0
            if not np.isnan(target) and abs(new_z - target) > Z_JUMP_TOL_K * local_mad:
                return False
        if abs(new_z - z_global_med[j]) > Z_OUT_TOL_K * z_global_mad[j]:
            return False
        return True

    def _commit_3d(f: int, j: int, new_pt: np.ndarray):
        """Project new 3D to both cameras, overwrite mp_L_c/mp_R_c."""
        pL, pR = _project_point_stereo(new_pt, calib)
        if pL is None or pR is None:
            return False
        mp_L_c[f, j, 0] = float(pL[0]); mp_L_c[f, j, 1] = float(pL[1])
        mp_R_c[f, j, 0] = float(pR[0]); mp_R_c[f, j, 1] = float(pR[1])
        mp_3d[f, j, :] = new_pt
        return True

    def _bone_medians_and_threshold():
        """Per-bone median length (across non-NaN frames); pooled (1-W)
        percentile dev threshold across all bones."""
        medians = {}
        pooled = []
        for (a, b) in HAND_SKELETON:
            dab = mp_3d[:, b] - mp_3d[:, a]
            lens = np.linalg.norm(dab, axis=1)
            vals = lens[~np.isnan(lens)]
            if vals.size < 3:
                continue
            med = float(np.median(vals))
            medians[(a, b)] = med
            pooled.extend(np.abs(vals - med).tolist())
        thr = None
        if pooled:
            pooled = np.array(sorted(pooled))
            idx = min(len(pooled) - 1,
                     max(0, int(len(pooled) * (1.0 - W))))
            thr = float(pooled[idx])
        return medians, thr

    def _wrist_pass():
        nonlocal mp_3d
        count = 0
        medians, _ = _bone_medians_and_threshold()
        # Per-frame mean signed deviation
        frame_mean_dev = np.full(N, 0.0)
        for f in range(N):
            devs = []
            for (a, b) in WRIST_BONES_LIST:
                if (a, b) not in medians:
                    continue
                d = mp_3d[f, b] - mp_3d[f, a]
                L = float(np.linalg.norm(d))
                if np.isnan(L):
                    continue
                devs.append(L - medians[(a, b)])
            frame_mean_dev[f] = float(np.mean(devs)) if devs else 0.0
        order = np.argsort(-np.abs(frame_mean_dev))
        for f in order:
            f = int(f)
            if abs(frame_mean_dev[f]) < 0.5:
                break
            mcps = [mp_3d[f, b] for (_a, b) in WRIST_BONES_LIST]
            if any(np.isnan(m).any() for m in mcps):
                continue
            targets = [medians[(a, b)] for (a, b) in WRIST_BONES_LIST if (a, b) in medians]
            if len(targets) < 3:
                continue
            wrist0 = mp_3d[f, 0].copy()
            if np.any(np.isnan(wrist0)):
                continue
            # Temporal target from ±10 neighbours (non-NaN, ≠ f)
            WIN = 10
            lo, hi = max(0, f - WIN), min(N, f + WIN + 1)
            idxs = np.arange(lo, hi)
            msk = ~np.isnan(mp_3d[lo:hi, 0, 0]) & (idxs != f)
            wrist_tp = (_local_poly_fit_3d(idxs[msk], mp_3d[lo:hi, 0][msk], f)
                        if msk.sum() >= 3 else wrist0)
            if np.any(np.isnan(wrist_tp)):
                wrist_tp = wrist0
            # 2D-stability weights for wrist joint
            wL, wR = _cam_weights(f, 0)
            orig_L = mp_L_c[f, 0].astype(np.float64)
            orig_R = mp_R_c[f, 0].astype(np.float64)

            def cost(w):
                bone = 0.0
                for (_a, b), tgt in zip(WRIST_BONES_LIST, targets):
                    bone += (np.linalg.norm(mp_3d[f, b] - w) - tgt) ** 2
                # Strong temporal-smoothness anchor: keep wrist close to its
                # polynomial prediction from surrounding frames.
                temp = float(np.sum((w - wrist_tp) ** 2))
                pL, pR = _project_point_stereo(np.asarray(w), calib)
                c2d = 0.0
                if pL is not None and pR is not None:
                    c2d = wL * float(np.sum((pL - orig_L) ** 2)) + \
                          wR * float(np.sum((pR - orig_R) ** 2))
                return bone + 1.5 * temp + 0.02 * c2d

            try:
                res = minimize(cost, wrist_tp, method='Nelder-Mead',
                               options={'xatol': 0.5, 'fatol': 0.1, 'maxiter': 150})
                new_w = res.x if res.success or res.fun < cost(wrist0) else None
            except Exception:
                new_w = None
            if new_w is None:
                continue
            if not _z_ok(f, 0, float(new_w[2]), mp_3d[:, 0, 2]):
                continue
            # Reject if we'd move the wrist a huge distance from its temporal
            # prediction (guard against degenerate optimisations).
            if float(np.linalg.norm(new_w - wrist_tp)) > 25.0:
                continue
            if _commit_3d(f, 0, np.asarray(new_w, dtype=np.float32)):
                count += 1
        return count

    def _finger_bone_pass(a: int, b: int):
        """Correct bone (a,b) by moving the distal endpoint `b`."""
        nonlocal mp_3d
        count = 0
        medians, thr = _bone_medians_and_threshold()
        if (a, b) not in medians or thr is None:
            return 0
        tgt_len = medians[(a, b)]
        lens = np.linalg.norm(mp_3d[:, b] - mp_3d[:, a], axis=1)
        devs = np.abs(lens - tgt_len)
        flagged = np.where((~np.isnan(lens)) & (devs > thr))[0]
        if len(flagged) == 0:
            return 0
        order = flagged[np.argsort(-devs[flagged])]
        for f in order:
            f = int(f)
            prox = mp_3d[f, a].copy()
            dist0 = mp_3d[f, b].copy()
            if np.any(np.isnan(prox)) or np.any(np.isnan(dist0)):
                continue
            # Temporal poly target for distal joint
            WIN = 10
            lo, hi = max(0, f - WIN), min(N, f + WIN + 1)
            idxs = np.arange(lo, hi)
            msk = ~np.isnan(mp_3d[lo:hi, b, 0]) & (idxs != f)
            dist_tp = (_local_poly_fit_3d(idxs[msk], mp_3d[lo:hi, b][msk], f)
                       if msk.sum() >= 3 else dist0)
            if np.any(np.isnan(dist_tp)):
                dist_tp = dist0
            wL, wR = _cam_weights(f, b)
            orig_L = mp_L_c[f, b].astype(np.float64)
            orig_R = mp_R_c[f, b].astype(np.float64)

            def cost(d):
                bone = (np.linalg.norm(d - prox) - tgt_len) ** 2
                temp = float(np.sum((d - dist_tp) ** 2))
                pL, pR = _project_point_stereo(np.asarray(d), calib)
                c2d = 0.0
                if pL is not None and pR is not None:
                    c2d = wL * float(np.sum((pL - orig_L) ** 2)) + \
                          wR * float(np.sum((pR - orig_R) ** 2))
                return 50.0 * bone + 0.01 * temp + 0.02 * c2d

            try:
                res = minimize(cost, dist0, method='Nelder-Mead',
                               options={'xatol': 0.5, 'fatol': 0.05, 'maxiter': 150})
                new_d = res.x if (res.success or res.fun < cost(dist0)) else None
            except Exception:
                new_d = None
            if new_d is None:
                continue
            if not _z_ok(f, b, float(new_d[2]), mp_3d[:, b, 2]):
                continue
            # New bone length must actually be closer to median than before
            new_L = float(np.linalg.norm(new_d - prox))
            if abs(new_L - tgt_len) >= abs(lens[f] - tgt_len):
                continue
            if _commit_3d(f, b, np.asarray(new_d, dtype=np.float32)):
                count += 1
        return count

    total = 0
    # Progress: 2 passes × (wrist + 5 fingers) = 12 checkpoints; report
    # fractional completion on [0, 100] so the caller can scale it.
    n_ck = 2 * 6
    i_ck = 0
    def _tick():
        nonlocal i_ck
        i_ck += 1
        if progress_callback:
            try: progress_callback(int(100 * i_ck / n_ck))
            except Exception: pass
    # Wrist is now reconstructed from MCPs in a dedicated stage-0 pass of
    # the pipeline (see reconstruct_wrist_from_mcps); skip re-optimising it
    # here so this correction only addresses finger bones.
    for _pass in range(2):  # main + cleanup
        _tick()
        for finger in FINGER_BONES_LIST:
            for (a, b) in finger:  # proximal → distal
                total += _finger_bone_pass(a, b)
            _tick()
    return mp_L_c, mp_R_c, total


def correct_bone_agreement_from_errors(
        mp_L: np.ndarray, mp_R: np.ndarray, calib: dict,
        bone_agreement_weight: float,
        jump_2d_scores: np.ndarray | None,
        progress_callback=None) -> tuple:
    """Bone-length-jump smoothing: per finger bone, fit a local quadratic
    polynomial to the bone's length across time; for frames whose
    |length − poly(f)| exceeds the pooled (1-W) percentile threshold
    (same slider semantics as bone_agreement), move the distal endpoint
    to restore the polynomial-predicted length.  Leaves descendants
    untouched (they'll re-settle on the next pass).

    Uses the same constraint suite as `correct_bone_length_from_errors`
    (no new Y-disparity, Z-jump, or Z-outlier), same camera-stability
    weighting, and the same iterative-ordering-by-error-magnitude.
    """
    from scipy.optimize import minimize

    W = float(bone_agreement_weight or 0.0)
    if W <= 0:
        return mp_L.copy(), mp_R.copy(), 0

    mp_L_c = mp_L.copy(); mp_R_c = mp_R.copy()
    N, J, _ = mp_L.shape

    # Triangulate current 3D
    mp_3d = np.full((N, J, 3), np.nan, dtype=np.float32)
    for j in range(J):
        mp_3d[:, j] = triangulate_points(
            mp_L_c[:, j], mp_R_c[:, j], calib).astype(np.float32)

    # Per-joint Z global statistics for Z-outlier guard
    z_global_med = np.zeros(J); z_global_mad = np.zeros(J)
    for j in range(J):
        z = mp_3d[:, j, 2]; valid = ~np.isnan(z)
        if valid.sum() < 3: continue
        z_global_med[j] = float(np.median(z[valid]))
        z_global_mad[j] = float(np.median(np.abs(z[valid] - z_global_med[j]))) or 1.0
    Z_JUMP_TOL_K = 6.0; Z_OUT_TOL_K = 10.0

    def _z_ok(f, j, new_z, z_all):
        WIN = 10
        lo, hi = max(0, f - WIN), min(N, f + WIN + 1)
        idxs = np.arange(lo, hi)
        msk = ~np.isnan(z_all[lo:hi]) & (idxs != f)
        if msk.sum() >= 3:
            target = _poly_predict_Z(idxs[msk], z_all[lo:hi][msk], f)
            local_mad = float(np.median(np.abs(z_all[lo:hi][msk] - np.median(z_all[lo:hi][msk])))) or 1.0
            if not np.isnan(target) and abs(new_z - target) > Z_JUMP_TOL_K * local_mad:
                return False
        if abs(new_z - z_global_med[j]) > Z_OUT_TOL_K * z_global_mad[j]:
            return False
        return True

    def _cam_weights(f, j):
        if jump_2d_scores is None: return 1.0, 1.0
        j0 = jump_2d_scores[f, j, 0]; j1 = jump_2d_scores[f, j, 1]
        j0 = 0.0 if np.isnan(j0) else float(j0)
        j1 = 0.0 if np.isnan(j1) else float(j1)
        return 1.0 / (j0 + 0.25), 1.0 / (j1 + 0.25)

    def _commit_3d(f, j, new_pt):
        pL, pR = _project_point_stereo(new_pt, calib)
        if pL is None or pR is None: return False
        mp_L_c[f, j, 0] = float(pL[0]); mp_L_c[f, j, 1] = float(pL[1])
        mp_R_c[f, j, 0] = float(pR[0]); mp_R_c[f, j, 1] = float(pR[1])
        mp_3d[f, j, :] = new_pt
        return True

    # Threshold: pool |length − poly(f)| residuals across all finger
    # bones and frames, take the (1−W) quantile (matches the UI
    # "bone_agreement" slider semantic).
    FINGER_BONES = [bone for finger in FINGER_BONES_LIST for bone in finger]
    def _fit_poly_len(series, WIN=12):
        """Local-quadratic smoothed length per frame.  NaN-aware."""
        out = np.full(len(series), np.nan, dtype=np.float32)
        for f in range(len(series)):
            lo, hi = max(0, f - WIN), min(len(series), f + WIN + 1)
            idxs = np.arange(lo, hi)
            msk = ~np.isnan(series[lo:hi])
            if msk.sum() < 3: continue
            try:
                c = np.polyfit(idxs[msk], series[lo:hi][msk], 2)
                out[f] = float(np.polyval(c, f))
            except Exception:
                pass
        return out

    pooled = []
    bone_info = []  # list of (a, b, lengths, poly)
    for (a, b) in FINGER_BONES:
        d = mp_3d[:, b] - mp_3d[:, a]
        lens = np.linalg.norm(d, axis=1)
        poly = _fit_poly_len(lens)
        bone_info.append((a, b, lens, poly))
        for v in np.abs(lens - poly):
            if np.isfinite(v): pooled.append(float(v))
    if not pooled:
        return mp_L_c, mp_R_c, 0
    pooled.sort()
    idx = min(len(pooled) - 1, max(0, int(len(pooled) * (1.0 - W))))
    thr = pooled[idx]
    if thr <= 0:
        return mp_L_c, mp_R_c, 0

    count = 0
    for pass_i, (a, b, _lens, _poly) in enumerate(bone_info):
        # Refit using the (possibly-adjusted) current 3D positions so
        # descendants of earlier-corrected joints see a smoothed trajectory.
        d = mp_3d[:, b] - mp_3d[:, a]
        lens = np.linalg.norm(d, axis=1)
        poly = _fit_poly_len(lens)
        dev = np.abs(lens - poly)
        flagged = np.where(np.isfinite(dev) & (dev > thr))[0]
        if len(flagged) == 0:
            if progress_callback:
                try: progress_callback(int(100 * (pass_i + 1) / len(bone_info)))
                except Exception: pass
            continue
        order = flagged[np.argsort(-dev[flagged])]
        for f in order:
            f = int(f)
            prox = mp_3d[f, a]; dist0 = mp_3d[f, b]
            if np.any(np.isnan(prox)) or np.any(np.isnan(dist0)): continue
            tgt_len = float(poly[f])
            if not np.isfinite(tgt_len) or tgt_len <= 0: continue
            # Temporal poly target for the distal joint itself (3D)
            WIN = 10
            lo, hi = max(0, f - WIN), min(N, f + WIN + 1)
            idxs = np.arange(lo, hi)
            msk = ~np.isnan(mp_3d[lo:hi, b, 0]) & (idxs != f)
            dist_tp = (_local_poly_fit_3d(idxs[msk], mp_3d[lo:hi, b][msk], f)
                       if msk.sum() >= 3 else dist0)
            if np.any(np.isnan(dist_tp)): dist_tp = dist0
            wL, wR = _cam_weights(f, b)
            orig_L = mp_L_c[f, b].astype(np.float64); orig_R = mp_R_c[f, b].astype(np.float64)
            def cost(d):
                bone = (np.linalg.norm(d - prox) - tgt_len) ** 2
                temp = float(np.sum((d - dist_tp) ** 2))
                pL, pR = _project_point_stereo(np.asarray(d), calib)
                c2d = 0.0
                if pL is not None and pR is not None:
                    c2d = wL * float(np.sum((pL - orig_L) ** 2)) + \
                          wR * float(np.sum((pR - orig_R) ** 2))
                return 50.0 * bone + 0.01 * temp + 0.02 * c2d
            try:
                res = minimize(cost, dist0, method='Nelder-Mead',
                               options={'xatol': 0.5, 'fatol': 0.05, 'maxiter': 150})
                new_d = res.x if (res.success or res.fun < cost(dist0)) else None
            except Exception:
                new_d = None
            if new_d is None: continue
            if not _z_ok(f, b, float(new_d[2]), mp_3d[:, b, 2]): continue
            new_L = float(np.linalg.norm(new_d - prox))
            if abs(new_L - tgt_len) >= abs(lens[f] - tgt_len): continue
            if _commit_3d(f, b, np.asarray(new_d, dtype=np.float32)):
                count += 1
        if progress_callback:
            try: progress_callback(int(100 * (pass_i + 1) / len(bone_info)))
            except Exception: pass
    return mp_L_c, mp_R_c, count


def _errors_for_factor(detection: dict, attribution: dict,
                        det_weights: dict, attr_weights: dict,
                        factor: str,
                        winner_take_all: bool = False) -> np.ndarray | None:
    """Run apply_thresholds with only the named detection factor active.

    Lets the correction pipeline pull out a clean per-factor error matrix
    so each pass only targets its own flag set.

    ``winner_take_all`` (default False): when True, a single bad camera
    is always picked even with no user-weighted attribution sliders
    (falls back to ``jump_2d`` / ``confidence``).  HRnet Correct uses
    this so the overlay shows one-sided blame out of the box.
    """
    w = float(det_weights.get(factor, 0.0))
    if w <= 0 or detection.get(factor) is None:
        return None
    single = {factor: w}
    return apply_thresholds(detection, attribution, single, attr_weights,
                             winner_take_all=winner_take_all)


class _CorrectionsCancelled(Exception):
    """Raised internally when a long-running pipeline / prefill should
    abort because its caller signalled the cancel event.  The router
    catches this to mark the job ``cancelled`` rather than ``failed``."""


def run_correction_pipeline(subject_name: str, trial_stem: str,
                             det_weights: dict, attr_weights: dict,
                             progress_callback=None,
                             cancel_event=None,
                             hrnet_source: str = "auto",
                             stereo_mode: str = "image",
                             stereo_mask_dilate_px: int = 10,
                             stereo_gauss_center_weight: float = 0.0,
                             stereo_conf: float = 0.0,
                             stereo_dist_px: float = 0.0,
                             stereo_occlusion_px: float = 0.0) -> dict:
    """Run the full correction pipeline, save result as ``mano_fit_v2.npz``.

    Step 0: Stereo-correction.  Runs ``run_stereo_align`` with the chosen
    mode + params, then for every (frame, joint) whose stereo confidence
    >= ``stereo_conf`` and whose MP↔stereo pixel distance > ``stereo_dist_px``,
    attributes the bad camera via the existing attribution score and
    replaces that camera's MP label with its stereo label.  Skipped when
    ``stereo_dist_px == 0``.

    Stage 1: Combined Y-disparity + Z-outlier correction.

    Returns ``{n_corrected, out_path, n_frames}``.
    """
    from .mediapipe_prelabel import load_mediapipe_prelabels
    from .mano_data import _load_trial_calibration, load_angle_priors
    from .video import build_trial_map
    import json as _json
    from datetime import datetime

    trials = build_trial_map(subject_name)
    trial = next((t for t in trials if t["trial_name"] == trial_stem), None)
    if trial is None:
        raise ValueError(f"Trial {trial_stem} not found for {subject_name}")
    N = trial["frame_count"]
    start = trial.get("start_frame", 0)

    mp_L, mp_R, conf_L, conf_R = _load_mp_labels(subject_name, trial_stem, start, N)

    calib = _load_trial_calibration(subject_name, trial_stem)
    if calib is None:
        raise ValueError(f"No stereo calibration for {subject_name}")
    priors = load_angle_priors().get("joints", [])

    def _cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    def _pg(pct):
        if _cancelled():
            raise _CorrectionsCancelled()
        if progress_callback:
            try: progress_callback(int(pct))
            except Exception: pass
    _pg(1)

    # Compute initial scores on original MP
    cache_key = (subject_name, trial_stem, N, start)
    det, attr = _get_scores(subject_name, trial_stem, mp_L, mp_R, conf_L, conf_R,
                              calib, priors, cache_key=cache_key)
    _pg(5)

    mp_L_c, mp_R_c = mp_L.copy(), mp_R.copy()
    n_corrected = 0

    # ── Step 0: Stereo-correction ─────────────────────────────────────
    # Run the stand-alone stereo-alignment first (saved as
    # stereo_align_<mode>.npz alongside the trial), then for every
    # joint whose stereo confidence clears the threshold AND whose
    # MP↔stereo distance exceeds the user's distance threshold, decide
    # which camera to blame using the existing per-camera attribution
    # score and replace that camera's MP label with its stereo label.
    # The downstream Y/Z passes then start from the stereo-corrected
    # MP labels rather than the raw ones.
    stereo_blame = np.zeros((N, 21, 2), dtype=bool)
    stereo_L_pts = np.full((N, 21, 2), np.nan, dtype=np.float32)
    stereo_R_pts = np.full((N, 21, 2), np.nan, dtype=np.float32)
    stereo_response = np.full((N, 21), np.nan, dtype=np.float32)
    stereo_ran = False  # False when Stereo-distance is 0 OR the align
                        # bake failed -- triggers the after_sc snapshot
                        # below to be left all-NaN so the data layer's
                        # has_skel_v2_sc flag is False and the stage row
                        # is hidden in the UI.
    if float(stereo_dist_px) > 0:
        try:
            from .stereo_align import run_stereo_align, load_stereo_align
            # Resolve trial_idx from trial_stem (run_stereo_align needs it).
            t_idx = next((i for i, t in enumerate(trials)
                          if t["trial_name"] == trial_stem), None)
            if t_idx is None:
                raise RuntimeError(f"Couldn't resolve trial_idx for {trial_stem}")
            run_stereo_align(
                subject_name, t_idx,
                progress_callback=lambda p: _pg(int(p * 0.10)),
                cancel_event=cancel_event,
                mode=str(stereo_mode),
                mask_dilate_px=int(stereo_mask_dilate_px),
                gauss_center_weight=float(stereo_gauss_center_weight))
            sa = load_stereo_align(subject_name, t_idx, mode=str(stereo_mode))
            if sa is not None:
                stereo_ran = True
                shifts = sa["shifts"]                # (N_sa, 21, 2)
                resp   = sa["response"]              # (N_sa, 21)
                n_sa   = min(N, int(shifts.shape[0]))
                # Per-camera stereo positions.
                stereo_L_pts[:n_sa] = mp_L[:n_sa] - shifts[:n_sa]
                stereo_R_pts[:n_sa] = mp_R[:n_sa] + shifts[:n_sa]
                stereo_response[:n_sa] = resp[:n_sa]
                # Per-joint MP↔Stereo distance == ||shifts|| (symmetric
                # by construction).
                dist = np.linalg.norm(shifts[:n_sa], axis=-1)  # (n_sa, 21)
                conf_ok = resp[:n_sa] >= float(stereo_conf)
                err_mask = conf_ok & (dist > float(stereo_dist_px))  # (n_sa, 21)
                # Per-camera attribution: use existing attr factors.
                attr_per_cam = _combined_attr_per_cam(attr, attr_weights)
                # Build per-(f, j) blame: True on the blamed camera.
                for fi in range(n_sa):
                    for j in range(21):
                        if not err_mask[fi, j]:
                            continue
                        sL, sR = stereo_L_pts[fi, j], stereo_R_pts[fi, j]
                        if np.any(np.isnan(sL)) or np.any(np.isnan(sR)):
                            continue
                        if attr_per_cam is not None:
                            a0 = float(attr_per_cam[fi, j, 0])
                            a1 = float(attr_per_cam[fi, j, 1])
                            blame_left = a0 >= a1
                        else:
                            # No attribution weights -- fall back to
                            # MP confidence (lower confidence = bad cam).
                            cL = float(conf_L[fi, j]) if conf_L is not None else 0.0
                            cR = float(conf_R[fi, j]) if conf_R is not None else 0.0
                            blame_left = (cL <= cR)
                        if blame_left:
                            mp_L_c[fi, j] = sL
                            stereo_blame[fi, j, 0] = True
                        else:
                            mp_R_c[fi, j] = sR
                            stereo_blame[fi, j, 1] = True
                        n_corrected += 1
        except Exception as e:
            logger.warning(f"Stereo-correct step skipped (non-fatal): {e}")

    # ── Step 0b: Occlusion-revert ─────────────────────────────────────
    # The stereo per-joint phase corr will sometimes lock onto a
    # nearby joint that's CLOSER to the camera (e.g. an MCP that
    # disappears behind a PIP/DIP).  The "corrected" 2D label snaps
    # onto the occluder, the triangulated Z drops toward the
    # occluder's Z, and the joint visually pops forward.  Detect this
    # and revert: for every stereo-blamed joint, if there's any OTHER
    # joint K within ``stereo_occlusion_px`` (2D) in either camera
    # that has a smaller Z than this joint's raw-MP Z, AND the
    # stereo-corrected Z is closer to K's Z than the raw-MP Z was,
    # roll the label back to MP.
    if float(stereo_occlusion_px) > 0 and n_corrected > 0:
        try:
            mp_3d_pre = np.full((N, 21, 3), np.nan, dtype=np.float32)
            mp_3d_sc  = np.full((N, 21, 3), np.nan, dtype=np.float32)
            for j in range(21):
                mp_3d_pre[:, j, :] = triangulate_points(
                    mp_L[:, j], mp_R[:, j], calib).astype(np.float32)
                mp_3d_sc[:, j, :] = triangulate_points(
                    mp_L_c[:, j], mp_R_c[:, j], calib).astype(np.float32)
            occ_r2 = float(stereo_occlusion_px) ** 2
            n_reverted = 0
            for fi in range(N):
                for j in range(21):
                    if not stereo_blame[fi, j].any():
                        continue
                    z_mp = float(mp_3d_pre[fi, j, 2])
                    z_sc = float(mp_3d_sc[fi, j, 2])
                    if not (np.isfinite(z_mp) and np.isfinite(z_sc)):
                        continue
                    # Look for any overlying joint K: 2D-close in
                    # either camera AND closer-to-camera Z.
                    pL = mp_L[fi, j]; pR = mp_R[fi, j]
                    for k in range(21):
                        if k == j:
                            continue
                        z_k = float(mp_3d_pre[fi, k, 2])
                        if not np.isfinite(z_k) or z_k >= z_mp:
                            continue
                        qL = mp_L[fi, k]; qR = mp_R[fi, k]
                        dL2 = (float(pL[0]) - float(qL[0])) ** 2 + \
                              (float(pL[1]) - float(qL[1])) ** 2 \
                              if np.isfinite(pL[0]) and np.isfinite(qL[0]) else float('inf')
                        dR2 = (float(pR[0]) - float(qR[0])) ** 2 + \
                              (float(pR[1]) - float(qR[1])) ** 2 \
                              if np.isfinite(pR[0]) and np.isfinite(qR[0]) else float('inf')
                        if dL2 > occ_r2 and dR2 > occ_r2:
                            continue
                        # SC pulled Z toward K's Z?
                        if abs(z_sc - z_k) < abs(z_mp - z_k):
                            mp_L_c[fi, j] = mp_L[fi, j]
                            mp_R_c[fi, j] = mp_R[fi, j]
                            stereo_blame[fi, j, 0] = False
                            stereo_blame[fi, j, 1] = False
                            n_reverted += 1
                            break
            if n_reverted > 0:
                logger.info(f"Stereo-correct occlusion-revert: {n_reverted} joints rolled back to MP")
                n_corrected = max(0, n_corrected - n_reverted)
        except Exception as e:
            logger.warning(f"Stereo-correct occlusion-revert skipped (non-fatal): {e}")
    _pg(10)

    # Snapshot after stereo-correction (stereo_correct stage view).
    # When stereo_dist_px == 0 the whole step is skipped; leave the
    # snapshot all-NaN (mirrors the HRnet-snap behaviour) so the data
    # layer's has_skel_v2_sc flag is False and the stage row is hidden
    # in the UI -- nothing was actually corrected so there's no view
    # to show.
    if stereo_ran:
        mp_L_after_sc = mp_L_c.copy()
        mp_R_after_sc = mp_R_c.copy()
    else:
        mp_L_after_sc = np.full_like(mp_L_c, np.nan)
        mp_R_after_sc = np.full_like(mp_R_c, np.nan)

    # ── Stage 1: Combined Y-disparity + Z-outlier correction ──────────
    # Errors driving the correction are the union of y_disp and z_outlier
    # flags.  The corrector picks the bad camera per-frame, interpolates Z
    # from frames clean of BOTH factors, lifts the good camera's pixel onto
    # the world-Z target plane, and projects back into the bad camera.
    y_errors = _errors_for_factor(det, attr, det_weights, attr_weights, "y_disp")
    z_errors = _errors_for_factor(det, attr, det_weights, attr_weights, "z_outlier")
    # Stereo-correct fallback: for joints that step-0 stereo-corrected,
    # first try substituting the raw MP label.  If raw MP isn't flagged
    # by y_disp / z_outlier, revert to MP (and clear the stereo_blame
    # flag) -- the SC adjustment wasn't necessary and shouldn't be
    # carried into Y/Z.  Joints where MP is also flagged keep their SC
    # label and let the usual Y/Z procedure act on them.
    if np.any(stereo_blame) and (y_errors is not None or z_errors is not None):
        n_yz_revert = 0
        for fi in range(N):
            for j in range(21):
                if not stereo_blame[fi, j].any():
                    continue
                yf = (y_errors is not None) and bool(y_errors[fi, j].any())
                zf = (z_errors is not None) and bool(z_errors[fi, j].any())
                if yf or zf:
                    continue  # MP is also an error -- keep SC
                mp_L_c[fi, j] = mp_L[fi, j]
                mp_R_c[fi, j] = mp_R[fi, j]
                stereo_blame[fi, j, 0] = False
                stereo_blame[fi, j, 1] = False
                n_yz_revert += 1
        if n_yz_revert > 0:
            logger.info(f"Y/Z-stage SC-fallback: {n_yz_revert} joints reverted to MP")
            n_corrected = max(0, n_corrected - n_yz_revert)
    if y_errors is not None or z_errors is not None:
        attr_per_cam = _combined_attr_per_cam(attr, attr_weights)
        mp_L_c, mp_R_c, n = correct_yz_from_errors(
            mp_L_c, mp_R_c, y_errors, z_errors, calib,
            jump_2d_scores=attr.get("jump_2d"),
            attr_per_cam=attr_per_cam,
            z_outlier_slider=float(det_weights.get("z_outlier", 0.0)))
        n_corrected += n

    # Snapshot after combined Y+Z correction (z_correct stage view).
    # Y-correct stage no longer exists; the legacy "after_y" snapshot is
    # written equal to the combined output for backwards compatibility.
    mp_L_after_y = mp_L_c.copy()
    mp_R_after_y = mp_R_c.copy()
    mp_L_after_z = mp_L_c.copy()
    mp_R_after_z = mp_R_c.copy()
    _pg(35)

    # ── Stage 3: Z-jump correction (same algorithm, z_jump-flagged frames) ──
    zj_det, zj_attr = compute_scores(mp_L_c, mp_R_c, conf_L, conf_R, calib, priors, subject_name=subject_name, trial_stem=trial_stem)
    zj_errors = _errors_for_factor(zj_det, zj_attr, det_weights, attr_weights, "z_jump")
    # Stereo-correct fallback at Z-smooth: same passthrough logic as
    # Y/Z above, gated on the z_jump factor of the RAW MP labels.  Any
    # remaining stereo-blamed joint whose raw-MP version isn't z_jump-
    # flagged is reverted to MP (and stereo_blame cleared) so Z-smooth
    # doesn't operate on an SC label that wasn't even needed.
    zj_errors_raw = _errors_for_factor(det, attr, det_weights, attr_weights, "z_jump")
    if np.any(stereo_blame) and zj_errors_raw is not None:
        n_zs_revert = 0
        for fi in range(N):
            for j in range(21):
                if not stereo_blame[fi, j].any():
                    continue
                if bool(zj_errors_raw[fi, j].any()):
                    continue  # MP is also a z_jump error -- keep SC
                mp_L_c[fi, j] = mp_L[fi, j]
                mp_R_c[fi, j] = mp_R[fi, j]
                stereo_blame[fi, j, 0] = False
                stereo_blame[fi, j, 1] = False
                n_zs_revert += 1
        if n_zs_revert > 0:
            logger.info(f"Z-smooth-stage SC-fallback: {n_zs_revert} joints reverted to MP")
            n_corrected = max(0, n_corrected - n_zs_revert)
    if zj_errors is not None:
        mp_L_c, mp_R_c, n = correct_z_from_errors(
            mp_L_c, mp_R_c, zj_errors, calib, zj_attr.get("jump_2d"))
        n_corrected += n

    # Snapshot after Y + Z-outlier + Z-jump (z_smooth stage view) — this
    # is the last view that still shows the MediaPipe wrist labels.
    mp_L_after_zs = mp_L_c.copy()
    mp_R_after_zs = mp_R_c.copy()
    _pg(50)

    # ── HRnet snap: replace MP labels with per-joint systematic-offset-
    # corrected HRnet peaks, then re-run the Y / Z-outlier / Z-jump passes
    # so any epipolar / outlier / jump introduced by the snap gets cleaned
    # up in place.
    #
    # When the user hasn't set the HRnet-mismatch threshold or no peak
    # assignments exist for this trial, leave the after_hr snapshot all
    # NaN -- the data layer's ``has_skel_v2_hr`` flag then evaluates to
    # False and the UI stops showing the HRnet-snap stage as identical
    # to Z-smooth (which is what you saw when the step didn't actually
    # fire).
    mp_L_after_hr = np.full_like(mp_L_c, np.nan)
    mp_R_after_hr = np.full_like(mp_R_c, np.nan)
    hrnet_offsets = None
    hr_threshold_px = float(det_weights.get("hrnet_mismatch", 0.0))
    # Only run HRnet snap when the user has set a non-zero threshold AND
    # HRnet peaks are actually available.  Threshold = 0 means "don't
    # snap anything" — so v3 runs without HRnet being required at all.
    try:
        hr_L, hr_R = (None, None)
        if hr_threshold_px > 0:
            hr_L, hr_R = _load_hrnet_peaks_arr(subject_name, trial_stem, start, N,
                                                source=hrnet_source)
        if hr_threshold_px > 0 and (hr_L is not None or hr_R is not None):
            mp_L_c, mp_R_c, hrnet_offsets = hrnet_snap_labels(
                mp_L_c, mp_R_c, hr_L, hr_R,
                mismatch_threshold_px=hr_threshold_px,
                calib=calib)
            # Mask out the cells the snap touched: their stereo-consistent
            # post-snap pixels are exactly where we want them (right on
            # the predicted-HR-peak − offset target), and re-running
            # Y/Z-correct on them would just walk them back toward the
            # empirical baseline and undo the snap.  The re-clean still
            # runs on UNSNAPPED cells so any residual y_disp / z_outlier
            # introduced by the snap's neighbours gets handled.
            snap_mask = (hrnet_offsets or {}).get("snap_mask")  # (N, 21) bool

            def _zero_snapped(err):
                """Set ``err[f, j, :]`` to False wherever ``snap_mask[f, j]`` is True."""
                if err is None or snap_mask is None:
                    return err
                err = err.copy()
                err[snap_mask] = False
                return err

            yz_det2, yz_attr2 = compute_scores(mp_L_c, mp_R_c, conf_L, conf_R, calib, priors, subject_name=subject_name, trial_stem=trial_stem)
            y_err2 = _zero_snapped(_errors_for_factor(
                yz_det2, yz_attr2, det_weights, attr_weights, "y_disp"))
            zo_err2 = _zero_snapped(_errors_for_factor(
                yz_det2, yz_attr2, det_weights, attr_weights, "z_outlier"))
            if y_err2 is not None or zo_err2 is not None:
                mp_L_c, mp_R_c, _ = correct_yz_from_errors(
                    mp_L_c, mp_R_c, y_err2, zo_err2, calib,
                    jump_2d_scores=yz_attr2.get("jump_2d"),
                    attr_per_cam=_combined_attr_per_cam(yz_attr2, attr_weights),
                    z_outlier_slider=float(det_weights.get("z_outlier", 0.0)))
            zj_det2, zj_attr2 = compute_scores(mp_L_c, mp_R_c, conf_L, conf_R, calib, priors, subject_name=subject_name, trial_stem=trial_stem)
            zj_err2 = _zero_snapped(_errors_for_factor(
                zj_det2, zj_attr2, det_weights, attr_weights, "z_jump"))
            if zj_err2 is not None:
                mp_L_c, mp_R_c, _ = correct_z_from_errors(
                    mp_L_c, mp_R_c, zj_err2, calib, zj_attr2.get("jump_2d"))
            mp_L_after_hr = mp_L_c.copy(); mp_R_after_hr = mp_R_c.copy()
    except Exception as e:
        logger.warning(f"HRnet snap failed (non-fatal): {e}")
    _pg(55)

    # ── Wrist reconstruction from MCPs ───────────────────────────────
    # Tied to the bone_length slider — wrist reconstruction is part of
    # the bone-length pass's geometry assumption (median bone lengths
    # per joint).  Skip when bone_length is 0 so v3 with a single
    # non-zero slider does only that one step.
    if float(det_weights.get("bone_length", 0.0)) > 0:
        try:
            mp_L_c, mp_R_c, _ = reconstruct_wrist_from_mcps(mp_L_c, mp_R_c, calib)
        except Exception:
            pass

    # ── Stage 4: Bone-length correction (finger bones only; wrist
    # already reconstructed above) ──
    bl_det, bl_attr = compute_scores(mp_L_c, mp_R_c, conf_L, conf_R, calib, priors, subject_name=subject_name, trial_stem=trial_stem)
    bl_w = float(det_weights.get("bone_length", 0.0))
    if bl_w > 0:
        mp_L_c, mp_R_c, n = correct_bone_length_from_errors(
            mp_L_c, mp_R_c, calib, bl_w, bl_attr.get("jump_2d"),
            progress_callback=lambda p: _pg(50 + int(p * 0.3)))
        n_corrected += n

    # Snapshot after bone-length correction (bone_correct stage view)
    mp_L_after_bc = mp_L_c.copy()
    mp_R_after_bc = mp_R_c.copy()
    _pg(85)

    # ── Stage 5: Bone-length-jump smoothing (bone_smooth stage view).
    # Re-score attributions on the post-BL labels and fire the dedicated
    # corrector if the bone_agreement slider is non-zero.
    ba_det, ba_attr = compute_scores(mp_L_c, mp_R_c, conf_L, conf_R, calib, priors, subject_name=subject_name, trial_stem=trial_stem)
    ba_w = float(det_weights.get("bone_agreement", 0.0))
    if ba_w > 0:
        mp_L_c, mp_R_c, n = correct_bone_agreement_from_errors(
            mp_L_c, mp_R_c, calib, ba_w, ba_attr.get("jump_2d"),
            progress_callback=lambda p: _pg(85 + int(p * 0.07)))
        n_corrected += n
    _pg(92)

    # Triangulate each pipeline checkpoint
    joints_3d_after_sc = np.full((N, 21, 3), np.nan, dtype=np.float32)
    joints_3d_after_y = np.full((N, 21, 3), np.nan, dtype=np.float32)
    joints_3d_after_z = np.full((N, 21, 3), np.nan, dtype=np.float32)
    joints_3d_after_zs = np.full((N, 21, 3), np.nan, dtype=np.float32)
    joints_3d_after_hr = np.full((N, 21, 3), np.nan, dtype=np.float32)
    joints_3d_after_bc = np.full((N, 21, 3), np.nan, dtype=np.float32)
    joints_3d = np.full((N, 21, 3), np.nan, dtype=np.float32)
    for j in range(21):
        joints_3d_after_sc[:, j, :] = triangulate_points(
            mp_L_after_sc[:, j], mp_R_after_sc[:, j], calib).astype(np.float32)
        joints_3d_after_y[:, j, :] = triangulate_points(
            mp_L_after_y[:, j], mp_R_after_y[:, j], calib).astype(np.float32)
        joints_3d_after_z[:, j, :] = triangulate_points(
            mp_L_after_z[:, j], mp_R_after_z[:, j], calib).astype(np.float32)
        joints_3d_after_zs[:, j, :] = triangulate_points(
            mp_L_after_zs[:, j], mp_R_after_zs[:, j], calib).astype(np.float32)
        joints_3d_after_hr[:, j, :] = triangulate_points(
            mp_L_after_hr[:, j], mp_R_after_hr[:, j], calib).astype(np.float32)
        joints_3d_after_bc[:, j, :] = triangulate_points(
            mp_L_after_bc[:, j], mp_R_after_bc[:, j], calib).astype(np.float32)
        joints_3d[:, j, :] = triangulate_points(
            mp_L_c[:, j], mp_R_c[:, j], calib).astype(np.float32)

    # Save in the same format as the existing Skeleton (v3) fit output
    out_dir = _mano_dir(subject_name) / trial_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mano_fit_v2.npz"
    params_path = out_dir / "mano_fit_v2_params.json"

    # Rotate history (same as v2 fit does)
    for i in range(3, 0, -1):
        src_npz = out_dir / f"mano_fit_v2_prev{i}.npz"
        src_json = out_dir / f"mano_fit_v2_prev{i}_params.json"
        if i == 3:
            if src_npz.exists(): src_npz.unlink()
            if src_json.exists(): src_json.unlink()
        else:
            if src_npz.exists(): src_npz.rename(out_dir / f"mano_fit_v2_prev{i+1}.npz")
            if src_json.exists(): src_json.rename(out_dir / f"mano_fit_v2_prev{i+1}_params.json")
    if out_path.exists():
        (out_dir / "mano_fit_v2_prev1.npz").unlink(missing_ok=True)
        out_path.rename(out_dir / "mano_fit_v2_prev1.npz")
    if params_path.exists():
        (out_dir / "mano_fit_v2_prev1_params.json").unlink(missing_ok=True)
        params_path.rename(out_dir / "mano_fit_v2_prev1_params.json")

    np.savez(
        str(out_path),
        joints_3d=joints_3d,                    # final (after Y+Z-out+Z-jump+BL)
        mp_L_corrected=mp_L_c,                  # final
        mp_R_corrected=mp_R_c,
        # Step 0 -- after stereo-correction.
        joints_3d_after_sc=joints_3d_after_sc,
        mp_L_after_sc=mp_L_after_sc,
        mp_R_after_sc=mp_R_after_sc,
        stereo_blame=stereo_blame.astype(np.bool_),
        stereo_L_pts=stereo_L_pts,
        stereo_R_pts=stereo_R_pts,
        stereo_response=stereo_response,
        stereo_mode=str(stereo_mode),
        stereo_conf=np.float32(stereo_conf),
        stereo_dist_px=np.float32(stereo_dist_px),
        stereo_occlusion_px=np.float32(stereo_occlusion_px),
        joints_3d_after_y=joints_3d_after_y,    # after Y only
        mp_L_after_y=mp_L_after_y,
        mp_R_after_y=mp_R_after_y,
        joints_3d_after_z=joints_3d_after_z,    # after Y + Z-outlier
        mp_L_after_z=mp_L_after_z,
        mp_R_after_z=mp_R_after_z,
        joints_3d_after_zs=joints_3d_after_zs,  # after Y + Z-outlier + Z-jump
        mp_L_after_zs=mp_L_after_zs,
        mp_R_after_zs=mp_R_after_zs,
        joints_3d_after_hr=joints_3d_after_hr,  # after HRnet snap + re-clean
        mp_L_after_hr=mp_L_after_hr,
        mp_R_after_hr=mp_R_after_hr,
        joints_3d_after_bc=joints_3d_after_bc,  # after bone-length correction
        mp_L_after_bc=mp_L_after_bc,
        mp_R_after_bc=mp_R_after_bc,
        hrnet_along_L=(hrnet_offsets["along_L"] if hrnet_offsets and hrnet_offsets.get("along_L") is not None else np.zeros(21, dtype=np.float32)),
        hrnet_perp_L=(hrnet_offsets["perp_L"] if hrnet_offsets and hrnet_offsets.get("perp_L") is not None else np.zeros(21, dtype=np.float32)),
        hrnet_along_R=(hrnet_offsets["along_R"] if hrnet_offsets and hrnet_offsets.get("along_R") is not None else np.zeros(21, dtype=np.float32)),
        hrnet_perp_R=(hrnet_offsets["perp_R"] if hrnet_offsets and hrnet_offsets.get("perp_R") is not None else np.zeros(21, dtype=np.float32)),
        hrnet_child=(hrnet_offsets["child_L"] if hrnet_offsets and hrnet_offsets.get("child_L") is not None else np.full(21, -1, dtype=np.int32)),
        n_frames=N,
        stage=5,
        fit_type="corrections_y_zout_zjump_hrnet_bl",
    )

    params_path.write_text(_json.dumps({
        "fit_type": "corrections_y_disp",
        "version": "corrections-stage1",
        "subject": subject_name,
        "trial": trial_stem,
        "n_frames": int(N),
        "n_corrected": int(n_corrected),
        # Nested under "params" so the frontend's v2_fit_params field
        # picks them up via _restoreV2Params().
        "params": {
            "detection": det_weights,
            "attribution": attr_weights,
            "stereo": {
                "mode": str(stereo_mode),
                "mask_dilate_px": int(stereo_mask_dilate_px),
                "gauss_center_weight": float(stereo_gauss_center_weight),
                "conf": float(stereo_conf),
                "dist_px": float(stereo_dist_px),
                "occlusion_px": float(stereo_occlusion_px),
            },
        },
        "timestamp": datetime.now().isoformat(),
    }, indent=2))

    from .mano_data import _load_mano_npz
    _load_mano_npz.cache_clear()

    return {
        "n_corrected": int(n_corrected),
        "out_path": str(out_path),
        "n_frames": int(N),
    }


def _stage_error_cache_path(subject_name: str, trial_stem: str, stage: str | None) -> Path:
    tag = stage or "mediapipe"
    return _mano_dir(subject_name) / trial_stem / f"mp_errors_{tag}.npz"


def _weights_match(saved: dict, current: dict) -> bool:
    """Shallow numeric comparison of two slider dicts (tolerant to missing keys)."""
    keys = set(saved.keys()) | set(current.keys())
    for k in keys:
        s = float(saved.get(k, 0.0) or 0.0)
        c = float(current.get(k, 0.0) or 0.0)
        if abs(s - c) > 1e-9:
            return False
    return True


def prefill_error_caches_for_all_subjects(progress_callback=None,
                                            cancel_event=None) -> dict:
    """Walk every subject + trial + stage and pre-compute the per-stage
    error matrix into ``mp_errors_<stage>.npz``.  Slider values are taken
    from each trial's last-saved fit params (``mano_fit_v2_params.json``)
    when available, otherwise zero — so the cache reflects what the UI
    will request next time the user opens that trial.

    Returns a summary dict.  Suitable for kicking off as a background job.
    """
    import json as _json
    from .video import build_trial_map
    from .mano_data import _mano_dir, list_mano_trials
    from ..db import get_db_ctx

    ALL_STAGES_LOCAL = ["mediapipe", "stereo_correct", "z_correct", "z_smooth",
                        "hrnet_snap", "bone_correct", "bone_smooth"]
    summary = {"subjects": 0, "trials": 0, "stages_computed": 0,
               "stages_skipped": 0, "errors": []}

    with get_db_ctx() as db:
        subjects = db.execute("SELECT id, name FROM subjects ORDER BY name").fetchall()
    subject_names = [r["name"] for r in subjects]
    total_steps = max(1, len(subject_names))
    def _cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    for s_idx, name in enumerate(subject_names):
        if _cancelled():
            raise _CorrectionsCancelled()
        try:
            trials = list_mano_trials(name)
        except Exception as e:
            summary["errors"].append(f"{name}: list_mano_trials failed — {e}")
            continue
        summary["subjects"] += 1
        if progress_callback:
            try: progress_callback(int(100 * s_idx / total_steps))
            except Exception: pass
        for t in trials:
            if _cancelled():
                raise _CorrectionsCancelled()
            trial_stem = t["trial_stem"]
            summary["trials"] += 1
            # Slider values: prefer the saved fit params, else zero defaults.
            params_path = _mano_dir(name) / trial_stem / "mano_fit_v2_params.json"
            det_w, attr_w = {}, {}
            if params_path.exists():
                try:
                    p = _json.loads(params_path.read_text())
                    inner = p.get("params") if isinstance(p.get("params"), dict) else p
                    det_w = inner.get("detection") or {}
                    attr_w = inner.get("attribution") or {}
                except Exception:
                    pass
            for stage in ALL_STAGES_LOCAL:
                # Filter detection weights to just the stage's factor (UI
                # behaviour mirror) so each stage's cache file is keyed
                # the same way the live recompute keys it.
                # mediapipe stage now reflects the union of y_disp and
                # z_outlier (those are what the combined yz corrector flags).
                stage_factor_map = {
                    "mediapipe": "stereo_dist",
                    "stereo_correct": ("y_disp", "z_outlier"),
                    "z_correct": "z_jump", "z_smooth": "hrnet_mismatch",
                    "hrnet_snap": "bone_length", "bone_correct": "bone_agreement",
                    "bone_smooth": "angle",
                }
                factor = stage_factor_map.get(stage)
                if isinstance(factor, tuple):
                    active = set(factor)
                else:
                    active = {factor} if factor else set()
                det_filtered = {k: (det_w.get(k, 0.0) if k in active else 0.0)
                                for k in DETECTION_FACTORS}
                try:
                    compute_errors_for_trial(name, trial_stem,
                                             det_filtered, attr_w,
                                             stage=stage)
                    summary["stages_computed"] += 1
                except Exception as e:
                    summary["stages_skipped"] += 1
                    summary["errors"].append(f"{name}/{trial_stem}/{stage}: {e}")
    if progress_callback:
        try: progress_callback(100)
        except Exception: pass
    return summary


def compute_errors_for_trial(subject_name: str, trial_stem: str,
                             det_weights: dict, attr_weights: dict,
                             corr_weights: dict | None = None,
                             stage: str | None = None) -> dict:
    """Load MP + calibration + priors, optionally apply corrections, then
    compute the final error matrix.

    Results are cached to disk per stage (``mp_errors_<stage>.npz``) along
    with the slider values that produced them.  A cached matrix is returned
    verbatim if the current slider values match the saved ones — so the UI
    can toggle Err checkboxes off/on without paying the recompute cost.
    Moving any slider invalidates the cache and triggers a fresh compute.

    ``stage`` controls which MP snapshot drives error detection:
      - None or 'mediapipe': raw MP landmarks (no corrections)
      - 'z_correct':    after combined Y-disparity + Z-outlier correction
      - 'z_smooth':     after Y/Z + Z-jump correction (pre-BL)
      - 'bone_correct': after Y/Z + Z-jump + HRnet-snap + bone-length
    Snapshots are loaded from ``mano_fit_v2.npz`` when present; missing
    snapshots fall back to raw MP.

    Returns dict with:
      - 'errors': (N, 21, 2) bool array
      - 'mp_L', 'mp_R': (N, 21, 2) — the corrected MP positions (equal to
         the original if no corrections were requested)
      - 'n_corrected': int count of (frame, joint, camera) corrections applied
    """
    from .mediapipe_prelabel import load_mediapipe_prelabels
    from .mano_data import _load_trial_calibration, load_angle_priors
    from .video import build_trial_map

    corr_weights = corr_weights or {}

    trials = build_trial_map(subject_name)
    trial = next((t for t in trials if t["trial_name"] == trial_stem), None)
    if trial is None:
        raise ValueError(f"Trial {trial_stem} not found for {subject_name}")
    N = trial["frame_count"]
    start = trial.get("start_frame", 0)

    # Disk cache check — return the saved error matrix if this stage's
    # detection + attribution sliders haven't changed since last compute.
    cache_path = _stage_error_cache_path(subject_name, trial_stem, stage)
    if cache_path.exists():
        try:
            cdata = np.load(str(cache_path), allow_pickle=True)
            saved_det  = dict(cdata["det_weights"].tolist())  if "det_weights"  in cdata.files else {}
            saved_attr = dict(cdata["attr_weights"].tolist()) if "attr_weights" in cdata.files else {}
            if (_weights_match(saved_det, det_weights) and
                _weights_match(saved_attr, attr_weights) and
                int(cdata.get("n_frames", -1)) == N):
                mp_L, mp_R, _, _ = _load_mp_labels(subject_name, trial_stem, start, N)
                return {
                    "errors": cdata["errors"].astype(bool),
                    "mp_L": mp_L, "mp_R": mp_R, "n_corrected": 0,
                }
        except Exception as e:
            logger.warning(f"Stage error-cache load failed ({cache_path.name}): {e}")

    mp_L, mp_R, conf_L, conf_R = _load_mp_labels(subject_name, trial_stem, start, N)

    # Short-circuit when there's no actual MP data (new subject, no
    # detection run yet) — return zero errors instead of crashing the
    # scoring pipeline with NaN-only inputs.
    if mp_L is None or not np.any(~np.isnan(mp_L[:, :, 0])):
        return {
            "errors": np.zeros((N, 21, 2), dtype=bool),
            "mp_L": mp_L, "mp_R": mp_R, "n_corrected": 0,
        }

    # Stage-specific MP snapshot (so each correction's own detection
    # measures from the most-up-to-date labels).
    stage_tag = ""
    if stage and stage != "mediapipe":
        v2_path = _mano_dir(subject_name) / trial_stem / "mano_fit_v2.npz"
        if v2_path.exists():
            try:
                with np.load(str(v2_path), allow_pickle=True) as z:
                    if stage == "stereo_correct" and "mp_L_after_sc" in z.files:
                        mp_L = z["mp_L_after_sc"].astype(np.float32)
                        mp_R = z["mp_R_after_sc"].astype(np.float32)
                        stage_tag = "sc"
                    elif stage == "z_correct" and "mp_L_after_z" in z.files:
                        mp_L = z["mp_L_after_z"].astype(np.float32)
                        mp_R = z["mp_R_after_z"].astype(np.float32)
                        stage_tag = "z"
                    elif stage == "z_smooth" and "mp_L_after_zs" in z.files:
                        mp_L = z["mp_L_after_zs"].astype(np.float32)
                        mp_R = z["mp_R_after_zs"].astype(np.float32)
                        stage_tag = "zs"
                    elif stage == "hrnet_snap" and "mp_L_after_hr" in z.files:
                        mp_L = z["mp_L_after_hr"].astype(np.float32)
                        mp_R = z["mp_R_after_hr"].astype(np.float32)
                        stage_tag = "hr"
                    elif stage == "bone_correct" and "mp_L_after_bc" in z.files:
                        mp_L = z["mp_L_after_bc"].astype(np.float32)
                        mp_R = z["mp_R_after_bc"].astype(np.float32)
                        stage_tag = "bc"
                    elif stage == "bone_smooth" and "mp_L_corrected" in z.files:
                        mp_L = z["mp_L_corrected"].astype(np.float32)
                        mp_R = z["mp_R_corrected"].astype(np.float32)
                        stage_tag = "bs"
            except Exception:
                pass  # fall back to raw MP

    calib = _load_trial_calibration(subject_name, trial_stem)
    priors = load_angle_priors().get("joints", [])

    cache_key = (subject_name, trial_stem, N, start, stage_tag)
    det, attr = _get_scores(subject_name, trial_stem, mp_L, mp_R, conf_L, conf_R,
                              calib, priors, cache_key=cache_key)
    # The mediapipe stage's corrections (combined Y/Z) are winner-take-all
    # — only one camera per (frame, joint) is ever moved.  Mirror that in
    # the error overlay so the user never sees the same joint marked red
    # in both cameras simultaneously for that stage.
    wta = (stage in (None, "mediapipe"))
    errors = apply_thresholds(det, attr, det_weights, attr_weights,
                              winner_take_all=wta)

    # Overlay stereo_dist (drives the mediapipe-stage error overlay
    # after the v3 step-0 stereo-correction).  Prefer the BAKED
    # ``stereo_blame`` mask written into mano_fit_v2.npz by the v3
    # fit -- that way the overlay always matches what the
    # Stereo-Correct stage view is actually showing (same MP labels
    # replaced for the same joints / cameras).  Fall back to a live
    # recompute from stereo_align_<mode>.npz only when no fit row
    # exists yet, so the user gets a useful preview before their
    # first bake.
    sd_w = float(det_weights.get("stereo_dist", 0.0))
    sc_w = float(det_weights.get("stereo_conf", 0.0))
    if sd_w > 0:
        v2_path = _mano_dir(subject_name) / trial_stem / "mano_fit_v2.npz"
        used_baked = False
        if v2_path.exists():
            try:
                with np.load(str(v2_path), allow_pickle=True) as z:
                    if "stereo_blame" in z.files:
                        sb = z["stereo_blame"].astype(bool)
                        n_sb = min(N, int(sb.shape[0]))
                        errors[:n_sb] = errors[:n_sb] | sb[:n_sb]
                        used_baked = True
            except Exception as e:
                logger.warning(f"stereo_blame overlay load failed: {e}")
        if not used_baked:
            try:
                from .stereo_align import load_stereo_align
                t_idx = next((i for i, t in enumerate(
                    build_trial_map(subject_name))
                    if t["trial_name"] == trial_stem), None)
                sa = None
                if t_idx is not None:
                    for _m in ("hybrid", "outline", "image"):
                        sa = load_stereo_align(subject_name, t_idx, mode=_m)
                        if sa is not None:
                            break
                if sa is not None:
                    shifts = sa["shifts"]                # (N_sa, 21, 2)
                    resp   = sa["response"]              # (N_sa, 21)
                    n_sa = min(N, int(shifts.shape[0]))
                    dist = np.linalg.norm(shifts[:n_sa], axis=-1)
                    conf_ok = resp[:n_sa] >= sc_w
                    err_mask = conf_ok & (dist > sd_w)
                    attr_per_cam = _combined_attr_per_cam(attr, attr_weights)
                    stereo_overlay = np.zeros((N, 21, 2), dtype=bool)
                    for fi in range(n_sa):
                        for j in range(21):
                            if not err_mask[fi, j]:
                                continue
                            if attr_per_cam is not None:
                                a0 = float(attr_per_cam[fi, j, 0])
                                a1 = float(attr_per_cam[fi, j, 1])
                                blame_left = a0 >= a1
                            else:
                                cL = float(conf_L[fi, j]) if conf_L is not None else 0.0
                                cR = float(conf_R[fi, j]) if conf_R is not None else 0.0
                                blame_left = (cL <= cR)
                            stereo_overlay[fi, j, 0 if blame_left else 1] = True
                    errors = errors | stereo_overlay
            except Exception as e:
                logger.warning(f"stereo-dist error overlay skipped: {e}")

    # Overlay hrnet_mismatch (drives the z_smooth stage's error overlay).
    # The slider is an absolute pixel-distance threshold compared against
    # the 2D distance between each camera's HRnet Peak Select label and
    # the location predicted by the Z-smooth 3D model + per-joint
    # systematic offset.  No per-camera attribution: if either camera's
    # distance exceeds the threshold, the joint is flagged in BOTH cams
    # (the snap step replaces both cameras' MP labels with the HRnet peaks
    # and lets the downstream Y/Z-correct + Z-jump passes clean them up).
    hr_w = float(det_weights.get("hrnet_mismatch", 0.0))
    if hr_w > 0:
        hr_L, hr_R = _load_hrnet_peaks_arr(subject_name, trial_stem, start, N)
        if calib is not None and hr_L is not None and hr_R is not None:
            # mp_L / mp_R here are the Z-smooth-stage MP labels (the
            # ``stage`` snapshot resolution above already remapped them).
            mp_3d_mm = np.full((N, 21, 3), np.nan, dtype=np.float32)
            hr_3d_mm = np.full((N, 21, 3), np.nan, dtype=np.float32)
            for j in range(21):
                mp_3d_mm[:, j] = triangulate_points(
                    mp_L[:, j], mp_R[:, j], calib).astype(np.float32)
                hr_3d_mm[:, j] = triangulate_points(
                    hr_L[:, j], hr_R[:, j], calib).astype(np.float32)
            along3, flex3, abd3, child3 = _hrnet_snap_compute_offsets_3d(
                mp_3d_mm, hr_3d_mm)
            joint_flag = np.zeros((N, 21), dtype=bool)
            for j in range(21):
                if j in _HRNET_EXEMPT_JOINTS: continue  # wrist + thumb CMC ignored
                if child3[j] < 0: continue
                for f in range(N):
                    mp_pt = mp_3d_mm[f, j]; hr_pt = hr_3d_mm[f, j]
                    if np.any(np.isnan(mp_pt)) or np.any(np.isnan(hr_pt)): continue
                    basis = _joint_basis_3d(mp_3d_mm[f], j)
                    if basis is None: continue
                    e_along, e_flex, e_abd = basis
                    # Predicted HRnet 3D = Z-smooth joint + median offset
                    # (offset is HR − MP, decomposed onto the per-frame basis).
                    pred_hr_3d = (mp_pt
                                  + float(along3[j]) * e_along
                                  + float(flex3[j])  * e_flex
                                  + float(abd3[j])   * e_abd)
                    # Project the prediction to both cameras and compare
                    # against the HRnet Peak-Select 2D peaks per camera.
                    pLp, pRp = _project_point_stereo(
                        pred_hr_3d.astype(np.float32), calib)
                    if pLp is None or pRp is None: continue
                    hrL2d = hr_L[f, j]; hrR2d = hr_R[f, j]
                    d_L = (np.hypot(pLp[0] - hrL2d[0], pLp[1] - hrL2d[1])
                           if not np.any(np.isnan(hrL2d)) else 0.0)
                    d_R = (np.hypot(pRp[0] - hrR2d[0], pRp[1] - hrR2d[1])
                           if not np.any(np.isnan(hrR2d)) else 0.0)
                    if max(float(d_L), float(d_R)) > hr_w:
                        joint_flag[f, j] = True
            # Flag both cameras when the joint mismatches in either camera.
            errors[:, :, 0] = errors[:, :, 0] | joint_flag
            errors[:, :, 1] = errors[:, :, 1] | joint_flag

    # Persist this (stage, sliders) → error matrix for instant re-display.
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(cache_path),
                 errors=errors.astype(np.bool_),
                 det_weights=np.array(list(det_weights.items()), dtype=object),
                 attr_weights=np.array(list(attr_weights.items()), dtype=object),
                 n_frames=int(N),
                 stage=str(stage or "mediapipe"))
    except Exception as e:
        logger.warning(f"Failed to save stage error cache ({cache_path.name}): {e}")

    return {
        "errors": errors,
        "mp_L": mp_L,
        "mp_R": mp_R,
        "n_corrected": 0,
    }


def save_errors(subject_name: str, trial_stem: str,
                det_weights: dict, attr_weights: dict,
                corr_weights: dict | None = None) -> Path:
    result = compute_errors_for_trial(subject_name, trial_stem,
                                       det_weights, attr_weights, corr_weights)
    errors = result["errors"]
    out_dir = _mano_dir(subject_name) / trial_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mp_errors.npz"
    np.savez(str(out_path),
             errors=errors.astype(np.bool_),
             det_weights=np.array(list(det_weights.items()), dtype=object),
             attr_weights=np.array(list(attr_weights.items()), dtype=object))
    return out_path


def load_saved_errors(subject_name: str, trial_stem: str):
    """Load saved error matrix + slider values if file exists; else None."""
    p = _mano_dir(subject_name) / trial_stem / "mp_errors.npz"
    if not p.exists():
        return None
    try:
        data = np.load(str(p), allow_pickle=True)
        errors = data["errors"]
        det = dict(data["det_weights"].tolist()) if "det_weights" in data.files else {}
        attr = dict(data["attr_weights"].tolist()) if "attr_weights" in data.files else {}
        return {"errors": errors, "det_weights": det, "attr_weights": attr}
    except Exception as e:
        logger.warning(f"Failed to load {p}: {e}")
        return None
