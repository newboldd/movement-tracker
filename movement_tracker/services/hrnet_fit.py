"""HRnet peak refinement pipeline (cluster centroid + joint stereo Hungarian).

Replaces the Peak-Select algorithm.  Zero MediaPipe involvement: candidates
come from heatmap geometry alone; assignment is decided by heatmap value,
empirical y-disparity, anchored temporal smoothness, and a post-assignment
overlap-repulsion rule.

Public entry points
-------------------
* ``run_hrnet_fit(heatmaps_L, heatmaps_R, bbox_L, bbox_R, params)``
  Returns ``(peaks_centroid_L, peaks_centroid_R,
            peaks_hungarian_L, peaks_hungarian_R, info)``.

* ``HRNET_FIT_GROUPS`` — joint-type groups assigned together.
"""
from __future__ import annotations

import logging
from itertools import permutations
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from .hrnet import (_cluster_centroid_peaks, _cluster_centroid_peak,
                     _cluster_centroid_peaks_with_auc)

logger = logging.getLogger(__name__)


# Wrist + thumb CMC merged into a new group; thumb CMC is removed from MCPs.
HRNET_FIT_GROUPS: tuple[tuple[int, ...], ...] = (
    (0, 1),                    # wrist + thumb CMC
    (5, 9, 13, 17),            # MCPs (no thumb)
    (2, 6, 10, 14, 18),        # PIPs (incl. T_MCP)
    (3, 7, 11, 15, 19),        # DIPs (incl. T_IP)
    (4, 8, 12, 16, 20),        # Tips
)


# ──────────────────────────── Candidate finding ─────────────────────────────

def _group_mip(heatmaps: np.ndarray, group: tuple[int, ...]) -> np.ndarray:
    """Per-frame MIP over a group's joint heatmaps.  ``heatmaps`` is
    (N, J, H, W); returns (N, H, W)."""
    return heatmaps[:, list(group)].max(axis=1)


def _topk_peaks_with_nms(mip: np.ndarray, K: int, nms_radius_hm: float):
    """Greedy NMS top-K peaks on a single (H, W) MIP.

    ``nms_radius_hm`` is the suppression radius in heatmap-pixel units.
    Returns a list of (py, px, value) up to length K, sorted by value
    descending.  Returns fewer than K when the MIP is too small / sparse.
    """
    H, W = mip.shape
    work = mip.copy()
    out: list[tuple[int, int, float]] = []
    r = max(1, int(round(nms_radius_hm)))
    for _ in range(K):
        idx = int(work.reshape(-1).argmax())
        py, px = idx // W, idx % W
        v = float(work[py, px])
        if v <= 0:
            break
        out.append((py, px, v))
        y0, y1 = max(0, py - r), min(H, py + r + 1)
        x0, x1 = max(0, px - r), min(W, px + r + 1)
        work[y0:y1, x0:x1] = 0.0
    return out


def _candidates_per_frame(
    heatmaps: np.ndarray, group: tuple[int, ...],
    bbox: tuple, cluster_size: int, nms_radius_hm: float,
    spike_support: float = 0.0, edge_margin: int = 0,
) -> np.ndarray:
    """Per-frame top-K candidates for a group on one camera.

    Workflow per frame:
      1. MIP across the group's joint heatmaps.
      2. NMS top-K from the MIP (K = group size).
      3. For each candidate, run cluster-centroid refinement against
         each of the group's individual joint heatmaps and keep the
         **best joint heatmap value** at that location plus the
         centroid-refined image coords.

    Returns ``(N, K, 2)`` image-coord candidates and a parallel
    ``(N, K, |group|)`` array of per-joint heatmap values at each
    candidate (used as the cost-cube input).
    """
    from .hrnet import _bbox_at
    N, J, H, W = heatmaps.shape
    K = len(group)

    cands = np.full((N, K, 2), np.nan, dtype=np.float32)
    cand_h = np.zeros((N, K, len(group)), dtype=np.float32)

    mip = _group_mip(heatmaps, group)  # (N, H, W)
    for f in range(N):
        x1, y1, x2, y2 = _bbox_at(bbox, f)
        bw, bh = float(x2 - x1), float(y2 - y1)
        peaks = _topk_peaks_with_nms(mip[f], K, nms_radius_hm)
        for ki, (py, px, _v) in enumerate(peaks):
            for gi, j in enumerate(group):
                cand_h[f, ki, gi] = float(heatmaps[f, j, py, px])
            best_gi = int(np.argmax(cand_h[f, ki]))
            cx, cy = _cluster_centroid_peak(
                heatmaps[f, group[best_gi]], cluster_size,
                spike_support=spike_support, edge_margin=edge_margin)
            if not (np.isnan(cx) or np.isnan(cy)):
                cands[f, ki, 0] = x1 + (cx / W) * bw
                cands[f, ki, 1] = y1 + (cy / H) * bh
            else:
                cands[f, ki, 0] = x1 + (px + 0.5) / W * bw
                cands[f, ki, 1] = y1 + (py + 0.5) / H * bh
    return cands, cand_h


# ─────────────────────────── Empirical y-disparity ──────────────────────────

def _per_joint_y_disparity(peaks_L: np.ndarray, peaks_R: np.ndarray) -> np.ndarray:
    """Median ``y_L − y_R`` per joint over valid frames.  Used as the
    target disparity in the cost cube.  Returns (J,) float32 with NaN
    where insufficient data."""
    N, J, _ = peaks_L.shape
    out = np.full(J, np.nan, dtype=np.float32)
    for j in range(J):
        m = ~np.isnan(peaks_L[:, j, 1]) & ~np.isnan(peaks_R[:, j, 1])
        if int(m.sum()) >= 5:
            out[j] = float(np.median(peaks_L[m, j, 1] - peaks_R[m, j, 1]))
    return out


# ───────────────────────────── Anchored temporal ────────────────────────────

def _polyfit_anchor_one(
    pts: np.ndarray, valid: np.ndarray, win: int = 3, deg: int = 2,
) -> np.ndarray:
    """For each frame, fit a degree-``deg`` polynomial to neighbour
    frames in a ±``win`` window (excluding f itself), evaluate at f.

    ``pts`` shape (N, 2) — per-frame 2D positions.
    ``valid`` shape (N,) bool — frames whose ``pts`` is trustworthy.

    Returns (N, 2) — predicted position at each frame.  NaN where the
    window has fewer than ``deg + 1`` clean neighbours.
    """
    N = pts.shape[0]
    out = np.full((N, 2), np.nan, dtype=np.float32)
    for f in range(N):
        lo, hi = max(0, f - win), min(N, f + win + 1)
        idxs = np.arange(lo, hi)
        keep = (idxs != f) & valid[lo:hi]
        if int(keep.sum()) < deg + 1:
            continue
        xs = idxs[keep].astype(np.float64)
        ys_x = pts[lo:hi, 0][keep].astype(np.float64)
        ys_y = pts[lo:hi, 1][keep].astype(np.float64)
        try:
            cx = np.polyfit(xs, ys_x, deg)
            cy = np.polyfit(xs, ys_y, deg)
            out[f, 0] = float(np.polyval(cx, f))
            out[f, 1] = float(np.polyval(cy, f))
        except Exception:
            continue
    return out


# ────────────────────────────── Cost cube ───────────────────────────────────

def _build_cost_cube(
    cands_L: np.ndarray, cand_h_L: np.ndarray,
    cands_R: np.ndarray, cand_h_R: np.ndarray,
    group: tuple[int, ...], emp_disp: np.ndarray,
    h_max: np.ndarray, w_hm: float, w_disp: float, w_temp: float,
    poly_L: np.ndarray | None, poly_R: np.ndarray | None,
    f: int,
    anchor_L: np.ndarray | None = None,
    anchor_R: np.ndarray | None = None,
    w_anchor: float = 0.0,
    anchor_scale: float = 1.0,
) -> np.ndarray:
    """Build the ``(J × K × K)`` cost cube for one frame.

    ``cost[ji, c_OS, c_OD]`` = full cost for assigning group joint ``ji``
    to OS candidate ``c_OS`` and OD candidate ``c_OD``.  Infinite where
    a candidate is invalid (NaN).
    """
    J = len(group)
    K = cands_L.shape[1]
    cube = np.full((J, K, K), np.inf, dtype=np.float64)
    p_L_f = cands_L[f]   # (K, 2)
    p_R_f = cands_R[f]
    cand_hL_f = cand_h_L[f]  # (K, J)
    cand_hR_f = cand_h_R[f]
    for ji, j in enumerate(group):
        for cL in range(K):
            if np.isnan(p_L_f[cL, 0]):
                continue
            for cR in range(K):
                if np.isnan(p_R_f[cR, 0]):
                    continue
                hL = float(cand_hL_f[cL, ji])
                hR = float(cand_hR_f[cR, ji])
                hm_max = float(h_max[j]) if h_max[j] > 1e-9 else 1.0
                cost_hm = (1.0 - hL / hm_max) + (1.0 - hR / hm_max)

                disp_pred = float(emp_disp[j]) if not np.isnan(emp_disp[j]) else 0.0
                disp_obs = float(p_L_f[cL, 1]) - float(p_R_f[cR, 1])
                cost_disp = abs(disp_obs - disp_pred)

                cost_temp = 0.0
                if w_temp > 0 and poly_L is not None and poly_R is not None:
                    pL = poly_L[ji, f]; pR = poly_R[ji, f]
                    if not (np.any(np.isnan(pL)) or np.any(np.isnan(pR))):
                        dxL = float(p_L_f[cL, 0]) - float(pL[0])
                        dyL = float(p_L_f[cL, 1]) - float(pL[1])
                        dxR = float(p_R_f[cR, 0]) - float(pR[0])
                        dyR = float(p_R_f[cR, 1]) - float(pR[1])
                        cost_temp = float(np.hypot(dxL, dyL) + np.hypot(dxR, dyR))

                cost_anchor = 0.0
                if w_anchor > 0 and anchor_L is not None and anchor_R is not None:
                    aL = anchor_L[f, j]; aR = anchor_R[f, j]
                    if not (np.any(np.isnan(aL)) or np.any(np.isnan(aR))):
                        dxL = float(p_L_f[cL, 0]) - float(aL[0])
                        dyL = float(p_L_f[cL, 1]) - float(aL[1])
                        dxR = float(p_R_f[cR, 0]) - float(aR[0])
                        dyR = float(p_R_f[cR, 1]) - float(aR[1])
                        # Normalised by overlap_px-style scale so w_anchor=1
                        # is comparable to the heatmap term magnitude.
                        cost_anchor = (np.hypot(dxL, dyL) + np.hypot(dxR, dyR)) / max(1e-6, anchor_scale)

                cube[ji, cL, cR] = (w_hm * cost_hm
                                    + w_disp * cost_disp
                                    + w_temp * cost_temp
                                    + w_anchor * cost_anchor)
    return cube


# ──────────────────────────── Joint stereo solve ─────────────────────────────

def _solve_3dap(cube: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """3-D assignment problem: pick (j → c_OS, j → c_OD) minimising total
    cost s.t. no c_OS / c_OD is reused.  ``cube`` shape (J, K, K).

    Algorithm: enumerate OS-candidate permutations of length J over K
    options; for each, solve a J×K Hungarian over OD candidates with
    the OS assignment fixed.  Track the minimum-cost solution.

    Returns (os_assign, od_assign, total_cost).  os_assign[ji] is the
    OS candidate index for group joint ``ji``; od_assign similarly.
    """
    J, K, _ = cube.shape
    if J > K:
        raise ValueError(f"3DAP needs J ≤ K, got J={J}, K={K}")

    best_cost = np.inf
    best_os: np.ndarray | None = None
    best_od: np.ndarray | None = None

    for os_perm in permutations(range(K), J):
        # cost_OD[ji, c_OD] = cube[ji, os_perm[ji], c_OD]
        cost_OD = np.empty((J, K), dtype=np.float64)
        for ji in range(J):
            cost_OD[ji, :] = cube[ji, os_perm[ji], :]
        # Hungarian's standard linear_sum_assignment requires square or
        # rectangular; replace inf with a very large finite penalty.
        finite = np.where(np.isfinite(cost_OD), cost_OD, 1e12)
        row, col = linear_sum_assignment(finite)
        total = float(finite[row, col].sum())
        if total < best_cost:
            best_cost = total
            best_os = np.array(os_perm, dtype=np.int64)
            # row is just 0..J-1 in order; col gives OD assignments
            order = np.argsort(row)
            best_od = col[order].astype(np.int64)

    if best_os is None or best_od is None:
        # Fully infeasible — return zeros so callers have something to log.
        return (np.zeros(J, dtype=np.int64),
                np.zeros(J, dtype=np.int64), float("inf"))
    return best_os, best_od, best_cost


# ─────────────────────────── Overlap repulsion ──────────────────────────────

# Joint pairs that are allowed to overlap (e.g. during pinching, the
# thumb tip and index tip legitimately share a peak).  Stored as a
# frozen set of joint-index pairs.
_OVERLAP_EXEMPT_PAIRS: frozenset = frozenset({
    (4, 8), (8, 4),     # thumb tip ↔ index tip — pinch
})


def _enforce_overlap(
    os_assign: np.ndarray, od_assign: np.ndarray,
    cube: np.ndarray, cands_L_f: np.ndarray, cands_R_f: np.ndarray,
    overlap_px: float,
    group: tuple[int, ...] | None = None,
):
    """Post-assignment overlap repulsion.  For any two group joints
    whose combined-camera distance is below ``overlap_px``, swap the
    higher-cost joint to its second-best candidate (per camera) until
    the rule is satisfied or no swap helps.  Mutates ``os_assign`` and
    ``od_assign`` in place.

    Combined-camera distance = ``sqrt(dx_OS² + dy_OS² + dx_OD² + dy_OD²)``
    treating both cameras' pixel deltas as components of a single vector.
    """
    if overlap_px <= 0:
        return
    J = len(os_assign)
    thr2 = overlap_px * overlap_px

    def pair_dist2(ja, jb):
        a_L = cands_L_f[os_assign[ja]]
        a_R = cands_R_f[od_assign[ja]]
        b_L = cands_L_f[os_assign[jb]]
        b_R = cands_R_f[od_assign[jb]]
        if np.any(np.isnan(a_L)) or np.any(np.isnan(b_L)) \
                or np.any(np.isnan(a_R)) or np.any(np.isnan(b_R)):
            return float("inf")
        return ((a_L[0] - b_L[0]) ** 2 + (a_L[1] - b_L[1]) ** 2
                + (a_R[0] - b_R[0]) ** 2 + (a_R[1] - b_R[1]) ** 2)

    for _ in range(J):  # at most J relaxation passes
        worst = None
        worst_d = float("inf")
        for ja in range(J):
            for jb in range(ja + 1, J):
                # Skip exempt joint pairs (e.g. thumb tip ↔ index tip).
                if group is not None:
                    pair = (group[ja], group[jb])
                    if pair in _OVERLAP_EXEMPT_PAIRS:
                        continue
                d2 = pair_dist2(ja, jb)
                if d2 < thr2 and d2 < worst_d:
                    worst_d = d2
                    worst = (ja, jb)
        if worst is None:
            return
        ja, jb = worst
        # Pick the higher-cost joint of the pair to move.
        cost_a = cube[ja, os_assign[ja], od_assign[ja]]
        cost_b = cube[jb, os_assign[jb], od_assign[jb]]
        loser = ja if cost_a >= cost_b else jb
        # Try its 2nd-, 3rd-, ... best (c_OS, c_OD) combo, picking the
        # cheapest one that satisfies overlap with all OTHER joints and
        # whose c_OS / c_OD aren't already used.
        used_os = {os_assign[k] for k in range(J) if k != loser}
        used_od = {od_assign[k] for k in range(J) if k != loser}
        K = cube.shape[1]
        order = []
        for cl in range(K):
            for cr in range(K):
                if cl in used_os or cr in used_od:
                    continue
                c = cube[loser, cl, cr]
                if np.isfinite(c):
                    order.append((c, cl, cr))
        order.sort()
        replaced = False
        for _, cl, cr in order:
            saved_os, saved_od = os_assign[loser], od_assign[loser]
            os_assign[loser] = cl
            od_assign[loser] = cr
            still_overlap = False
            for k in range(J):
                if k == loser: continue
                if pair_dist2(loser, k) < thr2:
                    still_overlap = True; break
            if not still_overlap:
                replaced = True
                break
            # restore and try next
            os_assign[loser] = saved_os
            od_assign[loser] = saved_od
        if not replaced:
            # No improvement available — leave as is and stop.
            return


# ───────────── Bone-length-driven single-joint re-pick ──────────────────

# Re-pick anchor map: joint → bone-length partner used as anchor when
# this joint is being re-picked.  We walk **tips → MCPs** (the user's
# preferred order — fingertips are the most reliable joints), so each
# joint's anchor is its kinematic CHILD (already finalised earlier in
# the walk), not its kinematic parent.  Wrist-MCP bones are excluded
# entirely; the wrist is reconstructed in a separate stage from
# constant-length constraints to all 5 MCPs.
_REPICK_ANCHOR = {
    # Thumb chain: T_CMC ← T_MCP ← T_IP ← T_tip
    3: 4, 2: 3, 1: 2,
    # Index, middle, ring, pinky: MCP ← PIP ← DIP ← tip
    7: 8, 6: 7, 5: 6,
    11: 12, 10: 11, 9: 10,
    15: 16, 14: 15, 13: 14,
    19: 20, 18: 19, 17: 18,
}
# Tips (4, 8, 12, 16, 20) are NEVER re-picked — they're the trusted
# anchors.  Wrist (0) is not in the order — handled separately.
_BONE_REPICK_ORDER = [
    # DIPs
    3, 7, 11, 15, 19,
    # PIPs (and T_MCP=2 since the thumb is one joint shorter)
    2, 6, 10, 14, 18,
    # MCPs (and T_CMC=1)
    1, 5, 9, 13, 17,
]


def _topk_image_candidates(hm: np.ndarray, bbox: tuple, K: int,
                            nms_radius_hm: float):
    """Top-K NMS peaks on a single (H, W) joint heatmap, returned as
    image-coord (x, y, value) tuples.  Used to seed bone-length re-pick."""
    H, W = hm.shape
    x1, y1, x2, y2 = bbox
    bw, bh = float(x2 - x1), float(y2 - y1)
    peaks = _topk_peaks_with_nms(hm, K, nms_radius_hm)
    return [(x1 + (px + 0.5) / W * bw,
             y1 + (py + 0.5) / H * bh,
             v) for (py, px, v) in peaks]


def _bone_length_repick(
    peaks_L_in: np.ndarray, peaks_R_in: np.ndarray,
    heatmaps_L: np.ndarray, heatmaps_R: np.ndarray,
    bbox_L: tuple, bbox_R: tuple,
    calib: dict,
    threshold_mm: float = 5.0,
    w_bone: float = 1.0,
    w_heatmap: float = 1.0,
    K_candidates: int = 8,
    nms_radius_px: float = 6.0,
) -> tuple:
    """For each frame and joint: if the bone to its parent deviates from
    the per-bone median by more than ``threshold_mm``, replace the joint's
    (OS, OD) labels with the heatmap-peak pair whose triangulated 3D
    sits closest to the target bone length (and on solid heatmap support).

    Walks parent → child so each joint's re-pick uses an already-finalised
    parent.  Returns ``(peaks_L_out, peaks_R_out)``.
    """
    from .calibration import triangulate_points
    N, J, _ = peaks_L_in.shape
    out_L = peaks_L_in.copy()
    out_R = peaks_R_in.copy()

    # Per-bone median length from current triangulated 3D.  Bones use
    # the re-pick anchor map (joint → its kinematic child for the
    # tips → MCPs walk).
    p3d_init = np.full((N, J, 3), np.nan, dtype=np.float32)
    for j in range(J):
        p3d_init[:, j] = triangulate_points(out_L[:, j], out_R[:, j], calib).astype(np.float32)
    bone_med = {}
    for j, anchor in _REPICK_ANCHOR.items():
        d = np.linalg.norm(p3d_init[:, j] - p3d_init[:, anchor], axis=1)
        valid = ~np.isnan(d)
        if int(valid.sum()) >= 5:
            bone_med[j] = float(np.median(d[valid]))

    from .hrnet import _bbox_at
    H_hm, W_hm = heatmaps_L.shape[2], heatmaps_L.shape[3]

    for j in _BONE_REPICK_ORDER:
        if j not in bone_med:
            continue
        p_idx = _REPICK_ANCHOR[j]    # kinematic child — already finalised
        target = bone_med[j]
        h_max_j = max(float(heatmaps_L[:, j].max()),
                      float(heatmaps_R[:, j].max())) or 1.0
        for f in range(N):
            parent_3d = triangulate_points(
                out_L[f:f+1, p_idx], out_R[f:f+1, p_idx], calib)[0]
            cur_3d = triangulate_points(
                out_L[f:f+1, j], out_R[f:f+1, j], calib)[0]
            if np.any(np.isnan(parent_3d)) or np.any(np.isnan(cur_3d)):
                continue
            cur_len = float(np.linalg.norm(cur_3d - parent_3d))
            if abs(cur_len - target) <= threshold_mm:
                continue
            # Per-frame bbox + NMS radius (frame-tight bbox can be smaller
            # than the trial union, so the radius converts differently).
            bb_L_f = _bbox_at(bbox_L, f)
            bb_R_f = _bbox_at(bbox_R, f)
            bw_L = max(1.0, float(bb_L_f[2] - bb_L_f[0]))
            bw_R = max(1.0, float(bb_R_f[2] - bb_R_f[0]))
            nms_hm_L = max(1.0, nms_radius_px / bw_L * W_hm)
            nms_hm_R = max(1.0, nms_radius_px / bw_R * W_hm)
            cands_L = _topk_image_candidates(
                heatmaps_L[f, j], bb_L_f, int(K_candidates), nms_hm_L)
            cands_R = _topk_image_candidates(
                heatmaps_R[f, j], bb_R_f, int(K_candidates), nms_hm_R)
            if not cands_L or not cands_R:
                continue
            best_cost = float("inf")
            best_pair = None
            for (xl, yl, vl) in cands_L:
                for (xr, yr, vr) in cands_R:
                    p3d = triangulate_points(
                        np.array([[xl, yl]], dtype=np.float32),
                        np.array([[xr, yr]], dtype=np.float32),
                        calib)[0]
                    if np.any(np.isnan(p3d)):
                        continue
                    new_len = float(np.linalg.norm(p3d - parent_3d))
                    bone_err = abs(new_len - target)
                    if bone_err > max(threshold_mm * 5.0, 20.0):
                        continue
                    hm_cost = (1.0 - float(vl) / h_max_j) + (1.0 - float(vr) / h_max_j)
                    cost = w_bone * bone_err + w_heatmap * hm_cost
                    if cost < best_cost:
                        best_cost = cost
                        best_pair = ((xl, yl), (xr, yr))
            if best_pair is not None:
                out_L[f, j] = best_pair[0]
                out_R[f, j] = best_pair[1]

    return out_L, out_R


# ───────── Constant-bone wrist reconstruction (HRnet Correct) ──────────

def _wrist_constant_bone_reconstruction(
    peaks_L: np.ndarray, peaks_R: np.ndarray,
    peaks_zsm_L: np.ndarray, peaks_zsm_R: np.ndarray,
    calib: dict,
    smooth_window: int = 5,
) -> tuple:
    """Reconstruct the wrist position so that all 5 wrist-MCP bones stay
    at their (Z-smooth-derived) median lengths, then temporally smooth
    and re-project the result into both cameras.

    Targets are computed on the Z-smoothed wrist + MCP positions (the
    cleanest pre-bone-repick view).  Solve is a per-frame
    Levenberg-Marquardt least-squares over the 3 wrist coordinates with
    5 distance residuals (one per MCP).  Y is anchored on the empirical
    wrist y-disparity to dodge calibration Ty bias.

    Returns ``(peaks_L_out, peaks_R_out)`` with wrist (joint 0) replaced.
    """
    from scipy.optimize import least_squares
    import cv2 as _cv2
    from .calibration import triangulate_points

    N = peaks_L.shape[0]
    out_L = peaks_L.copy()
    out_R = peaks_R.copy()
    if calib is None:
        return out_L, out_R

    MCP_INDICES = [1, 5, 9, 13, 17]    # T_CMC, I_MCP, M_MCP, R_MCP, P_MCP

    # Targets: median wrist-MCP bone lengths over the trial in the
    # Z-smooth output.  Anchored there so the rest of the pipeline's
    # bone-length re-pick can never warp the targets.
    wrist_zsm = triangulate_points(
        peaks_zsm_L[:, 0], peaks_zsm_R[:, 0], calib).astype(np.float64)
    target_lens = np.full(5, np.nan)
    mcp_zsm = np.full((N, 5, 3), np.nan, dtype=np.float64)
    for i, mi in enumerate(MCP_INDICES):
        mcp_zsm[:, i] = triangulate_points(
            peaks_zsm_L[:, mi], peaks_zsm_R[:, mi], calib)
    for i in range(5):
        d = np.linalg.norm(wrist_zsm - mcp_zsm[:, i], axis=1)
        valid = ~np.isnan(d)
        if int(valid.sum()) >= 5:
            target_lens[i] = float(np.median(d[valid]))

    # MCP positions from the now-corrected (post-bone-repick) labels.
    mcp_3d = np.full((N, 5, 3), np.nan, dtype=np.float64)
    for i, mi in enumerate(MCP_INDICES):
        mcp_3d[:, i] = triangulate_points(peaks_L[:, mi], peaks_R[:, mi], calib)

    # Per-frame least-squares solve.
    wrist_solved = np.full((N, 3), np.nan, dtype=np.float64)
    target_valid = ~np.isnan(target_lens)
    for f in range(N):
        mcps = mcp_3d[f]
        valid_mcps = ~np.isnan(mcps[:, 0]) & target_valid
        if int(valid_mcps.sum()) < 3:
            continue
        idx = np.where(valid_mcps)[0]
        # Initial guess: Z-smooth wrist if valid, else MCP centroid.
        init = wrist_zsm[f] if not np.any(np.isnan(wrist_zsm[f])) else mcps[idx].mean(axis=0)
        if np.any(np.isnan(init)):
            init = mcps[idx].mean(axis=0)
        try:
            res = least_squares(
                lambda w: np.array([np.linalg.norm(w - mcps[k]) - target_lens[k] for k in idx]),
                init, method='lm', max_nfev=20)
            wrist_solved[f] = res.x
        except Exception:
            pass

    # Temporal smoothing (degree-2 polynomial in a ±W window when there
    # are enough clean neighbours; else degree-1).
    smoothed = wrist_solved.copy()
    W = max(1, int(smooth_window))
    valid_solved = ~np.isnan(wrist_solved[:, 0])
    for f in range(N):
        if not valid_solved[f]:
            continue
        lo, hi = max(0, f - W), min(N, f + W + 1)
        keep = valid_solved[lo:hi]
        if int(keep.sum()) < 3:
            continue
        ts = np.arange(lo, hi)[keep].astype(np.float64)
        deg = 2 if int(keep.sum()) >= 5 else 1
        for ax in range(3):
            try:
                coef = np.polyfit(ts, wrist_solved[lo:hi, ax][keep], deg)
                smoothed[f, ax] = float(np.polyval(coef, f))
            except Exception:
                pass

    # Reproject into both cameras.  Anchor Y on empirical wrist
    # y-disparity from the Z-smooth pair to dodge the calibration Ty
    # bias (same trick the corrector uses).
    K1 = np.asarray(calib["K1"], dtype=np.float64)
    K2 = np.asarray(calib["K2"], dtype=np.float64)
    dist1 = np.asarray(calib["dist1"], dtype=np.float64)
    dist2 = np.asarray(calib["dist2"], dtype=np.float64)
    R = np.asarray(calib["R"], dtype=np.float64)
    T = np.asarray(calib["T"], dtype=np.float64).reshape(3, 1)
    rvec_I = np.zeros((3, 1)); tvec_0 = np.zeros((3, 1))
    rvec_R = _cv2.Rodrigues(R)[0]

    valid_pair = ~np.isnan(peaks_zsm_L[:, 0, 1]) & ~np.isnan(peaks_zsm_R[:, 0, 1])
    emp_disp_w = (float(np.median(peaks_zsm_L[valid_pair, 0, 1] - peaks_zsm_R[valid_pair, 0, 1]))
                  if int(valid_pair.sum()) >= 5 else float("nan"))

    for f in range(N):
        w = smoothed[f]
        if np.any(np.isnan(w)):
            continue
        pts = w.reshape(1, 1, 3).astype(np.float64)
        pL, _ = _cv2.projectPoints(pts, rvec_I, tvec_0, K1, dist1)
        pR, _ = _cv2.projectPoints(pts, rvec_R, T, K2, dist2)
        pL = pL.reshape(2); pR = pR.reshape(2)
        if not np.isnan(emp_disp_w):
            avg_y = 0.5 * (pL[1] + pR[1])
            pL[1] = avg_y + 0.5 * emp_disp_w
            pR[1] = avg_y - 0.5 * emp_disp_w
        out_L[f, 0] = pL
        out_R[f, 0] = pR

    return out_L, out_R


# ───────── AUC-delta camera-attribution score (HRnet Correct) ──────────

def _auc_at_pixel(hm: np.ndarray, ix: float, iy: float,
                  bbox: tuple, cluster_size: int) -> float:
    """Cluster AUC at image-coord ``(ix, iy)`` on heatmap ``hm`` (H, W).
    Maps the pixel back to heatmap coords, then runs the same greedy
    8-connected cluster grow as :func:`_cluster_centroid_peak` (no spike
    rejection — this is a probe at a *predicted* position, we just want
    raw cluster strength wherever we land).  Returns 0.0 outside the
    heatmap or when the seed pixel is non-positive."""
    H, W = hm.shape
    x1, y1, x2, y2 = bbox
    bw = float(x2 - x1); bh = float(y2 - y1)
    if bw <= 0 or bh <= 0:
        return 0.0
    u = (float(ix) - x1) / bw
    v = (float(iy) - y1) / bh
    hx = int(round(u * W - 0.5))
    hy = int(round(v * H - 0.5))
    if hx < 0 or hx >= W or hy < 0 or hy >= H:
        return 0.0
    seed_v = float(hm[hy, hx])
    if seed_v <= 0:
        return 0.0
    cluster_size = max(1, int(cluster_size))
    if cluster_size == 1:
        return seed_v
    in_cluster = np.zeros((H, W), dtype=bool)
    in_cluster[hy, hx] = True
    cluster_pixels = [(hy, hx)]
    wsum = seed_v
    NEIGH = ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1))
    while len(cluster_pixels) < cluster_size:
        best_val = -1.0
        best_pos = None
        for cy, cx in cluster_pixels:
            for dy, dx in NEIGH:
                ny, nx = cy + dy, cx + dx
                if not (0 <= ny < H and 0 <= nx < W):
                    continue
                if in_cluster[ny, nx]:
                    continue
                vv = float(hm[ny, nx])
                if vv > best_val:
                    best_val = vv
                    best_pos = (ny, nx)
        if best_pos is None:
            break
        in_cluster[best_pos] = True
        cluster_pixels.append(best_pos)
        wsum += float(hm[best_pos[0], best_pos[1]])
    return wsum


def _z_median_violation_mask(
    peaks_L: np.ndarray, peaks_R: np.ndarray,
    calib: dict, max_dev_mm: float,
) -> np.ndarray | None:
    """Per-(frame, joint, camera) bool: True for every frame of joints
    whose median Z deviates from the hand-wide median-of-medians by more
    than ``max_dev_mm``.  Anchored on the median of all joint medians so
    one bad joint can't drag the anchor.  Returns None when the slider
    is 0 or there's no usable Z signal.

    Use as a hard filter OR'd into the z_outlier error mask: catches
    sustained-contamination cases (>50% of one joint's frames bad) where
    the per-joint-distribution z_outlier detector is blind because the
    contamination has poisoned its own baseline.
    """
    from .calibration import triangulate_points
    if max_dev_mm <= 0:
        return None
    N, J, _ = peaks_L.shape
    z_meds = np.full(J, np.nan)
    for j in range(J):
        p3d = triangulate_points(peaks_L[:, j], peaks_R[:, j], calib).astype(np.float64)
        z = p3d[:, 2]
        valid = ~np.isnan(z)
        if int(valid.sum()) >= 5:
            z_meds[j] = float(np.median(z[valid]))
    valid_meds = ~np.isnan(z_meds)
    if int(valid_meds.sum()) < 5:
        return None
    anchor = float(np.median(z_meds[valid_meds]))
    out = np.zeros((N, J, 2), dtype=bool)
    for j in range(J):
        if np.isnan(z_meds[j]):
            continue
        if abs(z_meds[j] - anchor) > max_dev_mm:
            out[:, j, :] = True
    return out if out.any() else None


def _auc_delta_per_camera(
    peaks_L: np.ndarray, peaks_R: np.ndarray,
    auc_L: np.ndarray, auc_R: np.ndarray,
    heatmaps_L: np.ndarray, heatmaps_R: np.ndarray,
    bbox_L: tuple, bbox_R: tuple,
    calib: dict, cluster_size: int,
    return_predicted: bool = False,
):
    """Per-(frame, joint, camera) ΔAUC = AUC at predicted-correction pixel
    minus AUC at current pixel.

    For each (f, j) with a valid pair:
      * Hypothesis "blame L": lift R's pixel onto the joint's robust
        median Z-plane, project that 3D point into camera L → ``L_pred``;
        sample cluster AUC at ``L_pred`` on heatmap L → ``auc_L_pred``.
      * Hypothesis "blame R": symmetric.

    Returns ``(N, J, 2)`` float32 with cell ``[f, j, c] = auc_c_pred −
    auc_c_current``.  Higher Δ on a camera means moving that camera to
    its predicted position improves heatmap evidence the most → that
    camera is the more likely "bad" side.

    Used as the per-camera ``confidence``-equivalent attribution signal
    in HRnet Correct (replaces raw cluster AUC).

    When ``return_predicted=True``, also returns ``(pred_L, pred_R)`` in
    image coords (each ``(N, J, 2)``, NaN when the prediction is
    unavailable for that frame/joint).  Used by the UI to draw an empty
    circle at the predicted-corrected position so the user can see what
    the AUC delta represents.
    """
    import cv2 as _cv2
    from .calibration import triangulate_points

    N, J, _ = peaks_L.shape
    out = np.zeros((N, J, 2), dtype=np.float32)
    pred_L_out = np.full((N, J, 2), np.nan, dtype=np.float32) if return_predicted else None
    pred_R_out = np.full((N, J, 2), np.nan, dtype=np.float32) if return_predicted else None

    K1 = np.asarray(calib["K1"], dtype=np.float64)
    K2 = np.asarray(calib["K2"], dtype=np.float64)
    dist1 = np.asarray(calib["dist1"], dtype=np.float64)
    dist2 = np.asarray(calib["dist2"], dtype=np.float64)
    R = np.asarray(calib["R"], dtype=np.float64)
    T = np.asarray(calib["T"], dtype=np.float64).reshape(3)
    Rt = R.T
    rvec_I = np.zeros((3, 1)); tvec_0 = np.zeros((3, 1))
    rvec_R = _cv2.Rodrigues(R)[0]; tvec_R = T.reshape(3, 1)

    def _undistort(p, K, dist):
        p = np.asarray(p, dtype=np.float64).reshape(1, 1, 2)
        return _cv2.undistortPoints(p, K, dist).reshape(2)

    # Per-joint robust median Z over the trial.
    z_med = np.full(J, np.nan, dtype=np.float64)
    for j in range(J):
        p3d = triangulate_points(peaks_L[:, j], peaks_R[:, j], calib).astype(np.float64)
        z = p3d[:, 2]
        m = ~np.isnan(z)
        if int(m.sum()) >= 5:
            z_med[j] = float(np.median(z[m]))

    # Empirical y-disparity per joint (median observed ``y_L − y_R``).
    # Replaces the calibration-predicted disparity, which is biased by a
    # ~17 px extrinsic Ty error invisible to triangulation but fatal to
    # one-way reprojection.  The corrector uses the same anchor.
    emp_y_disp = np.full(J, np.nan, dtype=np.float64)
    for j in range(J):
        m = ~np.isnan(peaks_L[:, j, 1]) & ~np.isnan(peaks_R[:, j, 1])
        if int(m.sum()) >= 5:
            emp_y_disp[j] = float(np.median(peaks_L[m, j, 1] - peaks_R[m, j, 1]))

    from .hrnet import _bbox_at
    for f in range(N):
        bb_L_f = _bbox_at(bbox_L, f)
        bb_R_f = _bbox_at(bbox_R, f)
        for j in range(J):
            zt = z_med[j]
            if np.isnan(zt):
                continue
            pL = peaks_L[f, j]; pR = peaks_R[f, j]
            if np.any(np.isnan(pL)) or np.any(np.isnan(pR)):
                continue

            edisp = emp_y_disp[j]

            # Hypothesis "R correct" → lift R onto Z-plane → project to L
            u_n, v_n = _undistort(pR, K2, dist2)
            ray = np.array([u_n, v_n, 1.0])
            a = (Rt @ ray)[2]
            b = (Rt @ T)[2]
            if abs(a) > 1e-9:
                lam = (zt + b) / a
                X_world_R = Rt @ (lam * ray - T)
                pts = X_world_R.reshape(1, 1, 3).astype(np.float64)
                p, _ = _cv2.projectPoints(pts, rvec_I, tvec_0, K1, dist1)
                L_pred = p.reshape(2)
                if not np.isnan(edisp):
                    L_pred[1] = float(pR[1]) + edisp
                auc_L_pred = _auc_at_pixel(
                    heatmaps_L[f, j], L_pred[0], L_pred[1], bb_L_f, cluster_size)
                out[f, j, 0] = auc_L_pred - float(auc_L[f, j])
                if return_predicted:
                    pred_L_out[f, j] = L_pred

            # Hypothesis "L correct" → lift L onto Z-plane → project to R
            u_n, v_n = _undistort(pL, K1, dist1)
            X_world_L = np.array([u_n * zt, v_n * zt, zt])
            pts = X_world_L.reshape(1, 1, 3).astype(np.float64)
            p, _ = _cv2.projectPoints(pts, rvec_R, tvec_R, K2, dist2)
            R_pred = p.reshape(2)
            if not np.isnan(edisp):
                R_pred[1] = float(pL[1]) - edisp
            auc_R_pred = _auc_at_pixel(
                heatmaps_R[f, j], R_pred[0], R_pred[1], bb_R_f, cluster_size)
            out[f, j, 1] = auc_R_pred - float(auc_R[f, j])
            if return_predicted:
                pred_R_out[f, j] = R_pred

    if return_predicted:
        return out, pred_L_out, pred_R_out
    return out


# ───────────────────── Y/Z-correct preview (live overlay) ────────────────

# Tiny per-process cache for the live correction preview keyed by
# (subject, trial, cluster_size, spike_support, edge_margin).  Lets the
# UI re-threshold in milliseconds while the user drags a Y/Z slider —
# scoring is the slow part and doesn't depend on the slider thresholds.
_PREVIEW_CACHE: dict = {}
_PREVIEW_CACHE_MAX = 4


def _preview_cache_put(key, value):
    if len(_PREVIEW_CACHE) >= _PREVIEW_CACHE_MAX:
        _PREVIEW_CACHE.pop(next(iter(_PREVIEW_CACHE)))
    _PREVIEW_CACHE[key] = value


def compute_yzc_preview(
    subject_name: str, trial_stem: str,
    cluster_size: int = 1,
    spike_support: float = 0.0,
    edge_margin: int = 0,
    yzc_y_disp: float = 0.0,
    yzc_z_outlier: float = 0.0,
    yzc_attr_jump: float = 0.0,
    yzc_attr_auc: float = 0.0,
    yzc_z_median_mm: float = 0.0,
) -> dict:
    """Score Y/Z-correct error detections + per-camera attribution at the
    user's current Y/Z slider settings.  Returns lightweight per-(frame,
    joint, camera) flag arrays for live overlay rendering on the Peaks
    sub-stage.
    """
    from .skeleton_data import _load_trial_calibration, load_angle_priors, JOINT_NAMES
    from .skeleton_v3 import _load_hrnet_heatmaps
    from .video import build_trial_map
    from .mp_error_detection import (
        compute_scores_hrnet, _errors_for_factor, _combined_attr_per_cam,
        MIN_SIGMA_3D,
    )
    from .calibration import triangulate_points

    trials = build_trial_map(subject_name)
    trial = next((t for t in trials if t["trial_name"] == trial_stem), None)
    if trial is None:
        raise ValueError(f"Trial {trial_stem} not found")
    N = int(trial["frame_count"])
    start = int(trial.get("start_frame", 0))

    cache_key = (subject_name, trial_stem,
                 int(cluster_size), float(spike_support), int(edge_margin))
    cached = _PREVIEW_CACHE.get(cache_key)
    # Invalidate cache entries that pre-date pred_L/pred_R so users with
    # an in-memory cached score from before this change still see the
    # predicted-position black circles.
    if cached is not None and ("pred_L" not in cached or "peaks_L" not in cached):
        cached = None
    if cached is None:
        hm_L, bbox_L, hm_R, bbox_R = _load_hrnet_heatmaps(
            subject_name, trial_stem, start, N)
        if hm_L is None or hm_R is None:
            return {"available": False, "reason": "no_heatmaps"}
        peaks_L, auc_L = _cluster_centroid_peaks_with_auc(
            hm_L, bbox_L, int(cluster_size),
            spike_support=float(spike_support), edge_margin=int(edge_margin))
        peaks_R, auc_R = _cluster_centroid_peaks_with_auc(
            hm_R, bbox_R, int(cluster_size),
            spike_support=float(spike_support), edge_margin=int(edge_margin))
        calib = _load_trial_calibration(subject_name, trial_stem)
        if calib is None:
            return {"available": False, "reason": "no_calibration"}
        priors = load_angle_priors().get("joints", [])
        # AUC-delta attribution: higher Δ on a camera → moving that
        # camera improves the cluster AUC the most → that camera is the
        # likely bad one.  Also capture the predicted-corrected pixel
        # positions so the UI can show where each camera would move to.
        auc_delta, pred_L, pred_R = _auc_delta_per_camera(
            peaks_L, peaks_R, auc_L, auc_R, hm_L, hm_R,
            bbox_L, bbox_R, calib, int(cluster_size),
            return_predicted=True)
        det, attr = compute_scores_hrnet(
            peaks_L, peaks_R, auc_L, auc_R, calib, priors,
            factors={"y_disp", "z_outlier"},
            conf_attr_override=auc_delta)
        cached = {"det": det, "attr": attr,
                  "pred_L": pred_L, "pred_R": pred_R,
                  "peaks_L": peaks_L, "peaks_R": peaks_R, "calib": calib}
        _preview_cache_put(cache_key, cached)

    det = cached["det"]; attr = cached["attr"]
    det_w = {"y_disp": float(yzc_y_disp), "z_outlier": float(yzc_z_outlier)}
    attr_w = {"jump_2d": float(yzc_attr_jump), "confidence": float(yzc_attr_auc)}
    y_err = _errors_for_factor(det, attr, det_w, attr_w, "y_disp", winner_take_all=True)
    z_err = _errors_for_factor(det, attr, det_w, attr_w, "z_outlier", winner_take_all=True)
    # Hard filter: OR in any joint whose median Z is far from the
    # hand-wide consensus.  Catches sustained-contamination cases the
    # per-joint distribution detector misses.
    if yzc_z_median_mm > 0:
        peaks_L_c = cached["peaks_L"]; peaks_R_c = cached["peaks_R"]
        zm_mask = _z_median_violation_mask(
            peaks_L_c, peaks_R_c, cached["calib"], float(yzc_z_median_mm))
        if zm_mask is not None:
            z_err = zm_mask if z_err is None else (z_err | zm_mask)
    attr_per_cam = _combined_attr_per_cam(attr, attr_w)

    def _sparse_bool(arr):
        """Sparse-encode a (N, J, 2) bool array as a flat list of
        ``[frame, joint, cam]`` triples — typical preview has <5% cells
        True so this is 20–50× smaller than the dense JSON form."""
        if arr is None:
            return None
        idx = np.where(arr)
        if len(idx[0]) == 0:
            return []
        return [[int(f), int(j), int(c)]
                for f, j, c in zip(idx[0], idx[1], idx[2])]

    def _xy_grid(arr):
        """Sparse-encode a (N, J, 2) image-coord array as a flat list of
        ``[frame, joint, x, y]`` rows — skip NaN cells."""
        if arr is None:
            return None
        out = []
        for f in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                x, y = float(arr[f, j, 0]), float(arr[f, j, 1])
                if np.isnan(x) or np.isnan(y):
                    continue
                out.append([int(f), int(j), round(x, 1), round(y, 1)])
        return out

    # Per-joint Z-outlier display-space bounds — drawn as horizontal
    # reference lines on Pos: <Joint> Z plots when the user has the
    # Corrections checkbox on.  Mirrors the corrector's threshold logic:
    #   z_bound = med ± score_thresh × sigma
    # where score_thresh is the (1−slider) quantile of |z−med|/sigma over
    # clean frames and sigma = max(MAD × 1.4826, MIN_SIGMA_3D).  Mapped
    # into the same display space as _compute_joint_positions
    # (z − nanmean + 100).
    z_bounds = {}
    if cached.get("calib") is not None:
        peaks_L_c = cached["peaks_L"]; peaks_R_c = cached["peaks_R"]
        calib_c = cached["calib"]
        s = max(0.0, min(1.0, float(yzc_z_outlier)))
        for j in range(peaks_L_c.shape[1]):
            p3d = triangulate_points(peaks_L_c[:, j], peaks_R_c[:, j], calib_c).astype(np.float64)
            z = p3d[:, 2]
            valid = ~np.isnan(z)
            if int(valid.sum()) < 5:
                continue
            z_clean = z[valid]
            med = float(np.median(z_clean))
            mad = float(np.median(np.abs(z_clean - med)))
            sigma = max(mad * 1.4826, MIN_SIGMA_3D)
            if s > 0:
                raw_scores = np.abs(z_clean - med) / sigma
                score_thresh = float(np.quantile(raw_scores, max(0.0, 1.0 - s)))
            else:
                score_thresh = 6.0
            z_lo = med - score_thresh * sigma
            z_hi = med + score_thresh * sigma
            mean_z = float(np.nanmean(z))
            if not np.isfinite(mean_z):
                continue
            jname = JOINT_NAMES[j] if j < len(JOINT_NAMES) else f"J{j}"
            z_bounds[jname] = {
                "lo":  z_lo - mean_z + 100.0,
                "hi":  z_hi - mean_z + 100.0,
                "med": med  - mean_z + 100.0,
            }

    # Predicted-corrected positions are only needed by the UI when the
    # Cluster-AUC attribution slider is on (they drive the empty-black
    # circles).  Skip serialising them otherwise — saves ~N×J floats.
    include_pred = float(yzc_attr_auc) > 0
    return {
        "available": True,
        # Sparse: list of [frame, joint, cam] triples where flagged.
        "y_disp_errors":    _sparse_bool(y_err),
        "z_outlier_errors": _sparse_bool(z_err),
        # Sparse: list of [frame, joint, x, y].
        "pred_L":           _xy_grid(cached.get("pred_L")) if include_pred else None,
        "pred_R":           _xy_grid(cached.get("pred_R")) if include_pred else None,
        "z_outlier_bounds": z_bounds,
        "n_frames":         int(N),
    }


# ─────────────────────────────── Driver ─────────────────────────────────────

def run_hrnet_fit(
    heatmaps_L: np.ndarray, heatmaps_R: np.ndarray,
    bbox_L: tuple, bbox_R: tuple,
    cluster_size: int = 1,
    w_hm: float = 1.0,
    w_disp: float = 1.0,
    w_temp: float = 1.0,
    overlap_px: float = 8.0,
    spike_support: float = 0.0,
    edge_margin: int = 0,
    w_anchor: float = 1.0,
    yzc_y_disp: float = 0.0,
    yzc_z_outlier: float = 0.0,
    yzc_attr_jump: float = 0.0,
    yzc_attr_auc: float = 0.0,
    yzc_z_median_mm: float = 0.0,
    zsm_z_jump: float = 0.0,
    zsm_smooth_window: int = 15,
    bone_thresh_mm: float = 0.0,
    bone_K: int = 8,
    w_bone: float = 1.0,
    wrist_smooth_window: int = 5,
    calib: dict | None = None,
    angle_priors: list | None = None,
    progress_callback=None,
    cancel_event=None,
) -> dict[str, Any]:
    """Run cluster-centroid + joint stereo Hungarian.

    Returns a dict with:
      ``peaks_centroid_L``  (N, 21, 2)  — Stage-1 cluster centroids
      ``peaks_centroid_R``  (N, 21, 2)
      ``peaks_hungarian_L`` (N, 21, 2)  — Stage-2 stereo Hungarian
      ``peaks_hungarian_R`` (N, 21, 2)
      ``empirical_y_disp``  (21,)       — per-joint median y-disparity
                                          from cluster-centroid pairs
    """
    N, J, H, W = heatmaps_L.shape
    assert heatmaps_R.shape == heatmaps_L.shape, \
        "OS and OD heatmaps must have the same shape"

    # ── Stage 1: cluster centroids (with per-(frame,joint) cluster AUC) ──
    if progress_callback: progress_callback(2)
    peaks_c_L, auc_L = _cluster_centroid_peaks_with_auc(
        heatmaps_L, bbox_L, cluster_size,
        spike_support=spike_support, edge_margin=edge_margin)
    if progress_callback: progress_callback(8)
    peaks_c_R, auc_R = _cluster_centroid_peaks_with_auc(
        heatmaps_R, bbox_R, cluster_size,
        spike_support=spike_support, edge_margin=edge_margin)
    if progress_callback: progress_callback(12)

    # ── Stage 1b/c: Y/Z-correct + Z-smooth on the centroid peaks ──────
    # Reuses mp_error_detection's correctors with cluster-AUC standing in
    # for MP confidence.  The HRnet-mismatch attribution term is dropped
    # (it would be a feedback loop on the peaks we're correcting).
    peaks_yzc_L = peaks_c_L.copy()
    peaks_yzc_R = peaks_c_R.copy()
    peaks_zsm_L = peaks_c_L.copy()
    peaks_zsm_R = peaks_c_R.copy()
    try:
        if calib is not None and (yzc_y_disp > 0 or yzc_z_outlier > 0 or zsm_z_jump > 0):
            from .mp_error_detection import (
                compute_scores_hrnet, _errors_for_factor,
                _combined_attr_per_cam, correct_yz_from_errors,
                correct_z_from_errors,
            )
            priors = angle_priors or []
            # Y/Z-correct stage
            if yzc_y_disp > 0 or yzc_z_outlier > 0 or yzc_z_median_mm > 0:
                auc_delta = _auc_delta_per_camera(
                    peaks_yzc_L, peaks_yzc_R, auc_L, auc_R,
                    heatmaps_L, heatmaps_R,
                    bbox_L, bbox_R, calib, int(cluster_size))
                det, attr = compute_scores_hrnet(
                    peaks_yzc_L, peaks_yzc_R, auc_L, auc_R, calib, priors,
                    factors={"y_disp", "z_outlier"},
                    conf_attr_override=auc_delta)
                det_w = {"y_disp": float(yzc_y_disp),
                         "z_outlier": float(yzc_z_outlier)}
                attr_w: dict = {"jump_2d": float(yzc_attr_jump),
                                "confidence": float(yzc_attr_auc)}
                y_err = _errors_for_factor(det, attr, det_w, attr_w, "y_disp", winner_take_all=True)
                z_err = _errors_for_factor(det, attr, det_w, attr_w, "z_outlier", winner_take_all=True)
                # Hard "median Z must be in range" filter — OR'd into z_err
                if yzc_z_median_mm > 0:
                    zm_mask = _z_median_violation_mask(
                        peaks_yzc_L, peaks_yzc_R, calib, float(yzc_z_median_mm))
                    if zm_mask is not None:
                        z_err = zm_mask if z_err is None else (z_err | zm_mask)
                if y_err is not None or z_err is not None:
                    attr_per_cam = _combined_attr_per_cam(attr, attr_w)
                    peaks_yzc_L, peaks_yzc_R, _n = correct_yz_from_errors(
                        peaks_yzc_L, peaks_yzc_R, y_err, z_err, calib,
                        jump_2d_scores=attr.get("jump_2d"),
                        attr_per_cam=attr_per_cam,
                        z_outlier_slider=float(yzc_z_outlier))
            peaks_zsm_L = peaks_yzc_L.copy()
            peaks_zsm_R = peaks_yzc_R.copy()
            # Z-smooth stage (z_jump correction)
            if zsm_z_jump > 0:
                auc_delta2 = _auc_delta_per_camera(
                    peaks_zsm_L, peaks_zsm_R, auc_L, auc_R,
                    heatmaps_L, heatmaps_R,
                    bbox_L, bbox_R, calib, int(cluster_size))
                det2, attr2 = compute_scores_hrnet(
                    peaks_zsm_L, peaks_zsm_R, auc_L, auc_R, calib, priors,
                    factors={"z_jump"},
                    conf_attr_override=auc_delta2)
                det_w2 = {"z_jump": float(zsm_z_jump)}
                zj_err = _errors_for_factor(det2, attr2, det_w2, {}, "z_jump", winner_take_all=True)
                if zj_err is not None:
                    peaks_zsm_L, peaks_zsm_R, _n = correct_z_from_errors(
                        peaks_zsm_L, peaks_zsm_R, zj_err, calib,
                        attr2.get("jump_2d"),
                        window=int(zsm_smooth_window))
    except Exception as e:
        logger.warning(f"HRnet Y/Z-correct/Z-smooth failed (non-fatal): {e}")
        peaks_yzc_L = peaks_c_L.copy(); peaks_yzc_R = peaks_c_R.copy()
        peaks_zsm_L = peaks_c_L.copy(); peaks_zsm_R = peaks_c_R.copy()

    if progress_callback: progress_callback(15)

    # Stereo-Hungarian anchors on the Z-smoothed peaks (so the Hungarian
    # only swaps off a corrected centroid when heatmap evidence is clearly
    # better at the same stereo location in both cameras).
    emp_disp = _per_joint_y_disparity(peaks_zsm_L, peaks_zsm_R)

    # Per-joint global heatmap max — divisor for the cost-cube heatmap term.
    h_max = np.zeros(J, dtype=np.float32)
    for j in range(J):
        h_max[j] = max(float(heatmaps_L[:, j].max()),
                       float(heatmaps_R[:, j].max()))

    # ── Stage 2: bone-length-driven single-joint re-pick ──────────────
    # Walks parent → child.  For any frame whose triangulated bone to
    # parent deviates from the per-bone median by > bone_thresh_mm, brute
    # forces the top-K NMS peak pairs (OS × OD) and chooses the pair
    # whose triangulated 3D best satisfies the bone-length target while
    # still sitting on solid heatmap support.
    if bone_thresh_mm > 0 and calib is not None:
        out_L, out_R = _bone_length_repick(
            peaks_zsm_L, peaks_zsm_R,
            heatmaps_L, heatmaps_R,
            bbox_L, bbox_R, calib,
            threshold_mm=float(bone_thresh_mm),
            w_bone=float(w_bone),
            w_heatmap=float(w_hm),
            K_candidates=int(bone_K),
            nms_radius_px=float(overlap_px) if overlap_px > 0 else 6.0,
        )
    else:
        out_L = peaks_zsm_L.copy()
        out_R = peaks_zsm_R.copy()

    # ── Wrist reconstruction from constant wrist-MCP bone lengths ─────
    # Wrist-MCP bones are excluded from bone-length re-pick.  Instead the
    # wrist 3D is solved per-frame as the position satisfying all 5
    # wrist-MCP target lengths (medians from Z-smooth), temporally
    # smoothed, then re-projected into both cameras.  Always runs when
    # calibration is available — bypassed gracefully otherwise.
    if calib is not None:
        try:
            out_L, out_R = _wrist_constant_bone_reconstruction(
                out_L, out_R, peaks_zsm_L, peaks_zsm_R, calib,
                smooth_window=int(wrist_smooth_window))
        except Exception as _e:
            logger.warning(f"Wrist constant-bone reconstruction skipped: {_e}")

    if progress_callback: progress_callback(98)

    return {
        "peaks_centroid_L": peaks_c_L,
        "peaks_centroid_R": peaks_c_R,
        "peaks_yzc_L":      peaks_yzc_L,
        "peaks_yzc_R":      peaks_yzc_R,
        "peaks_zsmooth_L":  peaks_zsm_L,
        "peaks_zsmooth_R":  peaks_zsm_R,
        "peaks_hungarian_L": out_L,
        "peaks_hungarian_R": out_R,
        "empirical_y_disp": emp_disp,
        "auc_L": auc_L,
        "auc_R": auc_R,
    }


# ───────────────────────── Trial-level driver ────────────────────────────

def run_hrnet_fit_for_trial(
    subject_name: str, trial_stem: str,
    cluster_size: int = 1,
    w_hm: float = 1.0,
    w_disp: float = 1.0,
    w_temp: float = 1.0,
    overlap_px: float = 8.0,
    spike_support: float = 0.0,
    edge_margin: int = 0,
    w_anchor: float = 1.0,
    yzc_y_disp: float = 0.0,
    yzc_z_outlier: float = 0.0,
    yzc_attr_jump: float = 0.0,
    yzc_attr_auc: float = 0.0,
    yzc_z_median_mm: float = 0.0,
    zsm_z_jump: float = 0.0,
    zsm_smooth_window: int = 15,
    bone_thresh_mm: float = 0.0,
    bone_K: int = 8,
    w_bone: float = 1.0,
    wrist_smooth_window: int = 5,
    progress_callback=None,
    cancel_event=None,
) -> dict[str, Any]:
    """Run the HRnet Fit pipeline for one trial and persist results to
    ``hrnet_peak_assignments.json``.  Existing legacy fields
    (``peaks_L`` / ``peaks_R`` / ``peaks_*_raw``) are preserved.

    Returns ``{n_frames, n_joints, written: bool, params: {...}}``.
    """
    import json as _json
    from .skeleton_data import _skeleton_dir, JOINT_NAMES, _load_trial_calibration, load_angle_priors
    from .skeleton_v3 import _load_hrnet_heatmaps
    from .video import build_trial_map

    trials = build_trial_map(subject_name)
    trial = next((t for t in trials if t["trial_name"] == trial_stem), None)
    if trial is None:
        raise ValueError(f"Trial {trial_stem} not found for {subject_name}")
    N = int(trial["frame_count"])
    start = int(trial.get("start_frame", 0))

    hm_L, bbox_L, hm_R, bbox_R = _load_hrnet_heatmaps(
        subject_name, trial_stem, start, N)
    if hm_L is None or hm_R is None or bbox_L is None or bbox_R is None:
        raise ValueError("HRnet heatmaps + crop bboxes are required for HRnet Fit")

    if progress_callback: progress_callback(1)

    # Calibration + angle priors are needed only when one of the
    # Y/Z-correct or Z-smooth thresholds is non-zero.  Loading is cheap
    # so we always pass them in — the pipeline skips the corrector when
    # all three thresholds are 0.
    calib = _load_trial_calibration(subject_name, trial_stem)
    priors = load_angle_priors().get("joints", [])

    result = run_hrnet_fit(
        hm_L, hm_R, bbox_L, bbox_R,
        cluster_size=int(cluster_size),
        w_hm=float(w_hm), w_disp=float(w_disp), w_temp=float(w_temp),
        overlap_px=float(overlap_px),
        spike_support=float(spike_support),
        edge_margin=int(edge_margin),
        w_anchor=float(w_anchor),
        yzc_y_disp=float(yzc_y_disp),
        yzc_z_outlier=float(yzc_z_outlier),
        yzc_attr_jump=float(yzc_attr_jump),
        yzc_attr_auc=float(yzc_attr_auc),
        yzc_z_median_mm=float(yzc_z_median_mm),
        zsm_z_jump=float(zsm_z_jump),
        zsm_smooth_window=int(zsm_smooth_window),
        bone_thresh_mm=float(bone_thresh_mm),
        bone_K=int(bone_K),
        w_bone=float(w_bone),
        wrist_smooth_window=int(wrist_smooth_window),
        calib=calib,
        angle_priors=priors,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
    )

    if cancel_event is not None and cancel_event.is_set():
        return {"n_frames": N, "n_joints": 21, "written": False,
                "cancelled": True, "params": {}}

    # ── Merge into hrnet_peak_assignments.json (preserve legacy fields) ──
    skeleton_trial_dir = _skeleton_dir(subject_name) / trial_stem
    peak_path = skeleton_trial_dir / "hrnet_peak_assignments.json"
    existing = {}
    if peak_path.exists():
        try:
            existing = _json.loads(peak_path.read_text())
        except Exception:
            existing = {}

    def _pack(arr: np.ndarray) -> dict:
        out: dict[str, list] = {}
        for j in range(21):
            out[JOINT_NAMES[j]] = [
                None if np.isnan(arr[f, j, 0])
                else [round(float(arr[f, j, 0]), 1),
                      round(float(arr[f, j, 1]), 1)]
                for f in range(arr.shape[0])
            ]
        return out

    existing.setdefault("subject", subject_name)
    existing.setdefault("trial", trial_stem)
    existing.setdefault("n_frames", int(N))
    existing.setdefault("bbox_L", list(bbox_L))
    existing.setdefault("bbox_R", list(bbox_R))
    existing.setdefault("joint_names", JOINT_NAMES)
    existing["peaks_centroid_L"]  = _pack(result["peaks_centroid_L"])
    existing["peaks_centroid_R"]  = _pack(result["peaks_centroid_R"])
    existing["peaks_yzc_L"]       = _pack(result["peaks_yzc_L"])
    existing["peaks_yzc_R"]       = _pack(result["peaks_yzc_R"])
    existing["peaks_zsmooth_L"]   = _pack(result["peaks_zsmooth_L"])
    existing["peaks_zsmooth_R"]   = _pack(result["peaks_zsmooth_R"])
    existing["peaks_hungarian_L"] = _pack(result["peaks_hungarian_L"])
    existing["peaks_hungarian_R"] = _pack(result["peaks_hungarian_R"])
    existing["hrnet_fit_params"] = {
        "cluster_size": int(cluster_size),
        "w_hm": float(w_hm),
        "w_disp": float(w_disp),
        "w_temp": float(w_temp),
        "overlap_px": float(overlap_px),
        "spike_support": float(spike_support),
        "edge_margin": int(edge_margin),
        "w_anchor": float(w_anchor),
        "yzc_y_disp": float(yzc_y_disp),
        "yzc_z_outlier": float(yzc_z_outlier),
        "yzc_attr_jump": float(yzc_attr_jump),
        "yzc_attr_auc": float(yzc_attr_auc),
        "yzc_z_median_mm": float(yzc_z_median_mm),
        "zsm_z_jump": float(zsm_z_jump),
        "zsm_smooth_window": int(zsm_smooth_window),
        "bone_thresh_mm": float(bone_thresh_mm),
        "bone_K": int(bone_K),
        "w_bone": float(w_bone),
        "wrist_smooth_window": int(wrist_smooth_window),
    }
    existing["empirical_y_disp_per_joint"] = [
        None if np.isnan(v) else float(v)
        for v in result["empirical_y_disp"].tolist()
    ]

    peak_path.write_text(_json.dumps(existing, separators=(",", ":")))
    if progress_callback: progress_callback(100)

    return {
        "n_frames": int(N), "n_joints": 21, "written": True,
        "params": existing["hrnet_fit_params"],
    }

