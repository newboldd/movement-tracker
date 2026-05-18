"""Skeleton 3D hand model data loading, projection, and serving.

Loads skeleton_v3.npz, mediapipe.pkl, heatmaps, and calibration for each trial.
Projects 3D Skeleton joints to 2D camera coordinates.  Computes distance traces.
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

# Joint angle definitions: (parent, joint, child) for flexion, (j1a, j1b, j2a, j2b) for abduction
FLEX_ANGLE_OPTIONS: list[tuple[str, int, int, int]] = [
    ("Flex: Thumb CMC", 0, 1, 2), ("Flex: Thumb MCP", 1, 2, 3), ("Flex: Thumb IP", 2, 3, 4),
    ("Flex: Index MCP", 0, 5, 6), ("Flex: Index PIP", 5, 6, 7), ("Flex: Index DIP", 6, 7, 8),
    ("Flex: Middle MCP", 0, 9, 10), ("Flex: Middle PIP", 9, 10, 11), ("Flex: Middle DIP", 10, 11, 12),
    ("Flex: Ring MCP", 0, 13, 14), ("Flex: Ring PIP", 13, 14, 15), ("Flex: Ring DIP", 14, 15, 16),
    ("Flex: Pinky MCP", 0, 17, 18), ("Flex: Pinky PIP", 17, 18, 19), ("Flex: Pinky DIP", 18, 19, 20),
]
ABD_ANGLE_OPTIONS: list[tuple[str, int, int, int, int]] = [
    ("Abd: Thumb-Index", 1, 2, 5, 6),
    ("Abd: Index-Middle", 5, 6, 9, 10),
    ("Abd: Middle-Ring", 9, 10, 13, 14),
    ("Abd: Ring-Pinky", 13, 14, 17, 18),
]

# Maps joint index to its finger's metacarpal anchor (proximal, distal).
# All joints in a finger share the same flex/ext plane, defined once at the metacarpal level.
JOINT_FINGER_ANCHOR: dict[int, tuple[int, int]] = {
    1: (0, 1), 2: (0, 1), 3: (0, 1),        # thumb   → wrist→T_CMC
    5: (0, 5), 6: (0, 5), 7: (0, 5),        # index   → wrist→I_MCP
    9: (0, 9), 10: (0, 9), 11: (0, 9),      # middle  → wrist→M_MCP
    13: (0, 13), 14: (0, 13), 15: (0, 13),  # ring    → wrist→R_MCP
    17: (0, 17), 18: (0, 17), 19: (0, 17),  # pinky   → wrist→P_MCP
}

# Finger spread pairs: angle between wrist→MCP vectors for adjacent fingers
SPREAD_PAIRS: list[tuple[str, int, int]] = [
    ("Spread 1",  2,  5),   # Thumb MCP → Index MCP
    ("Spread 2",  5,  9),   # Index → Middle
    ("Spread 3",  9, 13),   # Middle → Ring
    ("Spread 4", 13, 17),   # Ring → Pinky
]

DISTANCE_OPTIONS: dict[str, tuple[int, int]] = {
    "Thumb-Index Aperture": (4, 8),
    "Thumb: CMC-MCP (1-2)": (1, 2),
    "Thumb: MCP-IP (2-3)": (2, 3),
    "Thumb: IP-Tip (3-4)": (3, 4),
    "Index: MCP-PIP (5-6)": (5, 6),
    "Index: PIP-DIP (6-7)": (6, 7),
    "Index: DIP-Tip (7-8)": (7, 8),
    "Middle: MCP-PIP (9-10)": (9, 10),
    "Middle: PIP-DIP (10-11)": (10, 11),
    "Middle: DIP-Tip (11-12)": (11, 12),
    "Ring: MCP-PIP (13-14)": (13, 14),
    "Ring: PIP-DIP (14-15)": (14, 15),
    "Ring: DIP-Tip (15-16)": (15, 16),
    "Pinky: MCP-PIP (17-18)": (17, 18),
    "Pinky: PIP-DIP (18-19)": (18, 19),
    "Pinky: DIP-Tip (19-20)": (19, 20),
    "Wrist-Thumb base (0-1)": (0, 1),
    "Wrist-Index base (0-5)": (0, 5),
    "Wrist-Middle base (0-9)": (0, 9),
    "Wrist-Ring base (0-13)": (0, 13),
    "Wrist-Pinky base (0-17)": (0, 17),
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

    Vectorised: ``np.round`` + single ``np.isnan`` + ``tolist`` then a sparse
    pass over NaN positions only.  ~30× faster than the prior per-coordinate
    loop while still rounding to 2 decimals to keep wire size small.
    """
    if arr is None or arr.size == 0:
        return []
    nan_mask = np.isnan(arr).any(axis=-1)
    rounded = np.round(arr.astype(np.float64), 2)
    out = rounded.tolist()
    if nan_mask.any():
        nz = np.argwhere(nan_mask)
        for f, j in nz:
            out[int(f)][int(j)] = None
    return out


def _array_to_list(arr: np.ndarray) -> list:
    """Convert 1D numpy array to list, NaN → None.  Rounded to 4 decimals."""
    if arr is None or arr.size == 0:
        return []
    a = np.round(np.asarray(arr, dtype=np.float64), 4)
    nan_mask = np.isnan(a)
    out = a.tolist()
    if nan_mask.any():
        for i in np.where(nan_mask)[0]:
            out[int(i)] = None
    return out


def _normalize(v: np.ndarray) -> np.ndarray:
    """Normalize (N,3) array row-wise; rows with near-zero norm become NaN."""
    n = np.linalg.norm(v, axis=1, keepdims=True)
    mask = n[:, 0] > 1e-6
    return np.where(mask[:, None], v / np.where(n > 1e-6, n, 1.0), np.nan)


def _compute_angles(joints_3d: np.ndarray) -> dict[str, list]:
    """Compute per-joint flex/abd angles using palm-normal decomposition.

    For each joint triplet (parent, joint, child) in FLEX_ANGLE_OPTIONS, produces:
      - "Flex: X": 0=straight, negative=flexion, positive=hyperextension
      - "Abd: X":  0=straight, positive=abduction, negative=adduction

    Both use atan2 decomposition against the incoming bone direction,
    so both are zero when the outgoing bone is collinear with the incoming bone.
    """
    N = joints_3d.shape[0]
    angles = {}

    # Palm normal: average of cross-products from all 4 adjacent MCP pairs.
    # Using all 5 palm landmarks makes the normal robust to depth errors in any
    # single joint (previously pinky MCP depth noise shifted the whole plane).
    w      = joints_3d[:, 0]
    e_idx  = joints_3d[:, 5]  - w   # wrist → index MCP
    e_mid  = joints_3d[:, 9]  - w   # wrist → middle MCP
    e_ring = joints_3d[:, 13] - w   # wrist → ring MCP
    e_pnk  = joints_3d[:, 17] - w   # wrist → pinky MCP
    # Four cross-product estimates, going radial→ulnar
    n_im = _normalize(np.cross(e_idx,  e_mid))
    n_mr = _normalize(np.cross(e_mid,  e_ring))
    n_rp = _normalize(np.cross(e_ring, e_pnk))
    n_ip = _normalize(np.cross(e_idx,  e_pnk))   # legacy estimate for sign reference
    # Sign-align all to n_ip so the average doesn't cancel
    def _align(n, ref):
        d = np.sum(n * ref, axis=1, keepdims=True)
        return np.where(d >= 0, n, -n)
    palm_n     = _normalize(n_ip + _align(n_im, n_ip) + _align(n_mr, n_ip) + _align(n_rp, n_ip))
    valid_palm = ~np.any(np.isnan(palm_n), axis=1)

    # Thumb reference: direction from wrist toward pinky MCP.
    # This becomes the dorsal axis (after projecting ⊥ b_in), so
    # flex_axis = cross(dorsal, b_in) points toward pinky MCP.
    pinky_dir   = _normalize(e_pnk)
    thumb_ref   = pinky_dir
    valid_thumb = ~np.any(np.isnan(thumb_ref), axis=1)
    THUMB_JOINTS = {1, 2, 3}

    # MCP local dorsal axes: cross product of adjacent inter-MCP segments.
    # More robust than the global palm normal for defining the flex/ext plane.
    # Each MCP's dorsal axis is perpendicular to the two inter-MCP segments meeting there.
    # For each MCP joint, compute a local dorsal axis from adjacent inter-MCP segments.
    # Sign is determined ONCE from the median geometry to avoid per-frame flipping.
    MCP_NEIGHBORS = {
        5:  (1, 9),    # I_MCP: between T_CMC and M_MCP
        9:  (5, 13),   # M_MCP: between I_MCP and R_MCP
        13: (9, 17),   # R_MCP: between M_MCP and P_MCP
    }
    mcp_local_ref = {}  # j -> (ref_vector, valid_mask)
    for mcp_j, (na, nb) in MCP_NEIGHBORS.items():
        va = _normalize(joints_3d[:, na] - joints_3d[:, mcp_j])
        vb = _normalize(joints_3d[:, nb] - joints_3d[:, mcp_j])
        local_n = _normalize(np.cross(va, vb))
        # Determine sign ONCE from median palm normal (stable across all frames).
        # Align OPPOSITE to palm_n: palm_n points dorsal, we want palmar.
        dots = np.sum(local_n * palm_n, axis=1)
        if np.any(~np.isnan(dots)):
            median_dot = np.nanmedian(dots)
            if median_dot > 0:
                local_n = -local_n
        v = ~np.any(np.isnan(local_n), axis=1)
        mcp_local_ref[mcp_j] = (local_n, v)
    # Pinky MCP: only one inter-MCP neighbor (R_MCP), use metacarpal direction as 2nd vector
    va_pnk = _normalize(joints_3d[:, 13] - joints_3d[:, 17])
    vb_pnk = _normalize(joints_3d[:, 0]  - joints_3d[:, 17])
    local_n_pnk = _normalize(np.cross(va_pnk, vb_pnk))
    dots_pnk = np.sum(local_n_pnk * palm_n, axis=1)
    if np.any(~np.isnan(dots_pnk)):
        median_dot_pnk = np.nanmedian(dots_pnk)
        if median_dot_pnk > 0:
            local_n_pnk = -local_n_pnk
    mcp_local_ref[17] = (local_n_pnk, ~np.any(np.isnan(local_n_pnk), axis=1))

    # Flex-propagated reference: for PIP/DIP joints, rotate the parent's dorsal axis
    # by the parent's flexion angle, using only the dorsal component of b_in_child
    # (invariant to abduction) as sin(θ).
    #
    # sin(θ) = -dot(b_in_child, da_parent): abduction-invariant measure of parent flex.
    #   In pure flex θ: b_in_child = cos(θ)*b_in_parent - sin(θ)*da_parent
    #   → dot(b_in_child, da_parent) = -sin(θ) regardless of any simultaneous abduction.
    # ref_child = sin(θ)*b_in_parent + cos(θ)*da_parent  (Rodrigues rotation of da_parent)
    # Then project ⊥ b_in_child as usual.
    joint_da_cache: dict[int, tuple] = {}  # j -> (dorsal_axis, b_in, valid)

    for name, p, j, c in FLEX_ANGLE_OPTIONS:
        flex_vals = np.full(N, np.nan)
        abd_vals  = np.full(N, np.nan)

        b_in  = _normalize(joints_3d[:, j] - joints_3d[:, p])
        b_out = _normalize(joints_3d[:, c] - joints_3d[:, j])

        is_thumb   = j in THUMB_JOINTS

        if p == 0 and j in mcp_local_ref:
            # MCP joints: use local dorsal axis from adjacent inter-MCP segments
            ref, valid_ref = mcp_local_ref[j]
        elif p == 0:
            # Thumb CMC or other root joint: use thumb/palm reference
            ref       = thumb_ref if is_thumb else palm_n
            valid_ref = valid_thumb if is_thumb else valid_palm
        elif p in joint_da_cache:
            da_par, bi_par, v_par = joint_da_cache[p]
            sin_t = np.clip(-np.sum(b_in * da_par, axis=1, keepdims=True), -1.0, 1.0)
            cos_t = np.sqrt(np.maximum(0.0, 1.0 - sin_t ** 2))
            ref       = sin_t * bi_par + cos_t * da_par
            valid_ref = v_par
        else:
            ref       = root_ref
            valid_ref = root_valid

        dorsal_axis = _normalize(ref - np.sum(ref * b_in, axis=1, keepdims=True) * b_in)
        flex_axis   = _normalize(np.cross(dorsal_axis, b_in))

        valid = (valid_ref
                 & ~np.any(np.isnan(b_in),        axis=1)
                 & ~np.any(np.isnan(b_out),        axis=1)
                 & ~np.any(np.isnan(dorsal_axis),  axis=1)
                 & ~np.any(np.isnan(flex_axis),    axis=1))

        if valid.any():
            bo = b_out[valid]
            da = dorsal_axis[valid]
            fa = flex_axis[valid]
            bi = b_in[valid]

            # Flex: angle in the dorsal/palmar plane.
            # dorsal_axis points palmar in typical 3D joint coordinate frames, so
            # negate its component so that flexion (bone toward palm) → negative,
            # extension/hyperextension → positive.
            fv = np.degrees(np.arctan2(
                -np.sum(bo * da, axis=1),
                np.sum(bo * bi, axis=1),
            ))
            flex_vals[valid] = fv

            # b_flex_end: where the outgoing bone would point after pure flexion only.
            # Negate da term to match the negated sign convention above.
            fr = np.radians(fv)
            bfe = np.cos(fr)[:, None] * bi - np.sin(fr)[:, None] * da
            bfe_len = np.linalg.norm(bfe, axis=1, keepdims=True)
            good = bfe_len[:, 0] > 1e-6
            bfe[good] /= bfe_len[good]
            bfe[~good] = np.nan

            # Abd: perpendicular to flex — from b_flex_end toward flex_axis
            # This guarantees the two decomposition axes are orthogonal.
            av = np.full(len(fv), np.nan)
            if good.any():
                av[good] = np.degrees(np.arctan2(
                    np.sum(bo[good] * fa[good], axis=1),
                    np.sum(bo[good] * bfe[good], axis=1),
                ))
            abd_vals[valid] = av

        # Cache dorsal_axis and b_in so children (PIP→DIP) can use the propagated ref
        joint_da_cache[j] = (dorsal_axis, b_in, valid)

        angles[name] = _array_to_list(flex_vals)
        angles[name.replace('Flex:', 'Abd:')] = _array_to_list(abd_vals)

    return angles


def _compute_wrist_angles(hand_3d: np.ndarray, elbow_3d: np.ndarray) -> dict[str, list]:
    """Compute wrist flex/abd angles from elbow→wrist→middle_MCP.

    Uses the same palm-normal decomposition as finger joints.
    b_in = normalize(wrist - elbow)  (forearm direction, toward hand)
    b_out = normalize(middle_MCP - wrist)  (hand direction)
    """
    N = hand_3d.shape[0]
    flex_vals = np.full(N, np.nan)
    abd_vals = np.full(N, np.nan)

    wrist = hand_3d[:, 0]
    mid_mcp = hand_3d[:, 9]

    b_in = _normalize(wrist - elbow_3d)       # forearm toward hand
    b_out = _normalize(mid_mcp - wrist)       # hand direction

    # Palm normal (same robust 4-cross-product average as _compute_angles)
    e_idx = hand_3d[:, 5] - wrist
    e_mid = hand_3d[:, 9] - wrist
    e_ring = hand_3d[:, 13] - wrist
    e_pnk = hand_3d[:, 17] - wrist
    n_im = _normalize(np.cross(e_idx, e_mid))
    n_mr = _normalize(np.cross(e_mid, e_ring))
    n_rp = _normalize(np.cross(e_ring, e_pnk))
    n_ip = _normalize(np.cross(e_idx, e_pnk))

    def _align(n, ref):
        d = np.sum(n * ref, axis=1, keepdims=True)
        return np.where(d >= 0, n, -n)

    palm_n = _normalize(n_ip + _align(n_im, n_ip) + _align(n_mr, n_ip) + _align(n_rp, n_ip))

    # Dorsal axis: palm normal projected ⊥ b_in
    dorsal = _normalize(palm_n - np.sum(palm_n * b_in, axis=1, keepdims=True) * b_in)
    flex_axis = _normalize(np.cross(dorsal, b_in))

    valid = (~np.any(np.isnan(b_in), axis=1)
             & ~np.any(np.isnan(b_out), axis=1)
             & ~np.any(np.isnan(dorsal), axis=1)
             & ~np.any(np.isnan(flex_axis), axis=1))

    if valid.any():
        bo = b_out[valid]
        da = dorsal[valid]
        fa = flex_axis[valid]
        bi = b_in[valid]

        fv = np.degrees(np.arctan2(-np.sum(bo * da, axis=1), np.sum(bo * bi, axis=1)))
        flex_vals[valid] = fv

        fr = np.radians(fv)
        bfe = np.cos(fr)[:, None] * bi - np.sin(fr)[:, None] * da
        bfe_len = np.linalg.norm(bfe, axis=1, keepdims=True)
        good = bfe_len[:, 0] > 1e-6
        bfe[good] /= bfe_len[good]
        bfe[~good] = np.nan

        av = np.full(len(fv), np.nan)
        if good.any():
            av[good] = np.degrees(np.arctan2(
                np.sum(bo[good] * fa[good], axis=1),
                np.sum(bo[good] * bfe[good], axis=1),
            ))
        abd_vals[valid] = av

    return {
        "Flex: Wrist": _array_to_list(flex_vals),
        "Abd: Wrist": _array_to_list(abd_vals),
    }


def _compute_spreads(joints_3d: np.ndarray) -> dict[str, list]:
    """Spread angle between adjacent finger pairs (angle between wrist→MCP vectors)."""
    N = joints_3d.shape[0]
    spreads = {}
    w = joints_3d[:, 0]
    for name, j1, j2 in SPREAD_PAIRS:
        v1 = _normalize(joints_3d[:, j1] - w)
        v2 = _normalize(joints_3d[:, j2] - w)
        dot = np.clip(np.sum(v1 * v2, axis=1), -1.0, 1.0)
        valid = (~np.any(np.isnan(v1), axis=1)) & (~np.any(np.isnan(v2), axis=1))
        vals = np.full(N, np.nan)
        vals[valid] = np.degrees(np.arccos(dot[valid]))
        spreads[name] = _array_to_list(vals)
    return spreads


def _compute_wrist_coords(joints_3d: np.ndarray) -> dict[str, list]:
    """Wrist 3D position (x, y) and mean-centered z (+100) per frame from joint 0.

    Z is mean-subtracted and offset by 100 so relative depth changes are visible
    within the normal distance y-axis range.
    """
    z = joints_3d[:, 0, 2].copy()
    if np.any(~np.isnan(z)):
        z_mean = np.nanmean(z)
        if np.isfinite(z_mean):
            z = z - z_mean + 100.0
    return {
        "Wrist X": _array_to_list(joints_3d[:, 0, 0]),
        "Wrist Y": _array_to_list(joints_3d[:, 0, 1]),
        "Wrist Z": _array_to_list(z),
    }


MCP_DIST_PAIRS = [
    # (name, ja, jb) — simple point-to-point distance
    ("MCP: Index-Middle", 5,  9),
    ("MCP: Middle-Ring",  9,  13),
    ("MCP: Ring-Pinky",   13, 17),
]


def _point_to_segment_dist(P: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Distance from point P to the nearest point on segment A→B.

    All inputs are (N, 3).  Returns (N,) distances.
    """
    AB = B - A                                              # (N, 3)
    AP = P - A                                              # (N, 3)
    t = np.einsum('ij,ij->i', AP, AB) / np.maximum(np.einsum('ij,ij->i', AB, AB), 1e-12)
    t = np.clip(t, 0.0, 1.0)                               # project onto segment
    closest = A + t[:, None] * AB                           # (N, 3)
    return np.linalg.norm(P - closest, axis=1)              # (N,)


def _compute_mcp_distances(joints_3d: np.ndarray) -> dict[str, list]:
    """Compute distances between adjacent MCP joints.

    Thumb-Index uses the distance from Thumb CMC (joint 1) to the
    nearest point on the index metacarpal segment (Wrist→I_MCP, joints 0→5).
    All others are simple point-to-point.
    """
    result = {}
    # Thumb-Index: point-to-segment
    d_ti = _point_to_segment_dist(joints_3d[:, 1], joints_3d[:, 0], joints_3d[:, 5])
    result["MCP: Thumb-Index"] = _array_to_list(d_ti)
    # Remaining pairs: point-to-point
    for name, ja, jb in MCP_DIST_PAIRS:
        d = np.linalg.norm(joints_3d[:, jb] - joints_3d[:, ja], axis=1)
        result[name] = _array_to_list(d)
    return result


MCP_KNUCKLE_ANGLES = [
    ("Knuckle: I-M-R", 5, 9, 13),   # angle at M_MCP between I_MCP→M_MCP and M_MCP→R_MCP
    ("Knuckle: M-R-P", 9, 13, 17),  # angle at R_MCP between M_MCP→R_MCP and R_MCP→P_MCP
]


def _compute_knuckle_angles(joints_3d: np.ndarray) -> dict[str, list]:
    """Compute angles between adjacent inter-MCP segments at the middle and ring MCP joints."""
    N = joints_3d.shape[0]
    result = {}
    for name, ja, jv, jb in MCP_KNUCKLE_ANGLES:
        va = joints_3d[:, ja] - joints_3d[:, jv]  # vertex → ja
        vb = joints_3d[:, jb] - joints_3d[:, jv]  # vertex → jb
        na = np.linalg.norm(va, axis=1, keepdims=True)
        nb = np.linalg.norm(vb, axis=1, keepdims=True)
        cos_ang = np.sum(va * vb, axis=1) / np.maximum(na[:, 0] * nb[:, 0], 1e-8)
        cos_ang = np.clip(cos_ang, -1, 1)
        ang = np.degrees(np.arccos(cos_ang))
        result[name] = _array_to_list(ang)
    return result


def _compute_joint_positions(joints_3d: np.ndarray) -> dict[str, list]:
    """De-meaned position traces for ALL 21 joints, each axis offset by +100mm.

    Keys: "Pos: {JointName} X", "Pos: {JointName} Y", "Pos: {JointName} Z"
    """
    result = {}
    for j in range(21):
        name = JOINT_NAMES[j] if j < len(JOINT_NAMES) else f"J{j}"
        for ax, ax_name in [(0, 'X'), (1, 'Y'), (2, 'Z')]:
            vals = joints_3d[:, j, ax].copy()
            if np.any(~np.isnan(vals)):
                mean_v = np.nanmean(vals)
                if np.isfinite(mean_v):
                    vals = vals - mean_v + 100.0
            result[f"Pos: {name} {ax_name}"] = _array_to_list(vals)
    return result


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


def _compute_distances_2d(joints_2d: np.ndarray) -> dict[str, list]:
    """Compute 2D pixel distance traces from (N,21,2) landmarks (fallback when no 3D)."""
    distances = {}
    for name, (ja, jb) in DISTANCE_OPTIONS.items():
        n = joints_2d.shape[0]
        dist = np.full(n, np.nan)
        valid = ~np.isnan(joints_2d[:, ja, 0]) & ~np.isnan(joints_2d[:, jb, 0])
        if valid.any():
            dist[valid] = np.linalg.norm(
                joints_2d[valid, ja, :] - joints_2d[valid, jb, :], axis=1
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

def _skeleton_dir(subject_name: str) -> Path:
    """Return path to skeleton data directory for a subject."""
    settings = get_settings()
    return settings.dlc_path / subject_name / "skeleton"


def list_v2_fit_history(subject_name: str, trial_stem: str) -> list[dict]:
    """List available previous skeleton v2 fits for a trial.

    Returns list of dicts with keys: slot (1-3), timestamp, params summary.
    """
    skeleton_trial_dir = _skeleton_dir(subject_name) / trial_stem
    history = []
    for i in range(1, 4):
        npz_path = skeleton_trial_dir / f"skeleton_v3_prev{i}.npz"
        params_path = skeleton_trial_dir / f"skeleton_v3_prev{i}_params.json"
        if not npz_path.exists():
            continue
        info = {"slot": i, "label": f"Run -{i}", "version": "v2"}
        if params_path.exists():
            try:
                params = json.loads(params_path.read_text())
                ts = params.get("timestamp", "")
                ver = params.get("version", "v2")
                # Map "corrections-stage*" to v3 short tag
                if isinstance(ver, str) and ver.startswith("corrections"):
                    ver = "v3"
                info["version"] = ver
                ver_tag = f" ({ver})" if ver else ""
                if ts:
                    # Format: "2024-04-15T10:30:00" → "Apr 15 10:30"
                    from datetime import datetime
                    try:
                        dt = datetime.fromisoformat(ts)
                        info["label"] = dt.strftime("%b %d %H:%M") + ver_tag
                    except ValueError:
                        info["label"] = f"Run -{i}{ver_tag}"
                else:
                    info["label"] = f"Run -{i}{ver_tag}"
                info["timestamp"] = ts
                p = params.get("params", {})
                info["params_summary"] = {k: p[k] for k in ("w_constraints", "w_smooth_z", "w_smooth_angles") if k in p}
            except Exception:
                pass
        history.append(info)
    return history


def load_v2_fit_history_slot(subject_name: str, trial_stem: str, slot: int, calib: dict | None = None) -> dict | None:
    """Load a historical skeleton v2 fit (slot 1-3) and compute projections + angles.

    Returns a dict with the same structure as the main skel_v2 data, or None.
    """
    if slot < 1 or slot > 3:
        return None
    skeleton_trial_dir = _skeleton_dir(subject_name) / trial_stem
    npz_path = skeleton_trial_dir / f"skeleton_v3_prev{slot}.npz"
    if not npz_path.exists():
        return None

    v2 = _load_skeleton_npz(str(npz_path))
    joints_3d = v2["joints_3d"]
    N = joints_3d.shape[0]
    fit_error_L = v2.get("fit_error_L", np.full(N, np.nan))
    fit_error_R = v2.get("fit_error_R", np.full(N, np.nan))

    # Set up projection helpers (used per stage below too).
    K1 = K2 = dist1 = dist2 = R = T = None
    if calib:
        K1, K2 = calib["K1"], calib["K2"]
        dist1, dist2 = calib["dist1"], calib["dist2"]
        R, T = calib["R"], calib["T"]
    R_eye = np.eye(3, dtype=np.float64)
    T_zero = np.zeros((3, 1), dtype=np.float64)

    def _proj_to_2d(joints):
        if joints is None or not calib:
            return np.full((N, 21, 2), np.nan), np.full((N, 21, 2), np.nan)
        return (_project_to_2d(joints, K1, dist1, R_eye, T_zero),
                _project_to_2d(joints, K2, dist2, R, T))

    # Final-stage 2D projections.
    proj_L, proj_R = _proj_to_2d(joints_3d)

    # Compute derived metrics for the final stage.
    distances = _compute_distances(joints_3d)
    distances.update(_compute_mcp_distances(joints_3d))
    angles = _compute_angles(joints_3d)
    spreads = _compute_spreads(joints_3d)
    positions = _compute_joint_positions(joints_3d)

    # Load params
    params_path = skeleton_trial_dir / f"skeleton_v3_prev{slot}_params.json"
    fit_params = None
    if params_path.exists():
        try:
            fit_params = json.loads(params_path.read_text())
        except Exception:
            pass

    # ── Per-stage snapshots (so the historical fit's stage rows can
    # show the same intermediate views as the live v3 fit). ──
    # Each tag matches the live-fit key suffix (sc, y, z, zs, hr, bc).
    # When the snapshot is missing or all-NaN (e.g. HRnet-snap step
    # was skipped, or this is a legacy fit pre-stereo-correct), the
    # corresponding fields are emitted as None and the frontend
    # falls back to the final-stage projection / 3D.
    stage_outputs: dict[str, object] = {}
    for tag, key in (
        ("sc", "joints_3d_after_sc"),
        ("y",  "joints_3d_after_y"),
        ("z",  "joints_3d_after_z"),
        ("zs", "joints_3d_after_zs"),
        ("hr", "joints_3d_after_hr"),
        ("bc", "joints_3d_after_bc"),
    ):
        arr = v2.get(key)
        has = arr is not None and np.any(~np.isnan(arr))
        if not has:
            stage_outputs[f"joints_3d_{tag}"] = None
            stage_outputs[f"proj_{tag}_L"] = None
            stage_outputs[f"proj_{tag}_R"] = None
            continue
        pL, pR = _proj_to_2d(arr)
        stage_outputs[f"joints_3d_{tag}"] = _points_to_list(arr)
        stage_outputs[f"proj_{tag}_L"] = _points_to_list(pL)
        stage_outputs[f"proj_{tag}_R"] = _points_to_list(pR)

    return {
        "proj_L": _points_to_list(proj_L),
        "proj_R": _points_to_list(proj_R),
        "joints_3d": _points_to_list(joints_3d),
        "fit_error_L": _array_to_list(fit_error_L),
        "fit_error_R": _array_to_list(fit_error_R),
        "distances": distances,
        "angles": angles,
        "spreads": spreads,
        "positions": positions,
        "fit_params": fit_params.get("params") if fit_params else None,
        "fit_constraints": fit_params.get("angle_constraints") if fit_params else None,
        **stage_outputs,
    }


def load_angle_priors() -> dict:
    """Load joint angle constraint priors.

    Checks for a user-customised file at ``{DATA_DIR}/custom_joint_angle_priors.json``
    first; falls back to the bundled defaults.  Returns the parsed JSON dict with
    ``flexion`` and ``abduction`` arrays.
    """
    from ..config import DATA_DIR
    custom_path = DATA_DIR / "custom_joint_angle_priors.json"
    if custom_path.exists():
        return json.loads(custom_path.read_text())
    default_path = Path(__file__).parent / "joint_angle_priors.json"
    return json.loads(default_path.read_text())


def list_skeleton_trials(subject_name: str) -> list[dict]:
    """List trials that have MediaPipe data and/or Skeleton fit results.

    Returns trials from the video trial map that have MediaPipe prelabels
    available.  Each entry includes ``has_skeleton_v1`` flag.
    """
    # Build video trial map for index alignment
    try:
        video_trials = build_trial_map(subject_name)
    except Exception:
        video_trials = []

    if not video_trials:
        return []

    # Check for MediaPipe prelabels (app's own data)
    from .mediapipe_prelabel import load_mediapipe_prelabels
    mp_data = load_mediapipe_prelabels(subject_name)

    # Check for mano_fit files
    skeleton_root = _skeleton_dir(subject_name)

    results = []
    for i, vt in enumerate(video_trials):
        trial_stem = vt["trial_name"]
        n_frames = vt["frame_count"]
        fps = vt["fps"]

        # Check if MediaPipe data exists for this trial
        has_mp = False
        if mp_data is not None:
            os_lm = mp_data.get("OS_landmarks")
            if os_lm is not None:
                # Check if any frames in this trial's range have valid data
                start = vt.get("start_frame", 0)
                end = min(start + n_frames, os_lm.shape[0])
                if end > start:
                    trial_slice = os_lm[start:end]
                    has_mp = np.any(~np.isnan(trial_slice[:, 0, 0]))

        # Check for skeleton_v3.npz or skeleton_v1.npz
        skeleton_trial_dir = skeleton_root / trial_stem
        has_skeleton_v1 = (
            (skeleton_trial_dir / "skeleton_v3.npz").exists()
            or (skeleton_trial_dir / "skeleton_v1.npz").exists()
        )
        has_heatmaps = (skeleton_trial_dir / "hrnet_w18_heatmaps.npz").exists()

        # Include every trial the subject's videos produce — even when no
        # MediaPipe/Skeleton data exists yet — so the viewer can show an empty
        # trial and the user can launch detection jobs from there.
        results.append({
            "trial_idx": i,
            "trial_stem": trial_stem,
            "n_frames": n_frames,
            "has_heatmaps": has_heatmaps,
            "has_skeleton_v1": has_skeleton_v1,
            "has_mp": bool(has_mp),
            "fps": fps,
        })

    return results


# ── Data loading ───────────────────────────────────────────────────────

@lru_cache(maxsize=8)
def _load_skeleton_npz(npz_path: str) -> dict:
    """Load and cache Skeleton fit npz.  Returns dict of numpy arrays."""
    data = np.load(npz_path, allow_pickle=True)
    return dict(data)


@lru_cache(maxsize=8)
def _load_mediapipe_pkl(pkl_path: str) -> dict:
    """Load and cache mediapipe pkl."""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _triangulate_hrnet_peaks(skeleton_trial_dir: Path, calib, N: int, kind: str = "refined",
                              peaks: dict | None = None):
    """Load HRNet peak JSON and triangulate per-frame 3D positions using
    the stereo calibration.  Returns the numpy (N, 21, 3) array, or None
    when peaks or calibration are missing.  ``kind`` selects which peak
    set written by the HRnet pipeline:
      * ``"centroid"``  — Stage-1 cluster-centroid peaks (HRnet Fit).
      * ``"yzc"``       — Y/Z-correct stage (HRnet Correct pipeline).
      * ``"zsmooth"``   — Z-smooth stage (HRnet Correct pipeline).
      * ``"hungarian"`` — Joint stereo Hungarian (HRnet Correct pipeline).
      * ``"raw"``       — legacy raw argmax (pre-HRnet-Fit pipeline).
      * ``"refined"``   — legacy MP-Hungarian Peak-Select (deprecated).

    Pass ``peaks`` to reuse a pre-parsed JSON (avoids re-reading +
    re-parsing the file once per kind — a measurable load-time win when
    the JSON contains all five stages × 2 cameras × 21 joints × N frames).
    """
    if calib is None:
        return None
    if peaks is None:
        peaks = _load_hrnet_peaks_json(skeleton_trial_dir)
    if not peaks:
        return None
    if kind == "raw":
        pL = peaks.get("peaks_L_raw") or {}
        pR = peaks.get("peaks_R_raw") or {}
    elif kind == "centroid":
        pL = peaks.get("peaks_centroid_L") or {}
        pR = peaks.get("peaks_centroid_R") or {}
    elif kind == "hungarian":
        pL = peaks.get("peaks_hungarian_L") or {}
        pR = peaks.get("peaks_hungarian_R") or {}
    elif kind == "yzc":
        pL = peaks.get("peaks_yzc_L") or {}
        pR = peaks.get("peaks_yzc_R") or {}
    elif kind == "zsmooth":
        pL = peaks.get("peaks_zsmooth_L") or {}
        pR = peaks.get("peaks_zsmooth_R") or {}
    else:  # legacy "refined"
        pL = peaks.get("peaks_L") or {}
        pR = peaks.get("peaks_R") or {}
    if not pL or not pR:
        return None
    pts_L = np.full((N, 21, 2), np.nan, dtype=np.float32)
    pts_R = np.full((N, 21, 2), np.nan, dtype=np.float32)
    for j in range(21):
        name = JOINT_NAMES[j] if j < len(JOINT_NAMES) else f"J{j}"
        fr_L = pL.get(name); fr_R = pR.get(name)
        for f in range(N):
            if fr_L and f < len(fr_L) and fr_L[f] is not None:
                pts_L[f, j] = fr_L[f]
            if fr_R and f < len(fr_R) and fr_R[f] is not None:
                pts_R[f, j] = fr_R[f]
    out = np.full((N, 21, 3), np.nan, dtype=np.float32)
    for j in range(21):
        out[:, j, :] = triangulate_points(pts_L[:, j], pts_R[:, j], calib).astype(np.float32)
    return out


def _load_hrnet_peaks_json(skeleton_trial_dir: Path) -> dict | None:
    """Load pre-computed HRNet peak assignments if available."""
    p = skeleton_trial_dir / "hrnet_peak_assignments.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        return {
            # Legacy MP-Hungarian + 90-percentile centroid (deprecated).
            "peaks_L": data.get("peaks_L", {}),
            "peaks_R": data.get("peaks_R", {}),
            # Legacy pure argmax.
            "peaks_L_raw": data.get("peaks_L_raw", {}),
            "peaks_R_raw": data.get("peaks_R_raw", {}),
            # New HRnet Fit pipeline outputs.
            "peaks_centroid_L":  data.get("peaks_centroid_L", {}),
            "peaks_centroid_R":  data.get("peaks_centroid_R", {}),
            "peaks_hungarian_L": data.get("peaks_hungarian_L", {}),
            "peaks_hungarian_R": data.get("peaks_hungarian_R", {}),
            # Y/Z-correct + Z-smooth substages (HRnet Correct pipeline).
            "peaks_yzc_L":       data.get("peaks_yzc_L", {}),
            "peaks_yzc_R":       data.get("peaks_yzc_R", {}),
            "peaks_zsmooth_L":   data.get("peaks_zsmooth_L", {}),
            "peaks_zsmooth_R":   data.get("peaks_zsmooth_R", {}),
            # Last-used HRnet Fit parameters (so the UI can restore sliders).
            "hrnet_fit_params":  data.get("hrnet_fit_params", None),
        }
    except Exception:
        return None


def _load_mp_errors_for_response(subject_name: str, trial_stem: str):
    """Return saved MP error matrix as a JSON-friendly nested list, or None."""
    from .mp_error_detection import load_saved_errors
    data = load_saved_errors(subject_name, trial_stem)
    if data is None:
        return None
    err = data["errors"]  # (N, 21, 2) bool
    return [[[int(bool(err[f, j, 0])), int(bool(err[f, j, 1]))]
             for j in range(err.shape[1])]
            for f in range(err.shape[0])]


def _load_trial_calibration(subject_name: str, trial_stem: str) -> dict | None:
    """Load calibration for a trial.

    Priority:
      1. Per-trial calibration.yaml in skeleton data dir
      2. Subject-level calibration from the app's calibration service
    """
    skeleton_trial_dir = _skeleton_dir(subject_name) / trial_stem
    local_calib = skeleton_trial_dir / "calibration.yaml"
    if local_calib.exists():
        try:
            return load_calibration(str(local_calib))
        except Exception as e:
            logger.warning(f"Failed to load trial calibration {local_calib}: {e}")

    calib = get_calibration_for_subject(subject_name)
    if calib is None:
        logger.debug(f"get_calibration_for_subject({subject_name!r}) returned None")
    return calib


def load_skeleton_trial_data(subject_name: str, trial_stem: str) -> dict[str, Any]:
    """Load all Skeleton viewer data for a single trial.

    Works in two modes:
    - Full mode: skeleton_v3.npz or skeleton_v1.npz exists → load Skeleton + MP
    - MP-only mode: no Skeleton fit → load MediaPipe from prelabels, return null Skeleton fields

    Returns a dict ready for JSON serialisation (numpy arrays converted to lists).
    """
    skeleton_trial_dir = _skeleton_dir(subject_name) / trial_stem

    # ── Determine frame count from video trial map ────────────
    N = 0
    fps = 30.0
    start_frame = 0
    try:
        trials = build_trial_map(subject_name)
        for t in trials:
            if t["trial_name"] == trial_stem:
                N = t["frame_count"]
                fps = t["fps"]
                start_frame = t.get("start_frame", 0)
                break
    except Exception:
        pass

    # ── Load Skeleton fit (if available) ───────────────────────────
    has_skeleton_fit = False
    # ── Load v1 skeleton fit (skeleton_v1.npz) ──────────────────────
    joints_3d = None
    fit_error_L = np.full(N, np.nan)
    fit_error_R = np.full(N, np.nan)
    mp_weights_L = None
    mp_weights_R = None

    v1_path = skeleton_trial_dir / "skeleton_v1.npz"
    v1_fit_params = None
    if v1_path.exists():
        v1 = _load_skeleton_npz(str(v1_path))
        joints_3d = v1["joints_3d"]
        if N == 0:
            N = joints_3d.shape[0]
        fit_error_L = v1.get("fit_error_L", np.full(N, np.nan))
        fit_error_R = v1.get("fit_error_R", np.full(N, np.nan))
        mp_weights_L = v1.get("mp_weights_L")
        mp_weights_R = v1.get("mp_weights_R")
        has_skeleton_fit = True
        # Load saved fit parameters
        v1_params_path = skeleton_trial_dir / "skeleton_v1_params.json"
        if v1_params_path.exists():
            try:
                v1_fit_params = json.loads(v1_params_path.read_text())
            except Exception:
                pass

    # ── Load v2 skeleton fit (skeleton_v3.npz) ─────────────────
    v2_joints_3d = None
    v2_fit_error_L = np.full(max(N, 1), np.nan)
    v2_fit_error_R = np.full(max(N, 1), np.nan)
    has_skel_v2 = False
    v2_fit_params = None

    v2_mp_L_corrected = None   # per-camera 2D from corrections-stage output
    v2_mp_R_corrected = None
    v2_mp_L_after_sc = None     # after stereo-correction (Step 0)
    v2_mp_R_after_sc = None
    v2_joints_3d_after_sc = None
    v2_stereo_blame = None      # (N, 21, 2) bool: which camera got its MP replaced
    v2_stereo_L_pts = None
    v2_stereo_R_pts = None
    v2_stereo_response = None
    v2_mp_L_after_y = None      # after Y-correction only (intermediate)
    v2_mp_R_after_y = None
    v2_joints_3d_after_y = None
    v2_mp_L_after_z = None      # after Y + Z-outlier (intermediate)
    v2_mp_R_after_z = None
    v2_joints_3d_after_z = None
    v2_mp_L_after_zs = None     # after Y + Z-outlier + Z-jump (pre-HRnet)
    v2_mp_R_after_zs = None
    v2_joints_3d_after_zs = None
    v2_mp_L_after_hr = None     # after HRnet snap + re-clean (pre-BL)
    v2_mp_R_after_hr = None
    v2_joints_3d_after_hr = None
    v2_mp_L_after_bc = None     # after bone-length correction (pre-BL-jump)
    v2_mp_R_after_bc = None
    v2_joints_3d_after_bc = None
    v2_hrnet_along_L = None
    v2_hrnet_perp_L  = None
    v2_hrnet_along_R = None
    v2_hrnet_perp_R  = None
    v2_hrnet_child   = None
    v2_path = skeleton_trial_dir / "skeleton_v3.npz"
    if v2_path.exists():
        v2 = _load_skeleton_npz(str(v2_path))
        v2_joints_3d = v2["joints_3d"]
        if N == 0:
            N = v2_joints_3d.shape[0]
        v2_fit_error_L = v2.get("fit_error_L", np.full(N, np.nan))
        v2_fit_error_R = v2.get("fit_error_R", np.full(N, np.nan))
        # Correction-stage outputs carry the 2D corrected MP positions
        # directly, so the Skeleton's 2D display uses them verbatim (no 3D
        # round-trip) and stays pixel-exact with the source MP on at least
        # one camera per joint.
        v2_mp_L_corrected = v2.get("mp_L_corrected")
        v2_mp_R_corrected = v2.get("mp_R_corrected")
        # After stereo-correction (Step 0 of v3 pipeline)
        v2_mp_L_after_sc = v2.get("mp_L_after_sc")
        v2_mp_R_after_sc = v2.get("mp_R_after_sc")
        v2_joints_3d_after_sc = v2.get("joints_3d_after_sc")
        v2_stereo_blame = v2.get("stereo_blame")
        v2_stereo_L_pts = v2.get("stereo_L_pts")
        v2_stereo_R_pts = v2.get("stereo_R_pts")
        v2_stereo_response = v2.get("stereo_response")
        # Legacy "after_y" snapshot — now equals the combined Y/Z output (kept for back-compat)
        v2_mp_L_after_y = v2.get("mp_L_after_y")
        v2_mp_R_after_y = v2.get("mp_R_after_y")
        v2_joints_3d_after_y = v2.get("joints_3d_after_y")
        # Intermediate after Y + Z-outlier — powers the 'z_correct' stage view
        v2_mp_L_after_z = v2.get("mp_L_after_z")
        v2_mp_R_after_z = v2.get("mp_R_after_z")
        v2_joints_3d_after_z = v2.get("joints_3d_after_z")
        # Intermediate after Y + Z-outlier + Z-jump — powers 'z_smooth' stage
        v2_mp_L_after_zs = v2.get("mp_L_after_zs")
        v2_mp_R_after_zs = v2.get("mp_R_after_zs")
        v2_joints_3d_after_zs = v2.get("joints_3d_after_zs")
        # Intermediate after HRnet snap + re-clean — powers 'hrnet_snap' stage
        v2_mp_L_after_hr = v2.get("mp_L_after_hr")
        v2_mp_R_after_hr = v2.get("mp_R_after_hr")
        v2_joints_3d_after_hr = v2.get("joints_3d_after_hr")
        # Intermediate after bone-length correction — powers 'bone_correct' stage
        v2_mp_L_after_bc = v2.get("mp_L_after_bc")
        v2_mp_R_after_bc = v2.get("mp_R_after_bc")
        v2_joints_3d_after_bc = v2.get("joints_3d_after_bc")
        v2_hrnet_along_L = v2.get("hrnet_along_L")
        v2_hrnet_perp_L  = v2.get("hrnet_perp_L")
        v2_hrnet_along_R = v2.get("hrnet_along_R")
        v2_hrnet_perp_R  = v2.get("hrnet_perp_R")
        v2_hrnet_child   = v2.get("hrnet_child")
        # Load saved fit parameters
        v2_params_path = skeleton_trial_dir / "skeleton_v3_params.json"
        if v2_params_path.exists():
            try:
                v2_fit_params = json.loads(v2_params_path.read_text())
            except Exception:
                pass
        has_skel_v2 = True

    # ── Load Skeleton v2 (legacy) fit ─────────────────────────
    legacy_joints_3d = None
    legacy_fit_params = None
    has_skel_legacy = False
    legacy_path = skeleton_trial_dir / "skeleton_v2.npz"
    if legacy_path.exists():
        leg = _load_skeleton_npz(str(legacy_path))
        legacy_joints_3d = leg["joints_3d"]
        if N == 0:
            N = legacy_joints_3d.shape[0]
        legacy_params_path = skeleton_trial_dir / "skeleton_v2_params.json"
        if legacy_params_path.exists():
            try:
                legacy_fit_params = json.loads(legacy_params_path.read_text())
            except Exception:
                pass
        has_skel_legacy = True

    if N == 0:
        raise FileNotFoundError(
            f"Cannot determine frame count for {subject_name}/{trial_stem}")

    # ── Load calibration (optional — needed for 3D but not 2D) ─
    calib = _load_trial_calibration(subject_name, trial_stem)
    has_calib = calib is not None

    if has_calib:
        K1, K2 = calib["K1"], calib["K2"]
        dist1, dist2 = calib["dist1"], calib["dist2"]
        R, T = calib["R"], calib["T"]
        logger.info(f"Loaded calibration for {subject_name}/{trial_stem}")
    else:
        logger.warning(f"No calibration for {subject_name}/{trial_stem} — 3D disabled")
        K1 = K2 = np.eye(3, dtype=np.float64)
        dist1 = dist2 = np.zeros((5, 1), dtype=np.float64)
        R = np.eye(3, dtype=np.float64)
        T = np.zeros((3, 1), dtype=np.float64)

    R_eye = np.eye(3, dtype=np.float64)
    T_zero = np.zeros((3, 1), dtype=np.float64)

    # ── Apply learned 2D offsets to 3D joints (before projection) ──
    if has_skeleton_fit and has_calib:
        offset_L = v1.get("offset_L")
        offset_R = v1.get("offset_R")
        # Convert the average of L/R 2D pixel offsets to a 3D world-space shift
        # using the left camera intrinsics and the average hand depth.
        if offset_L is not None or offset_R is not None:
            off_L = np.array(offset_L, dtype=np.float64).ravel()[:2] if offset_L is not None else np.zeros(2)
            off_R = np.array(offset_R, dtype=np.float64).ravel()[:2] if offset_R is not None else np.zeros(2)
            # Average the two camera offsets for a single 3D correction
            avg_off = (off_L + off_R) / 2.0
            # Compute average depth (Z in left camera frame) across all valid frames
            valid_mask = ~np.isnan(joints_3d[:, 0, 2])
            if valid_mask.any():
                avg_z = np.nanmean(joints_3d[valid_mask, :, 2])
                fx, fy = K1[0, 0], K1[1, 1]
                # 2D pixel offset → 3D world offset: dx_3d = dx_px * Z / fx
                dx_3d = avg_off[0] * avg_z / fx
                dy_3d = avg_off[1] * avg_z / fy
                joints_3d = joints_3d.copy()
                joints_3d[valid_mask, :, 0] += dx_3d
                joints_3d[valid_mask, :, 1] += dy_3d

    # ── Project Skeleton 3D→2D (if available) ──────────────────────
    if has_skeleton_fit and has_calib:
        skeleton_proj_L = _project_to_2d(joints_3d, K1, dist1, R_eye, T_zero)
        skeleton_proj_R = _project_to_2d(joints_3d, K2, dist2, R, T)
    else:
        skeleton_proj_L = np.full((N, 21, 2), np.nan)
        skeleton_proj_R = np.full((N, 21, 2), np.nan)
        if not has_skeleton_fit:
            joints_3d = np.full((N, 21, 3), np.nan)

    # ── Skeleton v2 2D: prefer directly saved corrected MP over 3D→2D ──
    # Correction stages save the 2D corrected MP positions alongside the
    # triangulated 3D.  Using them verbatim keeps the Skeleton's 2D Y
    # identical to the MP source on the unflagged camera (calibration
    # errors only affect the 3D view, not the 2D).
    if has_skel_v2:
        if v2_mp_L_corrected is not None and v2_mp_R_corrected is not None:
            v2_proj_L = np.array(v2_mp_L_corrected, dtype=np.float32)
            v2_proj_R = np.array(v2_mp_R_corrected, dtype=np.float32)
        elif has_calib:
            v2_proj_L = _project_to_2d(v2_joints_3d, K1, dist1, R_eye, T_zero)
            v2_proj_R = _project_to_2d(v2_joints_3d, K2, dist2, R, T)
        else:
            v2_proj_L = np.full((N, 21, 2), np.nan)
            v2_proj_R = np.full((N, 21, 2), np.nan)
    else:
        v2_proj_L = np.full((N, 21, 2), np.nan)
        v2_proj_R = np.full((N, 21, 2), np.nan)
        v2_joints_3d = np.full((N, 21, 3), np.nan)

    # ── Project Skeleton v2 (legacy) 3D→2D ─────────────────
    if has_skel_legacy and has_calib:
        legacy_proj_L = _project_to_2d(legacy_joints_3d, K1, dist1, R_eye, T_zero)
        legacy_proj_R = _project_to_2d(legacy_joints_3d, K2, dist2, R, T)
    else:
        legacy_proj_L = np.full((N, 21, 2), np.nan)
        legacy_proj_R = np.full((N, 21, 2), np.nan)
        if not has_skel_legacy:
            legacy_joints_3d = np.full((N, 21, 3), np.nan)

    # ── Load MediaPipe ─────────────────────────────────────────
    mp_tracked_L = np.full((N, 21, 2), np.nan)
    mp_tracked_R = np.full((N, 21, 2), np.nan)

    # Try skeleton-dir mediapipe.pkl first (legacy hand_tracking format)
    mp_path = skeleton_trial_dir / "mediapipe.pkl"
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
    else:
        # Fall back to app's own mediapipe_prelabels.npz
        from .mediapipe_prelabel import load_mediapipe_prelabels
        prelabels = load_mediapipe_prelabels(subject_name)
        if prelabels is not None:
            os_lm = prelabels.get("OS_landmarks")
            od_lm = prelabels.get("OD_landmarks")
            end = min(start_frame + N, os_lm.shape[0]) if os_lm is not None else start_frame
            if os_lm is not None and end > start_frame:
                n = end - start_frame
                mp_tracked_L[:n] = os_lm[start_frame:end]
            if od_lm is not None and end > start_frame:
                n = end - start_frame
                mp_tracked_R[:n] = od_lm[start_frame:end]


    # ── Triangulate MP to 3D (requires calibration) ─────────────
    mp_joints_3d = np.full((N, 21, 3), np.nan)
    if has_calib:
        for j in range(21):
            pts_L = mp_tracked_L[:, j, :]  # (N, 2)
            pts_R = mp_tracked_R[:, j, :]  # (N, 2)
            mp_joints_3d[:, j, :] = triangulate_points(pts_L, pts_R, calib)

    # ── Load reverse-pass MediaPipe (separate npz, optional) ───
    # Run MP with frames fed in reverse temporal order: tracker
    # enters cold-start frames already locked on.  Loaded as a
    # sibling layer to the forward MP labels so the Labels page
    # can show both side-by-side.
    reverse_tracked_L = np.full((N, 21, 2), np.nan)
    reverse_tracked_R = np.full((N, 21, 2), np.nan)
    from .mediapipe_prelabel import load_mediapipe_reverse_prelabels
    rev = load_mediapipe_reverse_prelabels(subject_name)
    if rev is not None:
        os_lm_r = rev.get("OS_landmarks")
        od_lm_r = rev.get("OD_landmarks")
        if os_lm_r is not None:
            end_r = min(start_frame + N, os_lm_r.shape[0])
            if end_r > start_frame:
                n = end_r - start_frame
                reverse_tracked_L[:n] = os_lm_r[start_frame:end_r]
        if od_lm_r is not None:
            end_r = min(start_frame + N, od_lm_r.shape[0])
            if end_r > start_frame:
                n = end_r - start_frame
                reverse_tracked_R[:n] = od_lm_r[start_frame:end_r]

    reverse_joints_3d = np.full((N, 21, 3), np.nan)
    if has_calib:
        for j in range(21):
            reverse_joints_3d[:, j, :] = triangulate_points(
                reverse_tracked_L[:, j, :], reverse_tracked_R[:, j, :], calib)

    # ── Load Vision (Apple Vision) landmarks ───────────────────
    vision_tracked_L = np.full((N, 21, 2), np.nan)
    vision_tracked_R = np.full((N, 21, 2), np.nan)
    vision_path = get_settings().dlc_path / subject_name / "vision_prelabels.npz"
    if vision_path.exists():
        try:
            v_data = np.load(str(vision_path))
            v_os = v_data.get("OS_landmarks")
            v_od = v_data.get("OD_landmarks")
            end = min(start_frame + N, v_os.shape[0]) if v_os is not None else start_frame
            if v_os is not None and end > start_frame:
                n = end - start_frame
                vision_tracked_L[:n] = v_os[start_frame:end]
            if v_od is not None and end > start_frame:
                n = end - start_frame
                vision_tracked_R[:n] = v_od[start_frame:end]
        except Exception as e:
            logger.warning(f"Failed to load vision_prelabels.npz: {e}")

    # ── Triangulate Vision to 3D (requires calibration) ────────
    vision_joints_3d = np.full((N, 21, 3), np.nan)
    if has_calib:
        for j in range(21):
            pts_L = vision_tracked_L[:, j, :]
            pts_R = vision_tracked_R[:, j, :]
            vision_joints_3d[:, j, :] = triangulate_points(pts_L, pts_R, calib)

    # ── Load Pose (body) landmarks ──────────────────────────────
    # Arm-chain indices from the 33-landmark pose model:
    # 11=L_shoulder, 12=R_shoulder, 13=L_elbow, 14=R_elbow,
    # 15=L_wrist, 16=R_wrist, 17=L_pinky, 18=R_pinky,
    # 19=L_index, 20=R_index, 21=L_thumb, 22=R_thumb
    POSE_ARM_INDICES = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    pose_arm_L = np.full((N, len(POSE_ARM_INDICES), 2), np.nan)
    pose_arm_R = np.full((N, len(POSE_ARM_INDICES), 2), np.nan)
    pose_arm_3d = np.full((N, len(POSE_ARM_INDICES), 3), np.nan)
    elbow_3d = np.full((N, 3), np.nan)
    pose_side = None  # 'left' or 'right' body side matching the tracked hand
    has_pose = False
    try:
        from .mediapipe_prelabel import load_pose_prelabels
        pose_data = load_pose_prelabels(subject_name)
        if pose_data is not None:
            os_pose = pose_data["OS_pose"]
            od_pose = pose_data["OD_pose"]
            end = min(start_frame + N, os_pose.shape[0])
            if end > start_frame:
                n = end - start_frame
                pose_arm_L[:n] = os_pose[start_frame:end][:, POSE_ARM_INDICES]
                pose_arm_R[:n] = od_pose[start_frame:end][:, POSE_ARM_INDICES]
                has_pose = True

            if has_calib and has_pose:
                # Triangulate all arm-chain landmarks to 3D
                for i, pidx in enumerate(POSE_ARM_INDICES):
                    pts_L = os_pose[start_frame:end][:, pidx]
                    pts_R = od_pose[start_frame:end][:, pidx]
                    pose_arm_3d[:n, i] = triangulate_points(pts_L, pts_R, calib)

                # Determine which body side matches the tracked hand by
                # comparing pose wrist 2D to hand wrist 2D in the OS camera.
                # Pose wrist left=idx 15 (arm_idx 4), right=idx 16 (arm_idx 5)
                hand_wrist_2d = mp_tracked_L[:, 0]  # (N, 2) — hand wrist in OS cam
                pose_wrist_L_2d = pose_arm_L[:, 4]  # left pose wrist in OS cam
                pose_wrist_R_2d = pose_arm_L[:, 5]  # right pose wrist in OS cam
                d_left = np.nanmedian(np.linalg.norm(hand_wrist_2d - pose_wrist_L_2d, axis=1))
                d_right = np.nanmedian(np.linalg.norm(hand_wrist_2d - pose_wrist_R_2d, axis=1))
                if not np.isnan(d_left) and not np.isnan(d_right):
                    pose_side = 'left' if d_left < d_right else 'right'
                elif not np.isnan(d_left):
                    pose_side = 'left'
                else:
                    pose_side = 'right'

                # Pick the matching elbow (arm_idx: L_elbow=2, R_elbow=3)
                elbow_arm_idx = 2 if pose_side == 'left' else 3
                elbow_3d[:] = pose_arm_3d[:, elbow_arm_idx]
                logger.info(f"  Pose side: {pose_side} (d_left={d_left:.1f}, d_right={d_right:.1f})")

                # Temporal median filter to smooth noisy elbow depth
                if np.any(~np.isnan(elbow_3d[:, 0])):
                    from scipy.ndimage import median_filter
                    for ax in range(3):
                        col = elbow_3d[:, ax]
                        valid = ~np.isnan(col)
                        if valid.sum() > 5:
                            col[valid] = median_filter(col[valid], size=5)
    except Exception as e:
        logger.warning(f"Failed to load pose data: {e}")

    # Project elbow to 2D
    elbow_proj_L = np.full((N, 2), np.nan)
    elbow_proj_R = np.full((N, 2), np.nan)
    if has_calib and np.any(~np.isnan(elbow_3d[:, 0])):
        rvec_L = cv2.Rodrigues(R_eye)[0]
        tvec_L = T_zero
        rvec_R = cv2.Rodrigues(R)[0]
        tvec_R = T.reshape(3, 1)
        for t in range(N):
            if np.isnan(elbow_3d[t, 0]):
                continue
            pt = elbow_3d[t:t+1].reshape(1, 1, 3).astype(np.float64)
            p2d_L, _ = cv2.projectPoints(pt, rvec_L, tvec_L, K1, dist1)
            elbow_proj_L[t] = p2d_L.reshape(2)
            p2d_R, _ = cv2.projectPoints(pt, rvec_R, tvec_R, K2, dist2)
            elbow_proj_R[t] = p2d_R.reshape(2)

    # ── Load DLC predictions (thumb+index only) ────────────────
    dlc_thumb_OS = np.full((N, 2), np.nan)
    dlc_thumb_OD = np.full((N, 2), np.nan)
    dlc_index_OS = np.full((N, 2), np.nan)
    dlc_index_OD = np.full((N, 2), np.nan)

    try:
        from .dlc_predictions import get_dlc_predictions_for_session
        settings = get_settings()
        dlc_data = get_dlc_predictions_for_session(subject_name)
        if dlc_data:
            cam_names = settings.camera_names
            cam_L, cam_R = cam_names[0], cam_names[1] if len(cam_names) > 1 else cam_names[0]
            for cam, thumb_arr, index_arr in [
                (cam_L, dlc_thumb_OS, dlc_index_OS),
                (cam_R, dlc_thumb_OD, dlc_index_OD),
            ]:
                cam_data = dlc_data.get(cam, {})
                for bp, target in [("thumb", thumb_arr), ("index", index_arr)]:
                    arr = cam_data.get(bp)
                    if arr:
                        for f_global, coord in enumerate(arr):
                            local = f_global - start_frame
                            if 0 <= local < N and coord is not None:
                                target[local] = coord
    except Exception as e:
        logger.warning(f"Failed to load DLC predictions: {e}")

    # ── Triangulate DLC to 3D (thumb=joint4, index=joint8) ──────
    dlc_3d_thumb = np.full((N, 3), np.nan)
    dlc_3d_index = np.full((N, 3), np.nan)
    if has_calib:
        dlc_3d_thumb[:] = triangulate_points(dlc_thumb_OS, dlc_thumb_OD, calib)
        dlc_3d_index[:] = triangulate_points(dlc_index_OS, dlc_index_OD, calib)

    # ── Compute DLC thumb-index distance ─────────────────────────
    dlc_ti_dist = np.full(N, np.nan)
    valid_dlc = ~np.isnan(dlc_3d_thumb[:, 0]) & ~np.isnan(dlc_3d_index[:, 0])
    if valid_dlc.any():
        dlc_ti_dist[valid_dlc] = np.linalg.norm(
            dlc_3d_thumb[valid_dlc] - dlc_3d_index[valid_dlc], axis=1
        )
    distances_dlc = {"Thumb-Index Aperture": _array_to_list(dlc_ti_dist)}

    # ── Compute distances ──────────────────────────────────────
    # Intermediate pipeline stages (may be None if not saved yet)
    has_skel_v2_sc = v2_joints_3d_after_sc is not None and np.any(~np.isnan(v2_joints_3d_after_sc))
    # When stereo-correct was skipped (Stereo-distance = 0 at bake), the
    # snapshot is all-NaN.  Drop so STAGE_CONFIGS falls back to MP.
    if not has_skel_v2_sc:
        v2_mp_L_after_sc = None
        v2_mp_R_after_sc = None
        v2_joints_3d_after_sc = None
    has_skel_v2_y = v2_joints_3d_after_y is not None and np.any(~np.isnan(v2_joints_3d_after_y))
    has_skel_v2_z = v2_joints_3d_after_z is not None and np.any(~np.isnan(v2_joints_3d_after_z))
    has_skel_v2_zs = v2_joints_3d_after_zs is not None and np.any(~np.isnan(v2_joints_3d_after_zs))
    has_skel_v2_hr = v2_joints_3d_after_hr is not None and np.any(~np.isnan(v2_joints_3d_after_hr))
    # When the HRnet snap step was skipped in run_correction_pipeline
    # (no peaks / threshold = 0), the snapshot arrays were written as
    # all-NaN.  Drop them entirely so the frontend's STAGE_CONFIGS
    # ``||`` fallback kicks in -- the HRnet-snap stage view then
    # falls back to the final Skeleton output instead of looking
    # identical to Z-smooth.
    if not has_skel_v2_hr:
        v2_mp_L_after_hr = None
        v2_mp_R_after_hr = None
        v2_joints_3d_after_hr = None
    has_skel_v2_bc = v2_joints_3d_after_bc is not None and np.any(~np.isnan(v2_joints_3d_after_bc))

    distances_mano = _compute_distances(joints_3d) if has_skeleton_fit else {}
    distances_skel_v2 = _compute_distances(v2_joints_3d) if has_skel_v2 else {}
    distances_skel_v2_sc = _compute_distances(v2_joints_3d_after_sc) if has_skel_v2_sc else {}
    distances_skel_v2_y = _compute_distances(v2_joints_3d_after_y) if has_skel_v2_y else {}
    distances_skel_v2_z = _compute_distances(v2_joints_3d_after_z) if has_skel_v2_z else {}
    distances_skel_v2_zs = _compute_distances(v2_joints_3d_after_zs) if has_skel_v2_zs else {}
    distances_skel_v2_hr = _compute_distances(v2_joints_3d_after_hr) if has_skel_v2_hr else {}
    distances_skel_v2_bc = _compute_distances(v2_joints_3d_after_bc) if has_skel_v2_bc else {}
    distances_skel_legacy = _compute_distances(legacy_joints_3d) if has_skel_legacy else {}
    distances_mp = _compute_distances(mp_joints_3d)
    distances_vision = _compute_distances(vision_joints_3d)
    # Parse the HRnet peaks JSON once and reuse for every kind below
    # (refined / raw / centroid / yzc / zsmooth / hungarian + the response
    # payload).  The JSON now contains all five HRnet Correct stages × 2
    # cameras × 21 joints × N frames — re-parsing it 6 times dominated
    # cold page load.
    hrnet_peaks_cached = _load_hrnet_peaks_json(skeleton_trial_dir)
    # HRNet heatmap-peak triangulated 3D — used as its own "model"
    hrnet_peaks_3d = _triangulate_hrnet_peaks(skeleton_trial_dir, calib, N, kind="refined", peaks=hrnet_peaks_cached)
    hrnet_peaks_raw_3d = _triangulate_hrnet_peaks(skeleton_trial_dir, calib, N, kind="raw",       peaks=hrnet_peaks_cached)
    # New HRnet Correct pipeline outputs (centroid → Y/Z-correct → Z-smooth → Hungarian).
    hrnet_centroid_3d  = _triangulate_hrnet_peaks(skeleton_trial_dir, calib, N, kind="centroid",  peaks=hrnet_peaks_cached)
    hrnet_yzc_3d       = _triangulate_hrnet_peaks(skeleton_trial_dir, calib, N, kind="yzc",       peaks=hrnet_peaks_cached)
    hrnet_zsmooth_3d   = _triangulate_hrnet_peaks(skeleton_trial_dir, calib, N, kind="zsmooth",   peaks=hrnet_peaks_cached)
    hrnet_hungarian_3d = _triangulate_hrnet_peaks(skeleton_trial_dir, calib, N, kind="hungarian", peaks=hrnet_peaks_cached)
    has_heatmap_3d = hrnet_peaks_3d is not None and np.any(~np.isnan(hrnet_peaks_3d))
    has_heatmap_raw_3d = hrnet_peaks_raw_3d is not None and np.any(~np.isnan(hrnet_peaks_raw_3d))
    has_centroid_3d  = hrnet_centroid_3d is not None and np.any(~np.isnan(hrnet_centroid_3d))
    has_yzc_3d       = hrnet_yzc_3d is not None and np.any(~np.isnan(hrnet_yzc_3d))
    has_zsmooth_3d   = hrnet_zsmooth_3d is not None and np.any(~np.isnan(hrnet_zsmooth_3d))
    has_hungarian_3d = hrnet_hungarian_3d is not None and np.any(~np.isnan(hrnet_hungarian_3d))
    distances_heatmap = _compute_distances(hrnet_peaks_3d) if has_heatmap_3d else {}
    if has_heatmap_3d: distances_heatmap.update(_compute_mcp_distances(hrnet_peaks_3d))
    # HRnet Fit pipeline outputs: cluster centroid + Stereo-Hungarian.
    distances_hrnet_centroid  = _compute_distances(hrnet_centroid_3d) if has_centroid_3d else {}
    if has_centroid_3d:  distances_hrnet_centroid.update(_compute_mcp_distances(hrnet_centroid_3d))
    distances_hrnet_yzc       = _compute_distances(hrnet_yzc_3d)      if has_yzc_3d      else {}
    if has_yzc_3d:       distances_hrnet_yzc.update(_compute_mcp_distances(hrnet_yzc_3d))
    distances_hrnet_zsmooth   = _compute_distances(hrnet_zsmooth_3d)  if has_zsmooth_3d  else {}
    if has_zsmooth_3d:   distances_hrnet_zsmooth.update(_compute_mcp_distances(hrnet_zsmooth_3d))
    distances_hrnet_hungarian = _compute_distances(hrnet_hungarian_3d) if has_hungarian_3d else {}
    if has_hungarian_3d: distances_hrnet_hungarian.update(_compute_mcp_distances(hrnet_hungarian_3d))
    # MCP inter-joint distances
    if has_skeleton_fit: distances_mano.update(_compute_mcp_distances(joints_3d))
    if has_skel_v2: distances_skel_v2.update(_compute_mcp_distances(v2_joints_3d))
    if has_skel_v2_sc: distances_skel_v2_sc.update(_compute_mcp_distances(v2_joints_3d_after_sc))
    if has_skel_v2_y: distances_skel_v2_y.update(_compute_mcp_distances(v2_joints_3d_after_y))
    if has_skel_v2_z: distances_skel_v2_z.update(_compute_mcp_distances(v2_joints_3d_after_z))
    if has_skel_v2_zs: distances_skel_v2_zs.update(_compute_mcp_distances(v2_joints_3d_after_zs))
    if has_skel_v2_hr: distances_skel_v2_hr.update(_compute_mcp_distances(v2_joints_3d_after_hr))
    if has_skel_v2_bc: distances_skel_v2_bc.update(_compute_mcp_distances(v2_joints_3d_after_bc))
    if has_skel_legacy: distances_skel_legacy.update(_compute_mcp_distances(legacy_joints_3d))
    distances_mp.update(_compute_mcp_distances(mp_joints_3d))
    distances_vision.update(_compute_mcp_distances(vision_joints_3d))

    # Compute joint angle traces
    angles_mano = _compute_angles(joints_3d) if has_skeleton_fit else {}
    angles_skel_v2 = _compute_angles(v2_joints_3d) if has_skel_v2 else {}
    angles_skel_v2_sc = _compute_angles(v2_joints_3d_after_sc) if has_skel_v2_sc else {}
    angles_skel_v2_y = _compute_angles(v2_joints_3d_after_y) if has_skel_v2_y else {}
    angles_skel_v2_z = _compute_angles(v2_joints_3d_after_z) if has_skel_v2_z else {}
    angles_skel_v2_zs = _compute_angles(v2_joints_3d_after_zs) if has_skel_v2_zs else {}
    angles_skel_v2_hr = _compute_angles(v2_joints_3d_after_hr) if has_skel_v2_hr else {}
    angles_skel_v2_bc = _compute_angles(v2_joints_3d_after_bc) if has_skel_v2_bc else {}
    angles_skel_legacy = _compute_angles(legacy_joints_3d) if has_skel_legacy else {}
    angles_mp = _compute_angles(mp_joints_3d)
    angles_vision = _compute_angles(vision_joints_3d)
    angles_hrnet_centroid  = _compute_angles(hrnet_centroid_3d)  if has_centroid_3d  else {}
    angles_hrnet_yzc       = _compute_angles(hrnet_yzc_3d)       if has_yzc_3d       else {}
    angles_hrnet_zsmooth   = _compute_angles(hrnet_zsmooth_3d)   if has_zsmooth_3d   else {}
    angles_hrnet_hungarian = _compute_angles(hrnet_hungarian_3d) if has_hungarian_3d else {}

    # Knuckle angles (inter-MCP segment angles at M_MCP and R_MCP)
    if has_skeleton_fit: angles_mano.update(_compute_knuckle_angles(joints_3d))
    if has_skel_v2: angles_skel_v2.update(_compute_knuckle_angles(v2_joints_3d))
    if has_skel_v2_sc: angles_skel_v2_sc.update(_compute_knuckle_angles(v2_joints_3d_after_sc))
    if has_skel_v2_y: angles_skel_v2_y.update(_compute_knuckle_angles(v2_joints_3d_after_y))
    if has_skel_v2_z: angles_skel_v2_z.update(_compute_knuckle_angles(v2_joints_3d_after_z))
    if has_skel_v2_zs: angles_skel_v2_zs.update(_compute_knuckle_angles(v2_joints_3d_after_zs))
    if has_skel_v2_hr: angles_skel_v2_hr.update(_compute_knuckle_angles(v2_joints_3d_after_hr))
    if has_skel_v2_bc: angles_skel_v2_bc.update(_compute_knuckle_angles(v2_joints_3d_after_bc))
    if has_skel_legacy: angles_skel_legacy.update(_compute_knuckle_angles(legacy_joints_3d))
    angles_mp.update(_compute_knuckle_angles(mp_joints_3d))
    angles_vision.update(_compute_knuckle_angles(vision_joints_3d))
    if has_centroid_3d:  angles_hrnet_centroid.update(_compute_knuckle_angles(hrnet_centroid_3d))
    if has_yzc_3d:       angles_hrnet_yzc.update(_compute_knuckle_angles(hrnet_yzc_3d))
    if has_zsmooth_3d:   angles_hrnet_zsmooth.update(_compute_knuckle_angles(hrnet_zsmooth_3d))
    if has_hungarian_3d: angles_hrnet_hungarian.update(_compute_knuckle_angles(hrnet_hungarian_3d))

    # Wrist flex/abd angles (requires elbow 3D)
    has_elbow = bool(np.any(~np.isnan(elbow_3d[:, 0])))
    if has_elbow:
        if has_skeleton_fit:    angles_mano.update(_compute_wrist_angles(joints_3d, elbow_3d))
        if has_skel_v2: angles_skel_v2.update(_compute_wrist_angles(v2_joints_3d, elbow_3d))
        if has_skel_v2_sc: angles_skel_v2_sc.update(_compute_wrist_angles(v2_joints_3d_after_sc, elbow_3d))
        if has_skel_v2_y: angles_skel_v2_y.update(_compute_wrist_angles(v2_joints_3d_after_y, elbow_3d))
        if has_skel_v2_z: angles_skel_v2_z.update(_compute_wrist_angles(v2_joints_3d_after_z, elbow_3d))
        if has_skel_v2_zs: angles_skel_v2_zs.update(_compute_wrist_angles(v2_joints_3d_after_zs, elbow_3d))
        if has_skel_v2_hr: angles_skel_v2_hr.update(_compute_wrist_angles(v2_joints_3d_after_hr, elbow_3d))
        if has_skel_v2_bc: angles_skel_v2_bc.update(_compute_wrist_angles(v2_joints_3d_after_bc, elbow_3d))
        if has_skel_legacy: angles_skel_legacy.update(_compute_wrist_angles(legacy_joints_3d, elbow_3d))
        angles_mp.update(_compute_wrist_angles(mp_joints_3d, elbow_3d))
        angles_vision.update(_compute_wrist_angles(vision_joints_3d, elbow_3d))
        if has_centroid_3d:  angles_hrnet_centroid.update(_compute_wrist_angles(hrnet_centroid_3d, elbow_3d))
        if has_yzc_3d:       angles_hrnet_yzc.update(_compute_wrist_angles(hrnet_yzc_3d, elbow_3d))
        if has_zsmooth_3d:   angles_hrnet_zsmooth.update(_compute_wrist_angles(hrnet_zsmooth_3d, elbow_3d))
        if has_hungarian_3d: angles_hrnet_hungarian.update(_compute_wrist_angles(hrnet_hungarian_3d, elbow_3d))

    # Compute finger spread angles
    spreads_mano = _compute_spreads(joints_3d) if has_skeleton_fit else {}
    spreads_skel_v2 = _compute_spreads(v2_joints_3d) if has_skel_v2 else {}
    spreads_skel_v2_sc = _compute_spreads(v2_joints_3d_after_sc) if has_skel_v2_sc else {}
    spreads_skel_v2_y = _compute_spreads(v2_joints_3d_after_y) if has_skel_v2_y else {}
    spreads_skel_v2_z = _compute_spreads(v2_joints_3d_after_z) if has_skel_v2_z else {}
    spreads_skel_v2_zs = _compute_spreads(v2_joints_3d_after_zs) if has_skel_v2_zs else {}
    spreads_skel_v2_hr = _compute_spreads(v2_joints_3d_after_hr) if has_skel_v2_hr else {}
    spreads_skel_v2_bc = _compute_spreads(v2_joints_3d_after_bc) if has_skel_v2_bc else {}
    spreads_skel_legacy = _compute_spreads(legacy_joints_3d) if has_skel_legacy else {}
    spreads_mp = _compute_spreads(mp_joints_3d)
    spreads_vision = _compute_spreads(vision_joints_3d)
    spreads_hrnet_centroid  = _compute_spreads(hrnet_centroid_3d)  if has_centroid_3d  else {}
    spreads_hrnet_yzc       = _compute_spreads(hrnet_yzc_3d)       if has_yzc_3d       else {}
    spreads_hrnet_zsmooth   = _compute_spreads(hrnet_zsmooth_3d)   if has_zsmooth_3d   else {}
    spreads_hrnet_hungarian = _compute_spreads(hrnet_hungarian_3d) if has_hungarian_3d else {}

    # Compute wrist 3D coordinates
    wrist_coords_mano = _compute_wrist_coords(joints_3d) if has_skeleton_fit else {}
    wrist_coords_skel_v2 = _compute_wrist_coords(v2_joints_3d) if has_skel_v2 else {}
    wrist_coords_skel_v2_sc = _compute_wrist_coords(v2_joints_3d_after_sc) if has_skel_v2_sc else {}
    wrist_coords_skel_v2_y = _compute_wrist_coords(v2_joints_3d_after_y) if has_skel_v2_y else {}
    wrist_coords_skel_v2_z = _compute_wrist_coords(v2_joints_3d_after_z) if has_skel_v2_z else {}
    wrist_coords_skel_v2_zs = _compute_wrist_coords(v2_joints_3d_after_zs) if has_skel_v2_zs else {}
    wrist_coords_skel_v2_hr = _compute_wrist_coords(v2_joints_3d_after_hr) if has_skel_v2_hr else {}
    wrist_coords_skel_v2_bc = _compute_wrist_coords(v2_joints_3d_after_bc) if has_skel_v2_bc else {}
    wrist_coords_skel_legacy = _compute_wrist_coords(legacy_joints_3d) if has_skel_legacy else {}
    wrist_coords_mp = _compute_wrist_coords(mp_joints_3d)
    wrist_coords_vision = _compute_wrist_coords(vision_joints_3d)
    wrist_coords_hrnet_centroid  = _compute_wrist_coords(hrnet_centroid_3d)  if has_centroid_3d  else {}
    wrist_coords_hrnet_yzc       = _compute_wrist_coords(hrnet_yzc_3d)       if has_yzc_3d       else {}
    wrist_coords_hrnet_zsmooth   = _compute_wrist_coords(hrnet_zsmooth_3d)   if has_zsmooth_3d   else {}
    wrist_coords_hrnet_hungarian = _compute_wrist_coords(hrnet_hungarian_3d) if has_hungarian_3d else {}

    # Compute per-joint position traces (de-meaned, +100mm)
    positions_mano = _compute_joint_positions(joints_3d) if has_skeleton_fit else {}
    positions_skel_v2 = _compute_joint_positions(v2_joints_3d) if has_skel_v2 else {}
    positions_skel_v2_sc = _compute_joint_positions(v2_joints_3d_after_sc) if has_skel_v2_sc else {}
    positions_skel_v2_y = _compute_joint_positions(v2_joints_3d_after_y) if has_skel_v2_y else {}
    positions_skel_v2_z = _compute_joint_positions(v2_joints_3d_after_z) if has_skel_v2_z else {}
    positions_skel_v2_zs = _compute_joint_positions(v2_joints_3d_after_zs) if has_skel_v2_zs else {}
    positions_skel_v2_hr = _compute_joint_positions(v2_joints_3d_after_hr) if has_skel_v2_hr else {}
    positions_skel_v2_bc = _compute_joint_positions(v2_joints_3d_after_bc) if has_skel_v2_bc else {}
    positions_skel_legacy = _compute_joint_positions(legacy_joints_3d) if has_skel_legacy else {}
    positions_mp = _compute_joint_positions(mp_joints_3d)
    positions_vision = _compute_joint_positions(vision_joints_3d)
    positions_hrnet_centroid  = _compute_joint_positions(hrnet_centroid_3d)  if has_centroid_3d  else {}
    positions_hrnet_yzc       = _compute_joint_positions(hrnet_yzc_3d)       if has_yzc_3d       else {}
    positions_hrnet_zsmooth   = _compute_joint_positions(hrnet_zsmooth_3d)   if has_zsmooth_3d   else {}
    positions_hrnet_hungarian = _compute_joint_positions(hrnet_hungarian_3d) if has_hungarian_3d else {}

    # Fallback: if 3D distances are all empty, try pre-computed from prelabels npz
    mp_has_3d = any(
        any(v is not None for v in vals) for vals in distances_mp.values()
    )
    if not mp_has_3d:
        # Try loading saved distances from mediapipe_prelabels.npz
        from .mediapipe_prelabel import load_mediapipe_prelabels as _load_mp
        _saved = _load_mp(subject_name)
        saved_dist = _saved.get("distances") if _saved else None
        if saved_dist is not None and len(saved_dist) > start_frame:
            # Slice to this trial's frame range
            trial_dist = saved_dist[start_frame:start_frame + N]
            # Wrap in the standard distance format (only thumb-index available)
            distances_mp = {"Thumb-Index Aperture": _array_to_list(trial_dist)}
            logger.info(f"Using saved distances for {subject_name}/{trial_stem}")
        else:
            logger.info(f"No 3D distances for {subject_name}/{trial_stem}, falling back to 2D")
            distances_mp = _compute_distances_2d(mp_tracked_L)

    # ── Assemble result ────────────────────────────────────────
    result: dict[str, Any] = {
        "n_frames": N,
        "fps": fps,
        "has_skeleton_v1": has_skeleton_fit,
        "has_skel_v2": has_skel_v2,
        # Per-stage flags so the frontend can hide stage rows that the
        # active v3 fit didn't actually produce (e.g. HRnet-snap when
        # the user kept "HRnet peak dist" at 0).
        "has_skel_v2_sc": bool(has_skel_v2_sc),
        "has_skel_v2_y":  bool(has_skel_v2_y),
        "has_skel_v2_z":  bool(has_skel_v2_z),
        "has_skel_v2_zs": bool(has_skel_v2_zs),
        "has_skel_v2_hr": bool(has_skel_v2_hr),
        "has_skel_v2_bc": bool(has_skel_v2_bc),
        "has_heatmaps": (skeleton_trial_dir / "hrnet_w18_heatmaps.npz").exists(),
        "has_mp": bool(np.any(~np.isnan(mp_tracked_L))),
        "has_mp_3d": bool(np.any(~np.isnan(mp_joints_3d))),
        "has_vision": bool(np.any(~np.isnan(vision_tracked_L))),
        "has_vision_3d": bool(np.any(~np.isnan(vision_joints_3d))),
        "has_dlc": bool(np.any(~np.isnan(dlc_thumb_OS)) or np.any(~np.isnan(dlc_index_OS))),
        "has_pose": has_pose,
        "has_wrist_angles": has_elbow,
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
        "skeleton_proj_L": _points_to_list(skeleton_proj_L),
        "skeleton_proj_R": _points_to_list(skeleton_proj_R),
        # 3D joints
        "skeleton_joints_3d": _points_to_list(joints_3d),
        # Skeleton v2
        "skel_v2_proj_L": _points_to_list(v2_proj_L),
        "skel_v2_proj_R": _points_to_list(v2_proj_R),
        "skel_v2_joints_3d": _points_to_list(v2_joints_3d),
        # Intermediate view after Stereo-correction (Step 0)
        "skel_v2_proj_sc_L": _points_to_list(v2_mp_L_after_sc) if v2_mp_L_after_sc is not None else None,
        "skel_v2_proj_sc_R": _points_to_list(v2_mp_R_after_sc) if v2_mp_R_after_sc is not None else None,
        "skel_v2_joints_sc_3d": _points_to_list(v2_joints_3d_after_sc) if v2_joints_3d_after_sc is not None else None,
        # (N, 21, 2) bool -- True on the camera whose MP was replaced
        # by its stereo label at Step 0.  Drives the pink-X overlay
        # on the MP-error display.
        "stereo_blame": (v2_stereo_blame.astype(bool).tolist()
                          if v2_stereo_blame is not None else None),
        # (N, 21, 2) float -- per-frame stereo-corrected position in each
        # camera's pixel space (== mp ± shifts).  None if not baked.
        "stereo_L_pts": (_points_to_list(v2_stereo_L_pts)
                          if v2_stereo_L_pts is not None else None),
        "stereo_R_pts": (_points_to_list(v2_stereo_R_pts)
                          if v2_stereo_R_pts is not None else None),
        "stereo_response_v3": (v2_stereo_response.tolist()
                                if v2_stereo_response is not None else None),
        # Intermediate view after Y-disparity correction only
        "skel_v2_proj_y_L": _points_to_list(v2_mp_L_after_y) if v2_mp_L_after_y is not None else None,
        "skel_v2_proj_y_R": _points_to_list(v2_mp_R_after_y) if v2_mp_R_after_y is not None else None,
        "skel_v2_joints_y_3d": _points_to_list(v2_joints_3d_after_y) if v2_joints_3d_after_y is not None else None,
        # Intermediate view after Y + Z-outlier correction
        "skel_v2_proj_z_L": _points_to_list(v2_mp_L_after_z) if v2_mp_L_after_z is not None else None,
        "skel_v2_proj_z_R": _points_to_list(v2_mp_R_after_z) if v2_mp_R_after_z is not None else None,
        "skel_v2_joints_z_3d": _points_to_list(v2_joints_3d_after_z) if v2_joints_3d_after_z is not None else None,
        # Intermediate view after Y + Z-outlier + Z-jump (pre-HRnet)
        "skel_v2_proj_zs_L": _points_to_list(v2_mp_L_after_zs) if v2_mp_L_after_zs is not None else None,
        "skel_v2_proj_zs_R": _points_to_list(v2_mp_R_after_zs) if v2_mp_R_after_zs is not None else None,
        "skel_v2_joints_zs_3d": _points_to_list(v2_joints_3d_after_zs) if v2_joints_3d_after_zs is not None else None,
        # Intermediate view after HRnet snap + re-clean
        "skel_v2_proj_hr_L": _points_to_list(v2_mp_L_after_hr) if v2_mp_L_after_hr is not None else None,
        "skel_v2_proj_hr_R": _points_to_list(v2_mp_R_after_hr) if v2_mp_R_after_hr is not None else None,
        "skel_v2_proj_bc_L": _points_to_list(v2_mp_L_after_bc) if v2_mp_L_after_bc is not None else None,
        "skel_v2_proj_bc_R": _points_to_list(v2_mp_R_after_bc) if v2_mp_R_after_bc is not None else None,
        "skel_v2_joints_bc_3d": _points_to_list(v2_joints_3d_after_bc) if v2_joints_3d_after_bc is not None else None,
        "skel_v2_joints_hr_3d": _points_to_list(v2_joints_3d_after_hr) if v2_joints_3d_after_hr is not None else None,
        "hrnet_along_L": v2_hrnet_along_L.tolist() if v2_hrnet_along_L is not None else None,
        "hrnet_perp_L":  v2_hrnet_perp_L.tolist()  if v2_hrnet_perp_L  is not None else None,
        "hrnet_along_R": v2_hrnet_along_R.tolist() if v2_hrnet_along_R is not None else None,
        "hrnet_perp_R":  v2_hrnet_perp_R.tolist()  if v2_hrnet_perp_R  is not None else None,
        "hrnet_child":   v2_hrnet_child.tolist()   if v2_hrnet_child   is not None else None,
        # Skeleton v2 (legacy/frozen — separate fit path)
        "skel_legacy_proj_L": _points_to_list(legacy_proj_L),
        "skel_legacy_proj_R": _points_to_list(legacy_proj_R),
        "skel_legacy_joints_3d": _points_to_list(legacy_joints_3d),
        "has_skel_legacy": bool(has_skel_legacy),
        # MediaPipe error matrix (if previously saved)
        "mp_errors": _load_mp_errors_for_response(subject_name, trial_stem),
        "has_mp_errors": bool((skeleton_trial_dir / "mp_errors.npz").exists()),
        # MediaPipe
        "mp_tracked_L": _points_to_list(mp_tracked_L),
        "mp_tracked_R": _points_to_list(mp_tracked_R),
        "mp_joints_3d": _points_to_list(mp_joints_3d),
        # Reverse-pass MediaPipe (sibling layer; same schema as MP).
        # ``has_reverse_mp`` lets the Labels page hide the row when no
        # reverse npz exists for this subject yet.
        "reverse_tracked_L": _points_to_list(reverse_tracked_L),
        "reverse_tracked_R": _points_to_list(reverse_tracked_R),
        "reverse_joints_3d": _points_to_list(reverse_joints_3d),
        "has_reverse_mp": bool(np.any(~np.isnan(reverse_tracked_L))
                                or np.any(~np.isnan(reverse_tracked_R))),
        # Stereo (image-based cross-camera alignment) — populated below
        # if a saved ``stereo_align.npz`` exists for this trial.
        "stereo_tracked_L": None,
        "stereo_tracked_R": None,
        "stereo_response": None,
        "has_stereo": False,
        # Stereo (outline-based) -- the same alignment but driven by
        # the preproc-baked hand outlines instead of raw pixels.
        "stereo_outline_tracked_L": None,
        "stereo_outline_tracked_R": None,
        "stereo_outline_response": None,
        "has_stereo_outline": False,
        # Stereo (hybrid) -- Pass 1 outline-vote, Pass 2 image phase-corr.
        "stereo_hybrid_tracked_L": None,
        "stereo_hybrid_tracked_R": None,
        "stereo_hybrid_response": None,
        "has_stereo_hybrid": False,
        # Per-frame hand outlines (original camera coords), one entry
        # per trial frame: {OS: [[x,y],...], OD: [...] | None}.
        # Populated below when outlines.json + camera_trajectory are
        # both present in preproc.
        "outlines_L": None,
        "outlines_R": None,
        "has_outlines": False,
        # Vision (Apple Vision)
        "vision_tracked_L": _points_to_list(vision_tracked_L),
        "vision_tracked_R": _points_to_list(vision_tracked_R),
        "vision_joints_3d": _points_to_list(vision_joints_3d),
        # DLC (thumb=joint4, index=joint8)
        "dlc_thumb_OS": [_nan_to_none_pair(dlc_thumb_OS[t]) for t in range(N)],
        "dlc_thumb_OD": [_nan_to_none_pair(dlc_thumb_OD[t]) for t in range(N)],
        "dlc_index_OS": [_nan_to_none_pair(dlc_index_OS[t]) for t in range(N)],
        "dlc_index_OD": [_nan_to_none_pair(dlc_index_OD[t]) for t in range(N)],
        # DLC 3D (thumb + index only)
        "dlc_3d_thumb": [_nan_to_none_triple(dlc_3d_thumb[t]) for t in range(N)],
        "dlc_3d_index": [_nan_to_none_triple(dlc_3d_index[t]) for t in range(N)],
        # Pose body landmarks (arm chain only: indices 11-22)
        "pose_tracked_L": _points_to_list(pose_arm_L),
        "pose_tracked_R": _points_to_list(pose_arm_R),
        "pose_arm_3d": _points_to_list(pose_arm_3d),
        "pose_arm_indices": POSE_ARM_INDICES,
        "pose_side": pose_side,
        # Elbow 3D + 2D projections
        "elbow_3d": [_nan_to_none_triple(elbow_3d[t]) for t in range(N)],
        "elbow_proj_L": [_nan_to_none_pair(elbow_proj_L[t]) for t in range(N)],
        "elbow_proj_R": [_nan_to_none_pair(elbow_proj_R[t]) for t in range(N)],
        # Distances
        "distances_mano": distances_mano,
        "distances_skel_v2": distances_skel_v2,
        "distances_skel_v2_sc": distances_skel_v2_sc,
        "distances_skel_v2_y": distances_skel_v2_y,
        "distances_skel_v2_z": distances_skel_v2_z,
        "distances_skel_v2_zs": distances_skel_v2_zs,
        "distances_skel_v2_hr": distances_skel_v2_hr,
        "distances_skel_v2_bc": distances_skel_v2_bc,
        "distances_skel_legacy": distances_skel_legacy,
        "distances_mp": distances_mp,
        "distances_vision": distances_vision,
        "distances_heatmap": distances_heatmap,
        "distances_hrnet_centroid":  distances_hrnet_centroid,
        "distances_hrnet_yzc":       distances_hrnet_yzc,
        "distances_hrnet_zsmooth":   distances_hrnet_zsmooth,
        "distances_hrnet_hungarian": distances_hrnet_hungarian,
        "distances_dlc": distances_dlc,
        # Joint angles (per-frame, degrees)
        "angles_mano": angles_mano,
        "angles_skel_v2": angles_skel_v2,
        "angles_skel_v2_sc": angles_skel_v2_sc,
        "angles_skel_v2_y": angles_skel_v2_y,
        "angles_skel_v2_z": angles_skel_v2_z,
        "angles_skel_v2_zs": angles_skel_v2_zs,
        "angles_skel_v2_hr": angles_skel_v2_hr,
        "angles_skel_v2_bc": angles_skel_v2_bc,
        "angles_skel_legacy": angles_skel_legacy,
        "angles_mp": angles_mp,
        "angles_vision": angles_vision,
        "angles_hrnet_centroid":  angles_hrnet_centroid,
        "angles_hrnet_yzc":       angles_hrnet_yzc,
        "angles_hrnet_zsmooth":   angles_hrnet_zsmooth,
        "angles_hrnet_hungarian": angles_hrnet_hungarian,
        # Finger spread angles (per-frame, degrees)
        "spreads_mano": spreads_mano,
        "spreads_skel_v2": spreads_skel_v2,
        "spreads_skel_v2_sc": spreads_skel_v2_sc,
        "spreads_skel_v2_y": spreads_skel_v2_y,
        "spreads_skel_v2_z": spreads_skel_v2_z,
        "spreads_skel_v2_zs": spreads_skel_v2_zs,
        "spreads_skel_v2_hr": spreads_skel_v2_hr,
        "spreads_skel_v2_bc": spreads_skel_v2_bc,
        "spreads_skel_legacy": spreads_skel_legacy,
        "spreads_mp": spreads_mp,
        "spreads_vision": spreads_vision,
        "spreads_hrnet_centroid":  spreads_hrnet_centroid,
        "spreads_hrnet_yzc":       spreads_hrnet_yzc,
        "spreads_hrnet_zsmooth":   spreads_hrnet_zsmooth,
        "spreads_hrnet_hungarian": spreads_hrnet_hungarian,
        # Wrist 3D coordinates (mm, per-frame)
        "wrist_coords_mano": wrist_coords_mano,
        "wrist_coords_skel_v2": wrist_coords_skel_v2,
        "wrist_coords_skel_v2_sc": wrist_coords_skel_v2_sc,
        "wrist_coords_skel_v2_y": wrist_coords_skel_v2_y,
        "wrist_coords_skel_v2_z": wrist_coords_skel_v2_z,
        "wrist_coords_skel_v2_zs": wrist_coords_skel_v2_zs,
        "wrist_coords_skel_v2_hr": wrist_coords_skel_v2_hr,
        "wrist_coords_skel_v2_bc": wrist_coords_skel_v2_bc,
        "wrist_coords_skel_legacy": wrist_coords_skel_legacy,
        "wrist_coords_mp": wrist_coords_mp,
        "wrist_coords_vision": wrist_coords_vision,
        "wrist_coords_hrnet_centroid":  wrist_coords_hrnet_centroid,
        "wrist_coords_hrnet_yzc":       wrist_coords_hrnet_yzc,
        "wrist_coords_hrnet_zsmooth":   wrist_coords_hrnet_zsmooth,
        "wrist_coords_hrnet_hungarian": wrist_coords_hrnet_hungarian,
        # Per-joint position traces (de-meaned +100mm)
        "positions_mano": positions_mano,
        "positions_skel_v2": positions_skel_v2,
        "positions_skel_v2_sc": positions_skel_v2_sc,
        "positions_skel_v2_y": positions_skel_v2_y,
        "positions_skel_v2_z": positions_skel_v2_z,
        "positions_skel_v2_zs": positions_skel_v2_zs,
        "positions_skel_v2_hr": positions_skel_v2_hr,
        "positions_skel_v2_bc": positions_skel_v2_bc,
        "positions_skel_legacy": positions_skel_legacy,
        "positions_mp": positions_mp,
        "positions_vision": positions_vision,
        "positions_hrnet_centroid":  positions_hrnet_centroid,
        "positions_hrnet_yzc":       positions_hrnet_yzc,
        "positions_hrnet_zsmooth":   positions_hrnet_zsmooth,
        "positions_hrnet_hungarian": positions_hrnet_hungarian,
        # Angle option metadata (for frontend)
        "flex_angle_options": [{"name": n, "parent": p, "joint": j, "child": c} for n, p, j, c in FLEX_ANGLE_OPTIONS]
            + ([{"name": "Flex: Wrist", "parent": -1, "joint": 0, "child": 9}] if has_elbow else []),
        "abd_angle_options": [{"name": n.replace('Flex:', 'Abd:'), "parent": p, "joint": j, "child": c} for n, p, j, c in FLEX_ANGLE_OPTIONS]
            + ([{"name": "Abd: Wrist", "parent": -1, "joint": 0, "child": 9}] if has_elbow else []),
        # Joint angle constraint boundaries (for plotting on the trace)
        "angle_constraints": load_angle_priors().get("joints", []),
        # Fit quality
        "fit_error_L": _array_to_list(fit_error_L),
        "fit_error_R": _array_to_list(fit_error_R),
        "v2_fit_error_L": _array_to_list(v2_fit_error_L),
        "v2_fit_error_R": _array_to_list(v2_fit_error_R),
        # Saved fit parameters (for restoring slider values)
        "v1_fit_params": v1_fit_params.get("params") if v1_fit_params else None,
        "v2_fit_params": v2_fit_params.get("params") if v2_fit_params else None,
        "v2_offset_L": v2.get("offset_L", np.zeros(2)).ravel()[:2].tolist() if has_skel_v2 else [0, 0],
        "v2_offset_R": v2.get("offset_R", np.zeros(2)).ravel()[:2].tolist() if has_skel_v2 else [0, 0],
        "v2_fit_constraints": v2_fit_params.get("angle_constraints") if v2_fit_params else None,
        # Previous fit history (metadata only — full data loaded on demand)
        "v2_fit_history": list_v2_fit_history(subject_name, trial_stem),
        # HRNet peak assignments (for display)
        "hrnet_peaks": hrnet_peaks_cached,
        "hrnet_peaks_3d": _points_to_list(hrnet_peaks_3d) if has_heatmap_3d else None,
        "hrnet_peaks_raw_3d": _points_to_list(hrnet_peaks_raw_3d) if has_heatmap_raw_3d else None,
        "hrnet_centroid_3d":  _points_to_list(hrnet_centroid_3d)  if has_centroid_3d  else None,
        "hrnet_yzc_3d":       _points_to_list(hrnet_yzc_3d)       if has_yzc_3d       else None,
        "hrnet_zsmooth_3d":   _points_to_list(hrnet_zsmooth_3d)   if has_zsmooth_3d   else None,
        "hrnet_hungarian_3d": _points_to_list(hrnet_hungarian_3d) if has_hungarian_3d else None,
    }

    # Optional: weights and residuals
    if mp_weights_L is not None:
        result["mp_weights_L"] = [[_nan_to_none(round(float(mp_weights_L[t, j]), 3))
                                    for j in range(21)] for t in range(N)]
    if mp_weights_R is not None:
        result["mp_weights_R"] = [[_nan_to_none(round(float(mp_weights_R[t, j]), 3))
                                    for j in range(21)] for t in range(N)]

    # ── Stereo image-based cross-camera label alignment ─────────────
    # Loads ``<skeleton>/<stem>/stereo_align.npz`` if present and emits the
    # OS / OD labels translated by the per-frame per-joint phase-corr
    # shift discovered between the two crops — the Stereo model on the
    # Auto page renders these alongside the local MP labels to surface
    # cross-camera disagreement at a glance.
    try:
        from .stereo_align import (
            load_stereo_align, crop_halves_per_joint as _default_per_joint,
            _DEFAULT_CROP_HALF, _HAND_CROP_HALF,
        )
        _stereo_trial_idx = next(
            (i for i, t in enumerate(build_trial_map(subject_name))
             if t.get("trial_name") == trial_stem),
            None,
        )
        def _emit_stereo(sa, key_L, key_R, key_resp, key_has):
            """Translate MP labels by the per-joint shifts from ``sa``
            and write them into ``result`` under the given keys."""
            if sa is None:
                return
            shifts = sa["shifts"]                 # (N_sa, 21, 2)
            resp = sa["response"]                 # (N_sa, 21)
            n_sa = min(N, shifts.shape[0])
            stereo_R = np.full((N, 21, 2), np.nan)
            stereo_L = np.full((N, 21, 2), np.nan)
            stereo_R[:n_sa, :, 0] = mp_tracked_R[:n_sa, :, 0] + shifts[:n_sa, :, 0]
            stereo_R[:n_sa, :, 1] = mp_tracked_R[:n_sa, :, 1] + shifts[:n_sa, :, 1]
            stereo_L[:n_sa, :, 0] = mp_tracked_L[:n_sa, :, 0] - shifts[:n_sa, :, 0]
            stereo_L[:n_sa, :, 1] = mp_tracked_L[:n_sa, :, 1] - shifts[:n_sa, :, 1]
            result[key_L] = _points_to_list(stereo_L)
            result[key_R] = _points_to_list(stereo_R)
            result[key_resp] = [[_nan_to_none(round(float(resp[t, j]), 3))
                                   for j in range(21)]
                                  for t in range(n_sa)]
            result[key_has] = bool(np.any(~np.isnan(shifts)))

        sa = (load_stereo_align(subject_name, _stereo_trial_idx, mode="image")
              if _stereo_trial_idx is not None else None)
        sa_out = (load_stereo_align(subject_name, _stereo_trial_idx, mode="outline")
                  if _stereo_trial_idx is not None else None)
        sa_hyb = (load_stereo_align(subject_name, _stereo_trial_idx, mode="hybrid")
                  if _stereo_trial_idx is not None else None)
        if sa is not None:
            _emit_stereo(sa, "stereo_tracked_L", "stereo_tracked_R",
                         "stereo_response", "has_stereo")
        if sa_out is not None:
            _emit_stereo(sa_out, "stereo_outline_tracked_L",
                         "stereo_outline_tracked_R",
                         "stereo_outline_response", "has_stereo_outline")
        if sa_hyb is not None:
            _emit_stereo(sa_hyb, "stereo_hybrid_tracked_L",
                         "stereo_hybrid_tracked_R",
                         "stereo_hybrid_response", "has_stereo_hybrid")
        # Per-joint + hand crop sizes (used by the frontend to draw
        # the local-registration bbox).  All three variants share the
        # same constants, so emit them if ANY bake produced an npz.
        _sa_meta = sa if sa is not None else (sa_out if sa_out is not None else sa_hyb)
        if _sa_meta is not None:
            result["stereo_crop_half"] = int(_sa_meta.get("crop_half", _DEFAULT_CROP_HALF))
            result["stereo_hand_crop_half"] = int(_sa_meta.get("hand_crop_half", _HAND_CROP_HALF))
            result["stereo_crop_halves_per_joint"] = [
                int(x) for x in (_sa_meta.get("crop_halves_per_joint")
                                  or _default_per_joint(21))
            ]
    except Exception as _e:
        logger.debug(f"stereo_align load skipped: {_e}")

    # ── Per-frame outlines (ref-space -> original camera coords) ────
    # Loaded from outlines.json baked by the preproc "Compute boundary
    # - all frames" step, inverse-warped via the camera trajectory so
    # they line up with the original-video MP labels the Labels page
    # draws.
    try:
        import json as _json
        from .background import _preproc_dir as _bg_preproc_dir
        from .camera_motion import load_camera_trajectory
        if _stereo_trial_idx is not None:
            _opath = _bg_preproc_dir(subject_name, trial_stem) / "outlines.json"
            if _opath.exists():
                with open(_opath) as _fh:
                    _ol = _json.load(_fh)
                _traj = load_camera_trajectory(subject_name, trial_stem)
                if _traj is not None:
                    _HL = _traj["H_to_ref_L"]
                    _HR = _traj["H_to_ref_R"]
                    def _warp_back(poly_ref, H):
                        if not poly_ref or len(poly_ref) < 3:
                            return None
                        try:
                            H_inv = np.linalg.inv(H.astype(np.float64))
                        except np.linalg.LinAlgError:
                            return None
                        pts = np.asarray(poly_ref, dtype=np.float64)
                        n = pts.shape[0]
                        ph = np.column_stack([pts, np.ones(n)]).T
                        w = H_inv @ ph
                        z = w[2]
                        if np.any(np.abs(z) < 1e-9):
                            return None
                        xy = (w[:2] / z).T
                        if not np.all(np.isfinite(xy)):
                            return None
                        return np.rint(xy).astype(int).tolist()
                    _frames = _ol.get("frames") or []
                    _ncap = min(N, len(_frames),
                                 int(_HL.shape[0]), int(_HR.shape[0]))
                    out_L = [None] * N
                    out_R = [None] * N
                    for _fi in range(_ncap):
                        _of = _frames[_fi]
                        out_L[_fi] = _warp_back(_of.get("OS"), _HL[_fi])
                        out_R[_fi] = _warp_back(_of.get("OD"), _HR[_fi])
                    result["outlines_L"] = out_L
                    result["outlines_R"] = out_R
                    result["has_outlines"] = any(p is not None for p in out_L)
    except Exception as _e:
        logger.debug(f"outlines load skipped: {_e}")

    return result


def _nan_to_none_pair(arr: np.ndarray) -> list | None:
    """Convert a (2,) array to [x,y] or None if NaN."""
    if np.isnan(arr[0]):
        return None
    return [round(float(arr[0]), 2), round(float(arr[1]), 2)]


def _nan_to_none_triple(arr: np.ndarray) -> list | None:
    """Convert a (3,) array to [x,y,z] or None if NaN."""
    if np.isnan(arr[0]):
        return None
    return [round(float(arr[0]), 2), round(float(arr[1]), 2), round(float(arr[2]), 2)]


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
_heatmap_cache: dict[str, tuple] = {}  # path → (mtime, np.load result)


def get_heatmap(subject_name: str, trial_stem: str,
                frame: int, joint: int, side: str) -> dict | None:
    """Load a single 64x64 heatmap for one joint at one frame.

    ``joint=-1`` returns the pre-computed MIP from ``hrnet_w18_mip.npz``
    (max over 21 joints) when available.  Uses memory-mapped access to
    avoid loading the entire ~350MB per-joint file.
    """
    skeleton_trial_dir = _skeleton_dir(subject_name) / trial_stem
    hm_path = skeleton_trial_dir / "hrnet_w18_heatmaps.npz"
    mip_path = skeleton_trial_dir / "hrnet_w18_mip.npz"
    crop_path = skeleton_trial_dir / "hand_crop.json"

    if not crop_path.exists():
        return None
    with open(crop_path) as f:
        crop_info = json.load(f)

    # Resolve the bbox for THIS frame.  Prefer per-frame array (new
    # outputs), fall back to legacy union ``crop_L``/``crop_R`` (older
    # outputs).  Heatmap-pixel ↔ image-pixel mapping must use the same
    # crop the network saw for this frame.
    if side == "OS":
        side_key, hm_key, mip_key = "crop_L", "heatmaps_L", "heatmaps_L_mip"
    else:
        side_key, hm_key, mip_key = "crop_R", "heatmaps_R", "heatmaps_R_mip"
    pf = crop_info.get(f"{side_key}_perframe")
    if pf is not None and frame < len(pf):
        bbox = pf[frame]
    else:
        bbox = crop_info.get(side_key)
    if bbox is None:
        return None

    # Precomputed-MIP fast path — one slice (~8KB) instead of 21.
    if joint == -1 and mip_path.exists():
        cache_key = str(mip_path)
        current_mtime = mip_path.stat().st_mtime
        cached = _heatmap_cache.get(cache_key)
        if cached is None or cached[0] != current_mtime:
            try:
                _heatmap_cache[cache_key] = (current_mtime, np.load(str(mip_path), mmap_mode="r"))
            except Exception as e:
                logger.error(f"Failed to load MIP: {e}")
                return None
        mip_data = _heatmap_cache[cache_key][1]
        if mip_key not in mip_data:
            return None
        mip_arr = mip_data[mip_key]
        if frame >= mip_arr.shape[0]:
            return None
        hm = mip_arr[frame].astype(np.float32)
        max_val = float(hm.max()); min_val = float(hm.min())
        rng = max_val - min_val
        hm_norm = (hm - min_val) / rng if rng > 1e-9 else hm
        return {"heatmap": hm_norm.tolist(), "bbox": list(bbox), "max_val": max_val}

    if not hm_path.exists():
        return None

    # Memory-map the heatmap file (invalidate cache if file changed)
    cache_key = str(hm_path)
    current_mtime = hm_path.stat().st_mtime
    cached = _heatmap_cache.get(cache_key)
    if cached is None or cached[0] != current_mtime:
        try:
            _heatmap_cache[cache_key] = (current_mtime, np.load(str(hm_path), mmap_mode="r"))
        except Exception as e:
            logger.error(f"Failed to load heatmaps: {e}")
            return None

    hm_data = _heatmap_cache[cache_key][1]

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

    # Normalise to [0, 1] using min-max stretch so the spatial peak
    # stands out even when the raw values span a narrow range (e.g.
    # after sigmoid + global normalisation in hrnet.py).
    min_val = float(hm.min())
    rng = max_val - min_val
    if rng > 1e-6:
        hm = (hm - min_val) / rng
    elif max_val > 0:
        hm = hm / max_val

    return {
        "heatmap": [[round(float(hm[r, c]), 4) for c in range(hm.shape[1])]
                     for r in range(hm.shape[0])],
        "bbox": bbox,
        "max_val": round(max_val, 4),
    }
