"""
Skeleton fitting v2: forward-kinematics parameterization with multi-source fusion.

Key differences from v1:
  1. FK parameterization — instead of directly optimizing joints_3d (N,21,3):
       wrist_pos   (N, 3)    wrist world position, per frame
       wrist_r6d   (N, 6)    wrist orientation as 6D rotation, per frame
       bone_lengths (20,)    ONE set of lengths, constant across all frames
       flex_angs   (N, 20)   flexion angle per bone per frame (radians)
       abd_angs    (N, 20)   abduction angle per bone per frame (radians)
     Bone lengths are a single shared parameter — naturally constant.

  2. Multi-source input fusion with independent weights:
       w_mediapipe — MediaPipe 2D landmarks (both cameras)
       w_vision    — Apple Vision landmarks (if npz present)
       w_dlc       — DLC fingertip labels for joints 4 & 8 (if CSV present)
       w_hrnet     — HRNet heatmap peak locations (if npz present)

  3. Separate temporal smoothing for wrist position vs joint angles.

  4. Joint angle constraints from joint_angle_priors.json (toggleable).
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..config import get_settings
from .calibration import get_calibration_for_subject, triangulate_points
from .mano_data import _mano_dir, HAND_SKELETON
from .mediapipe_prelabel import load_mediapipe_prelabels
from .video import build_trial_map

logger = logging.getLogger(__name__)

BONES = HAND_SKELETON   # 20 bones in BFS order from wrist

# ──────────────────────────────────────────────────────────────────────────────
# Differentiable FK utilities (all run under torch.no_grad or with grad)
# ──────────────────────────────────────────────────────────────────────────────

def _rot6d_to_mat(r6d):
    """(N,6) → (N,3,3) rotation matrix via Gram-Schmidt.  Columns = basis vecs."""
    import torch
    import torch.nn.functional as F
    a1 = r6d[:, 0:3]                                          # (N,3)
    a2 = r6d[:, 3:6]                                          # (N,3)
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (a2 * b1).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)                  # (N,3,3) columns


def _local_rot(flex, abd):
    """Per-frame local rotation matrix from flex/abd angles.

    Flex rotates around local-x (palm up/down), abd around local-y (finger spread).
    Rest direction (flex=0, abd=0) = local +z.

    Returns (N,3,3) where columns are the rotated local basis in parent frame.
    """
    import torch
    cf, sf = torch.cos(flex), torch.sin(flex)
    ca, sa = torch.cos(abd),  torch.sin(abd)
    z = torch.zeros_like(cf)
    # Ry(abd) @ Rx(flex): column form
    col0 = torch.stack([ca,   z,  -sa], dim=-1)          # (N,3)
    col1 = torch.stack([sf*sa, cf, sf*ca], dim=-1)
    col2 = torch.stack([cf*sa, -sf, cf*ca], dim=-1)       # = bone direction
    return torch.stack([col0, col1, col2], dim=-1)         # (N,3,3)


def _forward_kinematics(wrist_pos, wrist_r6d, bone_lengths, flex_angs, abd_angs):
    """Reconstruct (N,21,3) joint positions from FK parameters.

    Args:
        wrist_pos:    (N,3)
        wrist_r6d:    (N,6)
        bone_lengths: (20,)
        flex_angs:    (N,20)
        abd_angs:     (N,20)
    Returns:
        joints: (N,21,3)
    """
    import torch
    # Use dicts to avoid in-place assignments (which break autograd).
    pos = {0: wrist_pos}                                       # joint → (N,3)
    rot = {0: _rot6d_to_mat(wrist_r6d)}                        # joint → (N,3,3)

    for bi, (parent_j, child_j) in enumerate(BONES):
        R_local = _local_rot(flex_angs[:, bi], abd_angs[:, bi]) # (N,3,3)
        R_child = rot[parent_j] @ R_local                       # (N,3,3)
        d_world = R_child[:, :, 2]                               # (N,3) bone dir
        pos[child_j] = pos[parent_j] + bone_lengths[bi] * d_world
        rot[child_j] = R_child

    return torch.stack([pos[j] for j in range(21)], dim=1)     # (N,21,3)


# ──────────────────────────────────────────────────────────────────────────────
# Initialisation helpers (numpy)
# ──────────────────────────────────────────────────────────────────────────────

def _compute_wrist_frame(joints_3d):
    """Return (N,3,3) rotation matrices (columns = x/y/z world-frame basis).

    z → toward middle MCP, y → dorsal (palm-normal), x → radial.
    Falls back gracefully for nan frames.
    """
    pos = joints_3d  # (N,21,3)

    z = pos[:, 9] - pos[:, 0]                            # to middle MCP
    v1 = pos[:, 5] - pos[:, 0]                           # to index MCP
    v2 = pos[:, 17] - pos[:, 0]                          # to pinky MCP
    y = np.cross(v2, v1)                                 # dorsal normal

    for v in (z, y):
        nrm = np.linalg.norm(v, axis=-1, keepdims=True).clip(1e-6)
        v /= nrm

    # orthogonalise y wrt z
    y -= (y * z).sum(-1, keepdims=True) * z
    nrm = np.linalg.norm(y, axis=-1, keepdims=True).clip(1e-6)
    y /= nrm
    x = np.cross(y, z)

    R = np.stack([x, y, z], axis=-1)                     # (N,3,3) columns

    # For nan frames set to identity
    nan_mask = np.isnan(R).any(axis=(-1,-2))
    R[nan_mask] = np.eye(3)
    return R                                              # (N,3,3)


def _inverse_fk_init(joints_3d, R_wrist):
    """Compute initial flex/abd angles from triangulated 3D + wrist frame.

    Returns:
        flex_init: (N,20) radians
        abd_init:  (N,20) radians
    """
    N = joints_3d.shape[0]
    flex_init = np.zeros((N, 20), dtype=np.float32)
    abd_init  = np.zeros((N, 20), dtype=np.float32)

    R_joint = np.tile(np.eye(3), (N, 21, 1, 1))         # (N,21,3,3)
    R_joint[:, 0] = R_wrist

    for bi, (parent_j, child_j) in enumerate(BONES):
        d_world = joints_3d[:, child_j] - joints_3d[:, parent_j]
        nrm = np.linalg.norm(d_world, axis=-1, keepdims=True).clip(1e-6)
        d_world = d_world / nrm

        # Transform to parent local frame: d_local = R_parent.T @ d_world
        d_local = np.einsum('nij,nj->ni',
                             R_joint[:, parent_j].transpose(0, 2, 1),
                             d_world)

        # Invert: d_local = [cf*sa, -sf, cf*ca]
        # sin(flex) = -d_local[:,1]
        sf = np.clip(-d_local[:, 1], -1.0, 1.0)
        flex = np.arcsin(sf)
        # sin(abd)/cos(abd) = d_local[:,0] / d_local[:,2]
        abd = np.arctan2(d_local[:, 0], d_local[:, 2])

        # Clamp to anatomical ranges
        flex = np.clip(flex, -np.pi / 4, np.pi * 5 / 6)
        abd  = np.clip(abd,  -np.pi / 3, np.pi / 3)

        # Handle nan frames
        nan_mask = np.isnan(flex) | np.isnan(abd)
        flex[nan_mask] = 0.0
        abd[nan_mask]  = 0.0

        flex_init[:, bi] = flex.astype(np.float32)
        abd_init[:, bi]  = abd.astype(np.float32)

        # Propagate child rotation for next bones
        cf, sf2 = np.cos(flex), np.sin(flex)
        ca, sa  = np.cos(abd),  np.sin(abd)
        col0 = np.stack([ca,      np.zeros_like(cf), -sa],     axis=-1)
        col1 = np.stack([sf2*sa,  cf,                sf2*ca],  axis=-1)
        col2 = np.stack([cf*sa,  -sf2,               cf*ca],   axis=-1)
        R_local = np.stack([col0, col1, col2], axis=-1)        # (N,3,3)
        R_joint[:, child_j] = R_joint[:, parent_j] @ R_local

    return flex_init, abd_init


# ──────────────────────────────────────────────────────────────────────────────
# Auxiliary data loaders
# ──────────────────────────────────────────────────────────────────────────────

def _load_vision_lm(subject_name, start_frame, N):
    """Load Vision landmarks → (N,21,2) per camera, or None."""
    try:
        path = get_settings().dlc_path / subject_name / "vision_prelabels.npz"
        if not path.exists():
            return None, None
        d = np.load(str(path))
        os_lm = d.get("OS_landmarks")
        od_lm = d.get("OD_landmarks")
        if os_lm is None or od_lm is None:
            return None, None
        end = min(start_frame + N, os_lm.shape[0])
        n = end - start_frame
        vis_L = np.full((N, 21, 2), np.nan, dtype=np.float32)
        vis_R = np.full((N, 21, 2), np.nan, dtype=np.float32)
        vis_L[:n] = os_lm[start_frame:end].astype(np.float32)
        vis_R[:n] = od_lm[start_frame:end].astype(np.float32)
        return vis_L, vis_R
    except Exception as e:
        logger.warning(f"Vision load failed: {e}")
        return None, None


def _load_dlc_tips(subject_name, start_frame, N):
    """Load DLC predictions for joints 4 (thumb tip) and 8 (index tip).

    Returns two (N,2) arrays per camera for each tip, or all-nan on failure.
    thumb_L, thumb_R, index_L, index_R — pixel coords.
    """
    empty = np.full((N, 2), np.nan, dtype=np.float32)
    try:
        from .dlc_predictions import get_dlc_predictions_for_session
        dlc_data = get_dlc_predictions_for_session(subject_name)
        if not dlc_data:
            return empty, empty, empty, empty
        settings = get_settings()
        cam_names = settings.camera_names
        cam_L = cam_names[0]
        cam_R = cam_names[1] if len(cam_names) > 1 else cam_names[0]

        def _get(cam, key):
            arr = dlc_data.get(cam, {}).get(key)
            if arr is None:
                return empty.copy()
            end = min(start_frame + N, arr.shape[0])
            n = end - start_frame
            out = empty.copy()
            out[:n] = arr[start_frame:end, :2].astype(np.float32)
            return out

        return (_get(cam_L, "thumb_tip"), _get(cam_R, "thumb_tip"),
                _get(cam_L, "index_tip"), _get(cam_R, "index_tip"))
    except Exception as e:
        logger.warning(f"DLC load failed: {e}")
        return empty, empty, empty, empty


def _load_hrnet_peaks(subject_name, trial_stem, start_frame, N, calib):
    """Extract HRNet heatmap peak locations as 2D targets.

    Returns (N,21,2) per camera mapped to image coords, or None.
    """
    try:
        mano_dir = _mano_dir(subject_name) / trial_stem
        hm_path  = mano_dir / "hrnet_w18_heatmaps.npz"
        crop_path = mano_dir / "hand_crop.json"
        if not hm_path.exists() or not crop_path.exists():
            return None, None

        with open(crop_path) as f:
            crop = json.load(f)

        hm = np.load(str(hm_path), mmap_mode="r")

        def _peaks_for_side(hm_key, bbox_key):
            bbox = crop.get(bbox_key)
            if bbox is None:
                return None
            if hm_key in hm:
                arr = hm[hm_key]
            elif "heatmaps" in hm:
                arr = hm["heatmaps"]
            else:
                return None
            # arr: (total_frames, 21, 64, 64)
            H, W = arr.shape[2], arr.shape[3]
            end = min(start_frame + N, arr.shape[0])
            n = end - start_frame
            out = np.full((N, 21, 2), np.nan, dtype=np.float32)
            for f in range(n):
                for j in range(21):
                    hmap = arr[start_frame + f, j]
                    idx = np.unravel_index(np.argmax(hmap), hmap.shape)
                    py, px = idx
                    x1, y1, x2, y2 = bbox
                    out[f, j, 0] = x1 + (px + 0.5) / W * (x2 - x1)
                    out[f, j, 1] = y1 + (py + 0.5) / H * (y2 - y1)
            return out

        hrnet_L = _peaks_for_side("heatmaps_L", "crop_L")
        hrnet_R = _peaks_for_side("heatmaps_R", "crop_R")
        return hrnet_L, hrnet_R
    except Exception as e:
        logger.warning(f"HRNet peaks load failed: {e}")
        return None, None


def _load_hrnet_heatmaps(subject_name, trial_stem, start_frame, N):
    """Load full HRNet heatmaps + per-frame bboxes for heatmap-value loss.

    Returns ``(heatmaps_L, bboxes_L, heatmaps_R, bboxes_R)`` or
    ``(None, None, None, None)``.

    * ``heatmaps``: ``(N, 21, H, W)`` float32, min-max normalized per (frame, joint).
    * ``bboxes``: ``(N, 4)`` float32 array in image pixel coords —
      one bbox per frame.  Older outputs that only stored a single
      union bbox are broadcast to every frame so callers get a uniform
      shape regardless of producer version.
    """
    from .hrnet_bbox import read_per_frame_bboxes
    try:
        mano_dir = _mano_dir(subject_name) / trial_stem
        hm_path  = mano_dir / "hrnet_w18_heatmaps.npz"
        crop_path = mano_dir / "hand_crop.json"
        if not hm_path.exists() or not crop_path.exists():
            return None, None, None, None

        with open(crop_path) as f:
            crop = json.load(f)

        hm = np.load(str(hm_path), mmap_mode="r")

        def _load_side(hm_key, bbox_key):
            arr = hm[hm_key] if hm_key in hm else (hm["heatmaps"] if "heatmaps" in hm else None)
            if arr is None:
                return None, None
            end = min(start_frame + N, arr.shape[0])
            n = end - start_frame
            # Load and min-max normalize per (frame, joint) — vectorized
            raw = np.array(arr[start_frame:end], dtype=np.float32)  # (n, 21, H, W)
            flat = raw.reshape(n, 21, -1)  # (n, 21, H*W)
            mn = flat.min(axis=2, keepdims=True)
            mx = flat.max(axis=2, keepdims=True)
            rng = mx - mn
            rng[rng < 1e-6] = 1  # avoid div by zero
            flat = (flat - mn) / rng
            out = np.zeros((N, 21, raw.shape[2], raw.shape[3]), dtype=np.float32)
            out[:n] = flat.reshape(n, 21, raw.shape[2], raw.shape[3])
            # Per-frame bbox slice for the same trial frame range.
            bboxes_full = read_per_frame_bboxes(crop, bbox_key, start_frame + N)
            if bboxes_full is None:
                return None, None
            bboxes = bboxes_full[start_frame:start_frame + N]
            if bboxes.shape[0] < N:
                # Pad with the last bbox if file is shorter than expected.
                pad = np.tile(bboxes[-1:], (N - bboxes.shape[0], 1))
                bboxes = np.concatenate([bboxes, pad], axis=0)
            return out, bboxes

        hm_L, bb_L = _load_side("heatmaps_L", "crop_L")
        hm_R, bb_R = _load_side("heatmaps_R", "crop_R")
        return hm_L, bb_L, hm_R, bb_R
    except Exception as e:
        logger.warning(f"HRNet heatmap load failed: {e}")
        return None, None, None, None


_JOINT_TYPE_GROUPS = [
    [1],                    # T_CMC (unique)
    [5, 9, 13, 17],         # MCP joints
    [2, 6, 10, 14, 18],     # PIP joints (incl T_MCP)
    [3, 7, 11, 15, 19],     # DIP joints (incl T_IP)
    [4, 8, 12, 16, 20],     # Tip joints
]


def _assign_hm_targets(proj_2d, hm_tensor, bbox):
    """Assign each joint its own heatmap's peak, using Hungarian to resolve conflicts.

    For each frame and joint-type group:
      1. Find each joint's individual heatmap peak (not MIP).
      2. If peaks are well-separated, use them directly.
      3. If two joints' peaks overlap, use Hungarian assignment on the
         combined set of per-joint peaks to resolve conflicts.
      4. Pass 2: refine using temporal neighbors as priors.

    Returns (N, 21, 2) target positions in pixel coords (NaN for unassigned/wrist).
    """
    import torch
    from scipy.optimize import linear_sum_assignment

    x1, y1, x2, y2 = bbox
    N_, J, H, W = hm_tensor.shape
    bw, bh = x2 - x1, y2 - y1
    targets = torch.full((N_, J, 2), float('nan'), device=hm_tensor.device)

    def _hm_peak(hm):
        """Find the peak of a single (H, W) heatmap, return (img_x, img_y, val)."""
        idx = hm.reshape(-1).argmax().item()
        py, px = idx // W, idx % W
        return (x1 + (px + 0.5) / W * bw, y1 + (py + 0.5) / H * bh, hm[py, px].item())

    def _find_multi_peaks(hm, n_peaks=3):
        """Find top n_peaks from a single (H, W) heatmap with NMS."""
        peaks = []
        work = hm.clone()
        for _ in range(n_peaks):
            idx = work.reshape(-1).argmax().item()
            py, px = idx // W, idx % W
            val = work[py, px].item()
            if val < 0.02:
                break
            peaks.append((x1 + (px + 0.5) / W * bw, y1 + (py + 0.5) / H * bh, val))
            r = 3
            y0, y1_ = max(0, py - r), min(H, py + r + 1)
            x0, x1_ = max(0, px - r), min(W, px + r + 1)
            work[y0:y1_, x0:x1_] = 0
        return peaks

    def _assign_group_per_joint(group, priors):
        """Assign using multiple per-joint peaks + Hungarian.

        For each joint, find top-3 peaks from its own heatmap.
        Build a candidate pool of all unique peaks across the group.
        Hungarian assigns joints to candidates, with cost based on:
          1. Distance from prior
          2. Joint's own heatmap value at the candidate
        """
        K = len(group)
        for f in range(N_):
            # Find top-3 peaks per joint
            all_candidates = []  # list of (img_x, img_y)
            joint_multi_peaks = {}  # ki → [(img_x, img_y, val), ...]
            for ki, j in enumerate(group):
                peaks = _find_multi_peaks(hm_tensor[f, j], n_peaks=3)
                joint_multi_peaks[ki] = peaks
                for px, py, _ in peaks:
                    # Add to candidate pool if not too close to existing
                    dup = False
                    for cx, cy in all_candidates:
                        if (px - cx)**2 + (py - cy)**2 < 4:  # ~2px
                            dup = True
                            break
                    if not dup:
                        all_candidates.append((px, py))

            if not all_candidates:
                continue

            cand_pts = torch.tensor(all_candidates, device=hm_tensor.device)  # (C, 2)
            prior_pts = priors[f]  # (K, 2)
            C = len(all_candidates)

            # Compute each joint's heatmap value at each candidate
            cand_vals = np.zeros((K, C))
            for ci in range(C):
                cpx = int((all_candidates[ci][0] - x1) / bw * W)
                cpy = int((all_candidates[ci][1] - y1) / bh * H)
                cpx = max(0, min(W-1, cpx))
                cpy = max(0, min(H-1, cpy))
                for ki in range(K):
                    cand_vals[ki, ci] = hm_tensor[f, group[ki], cpy, cpx].item()

            # Cost matrix: (K joints × C candidates)
            # Balance distance vs heatmap value so a strong distant peak can
            # beat a weak nearby peak. Use sqrt(dist) to reduce distance dominance.
            cost = np.full((K, C), 1e6)
            for ki in range(K):
                pt = prior_pts[ki]
                if torch.isnan(pt[0]):
                    continue
                for ci in range(C):
                    dist = np.sqrt(((pt - cand_pts[ci]) ** 2).sum().item())  # linear px, not squared
                    my_val = cand_vals[ki, ci]
                    # Penalty for weak heatmap signal (scaled to compete with distance)
                    weakness = (1.0 - my_val) * 80.0  # 80px equivalent for zero-val heatmap
                    cost[ki, ci] = dist + weakness

            row_ind, col_ind = linear_sum_assignment(cost)
            for ki, ci in zip(row_ind, col_ind):
                if cost[ki, ci] < 1e5:
                    targets[f, group[ki]] = cand_pts[ci]

    with torch.no_grad():
        for group in _JOINT_TYPE_GROUPS:
            K = len(group)
            if K == 1:
                j = group[0]
                for f in range(N_):
                    px, py, _ = _hm_peak(hm_tensor[f, j])
                    targets[f, j, 0] = px
                    targets[f, j, 1] = py
                continue

            # Pass 1: assign using projected 2D positions as priors
            _assign_group_per_joint(group, proj_2d[:, group])

            # Pass 2: refine using wider temporal neighbors as priors
            HALF_WIN = 7
            priors_pass2 = torch.full((N_, K, 2), float('nan'), device=hm_tensor.device)
            for ki, j in enumerate(group):
                for f in range(N_):
                    neighbors = []
                    for df in range(-HALF_WIN, HALF_WIN + 1):
                        nf = f + df
                        if nf < 0 or nf >= N_ or nf == f:
                            continue
                        t = targets[nf, j]
                        if not torch.isnan(t[0]):
                            neighbors.append(t)
                    if neighbors:
                        priors_pass2[f, ki] = torch.stack(neighbors).median(dim=0).values
                    else:
                        t1 = targets[f, j]
                        priors_pass2[f, ki] = t1 if not torch.isnan(t1[0]) else proj_2d[f, j]

            for j in group:
                targets[:, j] = float('nan')
            _assign_group_per_joint(group, priors_pass2)

            # Pass 3: topology check — each joint's assigned peak must be
            # closer to its own prior (MP position) than to any other joint's
            # prior in the group. If not, the assignment is likely wrong.
            # Use the original MP priors (not temporal), since temporal priors
            # can propagate errors.
            orig_priors = proj_2d[:, group]  # (N_, K, 2) — MP positions
            for f in range(N_):
                for ki, j in enumerate(group):
                    curr = targets[f, j]
                    my_prior = orig_priors[f, ki]
                    if torch.isnan(curr[0]) or torch.isnan(my_prior[0]):
                        continue
                    my_dist = ((curr - my_prior) ** 2).sum().sqrt().item()
                    # Check if this peak is closer to another joint's prior
                    for oi in range(K):
                        if oi == ki:
                            continue
                        other_prior = orig_priors[f, oi]
                        if torch.isnan(other_prior[0]):
                            continue
                        other_dist = ((curr - other_prior) ** 2).sum().sqrt().item()
                        if other_dist < my_dist - 10:  # significantly closer to other joint
                            # This peak probably belongs to the other joint.
                            # Try to find a peak from our own heatmap near our prior.
                            peaks = _find_multi_peaks(hm_tensor[f, j], n_peaks=3)
                            best_pt = None
                            best_score = float('inf')
                            for px, py, val in peaks:
                                pt = torch.tensor([px, py], device=curr.device)
                                d = ((pt - my_prior) ** 2).sum().sqrt().item()
                                score = d + (1.0 - val) * 40.0
                                if score < best_score:
                                    best_score = score
                                    best_pt = pt
                            if best_pt is not None:
                                targets[f, j] = best_pt
                            break

    return targets


def assign_hm_targets_stereo(proj_L, proj_R, hm_L, hm_R, bbox_L, bbox_R):
    """Assign heatmap peaks for both cameras, then fix Z-jump discrepancies.

    After independent per-camera assignment, triangulate peaks to 3D.
    When a joint's Z jumps >150mm between adjacent frames, try secondary
    peaks from each camera to find a combination that eliminates the jump.
    """
    import torch
    from .calibration import triangulate_points as _tri

    targets_L = _assign_hm_targets(proj_L, hm_L, bbox_L)
    targets_R = _assign_hm_targets(proj_R, hm_R, bbox_R) if hm_R is not None else None

    if targets_R is None:
        return targets_L, targets_R

    N_ = targets_L.shape[0]
    H, W = hm_L.shape[2], hm_L.shape[3]
    x1L, y1L, x2L, y2L = bbox_L
    x1R, y1R, x2R, y2R = bbox_R
    bwL, bhL = x2L - x1L, y2L - y1L
    bwR, bhR = x2R - x1R, y2R - y1R

    # Build a calibration dict for triangulation (numpy arrays)
    calib_np = None
    try:
        from .mano_data import _load_trial_calibration
        # We need the calibration but don't have subject/trial here.
        # Use the bbox to find it — actually, we'll pass it through.
        # For now, triangulate using a simple approach.
        pass
    except Exception:
        pass

    with torch.no_grad():
        _nan2_tpl = torch.tensor([float('nan'), float('nan')])
        def _temporal_smoothness(cam_targets, f, j, pos):
            """How well `pos` fits the joint's trajectory at neighboring frames.
            Returns median pixel distance to neighbors (lower = smoother)."""
            dists = []
            for df in [-3, -2, -1, 1, 2, 3]:
                nf = f + df
                if nf < 0 or nf >= N_:
                    continue
                prev = cam_targets[nf, j]
                if torch.isnan(prev[0]):
                    continue
                dists.append(float(np.sqrt((float(pos[0]) - float(prev[0]))**2 +
                                           (float(pos[1]) - float(prev[1]))**2)))
            return np.median(dists) if dists else 999.0

        # ── Phase 1: Stereo cross-camera validation ────────────────────────
        # A joint's cam_L and cam_R positions should have matching Y (within
        # calibration error) and disparity near the group's median.  Y mismatch
        # means one camera is mislabeled.  Before NaN-ing, check if the peak
        # is actually stereo-consistent with a DIFFERENT same-group joint on
        # the other camera — that indicates a label swap.
        Y_THRESH = 15.0  # pixels
        for group in _JOINT_TYPE_GROUPS:
            if len(group) < 1:
                continue
            # Compute this group's median disparity across frames
            disps = []
            for j in group:
                for f in range(N_):
                    pL = targets_L[f, j]
                    pR = targets_R[f, j]
                    if torch.isnan(pL[0]) or torch.isnan(pR[0]):
                        continue
                    if abs(float(pL[1]) - float(pR[1])) > Y_THRESH:
                        continue  # don't include likely-wrong pairs
                    disps.append(float(pL[0]) - float(pR[0]))
            group_med_disp = np.median(disps) if disps else None
            disp_tol = max(np.median(np.abs(np.array(disps) - group_med_disp)) * 3.0, 10.0) if disps else 30.0

            for f in range(N_):
                for j in group:
                    pL = targets_L[f, j]
                    pR = targets_R[f, j]
                    if torch.isnan(pL[0]) or torch.isnan(pR[0]):
                        continue
                    y_diff = abs(float(pL[1]) - float(pR[1]))
                    disp = float(pL[0]) - float(pR[0])
                    disp_bad = (group_med_disp is not None and
                                abs(disp - group_med_disp) > disp_tol)
                    if y_diff <= Y_THRESH and not disp_bad:
                        continue

                    # Stereo mismatch — try matching with other joints in group
                    best_swap = None   # ('L'|'R', other_j)
                    best_swap_score = y_diff
                    for k in group:
                        if k == j:
                            continue
                        qL = targets_L[f, k]
                        qR = targets_R[f, k]
                        # Does cam_L's j-peak match cam_R's k-peak (label-swap on L)?
                        if not torch.isnan(qR[0]):
                            y2 = abs(float(pL[1]) - float(qR[1]))
                            d2 = float(pL[0]) - float(qR[0])
                            d_ok = group_med_disp is None or abs(d2 - group_med_disp) <= disp_tol
                            if y2 <= Y_THRESH and d_ok and y2 < best_swap_score:
                                best_swap_score = y2
                                best_swap = ('L_is_k', k)
                        # Does cam_R's j-peak match cam_L's k-peak (label-swap on R)?
                        if not torch.isnan(qL[0]):
                            y2 = abs(float(pR[1]) - float(qL[1]))
                            d2 = float(qL[0]) - float(pR[0])
                            d_ok = group_med_disp is None or abs(d2 - group_med_disp) <= disp_tol
                            if y2 <= Y_THRESH and d_ok and y2 < best_swap_score:
                                best_swap_score = y2
                                best_swap = ('R_is_k', k)

                    # Decide which camera is wrong using temporal smoothness.
                    # The camera with the less temporally smooth position is wrong.
                    smooth_L = _temporal_smoothness(targets_L, f, j, pL)
                    smooth_R = _temporal_smoothness(targets_R, f, j, pR)
                    if best_swap is not None:
                        # A swap is possible — the stereo-consistent camera keeps
                        # its peak (likely correct), the other is mislabeled.
                        side, k = best_swap
                        if side == 'L_is_k':
                            # cam_L's j-peak actually belongs to k — mislabeled. NaN it.
                            targets_L[f, j] = _nan2_tpl.clone().to(targets_L.device)
                        else:
                            targets_R[f, j] = _nan2_tpl.clone().to(targets_R.device)
                    else:
                        # No swap candidate — NaN the less smooth side (or both if tied)
                        if smooth_L > smooth_R + 5:
                            targets_L[f, j] = _nan2_tpl.clone().to(targets_L.device)
                        elif smooth_R > smooth_L + 5:
                            targets_R[f, j] = _nan2_tpl.clone().to(targets_R.device)
                        else:
                            targets_L[f, j] = _nan2_tpl.clone().to(targets_L.device)
                            targets_R[f, j] = _nan2_tpl.clone().to(targets_R.device)

        # ── Phase 2: Overlap resolution ────────────────────────────────────
        # Any two joints assigned to the same peak is an error — one of them
        # is wrong. Decide which to NaN using temporal smoothness and
        # cross-camera correspondence, never raw heatmap values.
        # Exception: thumb tip (4) and index tip (8) can legitimately overlap
        # during finger tapping.
        OVERLAP_EXEMPT = {(4, 8), (8, 4)}
        OVERLAP_DIST = 10.0  # pixels — peaks closer than this are "shared"
        _nan2 = _nan2_tpl
        _nan2 = torch.tensor([float('nan'), float('nan')])

        for group in _JOINT_TYPE_GROUPS:
            if len(group) < 2:
                continue
            for f in range(N_):
                for cam_targets, other_targets in [(targets_L, targets_R), (targets_R, targets_L)]:
                    # Find all assigned joints in this group on this camera
                    assigned = []
                    for j in group:
                        t = cam_targets[f, j]
                        if not torch.isnan(t[0]):
                            assigned.append(j)

                    # Check all pairs for overlap
                    for i_a in range(len(assigned)):
                        ja = assigned[i_a]
                        ta = cam_targets[f, ja]
                        if torch.isnan(ta[0]):
                            continue
                        for i_b in range(i_a + 1, len(assigned)):
                            jb = assigned[i_b]
                            if (ja, jb) in OVERLAP_EXEMPT:
                                continue
                            tb = cam_targets[f, jb]
                            if torch.isnan(tb[0]):
                                continue
                            if ((ta - tb) ** 2).sum().sqrt().item() >= OVERLAP_DIST:
                                continue

                            # Overlap detected — always an error. Resolve using
                            # temporal smoothness and cross-camera correspondence.

                            # 1. Temporal smoothness: which joint was at a different
                            #    position on neighboring frames? That one jumped here.
                            score_a = _temporal_smoothness(cam_targets, f, ja, cam_targets[f, ja])
                            score_b = _temporal_smoothness(cam_targets, f, jb, cam_targets[f, jb])

                            # Higher score = bigger jump from neighbors = more likely wrong
                            if abs(score_a - score_b) > 5.0:
                                if score_a > score_b:
                                    cam_targets[f, ja] = _nan2.clone().to(cam_targets.device)
                                else:
                                    cam_targets[f, jb] = _nan2.clone().to(cam_targets.device)
                                continue

                            # 2. Cross-camera: if other camera has them separated,
                            #    the one whose other-camera Y is further from the
                            #    shared position is wrong here.
                            oa = other_targets[f, ja]
                            ob = other_targets[f, jb]
                            if not torch.isnan(oa[0]) and not torch.isnan(ob[0]):
                                shared_y = ta[1].item()
                                ya_diff = abs(oa[1].item() - shared_y)
                                yb_diff = abs(ob[1].item() - shared_y)
                                if ya_diff > yb_diff + 3:
                                    cam_targets[f, ja] = _nan2.clone().to(cam_targets.device)
                                    continue
                                elif yb_diff > ya_diff + 3:
                                    cam_targets[f, jb] = _nan2.clone().to(cam_targets.device)
                                    continue

                            # 3. Can't resolve — NaN both rather than leave a
                            #    likely-wrong assignment.
                            cam_targets[f, ja] = _nan2.clone().to(cam_targets.device)
                            cam_targets[f, jb] = _nan2.clone().to(cam_targets.device)

        # ── Phase 3: Secondary peak recovery ──────────────────────────────────
        # When a joint is NaN on one camera but assigned on the other, predict
        # the NaN'd side's position from stereo (median group disparity) and
        # search the joint's heatmap for a local peak there.  Only accept if
        # the new peak is stereo-consistent AND temporally smooth.
        for group in _JOINT_TYPE_GROUPS:
            # Recompute group median disparity (earlier phase may have NaN'd some)
            disps = []
            for j in group:
                for f in range(N_):
                    pL = targets_L[f, j]
                    pR = targets_R[f, j]
                    if torch.isnan(pL[0]) or torch.isnan(pR[0]):
                        continue
                    if abs(float(pL[1]) - float(pR[1])) > Y_THRESH:
                        continue
                    disps.append(float(pL[0]) - float(pR[0]))
            if not disps:
                continue
            group_med_disp = float(np.median(disps))

            def _search_peak(hm_t, f, j, bbox, pred_x, pred_y, search_r=5, min_val=0.05):
                bx1, by1, bx2, by2 = bbox
                bw, bh = bx2 - bx1, by2 - by1
                hx = (pred_x - bx1) / bw * W
                hy = (pred_y - by1) / bh * H
                if not (0 <= hx < W and 0 <= hy < H):
                    return None
                ix0, ix1 = max(0, int(hx) - search_r), min(W, int(hx) + search_r + 1)
                iy0, iy1 = max(0, int(hy) - search_r), min(H, int(hy) + search_r + 1)
                patch = hm_t[f, j, iy0:iy1, ix0:ix1]
                idx = patch.reshape(-1).argmax().item()
                val = patch.reshape(-1)[idx].item()
                if val < min_val:
                    return None
                py = iy0 + idx // (ix1 - ix0)
                px = ix0 + idx % (ix1 - ix0)
                return (bx1 + (px + 0.5) / W * bw, by1 + (py + 0.5) / H * bh, val)

            # For each NaN point, project from the opposite camera's peak using
            # median disparity + matching Y, then search the joint's heatmap.
            # If another same-group joint is already at that position, flag
            # the conflict — one of the two is mislabeled, so NaN both.
            for j in group:
                for f in range(N_):
                    pL = targets_L[f, j]
                    pR = targets_R[f, j]
                    nan_L = bool(torch.isnan(pL[0]))
                    nan_R = bool(torch.isnan(pR[0]))

                    # Only attempt recovery when exactly one side is NaN
                    if nan_L == nan_R:
                        continue

                    if nan_L:
                        pred_x = float(pR[0]) + group_med_disp
                        pred_y = float(pR[1])
                        hm_t, bbox, target_side = hm_L, bbox_L, targets_L
                    else:
                        pred_x = float(pL[0]) - group_med_disp
                        pred_y = float(pL[1])
                        hm_t, bbox, target_side = hm_R, bbox_R, targets_R

                    peak = _search_peak(hm_t, f, j, bbox, pred_x, pred_y)
                    if peak is None:
                        continue
                    new_x, new_y, _val = peak

                    # Check for conflict with other same-group joint labels
                    conflict_k = None
                    for k in group:
                        if k == j: continue
                        if (k, j) in {(4,8),(8,4)} or (j, k) in {(4,8),(8,4)}:
                            continue
                        other = target_side[f, k]
                        if torch.isnan(other[0]): continue
                        if ((float(other[0])-new_x)**2 + (float(other[1])-new_y)**2) ** 0.5 < 10.0:
                            conflict_k = k
                            break

                    if conflict_k is not None:
                        # Another joint is already labeled here — one of them is
                        # mislabeled (either the existing label on this camera,
                        # or the reference peak on the other camera). NaN both.
                        target_side[f, conflict_k] = _nan2_tpl.clone().to(target_side.device)
                        # Also NaN the reference joint on the other camera
                        if nan_L:
                            targets_R[f, j] = _nan2_tpl.clone().to(targets_R.device)
                        else:
                            targets_L[f, j] = _nan2_tpl.clone().to(targets_L.device)
                    else:
                        target_side[f, j] = torch.tensor(
                            [new_x, new_y], device=target_side.device)

        # ── Phase 4: Temporal outlier detection ───────────────────────────
        # A joint's position at frame f should be close to its temporally
        # interpolated value from the nearest valid neighbors.  If the
        # deviation is large (>30px), the label is likely wrong — often it
        # was stolen by a neighboring finger's joint at that frame.
        for group in _JOINT_TYPE_GROUPS:
            for cam_targets in [targets_L, targets_R]:
                for j in group:
                    for f in range(N_):
                        p = cam_targets[f, j]
                        if torch.isnan(p[0]):
                            continue
                        # Get nearest valid neighbors in ±3 frames
                        prev_f = next((f-d for d in range(1,4) if f-d>=0
                                       and not torch.isnan(cam_targets[f-d,j,0])), None)
                        next_f = next((f+d for d in range(1,4) if f+d<N_
                                       and not torch.isnan(cam_targets[f+d,j,0])), None)
                        if prev_f is None or next_f is None:
                            continue
                        a = cam_targets[prev_f, j].cpu().numpy()
                        b = cam_targets[next_f, j].cpu().numpy()
                        t = (f - prev_f) / (next_f - prev_f)
                        exp_x = a[0]*(1-t) + b[0]*t
                        exp_y = a[1]*(1-t) + b[1]*t
                        dev = ((float(p[0])-exp_x)**2 + (float(p[1])-exp_y)**2) ** 0.5
                        if dev <= 30:
                            continue
                        # Outlier — check if this peak better matches another
                        # same-group joint's trajectory at this frame.
                        better_match = False
                        for k in group:
                            if k == j: continue
                            k_prev = next((f-d for d in range(1,4) if f-d>=0
                                           and not torch.isnan(cam_targets[f-d,k,0])), None)
                            k_next = next((f+d for d in range(1,4) if f+d<N_
                                           and not torch.isnan(cam_targets[f+d,k,0])), None)
                            if k_prev is None and k_next is None:
                                continue
                            # Interpolate or extrapolate k
                            if k_prev is not None and k_next is not None:
                                ka = cam_targets[k_prev, k].cpu().numpy()
                                kb = cam_targets[k_next, k].cpu().numpy()
                                kt = (f - k_prev) / (k_next - k_prev)
                                kx = ka[0]*(1-kt) + kb[0]*kt
                                ky = ka[1]*(1-kt) + kb[1]*kt
                            elif k_prev is not None:
                                ka = cam_targets[k_prev, k].cpu().numpy()
                                kx, ky = float(ka[0]), float(ka[1])
                            else:
                                ka = cam_targets[k_next, k].cpu().numpy()
                                kx, ky = float(ka[0]), float(ka[1])
                            k_dev = ((float(p[0])-kx)**2 + (float(p[1])-ky)**2) ** 0.5
                            if k_dev < dev - 10:
                                better_match = True
                                break
                        if better_match:
                            cam_targets[f, j] = _nan2_tpl.clone().to(cam_targets.device)

        # ── Disparity-based stereo validation ──────────────────────────────────
        Z_JUMP_THRESH = 150.0  # mm — max plausible single-frame Z change

        for j in range(1, 21):  # skip wrist
            # Triangulate all frames for this joint
            pk_L = targets_L[:, j].cpu().numpy()  # (N, 2)
            pk_R = targets_R[:, j].cpu().numpy()

            # Simple stereo Z estimate: Z ∝ baseline / disparity
            # disparity = pk_L[:, 0] - pk_R[:, 0] (x difference between cameras)
            disparity = pk_L[:, 0] - pk_R[:, 0]
            disparity[np.isnan(disparity)] = 0
            disparity[disparity < 1] = 1  # prevent div by zero

            # We don't need exact Z — just relative changes to detect jumps
            inv_disp = 1.0 / disparity  # proportional to Z

            # Global median disparity and MAD for this joint
            valid_mask = ~np.isnan(pk_L[:, 0]) & ~np.isnan(pk_R[:, 0]) & (disparity > 1)
            valid_disp = disparity[valid_mask]
            if len(valid_disp) > 10:
                global_med_disp = np.median(valid_disp)
                global_mad = np.median(np.abs(valid_disp - global_med_disp))
                global_sigma = max(global_mad * 1.4826, 2.0)
                # Frame-to-frame threshold (for detecting sudden jumps)
                diffs = np.abs(np.diff(valid_disp))
                med_diff = np.median(diffs)
                mad_diff = np.median(np.abs(diffs - med_diff))
                jump_thresh = max(med_diff + 4 * max(mad_diff * 1.4826, 1.0), 20.0)
            else:
                global_med_disp = np.median(valid_disp) if len(valid_disp) > 0 else 50.0
                global_sigma = 10.0
                jump_thresh = 20.0

            for f in range(1, N_):
                if np.isnan(pk_L[f, 0]) or np.isnan(pk_R[f, 0]):
                    continue

                # Find the most recent valid reference frame (look back up to 3 frames)
                ref_f = None
                for lookback in range(1, 4):
                    rf = f - lookback
                    if rf < 0:
                        break
                    if not np.isnan(pk_L[rf, 0]) and not np.isnan(pk_R[rf, 0]) and disparity[rf] > 1:
                        ref_f = rf
                        break
                if ref_f is None:
                    ref_f = max(0, f - 1)  # fallback

                prev_disp = disparity[ref_f] if not np.isnan(disparity[ref_f]) and disparity[ref_f] > 1 else global_med_disp

                # Check: disparity far from global median OR sudden frame-to-frame jump
                # OR Y-coordinate mismatch between cameras (stereo cameras are horizontally
                # aligned, so Y should match within calibration error ~10px)
                disp_from_global = abs(disparity[f] - global_med_disp)
                disp_change = abs(disparity[f] - prev_disp)
                y_diff = abs(pk_L[f, 1] - pk_R[f, 1])
                is_global_outlier = disp_from_global > 4 * global_sigma
                is_frame_jump = disp_change > jump_thresh
                is_y_mismatch = y_diff > 20  # >20px Y difference = wrong assignment

                # Check if one camera jumped while the other didn't.
                # Validate by checking same-finger joints: if the tip jumped
                # but the DIP/PIP on the same finger barely moved, it's a
                # tracking error, not real motion.
                # Finger chains: tip → DIP → PIP → MCP
                _FINGER_CHAIN = {
                    4: [3, 2, 1],     # T_Tip → T_IP → T_MCP → T_CMC
                    8: [7, 6, 5],     # I_Tip → I_DIP → I_PIP → I_MCP
                    12: [11, 10, 9],  # M_Tip
                    16: [15, 14, 13], # R_Tip
                    20: [19, 18, 17], # P_Tip
                }
                dx_L = pk_L[f, 0] - pk_L[ref_f, 0]
                dx_R = pk_R[f, 0] - pk_R[ref_f, 0]
                single_cam_jump = False

                def _check_finger_motion(targets_cam, cam_dx, j_idx):
                    """Check if same-finger joints moved consistently with the tip."""
                    chain = _FINGER_CHAIN.get(j_idx, [])
                    if not chain:
                        return False  # not a tip joint
                    finger_dx = []
                    for fj in chain[:2]:  # check DIP and PIP
                        tc = targets_cam[f, fj].cpu().numpy()
                        tr = targets_cam[ref_f, fj].cpu().numpy()
                        if not np.isnan(tc[0]) and not np.isnan(tr[0]):
                            finger_dx.append(tc[0] - tr[0])
                    if not finger_dx:
                        return False
                    # If the tip moved much more than its DIP/PIP, it's a tracking error
                    finger_med = np.median(finger_dx)
                    return abs(cam_dx - finger_med) > 15

                if abs(dx_L) > 20 and abs(dx_R) < abs(dx_L) * 0.5:
                    if _check_finger_motion(targets_L, dx_L, j):
                        single_cam_jump = True
                elif abs(dx_R) > 20 and abs(dx_L) < abs(dx_R) * 0.5:
                    if _check_finger_motion(targets_R, dx_R, j):
                        single_cam_jump = True

                if not is_global_outlier and not is_frame_jump and not single_cam_jump and not is_y_mismatch:
                    continue

                # Big disparity jump — try secondary peaks from each camera
                # to find the combination with smallest disparity change

                # Get top-3 peaks from each camera for this joint
                def _get_peaks(hm_tensor, f_idx, j_idx, bbox):
                    bx1, by1, bx2, by2 = bbox
                    bw, bh = bx2 - bx1, by2 - by1
                    hm = hm_tensor[f_idx, j_idx].clone()
                    peaks = []
                    for _ in range(3):
                        idx = hm.reshape(-1).argmax().item()
                        py, px = idx // W, idx % W
                        val = hm[py, px].item()
                        if val < 0.02:
                            break
                        peaks.append(np.array([bx1 + (px + 0.5) / W * bw,
                                               by1 + (py + 0.5) / H * bh]))
                        r = 3
                        y0, y1_ = max(0, py-r), min(H, py+r+1)
                        x0, x1_ = max(0, px-r), min(W, px+r+1)
                        hm[y0:y1_, x0:x1_] = 0
                    return peaks

                cands_L = _get_peaks(hm_L, f, j, bbox_L)
                cands_R = _get_peaks(hm_R, f, j, bbox_R)

                if not cands_L or not cands_R:
                    continue

                # Identify which camera jumped (relative to reference frame)
                dx_L = abs(pk_L[f, 0] - pk_L[ref_f, 0])
                dx_R = abs(pk_R[f, 0] - pk_R[ref_f, 0])

                if dx_L > dx_R * 1.5:
                    jumping_cam, stable_cam = 'L', 'R'
                elif dx_R > dx_L * 1.5:
                    jumping_cam, stable_cam = 'R', 'L'
                else:
                    jumping_cam = None  # both jumped equally

                if jumping_cam:
                    # Predict where the jumping camera's peak should be:
                    # translate the reference frame's peak by the stable camera's motion
                    if jumping_cam == 'L':
                        stable_motion = pk_R[f] - pk_R[ref_f]
                        predicted = pk_L[ref_f] + stable_motion
                        hm_jump = hm_L
                        bbox_jump = bbox_L
                    else:
                        stable_motion = pk_L[f] - pk_L[ref_f]
                        predicted = pk_R[ref_f] + stable_motion
                        hm_jump = hm_R
                        bbox_jump = bbox_R

                    # Search for a local peak near the predicted position
                    bx1, by1, bx2, by2 = bbox_jump
                    bw_j, bh_j = bx2 - bx1, by2 - by1
                    pred_hx = int((predicted[0] - bx1) / bw_j * W)
                    pred_hy = int((predicted[1] - by1) / bh_j * H)

                    # Sample a local region around predicted position
                    hm_j = hm_jump[f, j].clone()
                    search_r = 5  # heatmap pixels (~20 image pixels)
                    best_val = 0
                    best_px, best_py = pred_hx, pred_hy
                    for dy in range(-search_r, search_r + 1):
                        for dx in range(-search_r, search_r + 1):
                            sy, sx = pred_hy + dy, pred_hx + dx
                            if 0 <= sy < H and 0 <= sx < W:
                                v = hm_j[sy, sx].item()
                                if v > best_val:
                                    best_val = v
                                    best_px, best_py = sx, sy

                    if best_val > 0.05:
                        new_pt = np.array([bx1 + (best_px + 0.5) / W * bw_j,
                                           by1 + (best_py + 0.5) / H * bh_j])
                        # Check new point's Y consistency with stable camera
                        stable_y = pk_R[f, 1] if jumping_cam == 'L' else pk_L[f, 1]
                        new_y_diff = abs(new_pt[1] - stable_y)
                        if jumping_cam == 'L':
                            new_disp = new_pt[0] - pk_R[f, 0]
                        else:
                            new_disp = pk_L[f, 0] - new_pt[0]
                        new_dc = abs(new_disp - prev_disp)
                        max_issue = max(disp_change, disp_from_global)
                        if new_dc < max_issue * 0.7 and new_y_diff < 15:
                            if jumping_cam == 'L':
                                targets_L[f, j] = torch.tensor(new_pt, device=targets_L.device)
                            else:
                                targets_R[f, j] = torch.tensor(new_pt, device=targets_R.device)
                            disparity[f] = new_disp
                            continue

                    # No good local peak found — mark as NaN
                    if jumping_cam == 'L':
                        targets_L[f, j] = torch.tensor([float('nan'), float('nan')], device=targets_L.device)
                    else:
                        targets_R[f, j] = torch.tensor([float('nan'), float('nan')], device=targets_R.device)
                    disparity[f] = prev_disp
                else:
                    # Both jumped — try all combos
                    best_combo = None
                    best_disp_change = disp_change
                    for cl in cands_L:
                        for cr in cands_R:
                            d = cl[0] - cr[0]
                            if d < 1:
                                continue
                            dc = abs(d - prev_disp)
                            if dc < best_disp_change:
                                best_disp_change = dc
                                best_combo = (cl, cr)
                    if best_combo is not None and best_disp_change < disp_change * 0.5:
                        targets_L[f, j] = torch.tensor(best_combo[0], device=targets_L.device)
                        targets_R[f, j] = torch.tensor(best_combo[1], device=targets_R.device)
                        disparity[f] = best_combo[0][0] - best_combo[1][0]

    return targets_L, targets_R


# ──────────────────────────────────────────────────────────────────────────────
# Main v2 fitting entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_v2_fitting(
    subject_name: str,
    trial_stem: str,
    cancel_event: threading.Event | None = None,
    progress_callback: Callable[[float], None] | None = None,
    w_mediapipe: float = 1.0,
    w_vision: float = 1.0,
    w_dlc: float = 1.0,
    w_hrnet: float = 1.0,
    hrnet_fingertips_only: bool = False,
    w_bone: float = 0.5,
    w_smooth_wrist: float = 1.0,
    w_smooth_xy: float = 1.0,
    w_smooth_z: float = 1.0,
    w_smooth_angles: float = 1.0,
    use_angle_constraints: bool = True,
    w_constraints: float = 2.0,
    w_v1_ref: float = 0.5,
) -> dict[str, Any]:
    """Fit the v2 skeleton for one trial.

    Uses FK parameterisation with constant bone lengths and separate temporal
    smoothing for wrist position and joint angles.

    Returns dict with output path and quality metrics.
    """
    import torch
    import torch.nn.functional as F  # noqa: F401 — used implicitly in _rot6d_to_mat

    device = torch.device("cpu")

    def report(pct):
        if progress_callback:
            progress_callback(pct)

    def cancelled():
        return cancel_event is not None and cancel_event.is_set()

    report(0)
    logger.info(f"[Skeleton Fit v2] {subject_name}/{trial_stem}")

    # ── Trial info ──────────────────────────────────────────────
    trials = build_trial_map(subject_name)
    trial_info = next((t for t in trials if t["trial_name"] == trial_stem), None)
    if trial_info is None:
        raise ValueError(f"Trial {trial_stem} not found for {subject_name}")

    n_frames    = trial_info["frame_count"]
    start_frame = trial_info.get("start_frame", 0)

    # ── MediaPipe ───────────────────────────────────────────────
    prelabels = load_mediapipe_prelabels(subject_name)
    if prelabels is None:
        raise ValueError(f"No MediaPipe prelabels for {subject_name}")
    os_lm = prelabels["OS_landmarks"]
    od_lm = prelabels["OD_landmarks"]
    end = min(start_frame + n_frames, os_lm.shape[0])
    mp_L = os_lm[start_frame:end].copy().astype(np.float32)
    mp_R = od_lm[start_frame:end].copy().astype(np.float32)
    N = mp_L.shape[0]

    # ── Calibration ─────────────────────────────────────────────
    # Use _load_trial_calibration (same as display code) so the fit
    # uses the same calibration that the viewer uses for projection.
    from .mano_data import _load_trial_calibration
    calib = _load_trial_calibration(subject_name, trial_stem)
    if calib is None:
        calib = get_calibration_for_subject(subject_name)
    if calib is None:
        raise ValueError(f"No stereo calibration for {subject_name}")

    report(3)
    if cancelled():
        return {"cancelled": True}

    # ── Auxiliary sources ───────────────────────────────────────
    vis_L, vis_R = _load_vision_lm(subject_name, start_frame, N)
    has_vision = vis_L is not None and not np.isnan(vis_L).all()

    th_L, th_R, ix_L, ix_R = _load_dlc_tips(subject_name, start_frame, N)
    has_dlc = (not np.isnan(th_L).all()) or (not np.isnan(ix_L).all())

    hrnet_L, hrnet_R = _load_hrnet_peaks(subject_name, trial_stem, start_frame, N, calib)
    has_hrnet = hrnet_L is not None and not np.isnan(hrnet_L).all()

    # Load full heatmaps for heatmap-value loss.
    # ``_load_hrnet_heatmaps`` now returns per-frame bboxes ``(N, 4)``;
    # the v3 fit's heatmap-loss path was originally written for a single
    # static bbox.  Collapse the per-frame array to its union so existing
    # internals keep working — losing some per-frame resolution accuracy
    # but matching prior behaviour for now.
    hm_maps_L, _hm_bb_L_pf, hm_maps_R, _hm_bb_R_pf = _load_hrnet_heatmaps(
        subject_name, trial_stem, start_frame, N)
    def _union_of_pf(pf):
        if pf is None: return None
        v = ~np.isnan(pf[:, 0])
        if not v.any(): return None
        return [float(pf[v, 0].min()), float(pf[v, 1].min()),
                float(pf[v, 2].max()), float(pf[v, 3].max())]
    hm_bbox_L = _union_of_pf(_hm_bb_L_pf)
    hm_bbox_R = _union_of_pf(_hm_bb_R_pf)
    has_heatmaps = hm_maps_L is not None

    logger.info(f"  Sources: MP ✓  Vision {'✓' if has_vision else '✗'}  "
                f"DLC {'✓' if has_dlc else '✗'}  HRNet {'✓' if has_hrnet else '✗'}  "
                f"Heatmaps {'✓' if has_heatmaps else '✗'}")
    logger.info(f"  w_hrnet={w_hrnet}  has_heatmaps={has_heatmaps}")

    # ── Triangulate MediaPipe to 3D ─────────────────────────────
    logger.info(f"  Triangulating {N} frames...")
    mp_3d = np.full((N, 21, 3), np.nan, dtype=np.float32)
    for j in range(21):
        mp_3d[:, j] = triangulate_points(mp_L[:, j], mp_R[:, j], calib).astype(np.float32)

    # ── Calibration quality: round-trip reprojection error per camera ──
    # Previously we downweighted the less-accurate camera's 2D loss, but that
    # shifts the reprojection optimum along the trusted camera's ray and
    # biases 3D depth.  Now we weight both cameras equally and rely on the
    # refined per-trial calibration to keep round-trip error balanced.
    from .mano_data import _project_to_2d
    _rp_L = _project_to_2d(mp_3d, calib["K1"], calib["dist1"],
                            np.eye(3, dtype=np.float64),
                            np.zeros((3, 1), dtype=np.float64))
    _rp_R = _project_to_2d(mp_3d, calib["K2"], calib["dist2"],
                            calib["R"], calib["T"])
    _rt_err_L = np.nanmean(np.abs(_rp_L - mp_L))
    _rt_err_R = np.nanmean(np.abs(_rp_R - mp_R))
    cam_w_L = 1.0
    cam_w_R = 1.0
    logger.info(f"  Calib roundtrip: L={_rt_err_L:.1f}px  R={_rt_err_R:.1f}px  "
                f"(equal weighting)")

    valid_mask = (~np.isnan(mp_L[:, 0, 0])
                  & ~np.isnan(mp_R[:, 0, 0])
                  & ~np.isnan(mp_3d[:, 0, 0]))
    valid_idx  = np.where(valid_mask)[0]
    n_valid    = len(valid_idx)

    if n_valid < 10:
        raise ValueError(f"Only {n_valid} valid frames (need ≥ 10)")

    logger.info(f"  {n_valid}/{N} valid frames")
    report(8)

    # ── Robust bone-length targets (from triangulated MP) ───────
    bone_lengths_all = np.zeros((n_valid, 20), dtype=np.float32)
    for b, (j1, j2) in enumerate(BONES):
        diff = mp_3d[valid_idx, j2] - mp_3d[valid_idx, j1]
        bone_lengths_all[:, b] = np.linalg.norm(diff, axis=1)

    target_bone_lengths = np.zeros(20, dtype=np.float32)
    for b in range(20):
        bl = bone_lengths_all[:, b]
        bl = bl[~np.isnan(bl) & (bl > 0)]
        if len(bl) < 5:
            target_bone_lengths[b] = 30.0  # fallback
            continue
        med = np.median(bl)
        mad = np.median(np.abs(bl - med))
        sigma = max(mad * 1.4826, 0.1)
        inlier = np.abs(bl - med) < 3.0 * sigma
        bl_clean = bl[inlier] if inlier.sum() >= 5 else bl
        target_bone_lengths[b] = float(np.median(bl_clean))

    logger.info(f"  Bone targets: {target_bone_lengths[target_bone_lengths > 0].min():.1f}–"
                f"{target_bone_lengths.max():.1f} mm")

    # ── Inter-MCP span constraints (constant length, like bones) ──
    # All four inter-MCP distances are held constant throughout the video.
    # Thumb-Index uses point-to-segment distance (T_CMC → Wrist–I_MCP).
    # The other three are simple point-to-point.
    MCP_P2P_PAIRS = [(5, 9), (9, 13), (13, 17)]
    MCP_ALL_NAMES = ["Thumb-Index", "Index-Middle", "Middle-Ring", "Ring-Pinky"]
    from .mano_data import load_angle_priors, _point_to_segment_dist
    _priors_data = load_angle_priors()

    # Compute constant targets from data (robust median, same as bone lengths)
    # Thumb-Index: point-to-segment distance
    ti_raw = _point_to_segment_dist(
        mp_3d[valid_idx, 1], mp_3d[valid_idx, 0], mp_3d[valid_idx, 5])
    ti_raw = ti_raw[~np.isnan(ti_raw) & (ti_raw > 0)]
    if len(ti_raw) >= 5:
        med = np.median(ti_raw)
        mad = np.median(np.abs(ti_raw - med))
        sigma = max(mad * 1.4826, 0.1)
        inlier = np.abs(ti_raw - med) < 3.0 * sigma
        ti_target = float(np.median(ti_raw[inlier]) if inlier.sum() >= 5 else np.median(ti_raw))
    else:
        ti_target = float(np.median(ti_raw)) if len(ti_raw) > 0 else 20.0

    # Other three: point-to-point
    mcp_p2p_targets = np.zeros(3, dtype=np.float32)
    for i, (ja, jb) in enumerate(MCP_P2P_PAIRS):
        d = np.linalg.norm(mp_3d[valid_idx, jb] - mp_3d[valid_idx, ja], axis=1)
        d = d[~np.isnan(d) & (d > 0)]
        if len(d) >= 5:
            med = np.median(d)
            mad = np.median(np.abs(d - med))
            sigma = max(mad * 1.4826, 0.1)
            inlier = np.abs(d - med) < 3.0 * sigma
            d_clean = d[inlier] if inlier.sum() >= 5 else d
            mcp_p2p_targets[i] = float(np.median(d_clean))
        else:
            mcp_p2p_targets[i] = float(np.median(d)) if len(d) > 0 else 15.0

    mcp_all_targets = [ti_target] + mcp_p2p_targets.tolist()
    logger.info(f"  MCP span targets: {dict(zip(MCP_ALL_NAMES, mcp_all_targets))}")

    # ── Inter-MCP chain angle targets (constant, like bone lengths) ──
    # Angles between adjacent inter-MCP segments to prevent knuckle drift.
    # Angle at M_MCP between (I_MCP→M_MCP) and (M_MCP→R_MCP)
    # Angle at R_MCP between (M_MCP→R_MCP) and (R_MCP→P_MCP)
    MCP_CHAIN_ANGLES = [(5, 9, 13), (9, 13, 17)]  # (j_a, j_vertex, j_b)
    MCP_CHAIN_NAMES = ["I-M-R angle", "M-R-P angle"]
    mcp_chain_targets = np.zeros(2, dtype=np.float32)
    for i, (ja, jv, jb) in enumerate(MCP_CHAIN_ANGLES):
        va = mp_3d[valid_idx, ja] - mp_3d[valid_idx, jv]  # vertex→ja
        vb = mp_3d[valid_idx, jb] - mp_3d[valid_idx, jv]  # vertex→jb
        na = np.linalg.norm(va, axis=1, keepdims=True)
        nb = np.linalg.norm(vb, axis=1, keepdims=True)
        valid_len = (na[:, 0] > 1e-6) & (nb[:, 0] > 1e-6)
        cos_ang = np.sum(va * vb, axis=1) / (na[:, 0] * nb[:, 0])
        cos_ang = np.clip(cos_ang, -1, 1)
        ang = np.degrees(np.arccos(cos_ang))
        ang = ang[valid_len & ~np.isnan(ang)]
        if len(ang) >= 5:
            med = np.median(ang)
            mad = np.median(np.abs(ang - med))
            sigma = max(mad * 1.4826, 0.1)
            inlier = np.abs(ang - med) < 3.0 * sigma
            mcp_chain_targets[i] = float(np.median(ang[inlier]) if inlier.sum() >= 5 else np.median(ang))
        else:
            mcp_chain_targets[i] = float(np.median(ang)) if len(ang) > 0 else 160.0
    logger.info(f"  MCP chain angles: {dict(zip(MCP_CHAIN_NAMES, mcp_chain_targets.tolist()))}")

    # ── Per-joint per-frame confidence weights ───────────────────
    # Z is ~7× noisier than X/Y in stereo triangulation.  Problems are
    # joint-specific: a bad index-tip detection shouldn't penalise the
    # wrist.  For each (frame, joint), compare its Z-residual from a
    # temporal baseline to its X/Y residual.  Joints with Z >> X/Y on
    # a given frame are triangulation artifacts and get downweighted.
    joint_weights = np.ones((n_valid, 21), dtype=np.float32)

    tri_valid = mp_3d[valid_idx]  # (n_valid, 21, 3)

    if n_valid > 12:
        # --- Signal 1: Z residual from temporal baseline ---
        HALF_WIN = 5
        resid = np.full((n_valid, 21, 3), np.nan)
        for i in range(n_valid):
            lo, hi = max(0, i - HALF_WIN), min(n_valid, i + HALF_WIN + 1)
            with np.errstate(all='ignore'):
                baseline = np.nanmedian(tri_valid[lo:hi], axis=0)
            resid[i] = np.abs(tri_valid[i] - baseline)

        z_resid = resid[:, :, 2]                                   # (n_valid, 21)
        xy_resid = np.maximum(resid[:, :, 0], resid[:, :, 1])      # (n_valid, 21)

        with np.errstate(divide='ignore', invalid='ignore'):
            z_ratio = z_resid / np.maximum(xy_resid, 0.3)

        RATIO_THRESH = 3.0
        ABS_Z_THRESH = 8.0  # mm
        w_r = np.where(z_ratio > RATIO_THRESH,
                        RATIO_THRESH / np.maximum(z_ratio, RATIO_THRESH), 1.0)
        w_a = np.where(z_resid > ABS_Z_THRESH,
                        ABS_Z_THRESH / np.maximum(z_resid, ABS_Z_THRESH), 1.0)
        joint_weights = np.clip(np.minimum(w_r, w_a), 0.01, 1.0)

        # --- Signal 2: 2D velocity outliers (both cameras) ---
        # A 15+ px jump on a single joint between frames is almost certainly
        # a mislabel.  This catches multi-frame bursts that fool the Z median.
        D2D_THRESH = 15.0  # pixels
        mp_valid_L = mp_L[valid_idx]  # (n_valid, 21, 2)
        mp_valid_R = mp_R[valid_idx]
        for mp_2d in [mp_valid_L, mp_valid_R]:
            d2d = np.sqrt(np.sum(np.diff(mp_2d, axis=0) ** 2, axis=2))  # (n_valid-1, 21)
            for i in range(d2d.shape[0]):
                for j in range(21):
                    if np.isnan(d2d[i, j]):
                        continue
                    if d2d[i, j] > D2D_THRESH:
                        w = D2D_THRESH / d2d[i, j]
                        joint_weights[i, j]   = min(joint_weights[i, j],   w)
                        joint_weights[i+1, j] = min(joint_weights[i+1, j], w)

        joint_weights = np.clip(joint_weights, 0.01, 1.0)

    n_bad = (joint_weights < 0.5).sum()
    n_total = n_valid * 21
    logger.info(f"  Joint weights: {n_bad}/{n_total} ({100*n_bad/n_total:.0f}%) "
                f"joint-frames downweighted (<0.5)")

    report(13)

    # ── Fill NaN 3D positions for initialisation ────────────────
    init_3d = mp_3d[valid_idx].copy()
    for i in range(n_valid):
        for j in range(21):
            if np.isnan(init_3d[i, j, 0]):
                init_3d[i, j] = init_3d[i, 0] if not np.isnan(init_3d[i, 0, 0]) else np.array([0, 0, 500])

    # ── Load v1 skeleton fit angles as reference ─────────────────
    # The v1 fit (direct 3D optimization) is often more stable than v2
    # even if less anatomically correct.  Penalise v2 for deviating from
    # v1 angles to prevent wild solutions.
    v1_angles_ref = None  # will be (n_valid, n_angles, 2) tensor if available
    v1_path = _mano_dir(subject_name) / trial_stem / "mano_fit.npz"
    if v1_path.exists():
        from .mano_data import _compute_angles, FLEX_ANGLE_OPTIONS, _load_mano_npz
        v1_data = _load_mano_npz(str(v1_path))
        v1_j3d = v1_data["joints_3d"]  # (N_total, 21, 3)
        v1_valid = v1_j3d[valid_idx]   # (n_valid, 21, 3)
        v1_wrist_ref = v1_valid[:, 0].astype(np.float32)  # (n_valid, 3)
        # Compute angles using the same decomposition as plotting
        v1_angle_dict = _compute_angles(v1_valid)
        # Pack into a (n_valid, 15, 2) array: [flex, abd] per joint
        n_ang = len(FLEX_ANGLE_OPTIONS)
        v1_ang = np.full((n_valid, n_ang, 2), np.nan, dtype=np.float32)
        for ai, (fname, p, j, c) in enumerate(FLEX_ANGLE_OPTIONS):
            aname = fname.replace('Flex:', 'Abd:')
            ftrace = v1_angle_dict.get(fname, [])
            atrace = v1_angle_dict.get(aname, [])
            for fi in range(min(n_valid, len(ftrace))):
                if ftrace[fi] is not None:
                    v1_ang[fi, ai, 0] = ftrace[fi]
                if fi < len(atrace) and atrace[fi] is not None:
                    v1_ang[fi, ai, 1] = atrace[fi]
        v1_angles_ref = v1_ang
        n_v1_valid = np.count_nonzero(~np.isnan(v1_ang[:, :, 0]))
        logger.info(f"  Loaded v1 angle reference: {n_v1_valid}/{n_valid * n_ang} valid angle-frames")

    # ── FK parameter initialisation ─────────────────────────────
    R_wrist_init  = _compute_wrist_frame(init_3d)            # (n_valid, 3, 3)
    flex_init, abd_init = _inverse_fk_init(init_3d, R_wrist_init)

    # 6D representation: first two columns of R_wrist
    r6d_init = np.concatenate(
        [R_wrist_init[:, :, 0], R_wrist_init[:, :, 1]], axis=-1
    ).astype(np.float32)                                     # (n_valid, 6)

    report(18)
    if cancelled():
        return {"cancelled": True}

    # ── Prepare tensors ─────────────────────────────────────────
    K1       = torch.tensor(calib["K1"],           dtype=torch.float32)
    K2       = torch.tensor(calib["K2"],           dtype=torch.float32)
    d1       = torch.tensor(calib["dist1"].ravel(),dtype=torch.float32)
    d2       = torch.tensor(calib["dist2"].ravel(),dtype=torch.float32)
    R_stereo = torch.tensor(calib["R"],            dtype=torch.float32)
    T_stereo = torch.tensor(calib["T"].ravel(),    dtype=torch.float32)

    def _t(arr, req_grad=False):
        return torch.tensor(arr, dtype=torch.float32, device=device,
                            requires_grad=req_grad)

    # Learnable FK parameters
    wrist_pos    = _t(init_3d[:, 0],     req_grad=True)
    wrist_r6d    = _t(r6d_init,          req_grad=True)
    bone_lengths = _t(target_bone_lengths, req_grad=True)
    flex_angs    = _t(flex_init,         req_grad=True)
    abd_angs     = _t(abd_init,          req_grad=True)

    # Metacarpal angles (bones 1-4: wrist→I_MCP, wrist→M_MCP, wrist→R_MCP, wrist→P_MCP)
    # These are CONSTANT across all frames — the palm structure doesn't change.
    # Initialise from the median of the per-frame init values.
    META_BONES = [1, 2, 3, 4]  # bone indices for the 4 metacarpals
    meta_flex = _t(np.nanmedian(flex_init[:, META_BONES], axis=0), req_grad=True)  # (4,)
    meta_abd  = _t(np.nanmedian(abd_init[:, META_BONES],  axis=0), req_grad=True)  # (4,)

    # 2D targets — valid frames only
    tgt_mp_L = _t(mp_L[valid_idx])
    tgt_mp_R = _t(mp_R[valid_idx])

    # Per-joint per-frame confidence weights (N, 21) → used in reprojection losses
    jw = _t(joint_weights)  # (n_valid, 21)

    # Per-frame adaptive smoothing weights: reduce smoothing during rapid 2D
    # motion (real finger taps) so the fit can follow fast movements.
    # Scale = 1.0 for still frames, drops toward 0.1 for rapid motion.
    smooth_scale = np.ones(n_valid, dtype=np.float32)
    mp_valid_L = mp_L[valid_idx]
    if n_valid > 2:
        d2d = np.sqrt(np.sum(np.diff(mp_valid_L, axis=0) ** 2, axis=2))  # (N-1, 21)
        d2d_max = np.nanmax(d2d, axis=1)                                  # (N-1,) max across joints
        med_d2d = np.nanmedian(d2d_max)
        # Frames with >3× median 2D motion get reduced smoothing
        MOTION_THRESH = max(med_d2d * 3.0, 5.0)
        for i in range(len(d2d_max)):
            if d2d_max[i] > MOTION_THRESH:
                scale = max(MOTION_THRESH / d2d_max[i], 0.1)
                smooth_scale[i]   = min(smooth_scale[i],   scale)
                smooth_scale[i+1] = min(smooth_scale[i+1], scale)
        n_reduced = (smooth_scale < 0.9).sum()
        logger.info(f"  Adaptive smoothing: {n_reduced}/{n_valid} frames with reduced smoothing")
    smooth_w = _t(smooth_scale)  # (n_valid,)

    # v1 angle reference tensor (n_valid, 15, 2) — NaN where v1 had no data
    v1_ref_t = None
    v1_ref_mask = None
    v1_wrist_t = None
    if v1_angles_ref is not None:
        v1_ref_t = _t(np.nan_to_num(v1_angles_ref, nan=0.0))     # (n_valid, 15, 2)
        v1_ref_mask = _t((~np.isnan(v1_angles_ref)).astype(np.float32))  # 1 where valid
        v1_wrist_t = _t(v1_wrist_ref)                              # (n_valid, 3)

    def _valid_tgt(arr):
        if arr is None:
            return None
        sub = arr[valid_idx]
        t   = _t(sub)
        mask = ~torch.isnan(t[:, :, 0])                     # (n_valid, 21)
        t    = torch.nan_to_num(t, nan=0.0)
        return t, mask

    vis_tgt_L, vis_mask_L = _valid_tgt(vis_L) if has_vision else (None, None)
    vis_tgt_R, vis_mask_R = _valid_tgt(vis_R) if has_vision else (None, None)

    # DLC — joints 4 (thumb tip) and 8 (index tip) only
    if has_dlc:
        dlc_tgt_L = _t(np.stack([th_L[valid_idx], ix_L[valid_idx]], axis=1))  # (n_valid,2,2)
        dlc_tgt_R = _t(np.stack([th_R[valid_idx], ix_R[valid_idx]], axis=1))
        dlc_mask_L = ~torch.isnan(dlc_tgt_L[:, :, 0])
        dlc_mask_R = ~torch.isnan(dlc_tgt_R[:, :, 0])
        dlc_tgt_L  = torch.nan_to_num(dlc_tgt_L, nan=0.0)
        dlc_tgt_R  = torch.nan_to_num(dlc_tgt_R, nan=0.0)
    else:
        dlc_tgt_L = dlc_tgt_R = dlc_mask_L = dlc_mask_R = None

    if has_hrnet:
        hrnet_tgt_L, hrnet_mask_L = _valid_tgt(hrnet_L)
        hrnet_tgt_R, hrnet_mask_R = _valid_tgt(hrnet_R)
    else:
        hrnet_tgt_L = hrnet_tgt_R = hrnet_mask_L = hrnet_mask_R = None

    # Prepare heatmap tensors for heatmap-value loss
    hm_t_L = hm_t_R = None
    hm_bbox_L_t = hm_bbox_R_t = None
    if has_heatmaps:
        import torch.nn.functional as F_torch
        # Slice to valid frames and convert to tensors
        hm_t_L = _t(hm_maps_L[valid_idx])  # (n_valid, 21, H, W)
        hm_bbox_L_t = hm_bbox_L
        if hm_maps_R is not None:
            hm_t_R = _t(hm_maps_R[valid_idx])
            hm_bbox_R_t = hm_bbox_R
        logger.info(f"  Heatmaps loaded: L={hm_t_L.shape} bbox={hm_bbox_L}")

        # Load pre-computed peak assignments from JSON (generated by HRNet job)
        _mano_trial_dir = _mano_dir(subject_name) / trial_stem
        peak_json_path = _mano_trial_dir / "hrnet_peak_assignments.json"
        _hm_targets_L = None
        _hm_targets_R = None
        if peak_json_path.exists():
            peak_data = json.loads(peak_json_path.read_text())
            from .mano_data import JOINT_NAMES as _JN
            def _load_peaks(peaks_dict, n_frames):
                arr = np.full((n_frames, 21, 2), np.nan, dtype=np.float32)
                for j in range(21):
                    frames = peaks_dict.get(_JN[j], [])
                    for f in range(min(len(frames), n_frames)):
                        if frames[f] is not None:
                            arr[f, j] = frames[f]
                return arr
            peaks_all_L = _load_peaks(peak_data.get("peaks_L", {}), N)
            _hm_targets_L = _t(peaks_all_L[valid_idx])
            if "peaks_R" in peak_data:
                peaks_all_R = _load_peaks(peak_data.get("peaks_R", {}), N)
                _hm_targets_R = _t(peaks_all_R[valid_idx])
            if hrnet_fingertips_only:
                # Zero out (NaN) all non-fingertip joints so only tips contribute
                _TIPS = {4, 8, 12, 16, 20}
                _non_tip = [j for j in range(21) if j not in _TIPS]
                _hm_targets_L[:, _non_tip] = float('nan')
                if _hm_targets_R is not None:
                    _hm_targets_R[:, _non_tip] = float('nan')
            n_valid_L = (~torch.isnan(_hm_targets_L[:, 1:, 0])).sum().item()
            n_total = n_valid * 20
            logger.info(f"  Loaded peak assignments: L={n_valid_L}/{n_total} ({100*n_valid_L/max(n_total,1):.0f}%)" +
                        (" (fingertips only)" if hrnet_fingertips_only else ""))
        else:
            logger.warning(f"  No peak assignments found at {peak_json_path} — heatmap loss disabled")
            has_heatmaps = False

    # ── Joint angle priors (custom overrides or bundled defaults) ──
    angle_prior_data = None
    _angle_priors_list = []
    if use_angle_constraints:
        from .mano_data import load_angle_priors
        angle_prior_data = load_angle_priors()
        _angle_priors_list = angle_prior_data.get("joints", [])
        logger.info(f"  Loaded {len(_angle_priors_list)} joint angle priors (flex+abd per joint)")

    # ── Optimiser ───────────────────────────────────────────────
    n_iters   = 400
    optimizer = torch.optim.Adam([
        {"params": [wrist_pos],    "lr": 0.3},
        {"params": [wrist_r6d],    "lr": 0.01},
        {"params": [bone_lengths], "lr": 0.05},
        {"params": [flex_angs],    "lr": 0.01},
        {"params": [abd_angs],     "lr": 0.005},
        {"params": [meta_flex],    "lr": 0.005},   # constant metacarpal angles
        {"params": [meta_abd],     "lr": 0.005},
    ])

    from .mano_fitting import _project_torch  # reuse the same differentiable projector

    target_bl = _t(target_bone_lengths)
    mcp_p2p_idx    = torch.tensor(MCP_P2P_PAIRS, dtype=torch.long)  # (3, 2)
    mcp_p2p_tgt_t  = _t(mcp_p2p_targets)                           # (3,) [IM, MR, RP]
    ti_target_t    = torch.tensor(ti_target, device=device)
    mcp_chain_idx  = torch.tensor(MCP_CHAIN_ANGLES, dtype=torch.long)  # (2, 3)
    mcp_chain_tgt  = _t(np.radians(mcp_chain_targets))               # (2,) in radians

    logger.info(f"  Running v2 fitting ({n_iters} iters, {n_valid} frames)...")
    report(20)

    for it in range(n_iters):
        if cancelled():
            return {"cancelled": True}

        optimizer.zero_grad()

        # Clamp bone lengths to positive
        bl_clamped = torch.clamp(bone_lengths, min=1.0)

        # Broadcast constant metacarpal angles into per-frame tensors.
        # meta_flex/meta_abd are (4,) constants; expand to (N, 4) and overwrite
        # bones 1-4 so the palm structure is identical every frame.
        flex_combined = flex_angs.clone()
        abd_combined  = abd_angs.clone()
        flex_combined[:, META_BONES] = meta_flex.unsqueeze(0)
        abd_combined[:, META_BONES]  = meta_abd.unsqueeze(0)

        # Forward kinematics → joints_3d (n_valid, 21, 3)
        joints_3d = _forward_kinematics(wrist_pos, wrist_r6d,
                                        bl_clamped, flex_combined, abd_combined)

        # Project to both cameras
        pL = _project_torch(joints_3d, K1, d1)
        pR = _project_torch(joints_3d, K2, d2, R_stereo, T_stereo)

        loss = torch.tensor(0.0)

        # ── Source losses (per-joint weighted) ──────────────────
        # jw (n_valid, 21) downweights individual joints on frames where
        # their Z-residual is anomalous — good joints on the same frame
        # retain full weight.
        jw_sum = jw.sum().clamp(min=1)

        def _weighted_reproj(pred, tgt, mask=None):
            """Per-joint squared 2D error, weighted by jw and optional mask."""
            err = ((pred - tgt) ** 2).sum(-1)                    # (N, 21)
            w = jw if mask is None else jw * mask.float()
            return (err * w).sum() / w.sum().clamp(min=1)

        if w_mediapipe > 0:
            loss = loss + w_mediapipe * (
                cam_w_L * _weighted_reproj(pL, tgt_mp_L)
                + cam_w_R * _weighted_reproj(pR, tgt_mp_R)
            )

        if w_vision > 0 and has_vision:
            loss = loss + w_vision * (
                cam_w_L * _weighted_reproj(pL, vis_tgt_L, vis_mask_L)
                + cam_w_R * _weighted_reproj(pR, vis_tgt_R, vis_mask_R)
            )

        # HRNet/DLC target points differ from MediaPipe landmarks.
        # Each joint's HRNet target is shifted distally along the parent→joint bone
        # by a fraction of that bone's length (empirically measured).
        # {joint: (parent, extension_ratio)}
        HM_OFFSETS = {
            # Tips: HRNet targets actual nail/tip
            4: (3, 0.43),  8: (7, 0.32),  12: (11, 0.40), 16: (15, 0.40), 20: (19, 0.55),
            # DIPs: ~15-28% distal shift
            3: (2, 0.28),  7: (6, 0.19),  11: (10, 0.12), 15: (14, 0.17), 19: (18, 0.02),
            # PIPs: ~5-13% distal shift
            2: (1, 0.12),  6: (5, 0.10),  10: (9, 0.13),  14: (13, 0.04), 18: (17, 0.05),
        }
        joints_ext = joints_3d.clone()
        for j, (p, ext) in HM_OFFSETS.items():
            if ext > 0.01:
                bone_dir = joints_3d[:, j] - joints_3d[:, p]
                joints_ext[:, j] = joints_3d[:, j] + ext * bone_dir
        pL_ext = _project_torch(joints_ext, K1, d1)
        pR_ext = _project_torch(joints_ext, K2, d2, R_stereo, T_stereo)

        if w_dlc > 0 and has_dlc:
            pL_tips = torch.stack([pL_ext[:, 4], pL_ext[:, 8]], dim=1)
            pR_tips = torch.stack([pR_ext[:, 4], pR_ext[:, 8]], dim=1)
            jw_dlc = torch.stack([jw[:, 4], jw[:, 8]], dim=1)   # (N, 2)
            err_L = ((pL_tips - dlc_tgt_L) ** 2).sum(-1) * dlc_mask_L.float() * jw_dlc
            err_R = ((pR_tips - dlc_tgt_R) ** 2).sum(-1) * dlc_mask_R.float() * jw_dlc
            denom = (dlc_mask_L.float() * jw_dlc + dlc_mask_R.float() * jw_dlc).sum().clamp(min=1)
            loss = loss + w_dlc * (cam_w_L * err_L.sum() + cam_w_R * err_R.sum()) / denom

        if w_hrnet > 0 and has_heatmaps:


            def _heatmap_sample_loss_masked(proj_2d, hm_tensor, bbox, targets):
                """Smooth heatmap-value loss, masked by peak assignment validity.

                Only include joints where the assigned peak is within 30px of
                the projected position — if the peak is far away, the heatmap
                is confused about this joint and the gradient is untrustworthy.
                """
                x1, y1, x2, y2 = bbox
                N_, J, H, W = hm_tensor.shape
                bw = max(x2 - x1, 1)
                bh = max(y2 - y1, 1)

                # Sample heatmap values at projected positions
                gx = 2.0 * (proj_2d[:, :, 0] - x1) / bw - 1.0
                gy = 2.0 * (proj_2d[:, :, 1] - y1) / bh - 1.0
                grid = torch.stack([gx, gy], dim=-1).unsqueeze(2)
                hm_flat = hm_tensor.reshape(N_ * J, 1, H, W)
                grid_flat = grid.reshape(N_ * J, 1, 1, 2)
                sampled = torch.nn.functional.grid_sample(
                    hm_flat, grid_flat, mode='bilinear', padding_mode='zeros', align_corners=False
                )
                vals = sampled.reshape(N_, J)  # (N, 21)

                # Validity mask: assigned peak must be within 30px of projected position
                with torch.no_grad():
                    peak_dist = ((proj_2d - targets) ** 2).sum(-1).sqrt()  # (N, 21)
                    valid = (~torch.isnan(peak_dist)) & (peak_dist < 30.0)
                    # Exclude wrist (joint 0)
                    valid[:, 0] = False

                if valid.sum() == 0:
                    return torch.tensor(0.0, device=proj_2d.device)
                return ((1.0 - vals.clamp(0, 1)) * valid.float()).sum() / valid.sum()

            hm_loss = cam_w_L * _heatmap_sample_loss_masked(pL_ext, hm_t_L, hm_bbox_L_t, _hm_targets_L)
            if _hm_targets_R is not None and hm_t_R is not None:
                hm_loss = hm_loss + cam_w_R * _heatmap_sample_loss_masked(pR_ext, hm_t_R, hm_bbox_R_t, _hm_targets_R)
            loss = loss + w_hrnet * 1000 * hm_loss
            if it % 100 == 0:
                logger.info(f"    heatmap loss: {hm_loss.item():.4f} (w_hrnet={w_hrnet})")

        # ── Bone length constraint (soft pull toward targets) ───
        if w_bone > 0:
            loss = loss + w_bone * ((bl_clamped - target_bl) ** 2).mean()

        # Note: inter-MCP distances and knuckle angles are inherently constant
        # because metacarpal angles (bones 1-4) are parameterised as constants.
        # No soft constraint needed — the structure is rigid by construction.

        # ── Temporal smoothness — adaptive per-frame weights ─────
        # smooth_w (N,) is ~1.0 for still frames, drops to ~0.1 during
        # rapid finger taps so the fit can track fast real motion.
        # sw_vel (N-1,) weights velocity terms; sw_acc (N-2,) weights accel.
        sw_vel = torch.minimum(smooth_w[:-1], smooth_w[1:])        # (N-1,)
        sw_acc = torch.minimum(sw_vel[:-1], sw_vel[1:]) if n_valid > 2 else None
        sw_jrk = torch.minimum(sw_acc[:-1], sw_acc[1:]) if n_valid > 3 and sw_acc is not None else None

        # ── Temporal smoothness — wrist only for absolute position ──
        # Distal joint motion scales with bone length, so penalising it
        # biases bones shorter.  Finger motion is handled by w_smooth_angles
        # (length-invariant).  Wrist velocity smoothing stabilises global
        # depth and hand-level jitter.
        if w_smooth_xy > 0 and n_valid > 1:
            vel_w = joints_3d[1:, 0] - joints_3d[:-1, 0]
            vel_xy_w = vel_w[:, :2]
            loss_xy = 0.05 * (sw_vel * (vel_xy_w ** 2).sum(-1)).mean()
            if n_valid > 2:
                acc_xy_w = vel_xy_w[1:] - vel_xy_w[:-1]
                loss_xy = loss_xy + 0.02 * (sw_acc * (acc_xy_w ** 2).sum(-1)).mean()
            if n_valid > 3 and sw_jrk is not None:
                jrk_xy_w = acc_xy_w[1:] - acc_xy_w[:-1]
                loss_xy = loss_xy + 0.01 * (sw_jrk * (jrk_xy_w ** 2).sum(-1)).mean()
            loss = loss + w_smooth_xy * w_smooth_wrist * loss_xy

        if w_smooth_z > 0 and n_valid > 1:
            vel_z_w = joints_3d[1:, 0, 2] - joints_3d[:-1, 0, 2]
            loss_z = 0.05 * (sw_vel * vel_z_w ** 2).mean()
            if n_valid > 2:
                acc_z_w = vel_z_w[1:] - vel_z_w[:-1]
                loss_z = loss_z + 0.02 * (sw_acc * acc_z_w ** 2).mean()
            if n_valid > 3 and sw_jrk is not None:
                jrk_z_w = acc_z_w[1:] - acc_z_w[:-1]
                loss_z = loss_z + 0.01 * (sw_jrk * jrk_z_w ** 2).mean()
            loss = loss + w_smooth_z * w_smooth_wrist * loss_z

        # ── Temporal smoothness — joint angles ──────────────────
        if w_smooth_angles > 0 and n_valid > 1:
            vf = flex_angs[1:] - flex_angs[:-1]
            va = abd_angs[1:] - abd_angs[:-1]
            t_ang = 0.1 * (sw_vel.unsqueeze(1) * (vf ** 2 + va ** 2)).mean()
            if n_valid > 2:
                af = vf[1:] - vf[:-1]
                aa = va[1:] - va[:-1]
                t_ang = t_ang + 0.05 * (sw_acc.unsqueeze(1) * (af ** 2 + aa ** 2)).mean()
            if n_valid > 3 and sw_jrk is not None:
                jf = af[1:] - af[:-1]
                ja = aa[1:] - aa[:-1]
                t_ang = t_ang + 0.02 * (sw_jrk.unsqueeze(1) * (jf ** 2 + ja ** 2)).mean()
            loss = loss + w_smooth_angles * t_ang

        # ── Joint angle constraints (palm-normal flex/abd decomposition) ──
        if use_angle_constraints and _angle_priors_list and w_constraints > 0:
            from .angle_constraint_loss import compute_angle_constraint_loss
            _constraint_groups = angle_prior_data.get("constraint_groups") if angle_prior_data else None
            loss = loss + w_constraints * compute_angle_constraint_loss(joints_3d, _angle_priors_list, _constraint_groups)

        # ── Within-finger flex coupling constraint ────────────────
        # Finger bone indices in BONES/flex_angs order:
        #   Thumb:  5(1→2), 6(2→3), 7(3→4)
        #   Index:  8(5→6), 9(6→7), 10(7→8)
        #   Middle: 11(9→10), 12(10→11), 13(11→12)
        #   Ring:   14(13→14), 15(14→15), 16(15→16)
        #   Pinky:  17(17→18), 18(18→19), 19(19→20)
        # Only couple DIP-PIP pairs (not MCP). Each entry is [proximal, distal].
        FINGER_FLEX_PAIRS = [
            [6, 7],       # thumb: MCP→IP
            [9, 10],      # index: PIP→DIP
            [12, 13],     # middle: PIP→DIP
            [15, 16],     # ring:   PIP→DIP
            [18, 19],     # pinky:  PIP→DIP
        ]
        _flex_coupling_w = 1.0
        if _angle_priors_list:
            from .mano_data import load_angle_priors as _lap
            _flex_coupling_w = _lap().get("flex_coupling", 1.0)

        if w_constraints > 0 and _flex_coupling_w > 0:
            coupling_loss = torch.tensor(0.0, device=device)

            # 1. Velocity co-variance: paired joints should flex in the
            #    same direction, not opposite.
            if n_valid > 1:
                for prox_i, dist_i in FINGER_FLEX_PAIRS:
                    dfa = flex_angs[1:, prox_i] - flex_angs[:-1, prox_i]
                    dfb = flex_angs[1:, dist_i] - flex_angs[:-1, dist_i]
                    coupling_loss = coupling_loss + torch.clamp(-dfa * dfb, min=0).mean()

            # 2. Static ordering: a DIP cannot be significantly more
            #    flexed than its PIP.  flex_angs are in radians
            #    (negative = flexion).  Penalise when distal is more negative
            #    (more flexed) than proximal by more than ~10° (0.17 rad).
            FLEX_LEAD_MARGIN = 0.17  # ~10° in radians
            for prox_i, dist_i in FINGER_FLEX_PAIRS:
                prox = flex_angs[:, prox_i]    # (N,) more proximal
                dist = flex_angs[:, dist_i]    # (N,) more distal
                excess = torch.clamp(prox - dist - FLEX_LEAD_MARGIN, min=0)
                coupling_loss = coupling_loss + (excess ** 2).mean()

            loss = loss + w_constraints * _flex_coupling_w * coupling_loss


        # ── v1 skeleton reference penalty ──────────────────────
        # Penalise v2 angles that deviate from the v1 fit's angles.
        # Uses the same palm-normal decomposition for both.
        if v1_ref_t is not None and w_v1_ref > 0:
            from .angle_constraint_loss import _tnorm, _talign
            from .mano_data import FLEX_ANGLE_OPTIONS
            _w = joints_3d[:, 0]
            _ei = joints_3d[:, 5] - _w; _em = joints_3d[:, 9] - _w
            _er = joints_3d[:, 13] - _w; _ep = joints_3d[:, 17] - _w
            _nip = _tnorm(torch.cross(_ei, _ep, dim=1))
            _nim = _talign(_tnorm(torch.cross(_ei, _em, dim=1)), _nip)
            _nmr = _talign(_tnorm(torch.cross(_em, _er, dim=1)), _nip)
            _nrp = _talign(_tnorm(torch.cross(_er, _ep, dim=1)), _nip)
            _pn = _tnorm(_nip + _nim + _nmr + _nrp)
            _pd = _tnorm(_ep); _tref = _pd  # thumb_ref = pinky_dir (matches current setting)
            _THUMB = {1, 2, 3}
            _da_cache = {}
            PI = 3.141592653589793
            v2_angs = []  # will collect (n_valid,) flex and abd tensors
            for ai, (fname, p, j, c) in enumerate(FLEX_ANGLE_OPTIONS):
                bi = _tnorm(joints_3d[:, j] - joints_3d[:, p])
                bo = _tnorm(joints_3d[:, c] - joints_3d[:, j])
                rr = _tref if j in _THUMB else _pn
                if p != 0 and p in _da_cache:
                    dap, bip = _da_cache[p]
                    st = (-(bi * dap).sum(-1, keepdim=True)).clamp(-1, 1)
                    ct = (1 - st**2).clamp(min=0).sqrt()
                    rr = st * bip + ct * dap
                da = _tnorm(rr - (rr * bi).sum(-1, keepdim=True) * bi)
                fa = _tnorm(torch.cross(da, bi, dim=1))
                _da_cache[j] = (da, bi)
                flex_d = torch.atan2(-(bo * da).sum(-1), (bo * bi).sum(-1)) * (180.0 / PI)
                fr = flex_d * (PI / 180.0)
                bfe = _tnorm(fr.unsqueeze(-1).cos() * bi - fr.unsqueeze(-1).sin() * da)
                abd_d = torch.atan2((bo * fa).sum(-1), (bo * bfe).sum(-1)) * (180.0 / PI)
                v2_angs.append(torch.stack([flex_d, abd_d], dim=-1))  # (n_valid, 2)
            v2_angs_t = torch.stack(v2_angs, dim=1)  # (n_valid, 15, 2)
            v1_diff = (v2_angs_t - v1_ref_t) ** 2 * v1_ref_mask
            loss = loss + w_v1_ref * v1_diff.sum() / v1_ref_mask.sum().clamp(min=1)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        # Sanitise NaN/Inf gradients (atan2 in angle constraints can produce
        # undefined gradients at degenerate configurations)
        all_params = [wrist_pos, wrist_r6d, bone_lengths, flex_angs, abd_angs, meta_flex, meta_abd]
        for p in all_params:
            if p.grad is not None:
                torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
        # Zero out per-frame grads for metacarpal bones (they're overwritten by constants)
        if flex_angs.grad is not None:
            flex_angs.grad[:, META_BONES] = 0
        if abd_angs.grad is not None:
            abd_angs.grad[:, META_BONES] = 0
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=5.0)
        optimizer.step()

        # Hard projection: force mean MCP abduction to zero per frame.
        # Subtract the per-frame mean from each finger MCP abd angle.
        # This forces the wrist rotation to absorb any global abduction bias.
        MCP_ABD_BONES = [8, 11, 14, 17]  # I/M/R/P MCP→PIP
        with torch.no_grad():
            mcp_abd_mean = abd_angs[:, MCP_ABD_BONES].mean(dim=1, keepdim=True)  # (N, 1)
            abd_angs[:, MCP_ABD_BONES] -= mcp_abd_mean

        # Yield GIL
        if it % 5 == 0:
            import time; time.sleep(0)

        if it % 10 == 0:
            report(20 + (it / n_iters) * 70)

        if it % 100 == 0 or it == n_iters - 1:
            with torch.no_grad():
                eL = torch.sqrt(((pL - tgt_mp_L) ** 2).sum(-1)).mean().item()
                eR = torch.sqrt(((pR - tgt_mp_R) ** 2).sum(-1)).mean().item()
                be = (bl_clamped - target_bl).abs().mean().item()
            logger.info(f"    iter {it}: L={eL:.1f}px R={eR:.1f}px bone_err={be:.1f}mm")

    report(90)

    # ── Final projection errors ─────────────────────────────────
    with torch.no_grad():
        bl_final = torch.clamp(bone_lengths, min=1.0)
        # Apply constant metacarpal angles for final output
        flex_final = flex_angs.clone()
        abd_final  = abd_angs.clone()
        flex_final[:, META_BONES] = meta_flex.unsqueeze(0)
        abd_final[:, META_BONES]  = meta_abd.unsqueeze(0)
        joints_final = _forward_kinematics(wrist_pos, wrist_r6d,
                                           bl_final, flex_final, abd_final)
        pL_f = _project_torch(joints_final, K1, d1)
        pR_f = _project_torch(joints_final, K2, d2, R_stereo, T_stereo)
        err_L = torch.sqrt(((pL_f - tgt_mp_L) ** 2).sum(-1)).mean(1).cpu().numpy()
        err_R = torch.sqrt(((pR_f - tgt_mp_R) ** 2).sum(-1)).mean(1).cpu().numpy()

    # Compute per-camera 2D offset correction (compensates bad calibration)
    # Median offset between skeleton projection and MP 2D targets
    with torch.no_grad():
        off_L = (tgt_mp_L - pL_f).detach().cpu().numpy()  # (n_valid, 21, 2)
        off_R = (tgt_mp_R - pR_f).detach().cpu().numpy()
    corr_L = np.nanmedian(off_L.reshape(-1, 2), axis=0).astype(np.float32)
    corr_R = np.nanmedian(off_R.reshape(-1, 2), axis=0).astype(np.float32)
    # Only apply correction if it's significant (>5px)
    if np.linalg.norm(corr_L) < 5: corr_L = np.zeros(2, dtype=np.float32)
    if np.linalg.norm(corr_R) < 5: corr_R = np.zeros(2, dtype=np.float32)
    logger.info(f"  2D corrections: L=[{corr_L[0]:.1f},{corr_L[1]:.1f}]  R=[{corr_R[0]:.1f},{corr_R[1]:.1f}]")

    logger.info(f"  Final: L={err_L.mean():.1f}±{err_L.std():.1f}px  "
                f"R={err_R.mean():.1f}±{err_R.std():.1f}px")

    # ── Build full (N,21,3) output ──────────────────────────────
    all_joints = np.full((N, 21, 3), np.nan, dtype=np.float32)
    all_err_L  = np.full(N, np.nan,  dtype=np.float32)
    all_err_R  = np.full(N, np.nan,  dtype=np.float32)

    j3d_np = joints_final.detach().cpu().numpy()
    for i, t in enumerate(valid_idx):
        all_joints[t] = j3d_np[i]
        all_err_L[t]  = err_L[i]
        all_err_R[t]  = err_R[i]

    final_bone_lengths = bl_final.detach().cpu().numpy()

    # ── Save (rotate old fits to keep up to 3 previous runs) ───
    mano_trial_dir = _mano_dir(subject_name) / trial_stem
    mano_trial_dir.mkdir(parents=True, exist_ok=True)
    out_path = mano_trial_dir / "mano_fit_v2.npz"
    params_path = mano_trial_dir / "mano_fit_v2_params.json"

    # Rotate: prev3 → deleted, prev2 → prev3, prev1 → prev2, current → prev1
    for i in range(3, 0, -1):
        src_npz = mano_trial_dir / f"mano_fit_v2_prev{i}.npz"
        src_json = mano_trial_dir / f"mano_fit_v2_prev{i}_params.json"
        if i == 3:
            # Delete oldest
            if src_npz.exists(): src_npz.unlink()
            if src_json.exists(): src_json.unlink()
        else:
            dst_npz = mano_trial_dir / f"mano_fit_v2_prev{i+1}.npz"
            dst_json = mano_trial_dir / f"mano_fit_v2_prev{i+1}_params.json"
            if src_npz.exists(): src_npz.rename(dst_npz)
            if src_json.exists(): src_json.rename(dst_json)
    # Current → prev1
    if out_path.exists():
        (mano_trial_dir / "mano_fit_v2_prev1.npz").unlink(missing_ok=True)
        out_path.rename(mano_trial_dir / "mano_fit_v2_prev1.npz")
    if params_path.exists():
        (mano_trial_dir / "mano_fit_v2_prev1_params.json").unlink(missing_ok=True)
        params_path.rename(mano_trial_dir / "mano_fit_v2_prev1_params.json")

    np.savez(
        str(out_path),
        joints_3d=all_joints,
        fit_error_L=all_err_L,
        fit_error_R=all_err_R,
        offset_L=corr_L,
        offset_R=corr_R,
        target_bone_lengths=final_bone_lengths,
        n_frames=N,
        stage=2,
        fit_type="skeleton_v2",
    )

    # Save fitting parameters alongside the npz
    from datetime import datetime
    params_path = mano_trial_dir / "mano_fit_v2_params.json"
    params_path.write_text(json.dumps({
        "fit_type": "skeleton_v2",
        "version": "v2",
        "subject": subject_name,
        "trial": trial_stem,
        "n_frames": int(N),
        "n_fitted": int(n_valid),
        "n_iters": n_iters,
        "sources": {
            "mediapipe": True,
            "vision": has_vision,
            "dlc": has_dlc,
            "hrnet": has_hrnet,
        },
        "params": {
            "w_mediapipe": w_mediapipe,
            "w_vision": w_vision,
            "w_dlc": w_dlc,
            "w_hrnet": w_hrnet,
            "hrnet_fingertips_only": hrnet_fingertips_only,
            "w_bone": w_bone,
            "w_smooth_wrist": w_smooth_wrist,
            "w_smooth_xy": w_smooth_xy,
            "w_smooth_z": w_smooth_z,
            "w_smooth_angles": w_smooth_angles,
            "use_angle_constraints": use_angle_constraints,
            "w_constraints": w_constraints,
            "w_v1_ref": w_v1_ref,
        },
        "results": {
            "mean_error_L": float(np.nanmean(all_err_L)),
            "mean_error_R": float(np.nanmean(all_err_R)),
            "bone_lengths": final_bone_lengths.tolist(),
            "mcp_span_targets": dict(zip(MCP_ALL_NAMES, mcp_all_targets)),
            "mcp_chain_angle_targets": dict(zip(MCP_CHAIN_NAMES, mcp_chain_targets.tolist())),
        },
        "angle_constraints": angle_prior_data,
        "timestamp": datetime.now().isoformat(),
    }, indent=2))

    from .mano_data import _load_mano_npz
    _load_mano_npz.cache_clear()

    report(100)
    logger.info(f"  Saved {out_path}")

    return {
        "output_path": str(out_path),
        "n_frames": N,
        "n_fitted": n_valid,
        "mean_error_L": float(np.nanmean(all_err_L)),
        "mean_error_R": float(np.nanmean(all_err_R)),
        "bone_lengths": final_bone_lengths.tolist(),
    }
