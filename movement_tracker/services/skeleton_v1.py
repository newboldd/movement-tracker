"""Skeleton fitting: MediaPipe 2D → stereo triangulation → constrained 3D skeleton.

Fits a kinematic hand skeleton via differentiable 2D stereo reprojection:
  1. MediaPipe 2D keypoints (both cameras)
  2. Per-bone length targets from triangulated 3D (robust median)
  3. Temporal smoothness (velocity + acceleration regularization)
  4. Soft joint angle limits (anatomical constraints)

No external model data or licenses required — just PyTorch (CPU).
Install: pip install torch --index-url https://download.pytorch.org/whl/cpu

The optimization directly refines 3D joint positions (initialized from
stereo triangulation) subject to bone length, reprojection, and smoothness
constraints.  This gives consistent bone lengths, smooth trajectories, and
optimal dual-camera 2D fit without the Skeleton license.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..config import get_settings
from .calibration import get_calibration_for_subject, triangulate_points
from .skeleton_data import _skeleton_dir, HAND_SKELETON
from .mediapipe_prelabel import load_mediapipe_prelabels
from .video import build_trial_map

logger = logging.getLogger(__name__)

BONES = HAND_SKELETON

# Parent joint for each joint (wrist has no parent)
PARENT = {
    0: None,
    1: 0, 2: 1, 3: 2, 4: 3,       # thumb
    5: 0, 6: 5, 7: 6, 8: 7,       # index
    9: 0, 10: 9, 11: 10, 12: 11,  # middle
    13: 0, 14: 13, 15: 14, 16: 15, # ring
    17: 0, 18: 17, 19: 18, 20: 19, # pinky
}


def check_fitting_available() -> dict:
    """Check if fitting dependencies are available.

    Only requires PyTorch (CPU-only is fine, ~200MB).
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        return {
            "available": False,
            "message": (
                "PyTorch not installed. Run:\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cpu"
            ),
        }
    return {"available": True, "message": "Ready"}


# Keep old name as alias for the router
check_skeleton_available = check_fitting_available


def _get_hand_side(trial_stem: str) -> str:
    """Determine hand side from trial name (L1/L2 = left, R1/R2 = right)."""
    parts = trial_stem.split("_")
    if len(parts) >= 2:
        return "left" if parts[1][0].upper() == "L" else "right"
    return "right"


def run_skeleton_v1_fit(
    subject_name: str,
    trial_stem: str,
    cancel_event: threading.Event | None = None,
    progress_callback: Callable[[float], None] | None = None,
    w_reproj: float = 1.0,
    w_bone: float = 5.0,
    w_smooth: float = 1.0,
    snap_bones: bool = False,
    w_angle: float = 2.0,
) -> dict[str, Any]:
    """Run Stage 1 skeleton fitting for a single trial.

    Directly optimizes 3D joint positions with:
    - 2D stereo reprojection loss (both cameras)
    - Bone length constraint (robust targets from triangulated 3D)
    - Temporal smoothness (velocity + acceleration)
    - Bone length consistency across frames

    Runs on CPU. ~300 iterations.

    Args:
        subject_name: Subject identifier
        trial_stem: Trial name (e.g., "MSA01_L1")
        cancel_event: Threading event to check for cancellation
        progress_callback: Called with progress 0-100

    Returns:
        Dict with fit results and output path.
    """
    import torch

    device = torch.device("cpu")

    def report(pct):
        if progress_callback:
            progress_callback(pct)

    def cancelled():
        return cancel_event is not None and cancel_event.is_set()

    report(0)
    logger.info(f"[Skeleton Fit Stage 1] {subject_name}/{trial_stem}")

    # ── Load data ──────────────────────────────────────────────
    trials = build_trial_map(subject_name)
    trial_info = None
    for t in trials:
        if t["trial_name"] == trial_stem:
            trial_info = t
            break
    if trial_info is None:
        raise ValueError(f"Trial {trial_stem} not found for {subject_name}")

    n_frames = trial_info["frame_count"]
    start_frame = trial_info.get("start_frame", 0)

    # MediaPipe data
    prelabels = load_mediapipe_prelabels(subject_name)
    if prelabels is None:
        raise ValueError(f"No MediaPipe prelabels for {subject_name}")

    os_lm = prelabels["OS_landmarks"]
    od_lm = prelabels["OD_landmarks"]
    end = min(start_frame + n_frames, os_lm.shape[0])
    mp_L = os_lm[start_frame:end].copy()
    mp_R = od_lm[start_frame:end].copy()
    N = mp_L.shape[0]

    # Calibration — use trial-level first (same as display code)
    from .skeleton_data import _load_trial_calibration
    calib = _load_trial_calibration(subject_name, trial_stem)
    if calib is None:
        calib = get_calibration_for_subject(subject_name)
    if calib is None:
        raise ValueError(f"No stereo calibration for {subject_name}")

    report(5)
    if cancelled():
        return {"cancelled": True}

    # ── Triangulate MP to 3D ───────────────────────────────────
    logger.info(f"  Triangulating {N} frames...")
    mp_3d = np.full((N, 21, 3), np.nan)
    for j in range(21):
        mp_3d[:, j, :] = triangulate_points(mp_L[:, j, :], mp_R[:, j, :], calib)

    # Valid frames: both cameras + successful triangulation
    valid_mask = (
        ~np.isnan(mp_L[:, 0, 0])
        & ~np.isnan(mp_R[:, 0, 0])
        & ~np.isnan(mp_3d[:, 0, 0])
    )
    valid_idx = np.where(valid_mask)[0]
    n_valid = len(valid_idx)

    if n_valid < 10:
        raise ValueError(f"Only {n_valid} valid frames (need at least 10)")

    logger.info(f"  {n_valid}/{N} valid frames with stereo 3D")
    report(10)

    # ── Compute bone length targets (robust median) ────────────
    bone_lengths_all = np.zeros((n_valid, len(BONES)), dtype=np.float32)
    for b, (j1, j2) in enumerate(BONES):
        diff = mp_3d[valid_idx, j2] - mp_3d[valid_idx, j1]
        bone_lengths_all[:, b] = np.linalg.norm(diff, axis=1)

    target_bone_lengths = np.zeros(len(BONES), dtype=np.float32)
    bone_weights = np.zeros(len(BONES), dtype=np.float32)
    for b in range(len(BONES)):
        bl = bone_lengths_all[:, b]
        bl = bl[~np.isnan(bl) & (bl > 0)]
        if len(bl) < 10:
            continue
        # Two-pass trimmed median
        med = np.median(bl)
        mad = np.median(np.abs(bl - med))
        sigma = max(mad * 1.4826, 0.1)
        inlier = np.abs(bl - med) < 3.0 * sigma
        bl_clean = bl[inlier] if inlier.sum() >= 5 else bl
        target_bone_lengths[b] = np.median(bl_clean)
        mad2 = np.median(np.abs(bl_clean - target_bone_lengths[b]))
        sigma2 = max(mad2 * 1.4826, 0.1)
        bone_weights[b] = 1.0 / max(sigma2 ** 2, 1.0)

    logger.info(
        f"  Bone lengths: min={target_bone_lengths[target_bone_lengths > 0].min():.1f}mm, "
        f"max={target_bone_lengths.max():.1f}mm"
    )
    report(15)

    if cancelled():
        return {"cancelled": True}

    # ── Prepare tensors ────────────────────────────────────────
    K1 = torch.tensor(calib["K1"], dtype=torch.float32, device=device)
    K2 = torch.tensor(calib["K2"], dtype=torch.float32, device=device)
    d1 = torch.tensor(calib["dist1"].ravel(), dtype=torch.float32, device=device)
    d2 = torch.tensor(calib["dist2"].ravel(), dtype=torch.float32, device=device)
    R_stereo = torch.tensor(calib["R"], dtype=torch.float32, device=device)
    T_stereo = torch.tensor(calib["T"].ravel(), dtype=torch.float32, device=device)

    tgt_L = torch.tensor(mp_L[valid_idx], device=device, dtype=torch.float32)
    tgt_R = torch.tensor(mp_R[valid_idx], device=device, dtype=torch.float32)
    target_bl = torch.tensor(target_bone_lengths, device=device, dtype=torch.float32)
    bl_w = torch.tensor(bone_weights, device=device, dtype=torch.float32)
    bone_idx = torch.tensor(BONES, device=device, dtype=torch.long)

    # Initialize 3D positions from triangulated data
    init_3d = mp_3d[valid_idx].copy()
    # Fill any per-joint NaNs with interpolation from neighbors
    for i in range(n_valid):
        for j in range(21):
            if np.isnan(init_3d[i, j, 0]):
                # Use wrist position as fallback
                if not np.isnan(init_3d[i, 0, 0]):
                    init_3d[i, j] = init_3d[i, 0]
                else:
                    init_3d[i, j] = [0, 0, 500]  # arbitrary fallback

    # Optimizable: 3D joint positions + per-camera 2D offsets
    joints_3d = torch.tensor(init_3d, device=device, dtype=torch.float32, requires_grad=True)
    offset_L = torch.zeros(1, 1, 2, device=device, dtype=torch.float32, requires_grad=True)
    offset_R = torch.zeros(1, 1, 2, device=device, dtype=torch.float32, requires_grad=True)

    logger.info(f"  Fitting {n_valid} frames ({n_valid * 21 * 3} position params)")
    report(20)

    # ── Load joint angle priors (custom overrides or bundled defaults) ──
    angle_prior_data = None
    _angle_priors_list = []
    if w_angle > 0:
        from .skeleton_data import load_angle_priors
        angle_prior_data = load_angle_priors()
        _angle_priors_list = angle_prior_data.get("joints", [])
        logger.info(f"  Loaded {len(_angle_priors_list)} joint angle priors (flex+abd per joint)")

    # ── Stage 1 optimization ───────────────────────────────────
    n_iters = 300

    optimizer = torch.optim.Adam([
        {"params": [joints_3d], "lr": 0.5},
        {"params": [offset_L, offset_R], "lr": 0.1},
    ])

    # Build kinematic chain traversal order (BFS from wrist) for bone projection
    _chain_order = []  # list of (parent_joint, child_joint, bone_index)
    _bone_map = {(j1, j2): b for b, (j1, j2) in enumerate(BONES)}
    _visited = {0}
    _queue = [0]
    while _queue:
        p = _queue.pop(0)
        for cj, pj in PARENT.items():
            if pj == p and cj not in _visited:
                _visited.add(cj)
                _queue.append(cj)
                bi = _bone_map.get((pj, cj), _bone_map.get((cj, pj)))
                if bi is not None and target_bone_lengths[bi] > 0:
                    _chain_order.append((pj, cj, bi))

    logger.info(f"  Running Stage 1 ({n_iters} iterations)...")

    for it in range(n_iters):
        if cancelled():
            return {"cancelled": True}

        optimizer.zero_grad()

        # Project to both cameras
        pL = _project_torch(joints_3d, K1, d1) + offset_L
        pR = _project_torch(joints_3d, K2, d2, R_stereo, T_stereo) + offset_R

        # Loss 1: 2D reprojection (both cameras)
        loss_reproj = (
            ((pL - tgt_L) ** 2).sum(-1).mean()
            + ((pR - tgt_R) ** 2).sum(-1).mean()
        )

        # Loss 2: Bone length consistency (match robust targets)
        j1_pts = joints_3d[:, bone_idx[:, 0]]
        j2_pts = joints_3d[:, bone_idx[:, 1]]
        bone_lens = (j2_pts - j1_pts).norm(dim=2)
        loss_bone = ((bone_lens - target_bl) ** 2 * bl_w).mean()

        # Loss 3: Temporal smoothness
        loss_temporal = torch.tensor(0.0, device=device)
        if n_valid > 1:
            vel = joints_3d[1:] - joints_3d[:-1]
            loss_temporal = 0.1 * (vel ** 2).mean()

            if n_valid > 2:
                acc = vel[1:] - vel[:-1]
                loss_temporal = loss_temporal + 0.05 * (acc ** 2).mean()

        # Loss 4: Joint angle constraints (palm-normal flex/abd decomposition)
        loss_angle = torch.tensor(0.0, device=device)
        if w_angle > 0 and _angle_priors_list:
            from .angle_constraint_loss import compute_angle_constraint_loss
            _cg = angle_prior_data.get("constraint_groups") if angle_prior_data else None
            loss_angle = compute_angle_constraint_loss(joints_3d, _angle_priors_list, _cg)

        loss = w_reproj * loss_reproj + w_bone * loss_bone + w_smooth * loss_temporal + w_angle * loss_angle

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        for p in [joints_3d, offset_L, offset_R]:
            if p.grad is not None:
                torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
        torch.nn.utils.clip_grad_norm_([joints_3d, offset_L, offset_R], max_norm=5.0)
        optimizer.step()

        # Project bone lengths to target values (hard constraint)
        if snap_bones:
            with torch.no_grad():
                for pj, cj, bi in _chain_order:
                    vec = joints_3d[:, cj] - joints_3d[:, pj]
                    length = vec.norm(dim=1, keepdim=True).clamp(min=1e-6)
                    joints_3d[:, cj] = joints_3d[:, pj] + vec * (target_bl[bi] / length)

        # Yield GIL periodically so the web server can handle requests
        if it % 5 == 0:
            import time; time.sleep(0)

        # Report progress (20% → 90%)
        if it % 10 == 0:
            report(20 + (it / n_iters) * 70)

        if it % 100 == 0 or it == n_iters - 1:
            with torch.no_grad():
                eL = torch.sqrt(((pL - tgt_L) ** 2).sum(-1)).mean().item()
                eR = torch.sqrt(((pR - tgt_R) ** 2).sum(-1)).mean().item()
                be = (bone_lens - target_bl).abs().mean().item()
            angle_val = loss_angle.item()
            logger.info(
                f"    iter {it}: L={eL:.1f}px R={eR:.1f}px bone_err={be:.1f}mm angle={angle_val:.3f}"
                f" [w_r={w_reproj} w_b={w_bone} w_s={w_smooth} w_a={w_angle} snap={snap_bones}]"
            )

    report(90)

    # ── Compute final errors ───────────────────────────────────
    with torch.no_grad():
        pL = _project_torch(joints_3d, K1, d1) + offset_L
        pR = _project_torch(joints_3d, K2, d2, R_stereo, T_stereo) + offset_R
        err_L = torch.sqrt(((pL - tgt_L) ** 2).sum(-1)).mean(1).cpu().numpy()
        err_R = torch.sqrt(((pR - tgt_R) ** 2).sum(-1)).mean(1).cpu().numpy()

    logger.info(
        f"  Final: L={err_L.mean():.1f}\u00b1{err_L.std():.1f}px "
        f"R={err_R.mean():.1f}\u00b1{err_R.std():.1f}px"
    )

    # ── Save results ───────────────────────────────────────────
    all_joints = np.full((N, 21, 3), np.nan)
    all_err_L = np.full(N, np.nan)
    all_err_R = np.full(N, np.nan)

    j3d_np = joints_3d.detach().cpu().numpy()
    for i, t in enumerate(valid_idx):
        all_joints[t] = j3d_np[i]
        all_err_L[t] = err_L[i]
        all_err_R[t] = err_R[i]

    skeleton_trial_dir = _skeleton_dir(subject_name) / trial_stem
    skeleton_trial_dir.mkdir(parents=True, exist_ok=True)

    out_path = skeleton_trial_dir / "skeleton_v1.npz"
    np.savez(
        str(out_path),
        joints_3d=all_joints,
        fit_error_L=all_err_L,
        fit_error_R=all_err_R,
        offset_L=offset_L.detach().cpu().numpy().ravel(),
        offset_R=offset_R.detach().cpu().numpy().ravel(),
        target_bone_lengths=target_bone_lengths,
        n_frames=N,
        stage=1,
        fit_type="skeleton",
    )

    # Save fitting parameters alongside the npz
    import json as _json
    from datetime import datetime
    params_path = skeleton_trial_dir / "skeleton_v1_params.json"
    params_path.write_text(_json.dumps({
        "fit_type": "skeleton",
        "version": "v1.0",
        "subject": subject_name,
        "trial": trial_stem,
        "n_frames": int(N),
        "n_fitted": int(n_valid),
        "n_iters": n_iters,
        "params": {
            "w_reproj": w_reproj,
            "w_bone": w_bone,
            "w_smooth": w_smooth,
            "snap_bones": snap_bones,
            "w_angle": w_angle,
        },
        "results": {
            "mean_error_L": float(np.nanmean(all_err_L)),
            "mean_error_R": float(np.nanmean(all_err_R)),
            "target_bone_lengths": target_bone_lengths.tolist(),
        },
        "angle_constraints": angle_prior_data,
        "timestamp": datetime.now().isoformat(),
    }, indent=2))

    # Clear cached data so next load picks up the new fit
    from .skeleton_data import _load_skeleton_npz
    _load_skeleton_npz.cache_clear()

    report(100)
    logger.info(f"  Saved {out_path}")

    return {
        "output_path": str(out_path),
        "n_frames": N,
        "n_fitted": n_valid,
        "mean_error_L": float(np.nanmean(all_err_L)),
        "mean_error_R": float(np.nanmean(all_err_R)),
    }


def _project_torch(pts3d, K, dist, R_ext=None, t_ext=None, min_z=10.0):
    """Differentiable projection with OpenCV radial+tangential distortion."""
    import torch

    if R_ext is not None:
        pts3d = pts3d @ R_ext.T + t_ext
    X, Y = pts3d[..., 0], pts3d[..., 1]
    Z = torch.clamp(pts3d[..., 2], min=min_z)
    x, y = X / Z, Y / Z
    r2 = x * x + y * y
    r4, r6 = r2 * r2, r2 * r2 * r2
    k1, k2, p1, p2, k3 = dist[0], dist[1], dist[2], dist[3], dist[4]
    radial = 1 + k1 * r2 + k2 * r4 + k3 * r6
    x_d = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    y_d = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return torch.stack([K[0, 0] * x_d + K[0, 2], K[1, 1] * y_d + K[1, 2]], dim=-1)
