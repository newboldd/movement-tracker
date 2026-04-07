"""MANO Stage 1 fitting: MediaPipe 2D → stereo triangulation → MANO model.

Fits MANO hand model via differentiable 2D stereo reprojection using:
  1. MediaPipe 2D keypoints (both cameras)
  2. Per-bone length targets from triangulated 3D (robust median)

Stage 1 focuses on positioning + bone proportions.
Runs on CPU (no GPU required).

Requires optional dependencies: torch, manotorch
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..config import get_settings
from .calibration import get_calibration_for_subject, load_calibration, triangulate_points
from .mano_data import _mano_dir, HAND_SKELETON
from .mediapipe_prelabel import load_mediapipe_prelabels
from .video import build_trial_map

logger = logging.getLogger(__name__)

# Hand skeleton bones (same as HAND_SKELETON)
BONES = HAND_SKELETON

NCOMPS = 45


def check_mano_available() -> dict:
    """Check if torch and manotorch are available.

    Returns dict with 'available' bool and 'message' string.
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        return {"available": False, "message": "PyTorch not installed. Run: pip install torch"}
    try:
        from manotorch.manolayer import ManoLayer  # noqa: F401
    except ImportError:
        return {"available": False, "message": "manotorch not installed. Run: pip install manotorch"}

    # Check for MANO model data
    settings = get_settings()
    mano_root = settings.dlc_path / "mano_v1_2"
    if not mano_root.is_dir():
        # Also check next to the app
        app_dir = Path(__file__).parent.parent.parent
        mano_root = app_dir / "data" / "mano_v1_2"

    if not mano_root.is_dir():
        return {
            "available": False,
            "message": f"MANO model data not found. Download from https://mano.is.tue.mpg.de/ and place in {settings.dlc_path / 'mano_v1_2'}",
        }

    return {"available": True, "message": "Ready", "mano_root": str(mano_root)}


def _find_mano_root() -> str:
    """Find the MANO model data directory."""
    result = check_mano_available()
    if not result["available"]:
        raise RuntimeError(result["message"])
    return result["mano_root"]


def _get_mano_side(trial_stem: str) -> str:
    """Determine MANO model side from trial name.

    Patient's left hand (_L) → mirrored → MANO 'right', and vice versa.
    """
    parts = trial_stem.split("_")
    if len(parts) >= 2:
        hand = parts[1][0].upper()
        return "right" if hand == "L" else "left"
    return "right"


def run_stage1_fitting(
    subject_name: str,
    trial_stem: str,
    cancel_event: threading.Event | None = None,
    progress_callback: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """Run MANO Stage 1 fitting for a single trial.

    Stage 1: MP reprojection + strong bone length constraint.
    Gets gross positioning right. Runs on CPU.

    Args:
        subject_name: Subject identifier
        trial_stem: Trial name (e.g., "MSA01_L1")
        cancel_event: Threading event to check for cancellation
        progress_callback: Called with progress 0-100

    Returns:
        Dict with fit results and output path.
    """
    import torch
    from manotorch.manolayer import ManoLayer

    device = torch.device("cpu")

    def report(pct):
        if progress_callback:
            progress_callback(pct)

    def cancelled():
        return cancel_event is not None and cancel_event.is_set()

    report(0)
    logger.info(f"[MANO Stage 1] {subject_name}/{trial_stem}")

    # ── Load data ──────────────────────────────────────────────
    # Trial map for frame range
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

    # Calibration
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

    # Valid frames: both cameras have landmarks + triangulation succeeded
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

    # ── Compute bone length targets ────────────────────────────
    bone_lengths_all = np.zeros((n_valid, len(BONES)), dtype=np.float32)
    for b, (j1, j2) in enumerate(BONES):
        diff = mp_3d[valid_idx, j2] - mp_3d[valid_idx, j1]
        bone_lengths_all[:, b] = np.linalg.norm(diff, axis=1)

    # Robust median with MAD outlier removal
    target_bone_lengths = np.zeros(len(BONES), dtype=np.float32)
    bone_weights = np.zeros(len(BONES), dtype=np.float32)
    for b in range(len(BONES)):
        bl = bone_lengths_all[:, b]
        bl = bl[~np.isnan(bl) & (bl > 0)]
        if len(bl) < 10:
            continue
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

    # ── Initialize MANO model ──────────────────────────────────
    side = _get_mano_side(trial_stem)
    mano_root = _find_mano_root()
    logger.info(f"  Initializing MANO model (side={side})...")

    mano_model = ManoLayer(
        mano_assets_root=mano_root,
        side=side,
        use_pca=True,
        ncomps=NCOMPS,
        flat_hand_mean=False,
    )

    with torch.no_grad():
        zero_pose = torch.zeros(1, NCOMPS + 3)
        rest_output = mano_model(zero_pose)
        rest_joints = rest_output.joints[0].numpy() * 1000.0  # m → mm

    mano_model = mano_model.to(device)
    report(20)

    # ── Prepare camera tensors ─────────────────────────────────
    K1 = torch.tensor(calib["K1"], dtype=torch.float32, device=device)
    K2 = torch.tensor(calib["K2"], dtype=torch.float32, device=device)
    d1 = torch.tensor(calib["dist1"].ravel(), dtype=torch.float32, device=device)
    d2 = torch.tensor(calib["dist2"].ravel(), dtype=torch.float32, device=device)
    R_stereo = torch.tensor(calib["R"], dtype=torch.float32, device=device)
    T_stereo = torch.tensor(calib["T"].ravel(), dtype=torch.float32, device=device)

    # ── Prepare targets ────────────────────────────────────────
    tgt_L = torch.tensor(mp_L[valid_idx], device=device, dtype=torch.float32)
    tgt_R = torch.tensor(mp_R[valid_idx], device=device, dtype=torch.float32)
    target_bl = torch.tensor(target_bone_lengths, device=device, dtype=torch.float32)
    bl_w = torch.tensor(bone_weights, device=device, dtype=torch.float32)
    bone_idx = torch.tensor(BONES, device=device, dtype=torch.long)

    # ── Initialize parameters ──────────────────────────────────
    mano_wrist_to_middle = np.linalg.norm(rest_joints[12] - rest_joints[0]) / 1000.0

    init_trans = np.zeros((n_valid, 3), dtype=np.float32)
    init_scales = np.zeros(n_valid, dtype=np.float32)
    for i, t in enumerate(valid_idx):
        init_trans[i] = mp_3d[t, 0]  # wrist position
        hand_size = np.linalg.norm(mp_3d[t, 12] - mp_3d[t, 0])
        if 50 < hand_size < 300:
            init_scales[i] = hand_size / mano_wrist_to_middle
        else:
            init_scales[i] = 110.0 / mano_wrist_to_middle

    median_scale = float(np.median(init_scales))

    global_rot = torch.zeros(n_valid, 1, 3, device=device, requires_grad=True)
    pose_pca = torch.zeros(n_valid, 1, NCOMPS, device=device, requires_grad=True)
    trans = torch.tensor(init_trans, device=device, dtype=torch.float32, requires_grad=True)
    log_scale = torch.tensor(
        np.log(median_scale), device=device, dtype=torch.float32, requires_grad=True
    )
    beta = torch.zeros(1, 10, device=device, dtype=torch.float32, requires_grad=True)
    offset_L = torch.zeros(1, 1, 2, device=device, dtype=torch.float32, requires_grad=True)
    offset_R = torch.zeros(1, 1, 2, device=device, dtype=torch.float32, requires_grad=True)

    logger.info(f"  Scale init: {median_scale:.1f}, fitting {n_valid} frames")
    report(25)

    # ── Stage 1 optimization ───────────────────────────────────
    n_iters = 300
    lr = 0.05

    optimizer = torch.optim.Adam([
        {"params": [global_rot], "lr": lr},
        {"params": [trans], "lr": lr * 5},
        {"params": [log_scale], "lr": lr * 0.1},
        {"params": [pose_pca], "lr": lr},
        {"params": [beta], "lr": lr * 0.5},
        {"params": [offset_L, offset_R], "lr": lr * 2},
    ])

    logger.info(f"  Running Stage 1 ({n_iters} iterations)...")

    for it in range(n_iters):
        if cancelled():
            return {"cancelled": True}

        optimizer.zero_grad()

        pose_input = torch.cat([global_rot, pose_pca], dim=2).squeeze(1)
        out = mano_model(pose_input, betas=beta.expand(n_valid, -1))

        # Transform joints: MANO → OpenCV coords → world
        scale = torch.exp(log_scale)
        joints = out.joints * 1000.0  # m → mm
        joints_cv = joints.clone()
        joints_cv[..., 1] = -joints_cv[..., 1]
        joints_cv[..., 2] = -joints_cv[..., 2]
        wrist = joints_cv[:, 0:1, :]
        j3d = (joints_cv - wrist) * scale + trans.unsqueeze(1)

        # Project to both cameras
        pL = _project_torch(j3d, K1, d1) + offset_L
        pR = _project_torch(j3d, K2, d2, R_stereo, T_stereo) + offset_R

        # Loss 1: MP reprojection
        loss_mp = ((pL - tgt_L) ** 2).sum(-1).mean() + ((pR - tgt_R) ** 2).sum(-1).mean()

        # Loss 2: Bone length constraint
        j1_pts = j3d[:, bone_idx[:, 0]]
        j2_pts = j3d[:, bone_idx[:, 1]]
        bone_lens = (j2_pts - j1_pts).norm(dim=2)
        loss_bone = ((bone_lens - target_bl) ** 2 * bl_w).mean()

        # Regularization
        loss_pose = 0.001 * (pose_pca ** 2).mean()
        loss_shape = 0.01 * (beta ** 2).sum()

        # Temporal smoothness
        if n_valid > 1:
            pose_vel = (pose_pca[1:] - pose_pca[:-1]).squeeze(1)
            rot_vel = global_rot[1:] - global_rot[:-1]
            trans_vel = trans[1:] - trans[:-1]
            loss_temporal = 5.0 * (pose_vel ** 2).mean()
            loss_temporal += 5.0 * (rot_vel ** 2).mean()
            loss_temporal += 0.05 * (trans_vel ** 2).mean()
        else:
            loss_temporal = torch.tensor(0.0)

        loss = loss_mp + 5.0 * loss_bone + loss_pose + loss_shape + loss_temporal

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [global_rot, pose_pca, trans, log_scale, beta, offset_L, offset_R],
            max_norm=1.0,
        )
        optimizer.step()

        # Report progress (25% → 90% during fitting)
        if it % 10 == 0:
            report(25 + (it / n_iters) * 65)

        if it % 100 == 0 or it == n_iters - 1:
            with torch.no_grad():
                eL = torch.sqrt(((pL - tgt_L) ** 2).sum(-1)).mean().item()
                eR = torch.sqrt(((pR - tgt_R) ** 2).sum(-1)).mean().item()
            logger.info(
                f"    iter {it}: L={eL:.1f}px R={eR:.1f}px "
                f"scale={torch.exp(log_scale).item():.1f} bone={loss_bone.item():.1f}"
            )

    report(90)

    # ── Compute final errors ───────────────────────────────────
    with torch.no_grad():
        pose_input = torch.cat([global_rot, pose_pca], dim=2).squeeze(1)
        out = mano_model(pose_input, betas=beta.expand(n_valid, -1))
        scale = torch.exp(log_scale)
        joints = out.joints * 1000.0
        joints_cv = joints.clone()
        joints_cv[..., 1] = -joints_cv[..., 1]
        joints_cv[..., 2] = -joints_cv[..., 2]
        wrist = joints_cv[:, 0:1, :]
        j3d = (joints_cv - wrist) * scale + trans.unsqueeze(1)

        pL = _project_torch(j3d, K1, d1) + offset_L
        pR = _project_torch(j3d, K2, d2, R_stereo, T_stereo) + offset_R

        err_L = torch.sqrt(((pL - tgt_L) ** 2).sum(-1)).mean(1).cpu().numpy()
        err_R = torch.sqrt(((pR - tgt_R) ** 2).sum(-1)).mean(1).cpu().numpy()

    logger.info(
        f"  Final: L={err_L.mean():.1f}±{err_L.std():.1f}px "
        f"R={err_R.mean():.1f}±{err_R.std():.1f}px "
        f"scale={torch.exp(log_scale).item():.1f}"
    )

    # ── Save results ───────────────────────────────────────────
    # Scatter into full-frame arrays
    all_joints = np.full((N, 21, 3), np.nan)
    all_err_L = np.full(N, np.nan)
    all_err_R = np.full(N, np.nan)

    j3d_np = j3d.detach().cpu().numpy()
    for i, t in enumerate(valid_idx):
        all_joints[t] = j3d_np[i]
        all_err_L[t] = err_L[i]
        all_err_R[t] = err_R[i]

    # Save to mano directory
    mano_trial_dir = _mano_dir(subject_name) / trial_stem
    mano_trial_dir.mkdir(parents=True, exist_ok=True)

    out_path = mano_trial_dir / "mano_fit.npz"
    np.savez(
        str(out_path),
        joints_3d=all_joints,
        fit_error_L=all_err_L,
        fit_error_R=all_err_R,
        beta=beta.detach().cpu().numpy().ravel(),
        scale=np.full(N, torch.exp(log_scale).item()),
        offset_L=offset_L.detach().cpu().numpy().ravel(),
        offset_R=offset_R.detach().cpu().numpy().ravel(),
        target_bone_lengths=target_bone_lengths,
        n_frames=N,
        stage=1,
    )

    # Clear cached data so next load picks up the new fit
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
    }


def _project_torch(pts3d, K, dist, R_ext=None, t_ext=None, min_z=10.0):
    """Differentiable projection with OpenCV radial+tangential distortion."""
    if R_ext is not None:
        pts3d = pts3d @ R_ext.T + t_ext
    X, Y = pts3d[..., 0], pts3d[..., 1]
    import torch
    Z = torch.clamp(pts3d[..., 2], min=min_z)
    x, y = X / Z, Y / Z
    r2 = x * x + y * y
    r4, r6 = r2 * r2, r2 * r2 * r2
    k1, k2, p1, p2, k3 = dist[0], dist[1], dist[2], dist[3], dist[4]
    radial = 1 + k1 * r2 + k2 * r4 + k3 * r6
    x_d = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    y_d = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return torch.stack([K[0, 0] * x_d + K[0, 2], K[1, 1] * y_d + K[1, 2]], dim=-1)
