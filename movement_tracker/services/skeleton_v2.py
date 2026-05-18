"""
Skeleton v2 (FROZEN): FK skeleton fit with legacy absolute-position smoothing.

This is a snapshot of the skeleton-fit algorithm preserved as "Skeleton v2" in
the UI.  It uses:
  - Absolute-position XY/Z velocity-accel-jerk smoothing on ALL joints
  - Per-joint Z-constancy (mean-Z deviation penalty)
  - No HRNet, no Apple Vision — MediaPipe + DLC tips only

Future changes to `skeleton_v3.py` (the "Skeleton fit" button) must NOT
propagate here.  This module is a frozen copy; edit only to fix bugs present
at the time it was forked.
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
from .skeleton_data import _skeleton_dir, HAND_SKELETON
from .mediapipe_prelabel import load_mediapipe_prelabels
from .video import build_trial_map
# FK helpers are pulled from the active v2 module at time of forking.
# These are mathematical primitives (FK, inverse-FK, wrist-frame) and are
# considered stable enough to share — if they ever need to diverge, copy them
# in here.
from .skeleton_v3 import (
    _forward_kinematics,
    _compute_wrist_frame,
    _inverse_fk_init,
    _load_vision_lm,
    _load_dlc_tips,
)

logger = logging.getLogger(__name__)

BONES = HAND_SKELETON


def run_skeleton_v2_fit(
    subject_name: str,
    trial_stem: str,
    cancel_event: threading.Event | None = None,
    progress_callback: Callable[[float], None] | None = None,
    w_mediapipe: float = 1.0,
    w_dlc: float = 1.0,
    w_bone: float = 0.0,
    w_smooth_wrist: float = 1.0,
    w_smooth_xy: float = 10.0,
    w_smooth_z: float = 10.0,
    w_smooth_angles: float = 10.0,
    use_angle_constraints: bool = True,
    w_constraints: float = 10.0,
) -> dict[str, Any]:
    """Frozen Skeleton v2 fit: MP + DLC only, absolute-position smoothing + z-constancy."""
    import torch
    import torch.nn.functional as F  # noqa: F401

    device = torch.device("cpu")

    def report(pct):
        if progress_callback:
            progress_callback(pct)

    def cancelled():
        return cancel_event is not None and cancel_event.is_set()

    report(0)
    logger.info(f"[Skeleton v2 (legacy)] {subject_name}/{trial_stem}")

    trials = build_trial_map(subject_name)
    trial_info = next((t for t in trials if t["trial_name"] == trial_stem), None)
    if trial_info is None:
        raise ValueError(f"Trial {trial_stem} not found for {subject_name}")

    n_frames    = trial_info["frame_count"]
    start_frame = trial_info.get("start_frame", 0)

    prelabels = load_mediapipe_prelabels(subject_name)
    if prelabels is None:
        raise ValueError(f"No MediaPipe prelabels for {subject_name}")
    os_lm = prelabels["OS_landmarks"]
    od_lm = prelabels["OD_landmarks"]
    end = min(start_frame + n_frames, os_lm.shape[0])
    mp_L = os_lm[start_frame:end].copy().astype(np.float32)
    mp_R = od_lm[start_frame:end].copy().astype(np.float32)
    N = mp_L.shape[0]

    from .skeleton_data import _load_trial_calibration
    calib = _load_trial_calibration(subject_name, trial_stem)
    if calib is None:
        calib = get_calibration_for_subject(subject_name)
    if calib is None:
        raise ValueError(f"No stereo calibration for {subject_name}")

    report(3)
    if cancelled():
        return {"cancelled": True}

    th_L, th_R, ix_L, ix_R = _load_dlc_tips(subject_name, start_frame, N)
    has_dlc = (not np.isnan(th_L).all()) or (not np.isnan(ix_L).all())

    logger.info(f"  Sources: MP ✓  DLC {'✓' if has_dlc else '✗'}")

    logger.info(f"  Triangulating {N} frames...")
    mp_3d = np.full((N, 21, 3), np.nan, dtype=np.float32)
    for j in range(21):
        mp_3d[:, j] = triangulate_points(mp_L[:, j], mp_R[:, j], calib).astype(np.float32)

    from .skeleton_data import _project_to_2d
    _rp_L = _project_to_2d(mp_3d, calib["K1"], calib["dist1"],
                            np.eye(3, dtype=np.float64),
                            np.zeros((3, 1), dtype=np.float64))
    _rp_R = _project_to_2d(mp_3d, calib["K2"], calib["dist2"],
                            calib["R"], calib["T"])
    _rt_err_L = np.nanmean(np.abs(_rp_L - mp_L))
    _rt_err_R = np.nanmean(np.abs(_rp_R - mp_R))
    cam_w_L = 1.0
    cam_w_R = 1.0
    logger.info(f"  Calib roundtrip: L={_rt_err_L:.1f}px  R={_rt_err_R:.1f}px")

    valid_mask = (~np.isnan(mp_L[:, 0, 0])
                  & ~np.isnan(mp_R[:, 0, 0])
                  & ~np.isnan(mp_3d[:, 0, 0]))
    valid_idx  = np.where(valid_mask)[0]
    n_valid    = len(valid_idx)

    if n_valid < 10:
        raise ValueError(f"Only {n_valid} valid frames (need ≥ 10)")

    logger.info(f"  {n_valid}/{N} valid frames")
    report(8)

    # Robust bone-length targets
    bone_lengths_all = np.zeros((n_valid, 20), dtype=np.float32)
    for b, (j1, j2) in enumerate(BONES):
        diff = mp_3d[valid_idx, j2] - mp_3d[valid_idx, j1]
        bone_lengths_all[:, b] = np.linalg.norm(diff, axis=1)

    target_bone_lengths = np.zeros(20, dtype=np.float32)
    for b in range(20):
        bl = bone_lengths_all[:, b]
        bl = bl[~np.isnan(bl) & (bl > 0)]
        if len(bl) < 5:
            target_bone_lengths[b] = 30.0
            continue
        med = np.median(bl)
        mad = np.median(np.abs(bl - med))
        sigma = max(mad * 1.4826, 0.1)
        inlier = np.abs(bl - med) < 3.0 * sigma
        bl_clean = bl[inlier] if inlier.sum() >= 5 else bl
        target_bone_lengths[b] = float(np.median(bl_clean))

    # Per-joint per-frame confidence weights
    joint_weights = np.ones((n_valid, 21), dtype=np.float32)
    tri_valid = mp_3d[valid_idx]
    if n_valid > 12:
        HALF_WIN = 5
        resid = np.full((n_valid, 21, 3), np.nan)
        for i in range(n_valid):
            lo, hi = max(0, i - HALF_WIN), min(n_valid, i + HALF_WIN + 1)
            with np.errstate(all='ignore'):
                baseline = np.nanmedian(tri_valid[lo:hi], axis=0)
            resid[i] = np.abs(tri_valid[i] - baseline)
        z_resid = resid[:, :, 2]
        xy_resid = np.maximum(resid[:, :, 0], resid[:, :, 1])
        with np.errstate(divide='ignore', invalid='ignore'):
            z_ratio = z_resid / np.maximum(xy_resid, 0.3)
        RATIO_THRESH = 3.0
        ABS_Z_THRESH = 8.0
        w_r = np.where(z_ratio > RATIO_THRESH,
                        RATIO_THRESH / np.maximum(z_ratio, RATIO_THRESH), 1.0)
        w_a = np.where(z_resid > ABS_Z_THRESH,
                        ABS_Z_THRESH / np.maximum(z_resid, ABS_Z_THRESH), 1.0)
        joint_weights = np.clip(np.minimum(w_r, w_a), 0.01, 1.0)
        D2D_THRESH = 15.0
        mp_valid_L = mp_L[valid_idx]
        mp_valid_R = mp_R[valid_idx]
        for mp_2d in [mp_valid_L, mp_valid_R]:
            d2d = np.sqrt(np.sum(np.diff(mp_2d, axis=0) ** 2, axis=2))
            for i in range(d2d.shape[0]):
                for j in range(21):
                    if np.isnan(d2d[i, j]):
                        continue
                    if d2d[i, j] > D2D_THRESH:
                        w = D2D_THRESH / d2d[i, j]
                        joint_weights[i, j]   = min(joint_weights[i, j], w)
                        joint_weights[i+1, j] = min(joint_weights[i+1, j], w)
        joint_weights = np.clip(joint_weights, 0.01, 1.0)

    report(13)

    init_3d = mp_3d[valid_idx].copy()
    for i in range(n_valid):
        for j in range(21):
            if np.isnan(init_3d[i, j, 0]):
                init_3d[i, j] = init_3d[i, 0] if not np.isnan(init_3d[i, 0, 0]) else np.array([0, 0, 500])

    R_wrist_init  = _compute_wrist_frame(init_3d)
    flex_init, abd_init = _inverse_fk_init(init_3d, R_wrist_init)
    r6d_init = np.concatenate(
        [R_wrist_init[:, :, 0], R_wrist_init[:, :, 1]], axis=-1
    ).astype(np.float32)

    report(18)
    if cancelled():
        return {"cancelled": True}

    K1       = torch.tensor(calib["K1"],           dtype=torch.float32)
    K2       = torch.tensor(calib["K2"],           dtype=torch.float32)
    d1       = torch.tensor(calib["dist1"].ravel(),dtype=torch.float32)
    d2       = torch.tensor(calib["dist2"].ravel(),dtype=torch.float32)
    R_stereo = torch.tensor(calib["R"],            dtype=torch.float32)
    T_stereo = torch.tensor(calib["T"].ravel(),    dtype=torch.float32)

    def _t(arr, req_grad=False):
        return torch.tensor(arr, dtype=torch.float32, device=device,
                            requires_grad=req_grad)

    wrist_pos    = _t(init_3d[:, 0],     req_grad=True)
    wrist_r6d    = _t(r6d_init,          req_grad=True)
    bone_lengths = _t(target_bone_lengths, req_grad=True)
    flex_angs    = _t(flex_init,         req_grad=True)
    abd_angs     = _t(abd_init,          req_grad=True)

    META_BONES = [1, 2, 3, 4]
    meta_flex = _t(np.nanmedian(flex_init[:, META_BONES], axis=0), req_grad=True)
    meta_abd  = _t(np.nanmedian(abd_init[:, META_BONES],  axis=0), req_grad=True)

    tgt_mp_L = _t(mp_L[valid_idx])
    tgt_mp_R = _t(mp_R[valid_idx])
    jw = _t(joint_weights)

    # Adaptive smoothing
    smooth_scale = np.ones(n_valid, dtype=np.float32)
    mp_valid_L = mp_L[valid_idx]
    if n_valid > 2:
        d2d = np.sqrt(np.sum(np.diff(mp_valid_L, axis=0) ** 2, axis=2))
        d2d_max = np.nanmax(d2d, axis=1)
        med_d2d = np.nanmedian(d2d_max)
        MOTION_THRESH = max(med_d2d * 3.0, 5.0)
        for i in range(len(d2d_max)):
            if d2d_max[i] > MOTION_THRESH:
                scale = max(MOTION_THRESH / d2d_max[i], 0.1)
                smooth_scale[i]   = min(smooth_scale[i],   scale)
                smooth_scale[i+1] = min(smooth_scale[i+1], scale)
    smooth_w = _t(smooth_scale)

    if has_dlc:
        dlc_tgt_L = _t(np.stack([th_L[valid_idx], ix_L[valid_idx]], axis=1))
        dlc_tgt_R = _t(np.stack([th_R[valid_idx], ix_R[valid_idx]], axis=1))
        dlc_mask_L = ~torch.isnan(dlc_tgt_L[:, :, 0])
        dlc_mask_R = ~torch.isnan(dlc_tgt_R[:, :, 0])
        dlc_tgt_L  = torch.nan_to_num(dlc_tgt_L, nan=0.0)
        dlc_tgt_R  = torch.nan_to_num(dlc_tgt_R, nan=0.0)
    else:
        dlc_tgt_L = dlc_tgt_R = dlc_mask_L = dlc_mask_R = None

    angle_prior_data = None
    _angle_priors_list = []
    if use_angle_constraints:
        from .skeleton_data import load_angle_priors
        angle_prior_data = load_angle_priors()
        _angle_priors_list = angle_prior_data.get("joints", [])

    n_iters   = 400
    optimizer = torch.optim.Adam([
        {"params": [wrist_pos],    "lr": 0.3},
        {"params": [wrist_r6d],    "lr": 0.01},
        {"params": [bone_lengths], "lr": 0.05},
        {"params": [flex_angs],    "lr": 0.01},
        {"params": [abd_angs],     "lr": 0.005},
        {"params": [meta_flex],    "lr": 0.005},
        {"params": [meta_abd],     "lr": 0.005},
    ])

    from .skeleton_v1 import _project_torch

    target_bl = _t(target_bone_lengths)

    logger.info(f"  Running legacy fitting ({n_iters} iters, {n_valid} frames)...")
    report(20)

    # HRNet offset table (for DLC distal correction, same as active fit)
    HM_OFFSETS = {
        4: (3, 0.43),  8: (7, 0.32),  12: (11, 0.40), 16: (15, 0.40), 20: (19, 0.55),
        3: (2, 0.28),  7: (6, 0.19),  11: (10, 0.12), 15: (14, 0.17), 19: (18, 0.02),
        2: (1, 0.12),  6: (5, 0.10),  10: (9, 0.13),  14: (13, 0.04), 18: (17, 0.05),
    }

    for it in range(n_iters):
        if cancelled():
            return {"cancelled": True}

        optimizer.zero_grad()
        bl_clamped = torch.clamp(bone_lengths, min=1.0)
        flex_combined = flex_angs.clone()
        abd_combined  = abd_angs.clone()
        flex_combined[:, META_BONES] = meta_flex.unsqueeze(0)
        abd_combined[:, META_BONES]  = meta_abd.unsqueeze(0)

        joints_3d = _forward_kinematics(wrist_pos, wrist_r6d,
                                        bl_clamped, flex_combined, abd_combined)
        pL = _project_torch(joints_3d, K1, d1)
        pR = _project_torch(joints_3d, K2, d2, R_stereo, T_stereo)

        loss = torch.tensor(0.0)

        def _weighted_reproj(pred, tgt, mask=None):
            err = ((pred - tgt) ** 2).sum(-1)
            w = jw if mask is None else jw * mask.float()
            return (err * w).sum() / w.sum().clamp(min=1)

        if w_mediapipe > 0:
            loss = loss + w_mediapipe * (
                cam_w_L * _weighted_reproj(pL, tgt_mp_L)
                + cam_w_R * _weighted_reproj(pR, tgt_mp_R)
            )

        if w_dlc > 0 and has_dlc:
            # Apply distal offset for DLC (labels are at tip, skeleton joint is at knuckle)
            joints_ext = joints_3d.clone()
            for j, (p, ext) in HM_OFFSETS.items():
                if ext > 0.01:
                    bone_dir = joints_3d[:, j] - joints_3d[:, p]
                    joints_ext[:, j] = joints_3d[:, j] + ext * bone_dir
            pL_ext = _project_torch(joints_ext, K1, d1)
            pR_ext = _project_torch(joints_ext, K2, d2, R_stereo, T_stereo)
            pL_tips = torch.stack([pL_ext[:, 4], pL_ext[:, 8]], dim=1)
            pR_tips = torch.stack([pR_ext[:, 4], pR_ext[:, 8]], dim=1)
            jw_dlc = torch.stack([jw[:, 4], jw[:, 8]], dim=1)
            err_L = ((pL_tips - dlc_tgt_L) ** 2).sum(-1) * dlc_mask_L.float() * jw_dlc
            err_R = ((pR_tips - dlc_tgt_R) ** 2).sum(-1) * dlc_mask_R.float() * jw_dlc
            denom = (dlc_mask_L.float() * jw_dlc + dlc_mask_R.float() * jw_dlc).sum().clamp(min=1)
            loss = loss + w_dlc * (cam_w_L * err_L.sum() + cam_w_R * err_R.sum()) / denom

        if w_bone > 0:
            loss = loss + w_bone * ((bl_clamped - target_bl) ** 2).mean()

        sw_vel = torch.minimum(smooth_w[:-1], smooth_w[1:])
        sw_acc = torch.minimum(sw_vel[:-1], sw_vel[1:]) if n_valid > 2 else None
        sw_jrk = torch.minimum(sw_acc[:-1], sw_acc[1:]) if n_valid > 3 and sw_acc is not None else None

        # ── LEGACY SMOOTHING: absolute-position on ALL joints + z-constancy ──
        jw_smooth = torch.ones(21, device=device)
        jw_smooth[0] = w_smooth_wrist

        if (w_smooth_xy > 0 or w_smooth_z > 0) and n_valid > 1:
            vel_3d = joints_3d[1:] - joints_3d[:-1]
            sw_v = sw_vel.unsqueeze(1)
            jw_s = jw_smooth.unsqueeze(0)
            if w_smooth_xy > 0:
                vel_xy = vel_3d[:, :, :2]
                loss_xy = 0.05 * (sw_v.unsqueeze(2) * jw_s.unsqueeze(2) * vel_xy ** 2).mean()
                if n_valid > 2:
                    acc_xy = vel_xy[1:] - vel_xy[:-1]
                    loss_xy = loss_xy + 0.02 * (sw_acc.unsqueeze(1).unsqueeze(2) * jw_s.unsqueeze(2) * acc_xy ** 2).mean()
                if n_valid > 3 and sw_jrk is not None:
                    jrk_xy = acc_xy[1:] - acc_xy[:-1]
                    loss_xy = loss_xy + 0.01 * (sw_jrk.unsqueeze(1).unsqueeze(2) * jw_s.unsqueeze(2) * jrk_xy ** 2).mean()
                loss = loss + w_smooth_xy * loss_xy
            if w_smooth_z > 0:
                vel_z = vel_3d[:, :, 2:3]
                loss_z = 0.05 * (sw_v.unsqueeze(2) * jw_s.unsqueeze(2) * vel_z ** 2).mean()
                if n_valid > 2:
                    acc_z = vel_z[1:] - vel_z[:-1]
                    loss_z = loss_z + 0.02 * (sw_acc.unsqueeze(1).unsqueeze(2) * jw_s.unsqueeze(2) * acc_z ** 2).mean()
                if n_valid > 3 and sw_jrk is not None:
                    jrk_z = acc_z[1:] - acc_z[:-1]
                    loss_z = loss_z + 0.01 * (sw_jrk.unsqueeze(1).unsqueeze(2) * jw_s.unsqueeze(2) * jrk_z ** 2).mean()
                z_vals = joints_3d[:, :, 2]
                z_per_joint_mean = z_vals.mean(dim=0, keepdim=True)
                z_dev = z_vals - z_per_joint_mean
                loss_z = loss_z + 0.1 * (z_dev ** 2).mean()
                loss = loss + w_smooth_z * loss_z

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

        if use_angle_constraints and _angle_priors_list and w_constraints > 0:
            from .angle_constraint_loss import compute_angle_constraint_loss
            _constraint_groups = angle_prior_data.get("constraint_groups") if angle_prior_data else None
            loss = loss + w_constraints * compute_angle_constraint_loss(joints_3d, _angle_priors_list, _constraint_groups)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        all_params = [wrist_pos, wrist_r6d, bone_lengths, flex_angs, abd_angs, meta_flex, meta_abd]
        for p in all_params:
            if p.grad is not None:
                torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
        if flex_angs.grad is not None:
            flex_angs.grad[:, META_BONES] = 0
        if abd_angs.grad is not None:
            abd_angs.grad[:, META_BONES] = 0
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=5.0)
        optimizer.step()

        MCP_ABD_BONES = [8, 11, 14, 17]
        with torch.no_grad():
            mcp_abd_mean = abd_angs[:, MCP_ABD_BONES].mean(dim=1, keepdim=True)
            abd_angs[:, MCP_ABD_BONES] -= mcp_abd_mean

        if it % 10 == 0:
            report(20 + (it / n_iters) * 70)

        if it % 100 == 0 or it == n_iters - 1:
            with torch.no_grad():
                eL = torch.sqrt(((pL - tgt_mp_L) ** 2).sum(-1)).mean().item()
                eR = torch.sqrt(((pR - tgt_mp_R) ** 2).sum(-1)).mean().item()
                be = (bl_clamped - target_bl).abs().mean().item()
            logger.info(f"    iter {it}: L={eL:.1f}px R={eR:.1f}px bone_err={be:.1f}mm")

    report(90)

    with torch.no_grad():
        bl_final = torch.clamp(bone_lengths, min=1.0)
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

    with torch.no_grad():
        off_L = (tgt_mp_L - pL_f).detach().cpu().numpy()
        off_R = (tgt_mp_R - pR_f).detach().cpu().numpy()
    corr_L = np.nanmedian(off_L.reshape(-1, 2), axis=0).astype(np.float32)
    corr_R = np.nanmedian(off_R.reshape(-1, 2), axis=0).astype(np.float32)
    if np.linalg.norm(corr_L) < 5: corr_L = np.zeros(2, dtype=np.float32)
    if np.linalg.norm(corr_R) < 5: corr_R = np.zeros(2, dtype=np.float32)

    logger.info(f"  Final: L={err_L.mean():.1f}±{err_L.std():.1f}px  "
                f"R={err_R.mean():.1f}±{err_R.std():.1f}px")

    all_joints = np.full((N, 21, 3), np.nan, dtype=np.float32)
    all_err_L  = np.full(N, np.nan, dtype=np.float32)
    all_err_R  = np.full(N, np.nan, dtype=np.float32)
    j3d_np = joints_final.detach().cpu().numpy()
    for i, t in enumerate(valid_idx):
        all_joints[t] = j3d_np[i]
        all_err_L[t]  = err_L[i]
        all_err_R[t]  = err_R[i]

    final_bone_lengths = bl_final.detach().cpu().numpy()

    skeleton_trial_dir = _skeleton_dir(subject_name) / trial_stem
    skeleton_trial_dir.mkdir(parents=True, exist_ok=True)
    out_path = skeleton_trial_dir / "skeleton_v2.npz"
    params_path = skeleton_trial_dir / "skeleton_v2_params.json"
    # Legacy fit has no history rotation — it overwrites its own file.
    # The active "Skeleton" fit has its own history chain under skeleton_v3_prev*.npz.

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
        fit_type="skeleton_v2_legacy",
    )

    from datetime import datetime
    params_path.write_text(json.dumps({
        "fit_type": "skeleton_v2_legacy",
        "version": "v2-legacy",
        "subject": subject_name,
        "trial": trial_stem,
        "n_frames": int(N),
        "n_fitted": int(n_valid),
        "n_iters": n_iters,
        "sources": {"mediapipe": True, "dlc": has_dlc},
        "params": {
            "w_mediapipe": w_mediapipe,
            "w_dlc": w_dlc,
            "w_bone": w_bone,
            "w_smooth_wrist": w_smooth_wrist,
            "w_smooth_xy": w_smooth_xy,
            "w_smooth_z": w_smooth_z,
            "w_smooth_angles": w_smooth_angles,
            "use_angle_constraints": use_angle_constraints,
            "w_constraints": w_constraints,
        },
        "results": {
            "mean_error_L": float(np.nanmean(all_err_L)),
            "mean_error_R": float(np.nanmean(all_err_R)),
            "bone_lengths": final_bone_lengths.tolist(),
        },
        "angle_constraints": angle_prior_data,
        "timestamp": datetime.now().isoformat(),
    }, indent=2))

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
        "bone_lengths": final_bone_lengths.tolist(),
    }
