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
from .mediapipe_prelabel import (
    load_mediapipe_prelabels,
    load_mediapipe_combined_prelabels,
)
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
)

logger = logging.getLogger(__name__)

BONES = HAND_SKELETON


def run_skeleton_v2_fit(
    subject_name: str,
    trial_stem: str,
    cancel_event: threading.Event | None = None,
    progress_callback: Callable[[float], None] | None = None,
    w_bone: float = 0.0,
    w_smooth_wrist: float = 5.0,
    w_smooth_xy: float = 10.0,
    w_smooth_z: float = 10.0,
    w_smooth_angles: float = 10.0,
) -> dict[str, Any]:
    """Frozen Skeleton v2 fit: MP combined only, absolute-position smoothing + z-constancy.

    Previously took MP weight, DLC weight, angle-constraint flag, and
    constraint weight; all four were dropped — MP combined is now
    the sole 2D input (weight implicitly 1.0) and the constraint
    term has been removed.  Wrist smoothing multiplier default
    bumped from 1.0 to 5.0 so the wrist tracks more rigidly.
    """
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

    # Prefer the per-frame fused Combined layer; fall back to the
    # forward-only baseline when no combined file exists yet.
    prelabels = load_mediapipe_combined_prelabels(subject_name)
    if prelabels is None:
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

    logger.info("  Sources: MP combined ✓")

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

    # ── 2D reproj targets: Stereo Fill when available, MP combined otherwise ──
    # Mirrors the same path skeleton_v1 uses.  When a saved MP
    # Filter sidecar AND a stereo bake (Hybrid > Image) both exist
    # for the trial, the filtered + stereo-donated labels become
    # the optimiser's targets; donated cells contribute the
    # stereo-corrected 2D point with full weight, dropped cells
    # (validation failed, or both cameras filtered) contribute
    # zero per camera so the bone-length + smoothness terms have
    # to take over.  Joint-weight tensor is the per-camera finite
    # mask × the trial-level joint_weights confidence from above.
    target_L_np = mp_L[valid_idx].copy()
    target_R_np = mp_R[valid_idx].copy()
    mask_L_np = np.ones((n_valid, 21), dtype=np.float32)
    mask_R_np = np.ones((n_valid, 21), dtype=np.float32)

    _sf_sidecar = _skeleton_dir(subject_name) / trial_stem / "mp_filter_params.json"
    if _sf_sidecar.exists():
        try:
            from .mp_filter import (
                load_saved_filter_params, detect_mask_from_params,
                build_and_validate_stereo_fill,
            )
            _saved_params = load_saved_filter_params(_sf_sidecar)
            # MP confidence + HRnet for the camera-mask filter — same
            # subset slicing pattern as skeleton_v1.
            _conf_L = prelabels.get("confidence_OS") if hasattr(prelabels, "get") else None
            _conf_R = prelabels.get("confidence_OD") if hasattr(prelabels, "get") else None
            if _conf_L is not None: _conf_L = _conf_L[start_frame:end][valid_idx]
            if _conf_R is not None: _conf_R = _conf_R[start_frame:end][valid_idx]
            _hr_L = None; _hr_R = None
            try:
                _hm_path = _skeleton_dir(subject_name) / trial_stem / "hrnet_w18_heatmaps.npz"
                if _hm_path.exists():
                    with np.load(str(_hm_path), allow_pickle=False) as _hm:
                        if "heatmaps_L" in _hm.files:
                            _hr_L = _hm["heatmaps_L"].max(axis=(2, 3))[valid_idx].astype(np.float32)
                        if "heatmaps_R" in _hm.files:
                            _hr_R = _hm["heatmaps_R"].max(axis=(2, 3))[valid_idx].astype(np.float32)
            except Exception:
                _hr_L = None; _hr_R = None
            # Hybrid > image stereo source.  Reconstruct per-cell
            # stereo points the same way the display + v1 pipelines do.
            _shift_mag_v = None; _resp_v = None
            _stereo_L_v = None; _stereo_R_v = None
            try:
                from .stereo_align import load_stereo_align
                _trial_idx = next(
                    (i for i, tt in enumerate(build_trial_map(subject_name))
                     if tt.get("trial_name") == trial_stem),
                    None,
                )
                _sa_hyb = (load_stereo_align(subject_name, _trial_idx, mode="hybrid")
                           if _trial_idx is not None else None)
                _sa = _sa_hyb or (load_stereo_align(subject_name, _trial_idx, mode="image")
                                  if _trial_idx is not None else None)
                if _sa is not None and "shifts" in _sa and "response" in _sa:
                    _shifts = np.asarray(_sa["shifts"], dtype=np.float64)
                    _resp_arr = np.asarray(_sa["response"], dtype=np.float64)
                    _N_sa = min(N, _shifts.shape[0])
                    _stereo_L_v = np.full((N, 21, 2), np.nan)
                    _stereo_R_v = np.full((N, 21, 2), np.nan)
                    _stereo_L_v[:_N_sa, :, 0] = mp_L[:_N_sa, :, 0] - _shifts[:_N_sa, :, 0]
                    _stereo_L_v[:_N_sa, :, 1] = mp_L[:_N_sa, :, 1] - _shifts[:_N_sa, :, 1]
                    _stereo_R_v[:_N_sa, :, 0] = mp_R[:_N_sa, :, 0] + _shifts[:_N_sa, :, 0]
                    _stereo_R_v[:_N_sa, :, 1] = mp_R[:_N_sa, :, 1] + _shifts[:_N_sa, :, 1]
                    _resp_v = np.full((N, 21), np.nan)
                    _resp_v[:_N_sa] = _resp_arr[:_N_sa]
                    _shift_mag_v = np.linalg.norm(_shifts, axis=-1)
            except Exception as _e:
                logger.debug(f"  Stereo Fill source load skipped: {_e}")

            if _stereo_L_v is not None:
                # First pass: camera mask from saved filter applied to
                # MP combined.  init_3d (wrist-filled) is what v2 will
                # optimise from; pass it as the 3D-aware input.
                _, camera_mask_v, _ = detect_mask_from_params(
                    _saved_params, init_3d,
                    mp_L[valid_idx], mp_R[valid_idx], BONES,
                    calib=calib,
                    confidence_L=_conf_L, confidence_R=_conf_R,
                    hrnet_L=_hr_L,        hrnet_R=_hr_R,
                    stereo_shift_mag=(_shift_mag_v[valid_idx]
                                       if _shift_mag_v is not None else None),
                    stereo_response=(_resp_v[valid_idx]
                                      if _resp_v is not None else None),
                )
                # Second pass: build + validate stereo fill.
                _shift_mag_valid = (_shift_mag_v[valid_idx]
                                     if _shift_mag_v is not None else None)
                _resp_valid = (_resp_v[valid_idx]
                                if _resp_v is not None else None)
                _fL_valid, _fR_valid, _f3d_valid, _donated_v, _validated_v = (
                    build_and_validate_stereo_fill(
                        mp_L[valid_idx], mp_R[valid_idx],
                        _stereo_L_v[valid_idx], _stereo_R_v[valid_idx],
                        _resp_valid, _shift_mag_valid,
                        camera_mask_v, _saved_params,
                        init_3d, calib, BONES,
                        confidence_L=_conf_L, confidence_R=_conf_R,
                        hrnet_L=_hr_L,        hrnet_R=_hr_R,
                        conf_min=0.4,
                    ))
                logger.info(
                    f"  Stereo Fill: {int(_donated_v.sum())} donations placed, "
                    f"{int(_validated_v.sum())} survived validation"
                )
                target_L_np = _fL_valid.astype(np.float32)
                target_R_np = _fR_valid.astype(np.float32)
                # Per-camera finite mask: 1 where the fill has a usable
                # value, 0 where dropped.  Multiplied with jw below.
                mask_L_np = np.isfinite(target_L_np[:, :, 0]).astype(np.float32)
                mask_R_np = np.isfinite(target_R_np[:, :, 0]).astype(np.float32)
                logger.info("  Optimising against Stereo Fill targets.")
        except Exception as _e:
            logger.warning(f"  Stereo Fill prep failed; falling back to MP combined: {_e}")

    # NaN → 0 so torch doesn't NaN-poison the reproj loss (the
    # corresponding mask entries are 0, so the literal value is
    # multiplied out — but NaN times zero is still NaN in float).
    target_L_np = np.where(np.isfinite(target_L_np), target_L_np, 0.0).astype(np.float32)
    target_R_np = np.where(np.isfinite(target_R_np), target_R_np, 0.0).astype(np.float32)
    tgt_mp_L = _t(target_L_np)
    tgt_mp_R = _t(target_R_np)
    tgt_mask_L = _t(mask_L_np)
    tgt_mask_R = _t(mask_R_np)
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

    # DLC and angle constraints removed from v2 — MP combined is the
    # only 2D input; the smoothness + bone-length terms below handle
    # everything else.

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

        # 2D reproj: targets are Stereo Fill when available, MP
        # combined otherwise.  Per-camera masks zero out cells the
        # Stereo Fill validation dropped (NaN→0 on the target side,
        # mask side carries 0 there too).
        loss = loss + (
            cam_w_L * _weighted_reproj(pL, tgt_mp_L, mask=tgt_mask_L)
            + cam_w_R * _weighted_reproj(pR, tgt_mp_R, mask=tgt_mask_R)
        )

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
                # Mean reproj error over masked-in cells only (dropped
                # joints shouldn't drag the mean toward zero).
                _eL_pj = torch.sqrt(((pL - tgt_mp_L) ** 2).sum(-1))
                _eR_pj = torch.sqrt(((pR - tgt_mp_R) ** 2).sum(-1))
                eL = ((_eL_pj * tgt_mask_L).sum() / tgt_mask_L.sum().clamp(min=1)).item()
                eR = ((_eR_pj * tgt_mask_R).sum() / tgt_mask_R.sum().clamp(min=1)).item()
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
        # Per-frame mean reproj error — weighted by the per-camera
        # mask so dropped cells (zero target) don't drag the average
        # toward zero or inflate it with the fitted joint's distance
        # from origin.
        _err_L_pj = torch.sqrt(((pL_f - tgt_mp_L) ** 2).sum(-1))   # (n_valid, 21)
        _err_R_pj = torch.sqrt(((pR_f - tgt_mp_R) ** 2).sum(-1))
        _denom_L = tgt_mask_L.sum(1).clamp(min=1)
        _denom_R = tgt_mask_R.sum(1).clamp(min=1)
        err_L = ((_err_L_pj * tgt_mask_L).sum(1) / _denom_L).cpu().numpy()
        err_R = ((_err_R_pj * tgt_mask_R).sum(1) / _denom_R).cpu().numpy()

    with torch.no_grad():
        # Mask the offset reduction too — dropped cells where
        # tgt is 0 would otherwise contribute (0 - projected) to
        # the global-offset median, biasing it.
        _off_L_t = (tgt_mp_L - pL_f) * tgt_mask_L.unsqueeze(-1)
        _off_R_t = (tgt_mp_R - pR_f) * tgt_mask_R.unsqueeze(-1)
        # Replace masked-out cells with NaN so nanmedian skips them.
        _off_L_t = _off_L_t.masked_fill(tgt_mask_L.unsqueeze(-1) == 0, float('nan'))
        _off_R_t = _off_R_t.masked_fill(tgt_mask_R.unsqueeze(-1) == 0, float('nan'))
        off_L = _off_L_t.detach().cpu().numpy()
        off_R = _off_R_t.detach().cpu().numpy()
    corr_L = np.nanmedian(off_L.reshape(-1, 2), axis=0).astype(np.float32)
    corr_R = np.nanmedian(off_R.reshape(-1, 2), axis=0).astype(np.float32)
    # nanmedian → NaN when all masked.  Treat as no offset.
    corr_L = np.where(np.isnan(corr_L), 0.0, corr_L).astype(np.float32)
    corr_R = np.where(np.isnan(corr_R), 0.0, corr_R).astype(np.float32)
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
        "sources": {"mediapipe_combined": True},
        "params": {
            "w_bone": w_bone,
            "w_smooth_wrist": w_smooth_wrist,
            "w_smooth_xy": w_smooth_xy,
            "w_smooth_z": w_smooth_z,
            "w_smooth_angles": w_smooth_angles,
        },
        "results": {
            "mean_error_L": float(np.nanmean(all_err_L)),
            "mean_error_R": float(np.nanmean(all_err_R)),
            "bone_lengths": final_bone_lengths.tolist(),
        },
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
