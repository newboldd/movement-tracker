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
from .mediapipe_prelabel import (
    load_mediapipe_prelabels,
    load_mediapipe_combined_prelabels,
)
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
    # Outlier pre-filter thresholds (see _detect_outlier_per_joint).
    accel_k: float = 6.0,
    bone_k: float = 6.0,
    k_max: int = 30,
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

    # MediaPipe data — prefer the per-frame fused Combined layer; fall
    # back to the forward-only baseline when no combined file exists
    # yet (e.g. very old subjects, or trials with only one source).
    prelabels = load_mediapipe_combined_prelabels(subject_name)
    if prelabels is None:
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

    # ── Prepare scalar/static tensors ──────────────────────────
    K1 = torch.tensor(calib["K1"], dtype=torch.float32, device=device)
    K2 = torch.tensor(calib["K2"], dtype=torch.float32, device=device)
    d1 = torch.tensor(calib["dist1"].ravel(), dtype=torch.float32, device=device)
    d2 = torch.tensor(calib["dist2"].ravel(), dtype=torch.float32, device=device)
    R_stereo = torch.tensor(calib["R"], dtype=torch.float32, device=device)
    T_stereo = torch.tensor(calib["T"].ravel(), dtype=torch.float32, device=device)

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

    # ── Pre-pass: per-joint per-camera outlier mask ────────────
    #
    # Three signals composed (per joint, per frame; final mask
    # also per camera):
    #   1. Per-joint 3D acceleration-spike pairing identifies
    #      run-boundary frames for each joint.
    #   2. Per-bone length z-score flags anomalous bones; we
    #      blame ONE endpoint per anomalous bone by comparing
    #      the two endpoints' 3D step magnitudes
    #      (k_blame_joint = 2.0).  AND'ing with signal 1 gives
    #      the per-joint outlier mask.
    #   3. Per-joint per-camera 2D-pixel step magnitudes
    #      attribute blame to L, R, or both (k_blame_camera =
    #      2.0) for each masked joint.
    # Prefer the user's saved MP Filter sidecar when present —
    # that's the per-trial filter they tuned and "saved" from the
    # Labels page.  Falls back to the legacy internal detector
    # when no sidecar exists.
    skeleton_trial_dir = _skeleton_dir(subject_name) / trial_stem
    mp_filter_sidecar = skeleton_trial_dir / "mp_filter_params.json"
    # Default: 2D reproj targets are MP combined (or forward
    # fallback) — overridden below to Stereo Fill when a saved
    # filter AND a stereo bake both exist.
    target_L_np = mp_L
    target_R_np = mp_R
    _saved_params = None
    if mp_filter_sidecar.exists():
        from .mp_filter import (
            load_saved_filter_params, detect_mask_from_params,
            build_and_validate_stereo_fill,
        )
        _saved_params = load_saved_filter_params(mp_filter_sidecar)
        # MP confidence and HRnet data — best-effort, optional.
        _conf_L = prelabels.get("confidence_OS") if hasattr(prelabels, "get") else None
        _conf_R = prelabels.get("confidence_OD") if hasattr(prelabels, "get") else None
        if _conf_L is not None: _conf_L = _conf_L[start_frame:end][valid_idx]
        if _conf_R is not None: _conf_R = _conf_R[start_frame:end][valid_idx]
        # HRnet peak scores from heatmaps.npz (max over spatial dims).
        _hr_L = None; _hr_R = None
        try:
            _hm_path = skeleton_trial_dir / "hrnet_w18_heatmaps.npz"
            if _hm_path.exists():
                with np.load(str(_hm_path), allow_pickle=False) as _hm:
                    if "heatmaps_L" in _hm.files:
                        _hr_L = _hm["heatmaps_L"].max(axis=(2, 3))[valid_idx].astype(np.float32)
                    if "heatmaps_R" in _hm.files:
                        _hr_R = _hm["heatmaps_R"].max(axis=(2, 3))[valid_idx].astype(np.float32)
        except Exception:
            _hr_L = None; _hr_R = None
        # Hybrid stereo (preferred) + image fallback — same as the
        # Stereo Fill emission in skeleton_data.  Loads shifts +
        # response and rebuilds the per-cell stereo points in OS/OD
        # space so build_and_validate_stereo_fill can compose with
        # MP combined.
        _shift_mag_v = None
        _resp_v = None
        _stereo_L_v = None
        _stereo_R_v = None
        try:
            from .stereo_align import load_stereo_align
            from .video import build_trial_map
            _trial_idx = next(
                (i for i, t in enumerate(build_trial_map(subject_name))
                 if t.get("trial_name") == trial_stem),
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
                # Anchor stereo to MP combined where finite, forward
                # fallback elsewhere — matches the display code.
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

        # First pass: camera mask straight from the saved params on
        # the MP combined frames the optimiser would see.
        _, camera_mask, _per_sig = detect_mask_from_params(
            _saved_params, init_3d, mp_L[valid_idx], mp_R[valid_idx], BONES,
            calib=calib,
            confidence_L=_conf_L, confidence_R=_conf_R,
            hrnet_L=_hr_L,        hrnet_R=_hr_R,
            stereo_shift_mag=(_shift_mag_v[valid_idx]
                               if _shift_mag_v is not None else None),
            stereo_response=(_resp_v[valid_idx]
                              if _resp_v is not None else None),
        )
        joint_outlier_mask = camera_mask.any(axis=-1)
        # Second pass: build + validate stereo fill, then use it as
        # the optimiser's 2D target.  Requires a stereo bake; without
        # one we just keep MP combined as the target and downweight
        # filtered cells.
        if _stereo_L_v is not None:
            _shift_mag_valid = _shift_mag_v[valid_idx] if _shift_mag_v is not None else None
            _resp_valid = _resp_v[valid_idx] if _resp_v is not None else None
            _stereo_L_valid = _stereo_L_v[valid_idx]
            _stereo_R_valid = _stereo_R_v[valid_idx]
            _fL_valid, _fR_valid, _f3d_valid, _donated_v, _validated_v = (
                build_and_validate_stereo_fill(
                    mp_L[valid_idx], mp_R[valid_idx],
                    _stereo_L_valid, _stereo_R_valid,
                    _resp_valid, _shift_mag_valid,
                    camera_mask, _saved_params,
                    init_3d, calib, BONES,
                    confidence_L=_conf_L, confidence_R=_conf_R,
                    hrnet_L=_hr_L,        hrnet_R=_hr_R,
                    conf_min=0.4,
                ))
            n_donated_v = int(_donated_v.sum())
            n_validated_v = int(_validated_v.sum())
            logger.info(
                f"  Stereo Fill: {n_donated_v} donations placed, "
                f"{n_validated_v} survived validation"
            )
            # Splice fill back into trial-length target arrays so
            # downstream code (which still indexes by valid_idx) sees
            # the right values.  Non-valid frames stay at their MP
            # combined values — they're not part of the fit anyway.
            target_L_np = mp_L.copy()
            target_R_np = mp_R.copy()
            target_L_np[valid_idx] = _fL_valid
            target_R_np[valid_idx] = _fR_valid
            # The camera mask used for joint_weight switches to
            # "where is the fill target NaN?" — that's the truthful
            # picture of "this camera has no usable 2D for this
            # joint at this frame" after the fill+validate pass.
            _nan_L = ~np.isfinite(_fL_valid[:, :, 0])
            _nan_R = ~np.isfinite(_fR_valid[:, :, 0])
            camera_mask = np.stack([_nan_L, _nan_R], axis=-1)
        logger.info(f"  Outlier pre-filter: USING SAVED MP FILTER ({mp_filter_sidecar.name})")
    else:
        joint_outlier_mask, camera_mask = _detect_outlier_per_joint(
            init_3d, mp_L[valid_idx], mp_R[valid_idx], BONES,
            K_max=int(k_max), accel_k=float(accel_k), bone_k=float(bone_k),
            k_blame_joint=2.0, k_blame_camera=2.0,
        )
        logger.info("  Outlier pre-filter: internal detector (no MP Filter sidecar)")
    n_joint_outliers = int(joint_outlier_mask.sum())
    n_frames_any = int(joint_outlier_mask.any(axis=1).sum())
    n_cam_L = int(camera_mask[:, :, 0].sum())
    n_cam_R = int(camera_mask[:, :, 1].sum())
    logger.info(
        f"  → {n_joint_outliers} (joint, frame) cells masked "
        f"across {n_frames_any}/{n_valid} frames; "
        f"camera attribution L={n_cam_L} R={n_cam_R}"
    )

    # Per (frame, joint, camera) weights for the reproj loss.  Zero
    # on cells the filter (or fill+validate) marked as bad — the
    # smoothness + bone terms still apply, so masked joints get
    # pulled toward neighbor-interpolated positions through the
    # smoothness gradient.  A joint with one camera masked still
    # gets a reproj anchor from the OTHER camera.
    jw_L_np = (1.0 - camera_mask[:, :, 0].astype(np.float32))
    jw_R_np = (1.0 - camera_mask[:, :, 1].astype(np.float32))
    joint_weight_L = torch.tensor(jw_L_np, device=device, dtype=torch.float32)
    joint_weight_R = torch.tensor(jw_R_np, device=device, dtype=torch.float32)

    # 2D reproj targets: Stereo Fill when available, MP combined
    # otherwise.  Donated cells carry the stereo-corrected point as
    # the target; dropped cells carry NaN — replace with 0 here so
    # torch doesn't NaN-poison the loss (weight is zero there
    # regardless, so the literal value doesn't matter).
    _tgt_L_np = np.where(np.isfinite(target_L_np[valid_idx]),
                          target_L_np[valid_idx], 0.0).astype(np.float32)
    _tgt_R_np = np.where(np.isfinite(target_R_np[valid_idx]),
                          target_R_np[valid_idx], 0.0).astype(np.float32)
    tgt_L = torch.tensor(_tgt_L_np, device=device, dtype=torch.float32)
    tgt_R = torch.tensor(_tgt_R_np, device=device, dtype=torch.float32)

    # Optimizable: 3D joint positions only.
    #
    # We used to also optimize per-camera 2D offsets (offset_L /
    # offset_R) to soak up residual stereo-calibration error, but
    # that created a perverse outcome: with reproj-only settings
    # the optimizer would drift the offsets to non-zero and shift
    # joints_3d to compensate, so the saved 3D pose no longer
    # triangulates back to the MP-combined 2D points the user fed
    # in.  The display path then tried to "fold" the saved offsets
    # back into joints_3d via a single 3D shift (averaging the two
    # cameras and converting through the LEFT intrinsics), which
    # can't reproduce two independent per-camera 2D offsets — so
    # both cameras' reprojections came out shifted relative to
    # MP-combined.  Dropping the offsets makes the loss minimum
    # equal to a per-frame triangulation, which is the behaviour
    # the rest of the app assumes.
    joints_3d = torch.tensor(init_3d, device=device, dtype=torch.float32, requires_grad=True)
    offset_L = torch.zeros(1, 1, 2, device=device, dtype=torch.float32)
    offset_R = torch.zeros(1, 1, 2, device=device, dtype=torch.float32)

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

        # Loss 1: 2D reprojection per camera, weighted per
        # (frame, joint, camera) by the outlier mask.  Zero on
        # (frame, joint, camera) means MP's detection there was
        # attributed bad; that contribution drops from the data
        # term while the other camera's good detection (if any)
        # still anchors the joint.  Normalize by the sum of live
        # weights so the magnitude doesn't drift with the
        # outlier count.
        sq_L = ((pL - tgt_L) ** 2).sum(-1) * joint_weight_L
        sq_R = ((pR - tgt_R) ** 2).sum(-1) * joint_weight_R
        live_w = (joint_weight_L.sum() + joint_weight_R.sum()).clamp(min=1.0)
        loss_reproj = (sq_L.sum() + sq_R.sum()) / live_w

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
        if joints_3d.grad is not None:
            torch.nan_to_num_(joints_3d.grad, nan=0.0, posinf=0.0, neginf=0.0)
        torch.nn.utils.clip_grad_norm_([joints_3d], max_norm=5.0)
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
    # Outlier masks in full-trial coordinates.  Multiple
    # resolutions saved for downstream convenience:
    #   joint_outlier_mask[N, 21]      — per (frame, joint)
    #   camera_outlier_mask[N, 21, 2]  — per (frame, joint, camera)
    #   frame_outlier_mask[N]          — any-joint reduction
    #                                    (back-compat with v0 mask)
    all_joint_mask  = np.zeros((N, 21), dtype=bool)
    all_camera_mask = np.zeros((N, 21, 2), dtype=bool)
    all_outlier_mask = np.zeros(N, dtype=bool)

    j3d_np = joints_3d.detach().cpu().numpy()
    for i, t in enumerate(valid_idx):
        all_joints[t] = j3d_np[i]
        all_err_L[t] = err_L[i]
        all_err_R[t] = err_R[i]
        all_joint_mask[t]  = joint_outlier_mask[i]
        all_camera_mask[t] = camera_mask[i]
        all_outlier_mask[t] = bool(joint_outlier_mask[i].any())

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
        # Persist the fit weights inside the npz too so the Labels
        # page can repopulate the slider settings even if the
        # sibling skeleton_v1_params.json is missing (e.g. older
        # fits that pre-date the sidecar, or after a copy that
        # only carried the npz).
        w_reproj=float(w_reproj),
        w_bone=float(w_bone),
        w_smooth=float(w_smooth),
        snap_bones=bool(snap_bones),
        w_angle=float(w_angle),
        # Outlier masks in full-trial coordinates.  All three keys
        # are saved so downstream readers can pick the resolution
        # they need.  See _detect_outlier_per_joint for the
        # signal-composition details.
        frame_outlier_mask=all_outlier_mask,        # (N,) bool
        joint_outlier_mask=all_joint_mask,          # (N, 21) bool
        camera_outlier_mask=all_camera_mask,        # (N, 21, 2) bool
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
            "n_outliers_masked": int(n_joint_outliers),
            "n_frames_with_any_mask": int(n_frames_any),
            "n_cam_L_masked": int(n_cam_L),
            "n_cam_R_masked": int(n_cam_R),
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


def _detect_outlier_per_joint(
    init_3d: np.ndarray,
    mp_L: np.ndarray,
    mp_R: np.ndarray,
    bones,
    K_max: int = 30,
    accel_k: float = 6.0,
    bone_k: float = 6.0,
    k_blame_joint: float = 2.0,
    k_blame_camera: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-joint per-camera outlier detection.

    Returns
    -------
    joint_outlier_mask : (N, 21) bool
        True where the joint at the frame should be treated as
        misdetected (any camera).
    camera_mask : (N, 21, 2) bool
        True for the camera(s) attributed as the source of the bad
        detection.  camera_mask[t, j, 0] = left, [t, j, 1] = right.
        Both can be True simultaneously when the 2D step is
        comparable on both cameras.

    Three composed signals:
      1. Per-joint 3D acceleration-spike pairing — opposite-sign
         spikes within K_max frames, magnitude ratio in (0.5, 2)
         and direction cos ≤ -0.3, on each joint independently.
         Catches the run window.
      2. Bone-length anomaly with per-endpoint blame.  A bone is
         flagged when its length z-score (robust median / MAD)
         exceeds bone_k.  We blame ONE endpoint per anomalous
         bone via the ratio of 3D step magnitudes (k_blame_joint).
         AND'd with signal 1 produces joint_outlier_mask.
      3. Per-joint per-camera attribution via the ratio of 2D
         pixel-step magnitudes (k_blame_camera).  Both cameras
         masked when the steps are comparable.

    NaN-triangulated frames are *not* masked through this path —
    init_3d's wrist-fallback would create a spurious accel spike;
    those frames are filtered out before signal 1 instead.
    """
    N = init_3d.shape[0]
    n_joints = init_3d.shape[1]
    joint_mask = np.zeros((N, n_joints), dtype=bool)
    cam_mask = np.zeros((N, n_joints, 2), dtype=bool)
    if N < 5:
        return joint_mask, cam_mask

    # ── Per-joint 3D acceleration spikes ──────────────────────
    # a[t, j, :] = x[t+1, j] − 2·x[t, j] + x[t-1, j]   for t in 1..N-2
    a = init_3d[2:] - 2 * init_3d[1:-1] + init_3d[:-2]    # (N-2, 21, 3)
    a_norm = np.linalg.norm(a, axis=-1)                    # (N-2, 21)
    # Per-joint robust threshold using the trial's own statistics.
    med_j = np.median(a_norm, axis=0)                      # (21,)
    mad_j = np.median(np.abs(a_norm - med_j[None, :]),
                      axis=0) * 1.4826                     # (21,)
    thr_j = med_j + accel_k * np.maximum(mad_j, 0.1)       # (21,)

    # Pair opposite spikes per joint.  joint_candidate[t, j] True
    # iff frame t lies inside (or at) a matched spike pair on
    # joint j.
    joint_candidate = np.zeros((N, n_joints), dtype=bool)
    for j in range(n_joints):
        spikes = np.where(a_norm[:, j] > thr_j[j])[0]       # indices into a
        if spikes.size < 2:
            continue
        # Vector at each spike for direction check.
        vecs = a[spikes, j]                                # (n_sp, 3)
        norms = np.linalg.norm(vecs, axis=-1)              # (n_sp,)
        used = np.zeros(spikes.size, dtype=bool)
        for i in range(spikes.size):
            if used[i]:
                continue
            t1 = spikes[i] + 1                              # frame idx in init_3d
            v1, m1 = vecs[i], norms[i]
            if m1 < 1e-9:
                continue
            for k in range(i + 1, spikes.size):
                if used[k]:
                    continue
                t2 = spikes[k] + 1
                if t2 - t1 > K_max:
                    break
                v2, m2 = vecs[k], norms[k]
                if m2 < 1e-9:
                    continue
                cos = float(v1 @ v2) / (m1 * m2)
                if cos > -0.3:
                    continue
                ratio = m1 / m2
                if not (0.5 < ratio < 2.0):
                    continue
                joint_candidate[t1:t2 + 1, j] = True
                used[i] = True
                used[k] = True
                break

    # NaN frames in init_3d would have undefined accel — drop any
    # candidacy on those frames (they were filled with wrist
    # fallback, not real data).
    nan_frames = np.isnan(init_3d[:, :, 0])                # (N, 21) bool
    joint_candidate &= ~nan_frames

    # ── Per-bone length anomaly + per-endpoint blame ─────────
    n_bones = len(bones)
    bone_lens = np.zeros((N, n_bones), dtype=np.float64)
    for b, (j1, j2) in enumerate(bones):
        diff = init_3d[:, j2] - init_3d[:, j1]
        bone_lens[:, b] = np.linalg.norm(diff, axis=1)
    med_b = np.median(bone_lens, axis=0)
    mad_b = np.median(np.abs(bone_lens - med_b[None, :]),
                      axis=0) * 1.4826
    mad_b = np.maximum(mad_b, 0.5)  # mm; avoid /0 on near-rigid bones
    z = np.abs(bone_lens - med_b[None, :]) / mad_b[None, :]   # (N, n_bones)
    bone_bad = z > bone_k                                     # (N, n_bones)

    # Per-joint 3D step magnitude  v[t, j] = ‖x[t,j] − x[t-1,j]‖
    # used for blame disambiguation.  Frame 0 has no previous;
    # leave it at 0.
    step_3d = np.zeros((N, n_joints), dtype=np.float64)
    if N > 1:
        step_3d[1:] = np.linalg.norm(init_3d[1:] - init_3d[:-1], axis=-1)

    # Accumulate per-joint blame from each anomalous (t, bone).
    bone_blame = np.zeros((N, n_joints), dtype=np.float32)
    bad_idx_t, bad_idx_b = np.where(bone_bad)
    for t, b in zip(bad_idx_t, bad_idx_b):
        j1, j2 = bones[b]
        v1 = float(step_3d[t, j1])
        v2 = float(step_3d[t, j2])
        if v1 > k_blame_joint * v2:
            bone_blame[t, j1] += 1.0
        elif v2 > k_blame_joint * v1:
            bone_blame[t, j2] += 1.0
        else:
            bone_blame[t, j1] += 0.5
            bone_blame[t, j2] += 0.5

    # ── Combine: joint masked iff BOTH signals fire ──────────
    joint_mask = joint_candidate & (bone_blame > 0)

    # ── Per-joint per-camera attribution via 2D step ─────────
    # Compare ‖mp_L[t,j] − mp_L[t-1,j]‖ vs ‖mp_R[t,j] − mp_R[t-1,j]‖
    # for each masked (t, j).  Both flagged when the magnitudes
    # are within factor k_blame_camera of each other.
    step_2d_L = np.zeros((N, n_joints), dtype=np.float64)
    step_2d_R = np.zeros((N, n_joints), dtype=np.float64)
    if N > 1:
        step_2d_L[1:] = np.linalg.norm(mp_L[1:] - mp_L[:-1], axis=-1)
        step_2d_R[1:] = np.linalg.norm(mp_R[1:] - mp_R[:-1], axis=-1)
    # nan-safe — np.linalg.norm of a NaN vector is NaN; we treat
    # NaN as "no information", default to flagging both cameras.
    tj_idx = np.argwhere(joint_mask)
    for t, j in tj_idx:
        dL = step_2d_L[t, j]
        dR = step_2d_R[t, j]
        if np.isnan(dL) and np.isnan(dR):
            cam_mask[t, j, 0] = True
            cam_mask[t, j, 1] = True
        elif np.isnan(dL):
            cam_mask[t, j, 1] = True
        elif np.isnan(dR):
            cam_mask[t, j, 0] = True
        elif dL > k_blame_camera * dR:
            cam_mask[t, j, 0] = True
        elif dR > k_blame_camera * dL:
            cam_mask[t, j, 1] = True
        else:
            cam_mask[t, j, 0] = True
            cam_mask[t, j, 1] = True

    return joint_mask, cam_mask


def _detect_outlier_runs(
    init_3d: np.ndarray,
    bones,
    K_max: int = 30,
    accel_k: float = 6.0,
    bone_k: float = 6.0,
) -> np.ndarray:
    """Identify a per-frame mask of suspected outlier frames.

    Two independent signals AND'd together:

    1. **Acceleration-spike pairing** — a run of K consecutive
       mislabeled frames produces opposite-sign acceleration
       spikes at frames t1 and t2 = t1+K, with matching magnitudes
       and roughly opposite vector directions.  We scan all
       frames whose per-frame |a| exceeds median + accel_k·MAD,
       then pair each positive-direction spike to the next
       negative-direction spike within K_max frames whose
       magnitude ratio is in (0.5, 2.0) and whose vector cosine
       is ≤ -0.3.  Frames between matched spikes go in the
       candidate set.

    2. **Bone-length anomaly** — for each bone, compute a
       robust z-score (median / MAD) of its per-frame length
       across the trial.  Frame-level score is the max |z| over
       all bones.  Frames with score > bone_k are flagged.

    Both signals are needed for a frame to be masked, so genuine
    fast motion (high acceleration but normal bone lengths) and
    gradual MP drift on a single joint (anomalous bone length
    but smooth velocity) don't mask each other.

    Returns: bool array of shape (N,) where True means MASKED.
    """
    N = init_3d.shape[0]
    if N < 5:
        return np.zeros(N, dtype=bool)

    # ── Acceleration-spike pairing ────────────────────────────
    # a[t] = x[t+1] − 2·x[t] + x[t-1] for t in 1..N-2
    a = init_3d[2:] - 2 * init_3d[1:-1] + init_3d[:-2]   # (N-2, 21, 3)
    a_norm = np.linalg.norm(a, axis=-1)                  # (N-2, 21)
    a_frame_mag = a_norm.max(axis=-1)                    # (N-2,)
    med = float(np.median(a_frame_mag))
    mad = float(np.median(np.abs(a_frame_mag - med))) * 1.4826
    threshold = med + accel_k * max(mad, 0.1)
    spike_local = np.where(a_frame_mag > threshold)[0]   # indices into a[]
    spike_frames = spike_local + 1                       # convert to full-frame idx
    # Flatten per-spike vector and magnitude for direction matching.
    spike_vecs = a[spike_local].reshape(len(spike_local), -1)
    spike_norms = np.linalg.norm(spike_vecs, axis=-1)

    run_candidate = np.zeros(N, dtype=bool)
    used = np.zeros(len(spike_local), dtype=bool)
    for i in range(len(spike_local)):
        if used[i]:
            continue
        t1 = spike_frames[i]
        v1 = spike_vecs[i]
        m1 = spike_norms[i]
        if m1 < 1e-9:
            continue
        for j in range(i + 1, len(spike_local)):
            if used[j]:
                continue
            t2 = spike_frames[j]
            if t2 - t1 > K_max:
                break
            v2 = spike_vecs[j]
            m2 = spike_norms[j]
            if m2 < 1e-9:
                continue
            cos = float(v1 @ v2) / (m1 * m2)
            if cos > -0.3:
                continue
            ratio = m1 / m2
            if not (0.5 < ratio < 2.0):
                continue
            # Mark the interior frames; pair both spikes too.
            run_candidate[t1:t2 + 1] = True
            used[i] = True
            used[j] = True
            break

    # ── Bone-length anomaly ──────────────────────────────────
    n_bones = len(bones)
    bone_lens = np.zeros((N, n_bones), dtype=np.float64)
    for b, (j1, j2) in enumerate(bones):
        diff = init_3d[:, j2] - init_3d[:, j1]
        bone_lens[:, b] = np.linalg.norm(diff, axis=1)
    med_b = np.median(bone_lens, axis=0)
    mad_b = np.median(np.abs(bone_lens - med_b[None, :]), axis=0) * 1.4826
    mad_b = np.maximum(mad_b, 0.5)            # mm; avoid /0 on rigid bones
    z = np.abs(bone_lens - med_b[None, :]) / mad_b[None, :]
    bone_score = z.max(axis=1)                # (N,)
    bone_anomaly = bone_score > bone_k

    return run_candidate & bone_anomaly


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
