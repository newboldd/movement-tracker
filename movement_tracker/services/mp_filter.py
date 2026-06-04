"""Multi-signal MediaPipe outlier filter.

A library of independent per-(frame, joint, camera) flagging signals
that get OR'd together to produce the final outlier mask.  Each
signal exposes its own thresholds; the caller can enable any subset.

Signals:
  - velocity:    per-joint 3D step  ‖x[t,j] − x[t-1,j]‖  z-score
  - acceleration: per-joint 3D       ‖x[t+1,j] − 2x[t,j] + x[t-1,j]‖  z-score
  - bone:        per-bone length    |z| > k_bone  with per-endpoint blame
  - ydisp:       per-joint per-cam  |v_R_undist − v_L_undist|  pixels
                 (epipolar Y residual when cameras are nominally aligned)
  - z_outlier:   per-joint 3D Z (depth) z-score
  - mpconf:      per-frame per-camera MP confidence below threshold
                 → flags every joint on that camera, that frame
  - stereo:      per-(frame, joint) stereo reprojection error (px)
                 → flags whichever camera has the larger residual
  - hrnet:       per-(frame, joint, camera) HRnet peak score below threshold

Each signal returns a (N, 21, 2) bool tensor.  When disabled the
signal is skipped and contributes nothing.  Camera attribution
for 3D signals uses the 2D step-magnitude ratio (k_blame_camera
default 2.0), same convention as the previous v1 detector.
"""
from __future__ import annotations

import numpy as np
import cv2


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def _robust_z(values: np.ndarray, axis: int = 0):
    """Robust z = |value - median| / (MAD · 1.4826), per-axis."""
    med = np.median(values, axis=axis, keepdims=True)
    mad = np.median(np.abs(values - med), axis=axis, keepdims=True) * 1.4826
    mad = np.maximum(mad, np.finfo(np.float64).eps)
    return np.abs(values - med) / mad, med, mad


def _leave_one_out_poly_kernel(
    window: int = 5, degree: int = 2,
) -> np.ndarray:
    """Build a (2*window+1,)-length kernel ``c`` such that

        prediction_at_t = sum_k c[k] * arr[t - window + k]

    is the value at offset 0 of the polynomial of degree ``degree``
    least-squares-fit through ``arr`` at offsets
    ``[-window … -1, 1 … window]``.  The centre entry ``c[window]``
    is 0 by construction (the value at t is EXCLUDED from the fit).

    Cached at module level since (window, degree) is fixed.
    """
    offsets = np.concatenate([np.arange(-window, 0),
                              np.arange(1, window + 1)]).astype(np.float64)
    A = np.vander(offsets, degree + 1, increasing=True)   # (2w, d+1)
    # const-term row of the least-squares solver, equivalent to
    # picking out the polynomial's value at offset 0.
    coef = np.linalg.pinv(A.T @ A) @ A.T                  # (d+1, 2w)
    const = coef[0, :]                                    # (2w,)
    kernel = np.zeros(2 * window + 1, dtype=np.float64)
    kernel[:window] = const[:window]
    kernel[window + 1:] = const[window:]
    return kernel


_POLY_KERNEL_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _residual_vs_polyfit(
    arr: np.ndarray, window: int = 5, degree: int = 2,
) -> np.ndarray:
    """Per-(t, j) residual of ``arr[t, j]`` against a polynomial fit
    through its neighbours.

    For each (t, j) we fit a degree-``degree`` polynomial to the
    points ``arr[t-window:t+window+1, j]`` EXCLUDING t itself and
    report the distance from ``arr[t, j]`` to the polynomial's
    prediction at frame t.

    Rationale: raw frame-to-frame step magnitude treats fast natural
    motion the same as a sudden label glitch — both have big step.
    A polynomial residual separates them: smooth real motion matches
    the local fit so the residual stays small, while an outlier
    label deviates sharply from the surrounding trajectory.

    Vectorised: a single leave-one-out kernel built once from a
    ``(2*window+1)``-length design matrix is applied to a sliding
    window over ``arr`` via matrix multiplication.  Boundary
    frames within ``window`` of either edge get NaN residual
    (no full window available).  Cells where any neighbour in the
    window is NaN also produce NaN residual (the matmul
    propagates) — the consumer treats those as "can't blame this
    side", same as for boundary cells.

    Returns ``(N, J)`` float64.
    """
    N, J = arr.shape[0], arr.shape[1]
    out = np.full((N, J), np.nan, dtype=np.float64)
    W = 2 * window + 1
    if N < W:
        return out
    key = (window, degree)
    kernel = _POLY_KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = _leave_one_out_poly_kernel(window, degree)
        _POLY_KERNEL_CACHE[key] = kernel
    # sliding_window_view: (N - W + 1, J, W) for each dim.
    swv_x = np.lib.stride_tricks.sliding_window_view(arr[:, :, 0], W, axis=0)
    swv_y = np.lib.stride_tricks.sliding_window_view(arr[:, :, 1], W, axis=0)
    pred_x_int = swv_x @ kernel                           # (N - W + 1, J)
    pred_y_int = swv_y @ kernel
    pred_x = np.full((N, J), np.nan)
    pred_y = np.full((N, J), np.nan)
    pred_x[window:window + pred_x_int.shape[0]] = pred_x_int
    pred_y[window:window + pred_y_int.shape[0]] = pred_y_int
    dx = arr[..., 0] - pred_x
    dy = arr[..., 1] - pred_y
    out = np.sqrt(dx * dx + dy * dy)
    return out


def _camera_blame_from_2d_step(
    mp_L: np.ndarray, mp_R: np.ndarray,
    joint_mask: np.ndarray,
    k_blame: float = 2.0,
) -> np.ndarray:
    """Distribute a per-(frame, joint) flag onto the L/R cameras
    based on the polynomial-fit residual at frame t per camera.

    The residual is computed by :func:`_residual_vs_polyfit` over
    a centred 11-frame window (5 before + 5 after, excluding t).
    A camera whose label at t deviates sharply from its own local
    smooth trajectory gets the larger residual and the blame —
    real fast motion that's continuous on both sides of t has a
    small residual since the polynomial captures the trajectory.

    Decision ladder:
      * one side's residual is NaN  → blame the other side.
      * both NaN                    → blame both.
      * dL > k_blame × dR           → blame L only.
      * dR > k_blame × dL           → blame R only.
      * otherwise (comparable)      → blame the LARGER residual.
        Exact ties (rare with float residuals; typically dL ==
        dR == 0)                    → blame both.
    """
    N, J = mp_L.shape[0], mp_L.shape[1]
    out = np.zeros((N, J, 2), dtype=bool)
    if N < 2:
        return out
    step_L = _residual_vs_polyfit(mp_L)
    step_R = _residual_vs_polyfit(mp_R)
    tj = np.argwhere(joint_mask)
    for t, j in tj:
        dL = step_L[t, j]
        dR = step_R[t, j]
        nanL = np.isnan(dL)
        nanR = np.isnan(dR)
        if nanL and nanR:
            out[t, j, 0] = True; out[t, j, 1] = True
        elif nanL:
            out[t, j, 1] = True
        elif nanR:
            out[t, j, 0] = True
        elif dL > k_blame * dR:
            out[t, j, 0] = True
        elif dR > k_blame * dL:
            out[t, j, 1] = True
        elif dL > dR:
            # Comparable, but L still moved more — blame L alone.
            out[t, j, 0] = True
        elif dR > dL:
            out[t, j, 1] = True
        else:
            # Exact equality (almost always dL == dR == 0).
            out[t, j, 0] = True; out[t, j, 1] = True
    return out


def _project_with_distortion(pts3d, K, dist, R=None, T=None):
    """OpenCV projectPoints for an (M, 3) point set → (M, 2)."""
    if R is None:
        rvec = np.zeros(3)
        tvec = np.zeros(3)
    else:
        rvec = cv2.Rodrigues(R)[0]
        tvec = T.reshape(3, 1)
    pts = np.asarray(pts3d, dtype=np.float64).reshape(-1, 1, 3)
    out, _ = cv2.projectPoints(pts, rvec, tvec, K, dist)
    return out.reshape(-1, 2)


# ──────────────────────────────────────────────────────────────────
# Individual signals  — each returns (N, 21, 2) bool
# ──────────────────────────────────────────────────────────────────

def _signal_velocity(init_3d, mp_L, mp_R, vel_k: float, k_blame_camera: float):
    """Per-joint 3D velocity z-score > vel_k → flag (any camera per blame)."""
    N, J = init_3d.shape[0], init_3d.shape[1]
    if N < 2:
        return np.zeros((N, J, 2), dtype=bool)
    v = np.linalg.norm(init_3d[1:] - init_3d[:-1], axis=-1)   # (N-1, J)
    z, _, _ = _robust_z(v, axis=0)
    flag = np.zeros((N, J), dtype=bool)
    flag[1:] = z > vel_k
    return _camera_blame_from_2d_step(mp_L, mp_R, flag, k_blame_camera)


def _signal_acceleration(init_3d, mp_L, mp_R, accel_k: float, k_blame_camera: float):
    """Per-joint 3D acceleration z-score > accel_k → flag."""
    N, J = init_3d.shape[0], init_3d.shape[1]
    if N < 3:
        return np.zeros((N, J, 2), dtype=bool)
    a = init_3d[2:] - 2 * init_3d[1:-1] + init_3d[:-2]       # (N-2, J, 3)
    am = np.linalg.norm(a, axis=-1)                          # (N-2, J)
    z, _, _ = _robust_z(am, axis=0)
    flag = np.zeros((N, J), dtype=bool)
    flag[1:-1] = z > accel_k
    return _camera_blame_from_2d_step(mp_L, mp_R, flag, k_blame_camera)


def _signal_bone(init_3d, mp_L, mp_R, bones, bone_k: float,
                 k_blame_joint: float, k_blame_camera: float):
    """Per-bone z-score → per-endpoint blame via 3D step → cam attr."""
    N, J = init_3d.shape[0], init_3d.shape[1]
    n_b = len(bones)
    bone_lens = np.zeros((N, n_b), dtype=np.float64)
    for b, (j1, j2) in enumerate(bones):
        bone_lens[:, b] = np.linalg.norm(init_3d[:, j2] - init_3d[:, j1], axis=1)
    z, _, _ = _robust_z(bone_lens, axis=0)
    bone_bad = z > bone_k
    # Per-(frame, joint) 3D step for blame disambiguation.
    step_3d = np.zeros((N, J), dtype=np.float64)
    if N > 1:
        step_3d[1:] = np.linalg.norm(init_3d[1:] - init_3d[:-1], axis=-1)
    joint_flag = np.zeros((N, J), dtype=bool)
    bad_t, bad_b = np.where(bone_bad)
    for t, b in zip(bad_t, bad_b):
        j1, j2 = bones[b]
        v1 = float(step_3d[t, j1]); v2 = float(step_3d[t, j2])
        if v1 > k_blame_joint * v2:
            joint_flag[t, j1] = True
        elif v2 > k_blame_joint * v1:
            joint_flag[t, j2] = True
        else:
            joint_flag[t, j1] = True
            joint_flag[t, j2] = True
    return _camera_blame_from_2d_step(mp_L, mp_R, joint_flag, k_blame_camera)


def _signal_ydisp(mp_L, mp_R, calib, ydisp_px: float):
    """Y-disparity per joint, relative to the trial baseline.

    For each joint, compute the signed undistorted Y disparity
    dy[t, j] = y_R_und[t, j] − y_L_und[t, j], then take its
    per-joint median across the trial as the "expected" baseline
    (this absorbs the systematic Y offset from non-rectified
    calibration — see the /stereo_disparity diagnostic).  Flag a
    frame when |dy[t, j] − median_j| > ydisp_px.

    Flags BOTH cameras when triggered (the disparity can't tell us
    which side moved).  No calib → no-op.
    """
    N, J = mp_L.shape[0], mp_L.shape[1]
    out = np.zeros((N, J, 2), dtype=bool)
    if calib is None or N == 0:
        return out
    K1 = calib["K1"]; d1 = calib["dist1"]
    K2 = calib["K2"]; d2 = calib["dist2"]
    L = mp_L.reshape(-1, 1, 2).astype(np.float64)
    R = mp_R.reshape(-1, 1, 2).astype(np.float64)
    L_und = cv2.undistortPoints(L, K1, d1, P=K1).reshape(N, J, 2)
    R_und = cv2.undistortPoints(R, K2, d2, P=K2).reshape(N, J, 2)
    dy = R_und[..., 1] - L_und[..., 1]                       # (N, J), signed
    # Per-joint trial baseline.  np.nanmedian so missing frames
    # (NaN MP detections) don't contaminate it.
    baseline = np.nanmedian(dy, axis=0, keepdims=True)        # (1, J)
    dy_dev = np.abs(dy - baseline)
    flag = np.where(np.isfinite(dy_dev), dy_dev > ydisp_px, False)
    out[..., 0] = flag
    out[..., 1] = flag
    return out


def _signal_z_outlier(init_3d, mp_L, mp_R, z_k: float, k_blame_camera: float):
    """Per-joint Z (depth) z-score > z_k → flag, with cam attribution."""
    N, J = init_3d.shape[0], init_3d.shape[1]
    z = init_3d[..., 2]                                     # (N, J)
    z_score, _, _ = _robust_z(z, axis=0)
    flag = np.where(np.isfinite(z_score), z_score > z_k, False)
    return _camera_blame_from_2d_step(mp_L, mp_R, flag, k_blame_camera)


def _signal_mpconf(confidence_L, confidence_R, N, J, mpconf_min: float):
    """Per-frame per-camera MP confidence below threshold.

    MP confidence is HAND-LEVEL (one scalar per frame per camera),
    so we flag ALL joints on the offending camera at that frame.
    """
    out = np.zeros((N, J, 2), dtype=bool)
    if confidence_L is not None:
        bad_L = (np.asarray(confidence_L) < mpconf_min) & np.isfinite(confidence_L)
        # bad_L[:N] in case of length mismatch.
        m = min(N, len(bad_L))
        out[:m, :, 0] = bad_L[:m][:, None]
    if confidence_R is not None:
        bad_R = (np.asarray(confidence_R) < mpconf_min) & np.isfinite(confidence_R)
        m = min(N, len(bad_R))
        out[:m, :, 1] = bad_R[:m][:, None]
    return out


def _signal_stereo_reproj(init_3d, mp_L, mp_R, calib, stereo_px: float):
    """Per-(frame, joint, camera) stereo reprojection residual,
    thresholded against the per-joint per-camera trial baseline.

    A non-rectified stereo pair has a calibration-dependent
    irreducible reprojection error per joint per camera (the L2-
    optimal triangulation residual).  Treating "0 = perfect" would
    punish trials whose calibration happens to have a ~0.3 px
    natural offset.  Subtracting the per-(joint, camera) trial
    median (np.nanmedian) leaves only the per-frame DEVIATION
    above the baseline as the anomaly signal.

    Flag camera L when ``res_L - median_L > stereo_px`` (and
    similarly for R).  No calib → no-op.
    """
    N, J = init_3d.shape[0], init_3d.shape[1]
    out = np.zeros((N, J, 2), dtype=bool)
    if calib is None or N == 0:
        return out
    K1 = calib["K1"]; d1 = calib["dist1"]
    K2 = calib["K2"]; d2 = calib["dist2"]
    R = np.asarray(calib["R"])
    T = np.asarray(calib["T"]).reshape(3, 1)
    rvecR, _ = cv2.Rodrigues(R)
    pts = init_3d.reshape(-1, 1, 3).astype(np.float64)
    valid = np.isfinite(pts[..., 0]).ravel()
    if not valid.any():
        return out
    pL_2d, _ = cv2.projectPoints(pts[valid], np.zeros(3), np.zeros(3), K1, d1)
    pR_2d, _ = cv2.projectPoints(pts[valid], rvecR, T, K2, d2)
    proj_L = np.full((N * J, 2), np.nan)
    proj_R = np.full((N * J, 2), np.nan)
    proj_L[valid] = pL_2d.reshape(-1, 2)
    proj_R[valid] = pR_2d.reshape(-1, 2)
    proj_L = proj_L.reshape(N, J, 2)
    proj_R = proj_R.reshape(N, J, 2)
    res_L = np.linalg.norm(proj_L - mp_L, axis=-1)         # (N, J)
    res_R = np.linalg.norm(proj_R - mp_R, axis=-1)
    # Per-joint per-camera baseline (irreducible cal/MP residual).
    base_L = np.nanmedian(res_L, axis=0, keepdims=True)    # (1, J)
    base_R = np.nanmedian(res_R, axis=0, keepdims=True)
    # nan-safe fallback when an entire joint column is NaN.
    base_L = np.where(np.isfinite(base_L), base_L, 0.0)
    base_R = np.where(np.isfinite(base_R), base_R, 0.0)
    dev_L = res_L - base_L
    dev_R = res_R - base_R
    out[..., 0] = np.where(np.isfinite(dev_L), dev_L > stereo_px, False)
    out[..., 1] = np.where(np.isfinite(dev_R), dev_R > stereo_px, False)
    return out


def _signal_stereo_outline(stereo_shift_mag, stereo_response,
                           mp_L, mp_R, N, J,
                           stereo_outline_px: float,
                           stereo_outline_conf_min: float,
                           k_blame_camera: float):
    """Per-(frame, joint) outline-stereo correction distance, gated
    by phase-correlation confidence.

    Three-step decision:

    1. If the per-joint phase-corr response is below
       ``stereo_outline_conf_min``, skip — the shift baked into
       stereo_align_outline.npz is unreliable, so we have no basis
       to call its distance "wrong" either.
    2. If the shift magnitude is at-or-under
       ``stereo_outline_px``, the per-joint stereo distance is in
       the normal range — no flag.
    3. Otherwise (confident shift, large distance) attribute the
       error to one camera via the 2D-jump rule shared with every
       other MP-Filter signal.

    Returns the standard per-(frame, joint, camera) bool mask.
    NaN-safe; missing response treated as "fails the gate" so we
    don't false-positive on frames where stereo wasn't run.
    """
    out = np.zeros((N, J, 2), dtype=bool)
    if stereo_shift_mag is None:
        return out
    S = np.asarray(stereo_shift_mag, dtype=np.float64)
    if S.ndim != 2 or S.shape[1] != J:
        return out
    m = min(N, S.shape[0])
    big_shift = np.zeros((N, J), dtype=bool)
    big_shift[:m] = np.where(np.isfinite(S[:m]),
                               S[:m] > stereo_outline_px, False)
    conf_ok = np.zeros((N, J), dtype=bool)
    if stereo_response is not None:
        R = np.asarray(stereo_response, dtype=np.float64)
        if R.ndim == 2 and R.shape[1] == J:
            mr = min(N, R.shape[0])
            conf_ok[:mr] = np.where(np.isfinite(R[:mr]),
                                      R[:mr] >= stereo_outline_conf_min,
                                      False)
    else:
        # No response array → can't gate; trust the distance check.
        conf_ok[:] = True
    flag = big_shift & conf_ok
    return _camera_blame_from_2d_step(mp_L, mp_R, flag, k_blame_camera)


def _signal_hrnet(hrnet_L, hrnet_R, N, J, hrnet_min: float):
    """Per-(frame, joint, camera) HRnet score below threshold."""
    out = np.zeros((N, J, 2), dtype=bool)
    if hrnet_L is not None:
        H = np.asarray(hrnet_L)
        if H.ndim == 2 and H.shape[1] == J:
            m = min(N, H.shape[0])
            out[:m, :, 0] = (H[:m] < hrnet_min) & np.isfinite(H[:m])
    if hrnet_R is not None:
        H = np.asarray(hrnet_R)
        if H.ndim == 2 and H.shape[1] == J:
            m = min(N, H.shape[0])
            out[:m, :, 1] = (H[:m] < hrnet_min) & np.isfinite(H[:m])
    return out


# ──────────────────────────────────────────────────────────────────
# Precomputed signal arrays for the client-side composer
# ──────────────────────────────────────────────────────────────────

def build_signal_data(
    init_3d: np.ndarray,
    mp_L: np.ndarray,
    mp_R: np.ndarray,
    bones,
    calib=None,
    confidence_L=None,
    confidence_R=None,
    hrnet_L=None,
    hrnet_R=None,
    stereo_shift_mag=None,
    stereo_response=None,
) -> dict:
    """Compute every signal's RAW per-(frame, joint[, camera]) array
    once for a trial.  Returns float32 numpy arrays the caller can
    base64-encode for the wire.

    The returned arrays are in the SAME coordinate system (no
    per-signal valid-frame slicing) so the client can threshold + OR
    them with simple element-wise operations and no extra book-keeping.

    Signals returning a single number per (frame, joint) come in the
    same shape; 2-camera signals come as separate _L / _R arrays.
    """
    N, J = init_3d.shape[0], init_3d.shape[1]
    f32 = np.float32

    # ── Velocity z-score (3D step) ────────────────────────────
    vel_mag = np.zeros((N, J), dtype=np.float64)
    if N > 1:
        vel_mag[1:] = np.linalg.norm(init_3d[1:] - init_3d[:-1], axis=-1)
    vel_z, _, _ = _robust_z(vel_mag, axis=0)

    # ── Acceleration z-score (3D second diff) ─────────────────
    accel_mag = np.zeros((N, J), dtype=np.float64)
    if N > 2:
        a = init_3d[2:] - 2 * init_3d[1:-1] + init_3d[:-2]
        accel_mag[1:-1] = np.linalg.norm(a, axis=-1)
    accel_z, _, _ = _robust_z(accel_mag, axis=0)

    # ── Bone-length z-score ───────────────────────────────────
    n_b = len(bones)
    bone_lens = np.zeros((N, n_b), dtype=np.float64)
    for b, (j1, j2) in enumerate(bones):
        bone_lens[:, b] = np.linalg.norm(init_3d[:, j2] - init_3d[:, j1], axis=1)
    bone_z, _, _ = _robust_z(bone_lens, axis=0)

    # ── Per-joint Z (depth) z-score ───────────────────────────
    z_dev, _, _ = _robust_z(init_3d[..., 2], axis=0)

    # ── Y-disparity Δ (px) — deviation from per-joint baseline ─
    ydisp_dev = np.full((N, J), np.nan, dtype=np.float64)
    if calib is not None:
        K1 = calib["K1"]; d1 = calib["dist1"]
        K2 = calib["K2"]; d2 = calib["dist2"]
        L = mp_L.reshape(-1, 1, 2).astype(np.float64)
        R = mp_R.reshape(-1, 1, 2).astype(np.float64)
        L_und = cv2.undistortPoints(L, K1, d1, P=K1).reshape(N, J, 2)
        R_und = cv2.undistortPoints(R, K2, d2, P=K2).reshape(N, J, 2)
        dy = R_und[..., 1] - L_und[..., 1]
        base = np.nanmedian(dy, axis=0, keepdims=True)
        ydisp_dev = np.abs(dy - base)

    # ── Stereo reproj Δ (px) — per-camera, baseline-relative ───
    stereo_dev_L = np.full((N, J), np.nan, dtype=np.float64)
    stereo_dev_R = np.full((N, J), np.nan, dtype=np.float64)
    if calib is not None:
        K1 = calib["K1"]; d1 = calib["dist1"]
        K2 = calib["K2"]; d2 = calib["dist2"]
        R = np.asarray(calib["R"]); T = np.asarray(calib["T"]).reshape(3, 1)
        rvecR, _ = cv2.Rodrigues(R)
        pts = init_3d.reshape(-1, 1, 3).astype(np.float64)
        valid = np.isfinite(pts[..., 0]).ravel()
        if valid.any():
            pL_2d, _ = cv2.projectPoints(pts[valid], np.zeros(3), np.zeros(3), K1, d1)
            pR_2d, _ = cv2.projectPoints(pts[valid], rvecR, T, K2, d2)
            proj_L = np.full((N * J, 2), np.nan)
            proj_R = np.full((N * J, 2), np.nan)
            proj_L[valid] = pL_2d.reshape(-1, 2)
            proj_R[valid] = pR_2d.reshape(-1, 2)
            res_L = np.linalg.norm(proj_L.reshape(N, J, 2) - mp_L, axis=-1)
            res_R = np.linalg.norm(proj_R.reshape(N, J, 2) - mp_R, axis=-1)
            base_L = np.nanmedian(res_L, axis=0, keepdims=True)
            base_R = np.nanmedian(res_R, axis=0, keepdims=True)
            base_L = np.where(np.isfinite(base_L), base_L, 0.0)
            base_R = np.where(np.isfinite(base_R), base_R, 0.0)
            stereo_dev_L = res_L - base_L
            stereo_dev_R = res_R - base_R

    # ── Hybrid-stereo correction distance (px) ────────────────
    # Per-(frame, joint) 2D shift magnitude from the trial's
    # stereo_align_outline.npz, NaN-padded when missing.
    stereo_outline_mag = np.full((N, J), np.nan, dtype=np.float64)
    if stereo_shift_mag is not None:
        S = np.asarray(stereo_shift_mag, dtype=np.float64)
        if S.ndim == 2 and S.shape[1] == J:
            m = min(N, S.shape[0])
            stereo_outline_mag[:m] = S[:m]
    # ── Hybrid-stereo phase-corr response (≈[0,1]) ────────────
    # Per-(frame, joint) confidence baked alongside the shift in
    # the outline npz.  NaN where missing.
    stereo_outline_resp = np.full((N, J), np.nan, dtype=np.float64)
    if stereo_response is not None:
        R = np.asarray(stereo_response, dtype=np.float64)
        if R.ndim == 2 and R.shape[1] == J:
            m = min(N, R.shape[0])
            stereo_outline_resp[:m] = R[:m]

    # ── Camera-attribution helpers (2D residuals + 3D step) ──
    # ``step_2d_L`` / ``step_2d_R`` are NOT raw frame-to-frame
    # step magnitudes anymore — they're per-camera polynomial-fit
    # residuals over a centred 11-frame window (see
    # ``_residual_vs_polyfit``).  Real fast motion that's smooth
    # on both sides of t now has a small residual; a sudden label
    # jump has a large one, so blame attribution doesn't trip on
    # high-velocity but legitimate motion.  Keeping the legacy
    # name avoids churning the JS composer's attrib3D helper.
    step_2d_L = _residual_vs_polyfit(mp_L)
    step_2d_R = _residual_vs_polyfit(mp_R)
    step_3d = np.zeros((N, J), dtype=np.float64)
    if N > 1:
        step_3d[1:] = np.linalg.norm(init_3d[1:] - init_3d[:-1], axis=-1)

    # ── Per-camera scalars (MP conf / HRnet scores) ───────────
    mpconf_L_arr = np.full(N, np.nan, dtype=np.float64)
    mpconf_R_arr = np.full(N, np.nan, dtype=np.float64)
    if confidence_L is not None:
        m = min(N, len(confidence_L))
        mpconf_L_arr[:m] = np.asarray(confidence_L[:m], dtype=np.float64)
    if confidence_R is not None:
        m = min(N, len(confidence_R))
        mpconf_R_arr[:m] = np.asarray(confidence_R[:m], dtype=np.float64)
    hrnet_L_arr = np.full((N, J), np.nan, dtype=np.float64)
    hrnet_R_arr = np.full((N, J), np.nan, dtype=np.float64)
    if hrnet_L is not None:
        H = np.asarray(hrnet_L)
        if H.ndim == 2 and H.shape[1] == J:
            m = min(N, H.shape[0]); hrnet_L_arr[:m] = H[:m]
    if hrnet_R is not None:
        H = np.asarray(hrnet_R)
        if H.ndim == 2 and H.shape[1] == J:
            m = min(N, H.shape[0]); hrnet_R_arr[:m] = H[:m]

    # NaNs in float32 are preserved across the wire and round-trip
    # cleanly to JS Float32Array.  The client treats NaN as
    # "no information" (no flag).
    return {
        "vel_z":          vel_z.astype(f32),
        "accel_z":        accel_z.astype(f32),
        "bone_z":         bone_z.astype(f32),
        "z_dev":          z_dev.astype(f32),
        "ydisp_dev":      ydisp_dev.astype(f32),
        "stereo_dev_L":   stereo_dev_L.astype(f32),
        "stereo_dev_R":   stereo_dev_R.astype(f32),
        "stereo_outline_mag": stereo_outline_mag.astype(f32),
        "stereo_outline_resp": stereo_outline_resp.astype(f32),
        "step_2d_L":      step_2d_L.astype(f32),
        "step_2d_R":      step_2d_R.astype(f32),
        "step_3d":        step_3d.astype(f32),
        "mpconf_L":       mpconf_L_arr.astype(f32),
        "mpconf_R":       mpconf_R_arr.astype(f32),
        "hrnet_L":        hrnet_L_arr.astype(f32),
        "hrnet_R":        hrnet_R_arr.astype(f32),
        "bones":          [[int(a), int(b)] for a, b in bones],
        "N":              int(N),
        "J":              int(J),
    }


# ──────────────────────────────────────────────────────────────────
# Composer (server-side, used by Skel Fit v1's internal detector)
# ──────────────────────────────────────────────────────────────────

def detect_mask(
    init_3d: np.ndarray,
    mp_L: np.ndarray,
    mp_R: np.ndarray,
    bones,
    *,
    calib=None,
    confidence_L=None,
    confidence_R=None,
    hrnet_L=None,
    hrnet_R=None,
    # Velocity
    enable_vel: bool = False,
    vel_k: float = 6.0,
    # Acceleration
    enable_accel: bool = True,
    accel_k: float = 6.0,
    # Bone length
    enable_bone: bool = True,
    bone_k: float = 6.0,
    # Y disparity (px)
    enable_ydisp: bool = False,
    ydisp_px: float = 5.0,
    # Z outlier
    enable_z: bool = False,
    z_k: float = 6.0,
    # MP confidence (≥)
    enable_mpconf: bool = False,
    mpconf_min: float = 0.5,
    # Stereo reproj (px)
    enable_stereo: bool = False,
    stereo_px: float = 5.0,
    # Stereo error: dist threshold (px) gated by min conf (≥).
    enable_stereo_outline: bool = False,
    stereo_outline_px: float = 10.0,
    stereo_outline_conf_min: float = 0.2,
    stereo_shift_mag=None,
    stereo_response=None,
    # HRnet (≥)
    enable_hrnet: bool = False,
    hrnet_min: float = 0.2,
    # Blame disambiguation
    k_blame_joint: float = 2.0,
    k_blame_camera: float = 2.0,
):
    """Run all enabled signals and OR their (N, 21, 2) masks.

    Returns
    -------
    joint_mask : (N, 21) bool        — any-camera reduction
    camera_mask : (N, 21, 2) bool    — per (frame, joint, camera)
    per_signal : dict[str, int]      — per-signal flag count for the UI
    """
    N, J = init_3d.shape[0], init_3d.shape[1]
    cam = np.zeros((N, J, 2), dtype=bool)
    per_signal: dict[str, int] = {}

    def _add(name, m):
        nonlocal cam
        per_signal[name] = int(m.sum())
        cam |= m

    if enable_vel and N >= 2:
        _add("velocity",     _signal_velocity(init_3d, mp_L, mp_R, vel_k, k_blame_camera))
    if enable_accel and N >= 3:
        _add("acceleration", _signal_acceleration(init_3d, mp_L, mp_R, accel_k, k_blame_camera))
    if enable_bone:
        _add("bone",         _signal_bone(init_3d, mp_L, mp_R, bones, bone_k,
                                          k_blame_joint, k_blame_camera))
    if enable_ydisp and calib is not None:
        _add("ydisp",        _signal_ydisp(mp_L, mp_R, calib, ydisp_px))
    if enable_z:
        _add("z_outlier",    _signal_z_outlier(init_3d, mp_L, mp_R, z_k, k_blame_camera))
    if enable_mpconf and (confidence_L is not None or confidence_R is not None):
        _add("mp_confidence", _signal_mpconf(confidence_L, confidence_R, N, J, mpconf_min))
    if enable_stereo and calib is not None:
        _add("stereo_reproj", _signal_stereo_reproj(init_3d, mp_L, mp_R, calib, stereo_px))
    if enable_stereo_outline and stereo_shift_mag is not None:
        _add("stereo_outline",
             _signal_stereo_outline(stereo_shift_mag, stereo_response,
                                     mp_L, mp_R, N, J,
                                     stereo_outline_px,
                                     stereo_outline_conf_min,
                                     k_blame_camera))
    if enable_hrnet and (hrnet_L is not None or hrnet_R is not None):
        _add("hrnet",        _signal_hrnet(hrnet_L, hrnet_R, N, J, hrnet_min))

    joint = cam.any(axis=-1)
    return joint, cam, per_signal


# ──────────────────────────────────────────────────────────────────
# Sidecar helpers — both skeleton_data.py (Stereo Fill emission) and
# skeleton_v1.py (saved-filter pre-pass) read the same JSON sidecar
# and run detect_mask with the same defaults, so we share one call.
# ──────────────────────────────────────────────────────────────────

def load_saved_filter_params(sidecar_path) -> dict | None:
    """Read a per-trial ``mp_filter_params.json`` sidecar.

    Returns the decoded dict on success, ``None`` when the file
    is missing or unparseable.
    """
    from pathlib import Path
    p = Path(sidecar_path)
    if not p.exists():
        return None
    try:
        import json
        return json.loads(p.read_text())
    except Exception:
        return None


def detect_mask_from_params(
    saved_params: dict,
    init_3d: np.ndarray,
    mp_L: np.ndarray,
    mp_R: np.ndarray,
    bones,
    *,
    calib=None,
    confidence_L=None,
    confidence_R=None,
    hrnet_L=None,
    hrnet_R=None,
    stereo_shift_mag=None,
    stereo_response=None,
):
    """Run :func:`detect_mask` with thresholds pulled from a saved
    sidecar dict.  Defaults match the ones the UI ships with so
    older sidecars that don't include the newer keys still produce
    sensible output.
    """
    _mpf = saved_params or {}
    return detect_mask(
        init_3d, mp_L, mp_R, bones,
        calib=calib,
        confidence_L=confidence_L, confidence_R=confidence_R,
        hrnet_L=hrnet_L,           hrnet_R=hrnet_R,
        stereo_shift_mag=stereo_shift_mag,
        stereo_response=stereo_response,
        enable_vel=bool(_mpf.get("enable_vel", False)),
        vel_k=float(_mpf.get("vel_k", 6.0)),
        enable_accel=bool(_mpf.get("enable_accel", True)),
        accel_k=float(_mpf.get("accel_k", 6.0)),
        enable_bone=bool(_mpf.get("enable_bone", True)),
        bone_k=float(_mpf.get("bone_k", 6.0)),
        enable_ydisp=bool(_mpf.get("enable_ydisp", False)),
        ydisp_px=float(_mpf.get("ydisp_px", 5.0)),
        enable_z=bool(_mpf.get("enable_z", False)),
        z_k=float(_mpf.get("z_k", 6.0)),
        enable_mpconf=bool(_mpf.get("enable_mpconf", False)),
        mpconf_min=float(_mpf.get("mpconf_min", 0.5)),
        enable_stereo=bool(_mpf.get("enable_stereo", False)),
        stereo_px=float(_mpf.get("stereo_px", 5.0)),
        # Back-compat: pre-rename mp_filter_params.json used
        # ``enable_stereo_outline`` / ``stereo_hybrid_*`` keys.  Accept
        # both, prefer the new outline-named key.
        enable_stereo_outline=bool(
            _mpf.get("enable_stereo_outline",
                      _mpf.get("enable_stereo_outline", False))
        ),
        stereo_outline_px=float(
            _mpf.get("stereo_outline_px",
                      _mpf.get("stereo_outline_px", 10.0))
        ),
        stereo_outline_conf_min=float(
            _mpf.get("stereo_outline_conf_min",
                      _mpf.get("stereo_outline_conf_min", 0.2))
        ),
        enable_hrnet=bool(_mpf.get("enable_hrnet", False)),
        hrnet_min=float(_mpf.get("hrnet_min", 0.2)),
    )


def build_stereo_fill(
    combined_L: np.ndarray, combined_R: np.ndarray,
    stereo_L: np.ndarray, stereo_R: np.ndarray,
    stereo_response: np.ndarray,
    camera_mask: np.ndarray,
    conf_min: float = 0.4,
):
    """Substitute filtered MP labels with stereo labels where
    confident; otherwise drop the whole joint on both cameras.

    Parameters
    ----------
    combined_L, combined_R : (N, J, 2)
        MP combined 2D keypoints, NaN where missing.
    stereo_L, stereo_R : (N, J, 2)
        Stereo-corrected 2D keypoints from the chosen stereo
        variant (outline preferred over image).
    stereo_response : (N, J)
        Per-(frame, joint) phase-corr confidence baked alongside
        the shifts.  Cells with ``response > conf_min`` may donate.
    camera_mask : (N, J, 2) bool
        The MP-Filter per-camera flag mask (True = "this camera
        was filtered out for this (frame, joint)").
    conf_min : float
        Confidence cut for accepting a stereo donation.

    Returns
    -------
    fill_L, fill_R : (N, J, 2) float
        Per-(frame, joint) decision:
          * neither filtered                         → combined on both
          * exactly one filtered, response > conf_min → stereo on that
            camera, combined on the partner
          * exactly one filtered, response ≤ conf_min → NaN on BOTH
          * both filtered                             → NaN on both
    donated : (N, J) bool
        True where a stereo donation was actually placed into one
        of the cameras (i.e. one-side-filtered + conf OK).  False
        elsewhere — both the "no donation needed" and "dropped"
        paths report False here.
    """
    N, J = combined_L.shape[0], combined_L.shape[1]
    fill_L = combined_L.copy()
    fill_R = combined_R.copy()
    # Align stereo arrays to (N, J, 2) — shorter trials get NaN.
    def _pad(arr):
        out = np.full((N, J, 2), np.nan)
        if arr is None:
            return out
        m = min(N, arr.shape[0])
        out[:m] = arr[:m]
        return out
    sL = _pad(stereo_L)
    sR = _pad(stereo_R)
    resp = np.full((N, J), np.nan)
    if stereo_response is not None:
        m = min(N, stereo_response.shape[0])
        resp[:m] = stereo_response[:m]

    fL = camera_mask[..., 0]
    fR = camera_mask[..., 1]
    conf_ok = np.where(np.isfinite(resp), resp > conf_min, False)

    # Cells where EITHER camera was filtered.  For these we either
    # donate from stereo or drop the entire joint; combined-on-the-
    # unfiltered-side is never the answer (otherwise we'd be making
    # the joint look "fine" when 3D triangulation is impossible).
    either = fL | fR
    drop = either & ~conf_ok          # both filtered, or one filtered + bad conf
    donate_L = (fL & ~fR) & conf_ok   # exactly L filtered, conf OK
    donate_R = (fR & ~fL) & conf_ok   # exactly R filtered, conf OK

    fill_L[drop] = np.nan
    fill_R[drop] = np.nan
    fill_L[donate_L] = sL[donate_L]
    # combined_R was already correct for donate_L cells (R wasn't
    # filtered), so leave fill_R alone there.
    fill_R[donate_R] = sR[donate_R]

    # If BOTH cameras were filtered AND conf was high enough that
    # `drop` excluded the cell, we still need to drop it — donating
    # to both sides doesn't make sense (the shift is a single
    # vector that mirrors L vs R, so substituting both ends gives a
    # joint at the SAME image-space displacement on both cameras,
    # which is meaningless for 3D).
    both_filtered = fL & fR
    fill_L[both_filtered] = np.nan
    fill_R[both_filtered] = np.nan

    # Per-(frame, joint) bool: was a stereo donation actually
    # placed into either camera?  Caller uses this to neutralise
    # signals that are computed against the unfilled MP combined
    # (notably the Stereo error signal: by definition the distance
    # from "MP combined" to "stereo" is zero at a donated cell).
    donated = donate_L | donate_R
    return fill_L, fill_R, donated


def build_and_validate_stereo_fill(
    combined_L: np.ndarray, combined_R: np.ndarray,
    stereo_L: np.ndarray, stereo_R: np.ndarray,
    stereo_response: np.ndarray,
    stereo_shift_mag,
    camera_mask: np.ndarray,
    saved_params: dict,
    init_3d: np.ndarray,
    calib: dict | None,
    bones,
    *,
    n_joints: int = 21,
    conf_min: float = 0.4,
    confidence_L=None, confidence_R=None,
    hrnet_L=None, hrnet_R=None,
):
    """Build stereo-filled labels and prune donations that don't
    actually fix the problem the filter caught.

    Two-pass:

    1. :func:`build_stereo_fill` substitutes the stereo point
       into the filtered camera for cells where confidence is
       above ``conf_min``; drops the rest.
    2. Triangulate the filled labels, re-run
       :func:`detect_mask_from_params` on them, and NaN out any
       (frame, joint) that was originally flagged AND still
       flagged after the donation.  Donated cells have their
       ``stereo_shift_mag`` zeroed before the second pass so the
       Stereo error signal doesn't keep re-flagging the very
       donations it inspired (the post-fill MP-to-stereo distance
       is, by definition, zero at a donated cell).

    Returns
    -------
    fill_L, fill_R : (N, J, 2)   filled 2D labels (NaN where dropped)
    fill_3d        : (N, J, 3)   triangulated post-validation
    donated        : (N, J) bool donations placed in pass 1
    validated      : (N, J) bool donations that survived pass 2
    """
    from .calibration import triangulate_points
    fill_L, fill_R, donated = build_stereo_fill(
        combined_L, combined_R, stereo_L, stereo_R,
        stereo_response, camera_mask, conf_min=conf_min,
    )
    N = fill_L.shape[0]

    def _tri(pL, pR):
        out = np.full((N, n_joints, 3), np.nan)
        if calib is not None:
            for j in range(n_joints):
                out[:, j, :] = triangulate_points(pL[:, j, :], pR[:, j, :], calib)
        return out

    fill_3d = _tri(fill_L, fill_R)
    # Wrist-fallback the 3D so detect_mask's velocity/accel arrays
    # don't NaN-spike across frames where the fill dropped a joint.
    val_3d = np.where(np.isfinite(fill_3d), fill_3d, init_3d)

    # Zero shift_mag at donated cells — see docstring.
    shift_mag_val = None
    if stereo_shift_mag is not None:
        shift_mag_val = np.asarray(stereo_shift_mag, dtype=np.float64).copy()
        m = min(shift_mag_val.shape[0], donated.shape[0])
        shift_mag_val[:m][donated[:m]] = 0.0

    _, new_mask, _ = detect_mask_from_params(
        saved_params, val_3d, fill_L, fill_R, bones,
        calib=calib,
        confidence_L=confidence_L, confidence_R=confidence_R,
        hrnet_L=hrnet_L, hrnet_R=hrnet_R,
        stereo_shift_mag=shift_mag_val,
        stereo_response=stereo_response,
    )

    orig_either = camera_mask[..., 0] | camera_mask[..., 1]
    new_either  = new_mask[..., 0] | new_mask[..., 1]
    fail = orig_either & new_either
    fill_L[fail] = np.nan
    fill_R[fail] = np.nan
    fill_3d = _tri(fill_L, fill_R)
    validated = donated & ~fail
    return fill_L, fill_R, fill_3d, donated, validated
