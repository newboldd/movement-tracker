"""Camera-motion trajectory extraction for handheld stereo video.

For each frame and camera half (OS, OD), computes a homography that
maps that frame's pixel coordinates into a single per-trial reference
frame.  The hand is a tiny minority of pixels; ORB+RANSAC picks the
dominant homography supported by the background, so the trajectory ≈
camera motion with the hand naturally rejected as outliers.

Outputs (saved to ``<dlc>/<subject>/preproc/<stem>/camera_trajectory.npz``):

    H_to_ref_L   (N, 3, 3) float32  — frame i → reference frame, OS
    H_to_ref_R   (N, 3, 3) float32  — same, OD
    n_inliers_L  (N, ) int32        — RANSAC inliers in the pair (i-1 → i)
    n_inliers_R  (N, ) int32          (frame 0 has 0)
    repr_error_L (N, ) float32      — median reprojection error of inliers
    repr_error_R (N, ) float32
    jerk_flag    (N, ) bool         — frame's pairwise registration failed
                                      (too few inliers or too high error)
    reference_frame  int            — chosen reference frame index
    n_frames     int

The reference frame is selected as the frame with the highest sum of
inlier counts to its neighbors — i.e., the "most stable" view in the
trial, where ORB matching is cleanest on both sides.

Per-camera trajectories are computed independently.  The disagreement
between OS and OD trajectories doubles as a diagnostic on the stereo
rig's mechanical integrity — if the cameras flex relative to each
other, the two trajectories diverge.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


_DEFAULT_N_FEATURES = 2000
_DEFAULT_RANSAC_PX  = 3.0
_DEFAULT_MIN_INLIERS = 30
_DEFAULT_MAX_ERR_PX  = 5.0


def _preproc_dir(subject_name: str, trial_stem: str) -> Path:
    from ..config import get_settings
    settings = get_settings()
    return settings.dlc_path / subject_name / "preproc" / trial_stem


def _trajectory_path(subject_name: str, trial_stem: str) -> Path:
    return _preproc_dir(subject_name, trial_stem) / "camera_trajectory.npz"


def trajectory_exists(subject_name: str, trial_stem: str) -> bool:
    return _trajectory_path(subject_name, trial_stem).exists()


def _match_and_solve(kp_prev, des_prev, kp_curr, des_curr,
                     matcher, ransac_thresh: float = _DEFAULT_RANSAC_PX,
                     ) -> tuple[np.ndarray, int, float]:
    if (des_prev is None or des_curr is None
            or len(kp_prev) < 4 or len(kp_curr) < 4):
        return np.eye(3, dtype=np.float32), 0, float("inf")
    matches = matcher.match(des_prev, des_curr)
    if len(matches) < 4:
        return np.eye(3, dtype=np.float32), 0, float("inf")
    matches = sorted(matches, key=lambda m: m.distance)[:400]
    src = np.float32([kp_prev[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp_curr[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
    if H is None or inlier_mask is None:
        return np.eye(3, dtype=np.float32), 0, float("inf")
    inlier_mask = inlier_mask.ravel().astype(bool)
    n_in = int(inlier_mask.sum())
    if n_in < 4:
        return np.eye(3, dtype=np.float32), 0, float("inf")
    src_in = src[inlier_mask].reshape(-1, 2)
    dst_in = dst[inlier_mask].reshape(-1, 2)
    src_h = np.hstack([src_in, np.ones((n_in, 1), dtype=np.float32)])
    proj = (H @ src_h.T).T
    proj_xy = proj[:, :2] / proj[:, 2:3]
    err = np.linalg.norm(proj_xy - dst_in, axis=1)
    return H.astype(np.float32), n_in, float(np.median(err))


def _hand_visibility_scores(
    H_tref_L: np.ndarray, H_tref_R: np.ndarray | None,
    subject_name: str, start_frame: int, n_frames: int,
    w_half: int, h_full: int,
) -> np.ndarray | None:
    """Compute per-candidate hand-visibility score using MediaPipe prelabels.

    For each candidate reference frame ``k``, count how many hand
    keypoints across the whole trial would remain within image bounds
    when the trial is warped to ``k``'s coordinate system.  Higher
    scores ⇒ ``k`` is a better reference for keeping the hand visible.

    Returns ``None`` if MediaPipe prelabels for the subject aren't
    available — the caller should then fall back to the camera-motion
    heuristic.

    The math:
        Let G_j = H_to_tref[j] (frame_j → tentative_ref coords).
        For candidate ref ``k``: pos_in_k = inv(G_k) @ G_j @ p_j
        where p_j is the keypoint position in frame_j's pixel coords.
        We pre-compute q_j = G_j @ p_j (positions in tentative-ref
        coords) once, then for each candidate ``k`` apply ``inv(G_k)``
        to every q and bounds-check.
    """
    try:
        from .mediapipe_prelabel import load_mediapipe_prelabels
    except ImportError:
        return None
    try:
        mp = load_mediapipe_prelabels(subject_name)
    except Exception as e:
        logger.warning(f"hand_visibility_scores: MP load failed: {e}")
        return None
    if mp is None:
        return None

    os_lm = mp.get("OS_landmarks")
    od_lm = mp.get("OD_landmarks")
    if os_lm is None or os_lm.size == 0:
        return None

    is_stereo = H_tref_R is not None

    def _q_in_tref(H_chain: np.ndarray, all_lm: np.ndarray) -> np.ndarray:
        """Project every frame's valid keypoints into tentative-ref coords.
        Returns (M, 3) stack — homogeneous coords."""
        end = min(start_frame + n_frames, all_lm.shape[0])
        trial_lm = all_lm[start_frame:end]   # (N_lm, 21, 2)
        if trial_lm.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float64)
        N_lm = trial_lm.shape[0]
        K = trial_lm.shape[1]
        # Mask out NaN keypoints — they correspond to frames where MP failed.
        valid = ~np.isnan(trial_lm).any(axis=-1)  # (N_lm, K)
        if not valid.any():
            return np.zeros((0, 3), dtype=np.float64)
        # Augment to homogeneous: (N_lm, K, 3)
        pts_h = np.concatenate(
            [trial_lm, np.ones((N_lm, K, 1), dtype=trial_lm.dtype)],
            axis=-1,
        ).astype(np.float64)
        # Per-frame matmul: H_chain[j] (3,3) @ pts_h[j] (K,3) → q_j (K,3)
        H_use = H_chain[:N_lm]
        # einsum: q[j,k,i] = H_use[j,i,x] * pts_h[j,k,x]
        q = np.einsum('jix,jkx->jki', H_use.astype(np.float64), pts_h)
        # Collect only valid keypoints → (M, 3)
        return q[valid]

    q_L = _q_in_tref(H_tref_L, os_lm)
    q_R = _q_in_tref(H_tref_R, od_lm) if (is_stereo and od_lm is not None) else None

    if q_L.shape[0] == 0 and (q_R is None or q_R.shape[0] == 0):
        return None

    def _score_side(H_chain: np.ndarray, all_q: np.ndarray, w: int, h: int) -> np.ndarray:
        scores = np.zeros(H_chain.shape[0], dtype=np.float64)
        if all_q.shape[0] == 0:
            return scores
        for k in range(H_chain.shape[0]):
            try:
                G_inv = np.linalg.inv(H_chain[k].astype(np.float64))
            except np.linalg.LinAlgError:
                scores[k] = 0.0
                continue
            # warped = G_inv @ all_q.T → (3, M)
            warped = G_inv @ all_q.T
            wz = warped[2]
            ok = np.abs(wz) > 1e-9
            wx = np.where(ok, warped[0] / np.where(ok, wz, 1.0), np.nan)
            wy = np.where(ok, warped[1] / np.where(ok, wz, 1.0), np.nan)
            in_b = (wx >= 0) & (wx < w) & (wy >= 0) & (wy < h)
            scores[k] = int(in_b.sum())
        return scores

    scores_L = _score_side(H_tref_L, q_L, w_half, h_full)
    scores_R = (_score_side(H_tref_R, q_R, w_half, h_full)
                if (is_stereo and q_R is not None) else 0)
    total = scores_L + (scores_R if is_stereo else 0)
    return total


def _chain_to_reference(H_pairs: list[np.ndarray], ref: int) -> np.ndarray:
    N = len(H_pairs) + 1
    H_to_ref = np.empty((N, 3, 3), dtype=np.float32)
    H_to_ref[ref] = np.eye(3, dtype=np.float32)
    for i in range(ref - 1, -1, -1):
        H_to_ref[i] = H_to_ref[i + 1] @ H_pairs[i]
    for i in range(ref + 1, N):
        try:
            inv = np.linalg.inv(H_pairs[i - 1])
        except np.linalg.LinAlgError:
            inv = np.eye(3, dtype=np.float32)
        H_to_ref[i] = H_to_ref[i - 1] @ inv
    return H_to_ref


def compute_camera_trajectory(
    subject_name: str,
    trial_idx: int,
    progress_callback=None,
    cancel_event=None,
    nfeatures: int = _DEFAULT_N_FEATURES,
    ransac_thresh: float = _DEFAULT_RANSAC_PX,
    min_inliers: int = _DEFAULT_MIN_INLIERS,
    max_err_px:  float = _DEFAULT_MAX_ERR_PX,
) -> str:
    from .video import build_trial_map

    tmap = build_trial_map(subject_name)
    if trial_idx < 0 or trial_idx >= len(tmap):
        raise ValueError(f"trial_idx {trial_idx} out of range ({len(tmap)} trials)")
    trial = tmap[trial_idx]
    stem = trial["trial_name"]
    video_path = trial["video_path"]
    start_frame = int(trial.get("start_frame", 0))
    n_frames    = int(trial["frame_count"])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    orb = cv2.ORB_create(nfeatures=nfeatures, fastThreshold=12)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    H_pairs_OS = [None] * (n_frames - 1) if n_frames > 1 else []
    H_pairs_OD = [None] * (n_frames - 1) if n_frames > 1 else []
    inliers_pair_OS = np.zeros(n_frames - 1, dtype=np.int32) if n_frames > 1 else np.zeros(0, dtype=np.int32)
    inliers_pair_OD = np.zeros(n_frames - 1, dtype=np.int32) if n_frames > 1 else np.zeros(0, dtype=np.int32)
    err_pair_OS = np.full(n_frames - 1, np.inf, dtype=np.float32) if n_frames > 1 else np.zeros(0, dtype=np.float32)
    err_pair_OD = np.full(n_frames - 1, np.inf, dtype=np.float32) if n_frames > 1 else np.zeros(0, dtype=np.float32)

    prev_kp_OS = prev_des_OS = None
    prev_kp_OD = prev_des_OD = None
    is_stereo = None

    for i in range(n_frames):
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Job cancelled")
        ok, frame = cap.read()
        if not ok:
            logger.warning(f"camera_motion: read failed at frame {i}; truncating")
            n_frames = i
            break
        if is_stereo is None:
            h, full_w = frame.shape[:2]
            is_stereo = (full_w / h) > 1.7
            half_w = full_w // 2 if is_stereo else full_w
        if is_stereo:
            os_gray = cv2.cvtColor(frame[:, :half_w], cv2.COLOR_BGR2GRAY)
            od_gray = cv2.cvtColor(frame[:, half_w:], cv2.COLOR_BGR2GRAY)
        else:
            os_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            od_gray = None
        kp_OS, des_OS = orb.detectAndCompute(os_gray, None)
        kp_OD, des_OD = (orb.detectAndCompute(od_gray, None)
                          if od_gray is not None else (None, None))
        if i > 0:
            H_os, n_os, err_os = _match_and_solve(
                prev_kp_OS, prev_des_OS, kp_OS, des_OS, matcher, ransac_thresh)
            H_pairs_OS[i - 1] = H_os
            inliers_pair_OS[i - 1] = n_os
            err_pair_OS[i - 1] = err_os
            if od_gray is not None:
                H_od, n_od, err_od = _match_and_solve(
                    prev_kp_OD, prev_des_OD, kp_OD, des_OD, matcher, ransac_thresh)
                H_pairs_OD[i - 1] = H_od
                inliers_pair_OD[i - 1] = n_od
                err_pair_OD[i - 1] = err_od
        prev_kp_OS, prev_des_OS = kp_OS, des_OS
        prev_kp_OD, prev_des_OD = kp_OD, des_OD

        if progress_callback is not None:
            try:
                progress_callback(95.0 * (i + 1) / max(1, n_frames))
            except Exception:
                pass

    cap.release()

    if n_frames <= 1:
        raise RuntimeError(f"Trial {stem} has too few frames ({n_frames}) for trajectory")

    H_pairs_OS = H_pairs_OS[: n_frames - 1]
    H_pairs_OD = H_pairs_OD[: n_frames - 1]
    inliers_pair_OS = inliers_pair_OS[: n_frames - 1]
    inliers_pair_OD = inliers_pair_OD[: n_frames - 1]
    err_pair_OS = err_pair_OS[: n_frames - 1]
    err_pair_OD = err_pair_OD[: n_frames - 1]

    # ── Per-frame inlier / error / jerk-flag (independent of ref choice) ──
    nb_OS = np.zeros(n_frames, dtype=np.int32)
    nb_OD = np.zeros(n_frames, dtype=np.int32)
    if n_frames >= 2:
        nb_OS[1:] += inliers_pair_OS
        if is_stereo:
            nb_OD[1:] += inliers_pair_OD
        nb_OS[:-1] += inliers_pair_OS
        if is_stereo:
            nb_OD[:-1] += inliers_pair_OD
    stability = (nb_OS.astype(np.int64)
                 + (nb_OD.astype(np.int64) if is_stereo else 0))

    inliers_L = np.zeros(n_frames, dtype=np.int32)
    inliers_R = np.zeros(n_frames, dtype=np.int32)
    err_L = np.zeros(n_frames, dtype=np.float32)
    err_R = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        nb, nbR, nbE, nbER = [], [], [], []
        if i > 0:
            nb.append(int(inliers_pair_OS[i - 1]))
            nbE.append(float(err_pair_OS[i - 1]))
            if is_stereo:
                nbR.append(int(inliers_pair_OD[i - 1]))
                nbER.append(float(err_pair_OD[i - 1]))
        if i < n_frames - 1:
            nb.append(int(inliers_pair_OS[i]))
            nbE.append(float(err_pair_OS[i]))
            if is_stereo:
                nbR.append(int(inliers_pair_OD[i]))
                nbER.append(float(err_pair_OD[i]))
        inliers_L[i] = min(nb) if nb else 0
        err_L[i]     = max(nbE) if nbE else float("inf")
        inliers_R[i] = min(nbR) if (is_stereo and nbR) else 0
        err_R[i]     = max(nbER) if (is_stereo and nbER) else float("inf")

    jerk_flag = ((inliers_L < min_inliers)
                 | (err_L > max_err_px)
                 | ((inliers_R < min_inliers) if is_stereo else False)
                 | ((err_R > max_err_px) if is_stereo else False))

    # ── Pick reference frame ──────────────────────────────────────────
    # Primary criterion (when MediaPipe prelabels are available):
    # maximise the number of hand keypoints that remain inside the
    # image after the trial is warped to the candidate's ref coords.
    # This directly addresses the Con02_R2 failure mode — picking a
    # ref where the camera was off-centre meant stable.mp4 cropped
    # the hand out of view for many frames, breaking downstream MP /
    # HRnet detection on stable.mp4.
    #
    # Fallback (no MP yet): pick the frame whose camera position is
    # closest to the trial-wide median, smoothed so the winner sits in
    # the middle of a sustained stable region rather than at an
    # isolated near-median outlier.
    valid_mask = ~jerk_flag
    if not valid_mask.any():
        valid_mask = np.ones(n_frames, dtype=bool)
    stability_for_tentative = stability.copy()
    stability_for_tentative[~valid_mask] = -1
    tentative_ref = int(np.argmax(stability_for_tentative))

    H_tref_L = _chain_to_reference(H_pairs_OS, tentative_ref)
    H_tref_R = (_chain_to_reference(H_pairs_OD, tentative_ref)
                if is_stereo else
                np.tile(np.eye(3, dtype=np.float32), (n_frames, 1, 1)))

    # Try the hand-visibility metric first.  Quietly falls back when
    # MediaPipe prelabels aren't available or the trial range is empty.
    hand_scores = _hand_visibility_scores(
        H_tref_L,
        H_tref_R if is_stereo else None,
        subject_name, start_frame, n_frames,
        half_w, h,
    )
    win = int(max(11, min(101, n_frames // 20)))
    if win % 2 == 0:
        win += 1
    kernel = np.ones(win, dtype=np.float64) / win

    method = None
    if hand_scores is not None and np.any(hand_scores > 0):
        # Smooth hand-visibility scores; argmax over non-jerk frames.
        padded = np.pad(hand_scores, win // 2, mode='edge')
        score_smooth = np.convolve(padded, kernel, mode='valid')
        score_pick = score_smooth.copy()
        score_pick[jerk_flag] = -np.inf
        if not np.isfinite(score_pick).any() or score_pick.max() <= 0:
            score_pick = score_smooth
        ref = int(np.argmax(score_pick))
        method = (f"hand_visibility (score={score_pick[ref]:.1f} "
                  f"keypoints-in-view, smoothed over window={win})")
    else:
        # Fall back to camera-position median.
        tx_L = H_tref_L[:, 0, 2].astype(np.float64)
        ty_L = H_tref_L[:, 1, 2].astype(np.float64)
        tx_med_L = float(np.median(tx_L[valid_mask]))
        ty_med_L = float(np.median(ty_L[valid_mask]))
        d_L = np.sqrt((tx_L - tx_med_L) ** 2 + (ty_L - ty_med_L) ** 2)
        if is_stereo:
            tx_R = H_tref_R[:, 0, 2].astype(np.float64)
            ty_R = H_tref_R[:, 1, 2].astype(np.float64)
            tx_med_R = float(np.median(tx_R[valid_mask]))
            ty_med_R = float(np.median(ty_R[valid_mask]))
            d_R = np.sqrt((tx_R - tx_med_R) ** 2 + (ty_R - ty_med_R) ** 2)
        else:
            d_R = np.zeros(n_frames, dtype=np.float64)
        d_total = d_L + d_R
        padded = np.pad(d_total, win // 2, mode='edge')
        d_smooth = np.convolve(padded, kernel, mode='valid')
        d_for_pick = d_smooth.copy()
        d_for_pick[jerk_flag] = np.inf
        if not np.isfinite(d_for_pick).any():
            d_for_pick = d_smooth
        ref = int(np.argmin(d_for_pick))
        method = (f"camera_median (no MP labels; "
                  f"d_smooth[{ref}]={d_for_pick[ref]:.1f}px, window={win})")

    logger.info(
        f"camera_motion {subject_name}/{stem}: reference frame = {ref}  "
        f"(tentative was {tentative_ref}; method = {method})"
    )

    H_to_ref_L = (H_tref_L if ref == tentative_ref
                  else _chain_to_reference(H_pairs_OS, ref))
    H_to_ref_R = (H_tref_R if (ref == tentative_ref or not is_stereo)
                  else _chain_to_reference(H_pairs_OD, ref))

    out_dir = _preproc_dir(subject_name, stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = _trajectory_path(subject_name, stem)
    np.savez_compressed(
        str(out_path),
        H_to_ref_L=H_to_ref_L,
        H_to_ref_R=H_to_ref_R,
        n_inliers_L=inliers_L,
        n_inliers_R=inliers_R,
        repr_error_L=err_L,
        repr_error_R=err_R,
        jerk_flag=jerk_flag,
        reference_frame=np.array(ref, dtype=np.int32),
        n_frames=np.array(n_frames, dtype=np.int32),
        is_stereo=np.array(bool(is_stereo), dtype=bool),
        start_frame=np.array(start_frame, dtype=np.int32),
        nfeatures=np.array(nfeatures, dtype=np.int32),
        ransac_thresh=np.array(ransac_thresh, dtype=np.float32),
        min_inliers=np.array(min_inliers, dtype=np.int32),
        max_err_px=np.array(max_err_px, dtype=np.float32),
    )
    if progress_callback is not None:
        try: progress_callback(100.0)
        except Exception: pass
    logger.info(
        f"camera_motion saved: {out_path}  "
        f"N={n_frames}  ref={ref}  "
        f"jerk={int(jerk_flag.sum())}/{n_frames} frames"
    )
    return str(out_path)


def load_camera_trajectory(subject_name: str, trial_stem: str) -> dict | None:
    path = _trajectory_path(subject_name, trial_stem)
    if not path.exists():
        return None
    try:
        d = np.load(str(path))
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
        return None
    return {
        "H_to_ref_L": d["H_to_ref_L"],
        "H_to_ref_R": d["H_to_ref_R"],
        "n_inliers_L": d["n_inliers_L"],
        "n_inliers_R": d["n_inliers_R"],
        "repr_error_L": d["repr_error_L"],
        "repr_error_R": d["repr_error_R"],
        "jerk_flag": d["jerk_flag"],
        "reference_frame": int(d["reference_frame"]),
        "n_frames": int(d["n_frames"]),
        "is_stereo": bool(d["is_stereo"]) if "is_stereo" in d.files else True,
        "start_frame": int(d["start_frame"]) if "start_frame" in d.files else 0,
    }


def summarise_trajectory(traj: dict) -> dict:
    H_L = traj["H_to_ref_L"]
    H_R = traj["H_to_ref_R"]
    N = H_L.shape[0]
    is_stereo = traj["is_stereo"]

    def _decompose(H_stack: np.ndarray) -> dict:
        tx = H_stack[:, 0, 2].astype(float)
        ty = H_stack[:, 1, 2].astype(float)
        rot = np.arctan2(H_stack[:, 1, 0], H_stack[:, 0, 0]).astype(float)
        return {"tx": [float(x) for x in tx],
                "ty": [float(y) for y in ty],
                "rot_deg": [float(np.degrees(r)) for r in rot]}

    summary = {
        "n_frames": N,
        "reference_frame": traj["reference_frame"],
        "is_stereo": is_stereo,
        "jerk_flag": [bool(x) for x in traj["jerk_flag"]],
        "n_inliers_L": [int(x) for x in traj["n_inliers_L"]],
        "n_inliers_R": [int(x) for x in traj["n_inliers_R"]],
        "repr_error_L": [float(x) if np.isfinite(x) else None
                          for x in traj["repr_error_L"]],
        "repr_error_R": [float(x) if np.isfinite(x) else None
                          for x in traj["repr_error_R"]],
        "OS": _decompose(H_L),
        "OD": _decompose(H_R) if is_stereo else None,
    }
    if is_stereo:
        dx = np.array(summary["OS"]["tx"]) - np.array(summary["OD"]["tx"])
        dy = np.array(summary["OS"]["ty"]) - np.array(summary["OD"]["ty"])
        dev_x = dx - np.median(dx)
        dev_y = dy - np.median(dy)
        rms = float(np.sqrt(np.mean(dev_x**2 + dev_y**2)))
        summary["os_od_rms_disagreement_px"] = rms
    return summary
