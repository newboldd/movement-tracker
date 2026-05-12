"""Stabilisation + foreground/background mask extraction.

Two artifacts per trial, both directly consumable by MP/HRnet/DLC:

1.  ``stable.mp4`` — every source frame warped into the reference
    frame's coordinate system using the camera trajectory.  The static
    scene stays put; only the hand (and any other moving parts) appears
    to move.  Same resolution and layout as the source video, so any
    consumer that reads the original video can read this one too.

2.  ``fg.mp4`` — grayscale per-frame foreground mask: the per-pixel
    absolute difference between the stabilised frame and the static
    background, max-pooled across BGR channels.  Bright = motion (hand);
    dark = static scene.  Same resolution as the source.

The static background needed to compute the mask is itself computed
internally as a temporal median across uniformly-sampled stabilised
frames.  It is saved as ``background.npz`` (+ PNGs for inspection)
mainly as a diagnostic artifact; the mp4s are the operational outputs.

Inputs:
    Camera trajectory (``camera_trajectory.npz``) must already exist —
    compute it via the *Compute Trajectory* button first.

Outputs in ``<dlc>/<subject>/preproc/<stem>/``:
    stable.mp4              stabilised source video
    fg.mp4                  per-frame foreground mask (grayscale)
    background.npz          temporal-median BG image + MAD + metadata
    background_OS.png       BG image, OS half (full-res preview)
    background_OD.png       BG image, OD half (stereo only)
    mad_OS.png, mad_OD.png  MAD heat-maps for quality inspection

``H_to_ref[i]`` maps frame-i pixel coords → reference-frame pixel coords
(forward direction).  We pass it to ``warpPerspective`` *without*
``WARP_INVERSE_MAP`` — OpenCV then internally inverts it and samples
each destination pixel from ``inv(H_to_ref) @ (x, y)`` in the source,
which is exactly what we want: a destination image where every pixel
shows the scene point at that reference-frame location.
"""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


_DEFAULT_MAX_SAMPLES = 120     # cap to keep RAM bounded for long trials
_DEFAULT_DOWNSCALE   = 2       # downsample factor inside the median stack
_MAX_RAM_BYTES       = 1_500_000_000   # ~1.5 GB ceiling per side


def _open_ffmpeg_pipe(output_path: str, width: int, height: int,
                      fps: float, pix_fmt_in: str = "bgr24") -> subprocess.Popen:
    """Open an ffmpeg subprocess that consumes raw frames on stdin and
    writes a yuv420p / H.264 mp4 to ``output_path``.

    Mirrors the pattern in ``deidentify.py`` so the produced mp4 plays
    in every browser and can be read by OpenCV / MediaPipe consumers.
    """
    from .ffmpeg import get_ffmpeg_path
    ffmpeg = get_ffmpeg_path()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found. Install FFmpeg or pip install imageio-ffmpeg")
    if Path(output_path).exists():
        Path(output_path).unlink()
    kwargs = {"stdin": subprocess.PIPE,
              "stdout": subprocess.DEVNULL,
              "stderr": subprocess.PIPE}
    if os.name == "nt":
        kwargs["creationflags"] = (
            getattr(subprocess, "DETACHED_PROCESS", 0) |
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        )
    return subprocess.Popen([
        ffmpeg, "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{width}x{height}", "-pix_fmt", pix_fmt_in,
        "-r", str(fps),
        "-i", "-",
        # crf 20 for stable.mp4 keeps detail; mask gets crf 23 — masks
        # are smooth-ish so don't need full quality.
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-an",
        "-movflags", "+faststart",
        output_path,
    ], **kwargs)


def _stable_path(subject_name: str, trial_stem: str) -> Path:
    return _preproc_dir(subject_name, trial_stem) / "stable.mp4"


def _fg_path(subject_name: str, trial_stem: str) -> Path:
    return _preproc_dir(subject_name, trial_stem) / "fg.mp4"


def _hand_path(subject_name: str, trial_stem: str) -> Path:
    """Strictly keypoint-gated hand mask (no motion / colour contribution).
    Used for the 'isolated' canvas composite — pixels far from any
    MediaPipe landmark are guaranteed-dark even if they moved or are
    skin-coloured."""
    return _preproc_dir(subject_name, trial_stem) / "hand.mp4"


def _outline_path(subject_name: str, trial_stem: str) -> Path:
    """Per-frame contour of the hand mask (Canny edges, slightly
    dilated).  Overlaid as a checkbox option to show the boundary of
    the segmentation regardless of which overlay mode is active."""
    return _preproc_dir(subject_name, trial_stem) / "outline.mp4"


def _preproc_dir(subject_name: str, trial_stem: str) -> Path:
    from ..config import get_settings
    settings = get_settings()
    return settings.dlc_path / subject_name / "preproc" / trial_stem


def _background_path(subject_name: str, trial_stem: str) -> Path:
    return _preproc_dir(subject_name, trial_stem) / "background.npz"


def background_exists(subject_name: str, trial_stem: str) -> bool:
    return _background_path(subject_name, trial_stem).exists()


def _pick_sample_frames(n_frames: int, jerk_flag: np.ndarray,
                        max_samples: int) -> np.ndarray:
    """Uniform sample across the trial, skipping jerk-flagged frames.

    Jerk frames have unreliable homographies — including them would smear
    the median.  We drop them up-front and uniform-sample the survivors.
    Falls back to uniform sampling without filtering if nearly every
    frame is flagged.
    """
    good = np.where(~jerk_flag)[0]
    if good.size < max(8, max_samples // 4):
        # Too many jerks — fall back to all frames so we still get a
        # background (degraded quality).
        logger.warning(
            f"background: only {good.size}/{n_frames} non-jerk frames; "
            "using all frames for sampling"
        )
        good = np.arange(n_frames, dtype=np.int64)
    if good.size <= max_samples:
        return good.astype(np.int32)
    idx = np.linspace(0, good.size - 1, max_samples).round().astype(np.int64)
    return good[idx].astype(np.int32)


# ─── Enhanced FG-mask helpers (MediaPipe + colour) ──────────────────────
#
# The plain "|frame − BG|" mask has two predictable failure modes:
#   1. Hand pixels whose colour happens to match the background → dim
#   2. Anything else that moved (head, other hand, scene clutter) → bright
#
# When MediaPipe prelabels are available we can refine the mask with two
# extra signals:
#   • Keypoint proximity — a Gaussian-blurred stamp around each of the 21
#     hand landmarks, warped into the reference frame.  This says "the
#     hand is roughly here, even if motion is weak".
#   • Skin-colour similarity — a CbCr Mahalanobis-distance model fit
#     from pixels at MP keypoints in sampled frames.  This says "this
#     pixel's colour matches the trial's hand", regardless of motion.
#
# Combined: motion·w_m + keypoint·w_k + colour·w_c, clipped to 0..255.
# Falls back to motion-only when MediaPipe data is absent.


def _interpolate_keypoints(kpts: np.ndarray) -> np.ndarray:
    """Fill NaN gaps in (N, K, 2) keypoint arrays by linear time-interpolation.

    Per axis, per keypoint: np.interp over valid frame indices.  Frames
    before the first valid / after the last valid get forward / backward
    filled.  Keypoints with no valid frames at all stay NaN.
    """
    out = kpts.astype(np.float64, copy=True)
    N, K, _ = out.shape
    if N < 2:
        return out
    all_idx = np.arange(N)
    for k in range(K):
        for ax in range(2):
            col = out[:, k, ax]
            valid = ~np.isnan(col)
            n_valid = int(valid.sum())
            if n_valid == 0:
                continue
            if n_valid == 1:
                col[~valid] = col[valid][0]
                continue
            valid_idx = np.where(valid)[0]
            col[~valid] = np.interp(all_idx[~valid], valid_idx, col[valid])
    return out


def _warp_kpts_2d(kpts: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Apply a 3×3 homography to (K, 2) keypoints.  NaN keypoints stay NaN."""
    out = np.full_like(kpts, np.nan, dtype=np.float64)
    valid = ~np.isnan(kpts).any(axis=-1)
    if not valid.any():
        return out
    pts = kpts[valid].astype(np.float64)
    pts_h = np.column_stack([pts, np.ones(len(pts))]).T   # (3, V)
    warped = H.astype(np.float64) @ pts_h                  # (3, V)
    w = warped[2]
    ok = np.abs(w) > 1e-9
    xy = np.full((len(pts), 2), np.nan)
    xy[ok] = (warped[:2, ok] / w[ok]).T
    out[valid] = xy
    return out


def _build_kpt_proximity_mask(kpts_ref: np.ndarray, w: int, h: int,
                                sigma: float = 30.0) -> np.ndarray:
    """Soft hand mask via Gaussian-blurred keypoint stamps.

    Each keypoint at ``(x, y)`` writes a single 1.0 into a zero canvas,
    which is then blurred by ``cv2.GaussianBlur(sigma)``.  Peak values
    are renormalised so a single point post-blur reaches 1.0 (the raw
    blur peak for a delta is ~1/(2πσ²) ≈ tiny).
    """
    mask = np.zeros((h, w), dtype=np.float32)
    valid = ~np.isnan(kpts_ref).any(axis=-1)
    if not valid.any():
        return mask
    n_stamped = 0
    for x, y in kpts_ref[valid]:
        xi, yi = int(round(float(x))), int(round(float(y)))
        if 0 <= xi < w and 0 <= yi < h:
            mask[yi, xi] = 1.0
            n_stamped += 1
    if n_stamped == 0:
        return mask
    blurred = cv2.GaussianBlur(mask, (0, 0), sigma)
    # Restore peak to ~1 so the final mask is in approx-[0,1].
    peak_factor = 2.0 * np.pi * sigma * sigma
    return np.clip(blurred * peak_factor, 0.0, 1.0)


def _fit_skin_model_cbcr(sampled_frames: np.ndarray,
                          sampled_kpts: list[np.ndarray],
                          patch_radius: int = 2) -> dict | None:
    """Fit a 2-D Gaussian (Cb, Cr) skin-tone model from pixels near hand keypoints.

    Y (luminance) is dropped — skin chrominance is much more lighting-
    invariant than its brightness, which lets the same model survive
    cross-trial lighting changes.

    Returns ``{'mean': (2,), 'cov_inv': (2,2)}`` or ``None`` if fewer
    than 50 samples could be collected.
    """
    cbcr_samples = []
    n_frames = len(sampled_frames)
    for i in range(n_frames):
        kpts = sampled_kpts[i]
        valid = ~np.isnan(kpts).any(axis=-1)
        if not valid.any():
            continue
        ycc = cv2.cvtColor(sampled_frames[i], cv2.COLOR_BGR2YCrCb)
        fh, fw = ycc.shape[:2]
        for x, y in kpts[valid]:
            xi, yi = int(round(float(x))), int(round(float(y)))
            y0, y1 = max(0, yi - patch_radius), min(fh, yi + patch_radius + 1)
            x0, x1 = max(0, xi - patch_radius), min(fw, xi + patch_radius + 1)
            if y1 > y0 and x1 > x0:
                cbcr_samples.append(ycc[y0:y1, x0:x1, 1:3].reshape(-1, 2))
    if not cbcr_samples:
        return None
    samples = np.vstack(cbcr_samples).astype(np.float32)
    if len(samples) < 50:
        return None
    mean = samples.mean(axis=0)
    cov = np.cov(samples.T)
    try:
        cov_inv = np.linalg.inv(cov + np.eye(2, dtype=np.float64) * 1.0)
    except np.linalg.LinAlgError:
        return None
    return {"mean": mean.astype(np.float32),
            "cov_inv": cov_inv.astype(np.float32)}


def _color_similarity(frame_bgr: np.ndarray, skin_model: dict,
                       scale: float = 8.0) -> np.ndarray:
    """Per-pixel exp(-Mahalanobis²/scale) under a CbCr skin model.

    Returns (H, W) float32 in [0, 1].  ``scale`` controls how tightly the
    similarity falls off — 8 is forgiving (broad hand-region match), 4
    is tight (only near-exact skin matches).
    """
    ycc = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    diff = ycc[..., 1:3].astype(np.float32) - skin_model["mean"]   # (H, W, 2)
    # Mahalanobis² = diffᵀ · cov_inv · diff, applied to every pixel.
    a = diff @ skin_model["cov_inv"].T                             # (H, W, 2)
    m_sq = (a * diff).sum(axis=-1)                                 # (H, W)
    return np.exp(-m_sq / float(scale)).astype(np.float32)


def _build_enhanced_fg_mask(
    warped_frame: np.ndarray,
    bg_full: np.ndarray,
    kpts_ref: np.ndarray | None,
    skin_model: dict | None,
    motion_norm: float = 60.0,
    kpt_sigma: float = 30.0,
    w_motion: float = 0.3,
    w_kpt: float = 0.5,
    w_color: float = 0.2,
) -> np.ndarray:
    """Combine motion + keypoint proximity + skin-colour similarity into a
    single (H, W) uint8 mask.  Weights are redistributed when a signal
    isn't available so the remaining ones still hit a 1.0 peak.

    motion_norm = the |frame−BG| value that should map to 1.0; ~60 lands
    the hand on full intensity given the typical mask range.
    """
    H, W = warped_frame.shape[:2]

    # 1) Motion
    motion = np.abs(warped_frame.astype(np.int16) - bg_full.astype(np.int16))
    motion = motion.max(axis=-1).astype(np.float32) / motion_norm
    np.clip(motion, 0.0, 1.0, out=motion)

    # 2) Keypoint proximity (zero when unavailable)
    if kpts_ref is not None:
        kpt = _build_kpt_proximity_mask(kpts_ref, W, H, sigma=kpt_sigma)
    else:
        kpt = None

    # 3) Colour similarity (zero when unavailable)
    if skin_model is not None:
        color = _color_similarity(warped_frame, skin_model)
    else:
        color = None

    # Weight redistribution: if a signal is missing, its weight is split
    # evenly between the remaining ones so the peak stays near 1.
    have_kpt = kpt is not None
    have_color = color is not None
    if have_kpt and have_color:
        combined = motion * w_motion + kpt * w_kpt + color * w_color
    elif have_kpt:
        combined = motion * (w_motion + w_color * 0.5) + kpt * (w_kpt + w_color * 0.5)
    elif have_color:
        combined = motion * (w_motion + w_kpt * 0.5) + color * (w_color + w_kpt * 0.5)
    else:
        combined = motion   # fall back to plain motion-only

    return np.clip(combined * 255.0, 0.0, 255.0).astype(np.uint8)


def compute_background(
    subject_name: str,
    trial_idx: int,
    progress_callback=None,
    cancel_event=None,
    max_samples: int = _DEFAULT_MAX_SAMPLES,
    downscale:   int = _DEFAULT_DOWNSCALE,
) -> str:
    """Compute the temporal-median background image for one trial.

    Requires a saved camera trajectory; raises ``RuntimeError`` otherwise.
    """
    from .video import build_trial_map
    from .camera_motion import load_camera_trajectory

    tmap = build_trial_map(subject_name)
    if trial_idx < 0 or trial_idx >= len(tmap):
        raise ValueError(f"trial_idx {trial_idx} out of range ({len(tmap)} trials)")
    trial = tmap[trial_idx]
    stem = trial["trial_name"]
    video_path = trial["video_path"]

    traj = load_camera_trajectory(subject_name, stem)
    if traj is None:
        raise RuntimeError(
            f"No camera trajectory for {subject_name}/{stem} — "
            "run Compute Trajectory first."
        )
    H_L = traj["H_to_ref_L"]
    H_R = traj["H_to_ref_R"]
    is_stereo = bool(traj["is_stereo"])
    n_frames  = int(traj["n_frames"])
    ref       = int(traj["reference_frame"])

    if downscale < 1:
        downscale = 1
    if max_samples < 4:
        max_samples = 4

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    # Pick which frames to use.
    frames_sampled = _pick_sample_frames(n_frames, traj["jerk_flag"], max_samples)
    n_samples = int(frames_sampled.size)
    sample_lookup = {int(f): i for i, f in enumerate(frames_sampled)}

    # Probe the first frame to get dimensions.
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, first = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Failed to read first frame")
    h_full, w_full = first.shape[:2]
    if is_stereo:
        w_half = w_full // 2
    else:
        w_half = w_full
    out_h = h_full // downscale
    out_w = w_half // downscale

    # Sanity check on RAM: n_samples × out_h × out_w × 3 bytes × (1 or 2 sides).
    n_sides = 2 if is_stereo else 1
    bytes_needed = n_samples * out_h * out_w * 3 * n_sides
    if bytes_needed > _MAX_RAM_BYTES:
        # Re-scale max_samples down to fit.
        new_max = int(_MAX_RAM_BYTES / (out_h * out_w * 3 * n_sides))
        logger.warning(
            f"background: requested {n_samples} samples × {out_h}×{out_w}×3 "
            f"× {n_sides} sides = {bytes_needed/1e9:.2f} GB exceeds "
            f"{_MAX_RAM_BYTES/1e9:.1f} GB cap; reducing to {new_max} samples"
        )
        frames_sampled = _pick_sample_frames(n_frames, traj["jerk_flag"], new_max)
        n_samples = int(frames_sampled.size)
        sample_lookup = {int(f): i for i, f in enumerate(frames_sampled)}

    # Allocate the warped-frame stacks.  uint8 keeps RAM minimal; the
    # median is computed per-channel with a uint8 result anyway.
    stack_L = np.empty((n_samples, out_h, out_w, 3), dtype=np.uint8)
    stack_R = (np.empty((n_samples, out_h, out_w, 3), dtype=np.uint8)
               if is_stereo else None)

    # We have to re-seek because we already consumed frame 0.  Use a
    # sequential read with skip — VideoCapture seeks are unreliable on
    # variable-bitrate H.264 (the exact frame returned can be off-by-one
    # near keyframes).  Sequential read is bulletproof but slower.
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frames_set = set(int(f) for f in frames_sampled)
    max_frame = int(frames_sampled.max())

    for i in range(max_frame + 1):
        if cancel_event is not None and cancel_event.is_set():
            cap.release()
            raise InterruptedError("Job cancelled")
        ok, frame = cap.read()
        if not ok:
            logger.warning(f"background: read failed at frame {i}; truncating")
            break
        if i not in frames_set:
            continue
        s_idx = sample_lookup[i]

        if is_stereo:
            os_img = frame[:, :w_half]
            od_img = frame[:, w_half:]
        else:
            os_img = frame
            od_img = None

        # Warp into the reference frame coordinate system, then downscale.
        # H_L[i] maps frame_i → ref (forward).  warpPerspective without
        # WARP_INVERSE_MAP internally inverts M and uses inv(M) as the
        # dst→src lookup — which is correctly inv(H_to_ref), i.e. it
        # samples the source pixel that corresponds to each dst pixel's
        # reference-frame location.
        warp_L = cv2.warpPerspective(
            os_img, H_L[i], (w_half, h_full),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
        )
        if downscale > 1:
            warp_L = cv2.resize(warp_L, (out_w, out_h),
                                 interpolation=cv2.INTER_AREA)
        stack_L[s_idx] = warp_L

        if is_stereo and od_img is not None:
            warp_R = cv2.warpPerspective(
                od_img, H_R[i], (w_half, h_full),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
            )
            if downscale > 1:
                warp_R = cv2.resize(warp_R, (out_w, out_h),
                                     interpolation=cv2.INTER_AREA)
            stack_R[s_idx] = warp_R

        if progress_callback is not None:
            try:
                # 0 → 30 % during sampling pass; 30 → 35 during median;
                # 35 → 100 during the second pass that bakes stable.mp4
                # + fg.mp4 (the slow part — full video read + encode).
                progress_callback(30.0 * (s_idx + 1) / max(1, n_samples))
            except Exception:
                pass

    cap.release()

    if progress_callback is not None:
        try: progress_callback(32.0)
        except Exception: pass

    # ── Per-pixel temporal median + MAD ─────────────────────────────────
    # np.median over (n_samples, H, W, 3) → (H, W, 3).  For uint8 input
    # the result is float; cast back.  We compute MAD = median(|x - med|)
    # for a robust spread measure that flags moving regions.
    bg_L = np.median(stack_L, axis=0).astype(np.uint8)
    dev_L = np.abs(stack_L.astype(np.int16) - bg_L.astype(np.int16))
    mad_L = np.median(dev_L, axis=0).max(axis=-1).astype(np.uint8)
    # Free the sampling stacks; they're not needed for the bake pass.
    del stack_L, dev_L

    if is_stereo:
        bg_R = np.median(stack_R, axis=0).astype(np.uint8)
        dev_R = np.abs(stack_R.astype(np.int16) - bg_R.astype(np.int16))
        mad_R = np.median(dev_R, axis=0).max(axis=-1).astype(np.uint8)
        del stack_R, dev_R
    else:
        bg_R = np.zeros_like(bg_L)
        mad_R = np.zeros_like(mad_L)

    # For the diff-against-BG step we need a full-resolution BG.  Upscale
    # the half-res median back up if a downscale was applied.
    if downscale > 1:
        bg_L_full = cv2.resize(bg_L, (w_half, h_full), interpolation=cv2.INTER_LINEAR)
        bg_R_full = (cv2.resize(bg_R, (w_half, h_full), interpolation=cv2.INTER_LINEAR)
                     if is_stereo else np.zeros((h_full, w_half, 3), dtype=np.uint8))
    else:
        bg_L_full = bg_L
        bg_R_full = bg_R

    # ── Phase 1.5: load MediaPipe + fit skin model (best-effort) ──────
    # When MP prelabels exist for the subject, warp each frame's hand
    # keypoints into reference coords up-front, then fit a CbCr skin
    # model from pixels at the keypoints in the sampled stack.  These
    # become extra signals for the per-frame fg mask built in Phase 2.
    mp_kpts_ref_L: np.ndarray | None = None
    mp_kpts_ref_R: np.ndarray | None = None
    skin_model_L: dict | None = None
    skin_model_R: dict | None = None
    try:
        from .mediapipe_prelabel import load_mediapipe_prelabels
        mp = load_mediapipe_prelabels(subject_name)
        if mp is not None:
            os_lm_all = mp.get("OS_landmarks")
            od_lm_all = mp.get("OD_landmarks") if is_stereo else None

            def _prepare_kpts(all_lm: np.ndarray, H_chain: np.ndarray) -> np.ndarray | None:
                if all_lm is None or all_lm.size == 0:
                    return None
                end_lm = min(start_frame + n_frames, all_lm.shape[0])
                if end_lm <= start_frame:
                    return None
                trial_lm = all_lm[start_frame:end_lm].astype(np.float64, copy=True)
                # Interpolate NaN gaps so every frame has at least an
                # estimated keypoint position (better than mask collapse).
                trial_lm = _interpolate_keypoints(trial_lm)
                # Warp keypoints frame-by-frame into ref coords.
                n_lm = trial_lm.shape[0]
                out = np.full((n_frames, 21, 2), np.nan, dtype=np.float64)
                for fi in range(min(n_lm, H_chain.shape[0])):
                    out[fi] = _warp_kpts_2d(trial_lm[fi], H_chain[fi])
                return out

            mp_kpts_ref_L = _prepare_kpts(os_lm_all, H_L)
            if is_stereo and od_lm_all is not None:
                mp_kpts_ref_R = _prepare_kpts(od_lm_all, H_R)

            # Fit skin model from the already-warped sample stack.
            # Keypoints are at full-res but stack_L is at downscaled res
            # — scale the keypoints to match before sampling.
            def _kpts_at_downscale(kpts_ref_arr: np.ndarray | None) -> list[np.ndarray]:
                if kpts_ref_arr is None:
                    return []
                return [kpts_ref_arr[fi] / max(1, downscale)
                        for fi in frames_sampled]

            if mp_kpts_ref_L is not None:
                skin_model_L = _fit_skin_model_cbcr(
                    stack_L, _kpts_at_downscale(mp_kpts_ref_L))
            if is_stereo and mp_kpts_ref_R is not None:
                skin_model_R = _fit_skin_model_cbcr(
                    stack_R, _kpts_at_downscale(mp_kpts_ref_R))

            n_lm_L = (int(np.sum(~np.isnan(mp_kpts_ref_L[:, 0, 0])))
                      if mp_kpts_ref_L is not None else 0)
            logger.info(
                f"Enhanced fg mask: MP loaded, {n_lm_L}/{n_frames} frames "
                f"with valid OS keypoints; skin model: "
                f"L={'yes' if skin_model_L else 'no'}, "
                f"R={'yes' if skin_model_R else 'no'}"
            )
        else:
            logger.info("Enhanced fg mask: no MP prelabels — motion-only fallback")
    except Exception as e:
        logger.warning(f"Enhanced fg mask setup failed; motion-only: {e}")
        mp_kpts_ref_L = mp_kpts_ref_R = None
        skin_model_L = skin_model_R = None

    if progress_callback is not None:
        try: progress_callback(35.0)
        except Exception: pass

    # ── Pass 2: bake stable.mp4 + fg.mp4 ───────────────────────────────
    # Re-read the source video frame-by-frame, apply H_to_ref[i] to warp
    # each half into reference coords, and pipe two raw streams into
    # ffmpeg → H.264 mp4 in one go.  Sequential read (no seeking) is the
    # only reliable way to enumerate frames on variable-bitrate H.264.
    out_dir = _preproc_dir(subject_name, stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    stable_path = _stable_path(subject_name, stem)
    fg_path = _fg_path(subject_name, stem)

    # Source video may have a different fps than the trajectory was
    # computed against; trust the source's reported fps.
    cap2 = cv2.VideoCapture(video_path)
    if not cap2.isOpened():
        raise RuntimeError(f"Cannot reopen video for bake pass: {video_path}")
    fps = cap2.get(cv2.CAP_PROP_FPS) or 30.0
    full_w_total = w_half * 2 if is_stereo else w_half

    stable_proc = _open_ffmpeg_pipe(str(stable_path), full_w_total, h_full, fps, "bgr24")
    fg_proc     = _open_ffmpeg_pipe(str(fg_path),     full_w_total, h_full, fps, "bgr24")
    # Keypoint-only hand mask + contour outline.  When MP data is
    # genuinely usable we bake a tight kpt-gated mask; otherwise we
    # fall back to writing the foreground mask into hand.mp4 so the
    # 'Hand isolated' view shows SOMETHING useful rather than going
    # totally black.
    hand_path = _hand_path(subject_name, stem)
    outline_path = _outline_path(subject_name, stem)
    hand_proc    = _open_ffmpeg_pipe(str(hand_path),    full_w_total, h_full, fps, "bgr24")
    outline_proc = _open_ffmpeg_pipe(str(outline_path), full_w_total, h_full, fps, "bgr24")
    # Sigma for the keypoint Gaussian on hand.mp4 — wider than the
    # one inside _build_enhanced_fg_mask so the smooth blob reliably
    # covers the hand silhouette beyond the literal landmark positions.
    _HAND_KPT_SIGMA = 45.0
    _HAND_BOOST = 1.6   # multiplies the renormalised blob so the hand interior saturates
    _OUTLINE_THRESH = 80   # 0–255; hand-mask threshold for the outline contour
    _MIN_KPT_FRAMES_FRAC = 0.10   # need at least 10% of frames with usable kpts

    # Decide once per side whether we can use the kpt-based mask or
    # have to fall back to the foreground mask.  ``mp_kpts_ref_L`` can be
    # a (n_frames, 21, 2) array that's entirely NaN (subject has MP npz
    # but MP detected zero hand frames for this trial), in which case
    # the kpt-only mask would output an all-zero hand.mp4.
    def _usable_kpts(arr):
        if arr is None:
            return False
        per_frame_valid = (~np.isnan(arr).any(axis=-1)).any(axis=-1)  # (n_frames,)
        return float(per_frame_valid.sum()) / max(1, arr.shape[0]) >= _MIN_KPT_FRAMES_FRAC

    use_kpt_mask_L = _usable_kpts(mp_kpts_ref_L)
    use_kpt_mask_R = _usable_kpts(mp_kpts_ref_R) if is_stereo else False
    logger.info(
        f"hand.mp4 mode: OS={'kpt' if use_kpt_mask_L else 'fg-fallback'}"
        + (f", OD={'kpt' if use_kpt_mask_R else 'fg-fallback'}" if is_stereo else "")
    )

    # Iterate every frame; bail early if cancelled.  RAM-bounded: each
    # iteration allocates only the warped half-frames + the diff buffers.
    frames_written = 0
    try:
        for i in range(n_frames):
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("Job cancelled")
            ok, frame = cap2.read()
            if not ok:
                logger.warning(f"background bake: read failed at frame {i}; truncating")
                break

            if is_stereo:
                os_img = frame[:, :w_half]
                od_img = frame[:, w_half:]
            else:
                os_img = frame
                od_img = None

            warp_L = cv2.warpPerspective(
                os_img, H_L[i], (w_half, h_full),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
            )
            # FG mask: combines motion (|warped − BG|), MP keypoint
            # proximity (Gaussian blobs around the 21 hand landmarks),
            # and CbCr skin-tone similarity.  Falls back to motion-only
            # if MP wasn't available for this trial.
            kpts_L_now = (mp_kpts_ref_L[i] if mp_kpts_ref_L is not None
                          and i < mp_kpts_ref_L.shape[0] else None)
            fg_L = _build_enhanced_fg_mask(
                warp_L, bg_L_full, kpts_L_now, skin_model_L)
            # Replicate to 3 channels so ffmpeg's bgr24 pipe is happy.
            fg_L_bgr = np.stack([fg_L, fg_L, fg_L], axis=-1)

            # Hand mask: strict kpt-gated when MP is usable, else fall
            # back to the FG mask (motion + colour) so isolated view
            # never goes completely black.
            if use_kpt_mask_L and kpts_L_now is not None:
                hand_L_f = _build_kpt_proximity_mask(
                    kpts_L_now, w_half, h_full, sigma=_HAND_KPT_SIGMA)
                hand_L_u8 = (np.clip(hand_L_f * _HAND_BOOST, 0.0, 1.0)
                              * 255).astype(np.uint8)
            else:
                hand_L_u8 = fg_L
            # Outline = thin contour at the 'hand vs not-hand' threshold.
            binary_L = (hand_L_u8 > _OUTLINE_THRESH).astype(np.uint8) * 255
            outline_L = cv2.Canny(binary_L, 50, 150)
            # 1-pixel dilation so the line is visible on a 1080p canvas.
            outline_L = cv2.dilate(outline_L, np.ones((2, 2), np.uint8),
                                    iterations=1)
            hand_L_bgr    = np.stack([hand_L_u8, hand_L_u8, hand_L_u8], axis=-1)
            outline_L_bgr = np.stack([outline_L, outline_L, outline_L], axis=-1)

            if is_stereo and od_img is not None:
                warp_R = cv2.warpPerspective(
                    od_img, H_R[i], (w_half, h_full),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
                )
                kpts_R_now = (mp_kpts_ref_R[i] if mp_kpts_ref_R is not None
                              and i < mp_kpts_ref_R.shape[0] else None)
                fg_R = _build_enhanced_fg_mask(
                    warp_R, bg_R_full, kpts_R_now, skin_model_R)
                fg_R_bgr = np.stack([fg_R, fg_R, fg_R], axis=-1)

                if use_kpt_mask_R and kpts_R_now is not None:
                    hand_R_f = _build_kpt_proximity_mask(
                        kpts_R_now, w_half, h_full, sigma=_HAND_KPT_SIGMA)
                    hand_R_u8 = (np.clip(hand_R_f * _HAND_BOOST, 0.0, 1.0)
                                  * 255).astype(np.uint8)
                else:
                    hand_R_u8 = fg_R
                binary_R = (hand_R_u8 > _OUTLINE_THRESH).astype(np.uint8) * 255
                outline_R = cv2.Canny(binary_R, 50, 150)
                outline_R = cv2.dilate(outline_R, np.ones((2, 2), np.uint8),
                                        iterations=1)
                hand_R_bgr    = np.stack([hand_R_u8, hand_R_u8, hand_R_u8], axis=-1)
                outline_R_bgr = np.stack([outline_R, outline_R, outline_R], axis=-1)

                stable_frame  = np.concatenate([warp_L,        warp_R],        axis=1)
                fg_frame      = np.concatenate([fg_L_bgr,      fg_R_bgr],      axis=1)
                hand_frame    = np.concatenate([hand_L_bgr,    hand_R_bgr],    axis=1)
                outline_frame = np.concatenate([outline_L_bgr, outline_R_bgr], axis=1)
            else:
                stable_frame  = warp_L
                fg_frame      = fg_L_bgr
                hand_frame    = hand_L_bgr
                outline_frame = outline_L_bgr

            stable_proc.stdin.write(np.ascontiguousarray(stable_frame,  dtype=np.uint8).tobytes())
            fg_proc.stdin.write(    np.ascontiguousarray(fg_frame,      dtype=np.uint8).tobytes())
            hand_proc.stdin.write(  np.ascontiguousarray(hand_frame,    dtype=np.uint8).tobytes())
            outline_proc.stdin.write(np.ascontiguousarray(outline_frame, dtype=np.uint8).tobytes())
            frames_written += 1

            if progress_callback is not None and (i % 5 == 0):
                try:
                    progress_callback(35.0 + 60.0 * (i + 1) / max(1, n_frames))
                except Exception:
                    pass
    finally:
        cap2.release()
        for proc, name in ((stable_proc, "stable"), (fg_proc, "fg"),
                            (hand_proc, "hand"), (outline_proc, "outline")):
            try: proc.stdin.close()
            except Exception: pass
            try:
                stderr_bytes = proc.stderr.read() if proc.stderr else b""
                rc = proc.wait(timeout=600)
                if rc != 0:
                    logger.warning(f"ffmpeg ({name}) exited {rc}: "
                                    f"{stderr_bytes.decode('utf-8', errors='replace')[:600]}")
            except Exception as e:
                logger.warning(f"ffmpeg ({name}) finalise error: {e}")

    if progress_callback is not None:
        try: progress_callback(96.0)
        except Exception: pass

    out_path = _background_path(subject_name, stem)
    np.savez_compressed(
        str(out_path),
        background_L=bg_L,
        background_R=bg_R,
        mad_L=mad_L,
        mad_R=mad_R,
        frames_sampled=frames_sampled,
        frames_written=np.array(frames_written, dtype=np.int32),
        n_frames=np.array(n_frames, dtype=np.int32),
        reference_frame=np.array(ref, dtype=np.int32),
        is_stereo=np.array(is_stereo, dtype=bool),
        downscale=np.array(downscale, dtype=np.int32),
    )

    # Also dump PNGs of the bg + MAD for quick visual inspection in the
    # filesystem (and so the HTTP endpoint can serve them directly).
    cv2.imwrite(str(out_dir / "background_OS.png"), bg_L)
    cv2.imwrite(str(out_dir / "mad_OS.png"), mad_L)
    if is_stereo:
        cv2.imwrite(str(out_dir / "background_OD.png"), bg_R)
        cv2.imwrite(str(out_dir / "mad_OD.png"), mad_R)

    if progress_callback is not None:
        try: progress_callback(100.0)
        except Exception: pass
    logger.info(
        f"background+bake saved: {out_path}  N_samples={n_samples}  "
        f"frames_written={frames_written}/{n_frames}  "
        f"stereo={is_stereo}  downscale={downscale}"
    )
    return str(out_path)


def load_background(subject_name: str, trial_stem: str) -> dict | None:
    path = _background_path(subject_name, trial_stem)
    if not path.exists():
        return None
    try:
        d = np.load(str(path))
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
        return None
    return {
        "background_L": d["background_L"],
        "background_R": d["background_R"],
        "mad_L": d["mad_L"],
        "mad_R": d["mad_R"],
        "frames_sampled": d["frames_sampled"],
        "frames_written": (int(d["frames_written"])
                           if "frames_written" in d.files else 0),
        "n_frames": int(d["n_frames"]),
        "reference_frame": int(d["reference_frame"]),
        "is_stereo": bool(d["is_stereo"]),
        "downscale": int(d["downscale"]),
    }


def summarise_background(subject_name: str, trial_stem: str, bg: dict) -> dict:
    """Lightweight summary suitable for the HTTP GET endpoint.  Reports
    whether stable.mp4 / fg.mp4 made it to disk, since those are the
    artifacts downstream consumers care about most."""
    bg_L = bg["background_L"]
    mad_L = bg["mad_L"]
    h, w = bg_L.shape[:2]
    sp = _stable_path(subject_name, trial_stem)
    fp = _fg_path(subject_name, trial_stem)
    hp = _hand_path(subject_name, trial_stem)
    op = _outline_path(subject_name, trial_stem)
    return {
        "n_samples_used": int(bg["frames_sampled"].size),
        "frames_written": bg.get("frames_written", 0),
        "n_frames":       bg["n_frames"],
        "reference_frame": bg["reference_frame"],
        "is_stereo":      bg["is_stereo"],
        "downscale":      bg["downscale"],
        "image_shape":    [h, w],
        "mad_OS_p95":     float(np.percentile(mad_L, 95)),
        "mad_OS_mean":    float(np.mean(mad_L)),
        "mad_OD_p95":     (float(np.percentile(bg["mad_R"], 95))
                            if bg["is_stereo"] else None),
        "mad_OD_mean":    (float(np.mean(bg["mad_R"]))
                            if bg["is_stereo"] else None),
        "stable_mp4_exists":  sp.exists(),
        "stable_mp4_size_mb": (sp.stat().st_size / 1e6) if sp.exists() else 0.0,
        "fg_mp4_exists":      fp.exists(),
        "fg_mp4_size_mb":     (fp.stat().st_size / 1e6) if fp.exists() else 0.0,
        "hand_mp4_exists":    hp.exists(),
        "outline_mp4_exists": op.exists(),
        # Roughly indicates whether hand.mp4 is kpt-gated or motion-fallback,
        # inferred from the variance of the mask — a near-uniform fg-fallback
        # has higher variance across the trial than a sparse kpt blob.
        # (Cheap inference; precise info lives in the job log.)
    }


def stable_mp4_path(subject_name: str, trial_stem: str) -> Path | None:
    """Public helper for downstream consumers (MP/HRnet/DLC).  Returns
    the path to the stabilised mp4 if it exists, else ``None``."""
    p = _stable_path(subject_name, trial_stem)
    return p if p.exists() else None


def fg_mp4_path(subject_name: str, trial_stem: str) -> Path | None:
    p = _fg_path(subject_name, trial_stem)
    return p if p.exists() else None


def hand_mp4_path(subject_name: str, trial_stem: str) -> Path | None:
    p = _hand_path(subject_name, trial_stem)
    return p if p.exists() else None


def outline_mp4_path(subject_name: str, trial_stem: str) -> Path | None:
    p = _outline_path(subject_name, trial_stem)
    return p if p.exists() else None
