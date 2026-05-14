"""Stabilisation + on-demand hand-boundary extraction.

One bake per trial produces ``stable.mp4`` and ``background.npz``.
The hand boundary is computed on demand per frame by
:func:`compute_outline_frame`, which reads from those two artifacts
and returns a closed polygon -- no global fg.mp4 / outline.mp4 mp4s
are written.

``stable.mp4`` -- every source frame warped into the reference frame's
coordinate system using the camera trajectory.  The static scene stays
put; only the hand (and any other moving parts) appears to move.  Same
resolution and layout as the source video.

``background.npz`` -- temporal median of stabilised samples (with the
hand masked out via the dilated MP skeleton), plus the fitted CbCr
skin model and the MP-ref keypoints, so downstream callers can
reconstruct everything needed for per-frame mask work without
re-loading MP or re-fitting the model.

Inputs:
    Camera trajectory (``camera_trajectory.npz``) must already exist --
    compute it via the *Compute Trajectory* button first.

Outputs in ``<dlc>/<subject>/preproc/<stem>/``:
    stable.mp4              stabilised source video
    background.npz          BG image + MAD + skin model + MP-ref kpts
    background_OS.png       BG image, OS half (full-res preview)
    background_OD.png       BG image, OD half (stereo only)
    mad_OS.png, mad_OD.png  MAD heat-maps for quality inspection

``H_to_ref[i]`` maps frame-i pixel coords -> reference-frame pixel
coords (forward direction).  We pass it to ``warpPerspective`` *without*
``WARP_INVERSE_MAP`` -- OpenCV then internally inverts it and samples
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


_LIVE_FFMPEG_PROCS: "list[subprocess.Popen]" = []
"""Module-level registry of currently-active ffmpeg encoder subprocesses.
Used by an ``atexit`` handler below so that if the parent Python process
is shut down via SIGTERM (or finishes abruptly) the ffmpeg children get
killed immediately instead of being left as orphans writing to
half-finished mp4 files."""


def _kill_live_ffmpegs() -> None:
    for p in list(_LIVE_FFMPEG_PROCS):
        try:
            if p.poll() is None:
                p.kill()
        except Exception:
            pass


import atexit as _atexit
_atexit.register(_kill_live_ffmpegs)


def _kill_orphan_ffmpegs_for_dir(preproc_dir: Path) -> int:
    """Kill any ffmpeg process whose command line writes into
    ``preproc_dir``.

    Called at the start of every Stabilise / Foreground run so that
    even when a previous bake was hard-killed (SIGKILL, OOM, panic)
    and atexit didn't fire, the new bake doesn't race against a
    surviving encoder writing to the same mp4.  Returns the number of
    processes killed; logs a warning if any were found.
    """
    if os.name == "nt":
        return 0    # ps/SIGKILL path is POSIX-only; fine for our use case
    target = str(preproc_dir.resolve())
    try:
        out = subprocess.run(
            ["ps", "-axo", "pid=,command="],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except Exception:
        return 0
    killed = 0
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        if "ffmpeg" not in line:
            continue
        if target not in line:
            continue
        try:
            pid_str, _ = line.split(None, 1)
            pid = int(pid_str)
        except (ValueError, IndexError):
            continue
        # Don't kill ourselves -- we just registered new ffmpegs in
        # _LIVE_FFMPEG_PROCS, so leave those alone.
        if any(p.pid == pid for p in _LIVE_FFMPEG_PROCS):
            continue
        try:
            os.kill(pid, 9)
            killed += 1
        except Exception:
            pass
    if killed:
        logger.warning(
            f"killed {killed} orphan ffmpeg(s) writing to {target} "
            "before starting new bake")
    return killed


def _open_ffmpeg_pipe(output_path: str, width: int, height: int,
                      fps: float, pix_fmt_in: str = "bgr24") -> subprocess.Popen:
    """Open an ffmpeg subprocess that consumes raw frames on stdin and
    writes a yuv420p / H.264 mp4 to ``output_path``.

    The Popen is registered in ``_LIVE_FFMPEG_PROCS`` so the atexit
    handler can SIGKILL it if Python exits before the bake finishes a
    clean ``stdin.close() / wait()`` -- prevents the "orphan ffmpegs
    writing to the same mp4" failure mode that corrupted bakes earlier
    in development.
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
    proc = subprocess.Popen([
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
    _LIVE_FFMPEG_PROCS.append(proc)
    return proc


def _retire_ffmpeg(proc: subprocess.Popen) -> None:
    """Drop ``proc`` from the live-ffmpeg registry once the bake's
    ``finally:`` block has finished closing its stdin and waiting for
    the encoder to flush."""
    try:
        _LIVE_FFMPEG_PROCS.remove(proc)
    except ValueError:
        pass


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


# MediaPipe finger chains (joint indices).  Used to draw filled
# segments between adjacent joints so the kpt-hand-region covers each
# finger as a smooth shape rather than five isolated dots.
_FINGER_CHAINS = [
    [0, 1, 2, 3, 4],     # thumb : wrist → CMC → MCP → IP → TIP
    [0, 5, 6, 7, 8],     # index
    [0, 9, 10, 11, 12],  # middle
    [0, 13, 14, 15, 16], # ring
    [0, 17, 18, 19, 20], # pinky
]


def _build_kpt_hand_region(
    kpts_ref: np.ndarray,
    w: int, h: int,
    stamp_radius: int = 10,
    smooth_sigma: float = 4.0,
    extend_forearm_px: int = 0,
) -> np.ndarray:
    """Tight hand-skeleton silhouette from MP keypoints.

    Matches the shape of the deidentify hand mask (~1 cm dilation of
    the MP skeleton).  Procedure:

    1. Stamp a filled circle of radius ``stamp_radius`` at every valid
       MP keypoint.
    2. Draw a thick line (width = ``2 * stamp_radius``) between every
       pair of adjacent joints along each finger chain.  This actually
       traces the skeleton segments instead of scattering midpoint
       stamps that leave bone-thin gaps.
    3. Draw thick lines along the MCP arc (joints 1-5-9-13-17) to
       close the palm.
    4. Gaussian-blur with a small sigma to feather the edges without
       expanding the silhouette much.

    Returns a (h, w) float32 mask in [0, 1].  After the binary
    threshold in the caller (``> 0.5``), the gate ends up roughly
    ``stamp_radius + smooth_sigma`` pixels from each skeleton point --
    so 14-15 px at the default settings, matching the deidentify
    defaults of ``hand_mask_radius=10`` + ``hand_smooth=7``.
    """
    canvas = np.zeros((h, w), dtype=np.uint8)
    valid = ~np.isnan(kpts_ref).any(axis=-1)   # (21,) bool
    if not valid.any():
        return canvas.astype(np.float32)

    def _to_int(p):
        return (int(round(float(p[0]))), int(round(float(p[1]))))

    by_joint: dict[int, np.ndarray] = {
        int(j): kpts_ref[int(j)] for j in np.where(valid)[0]
    }

    # 1. Filled circle at every valid joint.
    for j, p in by_joint.items():
        xi, yi = _to_int(p)
        if 0 <= xi < w and 0 <= yi < h:
            cv2.circle(canvas, (xi, yi), stamp_radius, 255, thickness=-1)

    # 2. Thick line along each adjacent pair in each finger chain.
    line_thick = max(1, stamp_radius * 2)
    for chain in _FINGER_CHAINS:
        for ci in range(len(chain) - 1):
            a = by_joint.get(chain[ci]); b = by_joint.get(chain[ci + 1])
            if a is None or b is None:
                continue
            cv2.line(canvas, _to_int(a), _to_int(b),
                      color=255, thickness=line_thick, lineType=cv2.LINE_AA)

    # 3. Palm: thick lines along the MCP arc (thumb-CMC -> index-MCP ->
    #    middle-MCP -> ring-MCP -> pinky-MCP).  Closes the palm fill
    #    without needing per-frame triangle math.
    palm_chain = [1, 5, 9, 13, 17]
    for i in range(len(palm_chain) - 1):
        a = by_joint.get(palm_chain[i]); b = by_joint.get(palm_chain[i + 1])
        if a is None or b is None:
            continue
        cv2.line(canvas, _to_int(a), _to_int(b),
                  color=255, thickness=line_thick, lineType=cv2.LINE_AA)

    # 3a. Optional forearm extension.  Used at Stabilise-time so the BG
    #     median masks the distal forearm out alongside the hand;
    #     compute_outline_frame leaves this off (extend_forearm_px=0)
    #     so the visible polygon still ends at the wrist.
    if extend_forearm_px > 0 and 0 in by_joint:
        wrist = by_joint[0].astype(np.float64)
        mcp_pts = [by_joint[j].astype(np.float64) for j in (5, 9, 13, 17)
                    if j in by_joint]
        if mcp_pts:
            mcp_centroid = np.mean(mcp_pts, axis=0)
            forearm_dir = wrist - mcp_centroid
            n = float(np.linalg.norm(forearm_dir))
            if n > 1e-6:
                forearm_dir = forearm_dir / n
                forearm_end = wrist + forearm_dir * float(extend_forearm_px)
                # Forearm is wider than a finger -- 1.5x line thickness.
                forearm_thick = max(line_thick, int(round(line_thick * 1.5)))
                cv2.line(canvas, _to_int(wrist), _to_int(forearm_end),
                          color=255, thickness=forearm_thick,
                          lineType=cv2.LINE_AA)

    # 4. Light smoothing for edge feather.
    if smooth_sigma > 0:
        smoothed = cv2.GaussianBlur(canvas.astype(np.float32), (0, 0),
                                      float(smooth_sigma))
        return np.clip(smoothed / 255.0, 0.0, 1.0)
    return canvas.astype(np.float32) / 255.0


def _bg_edge_map(bg_full: np.ndarray) -> np.ndarray:
    """Normalised Sobel-magnitude of the background image.

    Strong edges in the BG (table seams, monitor bezels, high-contrast
    decor) produce huge |frame - BG| activations the instant the hand
    crosses them, even though the actual local pixel change is small.
    Subtracting this map from the motion score wipes those edge-only
    activations while leaving genuine smooth skin/background transitions
    intact.  Returns a (h, w) float32 in [0, 1].
    """
    g = cv2.cvtColor(bg_full, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sx * sx + sy * sy)
    # 99th percentile normalisation -- a few extreme pixels don't blow
    # out the rest of the map.
    p = float(np.percentile(mag, 99.0))
    if p <= 1e-6:
        return np.zeros_like(mag)
    return np.clip(mag / p, 0.0, 1.0)


def _smooth_binary_mask(
    binary: np.ndarray,
    *,
    min_width_px: int = 3,
) -> np.ndarray:
    """Remove thin webs and tiny speckles without shaving fingertips.

    1. Morphological opening with an elliptical kernel of radius
       ``min_width_px`` -- erodes everything that's narrower than the
       kernel, then dilates back.  Webs/strands narrower than
       ``2 * min_width_px`` disappear; fingertips (typically 15+ px
       wide at 1080p) survive because their interior has pixels well
       inside the kernel.
    2. Keep only the LARGEST connected component (the hand).  Any
       remaining specks elsewhere -- a moving piece of paper, a head
       glimpse -- get dropped.

    Input/output: (h, w) uint8 with values 0 or 255.
    """
    if not binary.any():
        return binary
    r = max(1, int(min_width_px))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    if n_lbl <= 1:
        return opened
    # stats[0] is the background label -- skip it.
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(lbl == largest, np.uint8(255), np.uint8(0))


def _build_image_aware_hand_mask(
    warped_frame: np.ndarray,
    bg_full: np.ndarray,
    kpts_ref: np.ndarray,
    skin_model: dict | None,
    *,
    stamp_radius: int = 10,           # matches deidentify hand_mask_radius default
    region_sigma: float = 4.0,        # smaller than before -- keeps the gate tight
    region_thresh: float = 0.5,       # binary cut at the silhouette boundary
    motion_norm: float = 50.0,
    image_thresh: float = 0.30,       # require stronger image signal inside the gate
    feather_sigma: float = 1.5,
    bg_edge_norm: np.ndarray | None = None,
    edge_subtract_strength: float = 0.7,
    min_width_px: int = 3,
) -> np.ndarray:
    """Hand mask that follows the actual hand pixels inside a deidentify-style
    keypoint-region gate.

    Pipeline per frame:
      1. ``kpt_region`` = filled silhouette from MP keypoints + finger
         chains, smoothed (see :func:`_build_kpt_hand_region`).  This is
         the GEOMETRIC bound -- "anywhere the hand might plausibly be".
      2. ``image_score`` = per-pixel max of:
           - motion: |warped - background| / motion_norm, clipped to [0,1],
             with the background edge map subtracted off so BG seams
             don't leak through.
           - skin colour: exp(-Mahalanobis(CbCr) / scale) if a skin model
             was fit.
      3. Multiply ``kpt_region`` x ``image_score``, threshold, feather edges,
         then a width-aware prune to drop thin webs / speckles.

    Result: a binary-ish uint8 mask, dark everywhere except where the
    image actually looks like a hand inside the keypoint silhouette.
    The boundary follows real pixel content, not just keypoint geometry.

    Returns (h, w) uint8 in [0, 255].
    """
    H, W = warped_frame.shape[:2]

    # 1. Geometric gate -- tight binary silhouette ~1 cm out from the
    #    MP skeleton (matching the deidentify hand mask in shape).
    kpt_region = _build_kpt_hand_region(kpts_ref, W, H,
                                          stamp_radius=stamp_radius,
                                          smooth_sigma=region_sigma)
    kpt_gate = (kpt_region > region_thresh).astype(np.float32)

    # 2a. Motion, with bg-edge penalty.  Pixels that are bright in the
    #     BG edge map get their motion suppressed -- a hand crossing a
    #     table seam shouldn't light up the whole seam.
    motion = np.abs(warped_frame.astype(np.int16) - bg_full.astype(np.int16))
    motion = motion.max(axis=-1).astype(np.float32) / float(motion_norm)
    if bg_edge_norm is not None and edge_subtract_strength > 0:
        motion = motion - float(edge_subtract_strength) * bg_edge_norm
    np.clip(motion, 0.0, 1.0, out=motion)

    # 2b. Colour similarity (when a skin model was fit).
    if skin_model is not None:
        color = _color_similarity(warped_frame, skin_model)
        image_score = np.maximum(motion, color)
    else:
        image_score = motion

    # 3. Gate + threshold + smoothness prune + feather.
    gated = kpt_gate * image_score
    binary = (gated > image_thresh).astype(np.uint8) * 255
    binary = _smooth_binary_mask(binary, min_width_px=min_width_px)

    if feather_sigma > 0:
        feathered = cv2.GaussianBlur(binary, (0, 0), float(feather_sigma))
    else:
        feathered = binary
    out = (feathered.astype(np.float32) * kpt_gate).clip(0, 255).astype(np.uint8)
    return out


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


def _build_fg_mask(
    warped_frame: np.ndarray,
    bg_full: np.ndarray,
    motion_norm: float = 60.0,
    bg_edge_norm: np.ndarray | None = None,
    edge_subtract_strength: float = 0.7,
) -> np.ndarray:
    """Raw motion mask: per-pixel ``|warped_frame - bg_full|`` scaled to
    a uint8 in [0, 255].

    motion_norm: the |frame-BG| value that maps to 1.0.

    bg_edge_norm: optional Sobel-magnitude map of the BG, normalised to
    [0,1].  When present, motion is reduced by
    ``edge_subtract_strength * bg_edge_norm`` so BG seams don't
    dominate when the hand crosses them.  This is a correction (removes
    a known false signal), not an enhancement.

    Previously this function also added an MP keypoint Gaussian and a
    CbCr skin-colour boost.  Both have been removed: the downstream
    pipeline already gates by MP, so adding non-motion signals here
    distorts what fg.mp4 represents.
    """
    motion = np.abs(warped_frame.astype(np.int16) - bg_full.astype(np.int16))
    motion = motion.max(axis=-1).astype(np.float32) / motion_norm
    if bg_edge_norm is not None and edge_subtract_strength > 0:
        motion = motion - float(edge_subtract_strength) * bg_edge_norm
    np.clip(motion, 0.0, 1.0, out=motion)
    return (motion * 255.0).astype(np.uint8)


def compute_stable(
    subject_name: str,
    trial_idx: int,
    progress_callback=None,
    cancel_event=None,
    max_samples: int = _DEFAULT_MAX_SAMPLES,
    downscale:   int = _DEFAULT_DOWNSCALE,
    dilation_px: int = 14,
) -> str:
    """Stage 1 of the preproc bake: produce stable.mp4 + background.npz.

    Requires a saved camera trajectory; raises ``RuntimeError`` otherwise.
    The companion :func:`compute_foreground` reads from the artifacts
    written here and produces fg.mp4 + outline.mp4 in a separate, much
    cheaper pass.

    ``dilation_px`` here only sizes the BG hand-mask used during the
    masked median -- it does NOT bake any hand-mask geometry into
    stable.mp4 itself.  Foreground re-runs can choose a different
    dilation without re-running this step.
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
    start_frame = int(trial.get("start_frame", 0))

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

    # ── Phase 0: load MP prelabels + warp to ref coords ──────────────
    # We need the per-sample-frame hand region during the sampling pass
    # so we can MASK OUT the hand from the temporal median.  Without
    # this step, near-stationary subjects bake the palm right into the
    # background -- exactly what produces the false shadow edges that
    # bleed into fg.mp4 and the outline.  Skin-model fitting needs the
    # warped stack and waits until Phase 1.5 below.
    mp_kpts_ref_L: np.ndarray | None = None
    mp_kpts_ref_R: np.ndarray | None = None
    try:
        from .mediapipe_prelabel import load_mediapipe_prelabels
        mp = load_mediapipe_prelabels(subject_name)
        if mp is not None:
            os_lm_all = mp.get("OS_landmarks")
            od_lm_all = mp.get("OD_landmarks") if is_stereo else None

            def _prepare_kpts(all_lm, H_chain):
                if all_lm is None or all_lm.size == 0:
                    return None
                end_lm = min(start_frame + n_frames, all_lm.shape[0])
                if end_lm <= start_frame:
                    return None
                trial_lm = all_lm[start_frame:end_lm].astype(np.float64, copy=True)
                trial_lm = _interpolate_keypoints(trial_lm)
                n_lm = trial_lm.shape[0]
                out = np.full((n_frames, 21, 2), np.nan, dtype=np.float64)
                for fi in range(min(n_lm, H_chain.shape[0])):
                    out[fi] = _warp_kpts_2d(trial_lm[fi], H_chain[fi])
                return out

            mp_kpts_ref_L = _prepare_kpts(os_lm_all, H_L)
            if is_stereo and od_lm_all is not None:
                mp_kpts_ref_R = _prepare_kpts(od_lm_all, H_R)
    except Exception as e:
        logger.warning(f"MP load for bg masking failed; bg will include hand: {e}")
        mp_kpts_ref_L = mp_kpts_ref_R = None

    # Allocate the warped-frame stacks.  uint8 keeps RAM minimal; the
    # median is computed per-channel with a uint8 result anyway.
    stack_L = np.empty((n_samples, out_h, out_w, 3), dtype=np.uint8)
    stack_R = (np.empty((n_samples, out_h, out_w, 3), dtype=np.uint8)
               if is_stereo else None)
    # Per-sample-frame hand-region mask (True = drop from BG median).
    # Built from the dilated MP skeleton, downscaled.  Size matches the
    # downscaled stack so we can mask the median directly without a
    # second resize.
    hand_mask_L = np.zeros((n_samples, out_h, out_w), dtype=bool)
    hand_mask_R = (np.zeros((n_samples, out_h, out_w), dtype=bool)
                   if is_stereo else None)
    # Skeleton stamp radius for BG masking, in downscaled pixels.  Use
    # the same dilation as the hand-mask gate so the patch removed from
    # BG matches the patch added back in by the fg mask -- otherwise the
    # outline ends up tracing the seam between the two.
    _BG_MASK_STAMP_DOWN = max(2, int(dilation_px) // max(1, downscale))
    # Extend the BG-mask stamp down the forearm so the distal forearm
    # also gets masked out of the median.  Outline-time gate doesn't
    # use this -- the visible polygon still cuts off at the wrist.
    # ~100 full-res px ~= a few cm of forearm in a typical 1080p clip.
    _BG_MASK_FOREARM_DOWN = max(0, 100 // max(1, downscale))

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
        if mp_kpts_ref_L is not None and i < mp_kpts_ref_L.shape[0]:
            kpts_L_down = mp_kpts_ref_L[i] / max(1, downscale)
            region_L = _build_kpt_hand_region(
                kpts_L_down, out_w, out_h,
                stamp_radius=_BG_MASK_STAMP_DOWN, smooth_sigma=2.0,
                extend_forearm_px=_BG_MASK_FOREARM_DOWN)
            hand_mask_L[s_idx] = region_L > 0.5

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
            if (mp_kpts_ref_R is not None
                    and i < mp_kpts_ref_R.shape[0]):
                kpts_R_down = mp_kpts_ref_R[i] / max(1, downscale)
                region_R = _build_kpt_hand_region(
                    kpts_R_down, out_w, out_h,
                    stamp_radius=_BG_MASK_STAMP_DOWN, smooth_sigma=2.0,
                    extend_forearm_px=_BG_MASK_FOREARM_DOWN)
                hand_mask_R[s_idx] = region_R > 0.5

        if progress_callback is not None:
            try:
                # Re-allocated 2026-05-13 to match real wall-clock split:
                #   sampling pass         : 0  → 10 %  (~n_samples frames,
                #                                       downscaled, no encode)
                #   median + MP setup     : 10 → 12 %  (numpy median + a few
                #                                       small fits, milliseconds)
                #   bake pass             : 12 → 97 %  (every frame, full-res
                #                                       warp + 4 ffmpeg pipes —
                #                                       this is the long phase)
                #   save npz + PNGs       : 97 → 100 %
                # The previous 0–30 sampling allocation reached 30 % in
                # ~10 % of the total time, then crawled to 100 % across
                # the rest, making the progress bar misleading.
                progress_callback(10.0 * (s_idx + 1) / max(1, n_samples))
            except Exception:
                pass

    cap.release()

    if progress_callback is not None:
        try: progress_callback(11.0)
        except Exception: pass

    # ── Per-pixel temporal median + MAD ─────────────────────────────────
    # np.ma.median over (n_samples, H, W, 3) -> (H, W, 3).  When MP
    # gave us a per-frame hand mask we exclude those pixels per-frame,
    # which keeps the near-stationary palm out of the BG median.  Pixels
    # that were masked in EVERY sample frame fall back to the plain
    # median so the BG is never undefined.  MAD = median(|x - med|) is a
    # robust spread measure used as a quality heat-map.
    def _masked_median(stack: np.ndarray, hand_mask: np.ndarray) -> np.ndarray:
        """Per-pixel temporal median ignoring hand-covered pixels.

        Pixels that were masked in EVERY sample frame (deep palm
        interior on a near-stationary hand) get filled with the GLOBAL
        median of non-hand pixels instead of the plain temporal median.
        That global median is a representative "table colour" -- nothing
        like skin -- so the per-frame fg mask still lights up across
        the whole hand silhouette and Canny picks up only the external
        boundary.  Losing the interior is fine: we only care about the
        edge of the hand.
        """
        if not hand_mask.any():
            return np.median(stack, axis=0).astype(np.uint8)
        m4 = np.broadcast_to(hand_mask[..., None], stack.shape)
        masked = np.ma.masked_array(stack, mask=m4)
        med = np.ma.median(masked, axis=0)
        # Global non-hand median (one BGR triplet) as the fill for
        # always-masked pixels.
        valid_samples = stack[~hand_mask]
        if valid_samples.size > 0:
            fill_color = np.median(valid_samples, axis=0)
        else:
            fill_color = np.array([128, 128, 128], dtype=np.float32)
        bg = np.where(np.ma.getmaskarray(med),
                       fill_color.reshape(1, 1, 3),
                       np.ma.getdata(med))
        return bg.astype(np.uint8)

    bg_L = _masked_median(stack_L, hand_mask_L)
    dev_L = np.abs(stack_L.astype(np.int16) - bg_L.astype(np.int16))
    mad_L = np.median(dev_L, axis=0).max(axis=-1).astype(np.uint8)
    del dev_L   # stack_L kept alive for Phase 1.5 skin-model fit

    if is_stereo:
        bg_R = _masked_median(stack_R, hand_mask_R)
        dev_R = np.abs(stack_R.astype(np.int16) - bg_R.astype(np.int16))
        mad_R = np.median(dev_R, axis=0).max(axis=-1).astype(np.uint8)
        del dev_R
    else:
        bg_R = np.zeros_like(bg_L)
        mad_R = np.zeros_like(mad_L)
    # Pixels that were under the dilated MP gate in EVERY sample frame.
    # These are guaranteed-hand: the BG median had to fall back to the
    # global non-skin colour for them.  Save the map (downscaled, bool)
    # so compute_outline_frame can force them inside every boundary --
    # otherwise per-frame Otsu can carve them out when the hand pose
    # makes them blend with the synthetic BG fill.
    always_hand_L_down = hand_mask_L.all(axis=0)
    always_hand_R_down = (hand_mask_R.all(axis=0)
                           if hand_mask_R is not None else None)
    del hand_mask_L
    if hand_mask_R is not None:
        del hand_mask_R

    # For the diff-against-BG step we need a full-resolution BG.  Upscale
    # the half-res median back up if a downscale was applied.
    if downscale > 1:
        bg_L_full = cv2.resize(bg_L, (w_half, h_full), interpolation=cv2.INTER_LINEAR)
        bg_R_full = (cv2.resize(bg_R, (w_half, h_full), interpolation=cv2.INTER_LINEAR)
                     if is_stereo else np.zeros((h_full, w_half, 3), dtype=np.uint8))
    else:
        bg_L_full = bg_L
        bg_R_full = bg_R

    # Background edge map -- subtracted from motion later so high-
    # contrast BG seams don't leak through the hand mask.
    bg_edge_L = _bg_edge_map(bg_L_full)
    bg_edge_R = _bg_edge_map(bg_R_full) if is_stereo else None

    # ── Phase 1.5: fit CbCr skin model from sampled stack ────────────
    # MP keypoints (already warped to ref in Phase 0) tell us where
    # skin pixels live in each sample frame.  We collect those pixels
    # and fit a Gaussian in CbCr space so per-frame fg masks can use
    # colour similarity alongside motion.  Best-effort: if any of this
    # fails the bake just falls back to motion-only masks.
    skin_model_L: dict | None = None
    skin_model_R: dict | None = None
    try:
        def _kpts_at_downscale(kpts_ref_arr):
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
            f"Enhanced fg mask: {n_lm_L}/{n_frames} frames with valid OS "
            f"keypoints; skin model: "
            f"L={'yes' if skin_model_L else 'no'}, "
            f"R={'yes' if skin_model_R else 'no'}"
        )
    except Exception as e:
        logger.warning(f"Skin-model fit failed; motion-only fg: {e}")
        skin_model_L = skin_model_R = None

    # Sampling stacks are large (n_samples × H × W × 3) — free them now
    # that the skin model has been fit; the bake pass below reads frames
    # one at a time from disk.
    try: del stack_L
    except NameError: pass
    try: del stack_R
    except NameError: pass

    if progress_callback is not None:
        try: progress_callback(12.0)
        except Exception: pass

    # ── Pass 2: bake stable.mp4 only ──────────────────────────────────
    # Re-read every source frame, apply H_to_ref[i], pipe a raw stream
    # into ffmpeg.  fg.mp4 + outline.mp4 are produced by the separate
    # ``compute_foreground`` step, which reads from stable.mp4 instead
    # of warping again.  Splitting the stages lets the user iterate on
    # mask/outline parameters cheaply without re-running the heavy
    # warp+median pass.
    out_dir = _preproc_dir(subject_name, stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    stable_path = _stable_path(subject_name, stem)
    # Wipe any orphan ffmpeg from a previous hard-killed bake before
    # opening our own pipes to the same files.
    _kill_orphan_ffmpegs_for_dir(out_dir)

    cap2 = cv2.VideoCapture(video_path)
    if not cap2.isOpened():
        raise RuntimeError(f"Cannot reopen video for bake pass: {video_path}")
    fps = cap2.get(cv2.CAP_PROP_FPS) or 30.0
    full_w_total = w_half * 2 if is_stereo else w_half

    stable_proc = _open_ffmpeg_pipe(str(stable_path), full_w_total, h_full, fps, "bgr24")
    # Stale fg/outline/hand from previous bakes are now out of sync
    # with the freshly-stabilised frames -- unlink them so the UI
    # shows "needs foreground" until that step is re-run.
    for p in (_fg_path(subject_name, stem),
              _outline_path(subject_name, stem),
              _hand_path(subject_name, stem)):
        try: p.unlink()
        except FileNotFoundError: pass

    frames_written = 0
    try:
        for i in range(n_frames):
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("Job cancelled")
            ok, frame = cap2.read()
            if not ok:
                logger.warning(f"stable bake: read failed at frame {i}; truncating")
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
            if is_stereo and od_img is not None:
                warp_R = cv2.warpPerspective(
                    od_img, H_R[i], (w_half, h_full),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
                )
                stable_frame = np.concatenate([warp_L, warp_R], axis=1)
            else:
                stable_frame = warp_L

            stable_proc.stdin.write(
                np.ascontiguousarray(stable_frame, dtype=np.uint8).tobytes())
            frames_written += 1

            if progress_callback is not None and (i % 5 == 0):
                try:
                    progress_callback(12.0 + 85.0 * (i + 1) / max(1, n_frames))
                except Exception:
                    pass
    finally:
        cap2.release()
        try: stable_proc.stdin.close()
        except Exception: pass
        try:
            stderr_bytes = stable_proc.stderr.read() if stable_proc.stderr else b""
            rc = stable_proc.wait(timeout=600)
            if rc != 0:
                logger.warning(f"ffmpeg (stable) exited {rc}: "
                                f"{stderr_bytes.decode('utf-8', errors='replace')[:600]}")
        except Exception as e:
            logger.warning(f"ffmpeg (stable) finalise error: {e}")
        finally:
            _retire_ffmpeg(stable_proc)

    if progress_callback is not None:
        try: progress_callback(97.0)
        except Exception: pass

    out_path = _background_path(subject_name, stem)
    save_kwargs = dict(
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
    # Save MP-ref keypoints and skin model so compute_foreground can run
    # without re-loading MP / re-fitting the model.
    if mp_kpts_ref_L is not None:
        save_kwargs["mp_kpts_ref_L"] = mp_kpts_ref_L.astype(np.float32)
    if mp_kpts_ref_R is not None:
        save_kwargs["mp_kpts_ref_R"] = mp_kpts_ref_R.astype(np.float32)
    if skin_model_L is not None:
        save_kwargs["skin_model_L_mean"]    = skin_model_L["mean"]
        save_kwargs["skin_model_L_cov_inv"] = skin_model_L["cov_inv"]
    if skin_model_R is not None:
        save_kwargs["skin_model_R_mean"]    = skin_model_R["mean"]
        save_kwargs["skin_model_R_cov_inv"] = skin_model_R["cov_inv"]
    # Always-hand maps so the live outline can force these pixels
    # inside the boundary regardless of per-frame motion.
    if always_hand_L_down is not None:
        save_kwargs["always_hand_L_down"] = always_hand_L_down
    if always_hand_R_down is not None:
        save_kwargs["always_hand_R_down"] = always_hand_R_down
    np.savez_compressed(str(out_path), **save_kwargs)

    cv2.imwrite(str(out_dir / "background_OS.png"), bg_L)
    cv2.imwrite(str(out_dir / "mad_OS.png"), mad_L)
    if is_stereo:
        cv2.imwrite(str(out_dir / "background_OD.png"), bg_R)
        cv2.imwrite(str(out_dir / "mad_OD.png"), mad_R)

    if progress_callback is not None:
        try: progress_callback(100.0)
        except Exception: pass
    logger.info(
        f"stable bake saved: {out_path}  N_samples={n_samples}  "
        f"frames_written={frames_written}/{n_frames}  "
        f"stereo={is_stereo}  downscale={downscale}"
    )
    return str(out_path)



def _encode_fg_png(motion: np.ndarray, gate: np.ndarray) -> dict | None:
    """Pack a small RGBA PNG of the foreground heatmap inside the gate.

    Returns ``{b64, bbox: [x0,y0,x1,y1]}`` or None if the gate is empty.
    The image covers the WHOLE dilated MP gate -- alpha is flat 255
    inside the gate and 0 outside, so the user sees the full region
    coloured by motion intensity (low motion = dark blue JET, high
    motion = red).  Overall fill opacity is left to the client so a
    slider can dial visibility against the underlying frame.

    The crop keeps the payload small (typically 20-80 KB per side for
    a 200 px hand crop) and lets the UI position the image directly
    via the bbox without resampling.
    """
    import base64
    ys, xs = np.where(gate > 0)
    if ys.size == 0:
        return None
    y0 = int(ys.min()); y1 = int(ys.max() + 1)
    x0 = int(xs.min()); x1 = int(xs.max() + 1)
    motion_crop = motion[y0:y1, x0:x1].copy()
    gate_crop = gate[y0:y1, x0:x1]
    motion_crop[gate_crop == 0] = 0
    color = cv2.applyColorMap(motion_crop, cv2.COLORMAP_JET)
    # Flat alpha: 255 inside the gate, 0 outside.  Decouples motion
    # intensity (encoded by colour) from visibility (controlled by the
    # client's opacity slider), so low-motion BG regions inside the
    # gate still appear as dark-blue fill instead of vanishing.
    alpha = (gate_crop > 0).astype(np.uint8) * 255
    rgba = np.concatenate([color, alpha[..., None]], axis=-1)
    ok, png = cv2.imencode('.png', rgba)
    if not ok:
        return None
    return {
        "b64": base64.b64encode(png.tobytes()).decode('ascii'),
        "bbox": [x0, y0, x1, y1],
    }


def compute_outline_frame(
    subject_name: str,
    trial_idx: int,
    frame: int,
    dilation_px: int = 14,
    close_radius_px: int = 5,
    simplify_eps_px: float = 0.5,
    include_fg: bool = False,
) -> dict:
    """On-demand hand boundary for one frame.

    Replaces the old compute_foreground bake: instead of writing an
    fg.mp4 + outline.mp4 over the entire trial, this function returns
    the contour of the hand for a single ``frame``, computed only
    inside the dilated MP gate.  The UI calls it lazily as the user
    scrubs frames or moves the dilation slider.

    Pipeline per side:
      1. Build dilated MP gate (filled silhouette of joints + chains).
      2. Compute |stable_frame - BG| inside the gate.
      3. Otsu-threshold the gate-restricted motion (bimodal: BG vs hand).
      4. OR with the always-hand map (pixels covered by the MP gate in
         every sample frame during Stabilise -- their BG fell back to
         a synthetic colour, so we force them to count as hand always).
      5. Morphological close (disk radius ``close_radius_px``) to fill
         palm-interior holes where skin happens to match BG.
      6. Keep only the largest connected component (drop speckles).
      7. Extract the outermost contour and simplify via Douglas-Peucker
         (``simplify_eps_px`` -- 0.5 keeps ~10x more points than 3.0,
         giving a smooth boundary while still removing 1 px jitter).

    Returns:
      ``{frame, is_stereo, OS: [[x,y],...], OD: [[x,y],...] | None}``.
      Each contour is a closed polygon in reference-frame pixel
      coordinates (the same space stable.mp4 frames live in).
    """
    from .video import build_trial_map

    tmap = build_trial_map(subject_name)
    if trial_idx < 0 or trial_idx >= len(tmap):
        raise ValueError(f"trial_idx {trial_idx} out of range ({len(tmap)} trials)")
    trial = tmap[trial_idx]
    stem = trial["trial_name"]

    bg_path = _background_path(subject_name, stem)
    if not bg_path.exists():
        raise RuntimeError(
            f"No background.npz for {subject_name}/{stem} -- "
            "run Stabilise first.")
    d = np.load(str(bg_path))
    bg_L = d["background_L"]
    is_stereo = bool(d["is_stereo"])
    bg_R = d["background_R"] if is_stereo else None
    downscale = int(d["downscale"])
    n_frames = int(d["n_frames"])
    mp_kpts_ref_L = d["mp_kpts_ref_L"] if "mp_kpts_ref_L" in d.files else None
    mp_kpts_ref_R = d["mp_kpts_ref_R"] if "mp_kpts_ref_R" in d.files else None
    always_hand_L_down = (d["always_hand_L_down"]
                          if "always_hand_L_down" in d.files else None)
    always_hand_R_down = (d["always_hand_R_down"]
                          if "always_hand_R_down" in d.files else None)

    if frame < 0 or frame >= n_frames:
        raise ValueError(f"frame {frame} out of range [0, {n_frames})")

    stable_path = _stable_path(subject_name, stem)
    if not stable_path.exists():
        raise RuntimeError(
            f"No stable.mp4 for {subject_name}/{stem} -- run Stabilise first.")

    cap = cv2.VideoCapture(str(stable_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open stable.mp4: {stable_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
    ok, img = cap.read()
    cap.release()
    if not ok or img is None:
        raise RuntimeError(f"Failed to read frame {frame} from stable.mp4")

    h_full = img.shape[0]
    full_w = img.shape[1]
    w_half = (full_w // 2) if is_stereo else full_w

    if downscale > 1:
        bg_L_full = cv2.resize(bg_L, (w_half, h_full), interpolation=cv2.INTER_LINEAR)
        bg_R_full = (cv2.resize(bg_R, (w_half, h_full), interpolation=cv2.INTER_LINEAR)
                     if is_stereo else None)
    else:
        bg_L_full = bg_L
        bg_R_full = bg_R

    # Upscale always-hand masks to full-res (nearest-neighbour preserves
    # the binary nature -- never interpolate a "maybe" pixel into being).
    def _upscale_bool(m, w, h):
        if m is None:
            return None
        u = cv2.resize(m.astype(np.uint8), (w, h),
                        interpolation=cv2.INTER_NEAREST)
        return u.astype(bool)
    always_hand_L = _upscale_bool(always_hand_L_down, w_half, h_full)
    always_hand_R = _upscale_bool(always_hand_R_down, w_half, h_full) \
                     if is_stereo else None

    def _side_outline(stable_side: np.ndarray, bg_full: np.ndarray,
                       kpts: np.ndarray | None,
                       always_hand: np.ndarray | None) -> dict:
        """Returns ``{contour: [[x,y],...], fg: {b64, bbox} | None}``."""
        empty = {"contour": [], "fg": None}
        if kpts is None or np.isnan(kpts).all():
            return empty
        H, W = stable_side.shape[:2]
        # 1. Dilated MP gate (outline-time -- no forearm extension; the
        # visible polygon stops at the wrist).
        kpt_region = _build_kpt_hand_region(
            kpts, W, H,
            stamp_radius=max(2, int(dilation_px)),
            smooth_sigma=4.0)
        gate = (kpt_region > 0.5).astype(np.uint8)
        if gate.sum() < 100:
            return empty
        # 2. Motion = max over BGR channels of |frame - BG|, uint8.
        motion = np.abs(stable_side.astype(np.int16) - bg_full.astype(np.int16))
        motion = motion.max(axis=-1).astype(np.uint8)
        # Optional: foreground heatmap PNG cropped to the gate bbox.
        # Sent back so the UI can paint a JET-coloured fill over the
        # gate region, alpha == motion intensity (dark = transparent).
        fg_pack = None
        if include_fg:
            fg_pack = _encode_fg_png(motion, gate)
        # 3. Otsu threshold over the gate-restricted motion only --
        # gives a per-frame adaptive cut between "moving" (hand) and
        # "still" (BG inside the gate).
        gated_vals = motion[gate > 0]
        if gated_vals.size < 100:
            return {"contour": [], "fg": fg_pack}
        thresh, _ = cv2.threshold(
            gated_vals, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = ((motion > thresh) & (gate > 0))
        # 4. OR with the always-hand map.  Pixels whose BG fell back to
        # the global non-skin fill were structurally inside the hand
        # every sample frame, so they belong inside the boundary even
        # when this frame's pose makes them blend with the synthetic
        # BG colour.  Also clip to the gate so we never extend outside
        # the dilated MP silhouette.
        if always_hand is not None:
            binary = binary | (always_hand & (gate > 0))
        binary = binary.astype(np.uint8) * 255
        # 5. Morphological close -- fills internal holes.
        r = max(1, int(close_radius_px))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        # 6. Largest connected component.
        n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if n_lbl <= 1:
            return {"contour": [], "fg": fg_pack}
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        # 7. Outermost contour + Douglas-Peucker smoothing.
        binary = (lbl == largest).astype(np.uint8) * 255
        contours, _hier = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return {"contour": [], "fg": fg_pack}
        c = max(contours, key=cv2.contourArea)
        if simplify_eps_px > 0:
            c = cv2.approxPolyDP(c, epsilon=float(simplify_eps_px), closed=True)
        return {
            "contour": c.reshape(-1, 2).astype(int).tolist(),
            "fg": fg_pack,
        }

    if is_stereo:
        os_img = img[:, :w_half]
        od_img = img[:, w_half:]
    else:
        os_img = img
        od_img = None

    kpts_L = (mp_kpts_ref_L[frame] if (mp_kpts_ref_L is not None
                                        and frame < mp_kpts_ref_L.shape[0])
              else None)
    kpts_R = (mp_kpts_ref_R[frame] if (mp_kpts_ref_R is not None
                                        and frame < mp_kpts_ref_R.shape[0])
              else None)
    os_out = _side_outline(os_img, bg_L_full, kpts_L, always_hand_L)
    od_out = (_side_outline(od_img, bg_R_full, kpts_R, always_hand_R)
                if is_stereo else None)
    return {
        "frame": int(frame),
        "is_stereo": is_stereo,
        "always_hand_available": always_hand_L is not None,
        "dilation_px": int(dilation_px),
        "OS": os_out["contour"],
        "OD": od_out["contour"] if od_out else None,
        # Foreground heatmap (when include_fg).  ``fg_OS`` /``fg_OD`` are
        # ``{b64: "...", bbox: [x0,y0,x1,y1]}`` or null.
        "fg_OS": os_out["fg"],
        "fg_OD": od_out["fg"] if od_out else None,
    }


# Backward-compat alias: older callers may still reach for
# ``compute_background``.  Points at the new Stage 1 (stable.mp4).
compute_background = compute_stable


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
    }


def stable_mp4_path(subject_name: str, trial_stem: str) -> Path | None:
    """Public helper for downstream consumers (MP/HRnet/DLC).  Returns
    the path to the stabilised mp4 if it exists, else ``None``."""
    p = _stable_path(subject_name, trial_stem)
    return p if p.exists() else None


