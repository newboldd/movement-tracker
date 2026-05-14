"""Stabilization + on-demand hand-boundary extraction.

One bake per trial produces ``stable.mp4`` and ``background.npz``.
The hand boundary is computed on demand per frame by
:func:`compute_outline_frame`, which reads from those two artifacts
and returns a closed polygon -- no global fg.mp4 / outline.mp4 mp4s
are written.

``stable.mp4`` -- every source frame warped into the reference frame's
coordinate system using the camera trajectory.  The static scene stays
put; only the hand (and any other moving parts) appears to move.  Same
resolution and layout as the source video.

``background.npz`` -- temporal median of stabilized samples (with the
hand masked out via the dilated MP skeleton), plus the fitted CbCr
skin model and the MP-ref keypoints, so downstream callers can
reconstruct everything needed for per-frame mask work without
re-loading MP or re-fitting the model.

Inputs:
    Camera trajectory (``camera_trajectory.npz``) must already exist --
    compute it via the *Compute Trajectory* button first.

Outputs in ``<dlc>/<subject>/preproc/<stem>/``:
    stable.mp4              stabilized source video
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

    Called at the start of every Stabilize / Foreground run so that
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
    """Strictly keypoint-gated hand mask (no motion / color contribution).
    Used for the 'isolated' canvas composite — pixels far from any
    MediaPipe landmark are guaranteed-dark even if they moved or are
    skin-colored."""
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


# ─── Enhanced FG-mask helpers (MediaPipe + color) ──────────────────────
#
# The plain "|frame − BG|" mask has two predictable failure modes:
#   1. Hand pixels whose color happens to match the background → dim
#   2. Anything else that moved (head, other hand, scene clutter) → bright
#
# When MediaPipe prelabels are available we can refine the mask with two
# extra signals:
#   • Keypoint proximity — a Gaussian-blurred stamp around each of the 21
#     hand landmarks, warped into the reference frame.  This says "the
#     hand is roughly here, even if motion is weak".
#   • Skin-color similarity — a CbCr Mahalanobis-distance model fit
#     from pixels at MP keypoints in sampled frames.  This says "this
#     pixel's color matches the trial's hand", regardless of motion.
#
# Combined: motion·w_m + keypoint·w_k + color·w_c, clipped to 0..255.
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


def _reflect_thumb_cmc(kpts_ref: np.ndarray) -> np.ndarray | None:
    """Synthetic ulnar-heel point: thumb CMC (joint 1) reflected across
    the wrist -> middle-MCP axis.

    MediaPipe has no landmark on the ulnar ("pinky-side") heel of the
    palm -- joint 1 covers the radial heel, joint 17 the ulnar
    knuckle, but the ulnar heel between them is bare.  That corner is
    a recurring hand-mask miss.  Reflecting the radial thumb-CMC
    corner across the palm's long axis (wrist 0 -> middle MCP 9)
    drops a point right where the missing corner should be.

    Returns ``(x, y)`` float64 or None if joints 0/1/9 aren't all
    present.
    """
    w_pt   = kpts_ref[0]
    cmc_pt = kpts_ref[1]
    mid_pt = kpts_ref[9]
    if (np.isnan(w_pt).any() or np.isnan(cmc_pt).any()
            or np.isnan(mid_pt).any()):
        return None
    w_pt = w_pt.astype(np.float64)
    axis = mid_pt.astype(np.float64) - w_pt
    n = float(np.linalg.norm(axis))
    if n < 1e-6:
        return None
    d = axis / n
    v = cmc_pt.astype(np.float64) - w_pt
    perp = v - np.dot(v, d) * d           # component perpendicular to the axis
    return w_pt + (v - perp) - perp        # = cmc - 2 * perp


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
       close the palm, plus a synthetic ulnar-heel point (thumb CMC
       reflected across the wrist -> middle-MCP axis) so the
       pinky-side heel of the palm -- which has no MP landmark -- is
       covered.
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

    # 3-ulnar. Synthetic ulnar-heel point -- thumb CMC reflected across
    #   the wrist -> middle-MCP axis.  Stamp it and close the ulnar
    #   palm edge (pinky-MCP -> reflected -> thumb-CMC) so the
    #   pinky-side heel of the palm, which has no MP landmark, stops
    #   being a recurring mask miss.
    refl = _reflect_thumb_cmc(kpts_ref)
    if refl is not None:
        rpt = _to_int(refl)
        if 0 <= rpt[0] < w and 0 <= rpt[1] < h:
            cv2.circle(canvas, rpt, stamp_radius, 255, thickness=-1)
        pinky = by_joint.get(17)
        thumb_cmc = by_joint.get(1)
        if pinky is not None:
            cv2.line(canvas, _to_int(pinky), rpt,
                      color=255, thickness=line_thick, lineType=cv2.LINE_AA)
        if thumb_cmc is not None:
            cv2.line(canvas, _to_int(thumb_cmc), rpt,
                      color=255, thickness=line_thick, lineType=cv2.LINE_AA)

    # 3a. Optional forearm extension.  Used at Stabilize-time so the BG
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


# Universal YCrCb skin range -- the FALLBACK used when no
# trial-specific range could be fit (MP missing / too few skin
# pixels).  Conservative enough to catch common skin tones, tight
# enough to reject typical table / desk / clothing colors.  The
# preferred path is _fit_skin_range_cbcr, which derives a window
# from this trial's own hand pixels.
_SKIN_CR_LO, _SKIN_CR_HI = 130, 175
_SKIN_CB_LO, _SKIN_CB_HI = 80, 130


def _is_skin_ycc(ycc: np.ndarray, leniency: float = 1.0,
                 skin_range: tuple | None = None) -> np.ndarray:
    """Boolean mask of skin-colored pixels.  Input is a YCrCb image
    (any shape ending in 3), output is the same shape minus the last
    axis.

    ``skin_range`` -- optional ``(cr_lo, cr_hi, cb_lo, cb_hi)`` window
    fit from this trial's own hand pixels (see _fit_skin_range_cbcr).
    When None, the universal ``_SKIN_*`` box is used.

    ``leniency`` scales whichever window is in play around its
    center: 1.0 = the window as-is, >1 widens it (more pixels count
    as skin), <1 narrows it.
    """
    cr = ycc[..., 1]
    cb = ycc[..., 2]
    if skin_range is not None:
        cr_lo, cr_hi, cb_lo, cb_hi = skin_range
    else:
        cr_lo, cr_hi = _SKIN_CR_LO, _SKIN_CR_HI
        cb_lo, cb_hi = _SKIN_CB_LO, _SKIN_CB_HI
    cr_c = (cr_lo + cr_hi) * 0.5
    cr_h = (cr_hi - cr_lo) * 0.5 * float(leniency)
    cb_c = (cb_lo + cb_hi) * 0.5
    cb_h = (cb_hi - cb_lo) * 0.5 * float(leniency)
    return ((cr >= cr_c - cr_h) & (cr <= cr_c + cr_h)
            & (cb >= cb_c - cb_h) & (cb <= cb_c + cb_h))


def _build_forearm_cone(
    kpts_per_frame: np.ndarray | None,
    frames_sampled: np.ndarray,
    out_w: int,
    out_h: int,
    downscale: int,
    max_length_px: int = 220,
    cone_factor: float = 1.8,
    fallback_width_px: int = 30,
) -> np.ndarray:
    """Cone (trapezoid) mask along the forearm, used to restrict the
    color-based BG refinement to a localised arm region.

    A constant-width capsule missed the forearm because the arm widens
    toward the elbow.  This draws a filled trapezoid:

      - Base edge: the palm-heel line -- thumb CMC (joint 1) to the
        reflected ulnar-heel point.  Anchoring the cone top to that
        segment makes the palm zone and forearm cone meet cleanly
        along it instead of leaving a gap or overlapping at the
        wrist.  (Falls back to a perpendicular segment at the wrist,
        |MCP_5 - MCP_17| x 1.2 wide, when joint 1 / the reflection
        aren't available.)
      - Direction: wrist - median(MCP_5/9/13/17) -- the forearm axis.
      - Tip width (at the far end): base x ``cone_factor``.
      - Length: ``max_length_px`` (downscaled).  Flesh coverage still
        stops naturally at the sleeve / non-skin boundary inside the
        cone.

    Returns an ``(out_h, out_w)`` uint8 mask (0 or 255) at downscaled
    BG resolution.
    """
    if kpts_per_frame is None or frames_sampled.size == 0:
        return np.zeros((out_h, out_w), dtype=np.uint8)
    wrists, mcp5s, mcp17s, centroids = [], [], [], []
    cmc_pts, refl_pts = [], []
    for fi in frames_sampled:
        if fi >= kpts_per_frame.shape[0]:
            continue
        k = kpts_per_frame[fi]
        if not np.isnan(k[0]).any():
            wrists.append(k[0])
        if not np.isnan(k[5]).any():
            mcp5s.append(k[5])
        if not np.isnan(k[17]).any():
            mcp17s.append(k[17])
        if not np.isnan(k[1]).any():
            cmc_pts.append(k[1])
        refl = _reflect_thumb_cmc(k)
        if refl is not None:
            refl_pts.append(refl)
        mcps = [k[j] for j in (5, 9, 13, 17) if not np.isnan(k[j]).any()]
        if mcps:
            centroids.append(np.mean(mcps, axis=0))
    if not wrists or not centroids:
        return np.zeros((out_h, out_w), dtype=np.uint8)

    ds = max(1, int(downscale))
    wrist_med = np.median(wrists, axis=0) / ds
    centroid_med = np.median(centroids, axis=0) / ds
    rough_forearm = wrist_med - centroid_med    # palm -> elbow, roughly
    if float(np.linalg.norm(rough_forearm)) < 1e-6:
        return np.zeros((out_h, out_w), dtype=np.uint8)

    # The palm-heel line (thumb CMC <-> reflected ulnar heel) anchors
    # the trapezoid in BOTH position and orientation:
    #   - base edge runs ALONG the heel line  (perp = heel_dir)
    #   - cone axis runs PERPENDICULAR to it  (forearm_dir)
    # so the trapezoid top is parallel to the heel line by
    # construction.  Falls back to the wrist + wrist->centroid axis
    # when the CMC / reflection aren't available.
    heel_dir = None
    if cmc_pts and refl_pts:
        cmc_med  = np.median(cmc_pts, axis=0) / ds
        refl_med = np.median(refl_pts, axis=0) / ds
        base_mid = (cmc_med + refl_med) * 0.5
        hd = refl_med - cmc_med
        hn = float(np.linalg.norm(hd))
        if hn > 1e-6:
            heel_dir = hd / hn

    if heel_dir is not None:
        perp = heel_dir
        # Cone axis = perpendicular to the heel line, pointing toward
        # the elbow (positive dot with the rough palm->elbow vector).
        axis = np.array([-heel_dir[1], heel_dir[0]], dtype=np.float64)
        if float(np.dot(axis, rough_forearm)) < 0:
            axis = -axis
        forearm_dir = axis
    else:
        # Fallback: orientation from the wrist -> MCP-centroid axis.
        forearm_dir = rough_forearm / float(np.linalg.norm(rough_forearm))
        perp = np.array([-forearm_dir[1], forearm_dir[0]], dtype=np.float64)
        base_mid = wrist_med

    # Base WIDTH -- independent of the heel-line length: keep it wide,
    # the MCP knuckle spread x 1.2 (the pre-heel-line behaviour).
    if mcp5s and mcp17s:
        m5 = np.median(mcp5s, axis=0) / ds
        m17 = np.median(mcp17s, axis=0) / ds
        base_w = max(float(np.linalg.norm(m5 - m17) * 1.2),
                      float(fallback_width_px // ds))
    else:
        base_w = float(max(8, fallback_width_px // ds))
    tip_w = base_w * float(cone_factor)

    length_down = max(2, max_length_px // ds)
    end = base_mid + forearm_dir * length_down

    # Trapezoid: narrow base on the heel line, wide tip at the elbow.
    base_a = base_mid + perp * (base_w / 2.0)
    base_b = base_mid - perp * (base_w / 2.0)
    tip_a  = end      + perp * (tip_w / 2.0)
    tip_b  = end      - perp * (tip_w / 2.0)
    poly = np.array([base_a, base_b, tip_b, tip_a], dtype=np.int32)

    cone = np.zeros((out_h, out_w), dtype=np.uint8)
    cv2.fillPoly(cone, [poly], color=255)
    return cone


def _build_palm_zone(
    hand_mask_stack: np.ndarray,
    grow_px: int,
) -> np.ndarray:
    """"Definitely-hand" palm zone: union of the per-sample-frame MP
    hand gates, dilated outward by ``grow_px``.

    Flesh-colored pixels inside this zone are forced to the green
    sentinel (always-hand) -- no color recovery is attempted, because
    the hand was demonstrably here and any "background color" the
    recovery would find is really just the hand at a different angle /
    lighting.  The ``grow_px`` slider lets the user reach chunks like
    the ulnar palm that the raw MP gate clips.

    ``hand_mask_stack`` is the ``(n_samples, h, w)`` bool array built
    during the sample pass.  Returns an ``(h, w)`` uint8 mask.
    """
    base = hand_mask_stack.any(axis=0).astype(np.uint8) * 255
    if grow_px > 0:
        r = int(grow_px)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        base = cv2.dilate(base, k)
    return base


def _refine_bg_color_based(
    bg_initial: np.ndarray,
    stack: np.ndarray,
    recover_mask: np.ndarray | None = None,
    force_green_mask: np.ndarray | None = None,
    skin_leniency: float = 1.0,
    skin_range: tuple | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Second BG-median pass that fixes flesh-colored BG pixels.

    Two regions, two behaviors:

    * ``force_green_mask`` (the palm zone -- grown MP gate).  Every
      flesh-colored pixel here is painted the green sentinel and
      recorded as always-hand.  No color recovery: the hand was
      demonstrably here, so any "background color" recovery would
      find is just the hand at a different angle / lighting.

    * ``recover_mask`` (the forearm cone, minus the palm zone).  For
      each flesh-colored pixel, classify its ``N`` sample colors
      skin / not-skin.  If at least one sample isn't skin, fill with
      the median of the non-skin samples (recovering the BG that
      peeked through between movements).  If every sample is skin,
      paint green + record always-hand.

    Returns ``(refined_bg, always_hand_color_mask)``.
    """
    bg_ycc = cv2.cvtColor(bg_initial, cv2.COLOR_BGR2YCrCb)
    flesh_in_bg = _is_skin_ycc(bg_ycc, leniency=skin_leniency,
                                skin_range=skin_range)
    refined = bg_initial.copy()
    always_hand = np.zeros(bg_initial.shape[:2], dtype=bool)
    green = np.array([0, 255, 0], dtype=np.uint8)

    # ── 1. Force-green zone: flesh here is definitely hand ───────────
    if force_green_mask is not None:
        fg = flesh_in_bg & (force_green_mask > 0)
        if fg.any():
            refined[fg] = green
            always_hand |= fg
    else:
        fg = np.zeros(bg_initial.shape[:2], dtype=bool)

    # ── 2. Recover zone: per-pixel non-skin median ───────────────────
    if recover_mask is not None:
        rec = flesh_in_bg & (recover_mask > 0) & ~fg
        if rec.any():
            rec_ys, rec_xs = np.where(rec)
            n = int(rec_ys.size)
            N = stack.shape[0]
            samples = stack[:, rec_ys, rec_xs, :]
            samples_ycc = cv2.cvtColor(
                samples.reshape(N * n, 1, 3),
                cv2.COLOR_BGR2YCrCb,
            ).reshape(N, n, 3)
            is_skin = _is_skin_ycc(samples_ycc, leniency=skin_leniency,
                                    skin_range=skin_range)              # (N, n)
            m = np.broadcast_to(is_skin[..., None], samples.shape)
            masked = np.ma.masked_array(samples, mask=m)
            med = np.ma.median(masked, axis=0)               # (n, 3)
            all_skin = np.ma.getmaskarray(med).any(axis=-1)   # (n,)
            refined[rec_ys, rec_xs] = np.ma.getdata(med).astype(np.uint8)
            if all_skin.any():
                gys = rec_ys[all_skin]
                gxs = rec_xs[all_skin]
                refined[gys, gxs] = green
                always_hand[gys, gxs] = True

    return refined, always_hand


def _bg_edge_map(bg_full: np.ndarray) -> np.ndarray:
    """Normalized Sobel-magnitude of the background image.

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
    # 99th percentile normalization -- a few extreme pixels don't blow
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
           - skin color: exp(-Mahalanobis(CbCr) / scale) if a skin model
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

    # 2b. Color similarity (when a skin model was fit).
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


def _fit_skin_range_cbcr(bgr_pixels: np.ndarray | None,
                          pct_lo: float = 2.0, pct_hi: float = 98.0,
                          min_count: int = 200) -> tuple | None:
    """Trial-specific skin Cr/Cb window from high-confidence hand pixels.

    ``bgr_pixels`` is an (M, 3) array of BGR pixels harvested from a
    TIGHT (eroded) MP hand mask across the sampled frames -- pixels
    we're confident are this subject's skin under this trial's
    lighting.  Converts to YCrCb and returns
    ``(cr_lo, cr_hi, cb_lo, cb_hi)`` from the [pct_lo, pct_hi]
    percentiles -- robust to the odd non-skin pixel that crept into
    the tight mask.  Returns None if fewer than ``min_count`` pixels
    were collected (caller then falls back to the universal box).

    Y (luminance) is dropped: skin chrominance is far more
    lighting-invariant than its brightness.
    """
    if bgr_pixels is None or len(bgr_pixels) < min_count:
        return None
    px = np.asarray(bgr_pixels, dtype=np.uint8).reshape(-1, 1, 3)
    ycc = cv2.cvtColor(px, cv2.COLOR_BGR2YCrCb).reshape(-1, 3)
    cr = ycc[:, 1].astype(np.float32)
    cb = ycc[:, 2].astype(np.float32)
    cr_lo, cr_hi = np.percentile(cr, [pct_lo, pct_hi])
    cb_lo, cb_hi = np.percentile(cb, [pct_lo, pct_hi])
    # Never let the window collapse to zero width.
    if cr_hi - cr_lo < 2.0:
        cr_lo, cr_hi = cr_lo - 1.0, cr_hi + 1.0
    if cb_hi - cb_lo < 2.0:
        cb_lo, cb_hi = cb_lo - 1.0, cb_hi + 1.0
    return (float(cr_lo), float(cr_hi), float(cb_lo), float(cb_hi))


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

    bg_edge_norm: optional Sobel-magnitude map of the BG, normalized to
    [0,1].  When present, motion is reduced by
    ``edge_subtract_strength * bg_edge_norm`` so BG seams don't
    dominate when the hand crosses them.  This is a correction (removes
    a known false signal), not an enhancement.

    Previously this function also added an MP keypoint Gaussian and a
    CbCr skin-color boost.  Both have been removed: the downstream
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
) -> str:
    """Stage 1 of the preproc bake: produce stable.mp4 only.

    Reads every source frame, warps it into the reference frame's
    coordinate system using the saved camera trajectory, and pipes
    the raw stream into ffmpeg -> H.264 mp4.  No background median,
    no MP load, no skin-model fit -- those all live in the separate
    :func:`compute_background` stage so the user can re-run the
    (relatively cheap) BG computation without re-warping every
    frame.

    Requires a saved camera trajectory; raises ``RuntimeError``
    otherwise.
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
            f"No camera trajectory for {subject_name}/{stem} -- "
            "run Compute Trajectory first."
        )
    H_L = traj["H_to_ref_L"]
    H_R = traj["H_to_ref_R"]
    is_stereo = bool(traj["is_stereo"])
    n_frames  = int(traj["n_frames"])

    # Probe dimensions + fps with a throwaway capture, then open a
    # FRESH one for the bake loop.  cv2.VideoCapture.set(POS_FRAMES, 0)
    # is unreliable on VBR H.264 -- a botched rewind truncated
    # stable.mp4 to a fraction of its frames -- so we never seek; the
    # bake capture reads strictly sequentially from a clean open.
    probe = cv2.VideoCapture(video_path)
    if not probe.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = probe.get(cv2.CAP_PROP_FPS) or 30.0
    ok, first = probe.read()
    probe.release()
    if not ok:
        raise RuntimeError("Failed to read first frame")
    h_full, w_full = first.shape[:2]
    w_half = (w_full // 2) if is_stereo else w_full
    full_w_total = w_half * 2 if is_stereo else w_half

    out_dir = _preproc_dir(subject_name, stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    stable_path = _stable_path(subject_name, stem)
    # Bake into a temp sibling and os.replace() it into place only on
    # a clean finish.  A cancelled / failed run then leaves the
    # previous good stable.mp4 (and its derivatives) completely
    # untouched -- the job is a no-op rather than a destructive one.
    # The temp name keeps the .mp4 extension (stable.tmp.mp4) so
    # ffmpeg can still infer the output container format from it.
    stable_tmp = stable_path.with_name(
        f"{stable_path.stem}.tmp{stable_path.suffix}")
    _kill_orphan_ffmpegs_for_dir(out_dir)

    stable_proc = _open_ffmpeg_pipe(
        str(stable_tmp), full_w_total, h_full, fps, "bgr24")

    # Fresh capture for the sequential bake read -- no seeking.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot reopen video for bake pass: {video_path}")

    frames_written = 0
    interrupted = False
    ok_to_promote = False
    try:
        for i in range(n_frames):
            if cancel_event is not None and cancel_event.is_set():
                interrupted = True
                raise InterruptedError("Job cancelled")
            ok, frame = cap.read()
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
                    progress_callback(100.0 * (i + 1) / max(1, n_frames))
                except Exception:
                    pass
        # Loop finished or hit a clean source-EOF break -- the .tmp is
        # a complete bake and may be promoted.  An exception (incl.
        # the cancel InterruptedError) skips this and leaves
        # ok_to_promote False.
        ok_to_promote = True
    finally:
        cap.release()
        try: stable_proc.stdin.close()
        except Exception: pass
        try:
            stderr_bytes = stable_proc.stderr.read() if stable_proc.stderr else b""
            rc = stable_proc.wait(timeout=600)
            if rc != 0:
                logger.warning(f"ffmpeg (stable) exited {rc}: "
                                f"{stderr_bytes.decode('utf-8', errors='replace')[:600]}")
                ok_to_promote = False
        except Exception as e:
            logger.warning(f"ffmpeg (stable) finalise error: {e}")
            ok_to_promote = False
        finally:
            _retire_ffmpeg(stable_proc)

        if ok_to_promote and not interrupted:
            # Clean finish: atomically swap the new bake into place,
            # then invalidate the now-stale derivatives (background.npz
            # + fg/outline/hand were computed against the OLD warp).
            os.replace(str(stable_tmp), str(stable_path))
            for p in (_background_path(subject_name, stem),
                      _fg_path(subject_name, stem),
                      _outline_path(subject_name, stem),
                      _hand_path(subject_name, stem)):
                try: p.unlink()
                except FileNotFoundError: pass
        else:
            # Cancelled / failed -> discard the partial .tmp; the
            # previous stable.mp4 + derivatives are left intact.
            try: stable_tmp.unlink()
            except FileNotFoundError: pass

    if progress_callback is not None:
        try: progress_callback(100.0)
        except Exception: pass
    logger.info(
        f"stable bake saved: {stable_path}  "
        f"frames_written={frames_written}/{n_frames}  stereo={is_stereo}")
    return str(stable_path)


def compute_background(
    subject_name: str,
    trial_idx: int,
    progress_callback=None,
    cancel_event=None,
    max_samples: int = _DEFAULT_MAX_SAMPLES,
    downscale:   int = _DEFAULT_DOWNSCALE,
    dilation_px: int = 14,
    palm_grow_px: int = 15,
    color_dilate_px: int = 0,
    skin_leniency: float = 1.0,
) -> str:
    """Stage 2 of the preproc bake: produce background.npz.

    Reads sampled frames from stable.mp4 (already warped, so no
    warpPerspective in the sample pass), stamps a downscaled MP-gate
    region per sample frame so the hand can be excluded from the
    temporal median, runs the color-based forearm refinement, fits
    a trial-specific skin range, and saves the result.

    Two independent MP-dilation knobs:
      - ``palm_grow_px`` -- "MP dilate (mask)": grows the palm zone
        (the force-green hard-boundary region).
      - ``color_dilate_px`` -- "MP dilate (color sample)": erodes
        (negative) or dilates (positive) the MP region that skin
        pixels are SAMPLED from for the trial skin-range fit.
        Negative lets the sampling region pull in tighter than the
        raw skeleton.

    ``skin_leniency`` scales the trial skin Cr/Cb window for the
    color refinement; ``skin_leniency == 0`` skips that refinement
    entirely -- nothing is force-greened and the background is just
    the plain masked temporal median.  Both ``palm_grow_px`` and
    ``color_dilate_px`` feed only the refinement, so they have no
    effect (and are dimmed in the UI) when ``skin_leniency == 0``.

    Re-runnable cheaply (~30 s for a typical trial) so the user can
    iterate without re-running the heavy Stabilize step.

    Requires :func:`compute_stable` to have produced stable.mp4
    first; raises ``RuntimeError`` otherwise.
    """
    from .video import build_trial_map
    from .camera_motion import load_camera_trajectory
    from .mediapipe_prelabel import load_mediapipe_prelabels

    tmap = build_trial_map(subject_name)
    if trial_idx < 0 or trial_idx >= len(tmap):
        raise ValueError(f"trial_idx {trial_idx} out of range ({len(tmap)} trials)")
    trial = tmap[trial_idx]
    stem = trial["trial_name"]

    traj = load_camera_trajectory(subject_name, stem)
    if traj is None:
        raise RuntimeError(
            f"No camera trajectory for {subject_name}/{stem} -- "
            "run Compute Trajectory first.")
    H_L = traj["H_to_ref_L"]
    H_R = traj["H_to_ref_R"]
    is_stereo = bool(traj["is_stereo"])
    n_frames  = int(traj["n_frames"])
    ref       = int(traj["reference_frame"])
    start_frame = int(trial.get("start_frame", 0))

    stable_path = _stable_path(subject_name, stem)
    if not stable_path.exists():
        raise RuntimeError(
            f"No stable.mp4 for {subject_name}/{stem} -- "
            "run Stabilize first.")

    if downscale < 1:
        downscale = 1
    if max_samples < 4:
        max_samples = 4

    # Probe stable.mp4 with a throwaway capture: dimensions + frame
    # count.  Never seek the bake capture (VBR H.264 seeks are
    # unreliable); read sequentially from a fresh open below.
    probe = cv2.VideoCapture(str(stable_path))
    if not probe.isOpened():
        raise RuntimeError(f"Cannot open stable.mp4: {stable_path}")
    stable_n = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, first = probe.read()
    probe.release()
    if not ok:
        raise RuntimeError("Failed to read first frame of stable.mp4")
    h_full, w_full = first.shape[:2]
    w_half = (w_full // 2) if is_stereo else w_full
    out_h = h_full // downscale
    out_w = w_half // downscale

    # Truncation guard.  stable.mp4 should have ~n_frames frames.  A
    # short one means Stabilize was interrupted -- building the BG
    # median from it silently produces a near-black image (most of the
    # sample stack stays at its zero-initialised value).  Fail loudly
    # instead so the user re-runs Stabilize.
    if stable_n < int(n_frames * 0.95):
        raise RuntimeError(
            f"stable.mp4 has only {stable_n} frames but the trajectory "
            f"expects {n_frames} -- Stabilize looks interrupted.  "
            f"Re-run Stabilize before Background.")
    effective_n = min(n_frames, stable_n)

    frames_sampled = _pick_sample_frames(
        effective_n, traj["jerk_flag"][:effective_n], max_samples)
    n_samples = int(frames_sampled.size)
    sample_lookup = {int(f): i for i, f in enumerate(frames_sampled)}

    cap = cv2.VideoCapture(str(stable_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open stable.mp4: {stable_path}")

    # RAM ceiling for the sample stack.
    n_sides = 2 if is_stereo else 1
    bytes_needed = n_samples * out_h * out_w * 3 * n_sides
    if bytes_needed > _MAX_RAM_BYTES:
        new_max = int(_MAX_RAM_BYTES / (out_h * out_w * 3 * n_sides))
        logger.warning(
            f"background: requested {n_samples} samples x {out_h}x{out_w}x3 "
            f"x {n_sides} sides = {bytes_needed/1e9:.2f} GB exceeds "
            f"{_MAX_RAM_BYTES/1e9:.1f} GB cap; reducing to {new_max} samples")
        frames_sampled = _pick_sample_frames(
            effective_n, traj["jerk_flag"][:effective_n], new_max)
        n_samples = int(frames_sampled.size)
        sample_lookup = {int(f): i for i, f in enumerate(frames_sampled)}

    # ── Phase 0: MP keypoints -> reference-frame coords ──────────────
    mp_kpts_ref_L: np.ndarray | None = None
    mp_kpts_ref_R: np.ndarray | None = None
    try:
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

    # Allocate stacks + hand-region masks at downscaled resolution.
    stack_L = np.empty((n_samples, out_h, out_w, 3), dtype=np.uint8)
    stack_R = (np.empty((n_samples, out_h, out_w, 3), dtype=np.uint8)
               if is_stereo else None)
    hand_mask_L = np.zeros((n_samples, out_h, out_w), dtype=bool)
    hand_mask_R = (np.zeros((n_samples, out_h, out_w), dtype=bool)
                   if is_stereo else None)
    _BG_MASK_STAMP_DOWN = max(2, int(dilation_px) // max(1, downscale))
    _BG_MASK_FOREARM_DOWN = max(0, 100 // max(1, downscale))
    # Tight MP mask for the trial-specific skin-range fit: half the
    # dilation stamp, no forearm extension, no feather -- a conservative
    # "definitely hand" core.  Its pixels feed _fit_skin_range_cbcr;
    # harvested per frame into these lists.  The "MP dilate (color
    # sample)" knob (``color_dilate_px``) then erodes (negative) or
    # dilates (positive) that core so the user can tighten the colour
    # sample away from non-hand pixels or loosen it to catch more skin.
    _TIGHT_STAMP_DOWN = max(2, _BG_MASK_STAMP_DOWN // 2)
    color_dilate_down = int(round(color_dilate_px / max(1, downscale)))
    skin_px_L: list[np.ndarray] = []
    skin_px_R: list[np.ndarray] = []

    def _harvest_skin(warp_d, kpts_down, sink):
        """Collect BGR pixels under a tight MP mask, adjusted by
        ``color_dilate_down`` (erode if negative, dilate if positive)."""
        tight = _build_kpt_hand_region(
            kpts_down, out_w, out_h,
            stamp_radius=_TIGHT_STAMP_DOWN, smooth_sigma=0.0,
            extend_forearm_px=0)
        tb = (tight > 0.5).astype(np.uint8)
        if color_dilate_down != 0:
            ksz = 2 * abs(color_dilate_down) + 1
            kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
            if color_dilate_down < 0:
                tb = cv2.erode(tb, kern)
            else:
                tb = cv2.dilate(tb, kern)
        else:
            # Default still trims a 1 px rim to stay off the boundary.
            tb = cv2.erode(tb, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (3, 3)))
        if tb.any():
            sink.append(warp_d[tb > 0])

    # ── Sample pass: sequential read of stable.mp4 ─────────────────
    # stable.mp4 is already warped, so we skip warpPerspective and
    # just downscale + stamp the hand region.  The capture is fresh
    # (opened above) and read strictly sequentially -- no seeking.
    frames_set = set(int(f) for f in frames_sampled)
    max_frame = int(frames_sampled.max())

    for i in range(max_frame + 1):
        if cancel_event is not None and cancel_event.is_set():
            cap.release()
            raise InterruptedError("Job cancelled")
        ok, frame = cap.read()
        if not ok:
            logger.warning(f"background sample: read failed at frame {i}; truncating")
            break
        if i not in frames_set:
            continue
        s_idx = sample_lookup[i]

        if is_stereo:
            warp_L = frame[:, :w_half]
            warp_R = frame[:, w_half:]
        else:
            warp_L = frame
            warp_R = None

        if downscale > 1:
            warp_L_d = cv2.resize(warp_L, (out_w, out_h),
                                    interpolation=cv2.INTER_AREA)
        else:
            warp_L_d = warp_L
        stack_L[s_idx] = warp_L_d
        if mp_kpts_ref_L is not None and i < mp_kpts_ref_L.shape[0]:
            kpts_L_down = mp_kpts_ref_L[i] / max(1, downscale)
            region_L = _build_kpt_hand_region(
                kpts_L_down, out_w, out_h,
                stamp_radius=_BG_MASK_STAMP_DOWN, smooth_sigma=2.0,
                extend_forearm_px=_BG_MASK_FOREARM_DOWN)
            hand_mask_L[s_idx] = region_L > 0.5
            _harvest_skin(warp_L_d, kpts_L_down, skin_px_L)

        if is_stereo and warp_R is not None:
            if downscale > 1:
                warp_R_d = cv2.resize(warp_R, (out_w, out_h),
                                       interpolation=cv2.INTER_AREA)
            else:
                warp_R_d = warp_R
            stack_R[s_idx] = warp_R_d
            if (mp_kpts_ref_R is not None
                    and i < mp_kpts_ref_R.shape[0]):
                kpts_R_down = mp_kpts_ref_R[i] / max(1, downscale)
                region_R = _build_kpt_hand_region(
                    kpts_R_down, out_w, out_h,
                    stamp_radius=_BG_MASK_STAMP_DOWN, smooth_sigma=2.0,
                    extend_forearm_px=_BG_MASK_FOREARM_DOWN)
                hand_mask_R[s_idx] = region_R > 0.5
                _harvest_skin(warp_R_d, kpts_R_down, skin_px_R)

        if progress_callback is not None:
            try:
                progress_callback(15.0 * (s_idx + 1) / max(1, n_samples))
            except Exception:
                pass
    cap.release()

    if progress_callback is not None:
        try: progress_callback(20.0)
        except Exception: pass

    # ── Per-pixel temporal median + MAD ─────────────────────────────
    def _masked_median(stack: np.ndarray, hand_mask: np.ndarray) -> np.ndarray:
        """Median over non-MP-covered samples per pixel.  Pixels under
        the MP gate in every sample fall back to a bright-green
        sentinel ``[0, 255, 0]`` (BGR) -- unmistakable in the BG
        image and guaranteed bright in the foreground heatmap."""
        if not hand_mask.any():
            return np.median(stack, axis=0).astype(np.uint8)
        m4 = np.broadcast_to(hand_mask[..., None], stack.shape)
        masked = np.ma.masked_array(stack, mask=m4)
        med = np.ma.median(masked, axis=0)
        fill_color = np.array([0, 255, 0], dtype=np.uint8)
        bg = np.where(np.ma.getmaskarray(med),
                       fill_color.reshape(1, 1, 3),
                       np.ma.getdata(med))
        return bg.astype(np.uint8)

    bg_L = _masked_median(stack_L, hand_mask_L)
    dev_L = np.abs(stack_L.astype(np.int16) - bg_L.astype(np.int16))
    mad_L = np.median(dev_L, axis=0).max(axis=-1).astype(np.uint8)
    del dev_L

    if is_stereo:
        bg_R = _masked_median(stack_R, hand_mask_R)
        dev_R = np.abs(stack_R.astype(np.int16) - bg_R.astype(np.int16))
        mad_R = np.median(dev_R, axis=0).max(axis=-1).astype(np.uint8)
        del dev_R
    else:
        bg_R = np.zeros_like(bg_L)
        mad_R = np.zeros_like(mad_L)

    # ── Color-based refinement: palm zone + forearm cone ───────────
    # Palm zone (grown MP gate): the hand was demonstrably present, so
    # it's force-greened -- no "background color" recovery.
    # ``palm_grow_px`` ("MP dilate (mask)") lets the user reach chunks
    # like the ulnar palm that the raw gate clips.
    # Forearm cone: flesh here gets per-pixel non-skin recovery, green
    # only where every sample is skin.
    #
    # ``skin_leniency == 0`` skips the color-based refinement
    # entirely: nothing is force-greened, the background is just the
    # plain masked temporal median.  Both MP-dilate knobs feed only
    # this refinement, so they're dimmed in the UI in that mode.
    if skin_leniency > 0:
        palm_grow_down = max(0, int(palm_grow_px) // max(1, downscale))
        # Trial-specific skin range from the tight-MP-mask pixels --
        # this subject's own hand under this trial's lighting; falls
        # back to the universal box when too few pixels were harvested.
        skin_px_L_all = np.vstack(skin_px_L) if skin_px_L else None
        skin_px_R_all = np.vstack(skin_px_R) if skin_px_R else None
        skin_range_L = _fit_skin_range_cbcr(skin_px_L_all)
        skin_range_R = (_fit_skin_range_cbcr(skin_px_R_all)
                        if is_stereo else None)
        logger.info(
            f"BG skin range: L={skin_range_L} (n={0 if skin_px_L_all is None else len(skin_px_L_all)})"
            + (f", R={skin_range_R}" if is_stereo else ""))

        palm_zone_L = _build_palm_zone(hand_mask_L, palm_grow_down)
        cone_L = _build_forearm_cone(mp_kpts_ref_L, frames_sampled,
                                        out_w, out_h, downscale)
        bg_L, always_hand_color_L_down = _refine_bg_color_based(
            bg_L, stack_L, recover_mask=cone_L, force_green_mask=palm_zone_L,
            skin_leniency=skin_leniency, skin_range=skin_range_L)
        if is_stereo:
            palm_zone_R = _build_palm_zone(hand_mask_R, palm_grow_down)
            cone_R = _build_forearm_cone(mp_kpts_ref_R, frames_sampled,
                                            out_w, out_h, downscale)
            bg_R, always_hand_color_R_down = _refine_bg_color_based(
                bg_R, stack_R, recover_mask=cone_R, force_green_mask=palm_zone_R,
                skin_leniency=skin_leniency, skin_range=skin_range_R)
        else:
            always_hand_color_R_down = None
    else:
        logger.info("BG skin_leniency=0: skipping color-based refinement "
                    "entirely; nothing force-greened")
        always_hand_color_L_down = np.zeros((out_h, out_w), dtype=bool)
        always_hand_color_R_down = (np.zeros((out_h, out_w), dtype=bool)
                                    if is_stereo else None)

    always_hand_L_down = hand_mask_L.all(axis=0) | always_hand_color_L_down
    if hand_mask_R is not None:
        always_hand_R_down = (hand_mask_R.all(axis=0)
                              | always_hand_color_R_down)
    else:
        always_hand_R_down = None
    del hand_mask_L
    if hand_mask_R is not None:
        del hand_mask_R

    if progress_callback is not None:
        try: progress_callback(75.0)
        except Exception: pass

    try: del stack_L
    except NameError: pass
    try: del stack_R
    except NameError: pass

    if progress_callback is not None:
        try: progress_callback(85.0)
        except Exception: pass

    # ── Save background.npz + PNGs ───────────────────────────────────
    out_dir = _preproc_dir(subject_name, stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = _background_path(subject_name, stem)
    save_kwargs = dict(
        background_L=bg_L,
        background_R=bg_R,
        mad_L=mad_L,
        mad_R=mad_R,
        frames_sampled=frames_sampled,
        frames_written=np.array(n_frames, dtype=np.int32),
        n_frames=np.array(n_frames, dtype=np.int32),
        reference_frame=np.array(ref, dtype=np.int32),
        is_stereo=np.array(is_stereo, dtype=bool),
        downscale=np.array(downscale, dtype=np.int32),
    )
    if mp_kpts_ref_L is not None:
        save_kwargs["mp_kpts_ref_L"] = mp_kpts_ref_L.astype(np.float32)
    if mp_kpts_ref_R is not None:
        save_kwargs["mp_kpts_ref_R"] = mp_kpts_ref_R.astype(np.float32)
    # Trial-specific skin Cr/Cb windows -- (cr_lo, cr_hi, cb_lo,
    # cb_hi), or absent when the universal box was used as fallback.
    if skin_range_L is not None:
        save_kwargs["skin_range_L"] = np.asarray(skin_range_L, dtype=np.float32)
    if skin_range_R is not None:
        save_kwargs["skin_range_R"] = np.asarray(skin_range_R, dtype=np.float32)
    if always_hand_L_down is not None:
        save_kwargs["always_hand_L_down"] = always_hand_L_down
    if always_hand_R_down is not None:
        save_kwargs["always_hand_R_down"] = always_hand_R_down

    # Write every artifact to a temp sibling, then os.replace() it
    # into place.  A crash mid-write can't corrupt the previous good
    # background.npz / PNGs -- the swap is atomic.  np.savez_compressed
    # gets a file handle (so it doesn't append a second .npz); imwrite
    # gets a *.tmp.png name so it still infers the PNG encoder.
    tmp_npz = out_dir / "background.npz.tmp"
    with open(tmp_npz, "wb") as _fh:
        np.savez_compressed(_fh, **save_kwargs)
    os.replace(str(tmp_npz), str(out_path))

    def _atomic_png(img, name):
        final = out_dir / name
        tmp = out_dir / (final.stem + ".tmp" + final.suffix)
        cv2.imwrite(str(tmp), img)
        os.replace(str(tmp), str(final))

    _atomic_png(bg_L,  "background_OS.png")
    _atomic_png(mad_L, "mad_OS.png")
    if is_stereo:
        _atomic_png(bg_R,  "background_OD.png")
        _atomic_png(mad_R, "mad_OD.png")

    if progress_callback is not None:
        try: progress_callback(100.0)
        except Exception: pass
    logger.info(
        f"background.npz saved: {out_path}  N_samples={n_samples}  "
        f"stereo={is_stereo}  downscale={downscale}")
    return str(out_path)




def _encode_fg_png(motion: np.ndarray, gate: np.ndarray) -> dict | None:
    """Pack a small RGBA PNG of the foreground heatmap inside the gate.

    Returns ``{b64, bbox: [x0,y0,x1,y1]}`` or None if the gate is empty.
    The image covers the WHOLE dilated MP gate -- alpha is flat 255
    inside the gate and 0 outside, so the user sees the full region
    colored by motion intensity (low motion = dark blue JET, high
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
    # intensity (encoded by color) from visibility (controlled by the
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
    open_radius_px: int = 0,
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
         every sample frame during Stabilize -- their BG fell back to
         a synthetic color, so we force them to count as hand always).
      5. Morphological close (disk radius ``close_radius_px``) to fill
         palm-interior holes where skin happens to match BG.
      6. Morphological open (disk radius ``open_radius_px``) to clip
         thin strands / webs jutting off the boundary.  0 = no
         clipping; bump it to trim noise without shaving fingertips
         (they're far wider than the kernel at sane values).
      7. Keep only the largest connected component (drop speckles).
      8. Extract the outermost contour and simplify via Douglas-Peucker
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
            "run Stabilize first.")
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
            f"No stable.mp4 for {subject_name}/{stem} -- run Stabilize first.")

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

    # Upscale always-hand masks to full-res (nearest-neighbor preserves
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

    # Background edge map -- Sobel magnitude normalized to [0, 1].
    # Subtracted from motion before Otsu so static high-contrast
    # seams in the BG (table edges, monitor bezels) don't produce
    # false outline boundaries when the hand crosses them.  True
    # hand-vs-BG contrast usually dominates the BG edge, so the
    # "true overlap" case is preserved by the subtraction surviving
    # at high motion values.
    bg_edge_L = _bg_edge_map(bg_L_full)
    bg_edge_R = _bg_edge_map(bg_R_full) if is_stereo else None
    # Stronger subtraction = fewer BG-edge artifacts but risks
    # killing true overlap.  60 motion units at peak edge ~= 1/4 of
    # the dynamic range, leaves room for the hand to dominate.
    _BG_EDGE_PENALTY = 60.0

    def _side_outline(stable_side: np.ndarray, bg_full: np.ndarray,
                       kpts: np.ndarray | None,
                       always_hand: np.ndarray | None,
                       bg_edge: np.ndarray | None) -> dict:
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
        # 2. Motion = max over BGR channels of |frame - BG|, with the
        # BG-edge map subtracted off so high-contrast BG seams don't
        # show as motion when the hand crosses them.  Float for the
        # subtraction, clipped back to uint8 [0, 255] afterwards.
        motion_raw = np.abs(stable_side.astype(np.int16) - bg_full.astype(np.int16))
        motion_f = motion_raw.max(axis=-1).astype(np.float32)
        if bg_edge is not None:
            motion_f = motion_f - _BG_EDGE_PENALTY * bg_edge
        motion = np.clip(motion_f, 0.0, 255.0).astype(np.uint8)
        # Detect bright-green sentinel pixels in the BG.  |frame - green|
        # is huge for any skin-colored hand pixel and would saturate
        # the JET heatmap, making nearby moderately-bright hand pixels
        # look dim by comparison.  Replace their motion with the 80th
        # percentile of nearby non-green gate motion so they read as
        # "bright but not max" -- in the orange/red band of JET, on the
        # high end of the rest of the heatmap but not pinned to red.
        green_mask = ((bg_full[..., 0] < 50)
                       & (bg_full[..., 1] > 200)
                       & (bg_full[..., 2] < 50)
                       & (gate > 0))
        if green_mask.any():
            non_green_gate = (gate > 0) & ~green_mask
            if non_green_gate.any():
                local_p80 = int(np.percentile(motion[non_green_gate], 80))
                motion[green_mask] = local_p80
        # Optional: foreground heatmap PNG cropped to the gate bbox.
        # Sent back so the UI can paint a JET-colored fill over the
        # gate region.  Now uniformly visible thanks to the green
        # normalization above.
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
        # BG color.  Also clip to the gate so we never extend outside
        # the dilated MP silhouette.
        if always_hand is not None:
            binary = binary | (always_hand & (gate > 0))
        binary = binary.astype(np.uint8) * 255
        # 5. Morphological close -- fills internal holes.
        r = max(1, int(close_radius_px))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        # 6. Morphological open -- clips thin strands / webs.  Erode
        # then dilate by the same disk: anything narrower than
        # 2 * open_radius_px disappears, fingertips (much wider)
        # survive.  0 = off.
        if open_radius_px > 0:
            ro = int(open_radius_px)
            ok_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * ro + 1, 2 * ro + 1))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, ok_kernel)
        # 7. Largest connected component.
        n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if n_lbl <= 1:
            return {"contour": [], "fg": fg_pack}
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        # 8. Outermost contour + Douglas-Peucker smoothing.
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
    os_out = _side_outline(os_img, bg_L_full, kpts_L, always_hand_L, bg_edge_L)
    od_out = (_side_outline(od_img, bg_R_full, kpts_R, always_hand_R, bg_edge_R)
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
    the path to the stabilized mp4 if it exists, else ``None``."""
    p = _stable_path(subject_name, trial_stem)
    return p if p.exists() else None


