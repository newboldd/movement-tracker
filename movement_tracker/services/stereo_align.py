"""Image-based cross-camera label alignment.

Coarse-to-fine pipeline per frame:

    Pass 1 (hand-wide)
        Crop a big window around each camera's hand centroid (mean of
        valid MP labels) and phase-correlate.  Background dominates the
        FFT; the discovered (dx, dy) is the frame's hand-wide bias.

    Pass 2 (per-joint)
        For every joint, crop a small window around its MP label in
        each camera and phase-correlate.  The result is clamped to be
        within ``_REFINE_CLAMP_PX`` of the hand-wide bias — kills the
        "joint flung off into nowhere" failure mode where phase corr
        latches onto an off-target feature.

Per-joint crop sizes are tuned: wrist (joint 0) gets a 50% larger box
(more texture to anchor against the palm/forearm boundary), every other
joint gets a 50% smaller box (less background contamination on
fingertips and PIPs).

A high-pass pre-filter (subtract Gaussian-blurred copy) suppresses
slowly-varying background gradients before each phase correlation.

Output (per trial, saved to ``<dlc>/<subject>/skeleton/<stem>/stereo_align.npz``)::

    shifts                 (N, 21, 2) float32  — (dx, dy) per joint per frame
    response               (N, 21)    float32  — phase-corr peak strength
    hand_shifts            (N, 2)     float32  — pass-1 hand-wide shift
    hand_response          (N,)       float32
    crop_half              int                 — legacy single crop half-size
    crop_halves_per_joint  (21,) int32
    hand_crop_half         int
    refine_clamp_px        int
    start_frame            int
    n_frames               int
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── Crop sizes ─────────────────────────────────────────────────────────────

_DEFAULT_CROP_HALF = 40       # 81×81 — original base size, kept for legacy npz fields

_WRIST_JOINT = 0
_DEFAULT_OTHER_CROP_HALF = int(round(_DEFAULT_CROP_HALF * 0.5))  # 20 → 41×41
_PER_JOINT_CROP_HALF: dict[int, int] = {
    _WRIST_JOINT: int(round(_DEFAULT_CROP_HALF * 1.5)),  # 60 → 121×121
}


def _crop_half_for_joint(j: int) -> int:
    return _PER_JOINT_CROP_HALF.get(int(j), _DEFAULT_OTHER_CROP_HALF)


def crop_halves_per_joint(n_joints: int = 21) -> list[int]:
    return [_crop_half_for_joint(j) for j in range(n_joints)]


_HAND_CROP_HALF = 80          # 161×161 around the per-frame hand centroid
_REFINE_CLAMP_PX = 6          # max per-joint shift residual on top of hand-wide

_HIGHPASS_SIGMA = 6.0         # high-pass pre-filter σ (px); kills background
                              # gradients while preserving skin/edge texture


# ── Crop / phase-correlate helpers ─────────────────────────────────────────

def _crop(img: np.ndarray, cx: int, cy: int, half: int) -> np.ndarray:
    """Return a (2*half+1)×(2*half+1)×C crop centered on (cx, cy),
    padded with reflection if it runs off the edge of the image."""
    h, w = img.shape[:2]
    x0, y0 = cx - half, cy - half
    x1, y1 = cx + half + 1, cy + half + 1
    pad_x0 = max(0, -x0); pad_y0 = max(0, -y0)
    pad_x1 = max(0, x1 - w); pad_y1 = max(0, y1 - h)
    sx0, sx1 = max(0, x0), min(w, x1)
    sy0, sy1 = max(0, y0), min(h, y1)
    sliced = img[sy0:sy1, sx0:sx1]
    if pad_x0 or pad_y0 or pad_x1 or pad_y1:
        sliced = cv2.copyMakeBorder(sliced, pad_y0, pad_y1, pad_x0, pad_x1,
                                    cv2.BORDER_REFLECT_101)
    return sliced


def _align_phase(os_crop: np.ndarray, od_crop: np.ndarray,
                  window: np.ndarray) -> tuple[float, float, float]:
    """Phase-correlate OS crop → OD crop and return (dx, dy, response).
    Pipeline: BGR → gray → high-pass (subtract Gaussian-blurred copy) →
    mean-subtract → Hanning window → ``cv2.phaseCorrelate``."""
    a = cv2.cvtColor(os_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(od_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    a = a - cv2.GaussianBlur(a, (0, 0), sigmaX=_HIGHPASS_SIGMA, sigmaY=_HIGHPASS_SIGMA)
    b = b - cv2.GaussianBlur(b, (0, 0), sigmaX=_HIGHPASS_SIGMA, sigmaY=_HIGHPASS_SIGMA)
    a = (a - a.mean()) * window
    b = (b - b.mean()) * window
    (dx, dy), response = cv2.phaseCorrelate(a, b)
    return float(dx), float(dy), float(response)


def _align_phase_mask(os_crop: np.ndarray, od_crop: np.ndarray,
                       window: np.ndarray) -> tuple[float, float, float]:
    """Phase-correlate two single-channel outline-mask crops.

    Same FFT pipeline as :func:`_align_phase` but skips the BGR->gray
    + high-pass steps -- the input is already a feature image
    (filled-polygon silhouette), so all phase correlation needs is the
    mean-subtract + Hanning window."""
    a = os_crop.astype(np.float32)
    b = od_crop.astype(np.float32)
    a = (a - a.mean()) * window
    b = (b - b.mean()) * window
    (dx, dy), response = cv2.phaseCorrelate(a, b)
    return float(dx), float(dy), float(response)


def _gauss_centre_weight(shape: tuple[int, int], strength: float) -> np.ndarray | None:
    """Build a 2D Gaussian weight image centred on the crop centre
    (which is where each per-joint phase-corr crop is anchored on the
    MP label).  ``strength`` in [0, 1]: 0 returns None (caller skips
    the Gaussian multiply), 1 produces a tight Gaussian whose sigma
    is ~0.3 * half-size.  Linearly interpolated in between."""
    if strength <= 0.0:
        return None
    H, W = shape
    cy = (H - 1) * 0.5
    cx = (W - 1) * 0.5
    jh = (min(H, W) - 1) * 0.5  # half-size in pixels
    # sigma_frac (units of jh): 5.0 at strength=0+, 0.3 at strength=1.
    sigma_frac = 5.0 - 4.7 * float(min(max(strength, 0.0), 1.0))
    sigma = max(1e-3, jh * sigma_frac)
    ys = np.arange(H, dtype=np.float32)[:, None]
    xs = np.arange(W, dtype=np.float32)[None, :]
    inv2s2 = 1.0 / (2.0 * sigma * sigma)
    return np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) * inv2s2).astype(np.float32)


def _align_phase_weighted(os_crop: np.ndarray, od_crop: np.ndarray,
                           window: np.ndarray,
                           os_mask: np.ndarray | None = None,
                           od_mask: np.ndarray | None = None,
                           gauss_strength: float = 0.0
                           ) -> tuple[float, float, float]:
    """Phase-correlate two BGR crops with optional per-pixel weighting.

    Final per-pixel weight = ``window`` x (mask if provided)
    x (Gaussian if ``gauss_strength`` > 0).

    - Background pixels (mask == 0) contribute LITERAL ZERO to the
      FFT input -- alignment is driven only by foreground content.
    - The Gaussian falloff (centred on the crop centre, which sits on
      the MP label) emphasises pixels NEAR the joint over pixels at
      the edge of the bbox.  Strength 0 disables it (uniform within
      the mask + Hanning).
    - The FG-only mean is subtracted so the masked-out / heavily-
      attenuated regions stay at exactly zero (no DC leakage)."""
    a = cv2.cvtColor(os_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(od_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    a = a - cv2.GaussianBlur(a, (0, 0), sigmaX=_HIGHPASS_SIGMA, sigmaY=_HIGHPASS_SIGMA)
    b = b - cv2.GaussianBlur(b, (0, 0), sigmaX=_HIGHPASS_SIGMA, sigmaY=_HIGHPASS_SIGMA)
    wa = window.astype(np.float32, copy=True)
    wb = window.astype(np.float32, copy=True)
    if os_mask is not None:
        wa = wa * (os_mask > 0).astype(np.float32)
    if od_mask is not None:
        wb = wb * (od_mask > 0).astype(np.float32)
    g = _gauss_centre_weight(a.shape, gauss_strength)
    if g is not None:
        wa = wa * g
        wb = wb * g
    # FG-only means (pixels with non-trivial weight).
    fa = wa > 1e-4
    fb = wb > 1e-4
    amean = float(a[fa].mean()) if fa.any() else 0.0
    bmean = float(b[fb].mean()) if fb.any() else 0.0
    a = (a - amean) * wa
    b = (b - bmean) * wb
    (dx, dy), response = cv2.phaseCorrelate(a, b)
    return float(dx), float(dy), float(response)


def _warp_poly_ref_to_orig(poly_ref, H_to_ref: np.ndarray):
    """Apply ``H_to_ref^-1`` (ref -> original camera coords) to every
    point of the polygon.  Returns an int32 (N, 1, 2) array suitable
    for ``cv2.fillPoly``, or None if the polygon is missing / a point
    fails to project."""
    if poly_ref is None or len(poly_ref) < 3:
        return None
    try:
        H_inv = np.linalg.inv(H_to_ref.astype(np.float64))
    except np.linalg.LinAlgError:
        return None
    pts = np.asarray(poly_ref, dtype=np.float64)
    n = pts.shape[0]
    pts_h = np.column_stack([pts, np.ones(n)]).T          # (3, n)
    w = H_inv @ pts_h
    z = w[2]
    if np.any(np.abs(z) < 1e-9):
        return None
    xy = (w[:2] / z).T
    if not np.all(np.isfinite(xy)):
        return None
    return np.rint(xy).astype(np.int32).reshape(-1, 1, 2)


def _outline_mask_image(poly_orig, h_full: int, half_w: int) -> np.ndarray:
    """Filled-polygon silhouette of an outline (in original camera
    coords) rasterized to a ``(h_full, half_w)`` uint8 mask -- 255
    inside the hand, 0 elsewhere.  Returns an all-zero mask when the
    polygon is missing / degenerate."""
    mask = np.zeros((h_full, half_w), dtype=np.uint8)
    if poly_orig is not None and len(poly_orig) >= 3:
        cv2.fillPoly(mask, [poly_orig], 255)
    return mask


def _dilate_mask(mask: np.ndarray, dilate_px: int) -> np.ndarray:
    """Dilate a uint8 mask by ``dilate_px`` using an elliptical kernel.
    No-op if ``dilate_px <= 0``."""
    if dilate_px <= 0:
        return mask
    k = 2 * int(dilate_px) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask, kernel)


def _densify_poly(poly, sp: float = 3.0) -> np.ndarray:
    """Resample a closed polygon (Nx1x2 int32 or Nx2 array-like) at
    ~``sp`` px spacing.  Returns an (M, 2) float32 array.  Empty if
    the polygon is missing / degenerate."""
    if poly is None:
        return np.zeros((0, 2), dtype=np.float32)
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 3:
        return np.zeros((0, 2), dtype=np.float32)
    out = []
    n = pts.shape[0]
    for i in range(n):
        a = pts[i]
        b = pts[(i + 1) % n]
        d = b - a
        L = float(np.hypot(d[0], d[1]))
        steps = max(1, int(round(L / float(sp))))
        for k in range(steps):
            t = k / steps
            out.append(a + d * t)
    return np.asarray(out, dtype=np.float32)


def _voting_align(os_poly, od_poly,
                   os_mp_cent: np.ndarray, od_mp_cent: np.ndarray,
                   sample_sp: float = 3.0,
                   max_radius: int = 130) -> tuple[float, float, float]:
    """Vote pairwise offsets between two dense polygon samples into a
    2D accumulator; the peak (of the 3x3-summed accumulator) is the
    translation that aligns the MOST sampled points.

    Both polygons are first re-centred on their camera's MP-keypoint
    centroid so the returned ``(dx, dy)`` is the **residual** offset
    after MP-centering -- the same convention the image-based
    phase-correlation path produces, which downstream consumers
    (mp_label + shift) depend on.  Mirrors the preproc page's
    _computeOtherAlign algorithm but the seed / output convention is
    swapped from "polygon-centroid-relative" to "MP-centroid-relative".

    Response = ``peak_votes / max(1, len(os_samples))`` -- roughly the
    fraction of OS points that found an OD match within +/- 1 px.
    """
    os_pts = _densify_poly(os_poly, sample_sp)
    od_pts = _densify_poly(od_poly, sample_sp)
    if os_pts.shape[0] < 8 or od_pts.shape[0] < 8:
        return 0.0, 0.0, float('nan')
    # Centre each polygon on its own MP centroid; the voting result is
    # then the residual after MP-centering -- same as phase_corr.
    os_c = os_pts - np.asarray(os_mp_cent, dtype=np.float32).reshape(1, 2)
    od_c = od_pts - np.asarray(od_mp_cent, dtype=np.float32).reshape(1, 2)
    R = int(max_radius)
    W = 2 * R + 1
    acc = np.zeros(W * W, dtype=np.uint32)
    for i in range(os_c.shape[0]):
        # ix = round(od_c.x - os_c[i].x) + R; vectorise over od.
        ix = np.rint(od_c[:, 0] - os_c[i, 0]).astype(np.int64) + R
        iy = np.rint(od_c[:, 1] - os_c[i, 1]).astype(np.int64) + R
        ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < W)
        if not ok.any():
            continue
        flat = iy[ok] * W + ix[ok]
        np.add.at(acc, flat, 1)
    acc = acc.reshape(W, W)
    # 3x3-summed peak -- "exactly aligned within +/-1 px".
    s = (acc[1:-1, 1:-1]
         + acc[:-2, 1:-1] + acc[2:, 1:-1]
         + acc[1:-1, :-2] + acc[1:-1, 2:]
         + acc[:-2, :-2] + acc[:-2, 2:]
         + acc[2:, :-2] + acc[2:, 2:])
    by_x = int(np.argmax(s))
    by, bx = divmod(by_x, W - 2)
    by += 1; bx += 1   # shift back into the un-sliced grid
    peak = int(s.flat[by_x])
    dx = bx - R
    dy = by - R
    resp = peak / float(max(1, os_pts.shape[0]))
    return float(dx), float(dy), float(min(1.0, resp))


# ── Main entry point ───────────────────────────────────────────────────────

def run_stereo_align(subject_name: str, trial_idx: int,
                      progress_callback=None,
                      crop_half: int = _DEFAULT_CROP_HALF,
                      cancel_event=None,
                      use_outline: bool = False,
                      mode: str = "image",
                      mask_dilate_px: int = 10,
                      gauss_center_weight: float = 0.0) -> str:
    """Run cross-camera stereo label alignment for one trial.

    The ``mode`` argument selects which feature image drives each
    stage of the alignment:

    - ``"image"`` (default): both Pass 1 (hand-wide) and Pass 2
      (per-joint) use raw video frame crops with a BGR -> gray ->
      high-pass pre-filter.  Output: ``stereo_align.npz``.
    - ``"outline"``: both passes use a filled-polygon silhouette of
      the preproc-baked hand outline (inverse-warped from stable-
      frame to original camera coords via ``H_to_ref^-1``).  Pass 1
      uses dense-polygon-offset voting on the entire outline; Pass 2
      uses phase-corr on outline-mask crops.  Output:
      ``stereo_align_outline.npz``.  Requires ``outlines.json`` +
      ``camera_trajectory.npz`` from the preproc pipeline.
    - ``"hybrid"``: Pass 1 uses outline voting (large-scale features,
      robust to hand pose); Pass 2 uses raw image phase-corr (fine
      pixel-level features).  Output: ``stereo_align_hybrid.npz``.
      Requires both the video and the outline + trajectory data.

    The legacy ``use_outline=True`` flag is equivalent to
    ``mode="outline"``.

    ``mask_dilate_px`` (hybrid only): the outline mask is dilated by
    this many pixels before being applied to the Pass-2 raw-image
    crops -- background pixels outside the dilated mask are weighted
    to zero so phase correlation is driven only by hand-region
    content.  Set to 0 to disable.

    ``gauss_center_weight`` (image + hybrid): in [0, 1], strength of
    a 2D Gaussian centred on each joint's MP label that further
    weights the Pass-2 phase correlation toward pixels near the
    joint.  0 = uniform (within the mask / Hanning); 1 = sigma ~0.3
    of the per-joint half-size.  Ignored in outline mode.

    Returns the saved npz path as a string.
    """
    # Resolve mode (legacy boolean takes precedence if True).
    if use_outline:
        mode = "outline"
    if mode not in ("image", "outline", "hybrid"):
        raise ValueError(f"Unknown stereo mode: {mode!r}")
    needs_outline = mode in ("outline", "hybrid")
    needs_video   = mode in ("image", "hybrid")
    import json as _json
    from ..config import get_settings
    from .video import build_trial_map
    from .skeleton_data import _skeleton_dir

    settings = get_settings()
    tmap = build_trial_map(subject_name)
    if trial_idx < 0 or trial_idx >= len(tmap):
        raise ValueError(f"trial_idx {trial_idx} out of range ({len(tmap)} trials)")
    trial = tmap[trial_idx]
    start_frame = int(trial["start_frame"])
    n_frames    = int(trial["frame_count"])
    video_path  = trial["video_path"]
    stem        = trial["trial_name"]

    from .mediapipe_prelabel import (
        has_mediapipe_data, load_mediapipe_prelabels,
        load_mediapipe_combined_prelabels,
    )
    if not has_mediapipe_data(subject_name):
        raise FileNotFoundError(f"MediaPipe prelabels not found for {subject_name}")
    # Prefer the per-trial MP Combined fusion (forward + reverse +
    # static + cropped, per-camera bone-length tie-break) when it
    # exists — that's the canonical per-camera input the rest of
    # the app now treats as the "MP" labels.  Falls back to the
    # forward-only pass for trials that don't have a combined npz
    # yet.
    mp_source = "combined"
    mp = load_mediapipe_combined_prelabels(subject_name)
    if mp is not None:
        # Per-trial loader stitches a subject-wide array and leaves
        # NaN for trials that don't have a combined.npz on disk.  If
        # the window we actually care about is all-NaN, combined
        # doesn't cover this trial — fall back to forward instead of
        # running phase correlation against a sea of NaN.
        _os_win = mp["OS_landmarks"][start_frame:start_frame + max(n_frames, 1)]
        if _os_win.size and not np.isfinite(_os_win).any():
            mp = None
            mp_source = "forward (combined empty for this trial)"
    else:
        mp_source = "forward (combined missing)"
    if mp is None:
        mp = load_mediapipe_prelabels(subject_name)
    if mp is None:
        raise FileNotFoundError(f"MediaPipe prelabels not found for {subject_name}")
    print(f"[stereo_align] {subject_name} frames {start_frame}+{n_frames}: "
          f"MP source = {mp_source}", flush=True)
    OS_lm = mp["OS_landmarks"]
    OD_lm = mp["OD_landmarks"]
    N_total, N_joints, _ = OS_lm.shape
    if start_frame + n_frames > N_total:
        n_frames = max(0, N_total - start_frame)

    # Outline data prerequisites (outline + hybrid modes): load
    # outlines.json + camera trajectory.
    out_frames = None
    H_L_all = H_R_all = None
    h_full = half_w = None
    if needs_outline:
        from .background import _preproc_dir as _bg_preproc_dir
        from .camera_motion import load_camera_trajectory
        outlines_path = _bg_preproc_dir(subject_name, stem) / "outlines.json"
        if not outlines_path.exists():
            raise FileNotFoundError(
                f"No outlines.json for {subject_name}/{stem} -- "
                "run 'Compute boundary - all frames' in Pre-proc first.")
        with open(outlines_path) as _fh:
            outlines = _json.load(_fh)
        out_frames = outlines.get("frames") or []
        traj = load_camera_trajectory(subject_name, stem)
        if traj is None:
            raise FileNotFoundError(
                f"No camera_trajectory.npz for {subject_name}/{stem} -- "
                "run Compute Trajectory + Stabilize in Pre-proc first.")
        H_L_all = traj["H_to_ref_L"]
        H_R_all = traj["H_to_ref_R"]
        n_frames = min(n_frames, len(out_frames),
                        int(H_L_all.shape[0]), int(H_R_all.shape[0]))

    cap = None
    if needs_video:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    elif needs_outline:
        # Outline-only mode needs the frame dimensions for mask
        # rasterisation -- read once and close.
        _probe = cv2.VideoCapture(video_path)
        if not _probe.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        _ok, _first = _probe.read()
        _probe.release()
        if not _ok or _first is None:
            raise RuntimeError("Failed to read first frame for dimensions")
        h_full, _full_w = _first.shape[:2]
        half_w = _full_w // 2

    # Pre-build the Hanning windows.  Per-joint sizes are cached by
    # half-size so duplicates (every non-wrist joint shares one) only
    # build the FFT window once.
    per_joint_halves = [_crop_half_for_joint(j) for j in range(N_joints)]
    _win_cache: dict[int, np.ndarray] = {}
    def _win_for_half(h: int) -> np.ndarray:
        if h not in _win_cache:
            s = 2 * h + 1
            _win_cache[h] = cv2.createHanningWindow((s, s), cv2.CV_32F)
        return _win_cache[h]
    hand_win_size = 2 * _HAND_CROP_HALF + 1
    hand_window = cv2.createHanningWindow((hand_win_size, hand_win_size), cv2.CV_32F)

    shifts = np.full((n_frames, N_joints, 2), np.nan, dtype=np.float32)
    response = np.full((n_frames, N_joints), np.nan, dtype=np.float32)
    hand_shifts = np.full((n_frames, 2), np.nan, dtype=np.float32)
    hand_response = np.full(n_frames, np.nan, dtype=np.float32)

    out_dir = _skeleton_dir(subject_name) / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / {
        "image":   "stereo_align.npz",
        "outline": "stereo_align_outline.npz",
        "hybrid":  "stereo_align_hybrid.npz",
    }[mode]
    # Pass 2 align fn: outline-mask phase-corr ONLY for pure outline
    # mode.  Hybrid uses raw-image phase-corr at the per-joint stage
    # (only its Pass 1 is outline-driven).
    align_fn = _align_phase_mask if mode == "outline" else _align_phase

    last_pct = -1
    for fi in range(n_frames):
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Job cancelled")
        # Outline polygons (Pass 1 voting + Pass 2 mask source for
        # pure outline mode).
        os_poly = od_poly = None
        if needs_outline:
            of = out_frames[fi] if fi < len(out_frames) else {}
            os_poly = _warp_poly_ref_to_orig(of.get("OS"), H_L_all[fi])
            od_poly = _warp_poly_ref_to_orig(of.get("OD"), H_R_all[fi])
        # Pass 2 source image: raw frame (image / hybrid) or outline
        # mask (pure outline mode).
        # Hybrid keeps the raw frame halves and the dilated outline
        # masks; the per-joint loop below crops both and routes through
        # _align_phase_weighted so background pixels contribute literal
        # zero to the FFT -- cleaner than mean-filling the image.
        os_mask = od_mask = None
        if needs_video:
            ok, frame = cap.read()
            if not ok:
                break
            h_full, _fw = frame.shape[:2]
            half_w = _fw // 2
            img_OS = frame[:, :half_w, :]
            img_OD = frame[:, half_w:, :]
            if mode == "hybrid":
                os_mask = _dilate_mask(_outline_mask_image(os_poly, h_full, half_w),
                                       mask_dilate_px)
                od_mask = _dilate_mask(_outline_mask_image(od_poly, h_full, half_w),
                                       mask_dilate_px)
        else:
            img_OS = _outline_mask_image(os_poly, h_full, half_w)
            img_OD = _outline_mask_image(od_poly, h_full, half_w)

        gf = start_frame + fi
        os_frame = OS_lm[gf]
        od_frame = OD_lm[gf]
        # MP centroid per camera (always the mean of valid MP joints).
        # Used by Pass 1 to anchor the hand-wide alignment AND by
        # Pass 2 to estimate the *expected* per-joint disparity.
        # ``base_dx/dy`` from Pass 1 is the residual phase-corr shift
        # AFTER MP-centroid alignment — NOT the full inter-camera
        # image disparity.  To predict where a joint should land in
        # the partner camera we need
        #     expected_disp = (od_centroid - os_centroid) + base.
        _os_valid_pre = ~np.isnan(os_frame[:, 0])
        _od_valid_pre = ~np.isnan(od_frame[:, 0])
        if _os_valid_pre.any() and _od_valid_pre.any():
            _os_mp_cent_xy = os_frame[_os_valid_pre].mean(axis=0)
            _od_mp_cent_xy = od_frame[_od_valid_pre].mean(axis=0)
        else:
            _os_mp_cent_xy = _od_mp_cent_xy = None

        # ── Pass 1: hand-wide alignment ────────────────────────────
        # Image mode: phase-correlate two big crops around the hand
        # centroid.  Outline mode: vote pairwise offsets of the
        # densely-sampled ENTIRE outline (same approach the preproc
        # page's cross-camera overlay uses) -- no crop, no patch.
        os_valid = ~np.isnan(os_frame[:, 0])
        od_valid = ~np.isnan(od_frame[:, 0])
        if needs_outline:
            # Vote needs MP centroids so the returned (dx,dy) is the
            # residual after MP-centering -- same convention as the
            # image-mode phase-corr it replaces (which crops are taken
            # around the MP centroid in each camera).  Without this,
            # the vote would return the absolute OS->OD translation
            # which the downstream ``stereo_R = mp_R + shifts`` math
            # is not expecting.
            if os_valid.any() and od_valid.any():
                os_mp_cent = os_frame[os_valid].mean(axis=0)
                od_mp_cent = od_frame[od_valid].mean(axis=0)
                try:
                    ref_dx, ref_dy, ref_r = _voting_align(
                        os_poly, od_poly, os_mp_cent, od_mp_cent)
                except Exception as e:
                    logger.debug(f"voting align failed at frame {gf}: {e}")
                    ref_dx = ref_dy = 0.0
                    ref_r = float('nan')
                hand_shifts[fi, 0] = ref_dx
                hand_shifts[fi, 1] = ref_dy
                hand_response[fi] = ref_r
                base_dx = float(ref_dx) if np.isfinite(ref_dx) else None
                base_dy = float(ref_dy) if np.isfinite(ref_dy) else None
            else:
                # No MP keypoints this frame -- can't define the
                # MP-centered residual, so leave the per-joint loop to
                # skip this frame's joints anyway.
                base_dx = base_dy = None
        elif os_valid.any() and od_valid.any():
            os_cent = os_frame[os_valid].mean(axis=0)
            od_cent = od_frame[od_valid].mean(axis=0)
            os_big = _crop(img_OS, int(round(float(os_cent[0]))),
                                    int(round(float(os_cent[1]))), _HAND_CROP_HALF)
            od_big = _crop(img_OD, int(round(float(od_cent[0]))),
                                    int(round(float(od_cent[1]))), _HAND_CROP_HALF)
            try:
                ref_dx, ref_dy, ref_r = align_fn(os_big, od_big, hand_window)
            except Exception as e:
                logger.debug(f"hand-wide phase corr failed at frame {gf}: {e}")
                ref_dx = ref_dy = 0.0
                ref_r = float('nan')
            hand_shifts[fi, 0] = ref_dx
            hand_shifts[fi, 1] = ref_dy
            hand_response[fi] = ref_r
            base_dx = float(ref_dx)
            base_dy = float(ref_dy)
        else:
            base_dx = base_dy = None

        # ── Pass 2: per-joint with clamping ───────────────────────
        for j in range(N_joints):
            os_x = os_frame[j, 0]; os_y = os_frame[j, 1]
            od_x = od_frame[j, 0]; od_y = od_frame[j, 1]
            if np.isnan(os_x) or np.isnan(od_x):
                continue
            os_cx, os_cy = int(round(float(os_x))), int(round(float(os_y)))
            od_cx, od_cy = int(round(float(od_x))), int(round(float(od_y)))
            jh = per_joint_halves[j]
            # If MP labels disagree by more than the crop half-size,
            # the OS and OD crops at the literal label positions sit
            # on entirely non-overlapping scene patches and phase
            # correlation can only return noise (user-visible
            # symptom: "Stereo correct gets tripped up when MP
            # labels are too far apart").  Use 2D jump magnitude
            # from the previous frame to attribute the bad label
            # to one camera, then re-anchor THAT camera's crop on
            # the partner's prediction (os_label + base_shift, or
            # od_label - base_shift) so both crops see the same
            # world patch.  Ambiguous case (similar jumps on both
            # sides): fall back to no-reanchor — phase corr returns
            # whatever and refine_clamp_px bounds the damage.
            #
            # The shift compensation we add downstream is
            # algebraically identical for both blame choices:
            # (predicted disparity) − (observed disparity), so we
            # compute it once and apply it whenever any re-anchor
            # happened (zero otherwise).
            od_cx_used, od_cy_used = od_cx, od_cy
            os_cx_used, os_cy_used = os_cx, os_cy
            ox = oy = 0
            if (base_dx is not None and base_dy is not None
                    and _os_mp_cent_xy is not None
                    and _od_mp_cent_xy is not None):
                # Expected per-joint disparity = inter-camera centroid
                # offset + residual from the hand-wide pass.  Earlier
                # versions of this code used base_dx directly as the
                # disparity, which over-triggered the re-anchor for
                # any camera pair with non-rectified geometry.
                _exp_disp_x = (float(_od_mp_cent_xy[0])
                                - float(_os_mp_cent_xy[0])
                                + float(base_dx))
                _exp_disp_y = (float(_od_mp_cent_xy[1])
                                - float(_os_mp_cent_xy[1])
                                + float(base_dy))
                pred_od_cx = os_cx + _exp_disp_x
                pred_od_cy = os_cy + _exp_disp_y
                gap = (((od_cx - pred_od_cx) ** 2
                        + (od_cy - pred_od_cy) ** 2) ** 0.5)
                if gap > jh:
                    # 2D jump from previous frame, per-cam.  Mirrors
                    # mp_filter._camera_blame_from_2d_step's k=2.0
                    # convention — the larger jump indicates the
                    # likely-mislabeled camera.
                    _K_BLAME = 2.0
                    step_OS = step_OD = None
                    if fi > 0:
                        p_os = OS_lm[start_frame + fi - 1, j]
                        p_od = OD_lm[start_frame + fi - 1, j]
                        if (np.isfinite(p_os[0]) and np.isfinite(p_os[1])
                                and np.isfinite(p_od[0]) and np.isfinite(p_od[1])):
                            step_OS = float(np.hypot(os_x - p_os[0], os_y - p_os[1]))
                            step_OD = float(np.hypot(od_x - p_od[0], od_y - p_od[1]))
                    blamed = None   # 'OD' / 'OS' / None
                    if step_OS is not None and step_OD is not None:
                        if step_OD > _K_BLAME * max(step_OS, 1.0):
                            blamed = 'OD'
                        elif step_OS > _K_BLAME * max(step_OD, 1.0):
                            blamed = 'OS'
                    if blamed == 'OD':
                        od_cx_used = int(round(pred_od_cx))
                        od_cy_used = int(round(pred_od_cy))
                    elif blamed == 'OS':
                        # Mirror image of the OD-blame anchor: OS
                        # predicted from OD by subtracting the same
                        # full expected disparity.
                        os_cx_used = int(round(od_cx - _exp_disp_x))
                        os_cy_used = int(round(od_cy - _exp_disp_y))
                    # else: ambiguous (or no prev frame) → no
                    # re-anchor.  Phase corr runs on whatever
                    # patches sit at the literal labels and the
                    # refine_clamp_px bound below limits the per-
                    # joint shift to ±6 px of the hand-wide prior.
                    if blamed is not None:
                        # (expected disparity) − (observed disparity),
                        # rounded to match the integer crop centers.
                        # NB: expected_disp here includes the centroid
                        # offset between cameras, NOT just base_dx.
                        ox = int(round(_exp_disp_x)) - (od_cx - os_cx)
                        oy = int(round(_exp_disp_y)) - (od_cy - os_cy)
            os_crop = _crop(img_OS, os_cx_used, os_cy_used, jh)
            od_crop = _crop(img_OD, od_cx_used, od_cy_used, jh)
            # Hybrid: weight the per-joint phase corr by the dilated
            # outline mask so BG pixels contribute literal zero to the
            # FFT.  Failsafe: if either MP label falls OUTSIDE the
            # mask, skip the mask for THAT joint (the Gaussian centre-
            # weight is still applied -- it doesn't zero anything out).
            use_mask = False
            if mode == "hybrid" and os_mask is not None and od_mask is not None:
                h_m, w_m = os_mask.shape
                in_os = (0 <= os_cx < w_m and 0 <= os_cy < h_m
                         and os_mask[os_cy, os_cx] > 0)
                in_od = (0 <= od_cx < w_m and 0 <= od_cy < h_m
                         and od_mask[od_cy, od_cx] > 0)
                use_mask = in_os and in_od
            try:
                if mode == "outline":
                    # Outline-only mode keeps its bespoke mask-image
                    # phase corr -- the inputs ARE silhouettes already.
                    raw_dx, raw_dy, r = align_fn(os_crop, od_crop, _win_for_half(jh))
                else:
                    # Image + hybrid go through the unified weighted
                    # phase corr (mask optional, Gaussian optional).
                    os_m_crop = od_m_crop = None
                    if use_mask:
                        os_m_crop = _crop(os_mask, os_cx_used, os_cy_used, jh)
                        od_m_crop = _crop(od_mask, od_cx_used, od_cy_used, jh)
                    raw_dx, raw_dy, r = _align_phase_weighted(
                        os_crop, od_crop, _win_for_half(jh),
                        os_m_crop, od_m_crop,
                        gauss_strength=float(gauss_center_weight))
            except Exception as e:
                logger.debug(f"per-joint phase corr failed at frame {gf} joint {j}: {e}")
                continue
            if base_dx is not None:
                dx = base_dx + max(-_REFINE_CLAMP_PX,
                                    min(_REFINE_CLAMP_PX, raw_dx - base_dx))
                dy = base_dy + max(-_REFINE_CLAMP_PX,
                                    min(_REFINE_CLAMP_PX, raw_dy - base_dy))
            else:
                dx, dy = raw_dx, raw_dy
            # Compensate for either crop being re-anchored on the
            # partner's prediction above (non-zero only when MP
            # labels disagreed AND 2D-jump blame picked a side).
            # ox / oy are intentionally NOT inside the refine clamp:
            # the clamp limits per-joint *refinement noise* on top
            # of the global prior, but the re-anchor offset IS the
            # signal we want when MP genuinely failed.
            dx += ox
            dy += oy
            shifts[fi, j, 0] = dx
            shifts[fi, j, 1] = dy
            response[fi, j] = r

        pct = int(100.0 * (fi + 1) / max(1, n_frames))
        if pct != last_pct and progress_callback is not None:
            last_pct = pct
            try: progress_callback(pct)
            except Exception: pass

    if cap is not None:
        cap.release()

    np.savez_compressed(
        str(out_path),
        shifts=shifts,
        response=response,
        hand_shifts=hand_shifts,
        hand_response=hand_response,
        crop_half=np.array(crop_half, dtype=np.int32),
        crop_halves_per_joint=np.array(per_joint_halves, dtype=np.int32),
        hand_crop_half=np.array(_HAND_CROP_HALF, dtype=np.int32),
        refine_clamp_px=np.array(_REFINE_CLAMP_PX, dtype=np.int32),
        start_frame=np.array(start_frame, dtype=np.int32),
        n_frames=np.array(n_frames, dtype=np.int32),
        mask_dilate_px=np.array(int(mask_dilate_px) if mode == "hybrid" else -1,
                                 dtype=np.int32),
        gauss_center_weight=np.array(
            float(gauss_center_weight) if mode != "outline" else -1.0,
            dtype=np.float32),
    )
    valid = int(np.sum(~np.isnan(shifts[:, :, 0])))
    logger.info(
        f"stereo_align ({mode}) saved: {out_path}  shape={shifts.shape}  "
        f"valid={valid}/{n_frames * N_joints}"
    )
    return str(out_path)


def load_stereo_align(subject_name: str, trial_idx: int,
                       use_outline: bool = False,
                       mode: str | None = None) -> dict | None:
    """Load saved stereo-align npz for a trial.  Returns None if not
    present.

    ``mode`` selects between ``"image"``, ``"outline"``, and
    ``"hybrid"`` variants (``stereo_align.npz``,
    ``stereo_align_outline.npz``, ``stereo_align_hybrid.npz``).  The
    legacy ``use_outline=True`` flag is equivalent to
    ``mode="outline"``."""
    if mode is None:
        mode = "outline" if use_outline else "image"
    if mode not in ("image", "outline", "hybrid"):
        raise ValueError(f"Unknown stereo mode: {mode!r}")
    from .video import build_trial_map
    from .skeleton_data import _skeleton_dir
    tmap = build_trial_map(subject_name)
    if trial_idx < 0 or trial_idx >= len(tmap):
        return None
    stem = tmap[trial_idx]["trial_name"]
    fname = {
        "image":   "stereo_align.npz",
        "outline": "stereo_align_outline.npz",
        "hybrid":  "stereo_align_hybrid.npz",
    }[mode]
    path = _skeleton_dir(subject_name) / stem / fname
    if not path.exists():
        return None
    try:
        d = np.load(str(path))
        result = {
            "shifts": d["shifts"],
            "response": d["response"],
            "start_frame": int(d["start_frame"]) if "start_frame" in d.files else 0,
            "n_frames": int(d["n_frames"]) if "n_frames" in d.files else int(d["shifts"].shape[0]),
        }
        if "crop_halves_per_joint" in d.files:
            result["crop_halves_per_joint"] = [int(x) for x in d["crop_halves_per_joint"]]
        if "crop_half" in d.files:
            result["crop_half"] = int(d["crop_half"])
        if "hand_crop_half" in d.files:
            result["hand_crop_half"] = int(d["hand_crop_half"])
        # Per-run knobs from the Stereo panel — the Labels UI reads
        # these back so opening the panel restores the exact settings
        # used to bake the current npz for the selected mode.
        if "mask_dilate_px" in d.files:
            result["mask_dilate_px"] = int(d["mask_dilate_px"])
        if "gauss_center_weight" in d.files:
            result["gauss_center_weight"] = float(d["gauss_center_weight"])
        return result
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
        return None
