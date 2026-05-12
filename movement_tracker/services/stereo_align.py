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

Output (per trial, saved to ``<dlc>/<subject>/mano/<stem>/stereo_align.npz``)::

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


# ── Main entry point ───────────────────────────────────────────────────────

def run_stereo_align(subject_name: str, trial_idx: int,
                      progress_callback=None,
                      crop_half: int = _DEFAULT_CROP_HALF,
                      cancel_event=None) -> str:
    """Run image-based stereo label alignment for one trial.

    Returns the saved npz path as a string.
    """
    from ..config import get_settings
    from .video import build_trial_map
    from .mano_data import _mano_dir

    settings = get_settings()
    tmap = build_trial_map(subject_name)
    if trial_idx < 0 or trial_idx >= len(tmap):
        raise ValueError(f"trial_idx {trial_idx} out of range ({len(tmap)} trials)")
    trial = tmap[trial_idx]
    start_frame = int(trial["start_frame"])
    n_frames    = int(trial["frame_count"])
    video_path  = trial["video_path"]
    stem        = trial["trial_name"]

    npz_path = settings.dlc_path / subject_name / "mediapipe_prelabels.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"MediaPipe prelabels not found: {npz_path}")
    mp = np.load(str(npz_path))
    OS_lm = mp["OS_landmarks"]
    OD_lm = mp["OD_landmarks"]
    N_total, N_joints, _ = OS_lm.shape
    if start_frame + n_frames > N_total:
        n_frames = max(0, N_total - start_frame)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

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

    out_dir = _mano_dir(subject_name) / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "stereo_align.npz"

    last_pct = -1
    for fi in range(n_frames):
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Job cancelled")
        ok, frame = cap.read()
        if not ok:
            break
        h, full_w = frame.shape[:2]
        half_w = full_w // 2
        img_OS = frame[:, :half_w, :]
        img_OD = frame[:, half_w:, :]

        gf = start_frame + fi
        os_frame = OS_lm[gf]
        od_frame = OD_lm[gf]

        # ── Pass 1: hand-wide alignment ────────────────────────────
        os_valid = ~np.isnan(os_frame[:, 0])
        od_valid = ~np.isnan(od_frame[:, 0])
        if os_valid.any() and od_valid.any():
            os_cent = os_frame[os_valid].mean(axis=0)
            od_cent = od_frame[od_valid].mean(axis=0)
            os_big = _crop(img_OS, int(round(float(os_cent[0]))),
                                    int(round(float(os_cent[1]))), _HAND_CROP_HALF)
            od_big = _crop(img_OD, int(round(float(od_cent[0]))),
                                    int(round(float(od_cent[1]))), _HAND_CROP_HALF)
            try:
                ref_dx, ref_dy, ref_r = _align_phase(os_big, od_big, hand_window)
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
            os_crop = _crop(img_OS, os_cx, os_cy, jh)
            od_crop = _crop(img_OD, od_cx, od_cy, jh)
            try:
                raw_dx, raw_dy, r = _align_phase(os_crop, od_crop, _win_for_half(jh))
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
            shifts[fi, j, 0] = dx
            shifts[fi, j, 1] = dy
            response[fi, j] = r

        pct = int(100.0 * (fi + 1) / max(1, n_frames))
        if pct != last_pct and progress_callback is not None:
            last_pct = pct
            try: progress_callback(pct)
            except Exception: pass

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
    )
    valid = int(np.sum(~np.isnan(shifts[:, :, 0])))
    logger.info(
        f"stereo_align saved: {out_path}  shape={shifts.shape}  "
        f"valid={valid}/{n_frames * N_joints}"
    )
    return str(out_path)


def load_stereo_align(subject_name: str, trial_idx: int) -> dict | None:
    """Load saved ``stereo_align.npz`` for a trial.  Returns None if not present."""
    from .video import build_trial_map
    from .mano_data import _mano_dir
    tmap = build_trial_map(subject_name)
    if trial_idx < 0 or trial_idx >= len(tmap):
        return None
    stem = tmap[trial_idx]["trial_name"]
    path = _mano_dir(subject_name) / stem / "stereo_align.npz"
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
        return result
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
        return None
