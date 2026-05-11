"""Per-frame HRnet bbox helpers.

Produces a tight per-frame crop rectangle (one per video frame) from
MediaPipe hand landmarks, with temporal smoothing and gap-fill so the
bounding box doesn't jitter or vanish on frames where MP failed.

The HRnet inference path crops each frame by its own bbox before the
network sees it, then heatmaps stay frame-aligned to that crop.  Loaders
read the per-frame bboxes back from ``hand_crop.json`` so the
heatmap-pixel ↔ image-pixel conversion is correct on every frame.

Schema written to ``hand_crop.json``::

    {
        "crop_L": [x1, y1, x2, y2],          # legacy: union bbox over the trial
        "crop_R": [x1, y1, x2, y2],
        "crop_L_perframe": [[x1,y1,x2,y2], ...],   # new: one entry per frame
        "crop_R_perframe": [[...], ...]
    }

The two ``crop_*`` fields are kept for back-compat with older readers.
"""
from __future__ import annotations

import numpy as np


BBOX_PADDING = 0.25         # ¼ of the longer landmark side on each axis →
                            # final bbox = 1.5× the landmark span (close to
                            # HRnet hand-pose training-time crop scale).
TEMPORAL_WINDOW = 3         # ±N frames for median smoothing of the bbox
MIN_BBOX_PX = 64            # minimum bbox dimension to avoid degenerate crops


def _per_frame_raw_centers_and_spans(landmarks: np.ndarray):
    """Per-frame (cx, cy, span) where span = max(landmark_w, landmark_h).
    NaN rows where MP failed."""
    N = landmarks.shape[0]
    centers = np.full((N, 2), np.nan, dtype=np.float32)
    spans = np.full(N, np.nan, dtype=np.float32)
    for f in range(N):
        pts = landmarks[f]
        if pts is None or np.isnan(pts).any():
            continue
        x1, y1 = pts[:, 0].min(), pts[:, 1].min()
        x2, y2 = pts[:, 0].max(), pts[:, 1].max()
        centers[f] = [(x1 + x2) * 0.5, (y1 + y2) * 0.5]
        spans[f] = max(x2 - x1, y2 - y1)
    return centers, spans


def _hold_fill(bb: np.ndarray) -> np.ndarray:
    """Fill NaN frames by holding the last-good value (and forward-fill
    the leading-NaN block from the first good value)."""
    out = bb.copy()
    last = None
    for f in range(out.shape[0]):
        if np.isnan(out[f, 0]):
            if last is not None:
                out[f] = last
        else:
            last = out[f].copy()
    # Forward-fill leading NaN block
    first_good = None
    for f in range(out.shape[0]):
        if not np.isnan(out[f, 0]):
            first_good = out[f].copy()
            break
    if first_good is not None:
        for f in range(out.shape[0]):
            if np.isnan(out[f, 0]):
                out[f] = first_good
            else:
                break
    return out


def _temporal_median(bb: np.ndarray, half_win: int) -> np.ndarray:
    """Per-axis temporal median over a ±``half_win`` frame window.
    Reduces sub-pixel jitter from MP wobble."""
    if half_win <= 0:
        return bb
    N = bb.shape[0]
    out = np.empty_like(bb)
    for f in range(N):
        lo, hi = max(0, f - half_win), min(N, f + half_win + 1)
        win = bb[lo:hi]
        out[f, 0] = np.nanmedian(win[:, 0])
        out[f, 1] = np.nanmedian(win[:, 1])
        out[f, 2] = np.nanmedian(win[:, 2])
        out[f, 3] = np.nanmedian(win[:, 3])
    return out


def _enforce_min_size(bb: np.ndarray, min_px: int) -> np.ndarray:
    """Grow tiny bboxes (degenerate crops) symmetrically about their center
    to at least ``min_px`` on each axis."""
    out = bb.copy()
    for f in range(out.shape[0]):
        if np.isnan(out[f, 0]):
            continue
        w = out[f, 2] - out[f, 0]
        h = out[f, 3] - out[f, 1]
        if w < min_px:
            cx = (out[f, 0] + out[f, 2]) * 0.5
            out[f, 0] = cx - min_px * 0.5
            out[f, 2] = cx + min_px * 0.5
        if h < min_px:
            cy = (out[f, 1] + out[f, 3]) * 0.5
            out[f, 1] = cy - min_px * 0.5
            out[f, 3] = cy + min_px * 0.5
    return out


def compute_per_frame_bboxes(landmarks: np.ndarray,
                              padding: float = BBOX_PADDING,
                              smooth_window: int = TEMPORAL_WINDOW,
                              min_size_px: int = MIN_BBOX_PX) -> np.ndarray:
    """Per-frame bbox ``(N, 4)`` from MediaPipe landmarks ``(N, 21, 2)``.

    Bbox **size is constant across the trial**; only the *center* tracks
    the hand frame-to-frame.  Rationale: for finger-tapping the subject's
    hand stays at roughly the same camera distance, so the actual hand
    pixel-size is stable.  The landmark span oscillates with the tap
    (open vs. closed fingers) — letting the bbox follow that creates
    rhythm-locked scale noise that biases tap-amplitude / timing measures
    at exactly the frequency we're trying to measure.  Holding size fixed
    keeps the heatmap-pixel ↔ image-pixel ratio constant and lets HRnet
    see the hand at a consistent training-time scale.

    Steps:
      1. Per-frame center (cx, cy) + span = max(landmark_w, landmark_h).
      2. Trial-wide fixed side = 95th-percentile(span) × (1 + 2·padding).
         Using p95 (not max) protects against single-frame outliers from
         noisy MP detections.
      3. Hold-fill NaN centers from neighbours and temporally median them.
      4. Build square bbox = (center, fixed_side); enforce min size.

    Returns float32 ``(N, 4)`` array.
    """
    centers, spans = _per_frame_raw_centers_and_spans(landmarks)

    # Fixed trial-wide side from the 95th-percentile span.
    valid_spans = spans[~np.isnan(spans)]
    if valid_spans.size == 0:
        # No MP detections — degenerate case, return all-NaN.
        return np.full((landmarks.shape[0], 4), np.nan, dtype=np.float32)
    p95_span = float(np.percentile(valid_spans, 95))
    fixed_side = p95_span * (1.0 + 2.0 * padding)
    fixed_half = fixed_side * 0.5

    # Hold-fill missing centers, then median-smooth to reduce MP jitter.
    # Re-use the existing helpers by packing centers into a (N, 4)-shaped
    # ndarray of (cx, cy, cx, cy) so _hold_fill / _temporal_median work
    # unchanged on it.
    cc = np.full((centers.shape[0], 4), np.nan, dtype=np.float32)
    cc[:, 0] = centers[:, 0]; cc[:, 1] = centers[:, 1]
    cc[:, 2] = centers[:, 0]; cc[:, 3] = centers[:, 1]
    cc = _hold_fill(cc)
    cc = _temporal_median(cc, smooth_window)

    out = np.empty((cc.shape[0], 4), dtype=np.float32)
    out[:, 0] = cc[:, 0] - fixed_half
    out[:, 1] = cc[:, 1] - fixed_half
    out[:, 2] = cc[:, 0] + fixed_half
    out[:, 3] = cc[:, 1] + fixed_half
    return _enforce_min_size(out, min_size_px).astype(np.float32)


def union_bbox(bboxes_per_frame: np.ndarray) -> list[float] | None:
    """Compute the trial-wide union bbox from per-frame bboxes (used as the
    legacy ``crop_L``/``crop_R`` field for back-compat readers)."""
    valid = ~np.isnan(bboxes_per_frame[:, 0])
    if not valid.any():
        return None
    bb = bboxes_per_frame[valid]
    return [float(bb[:, 0].min()), float(bb[:, 1].min()),
            float(bb[:, 2].max()), float(bb[:, 3].max())]


def read_per_frame_bboxes(crop_info: dict, side_key: str,
                           n_frames: int) -> np.ndarray | None:
    """Read per-frame bboxes from a ``hand_crop.json`` dict.

    ``side_key`` is ``"crop_L"`` or ``"crop_R"``.  Prefers the
    ``"<side_key>_perframe"`` array; if missing (legacy file), broadcasts
    the static ``side_key`` to ``(n_frames, 4)``.
    Returns ``None`` when neither field is present.
    """
    pf_key = f"{side_key}_perframe"
    if pf_key in crop_info and crop_info[pf_key] is not None:
        arr = np.asarray(crop_info[pf_key], dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] == 4:
            # Tolerate length mismatch — broadcast last frame on shortfall,
            # truncate on excess.
            if arr.shape[0] >= n_frames:
                return arr[:n_frames]
            pad = np.tile(arr[-1:], (n_frames - arr.shape[0], 1))
            return np.concatenate([arr, pad], axis=0)
    static = crop_info.get(side_key)
    if static is None:
        return None
    arr = np.asarray(static, dtype=np.float32).reshape(1, 4)
    return np.tile(arr, (n_frames, 1))
