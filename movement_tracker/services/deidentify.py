"""Face de-identification with hand protection for stereo/single-camera videos.

Adapted from the skeleton pipeline's deidentify_videos.py, simplified for the
movement-tracker context. Uses only MediaPipe (no HRNet dependency).

Usage:
    from movement_tracker.services.deidentify import deidentify_video
    deidentify_video("input.mp4", "output.mp4")
"""
from __future__ import annotations

import logging
import os
import subprocess

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

logger = logging.getLogger(__name__)


# Cache: ffmpeg_path -> chosen encoder name.  Probing ffmpeg -encoders
# is ~50 ms so we don't want to do it per-render.
_H264_ENCODER_CACHE: dict[str, str] = {}


def _pick_h264_encoder(ffmpeg: str) -> str:
    """Return the best available H.264 encoder name for this ffmpeg.

    Default-channel conda ffmpeg on Windows ships WITHOUT libx264
    (license reasons), so hard-coding ``-c:v libx264`` blows up with
    "Unknown encoder 'libx264'".  Probe ``ffmpeg -encoders`` once and
    pick the first available of:

      libx264     -- best quality, but missing from minimal builds
      libopenh264 -- royalty-free H.264, in nearly every build
      h264_mf     -- Windows Media Foundation hardware encoder

    Raises RuntimeError if none is found, with a fix hint pointing
    at conda-forge ffmpeg.
    """
    cached = _H264_ENCODER_CACHE.get(ffmpeg)
    if cached:
        return cached
    try:
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
        output = (proc.stdout or "") + (proc.stderr or "")
    except (OSError, subprocess.TimeoutExpired) as e:
        raise RuntimeError(f"Could not probe ffmpeg encoders: {e}")
    # Parse ``ffmpeg -encoders`` output.  Each encoder line begins
    # with " V....." / " A....." (lots of dots-padding for the flags
    # column), then the encoder name, then a description.  Splitting
    # on whitespace and picking the name token avoids the brittle
    # "look for ' libx264 '" check that missed multi-space padding
    # in some builds.
    available: set[str] = set()
    for line in output.splitlines():
        parts = line.split()
        # Encoder lines look like ['V.....', 'libx264', 'H.264', ...]
        # (or 'V..S..' etc.).  Take the second token as the name when
        # the first looks like a flags column.
        if len(parts) >= 2 and parts[0] and parts[0][0] in ("V", "A", "S"):
            available.add(parts[1])
    candidates = ("libx264", "libopenh264", "h264_mf")
    for name in candidates:
        if name in available:
            _H264_ENCODER_CACHE[ffmpeg] = name
            logger.info(f"Using H.264 encoder: {name}")
            return name
    # Diagnostic: log the first ~30 video-encoder lines so the user
    # / dev can see what THIS ffmpeg actually supports.
    v_lines = [l for l in output.splitlines()
               if l.lstrip().startswith("V")][:30]
    sample = "\n".join(v_lines)
    raise RuntimeError(
        "No H.264 encoder available in this ffmpeg build "
        f"({ffmpeg}).  Tried: {', '.join(candidates)}.  Fix on a "
        "conda env: 'conda install -c conda-forge -y ffmpeg' (the "
        "conda-forge build ships libx264).\n"
        f"Detected video encoders in this build:\n{sample}"
    )

# ── Configuration ────────────────────────────────────────────────────────
FACE_CONF_THRESHOLD = 0.2
FACE_MODEL_SELECTION = 1        # full-range model (up to 5m)
BLUR_KERNEL_SIZE = 99
BLUR_SIGMA = 50
FACE_BOX_EXPAND = 0.4
FACE_BOX_EXPAND_UP = 0.6
HAND_HULL_DILATION_PX = 30
TEMPORAL_SIGMA_FRAMES = 3.0
FEATHER_KERNEL = 15
IOU_MATCH_THRESHOLD = 0.1


# ── Face detection ───────────────────────────────────────────────────────

def _expand_face_bbox(bbox, w, h):
    """Expand face bbox for safe coverage."""
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    x1 -= bw * FACE_BOX_EXPAND
    x2 += bw * FACE_BOX_EXPAND
    y1 -= bh * (FACE_BOX_EXPAND + FACE_BOX_EXPAND_UP)
    y2 += bh * FACE_BOX_EXPAND
    return (max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2)))


def _detect_faces_in_half(rgb_half, face_det, w, h):
    """Run face detection on a single frame half, return list of expanded bboxes."""
    results = face_det.process(rgb_half)
    faces = []
    if results.detections:
        for det in results.detections:
            bb = det.location_data.relative_bounding_box
            x1 = int(bb.xmin * w)
            y1 = int(bb.ymin * h)
            x2 = int((bb.xmin + bb.width) * w)
            y2 = int((bb.ymin + bb.height) * h)
            faces.append(_expand_face_bbox((x1, y1, x2, y2), w, h))
    return faces


# ── Temporal smoothing ───────────────────────────────────────────────────

def _iou(a, b):
    """IoU of two (x1,y1,x2,y2) boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def _smooth_face_detections(faces_per_frame, n_frames, w, h):
    """Track, interpolate, forward-fill, and Gaussian-smooth face bboxes."""
    tracks = []

    for t in range(n_frames):
        detections = faces_per_frame[t]
        matched_det = set()
        matched_track = set()

        for di, det in enumerate(detections):
            best_iou, best_ti = 0, -1
            for ti, track in enumerate(tracks):
                if ti in matched_track:
                    continue
                last_t = track["last_valid"]
                if last_t < 0:
                    continue
                iou_val = _iou(det, tuple(track["boxes"][last_t].astype(int)))
                if iou_val > best_iou:
                    best_iou, best_ti = iou_val, ti

            if best_iou > IOU_MATCH_THRESHOLD and best_ti >= 0:
                tracks[best_ti]["boxes"][t] = det
                tracks[best_ti]["last_valid"] = t
                matched_det.add(di)
                matched_track.add(best_ti)

        for di, det in enumerate(detections):
            if di not in matched_det:
                boxes = np.full((n_frames, 4), np.nan)
                boxes[t] = det
                tracks.append({"boxes": boxes, "last_valid": t})

    # Interpolate, forward-fill, smooth
    for track in tracks:
        boxes = track["boxes"]
        valid = ~np.isnan(boxes[:, 0])
        valid_indices = np.where(valid)[0]
        if len(valid_indices) == 0:
            continue

        if len(valid_indices) >= 2:
            for i in range(len(valid_indices) - 1):
                t0, t1 = valid_indices[i], valid_indices[i + 1]
                for t in range(t0 + 1, t1):
                    alpha = (t - t0) / (t1 - t0)
                    boxes[t] = boxes[t0] * (1 - alpha) + boxes[t1] * alpha

        last = valid_indices[-1]
        if last < n_frames - 1:
            for t in range(last + 1, n_frames):
                boxes[t] = boxes[last]

        first = valid_indices[0]
        if n_frames - first > 1:
            for col in range(4):
                boxes[first:, col] = gaussian_filter1d(
                    boxes[first:, col], TEMPORAL_SIGMA_FRAMES)

    result = [[] for _ in range(n_frames)]
    for track in tracks:
        for t in range(n_frames):
            if not np.isnan(track["boxes"][t, 0]):
                box = tuple(np.clip(track["boxes"][t], 0, [w, h, w, h]).astype(int))
                result[t].append(box)

    return result


# ── Landmark interpolation across missing frames ────────────────────────

def interpolate_landmarks_inplace(arr, start: int = 0, end: int | None = None):
    """Linearly interpolate NaN values per (joint, dim) across a frame range.

    Mutates ``arr`` in place.  ``arr`` is expected to be ``(N, J, 2)`` —
    e.g. ``OS_landmarks`` from ``mediapipe_prelabels.npz``.  Interpolation
    is restricted to the inclusive range ``[start, end]`` so we never
    bridge a gap across trial boundaries (NaN frames in one trial
    shouldn't be filled from another trial's landmarks).

    Edge behaviour: ``np.interp`` holds endpoints by default, so leading
    / trailing NaN gaps inside the range get filled with the nearest
    valid value.  This is exactly what the hand mask wants — protection
    should stay active across MediaPipe drop-outs, including drop-outs
    at the very start or end of the trial.

    Joints / dims with NO valid frames in the range stay all-NaN.
    """
    import numpy as _np
    if arr is None or arr.size == 0:
        return arr
    N = arr.shape[0]
    if end is None:
        end = N - 1
    start = max(0, int(start))
    end = min(N - 1, int(end))
    if end <= start:
        return arr
    seg = arr[start:end + 1]            # view, in-place writes propagate
    M = seg.shape[0]
    idx = _np.arange(M)
    for j in range(seg.shape[1]):
        for d in range(seg.shape[2]):
            col = seg[:, j, d]
            valid = ~_np.isnan(col)
            if not valid.any() or valid.all():
                continue
            # ``np.interp`` holds the nearest endpoint outside the convex
            # hull of valid indices, which is the behaviour we want.
            seg[:, j, d] = _np.interp(idx, idx[valid], col[valid])
    return arr


# ── Hand protection (MediaPipe Hands) ────────────────────────────────────

def _build_hand_mask(shape, hand_results):
    """Build hand exclusion mask from MediaPipe Hands results.

    Uses convex hull of detected landmarks + dilation.
    """
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if not hand_results or not hand_results.multi_hand_landmarks:
        return mask > 0

    for hand_lm in hand_results.multi_hand_landmarks:
        pts = np.array(
            [(int(lm.x * w), int(lm.y * h)) for lm in hand_lm.landmark],
            dtype=np.float32,
        )
        valid = pts[~np.isnan(pts[:, 0])]
        if len(valid) < 3:
            continue
        hull = cv2.convexHull(valid.astype(np.float32))
        cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)

    if HAND_HULL_DILATION_PX > 0:
        k = 2 * HAND_HULL_DILATION_PX + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, kernel)

    return mask > 0


# ── HRnet MIP-based hand mask (preview + render share this) ─────────────
# Module-level so the live-preview endpoint can build the same HRnet
# hand mask the full render pipeline uses, instead of duplicating the
# closure-captured version inside ``render_with_blur_specs``.

def build_hand_mask_from_hrnet_mip(
    subject_name: str, trial_idx: int, frame_num: int,
    side_label: str, w_side: int, h_side: int,
    thresh: float = 0.30, smooth_px: int = 7,
):
    """Build an HRnet-MIP-based hand mask for one camera half.

    Reads ``hrnet_w18_mip.npz`` + ``hand_crop.json`` from the subject's
    skeleton output dir, thresholds the 64×64 MIP at ``thresh``, drops it
    into the per-frame bbox on a (h_side, w_side) canvas, then runs a
    Gaussian-blur + re-threshold smoothing pass (``smooth_px`` controls
    the dilation amount).

    Returns a (h, w) bool array, or ``None`` if the HRnet data isn't
    available for this subject/trial.  The renderer falls back to the
    MediaPipe-circle mask in that case.

    Args:
        subject_name: e.g. ``"MSA09"``.
        trial_idx: index into ``build_trial_map(subject_name)``.
        frame_num: GLOBAL frame number (across all trials).
        side_label: ``"left"`` / ``"right"`` / ``"full"`` — ``"left"``
            and ``"full"`` use the OS MIP, ``"right"`` the OD.
        w_side, h_side: dimensions of the destination canvas (single
            camera half for stereo, full frame for multicam).
        thresh: MIP threshold (0..1).  Higher = tighter mask.
        smooth_px: dilation strength.  0 = no smoothing.
    """
    import numpy as _np
    from .skeleton_data import _skeleton_dir
    from .video import build_trial_map as _btm
    import json as _jjson

    try:
        tmap = _btm(subject_name)
    except Exception as e:
        logger.warning(f"build_hand_mask_from_hrnet_mip: build_trial_map failed: {e}")
        return None
    if trial_idx < 0 or trial_idx >= len(tmap):
        return None
    tdef = tmap[trial_idx]
    stem = tdef["trial_name"]
    mip_path = _skeleton_dir(subject_name) / stem / "hrnet_w18_mip.npz"
    crop_path = _skeleton_dir(subject_name) / stem / "hand_crop.json"
    if not mip_path.exists() or not crop_path.exists():
        return None
    try:
        d = _np.load(mip_path)
        mip_L = d["heatmaps_L_mip"] if "heatmaps_L_mip" in d.files else None
        mip_R = d["heatmaps_R_mip"] if "heatmaps_R_mip" in d.files else None
        with open(crop_path) as f:
            cj = _jjson.load(f)
        bbox_L = cj.get("crop_L_perframe") or [cj.get("crop_L")]
        bbox_R = cj.get("crop_R_perframe") or [cj.get("crop_R")]
    except Exception as e:
        logger.warning(f"build_hand_mask_from_hrnet_mip: load failed: {e}")
        return None
    trial_start = int(tdef.get("start_frame", 0))
    mip = mip_L if side_label in ("left", "full") else mip_R
    bboxes = bbox_L if side_label in ("left", "full") else bbox_R
    if mip is None or not bboxes:
        return None
    f_in_trial = frame_num - trial_start
    if f_in_trial < 0 or f_in_trial >= mip.shape[0]:
        return None
    bbox = bboxes[f_in_trial] if f_in_trial < len(bboxes) else bboxes[-1]
    if bbox is None or len(bbox) != 4:
        return None
    bx1, by1, bx2, by2 = (int(round(v)) for v in bbox)
    bw, bh = bx2 - bx1, by2 - by1
    if bw <= 0 or bh <= 0:
        return None
    # Re-normalise the per-frame 64×64 MIP so the threshold slider
    # operates on a meaningful range.  The saved MIPs come from
    # ``expit(logits)`` in remote_hrnet_script.py, which gives a
    # bimodal distribution: a huge background spike at sigmoid(~0) ≈
    # 0.7 (most of the bbox area) plus a thin tail of joint peaks up
    # to 1.0.  Using per-frame min/max as the anchors snaps the entire
    # background spike to exactly 0 (because float16 quantises most
    # of them to a single value), creating a cliff at threshold ≈ 0
    # where 79% of pixels suddenly drop out for any thresh > 0.
    #
    # Better: anchor the lower end at the per-frame *median* (which
    # falls inside the background spike) and the upper end at the
    # max.  Then background normalises to ≤ 0 (clipped), the shoulder
    # of each joint Gaussian gets a smooth fraction in (0, 1), and the
    # slider's full [0, 1] range maps to "include the shoulder vs.
    # only the peak."
    frame_mip = mip[f_in_trial].astype(np.float32)
    baseline = float(np.median(frame_mip))
    hi = float(frame_mip.max())
    if hi - baseline > 1e-6:
        frame_mip = np.clip((frame_mip - baseline) / (hi - baseline), 0.0, 1.0)
    else:
        frame_mip = np.zeros_like(frame_mip)
    small = (frame_mip > thresh).astype(np.uint8) * 255
    big = cv2.resize(small, (bw, bh), interpolation=cv2.INTER_LINEAR)
    big = (big > 127).astype(np.uint8) * 255
    out = np.zeros((h_side, w_side), dtype=np.uint8)
    sx1, sy1 = max(0, bx1), max(0, by1)
    sx2, sy2 = min(w_side, bx2), min(h_side, by2)
    if sx2 > sx1 and sy2 > sy1:
        out[sy1:sy2, sx1:sx2] = big[sy1 - by1:sy2 - by1, sx1 - bx1:sx2 - bx1]
    if smooth_px and smooth_px > 0:
        k = max(1, int(smooth_px) | 1)
        out = cv2.GaussianBlur(out, (k, k), 0)
        out = (out > 127).astype(np.uint8) * 255
    return out > 0


# ── Blur compositing ─────────────────────────────────────────────────────

def _apply_face_blur(frame_half, face_boxes, hand_mask):
    """Apply Gaussian blur to face regions, excluding hand region."""
    if not face_boxes:
        return frame_half

    h, w = frame_half.shape[:2]

    face_mask = np.zeros((h, w), dtype=np.uint8)
    for (x1, y1, x2, y2) in face_boxes:
        face_mask[y1:y2, x1:x2] = 255

    face_mask[hand_mask] = 0

    if face_mask.max() == 0:
        return frame_half

    face_mask_f = cv2.GaussianBlur(
        face_mask.astype(np.float32) / 255.0,
        (FEATHER_KERNEL, FEATHER_KERNEL), 5)

    blurred = cv2.GaussianBlur(frame_half, (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), BLUR_SIGMA)

    mask_3ch = face_mask_f[:, :, np.newaxis]
    result = (frame_half.astype(np.float32) * (1 - mask_3ch) +
              blurred.astype(np.float32) * mask_3ch).astype(np.uint8)

    return result


# ── Main entry point ─────────────────────────────────────────────────────

def deidentify_video(input_path: str, output_path: str,
                     progress_callback=None) -> dict:
    """De-identify a video: blur faces, preserve hands.

    Handles both stereo (split at midline) and single-camera videos.
    Stereo detection: width/height > 1.7.

    Args:
        input_path: Source video path
        output_path: Output video path
        progress_callback: Optional callable(pct: float) for progress updates

    Returns:
        dict with stats: n_frames, face_det_rate, is_stereo
    """
    import mediapipe as mp_lib

    cap = cv2.VideoCapture(input_path)
    reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    full_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if full_w == 0 or h == 0:
        cap.release()
        raise RuntimeError(f"Cannot read video: {input_path}")

    is_stereo = (full_w / h) > 1.7

    if is_stereo:
        half_w = full_w // 2
    else:
        half_w = full_w

    logger.info(f"Deidentifying: reported {reported_frames} frames, "
                f"{'stereo' if is_stereo else 'mono'}, {half_w}x{h}")

    # Initialize MediaPipe
    face_det = mp_lib.solutions.face_detection.FaceDetection(
        model_selection=FACE_MODEL_SELECTION,
        min_detection_confidence=FACE_CONF_THRESHOLD,
    )
    hands = mp_lib.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    )

    # ── Pass 1: Detect faces (read until decode fails) ──
    if is_stereo:
        faces_L_raw, faces_R_raw = [], []
    else:
        faces_raw = []

    for t in range(reported_frames):
        ret, frame = cap.read()
        if not ret:
            break  # actual end of decodable frames

        frame = np.ascontiguousarray(frame, dtype=np.uint8)

        if is_stereo:
            left = frame[:, :half_w, :]
            right = frame[:, half_w:, :]

            rgb_l = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
            rgb_r = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)

            faces_L_raw.append(_detect_faces_in_half(rgb_l, face_det, half_w, h))
            faces_R_raw.append(_detect_faces_in_half(rgb_r, face_det, full_w - half_w, h))
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces_raw.append(_detect_faces_in_half(rgb, face_det, full_w, h))

        if progress_callback and t % 10 == 0:
            progress_callback(t / reported_frames * 40)  # Pass 1 = 0-40%

    face_det.close()

    # Use actual readable frame count (may be less than container metadata)
    n_frames = len(faces_L_raw) if is_stereo else len(faces_raw)
    if n_frames < reported_frames:
        logger.info(f"  Note: {reported_frames - n_frames} trailing undecodable frames "
                    f"(reported={reported_frames}, actual={n_frames})")

    # ── Temporal smoothing ──
    if is_stereo:
        faces_L = _smooth_face_detections(faces_L_raw, n_frames, half_w, h)
        faces_R = _smooth_face_detections(faces_R_raw, n_frames, full_w - half_w, h)
    else:
        faces = _smooth_face_detections(faces_raw, n_frames, full_w, h)

    # ── Pass 2: Render with hand protection ──
    # Reopen video instead of seeking (seeking can be inaccurate with H.264)
    cap.release()
    cap = cv2.VideoCapture(input_path)
    # Remove existing output file — macOS AVFoundation refuses to overwrite
    import os, sys
    if os.path.exists(output_path):
        os.remove(output_path)

    # Pipe raw frames into ffmpeg → H.264 yuv420p so the output is
    # decodable in HTML5 <video> on every browser.  The previous code
    # used cv2.VideoWriter('mp4v') on Windows, which produced MPEG-4 ASP
    # files browsers refuse to play.
    from .ffmpeg import get_ffmpeg_path
    ffmpeg = get_ffmpeg_path()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found. Install FFmpeg or pip install imageio-ffmpeg")
    _popen_kwargs = {"stdin": subprocess.PIPE,
                      "stdout": subprocess.DEVNULL,
                      "stderr": subprocess.PIPE}
    # Intentionally NO creationflags on Windows.  Both DETACHED_PROCESS
    # and CREATE_NO_WINDOW silently break Python's wiring of the
    # explicit stdin=PIPE handle into the child, causing the first
    # write to fail with [Errno 22] Invalid argument.  ffmpeg here is
    # a GRANDCHILD of the SSH session (owned by the worker process,
    # which is itself CREATE_BREAKAWAY_FROM_JOB), so it inherits the
    # SSH-exit-survival behaviour transitively without needing its
    # own job-breakaway / detach flag.
    h264_encoder = _pick_h264_encoder(ffmpeg)
    if h264_encoder == "libx264":
        encoder_args = ["-c:v", "libx264", "-preset", "slow", "-crf", "18"]
    else:
        encoder_args = ["-c:v", h264_encoder, "-b:v", "8M"]
    # -nostats + -loglevel error: keep stderr small so the 64 KB
    # OS pipe buffer doesn't fill mid-render and deadlock our
    # stdin-frame-write loop.  The finally block reads stderr once
    # at the end and that's enough for surfacing errors.
    proc = subprocess.Popen([
        ffmpeg, "-y", "-nostats", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{full_w}x{h}", "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "-",
        *encoder_args,
        "-pix_fmt", "yuv420p", "-an",
        "-movflags", "+faststart",
        output_path,
    ], **_popen_kwargs)

    face_det_count = 0

    for t in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        frame = np.ascontiguousarray(frame, dtype=np.uint8)

        if is_stereo:
            left = frame[:, :half_w, :].copy()
            right = frame[:, half_w:, :].copy()

            # Hand detection on each half
            rgb_l = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
            rgb_r = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)

            hand_res_l = hands.process(rgb_l)
            hand_res_r = hands.process(rgb_r)

            hand_mask_l = _build_hand_mask(left.shape, hand_res_l)
            hand_mask_r = _build_hand_mask(right.shape, hand_res_r)

            left = _apply_face_blur(left, faces_L[t], hand_mask_l)
            right = _apply_face_blur(right, faces_R[t], hand_mask_r)

            out_frame = np.concatenate([left, right], axis=1)

            if faces_L[t] or faces_R[t]:
                face_det_count += 1
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_res = hands.process(rgb)
            hand_mask = _build_hand_mask(frame.shape, hand_res)

            frame = _apply_face_blur(frame, faces[t], hand_mask)
            out_frame = frame

            if faces[t]:
                face_det_count += 1

        proc.stdin.write(np.ascontiguousarray(out_frame, dtype=np.uint8).tobytes())

        if progress_callback and t % 10 == 0:
            progress_callback(40 + t / n_frames * 60)  # Pass 2 = 40-100%

    cap.release()
    try:
        proc.stdin.close()
    except Exception:
        pass
    stderr_bytes = proc.stderr.read() if proc.stderr else b""
    proc.wait(timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg encoding failed: {stderr_bytes.decode(errors='replace')[:500]}")
    hands.close()

    if progress_callback:
        progress_callback(100)

    stats = {
        "n_frames": n_frames,
        "face_det_rate": face_det_count / n_frames if n_frames > 0 else 0,
        "is_stereo": is_stereo,
    }

    logger.info(f"Deidentify complete: {n_frames} frames, face rate={stats['face_det_rate']:.1%}")
    return stats


# ── Interactive deidentify: face detection + spec-based render ──────────

def detect_faces_in_video(video_path: str, start_frame: int = 0,
                          frame_count: int | None = None,
                          progress_callback=None) -> dict:
    """Run face detection on a video segment and return per-frame bounding boxes.

    Returns dict with:
        faces: list of {frame: int, faces: [{x1, y1, x2, y2, confidence}]}
        is_stereo: bool
        frame_width: int (half-width for stereo)
        frame_height: int
    """
    import mediapipe as mp_lib

    cap = cv2.VideoCapture(video_path)
    reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    full_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if full_w == 0 or h == 0:
        cap.release()
        raise RuntimeError(f"Cannot read video: {video_path}")

    is_stereo = (full_w / h) > 1.7
    half_w = full_w // 2 if is_stereo else full_w
    total = frame_count if frame_count else (reported - start_frame)

    face_det = mp_lib.solutions.face_detection.FaceDetection(
        model_selection=FACE_MODEL_SELECTION,
        min_detection_confidence=FACE_CONF_THRESHOLD,
    )

    # Only seek if start_frame is within this file's frame count.
    # start_frame is a global offset — for multi-file subjects each trial
    # has its own video starting at local frame 0, so don't seek past the end.
    if start_frame > 0 and start_frame < reported:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    raw_L, raw_R, raw_mono = [], [], []
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        frame = np.ascontiguousarray(frame, dtype=np.uint8)

        if is_stereo:
            left = frame[:, :half_w, :]
            right = frame[:, half_w:, :]
            rgb_l = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
            rgb_r = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)
            raw_L.append(_detect_faces_in_half(rgb_l, face_det, half_w, h))
            raw_R.append(_detect_faces_in_half(rgb_r, face_det, full_w - half_w, h))
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            raw_mono.append(_detect_faces_in_half(rgb, face_det, full_w, h))

        if progress_callback and i % 10 == 0:
            progress_callback(i / total * 100)

    face_det.close()
    cap.release()

    n = len(raw_L) if is_stereo else len(raw_mono)

    # Temporal smoothing
    if is_stereo:
        smoothed_L = _smooth_face_detections(raw_L, n, half_w, h)
        smoothed_R = _smooth_face_detections(raw_R, n, full_w - half_w, h)
    else:
        smoothed = _smooth_face_detections(raw_mono, n, full_w, h)

    # Build response
    faces_per_frame = []
    for i in range(n):
        entry = {"frame": start_frame + i, "faces": []}
        if is_stereo:
            for (x1, y1, x2, y2) in smoothed_L[i]:
                entry["faces"].append({
                    "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2), "side": "left",
                })
            for (x1, y1, x2, y2) in smoothed_R[i]:
                entry["faces"].append({
                    "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2), "side": "right",
                })
        else:
            for (x1, y1, x2, y2) in smoothed[i]:
                entry["faces"].append({
                    "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2), "side": "full",
                })
        faces_per_frame.append(entry)

    if progress_callback:
        progress_callback(100)

    return {
        "faces": faces_per_frame,
        "is_stereo": bool(is_stereo),
        "frame_width": int(half_w),
        "frame_height": int(h),
        "n_frames": int(n),
    }


def detect_hands_single_frame(video_path: str, frame_num: int,
                              is_stereo: bool = False) -> dict:
    """Run MediaPipe Hands on a single frame, return landmark positions.

    Returns dict with landmarks: [{x, y, side}] for each detected hand point.
    """
    import mediapipe as mp_lib

    cap = cv2.VideoCapture(video_path)
    full_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    half_w = full_w // 2 if is_stereo else full_w

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return {"landmarks": [], "hands": []}

    frame = np.ascontiguousarray(frame, dtype=np.uint8)
    hands_det = mp_lib.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=4,
        min_detection_confidence=0.3,
    )

    all_landmarks = []
    all_hands = []

    if is_stereo:
        for side_name, half_frame, w in [
            ("left", frame[:, :half_w, :], half_w),
            ("right", frame[:, half_w:, :], full_w - half_w),
        ]:
            rgb = cv2.cvtColor(half_frame, cv2.COLOR_BGR2RGB)
            res = hands_det.process(rgb)
            if res.multi_hand_landmarks:
                for hand_lm in res.multi_hand_landmarks:
                    hand_pts = []
                    for lm in hand_lm.landmark:
                        hand_pts.append({
                            "x": round(lm.x * w, 1),
                            "y": round(lm.y * h, 1),
                            "side": side_name,
                        })
                    all_hands.append(hand_pts)
                    all_landmarks.extend(hand_pts)
    else:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands_det.process(rgb)
        if res.multi_hand_landmarks:
            for hand_lm in res.multi_hand_landmarks:
                hand_pts = []
                for lm in hand_lm.landmark:
                    hand_pts.append({
                        "x": round(lm.x * full_w, 1),
                        "y": round(lm.y * h, 1),
                        "side": "full",
                    })
                all_hands.append(hand_pts)
                all_landmarks.extend(hand_pts)

    hands_det.close()
    return {"landmarks": all_landmarks, "hands": all_hands}


def render_with_blur_specs(input_path: str, output_path: str,
                           blur_specs: list[dict],
                           hand_settings: dict | None = None,
                           face_detections: list[dict] | None = None,
                           subject_name: str | None = None,
                           start_frame: int = 0,
                           frame_count: int | None = None,
                           progress_callback=None,
                           mp_data=None,
                           pose_data=None) -> dict:
    """Render a video with blur specs matching the frontend preview.

    Face-type spots track face detection centroids per frame.
    Blur regions are ellipses (width x height) with offset from centroid.
    Hand protection uses stored MediaPipe landmarks from npz, not live detection.
    """
    cap = cv2.VideoCapture(input_path)
    reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    full_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if full_w == 0 or fh == 0:
        cap.release()
        raise RuntimeError(f"Cannot read video: {input_path}")

    is_stereo = (full_w / fh) > 1.7
    half_w = full_w // 2 if is_stereo else full_w
    total = frame_count if frame_count else (reported - start_frame)

    # Re-anchor any saved blur-spec frame ranges that fall entirely
    # outside this trial's [start_frame, end_frame] window -- mirrors
    # the frontend's selectTrial re-anchor.  Without this, if the
    # trial got re-trimmed since the specs were last saved, every
    # frame's ``active_specs`` filter (``frame_start <= global_frame
    # <= frame_end``) would come back empty and the renderer would
    # write back the ORIGINAL video unchanged.  Specs with valid
    # overlap are clamped to the trial range; specs with no overlap
    # get the full trial range.
    _t_start = int(start_frame)
    _t_end = _t_start + int(total) - 1
    blur_specs = list(blur_specs or [])
    for _s in blur_specs:
        fs = _s.get("frame_start")
        fe = _s.get("frame_end")
        if fs is None or fe is None or fe < _t_start or fs > _t_end:
            _s["frame_start"] = _t_start
            _s["frame_end"] = _t_end
        else:
            _s["frame_start"] = max(_t_start, int(fs))
            _s["frame_end"] = min(_t_end, int(fe))

    # Index face detections by frame_num for fast lookup
    face_by_frame = {}
    for fd in (face_detections or []):
        fn = fd["frame_num"]
        if fn not in face_by_frame:
            face_by_frame[fn] = []
        face_by_frame[fn].append(fd)

    # Load hand landmarks from npz for hand protection
    hand_lm_data = None  # {frame: [{x, y, side}]}
    hand_segments = []
    hand_mask_radius = 5
    hand_smooth = 7
    forearm_radius = 10
    forearm_extent = 0.4
    hand_smooth2 = 5
    dlc_radius_val = 15
    arm_dorsal_dilate = 0
    arm_ventral_dilate = 0
    if hand_settings:
        import json as _json
        hand_mask_radius = hand_settings.get("hand_mask_radius") or hand_settings.get("mask_radius") or 10
        hand_smooth = hand_settings.get("hand_smooth", 10)
        forearm_radius = hand_settings.get("forearm_radius", 10)
        forearm_extent = hand_settings.get("forearm_extent", 0.7)
        hand_smooth2 = hand_settings.get("hand_smooth2", 0)
        dlc_radius_val = hand_settings.get("dlc_radius", 15)
        arm_dorsal_dilate = int(hand_settings.get("arm_dorsal_dilate", 0) or 0)
        arm_ventral_dilate = int(hand_settings.get("arm_ventral_dilate", 0) or 0)
        seg_json = hand_settings.get("segments_json", "[]")
        try:
            hand_segments = _json.loads(seg_json) if isinstance(seg_json, str) else (seg_json or [])
        except (ValueError, TypeError):
            hand_segments = []
        if not hand_segments:
            hand_segments = []

    # Mirror the preview's default-synthesis behaviour (see
    # routers/deidentify.py preview path): when no hand-protection
    # segments are present, cover the whole trial with a per-camera
    # segment so hand protection is on by default.  Previously gated
    # only on ``hand_settings is None``, which let the LOCAL worker
    # (which always builds a hand_settings dict) and the REMOTE worker
    # (whose DB-row bundle has ``segments_json`` but possibly empty)
    # both fall through with hand_segments == [].  Result: live
    # preview showed the auto-generated default mask but the
    # rendered video had no hand protection at all.  Now we fall
    # back to the default whenever hand_segments is empty.
    if not hand_segments:
        _t_start = int(start_frame or 0)
        _t_end = _t_start + (int(frame_count) if frame_count else 0) - 1
        if _t_end >= _t_start:
            if is_stereo:
                hand_segments = [
                    {"start": _t_start, "end": _t_end,
                     "radius": hand_mask_radius, "smooth": hand_smooth,
                     "side": "left"},
                    {"start": _t_start, "end": _t_end,
                     "radius": hand_mask_radius, "smooth": hand_smooth,
                     "side": "right"},
                ]
            else:
                hand_segments = [{
                    "start": _t_start, "end": _t_end,
                    "radius": hand_mask_radius, "smooth": hand_smooth,
                    "side": "full",
                }]

    # Load from npz if subject_name provided.
    # Accepts pre-loaded data via mp_data/pose_data kwargs (used by remote worker
    # to avoid relative import failures in standalone context).
    if subject_name and hand_segments:
        import numpy as np2

        def _build_lm_data(os_lm, od_lm, existing=None,
                            os_was_nan=None, od_was_nan=None):
            """Build hand_lm_data dict from OS/OD landmark arrays.

            ``os_was_nan`` / ``od_was_nan`` are optional (N, J) bool arrays
            captured BEFORE interpolation; when present, each landmark
            dict is tagged ``interpolated: True`` for joints whose
            position was filled in by ``interpolate_landmarks_inplace``.
            The DLC tip-disagreement path in
            ``_build_hand_mask_from_landmarks`` uses this flag to also
            fire when MP "saw" the tip only via interpolation.
            """
            lm_data = existing or {}
            if os_lm is not None:
                for f in range(os_lm.shape[0]):
                    if f not in lm_data:
                        lm_data[f] = []
                    for j in range(os_lm.shape[1]):
                        x, y = os_lm[f, j, 0], os_lm[f, j, 1]
                        if not np.isnan(x):
                            was_interp = bool(os_was_nan is not None and os_was_nan[f, j])
                            lm_data[f].append({"x": float(x), "y": float(y), "side": "left", "type": "hand", "joint": j, "interpolated": was_interp})
            if od_lm is not None:
                for f in range(od_lm.shape[0]):
                    if f not in lm_data:
                        lm_data[f] = []
                    for j in range(od_lm.shape[1]):
                        x, y = od_lm[f, j, 0], od_lm[f, j, 1]
                        if not np.isnan(x):
                            was_interp = bool(od_was_nan is not None and od_was_nan[f, j])
                            lm_data[f].append({"x": float(x), "y": float(y), "side": "right", "type": "hand", "joint": j, "interpolated": was_interp})
            return lm_data

        def _build_pose_data(pose_os, pose_od, existing):
            """Merge pose landmarks into existing lm_data dict."""
            lm_data = existing
            if pose_os is not None:
                for f in range(pose_os.shape[0]):
                    if f not in lm_data:
                        lm_data[f] = []
                    for j in range(pose_os.shape[1]):
                        x, y = pose_os[f, j, 0], pose_os[f, j, 1]
                        if not np.isnan(x):
                            lm_data[f].append({"x": float(x), "y": float(y), "side": "left", "type": "pose", "joint": j})
            if pose_od is not None:
                for f in range(pose_od.shape[0]):
                    if f not in lm_data:
                        lm_data[f] = []
                    for j in range(pose_od.shape[1]):
                        x, y = pose_od[f, j, 0], pose_od[f, j, 1]
                        if not np.isnan(x):
                            lm_data[f].append({"x": float(x), "y": float(y), "side": "right", "type": "pose", "joint": j})
            return lm_data

        # Interpolate NaN gaps inside this trial's frame range so the
        # hand mask stays active when MediaPipe drops detection on
        # individual frames.  ``start_frame`` and ``total`` come from
        # the renderer scope above.
        _interp_start = int(start_frame)
        _interp_end = _interp_start + int(total) - 1

        if mp_data is not None:
            # Pre-loaded data provided (e.g., remote worker) — use directly,
            # no file I/O or relative imports needed.
            os_lm = mp_data.get("OS_landmarks")
            od_lm = mp_data.get("OD_landmarks")
            os_was_nan = None
            od_was_nan = None
            if os_lm is not None:
                os_lm = np2.array(os_lm, copy=True)
                os_was_nan = np2.isnan(os_lm[..., 0]).copy()
                interpolate_landmarks_inplace(os_lm, _interp_start, _interp_end)
            if od_lm is not None:
                od_lm = np2.array(od_lm, copy=True)
                od_was_nan = np2.isnan(od_lm[..., 0]).copy()
                interpolate_landmarks_inplace(od_lm, _interp_start, _interp_end)
            hand_lm_data = _build_lm_data(os_lm, od_lm,
                                          os_was_nan=os_was_nan,
                                          od_was_nan=od_was_nan)
            if pose_data is not None:
                pose_os = pose_data.get("OS_pose")
                pose_od = pose_data.get("OD_pose")
                if pose_os is not None:
                    pose_os = np2.array(pose_os, copy=True)
                    interpolate_landmarks_inplace(pose_os, _interp_start, _interp_end)
                if pose_od is not None:
                    pose_od = np2.array(pose_od, copy=True)
                    interpolate_landmarks_inplace(pose_od, _interp_start, _interp_end)
                hand_lm_data = _build_pose_data(pose_os, pose_od, hand_lm_data)
        else:
            from ..config import get_settings
            from .mediapipe_prelabel import _detect_frame_offset
            settings = get_settings()
            npz_path = settings.dlc_path / subject_name / "mediapipe_prelabels.npz"
            if npz_path.exists():
                npz = np2.load(str(npz_path))
                os_lm = npz.get("OS_landmarks")
                od_lm = npz.get("OD_landmarks")
                # Trim pre-roll frames if NPZ was built on a machine with different
                # codec frame-counting behaviour (e.g. OpenCV exposing negative-PTS frames).
                _mp_offset = _detect_frame_offset(subject_name, npz)
                if _mp_offset > 0 and os_lm is not None:
                    os_lm = os_lm[_mp_offset:]
                if _mp_offset > 0 and od_lm is not None:
                    od_lm = od_lm[_mp_offset:]
                # Interpolate AFTER the offset trim (so frame indices
                # align with the renderer's start_frame/total range).
                os_was_nan = None
                od_was_nan = None
                if os_lm is not None:
                    os_lm = np2.array(os_lm, copy=True)
                    os_was_nan = np2.isnan(os_lm[..., 0]).copy()
                    interpolate_landmarks_inplace(os_lm, _interp_start, _interp_end)
                if od_lm is not None:
                    od_lm = np2.array(od_lm, copy=True)
                    od_was_nan = np2.isnan(od_lm[..., 0]).copy()
                    interpolate_landmarks_inplace(od_lm, _interp_start, _interp_end)
                hand_lm_data = _build_lm_data(os_lm, od_lm,
                                              os_was_nan=os_was_nan,
                                              od_was_nan=od_was_nan)

                # Also load pose landmarks for forearm triangle (elbow)
                pose_npz_path = settings.dlc_path / subject_name / "pose_prelabels.npz"
                if pose_npz_path.exists():
                    pose_npz = np2.load(str(pose_npz_path))
                    pose_os = pose_npz.get("OS_pose")
                    pose_od = pose_npz.get("OD_pose")
                    # Apply same pre-roll trim to pose data
                    _pose_offset = _detect_frame_offset(subject_name, pose_npz)
                    if _pose_offset > 0 and pose_os is not None:
                        pose_os = pose_os[_pose_offset:]
                    if _pose_offset > 0 and pose_od is not None:
                        pose_od = pose_od[_pose_offset:]
                    if pose_os is not None:
                        pose_os = np2.array(pose_os, copy=True)
                        interpolate_landmarks_inplace(pose_os, _interp_start, _interp_end)
                    if pose_od is not None:
                        pose_od = np2.array(pose_od, copy=True)
                        interpolate_landmarks_inplace(pose_od, _interp_start, _interp_end)
                    hand_lm_data = _build_pose_data(pose_os, pose_od, hand_lm_data)

    import os, subprocess

    # Remove existing output file
    if os.path.exists(output_path):
        os.remove(output_path)

    try:
        from .ffmpeg import get_ffmpeg_path
        ffmpeg = get_ffmpeg_path()
    except (ImportError, FileNotFoundError) as _e:
        cap.release()
        # Preserve the underlying lookup failure -- the remote worker's
        # ``get_ffmpeg_path`` includes specifics like "imageio_ffmpeg
        # returned X but path doesn't exist" that vanished here, leaving
        # the user with a generic message that didn't match the diagnostic
        # already printed to the job log.
        raise RuntimeError(
            f"ffmpeg not found. Install FFmpeg or pip install imageio-ffmpeg. "
            f"Underlying error: {type(_e).__name__}: {_e}"
        )

    # Each trial has its own video file starting at local frame 0.
    # start_frame is the global subject-level offset used only for looking up
    # face_detections and blur_specs by global frame number — NOT a seek position.
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Hand mask cache: reuse when landmarks haven't moved
    _hand_cache = {}
    HAND_CACHE_THRESHOLD = 3.0

    def _lm_hash(lms):
        if not lms:
            return None
        return tuple((round(lm["x"], 1), round(lm["y"], 1)) for lm in lms)

    def _lm_moved(old_hash, new_hash):
        if old_hash is None or new_hash is None:
            return True
        if len(old_hash) != len(new_hash):
            return True
        return max(max(abs(a[0]-b[0]), abs(a[1]-b[1]))
                   for a, b in zip(old_hash, new_hash)) > HAND_CACHE_THRESHOLD

    def _get_hand_mask_cached(side_label, lms, w_side, h_side, r, sm, fa_r, fa_e, sm2, dlc_rad=0):
        key = side_label
        new_hash = _lm_hash(lms)
        params = (r, sm, fa_r, fa_e, sm2, dlc_rad,
                  arm_dorsal_dilate, arm_ventral_dilate)
        cached = _hand_cache.get(key)
        if cached and cached["params"] == params and not _lm_moved(cached["lm_hash"], new_hash):
            return cached["mask"]
        mask = _build_hand_mask_from_landmarks(lms, w_side, h_side, r, sm, fa_r, fa_e, sm2,
                                               arm_dorsal_dilate=arm_dorsal_dilate,
                                               arm_ventral_dilate=arm_ventral_dilate,
                                               dlc_radius=dlc_rad)
        _hand_cache[key] = {"lm_hash": new_hash, "mask": mask, "params": params}
        return mask

    # ── HRnet MIP-based mask (alternative to MP-circle mask) ───────────
    # Loaded lazily on first use so the renderer doesn't pay the cost when
    # the user picked MediaPipe.  Each side gets a (N_trial, 64, 64) MIP
    # plus a (N_trial, 4) per-frame bbox.
    _hrnet_mip = {"loaded": False, "L": None, "R": None,
                   "bbox_L": None, "bbox_R": None,
                   "trial_start": 0, "n_trial": 0}

    def _load_hrnet_mip():
        if _hrnet_mip["loaded"] or not subject_name:
            return
        _hrnet_mip["loaded"] = True
        try:
            from .skeleton_data import _skeleton_dir
            from .video import build_trial_map as _btm
            import json as _jjson, numpy as _np
            tmap = _btm(subject_name)
            # Pick the trial whose start_frame matches the renderer's offset.
            tdef = next((t for t in tmap if t.get("start_frame", 0) == start_frame), None)
            if tdef is None:
                return
            stem = tdef["trial_name"]
            mip_path = _skeleton_dir(subject_name) / stem / "hrnet_w18_mip.npz"
            crop_path = _skeleton_dir(subject_name) / stem / "hand_crop.json"
            if not mip_path.exists() or not crop_path.exists():
                return
            d = _np.load(mip_path)
            _hrnet_mip["L"] = d["heatmaps_L_mip"] if "heatmaps_L_mip" in d.files else None
            _hrnet_mip["R"] = d["heatmaps_R_mip"] if "heatmaps_R_mip" in d.files else None
            with open(crop_path) as f:
                cj = _jjson.load(f)
            _hrnet_mip["bbox_L"] = cj.get("crop_L_perframe") or [cj.get("crop_L")]
            _hrnet_mip["bbox_R"] = cj.get("crop_R_perframe") or [cj.get("crop_R")]
            _hrnet_mip["trial_start"] = int(tdef.get("start_frame", 0))
            _hrnet_mip["n_trial"] = int(tdef.get("frame_count", 0))
        except Exception as e:
            logger.warning(f"HRnet MIP load failed for {subject_name}: {e}")

    def _build_hand_mask_from_hrnet(side_label, w_side, h_side, global_frame, thresh, smooth_px):
        """Threshold the HRnet MIP at the given frame, place into bbox,
        smooth via Gaussian blur + threshold.  Returns (h, w) bool mask
        or None if no HRnet data is available for this side / frame.
        """
        import numpy as _np
        if not _hrnet_mip["loaded"]:
            _load_hrnet_mip()
        # Side keys: in deidentify.py the renderer uses 'left'/'right'
        # for stereo halves (matching frame layout); the HRnet MIP has
        # 'L'/'R' keyed to the same OS/OD split.
        mip = _hrnet_mip["L"] if side_label in ("left", "full") else _hrnet_mip["R"]
        bboxes = _hrnet_mip["bbox_L"] if side_label in ("left", "full") else _hrnet_mip["bbox_R"]
        if mip is None or not bboxes:
            return _np.zeros((h_side, w_side), dtype=bool)
        f_in_trial = global_frame - _hrnet_mip["trial_start"]
        if f_in_trial < 0 or f_in_trial >= mip.shape[0]:
            return _np.zeros((h_side, w_side), dtype=bool)
        bbox = bboxes[f_in_trial] if f_in_trial < len(bboxes) else bboxes[-1]
        if bbox is None or len(bbox) != 4:
            return _np.zeros((h_side, w_side), dtype=bool)
        bx1, by1, bx2, by2 = (int(round(v)) for v in bbox)
        bw, bh = bx2 - bx1, by2 - by1
        if bw <= 0 or bh <= 0:
            return _np.zeros((h_side, w_side), dtype=bool)
        # Re-normalise the per-frame MIP: anchor low end at the
        # MEDIAN (background) and high end at the max.  See comment in
        # the module-level ``build_hand_mask_from_hrnet_mip`` for why
        # the min anchor created a cliff in the slider response.
        frame_mip = mip[f_in_trial].astype(_np.float32)
        _baseline = float(_np.median(frame_mip))
        _hi = float(frame_mip.max())
        if _hi - _baseline > 1e-6:
            frame_mip = _np.clip((frame_mip - _baseline) / (_hi - _baseline), 0.0, 1.0)
        else:
            frame_mip = _np.zeros_like(frame_mip)
        # Threshold + resize the 64×64 MIP to the bbox.
        small = (frame_mip > thresh).astype(np_uint8 := __import__('numpy').uint8) * 255
        big = cv2.resize(small, (bw, bh), interpolation=cv2.INTER_LINEAR)
        big = (big > 127).astype(__import__('numpy').uint8) * 255
        out = __import__('numpy').zeros((h_side, w_side), dtype=__import__('numpy').uint8)
        # Place big into out, clipping at frame edges.
        sx1, sy1 = max(0, bx1), max(0, by1)
        sx2, sy2 = min(w_side, bx2), min(h_side, by2)
        if sx2 > sx1 and sy2 > sy1:
            out[sy1:sy2, sx1:sx2] = big[sy1 - by1:sy2 - by1, sx1 - bx1:sx2 - bx1]
        if smooth_px and smooth_px > 0:
            k = max(1, int(smooth_px) | 1)
            out = cv2.GaussianBlur(out, (k, k), 0)
            out = (out > 127).astype(__import__('numpy').uint8) * 255
        return out > 0

    # Per-frame hand-mask dispatch — used by the per-side render code below.
    # The HRnet hand-mask source was removed; MediaPipe is the only path.
    def _get_hand_mask_for_render(side_label, lms, w_side, h_side, r, sm, fa_r, fa_e, sm2,
                                   dlc_rad=0, global_frame=0):
        return _get_hand_mask_cached(side_label, lms, w_side, h_side, r, sm, fa_r, fa_e, sm2, dlc_rad=dlc_rad)

    # Always pipe raw frames into ffmpeg → H.264 yuv420p.  Browsers refuse
    # to decode the MPEG-4 ASP files cv2.VideoWriter('mp4v') produces on
    # Windows, which broke the Auto page video pane for any subject whose
    # de-identification ran on the Windows GPU host.  H.264 yuv420p is
    # universally supported in HTML5 <video>.
    #
    # On Windows the child ffmpeg process is detached from our SSH session
    # (DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP) so it survives if a
    # remote launcher exits early — same fix we use for HRnet.
    _popen_kwargs = {"stdin": subprocess.PIPE,
                      "stdout": subprocess.DEVNULL,
                      "stderr": subprocess.PIPE}
    # Intentionally NO creationflags on Windows.  Both DETACHED_PROCESS
    # and CREATE_NO_WINDOW silently break Python's wiring of the
    # explicit stdin=PIPE handle into the child, causing the first
    # write to fail with [Errno 22] Invalid argument.  ffmpeg here is
    # a GRANDCHILD of the SSH session (owned by the worker process,
    # which is itself CREATE_BREAKAWAY_FROM_JOB), so it inherits the
    # SSH-exit-survival behaviour transitively without needing its
    # own job-breakaway / detach flag.
    # Pick an H.264 encoder this ffmpeg build actually has.  Conda's
    # default-channel ffmpeg often ships WITHOUT libx264 (license
    # reasons), in which case "-c:v libx264" fails immediately with
    # "Unknown encoder 'libx264'".  Probe once: prefer libx264 (best
    # quality), fall back to libopenh264 (royalty-free, in most
    # builds), then h264_mf (Windows Media Foundation hardware
    # encoder).  If none is available, raise with a fix hint.
    h264_encoder = _pick_h264_encoder(ffmpeg)
    # libopenh264 doesn't support -preset/-crf; use bitrate.  libx264
    # supports both.  h264_mf takes bitrate too.
    if h264_encoder == "libx264":
        encoder_args = ["-c:v", "libx264", "-preset", "slow", "-crf", "18"]
    else:
        encoder_args = ["-c:v", h264_encoder, "-b:v", "8M"]
    # -nostats + -loglevel error: keep stderr small so the 64 KB
    # OS pipe buffer doesn't fill mid-render and deadlock our
    # stdin-frame-write loop.  The finally block reads stderr once
    # at the end and that's enough for surfacing errors.
    proc = subprocess.Popen([
        ffmpeg, "-y", "-nostats", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{full_w}x{fh}", "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "-",
        *encoder_args,
        "-pix_fmt", "yuv420p", "-an",
        # +faststart moves the moov atom to the front so HTML5 <video>
        # can start playing before the entire file is downloaded.
        "-movflags", "+faststart",
        output_path,
    ], **_popen_kwargs)

    # Drain ffmpeg's stderr in a background thread so a 64 KB
    # accumulation can't ever deadlock the stdin-write loop, AND so
    # the captured text is available immediately on a crash instead
    # of only after the finally block.  Even with -nostats and
    # -loglevel error, a single libx264 warning can be enough to
    # block the writer on a long render.
    import threading as _thr
    _stderr_buf: list[bytes] = []
    def _drain_stderr():
        try:
            for line in iter(proc.stderr.readline, b""):
                _stderr_buf.append(line)
        except (OSError, ValueError):
            pass
    _stderr_thr = _thr.Thread(target=_drain_stderr, daemon=True)
    _stderr_thr.start()

    # If ffmpeg died before we even started writing (bad output path
    # permissions, missing codec, etc.) the first stdin.write fails
    # with [Errno 22] Invalid argument on Windows -- which obscures
    # the real cause.  Peek at proc.poll() first; if the child has
    # already exited, surface the stderr we just collected.
    import time as _time
    _time.sleep(0.1)  # tiny grace period for ffmpeg startup error
    if proc.poll() is not None:
        cap.release()
        _stderr_thr.join(timeout=0.5)
        stderr = b"".join(_stderr_buf).decode(errors="replace")
        raise RuntimeError(
            f"ffmpeg exited immediately (rc={proc.returncode}). "
            f"stderr: {stderr[:500]}"
        )

    frames_written = 0
    try:
        for i in range(total):
            global_frame = start_frame + i
            ret, frame = cap.read()
            if not ret:
                break

            # Collect active blur specs for this frame
            active_specs = [s for s in blur_specs
                            if s["frame_start"] <= global_frame <= s["frame_end"]]

            if active_specs:
                frame = np.ascontiguousarray(frame, dtype=np.uint8)

                # Hand protection activates for both cameras whenever ANY
                # segment overlaps the current frame -- matching the
                # preview's permissive behavior.  When a side-matching
                # segment exists we use its radius/smooth; otherwise we
                # fall back to the first segment that covers the frame so
                # both cameras render with the same look the preview shows.
                def _hand_active_for_side(side_label):
                    side_match = None
                    any_match = None
                    for seg in hand_segments:
                        if not (seg.get("start", 0) <= global_frame <= seg.get("end", 0)):
                            continue
                        seg_side = seg.get("side")
                        if any_match is None:
                            any_match = seg
                        if seg_side is None or seg_side == "full" or seg_side == side_label:
                            side_match = seg
                            break
                    chosen = side_match or any_match
                    if chosen is None:
                        return False, hand_mask_radius, hand_smooth
                    return (True,
                            chosen.get("radius", hand_mask_radius),
                            chosen.get("smooth", hand_smooth))

                if is_stereo:
                    left = frame[:, :half_w, :].copy()
                    right = frame[:, half_w:, :].copy()

                    left_specs = [s for s in active_specs if s.get("side", "left") == "left"]
                    right_specs = [s for s in active_specs if s.get("side", "right") == "right"]

                    if left_specs:
                        left_mask = _build_blur_mask(left_specs, half_w, fh, global_frame, face_by_frame, "left")
                        hand_mask_l = np.zeros((fh, half_w), dtype=bool)
                        hand_active_l, active_radius_l, active_smooth_l = _hand_active_for_side("left")
                        if hand_active_l:
                            lms_l = ([lm for lm in hand_lm_data[global_frame] if lm["side"] == "left"]
                                     if hand_lm_data and global_frame in hand_lm_data else [])
                            if lms_l:
                                hand_mask_l = _get_hand_mask_for_render(
                                    "left", lms_l, half_w, fh, active_radius_l, active_smooth_l,
                                    forearm_radius, forearm_extent, hand_smooth2,
                                    dlc_rad=dlc_radius_val, global_frame=global_frame)
                        left = _apply_blur_roi(left, left_mask, hand_mask_l)

                    if right_specs:
                        right_mask = _build_blur_mask(right_specs, full_w - half_w, fh, global_frame, face_by_frame, "right")
                        hand_mask_r = np.zeros((fh, full_w - half_w), dtype=bool)
                        hand_active_r, active_radius_r, active_smooth_r = _hand_active_for_side("right")
                        if hand_active_r:
                            lms_r = ([lm for lm in hand_lm_data[global_frame] if lm["side"] == "right"]
                                     if hand_lm_data and global_frame in hand_lm_data else [])
                            if lms_r:
                                hand_mask_r = _get_hand_mask_for_render(
                                    "right", lms_r, full_w - half_w, fh, active_radius_r, active_smooth_r,
                                    forearm_radius, forearm_extent, hand_smooth2,
                                    dlc_rad=dlc_radius_val, global_frame=global_frame)
                        right = _apply_blur_roi(right, right_mask, hand_mask_r)

                    frame = np.concatenate([left, right], axis=1)
                else:
                    blur_mask = _build_blur_mask(active_specs, full_w, fh, global_frame, face_by_frame, "full")
                    hand_mask = np.zeros((fh, full_w), dtype=bool)
                    hand_active_f, active_radius_f, active_smooth_f = _hand_active_for_side("full")
                    if hand_active_f:
                        lms_f = (hand_lm_data[global_frame]
                                 if hand_lm_data and global_frame in hand_lm_data else [])
                        if lms_f:
                            hand_mask = _get_hand_mask_for_render(
                                "full", lms_f, full_w, fh, active_radius_f, active_smooth_f,
                                forearm_radius, forearm_extent, hand_smooth2,
                                dlc_rad=dlc_radius_val, global_frame=global_frame)
                    frame = _apply_blur_roi(frame, blur_mask, hand_mask)

            # Write frame to ffmpeg's stdin (raw bgr24 bytes).
            # ──────────────────────────────────────────────────────
            # On Windows, the BufferedWriter wrapping proc.stdin has
            # produced an unreproducible-locally [Errno 22] on the
            # very first 12 MB write into a piped ffmpeg.  Writing
            # in 64 KB chunks via os.write directly side-steps the
            # BufferedWriter / WriteFile interaction completely and
            # works reliably on every Windows host we've tested it
            # on.  Same logic runs on POSIX too -- it's just slower
            # by a few % which is fine for I/O-bound rendering.
            _CHUNK = 65536
            _buf = frame.tobytes()
            _fd = proc.stdin.fileno()
            _view = memoryview(_buf)
            _off = 0
            _n = len(_buf)
            while _off < _n:
                # If ffmpeg died mid-render, surface the real error.
                if proc.poll() is not None:
                    _stderr_thr.join(timeout=0.5)
                    stderr = b"".join(_stderr_buf).decode(errors="replace")
                    raise RuntimeError(
                        f"ffmpeg exited mid-render after {frames_written} "
                        f"frames (rc={proc.returncode}). stderr: {stderr[:500]}"
                    )
                _written = os.write(_fd, _view[_off:_off + _CHUNK])
                if _written <= 0:
                    raise RuntimeError(
                        f"os.write returned {_written} on ffmpeg stdin "
                        f"after {frames_written} frames"
                    )
                _off += _written
            frames_written += 1

            if progress_callback and i % 10 == 0:
                progress_callback(i / total * 100)

    finally:
        cap.release()
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.wait(timeout=600)
        # Background drainer already pulled everything; just join it.
        _stderr_thr.join(timeout=1.0)
        stderr_bytes = b"".join(_stderr_buf)

    if proc.returncode != 0:
        stderr = stderr_bytes.decode(errors="replace") if stderr_bytes else ""
        raise RuntimeError(f"ffmpeg encoding failed: {stderr[:500]}")

    if progress_callback:
        progress_callback(100)

    # ── Sidecar metadata ────────────────────────────────────────────
    # Drop ``<output>.params.json`` next to the rendered .mp4 capturing
    # the hand-mask + arm-geometry parameters used to produce it.  The
    # Deidentify page reads this on load to show whether the current
    # slider values diverge from what produced the existing render.
    import json as _json
    from datetime import datetime as _dt
    _params = {
        "job_type": "deidentify",
        "hand_mask_radius": int(hand_mask_radius),
        "hand_smooth": int(hand_smooth),
        "hand_smooth2": int(hand_smooth2),
        "forearm_radius": int(forearm_radius),
        "forearm_extent": float(forearm_extent),
        "arm_dorsal_dilate": int(arm_dorsal_dilate),
        "arm_ventral_dilate": int(arm_ventral_dilate),
        "dlc_radius": int(dlc_radius_val),
        "ran_at": _dt.utcnow().isoformat(timespec="seconds") + "Z",
    }
    _sidecar = os.path.splitext(output_path)[0] + ".params.json"
    try:
        with open(_sidecar, "w") as _f:
            _json.dump(_params, _f, indent=2)
    except OSError as _e:
        logger.warning(f"Failed to write {_sidecar}: {_e}")

    logger.info(f"Rendered {frames_written} frames → {output_path}")
    return {"n_frames": frames_written, "is_stereo": is_stereo}


def _get_face_centroid(face_by_frame: dict, global_frame: int, side: str,
                       ref_x: float, ref_y: float) -> tuple | None:
    """Find the face detection closest to (ref_x, ref_y) for this frame and side."""
    faces = face_by_frame.get(global_frame, [])
    best = None
    best_dist = float("inf")
    for f in faces:
        if f.get("side", "full") != side:
            continue
        cx = (f["x1"] + f["x2"]) / 2
        cy = (f["y1"] + f["y2"]) / 2
        dist = (cx - ref_x) ** 2 + (cy - ref_y) ** 2
        if dist < best_dist:
            best_dist = dist
            best = (cx, cy)
    return best


def _build_blur_mask(specs: list[dict], w: int, h: int,
                     global_frame: int, face_by_frame: dict,
                     side: str) -> np.ndarray:
    """Build blur mask with ellipses, tracking face centroids for face spots."""
    mask = np.zeros((h, w), dtype=np.uint8)

    for s in specs:
        # Determine center position
        cx, cy = float(s["x"]), float(s["y"])
        ox = float(s.get("offset_x") or 0)
        oy = float(s.get("offset_y") or 0)

        if s.get("spot_type") == "face":
            # Track face detection centroid for this frame
            centroid = _get_face_centroid(face_by_frame, global_frame, side, cx, cy)
            if centroid:
                cx, cy = centroid[0] + ox, centroid[1] + oy
            else:
                cx, cy = cx + ox, cy + oy
        else:
            # Custom spots: apply offset to stored position
            cx, cy = cx + ox, cy + oy

        # Use width/height if available, otherwise fall back to radius
        # width/height are full dimensions; cv2.ellipse needs semi-axes (half)
        bw = float(s.get("width") or s.get("radius") or 50)
        bh = float(s.get("height") or s.get("radius") or 50)

        # Draw filled shape
        shape = s.get("shape", "oval")
        center = (int(cx), int(cy))
        axes = (int(bw / 2), int(bh / 2))
        if axes[0] > 0 and axes[1] > 0:
            if shape == "rect":
                x1r = max(0, int(cx - bw / 2))
                y1r = max(0, int(cy - bh / 2))
                x2r = min(w, int(cx + bw / 2))
                y2r = min(h, int(cy + bh / 2))
                cv2.rectangle(mask, (x1r, y1r), (x2r, y2r), 255, -1)
            else:
                cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

    return mask


def _build_hand_mask_from_landmarks(landmarks: list[dict], w: int, h: int,
                                     radius: int, smooth: int = 0,
                                     forearm_radius: int = 10,
                                     forearm_extent: float = 0.7,
                                     smooth2: int = 0,
                                     arm_dorsal_dilate: int = 0,
                                     arm_ventral_dilate: int = 0,
                                     **kwargs) -> np.ndarray:
    """Build hand protection mask from stored landmarks (matching frontend behavior).

    Builds a HAND mask (circles at every keypoint, dilated by ``smooth``)
    and a separate ARM-triangle mask (filled pinky-MCP → elbow → thumb-CMC
    polygon, plus per-edge dilation: ``arm_dorsal_dilate`` thickens the
    dorsal line elbow→pinky-MCP, ``arm_ventral_dilate`` thickens the
    ventral line elbow→thumb-CMC).  The two are unioned at the very end
    so the Hand-dilate slider grows the hand region WITHOUT also growing
    the arm.
    """
    mask = np.zeros((h, w), dtype=np.uint8)

    if not landmarks:
        return mask > 0

    # Hand landmarks (MediaPipe) + DLC fallback ONLY for joints MP
    # missed.  Two earlier strategies both regressed:
    #   - "replace MP with DLC" (original): when DLC's thumb/index
    #     prediction was wrong, the mask drifted off the MP fingertip
    #     the user could still see on screen (Con02_R2 frame 17).
    #   - "add DLC alongside MP" (previous fix): when DLC was bad on
    #     one camera, the mask grew an extra circle far from the hand
    #     -- visible as the OS-mask-too-large / slider-frozen feel.
    # Treating DLC strictly as a fill-in keeps MP authoritative
    # whenever it detected the joint, and only draws a DLC-derived
    # circle when MP genuinely lacked that joint.  Both regressions
    # are avoided and the mask is symmetric across the two cameras.
    dlc_lms = [lm for lm in landmarks if lm.get("type") == "dlc"]
    hand_lms = [lm for lm in landmarks if lm.get("type") not in ("pose", "dlc")]
    pose_lms = [lm for lm in landmarks if lm.get("type") == "pose"]

    if dlc_lms:
        mp_joints = {lm.get("joint") for lm in hand_lms
                      if lm.get("type") == "hand" and lm.get("joint") is not None}
        for dlc in dlc_lms:
            if dlc.get("joint") not in mp_joints:
                hand_lms.append({**dlc, "type": "hand"})

    # Slider values are in image pixel units — the same coordinate space as the
    # stored landmarks.  The frontend draws circles at (radius * canvasScale)
    # canvas pixels, where canvasScale = canvasWidth / imageWidth, which means
    # the effective image-pixel coverage is just `radius` unchanged.  No
    # conversion is needed here; applying image_w/canvas_w was a factor-of-2–3
    # overscaling that caused the preview to look larger than the masks overlay.

    # Interpolate midpoints along each finger segment for smoother coverage.
    # MediaPipe joints: 0=wrist, 1-4=thumb, 5-8=index, 9-12=middle,
    #                   13-16=ring, 17-20=pinky.
    # Finger inter-joint segments (e.g. MCP→PIP) get a single 1/2 fill —
    # fingers are thin so one extra circle is enough.  The wrist→MCP
    # segment (the "palm" — segment 0 of each chain) gets three rows of
    # fills at 1/4, 1/2, and 3/4 the distance from the wrist toward the
    # knuckle so the wider palm region is fully covered without leaving
    # gaps between rows.
    finger_chains = [
        [0, 1, 2, 3, 4],
        [0, 5, 6, 7, 8],
        [0, 9, 10, 11, 12],
        [0, 13, 14, 15, 16],
        [0, 17, 18, 19, 20],
    ]
    by_joint = {lm.get("joint", -1): lm for lm in hand_lms}
    all_points = list(hand_lms)
    for chain in finger_chains:
        for ci in range(len(chain) - 1):
            a = by_joint.get(chain[ci])
            b = by_joint.get(chain[ci + 1])
            if not (a and b):
                continue
            # Three palm-fill rows on the wrist→MCP segment (ci=0); one
            # midpoint elsewhere.
            fracs = (0.25, 0.5, 0.75) if ci == 0 else (0.5,)
            for f in fracs:
                all_points.append({
                    "x": a["x"] + f * (b["x"] - a["x"]),
                    "y": a["y"] + f * (b["y"] - a["y"]),
                    "type": "interp",
                })

    # Extra midpoint between thumb MCP (joint 2) and index MCP (joint 5)
    # Fills the gap in the web space between thumb and index finger
    thumb_mcp = by_joint.get(2)
    index_mcp = by_joint.get(5)
    if thumb_mcp and index_mcp:
        all_points.append({
            "x": (thumb_mcp["x"] + index_mcp["x"]) / 2,
            "y": (thumb_mcp["y"] + index_mcp["y"]) / 2,
            "type": "interp",
        })

    # DLC tip-disagreement: seed extra circles along MCP/CMC -> DLC-tip
    # line so the mask covers both possibilities whenever EITHER:
    #   * DLC's tip is > 20 px from MP's tip (the original signal — MP
    #     and DLC disagree on where the tip is), OR
    #   * MP's tip on this frame is interpolated (the original
    #     MediaPipe pass had NaN for this joint on this frame, so the
    #     MP tip we're using is just a linear blend of neighbours; DLC
    #     is the only actual per-frame measurement).
    # Joint numbers: thumb 1=CMC, 4=tip; index 5=MCP, 8=tip.
    _DLC_TIP_DISAGREE_PX = 20.0
    _dlc_by_joint = {lm.get("joint"): lm for lm in dlc_lms
                      if lm.get("joint") is not None}
    for (mcp_id, tip_id) in ((5, 8), (1, 4)):
        mcp_lm = by_joint.get(mcp_id)
        mp_tip = by_joint.get(tip_id)
        dlc_tip = _dlc_by_joint.get(tip_id)
        if not (mcp_lm and mp_tip and dlc_tip):
            continue
        dx_t = float(mp_tip["x"]) - float(dlc_tip["x"])
        dy_t = float(mp_tip["y"]) - float(dlc_tip["y"])
        far_apart = (dx_t * dx_t + dy_t * dy_t) ** 0.5 > _DLC_TIP_DISAGREE_PX
        mp_interpolated = bool(mp_tip.get("interpolated"))
        if not (far_apart or mp_interpolated):
            continue
        dx = float(dlc_tip["x"]) - float(mcp_lm["x"])
        dy = float(dlc_tip["y"]) - float(mcp_lm["y"])
        for f in (1/6, 1/3, 1/2, 2/3, 5/6, 1.0):
            all_points.append({
                "x": float(mcp_lm["x"]) + f * dx,
                "y": float(mcp_lm["y"]) + f * dy,
                "type": "dlc_interp",
            })

    # Extrapolate any missing fingertip (joints 4, 8, 12, 16, 20) when
    # the two joints below it (DIP + PIP / IP + MCP for the thumb) are
    # present.  Without this, MP drops on the very tip leave the
    # fingertip pad uncovered -- which is the most identifying piece of
    # the hand and the part the user most needs blurred.
    #
    # Geometry: assume the finger is roughly straight at the tip, i.e.
    # tip - PIP ≈ PIP - DIP, so tip ≈ 2*PIP - DIP.  Falls back to a
    # half-segment extension when only one joint below the tip exists.
    for chain in finger_chains:
        tip = chain[-1]
        if tip in by_joint:
            continue
        pip = by_joint.get(chain[-2])
        dip = by_joint.get(chain[-3]) if len(chain) >= 3 else None
        ext = None
        if pip and dip:
            ext = {
                "x": 2 * pip["x"] - dip["x"],
                "y": 2 * pip["y"] - dip["y"],
                "type": "extrap", "joint": tip,
            }
        elif pip:
            # Only one joint below — extend half a segment in the
            # palm-to-PIP direction, using the wrist as the anchor.
            wrist = by_joint.get(0)
            if wrist:
                ext = {
                    "x": pip["x"] + 0.5 * (pip["x"] - wrist["x"]),
                    "y": pip["y"] + 0.5 * (pip["y"] - wrist["y"]),
                    "type": "extrap", "joint": tip,
                }
        if ext is not None:
            by_joint[tip] = ext
            hand_lms.append(ext)
            all_points.append(ext)
            # Also bridge the half-segment between PIP and the
            # extrapolated tip so there isn't a coverage gap.
            all_points.append({
                "x": (pip["x"] + ext["x"]) / 2,
                "y": (pip["y"] + ext["y"]) / 2,
                "type": "interp",
            })

    # ── Hand-only mask: circles at each landmark + interpolated /
    # extrapolated points, optionally smoothed (the Hand-dilate slider).
    # The arm triangle is built INDEPENDENTLY below so Hand-dilate
    # doesn't also grow the arm.
    hand_mask = np.zeros((h, w), dtype=np.uint8)
    import math
    for lm in all_points:
        xv, yv = lm.get("x"), lm.get("y")
        if xv is None or yv is None:
            continue
        try:
            xf, yf = float(xv), float(yv)
        except (TypeError, ValueError):
            continue
        if math.isnan(xf) or math.isnan(yf):
            continue
        cv2.circle(hand_mask, (int(xf), int(yf)), radius, 255, -1)
    if smooth > 0 and hand_mask.any():
        # Pixel-exact mirror of the frontend's ``_morphClose`` in
        # static/js/deidentify.js: Gaussian blur, then 8 alpha-stacked
        # source-over draws (= ``1 - (1 - α)^8``), then threshold at
        # 30/255.  Using the same compositing math here keeps the
        # rendered/preview hand mask identical to the green outline
        # shown in the edit view.
        blurred = cv2.GaussianBlur(hand_mask.astype(np.float32) / 255.0,
                                    (0, 0), smooth)
        stacked = 1.0 - np.power(1.0 - np.clip(blurred, 0.0, 1.0), 8)
        hand_mask = (stacked > 30.0 / 255.0).astype(np.uint8) * 255

    # ── Arm-triangle mask (independent of Hand-dilate).
    # Triangle: pinky MCP (17) → elbow → thumb CMC (1).  Dorsal edge =
    # elbow→pinky-MCP (line thickness = arm_dorsal_dilate * 2), ventral
    # edge = elbow→thumb-CMC (line thickness = arm_ventral_dilate * 2).
    arm_mask = np.zeros((h, w), dtype=np.uint8)
    pinky_mcp = next((lm for lm in hand_lms if lm.get("joint") == 17), None)
    thumb_cmc = next((lm for lm in hand_lms if lm.get("joint") == 1), None)
    hand_wrist = next((lm for lm in hand_lms if lm.get("joint") == 0), None)

    # Pick the elbow closest to the hand WRIST.  Mirrors the live JS
    # overlay (_buildHandMask in deidentify.js) -- previously this
    # path used the hand-centroid heuristic, which could land on the
    # OTHER arm's elbow when both elbows were visible in the camera
    # and the hand pose biased the centroid away from the wrist.
    elbows = [lm for lm in pose_lms if lm.get("joint") in (13, 14)]
    elbow = None
    if hand_wrist and elbows:
        wx, wy = hand_wrist["x"], hand_wrist["y"]
        elbow = min(elbows, key=lambda e: (e["x"] - wx) ** 2 + (e["y"] - wy) ** 2)

    if pinky_mcp and thumb_cmc and elbow and hand_wrist:
        # Apply forearm extent: interpolate elbow toward wrist
        wx, wy = hand_wrist["x"], hand_wrist["y"]
        ex, ey = elbow["x"], elbow["y"]
        ext_x = wx + forearm_extent * (ex - wx)
        ext_y = wy + forearm_extent * (ey - wy)
        pts = np.array([
            [int(pinky_mcp["x"]), int(pinky_mcp["y"])],
            [int(ext_x), int(ext_y)],
            [int(thumb_cmc["x"]), int(thumb_cmc["y"])],
        ], dtype=np.int32)
        cv2.fillPoly(arm_mask, [pts], 255)
        # Dorsal edge dilation: elbow → pinky MCP.
        if int(arm_dorsal_dilate) > 0:
            cv2.line(arm_mask,
                     (int(ext_x), int(ext_y)),
                     (int(pinky_mcp["x"]), int(pinky_mcp["y"])),
                     255, int(arm_dorsal_dilate) * 2)
        # Ventral edge dilation: elbow → thumb CMC.
        if int(arm_ventral_dilate) > 0:
            cv2.line(arm_mask,
                     (int(ext_x), int(ext_y)),
                     (int(thumb_cmc["x"]), int(thumb_cmc["y"])),
                     255, int(arm_ventral_dilate) * 2)

    # Union hand + arm.
    mask = cv2.bitwise_or(hand_mask, arm_mask)
    return mask > 0


def _apply_blur_with_mask(frame_half, blur_mask, hand_mask):
    """Apply Gaussian blur to blur_mask regions, excluding hand_mask."""
    if blur_mask.max() == 0:
        return frame_half

    h, w = frame_half.shape[:2]

    # Subtract hand protection from blur mask
    final_mask = blur_mask.copy()
    if hand_mask is not None and hand_mask.any():
        final_mask[hand_mask] = 0

    if final_mask.max() == 0:
        return frame_half

    # Feather the mask edges
    mask_f = cv2.GaussianBlur(
        final_mask.astype(np.float32) / 255.0,
        (FEATHER_KERNEL, FEATHER_KERNEL), 5)

    blurred = cv2.GaussianBlur(frame_half, (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), BLUR_SIGMA)

    mask_3ch = mask_f[:, :, np.newaxis]
    result = (frame_half.astype(np.float32) * (1 - mask_3ch) +
              blurred.astype(np.float32) * mask_3ch).astype(np.uint8)

    # Restore original pixels in hand protection area
    if hand_mask is not None and hand_mask.any():
        result[hand_mask] = frame_half[hand_mask]

    return result


def _apply_blur_roi(frame_half, blur_mask, hand_mask):
    """Apply blur only to the bounding box of the mask region (ROI optimization).

    Instead of blurring the entire frame (expensive for 3840x1080), finds the
    bounding box of the blur mask, blurs only that region, and composites back.
    For a typical 200x200 face region on a 1920x1080 half-frame, this is ~50x faster.
    """
    if blur_mask.max() == 0:
        return frame_half

    # Subtract hand protection from blur mask
    final_mask = blur_mask.copy()
    if hand_mask is not None and hand_mask.any():
        final_mask[hand_mask] = 0

    if final_mask.max() == 0:
        return frame_half

    # Find bounding box of blur mask and process just that ROI.
    # Padding ensures the blur kernel has enough surrounding context.
    pad = BLUR_KERNEL_SIZE // 2 + FEATHER_KERNEL
    h, w = frame_half.shape[:2]

    nz = np.nonzero(final_mask)
    if len(nz[0]) == 0:
        return frame_half

    y1 = max(0, nz[0].min() - pad)
    y2 = min(h, nz[0].max() + pad + 1)
    x1 = max(0, nz[1].min() - pad)
    x2 = min(w, nz[1].max() + pad + 1)

    result = frame_half.copy()

    roi_mask = final_mask[y1:y2, x1:x2]
    roi = frame_half[y1:y2, x1:x2].copy()

    mask_f = cv2.GaussianBlur(
        roi_mask.astype(np.float32) / 255.0,
        (FEATHER_KERNEL, FEATHER_KERNEL), 5)

    # Zero out feathered mask wherever hand protection is active
    if hand_mask is not None and hand_mask.any():
        roi_hand = hand_mask[y1:y2, x1:x2]
        mask_f[roi_hand] = 0

    blurred_roi = cv2.GaussianBlur(roi, (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), BLUR_SIGMA)

    mask_3ch = mask_f[:, :, np.newaxis]
    composited = (roi.astype(np.float32) * (1 - mask_3ch) +
                  blurred_roi.astype(np.float32) * mask_3ch).astype(np.uint8)

    result[y1:y2, x1:x2] = composited

    # Final safety: restore original pixels in hand protection area
    if hand_mask is not None and hand_mask.any():
        result[hand_mask] = frame_half[hand_mask]

    return result


def _composite_with_preblurred(original, preblurred, blur_mask, hand_mask):
    """Composite original and pre-blurred frames using the blur mask.

    Like _apply_blur_with_mask but uses a pre-blurred frame instead of
    computing GaussianBlur per-frame. Much faster for video rendering.
    """
    if blur_mask.max() == 0:
        return original

    # Subtract hand protection from blur mask
    final_mask = blur_mask.copy()
    if hand_mask is not None and hand_mask.any():
        final_mask[hand_mask] = 0

    if final_mask.max() == 0:
        return original

    # Feather the mask edges
    mask_f = cv2.GaussianBlur(
        final_mask.astype(np.float32) / 255.0,
        (FEATHER_KERNEL, FEATHER_KERNEL), 5)

    # Composite: original where mask=0, preblurred where mask=1
    mask_3ch = mask_f[:, :, np.newaxis]
    result = (original.astype(np.float32) * (1 - mask_3ch) +
              preblurred.astype(np.float32) * mask_3ch).astype(np.uint8)

    # Restore original pixels in hand protection area
    if hand_mask is not None and hand_mask.any():
        result[hand_mask] = original[hand_mask]

    return result
