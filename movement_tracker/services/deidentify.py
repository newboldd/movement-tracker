"""Face de-identification with hand protection for stereo/single-camera videos.

Adapted from the mano pipeline's deidentify_videos.py, simplified for the
movement-tracker context. Uses only MediaPipe (no HRNet dependency).

Usage:
    from movement_tracker.services.deidentify import deidentify_video
    deidentify_video("input.mp4", "output.mp4")
"""
from __future__ import annotations

import logging
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

logger = logging.getLogger(__name__)

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
    # Try avc1 (H.264 via VideoToolbox) on macOS, fall back to mp4v
    import sys
    writer = None
    for codec in (["avc1", "mp4v"] if sys.platform == "darwin" else ["mp4v"]):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (full_w, h))
        if writer.isOpened():
            break
        writer.release()
        writer = None
    if writer is None:
        raise RuntimeError(f"cv2.VideoWriter failed to open: {output_path}")

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

        writer.write(out_frame)

        if progress_callback and t % 10 == 0:
            progress_callback(40 + t / n_frames * 60)  # Pass 2 = 40-100%

    cap.release()
    writer.release()
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

    if start_frame > 0:
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
                           start_frame: int = 0,
                           frame_count: int | None = None,
                           progress_callback=None) -> dict:
    """Render a video with explicit blur specifications.

    Args:
        input_path: Source video path
        output_path: Output video path
        blur_specs: List of {x, y, radius, frame_start, frame_end, side}
            Coordinates are in pixels (not normalized)
        hand_settings: {enabled: bool, mask_radius: int} or None
        start_frame: First frame to process
        frame_count: Number of frames (None = all from start)
        progress_callback: Optional callable(pct: float)

    Returns:
        dict with stats: n_frames, is_stereo
    """
    import mediapipe as mp_lib

    cap = cv2.VideoCapture(input_path)
    reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    full_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if full_w == 0 or h == 0:
        cap.release()
        raise RuntimeError(f"Cannot read video: {input_path}")

    is_stereo = (full_w / h) > 1.7
    half_w = full_w // 2 if is_stereo else full_w
    total = frame_count if frame_count else (reported - start_frame)

    use_hands = hand_settings and hand_settings.get("enabled", False)
    hand_mask_radius = hand_settings.get("mask_radius", 30) if hand_settings else 30

    hands = None
    if use_hands:
        hands = mp_lib.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=2,
            min_detection_confidence=0.3, min_tracking_confidence=0.3,
        )

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    import sys
    writer = None
    for codec in (["avc1", "mp4v"] if sys.platform == "darwin" else ["mp4v"]):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (full_w, h))
        if writer.isOpened():
            break
        writer.release()
        writer = None
    if writer is None:
        cap.release()
        raise RuntimeError(f"cv2.VideoWriter failed: {output_path}")

    for i in range(total):
        global_frame = start_frame + i
        ret, frame = cap.read()
        if not ret:
            break

        frame = np.ascontiguousarray(frame, dtype=np.uint8)

        # Collect active blur specs for this frame
        active_specs = [s for s in blur_specs
                        if s["frame_start"] <= global_frame <= s["frame_end"]]

        if active_specs:
            if is_stereo:
                left = frame[:, :half_w, :].copy()
                right = frame[:, half_w:, :].copy()

                left_specs = [s for s in active_specs if s.get("side", "left") == "left"]
                right_specs = [s for s in active_specs if s.get("side", "right") == "right"]

                # Build face boxes from specs
                left_boxes = _specs_to_boxes(left_specs, half_w, h)
                right_boxes = _specs_to_boxes(right_specs, full_w - half_w, h)

                # Hand protection
                hand_mask_l = np.zeros((h, half_w), dtype=bool)
                hand_mask_r = np.zeros((h, full_w - half_w), dtype=bool)
                if use_hands and hands:
                    rgb_l = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
                    rgb_r = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)
                    hand_mask_l = _build_hand_mask_with_radius(
                        left.shape, hands.process(rgb_l), hand_mask_radius)
                    hand_mask_r = _build_hand_mask_with_radius(
                        right.shape, hands.process(rgb_r), hand_mask_radius)

                left = _apply_face_blur(left, left_boxes, hand_mask_l)
                right = _apply_face_blur(right, right_boxes, hand_mask_r)
                frame = np.concatenate([left, right], axis=1)
            else:
                full_specs = [s for s in active_specs]
                boxes = _specs_to_boxes(full_specs, full_w, h)

                hand_mask = np.zeros((h, full_w), dtype=bool)
                if use_hands and hands:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    hand_mask = _build_hand_mask_with_radius(
                        frame.shape, hands.process(rgb), hand_mask_radius)

                frame = _apply_face_blur(frame, boxes, hand_mask)

        writer.write(frame)

        if progress_callback and i % 10 == 0:
            progress_callback(i / total * 100)

    cap.release()
    writer.release()
    if hands:
        hands.close()
    if progress_callback:
        progress_callback(100)

    return {"n_frames": total, "is_stereo": is_stereo}


def _specs_to_boxes(specs: list[dict], w: int, h: int) -> list[tuple]:
    """Convert blur specs (center + radius) to bounding box tuples."""
    boxes = []
    for s in specs:
        cx, cy, r = s["x"], s["y"], s["radius"]
        x1 = max(0, int(cx - r))
        y1 = max(0, int(cy - r))
        x2 = min(w, int(cx + r))
        y2 = min(h, int(cy + r))
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def _build_hand_mask_with_radius(shape, hand_results, radius: int):
    """Build hand exclusion mask with configurable dilation radius."""
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

    if radius > 0:
        k = 2 * radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, kernel)

    return mask > 0
