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
    # Remove existing output file — macOS AVFoundation refuses to overwrite
    import os, sys
    if os.path.exists(output_path):
        os.remove(output_path)

    # Try avc1 (H.264 via VideoToolbox) on macOS, fall back to mp4v
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
                           progress_callback=None) -> dict:
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
    if hand_settings:
        import json as _json
        hand_mask_radius = hand_settings.get("hand_mask_radius") or hand_settings.get("mask_radius") or 10
        hand_smooth = hand_settings.get("hand_smooth", 10)
        forearm_radius = hand_settings.get("forearm_radius", 10)
        forearm_extent = hand_settings.get("forearm_extent", 0.7)
        hand_smooth2 = hand_settings.get("hand_smooth2", 0)
        seg_json = hand_settings.get("segments_json", "[]")
        try:
            hand_segments = _json.loads(seg_json) if isinstance(seg_json, str) else (seg_json or [])
        except (ValueError, TypeError):
            hand_segments = []
        if not hand_segments:
            hand_segments = []

        # Load from npz if subject_name provided
        if subject_name and hand_segments:
            from ..config import get_settings
            settings = get_settings()
            npz_path = settings.dlc_path / subject_name / "mediapipe_prelabels.npz"
            if npz_path.exists():
                import numpy as np2
                npz = np2.load(str(npz_path))
                os_lm = npz.get("OS_landmarks")
                od_lm = npz.get("OD_landmarks")
                hand_lm_data = {}
                if os_lm is not None:
                    for f in range(os_lm.shape[0]):
                        if f not in hand_lm_data:
                            hand_lm_data[f] = []
                        for j in range(os_lm.shape[1]):
                            x, y = os_lm[f, j, 0], os_lm[f, j, 1]
                            if not np.isnan(x):
                                hand_lm_data[f].append({"x": float(x), "y": float(y), "side": "left", "type": "hand", "joint": j})
                if od_lm is not None:
                    for f in range(od_lm.shape[0]):
                        if f not in hand_lm_data:
                            hand_lm_data[f] = []
                        for j in range(od_lm.shape[1]):
                            x, y = od_lm[f, j, 0], od_lm[f, j, 1]
                            if not np.isnan(x):
                                hand_lm_data[f].append({"x": float(x), "y": float(y), "side": "right", "type": "hand", "joint": j})

                # Also load pose landmarks for forearm triangle (elbow)
                pose_path = settings.dlc_path / subject_name / "pose_prelabels.npz"
                if pose_path.exists():
                    pose_npz = np2.load(str(pose_path))
                    pose_os = pose_npz.get("OS_pose")
                    pose_od = pose_npz.get("OD_pose")
                    if pose_os is not None:
                        for f in range(pose_os.shape[0]):
                            if f not in hand_lm_data:
                                hand_lm_data[f] = []
                            for j in range(pose_os.shape[1]):
                                x, y = pose_os[f, j, 0], pose_os[f, j, 1]
                                if not np.isnan(x):
                                    hand_lm_data[f].append({"x": float(x), "y": float(y), "side": "left", "type": "pose", "joint": j})
                    if pose_od is not None:
                        for f in range(pose_od.shape[0]):
                            if f not in hand_lm_data:
                                hand_lm_data[f] = []
                            for j in range(pose_od.shape[1]):
                                x, y = pose_od[f, j, 0], pose_od[f, j, 1]
                                if not np.isnan(x):
                                    hand_lm_data[f].append({"x": float(x), "y": float(y), "side": "right", "type": "pose", "joint": j})

    # Only seek if start_frame is within this file's frame count.
    # start_frame is a global offset — for multi-file subjects each trial
    # has its own video starting at local frame 0, so don't seek past the end.
    if start_frame > 0 and start_frame < reported:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    import os, subprocess, tempfile

    # Remove existing output file
    if os.path.exists(output_path):
        os.remove(output_path)

    try:
        from .ffmpeg import get_ffmpeg_path
        ffmpeg = get_ffmpeg_path()
    except (ImportError, FileNotFoundError):
        cap.release()
        raise RuntimeError("ffmpeg not found. Install FFmpeg or pip install imageio-ffmpeg")

    # ── Pass 1: Pre-blur entire video with ffmpeg (fast, hardware-friendly) ──
    # This eliminates the expensive per-frame cv2.GaussianBlur.
    # boxblur luma=25:25 is roughly equivalent to GaussianBlur(99, sigma=50)
    tmp_dir = tempfile.mkdtemp(prefix="deid_")
    preblurred_path = os.path.join(tmp_dir, "preblurred.mp4")

    if progress_callback:
        progress_callback(1)

    # Build ffmpeg command: seek to start_frame, process 'total' frames
    blur_cmd = [
        ffmpeg, "-y",
        "-i", input_path,
    ]
    if start_frame > 0:
        blur_cmd += ["-ss", str(start_frame / fps)]
    blur_cmd += [
        "-frames:v", str(total),
        "-vf", "boxblur=luma=25:chroma=25",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "15",
        "-pix_fmt", "yuv420p", "-an",
        preblurred_path,
    ]

    result = subprocess.run(blur_cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        cap.release()
        raise RuntimeError(f"Pre-blur failed: {result.stderr[:500]}")

    if progress_callback:
        progress_callback(10)  # Pre-blur done (~10% of work)

    # ── Pass 2: Read original + pre-blurred, composite with masks, pipe to output ──
    blur_cap = cv2.VideoCapture(preblurred_path)
    if not blur_cap.isOpened():
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        cap.release()
        raise RuntimeError(f"Cannot open pre-blurred video: {preblurred_path}")

    # Reset original video to start
    if start_frame > 0 and start_frame < reported:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    out_cmd = [
        ffmpeg, "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{full_w}x{fh}", "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-an",
        output_path,
    ]

    proc = subprocess.Popen(
        out_cmd, stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )

    # Hand mask cache
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

    def _get_hand_mask_cached(side_label, lms, w_side, h_side, r, sm, fa_r, fa_e, sm2):
        key = side_label
        new_hash = _lm_hash(lms)
        params = (r, sm, fa_r, fa_e, sm2)
        cached = _hand_cache.get(key)
        if cached and cached["params"] == params and not _lm_moved(cached["lm_hash"], new_hash):
            return cached["mask"]
        mask = _build_hand_mask_from_landmarks(lms, w_side, h_side, r, sm, fa_r, fa_e, sm2)
        _hand_cache[key] = {"lm_hash": new_hash, "mask": mask, "params": params}
        return mask

    frames_written = 0
    try:
        for i in range(total):
            global_frame = start_frame + i
            ret_orig, frame = cap.read()
            ret_blur, blurred = blur_cap.read()
            if not ret_orig or not ret_blur:
                break

            # Collect active blur specs for this frame
            active_specs = [s for s in blur_specs
                            if s["frame_start"] <= global_frame <= s["frame_end"]]

            if active_specs:
                frame = np.ascontiguousarray(frame, dtype=np.uint8)
                blurred = np.ascontiguousarray(blurred, dtype=np.uint8)

                # Check if hand protection is active
                hand_active = False
                active_radius = hand_mask_radius
                active_smooth = hand_smooth
                for seg in hand_segments:
                    if seg.get("start", 0) <= global_frame <= seg.get("end", 0):
                        hand_active = True
                        active_radius = seg.get("radius", hand_mask_radius)
                        active_smooth = seg.get("smooth", hand_smooth)
                        break

                if is_stereo:
                    left_o = frame[:, :half_w, :]
                    right_o = frame[:, half_w:, :]
                    left_b = blurred[:, :half_w, :]
                    right_b = blurred[:, half_w:, :]

                    left_specs = [s for s in active_specs if s.get("side", "left") == "left"]
                    right_specs = [s for s in active_specs if s.get("side", "right") == "right"]

                    if left_specs:
                        left_mask = _build_blur_mask(left_specs, half_w, fh, global_frame, face_by_frame, "left")
                        hand_mask_l = np.zeros((fh, half_w), dtype=bool)
                        if hand_active and hand_lm_data and global_frame in hand_lm_data:
                            lms_l = [lm for lm in hand_lm_data[global_frame] if lm["side"] == "left"]
                            hand_mask_l = _get_hand_mask_cached(
                                "left", lms_l, half_w, fh, active_radius, active_smooth,
                                forearm_radius, forearm_extent, hand_smooth2)
                        left_o = _composite_with_preblurred(left_o, left_b, left_mask, hand_mask_l)

                    if right_specs:
                        right_mask = _build_blur_mask(right_specs, full_w - half_w, fh, global_frame, face_by_frame, "right")
                        hand_mask_r = np.zeros((fh, full_w - half_w), dtype=bool)
                        if hand_active and hand_lm_data and global_frame in hand_lm_data:
                            lms_r = [lm for lm in hand_lm_data[global_frame] if lm["side"] == "right"]
                            hand_mask_r = _get_hand_mask_cached(
                                "right", lms_r, full_w - half_w, fh, active_radius, active_smooth,
                                forearm_radius, forearm_extent, hand_smooth2)
                        right_o = _composite_with_preblurred(right_o, right_b, right_mask, hand_mask_r)

                    frame = np.concatenate([left_o, right_o], axis=1)
                else:
                    blur_mask = _build_blur_mask(active_specs, full_w, fh, global_frame, face_by_frame, "full")
                    hand_mask = np.zeros((fh, full_w), dtype=bool)
                    if hand_active and hand_lm_data and global_frame in hand_lm_data:
                        hand_mask = _get_hand_mask_cached(
                            "full", hand_lm_data[global_frame], full_w, fh, active_radius, active_smooth,
                            forearm_radius, forearm_extent, hand_smooth2)
                    frame = _composite_with_preblurred(frame, blurred, blur_mask, hand_mask)

            # Write raw BGR frame to ffmpeg stdin
            proc.stdin.write(frame.tobytes())
            frames_written += 1

            if progress_callback and i % 10 == 0:
                pct = 10 + (i / total * 90)  # 10-100% for pass 2
                progress_callback(pct)

    finally:
        cap.release()
        blur_cap.release()
        proc.stdin.close()
        proc.wait(timeout=120)
        # Clean up temp files
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if proc.returncode != 0:
        stderr = proc.stderr.read().decode() if proc.stderr else ""
        raise RuntimeError(f"ffmpeg encoding failed: {stderr[:500]}")

    if progress_callback:
        progress_callback(100)

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
                                     **kwargs) -> np.ndarray:
    """Build hand protection mask from stored landmarks (matching frontend behavior).

    Draws circles at hand keypoints, applies morphological close (smooth),
    then adds a forearm triangle (pinky MCP → elbow → thumb CMC) and
    applies a second smooth step.

    The frontend renders at screen pixel scale (typically ~700px viewport for
    a 1920px half-frame). To match the visual effect, we scale radius and smooth
    parameters proportionally to image resolution vs a reference display size.
    """
    mask = np.zeros((h, w), dtype=np.uint8)

    if not landmarks:
        return mask > 0

    hand_lms = [lm for lm in landmarks if lm.get("type") != "pose"]
    pose_lms = [lm for lm in landmarks if lm.get("type") == "pose"]

    # Slider values are in screen pixels (CSS px at typical canvas zoom).
    # Convert to image pixels: value * (imageWidth / typicalCanvasWidth).
    # The frontend draws at (value * canvasScale) screen px, where
    # canvasScale ≈ canvasWidth / imageWidth. Both produce the same
    # fractional coverage: value / imageWidth * (imageWidth / canvasWidth) = value / canvasWidth.
    canvas_w = kwargs.get("canvas_w", 700)
    scale_factor = max(1.0, w / canvas_w)
    radius = max(1, int(radius * scale_factor))
    smooth = max(0, int(smooth * scale_factor))
    forearm_radius = max(1, int(forearm_radius * scale_factor))
    smooth2 = max(0, int(smooth2 * scale_factor))

    # Interpolate midpoints along each finger segment for smoother coverage
    # MediaPipe joints: 0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky
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
            if a and b:
                all_points.append({
                    "x": (a["x"] + b["x"]) / 2,
                    "y": (a["y"] + b["y"]) / 2,
                    "type": "interp",
                })

    # Draw circles at each hand landmark + interpolated midpoints
    for lm in all_points:
        x, y = int(lm["x"]), int(lm["y"])
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(mask, (x, y), radius, 255, -1)

    # Apply blur + threshold for smooth (morphological close approximation)
    if smooth > 0 and mask.any():
        pre_count = np.count_nonzero(mask)
        k = 2 * smooth + 1
        if k % 2 == 0:
            k += 1
        blurred = cv2.GaussianBlur(mask, (k, k), 0)
        # Adaptive threshold: lower threshold if kernel is large relative to features
        # to avoid erasing the mask entirely
        for thresh in (30, 15, 5, 1):
            candidate = (blurred > thresh).astype(np.uint8) * 255
            if np.count_nonzero(candidate) >= pre_count * 0.5:
                mask = candidate
                break
        else:
            mask = (blurred > 0).astype(np.uint8) * 255

    # Add forearm triangle: pinky MCP (joint 17) → elbow → thumb CMC (joint 1)
    pinky_mcp = next((lm for lm in hand_lms if lm.get("joint") == 17), None)
    thumb_cmc = next((lm for lm in hand_lms if lm.get("joint") == 1), None)
    hand_wrist = next((lm for lm in hand_lms if lm.get("joint") == 0), None)

    # Pick the elbow closest to the hand centroid
    elbows = [lm for lm in pose_lms if lm.get("joint") in (13, 14)]
    elbow = None
    if hand_wrist and elbows:
        # Use centroid of hand landmarks to pick the right arm's elbow
        hand_xs = [lm["x"] for lm in hand_lms]
        hand_ys = [lm["y"] for lm in hand_lms]
        cx = sum(hand_xs) / len(hand_xs)
        cy = sum(hand_ys) / len(hand_ys)
        elbow = min(elbows, key=lambda e: (e["x"] - cx)**2 + (e["y"] - cy)**2)

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

        # Filled triangle
        cv2.fillPoly(mask, [pts], 255)

        # Dilate palmar edge (thumb CMC → elbow) by radius
        cv2.line(mask, (int(thumb_cmc["x"]), int(thumb_cmc["y"])),
                 (int(ext_x), int(ext_y)), 255, radius * 2)
        # Dilate dorsal edge (pinky MCP → elbow) by forearm_radius
        cv2.line(mask, (int(pinky_mcp["x"]), int(pinky_mcp["y"])),
                 (int(ext_x), int(ext_y)), 255, forearm_radius * 2)

    # Apply second blur + threshold after adding forearm (matches frontend)
    if smooth2 > 0 and mask.any():
        pre_count = np.count_nonzero(mask)
        k2 = 2 * smooth2 + 1
        if k2 % 2 == 0:
            k2 += 1
        blurred2 = cv2.GaussianBlur(mask, (k2, k2), 0)
        for thresh in (30, 15, 5, 1):
            candidate = (blurred2 > thresh).astype(np.uint8) * 255
            if np.count_nonzero(candidate) >= pre_count * 0.5:
                mask = candidate
                break
        else:
            mask = (blurred2 > 0).astype(np.uint8) * 255

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

    # Restore original pixels in hand protection area (feathering can bleed into it)
    # Dilate the hand mask slightly to ensure full coverage matches frontend display
    if hand_mask is not None and hand_mask.any():
        # Dilate by the feather kernel radius to cover any bleed from edge feathering
        dilate_r = FEATHER_KERNEL // 2
        if dilate_r > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_r * 2 + 1, dilate_r * 2 + 1))
            expanded_mask = cv2.dilate(hand_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        else:
            expanded_mask = hand_mask
        result[expanded_mask] = frame_half[expanded_mask]

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
        dilate_r = FEATHER_KERNEL // 2
        if dilate_r > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_r * 2 + 1, dilate_r * 2 + 1))
            expanded_mask = cv2.dilate(hand_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        else:
            expanded_mask = hand_mask
        result[expanded_mask] = original[expanded_mask]

    return result
