#!/usr/bin/env python
"""Remote preprocessing script for DLC labeler.

Self-contained (no dlc_app imports). Uploaded to remote host and executed via SSH.

Subcommands:
    mp       Run MediaPipe hand landmark extraction on stereo videos
    blur     Run face de-identification (blur) on stereo videos
    pipeline Run MP + blur as a detached process with status.json polling

Progress output: PROGRESS:<subject>:<pct> to stdout (parsed by orchestrator).
Pipeline mode writes status.json atomically for the local monitor to poll.

Usage:
    python remote_preprocess_script.py mp <video_dir> <output_dir> [--subjects ...]
    python remote_preprocess_script.py blur <video_dir> <output_dir> [--subjects ...]
    python remote_preprocess_script.py pipeline <video_dir> <output_dir> --steps mp blur [--subjects ...]
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import sys
import tempfile
import traceback
from collections import defaultdict
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# cv2 and numpy are deferred to inside functions so that the --log-file
# redirect in __main__ can capture any import errors from the detached process.

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────
N_JOINTS = 21
THUMB_TIP = 4
INDEX_TIP = 8
WRIST = 0

# Blur configuration (matches deidentify.py)
FACE_CONF_THRESHOLD = 0.2
FACE_MODEL_SELECTION = 1
BLUR_KERNEL_SIZE = 99
BLUR_SIGMA = 50
FACE_BOX_EXPAND = 0.4
FACE_BOX_EXPAND_UP = 0.6
HAND_HULL_DILATION_PX = 30
HAND_OVERLAP_PAD = 80
HAND_OUTLIER_WINDOW = 15
HAND_OUTLIER_THRESHOLD = 100
TEMPORAL_SIGMA_FRAMES = 3.0
FEATHER_KERNEL = 15
IOU_MATCH_THRESHOLD = 0.1


# ── Subject discovery ────────────────────────────────────────────────────

def discover_subjects(video_dir: str, subject_filter: list[str] | str | None = None) -> dict[str, list[str]]:
    """Discover subjects from video filenames.

    Expects files like {SubjectName}_{Trial}.mp4 (e.g. MSA01_L1.mp4).
    Returns dict mapping subject_name -> sorted list of video paths.
    """
    videos = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    subjects: dict[str, list[str]] = defaultdict(list)

    for v in videos:
        name = Path(v).stem
        m = re.match(r"^(.+?)_[LR]\d", name)
        if m:
            subj = m.group(1)
            subjects[subj].append(v)

    if subject_filter:
        if isinstance(subject_filter, str):
            subjects = {k: v for k, v in subjects.items() if k == subject_filter}
        else:
            filter_set = set(subject_filter)
            subjects = {k: v for k, v in subjects.items() if k in filter_set}

    return dict(subjects)


def emit_progress(subject: str, pct: float):
    """Print progress line for orchestrator parsing."""
    print(f"PROGRESS:{subject}:{pct:.1f}", flush=True)


def _write_status(status_file: str, phase: str, status: str,
                  progress_pct: float = 0.0, error: str | None = None,
                  current_subject: str = ""):
    """Write status.json atomically (temp + os.replace)."""
    data = {
        "phase": phase,
        "status": status,
        "progress_pct": round(progress_pct, 1),
        "error": error,
        "current_subject": current_subject,
        "pid": os.getpid(),
    }
    dir_name = os.path.dirname(os.path.abspath(status_file))
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        os.replace(tmp, status_file)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ── MediaPipe extraction ────────────────────────────────────────────────

def _extract_hands(results, width, height):
    """Extract detected hands as list of (21,2) arrays + confidence scores."""
    import numpy as np
    hands = []
    scores = []
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_lm, hand_cls in zip(results.multi_hand_landmarks,
                                     results.multi_handedness):
            kp = np.array([(lm.x * width, lm.y * height)
                           for lm in hand_lm.landmark])
            hands.append(kp)
            scores.append(hand_cls.classification[0].score)
    elif results.multi_hand_landmarks:
        for hand_lm in results.multi_hand_landmarks:
            kp = np.array([(lm.x * width, lm.y * height)
                           for lm in hand_lm.landmark])
            hands.append(kp)
            scores.append(0.5)
    return hands, scores


def _target_hand_label(video_name: str) -> str | None:
    """Parse target hand from video name (MediaPipe mirrored convention)."""
    m = re.search(r"_([LR])\d", video_name)
    if m:
        return "Right" if m.group(1) == "L" else "Left"
    return None


def run_mp_subject(subject_name: str, videos: list[str], output_dir: str):
    """Run MediaPipe on all stereo videos for a subject, save landmarks npz."""
    import cv2
    import numpy as np
    import mediapipe as mp_lib

    # Build trial map (contiguous frame offsets)
    trials = []
    offset = 0
    for vpath in videos:
        cap = cv2.VideoCapture(vpath)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        trials.append({"path": vpath, "start": offset, "n_frames": n_frames})
        offset += n_frames
    total_frames = offset

    if total_frames == 0:
        logger.warning(f"No frames found for {subject_name}")
        return

    # Allocate arrays: all 21 joints per camera, shape (total_frames, 21, 2)
    OS_landmarks = np.full((total_frames, N_JOINTS, 2), np.nan)
    OD_landmarks = np.full((total_frames, N_JOINTS, 2), np.nan)
    confidence_OS = np.full(total_frames, np.nan)
    confidence_OD = np.full(total_frames, np.nan)

    frames_done = 0

    for trial in trials:
        vpath = trial["path"]
        start_frame = trial["start"]
        n_frames = trial["n_frames"]
        video_name = Path(vpath).stem
        target_label = _target_hand_label(video_name)

        cap = cv2.VideoCapture(vpath)
        full_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if full_w == 0 or h == 0:
            cap.release()
            logger.warning(f"Cannot read video: {vpath}")
            frames_done += n_frames
            continue
        midline = full_w // 2

        mp_hands = mp_lib.solutions.hands
        n_hands = 1 if target_label else 2
        det_L = mp_hands.Hands(
            static_image_mode=False, max_num_hands=n_hands,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )
        det_R = mp_hands.Hands(
            static_image_mode=False, max_num_hands=n_hands,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )

        prev_wrist_L = None
        prev_wrist_R = None

        for local_frame in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                break

            frame = np.ascontiguousarray(frame, dtype=np.uint8)
            frame_L = frame[:, :midline, :]
            frame_R = frame[:, midline:, :]
            global_frame = start_frame + local_frame

            # Process left camera (OS)
            rgb_L = cv2.cvtColor(frame_L, cv2.COLOR_BGR2RGB)
            res_L = det_L.process(rgb_L)
            hands_L, scores_L = _extract_hands(res_L, midline, h)

            # Process right camera (OD)
            rgb_R = cv2.cvtColor(frame_R, cv2.COLOR_BGR2RGB)
            res_R = det_R.process(rgb_R)
            hands_R, scores_R = _extract_hands(res_R, full_w - midline, h)

            # Track hand identity by wrist proximity
            if hands_L:
                if prev_wrist_L is not None and len(hands_L) > 1:
                    idx = int(np.argmin([
                        np.linalg.norm(hh[WRIST] - prev_wrist_L) for hh in hands_L
                    ]))
                else:
                    idx = 0
                OS_landmarks[global_frame] = hands_L[idx]
                confidence_OS[global_frame] = scores_L[idx]
                prev_wrist_L = hands_L[idx][WRIST].copy()

            if hands_R:
                if prev_wrist_R is not None and len(hands_R) > 1:
                    idx = int(np.argmin([
                        np.linalg.norm(hh[WRIST] - prev_wrist_R) for hh in hands_R
                    ]))
                else:
                    idx = 0
                OD_landmarks[global_frame] = hands_R[idx]
                confidence_OD[global_frame] = scores_R[idx]
                prev_wrist_R = hands_R[idx][WRIST].copy()

            frames_done += 1
            if frames_done % 50 == 0:
                emit_progress(subject_name, (frames_done / total_frames) * 100)

        cap.release()
        det_L.close()
        det_R.close()

    # Save npz (distances left as NaN — computed locally with calibration)
    out_dir = Path(output_dir) / subject_name
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = str(out_dir / "mediapipe_prelabels.npz")

    np.savez(
        npz_path,
        OS_landmarks=OS_landmarks,
        OD_landmarks=OD_landmarks,
        confidence_OS=confidence_OS,
        confidence_OD=confidence_OD,
        distances=np.full(total_frames, np.nan),
        total_frames=total_frames,
    )

    valid_OS = np.sum(~np.isnan(OS_landmarks[:, 0, 0]))
    valid_OD = np.sum(~np.isnan(OD_landmarks[:, 0, 0]))
    logger.info(
        f"Saved {npz_path}: OS={valid_OS}/{total_frames}, OD={valid_OD}/{total_frames}"
    )
    emit_progress(subject_name, 100)


def cmd_mp(args):
    """MediaPipe subcommand handler."""
    subjects = discover_subjects(args.video_dir, args.subjects)
    if not subjects:
        logger.error("No subjects found")
        sys.exit(1)

    logger.info(f"Found {len(subjects)} subjects for MediaPipe")
    for name, videos in sorted(subjects.items()):
        logger.info(f"Processing {name} ({len(videos)} videos)")
        run_mp_subject(name, videos, args.output_dir)

    print("MP_COMPLETE", flush=True)


# ── Face blur (de-identification) ───────────────────────────────────────

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
    """Run face detection on a single frame half."""
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


def _load_overrides(video_name: str, overrides_file: str | None = None) -> dict:
    """Load per-video overrides from JSON.

    Returns dict with fields: fixed_faces_L, fixed_faces_R, extra_blur_L,
    extra_blur_R, max_tracks_L, max_tracks_R, reject_hand_outliers, skip.
    """
    defaults = {
        "fixed_faces_L": None, "fixed_faces_R": None,
        "extra_blur_L": None, "extra_blur_R": None,
        "max_tracks_L": None, "max_tracks_R": None,
        "reject_hand_outliers": False, "skip": False,
    }
    if not overrides_file or not os.path.exists(overrides_file):
        return defaults
    with open(overrides_file) as f:
        all_overrides = json.load(f)
    entry = all_overrides.get(video_name, {})
    for k in defaults:
        if k in entry:
            defaults[k] = entry[k]
    return defaults


def _filter_hand_overlapping_faces(faces_per_frame, hand_kps_per_frame):
    """Remove face detections whose center falls inside padded hand bbox."""
    import numpy as np
    if hand_kps_per_frame is None:
        return
    for t in range(min(len(faces_per_frame), len(hand_kps_per_frame))):
        kps = hand_kps_per_frame[t]
        if kps is None:
            continue
        valid = kps[~np.isnan(kps[:, 0])]
        if len(valid) < 3:
            continue
        hx1 = valid[:, 0].min() - HAND_OVERLAP_PAD
        hy1 = valid[:, 1].min() - HAND_OVERLAP_PAD
        hx2 = valid[:, 0].max() + HAND_OVERLAP_PAD
        hy2 = valid[:, 1].max() + HAND_OVERLAP_PAD
        kept = []
        for (fx1, fy1, fx2, fy2) in faces_per_frame[t]:
            cx = (fx1 + fx2) / 2
            cy = (fy1 + fy2) / 2
            if hx1 <= cx <= hx2 and hy1 <= cy <= hy2:
                continue
            kept.append((fx1, fy1, fx2, fy2))
        faces_per_frame[t] = kept


def _reject_hand_outliers(hand_kps_per_frame):
    """Set hand keypoints to None on frames with outlier centroids."""
    import numpy as np
    if hand_kps_per_frame is None:
        return
    n = len(hand_kps_per_frame)
    centroids = np.full((n, 2), np.nan)
    for t in range(n):
        kps = hand_kps_per_frame[t]
        if kps is None:
            continue
        valid = kps[~np.isnan(kps[:, 0])]
        if len(valid) >= 3:
            centroids[t] = valid.mean(axis=0)

    for t in range(n):
        if np.isnan(centroids[t, 0]):
            continue
        lo = max(0, t - HAND_OUTLIER_WINDOW)
        hi = min(n, t + HAND_OUTLIER_WINDOW + 1)
        window = centroids[lo:hi]
        valid_w = window[~np.isnan(window[:, 0])]
        if len(valid_w) < 3:
            continue
        med = np.median(valid_w, axis=0)
        dist = np.linalg.norm(centroids[t] - med)
        if dist > HAND_OUTLIER_THRESHOLD:
            hand_kps_per_frame[t] = None


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


def _smooth_face_detections(faces_per_frame, n_frames, w, h, max_tracks=None):
    """Track, interpolate, backward/forward-fill, and Gaussian-smooth face bboxes.

    Args:
        max_tracks: if set, keep only the N longest tracks (by detection count).
            Use 0 to suppress all face tracks for a camera.
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter1d

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

    # Optionally keep only the N longest tracks
    if max_tracks is not None and len(tracks) > max_tracks:
        tracks.sort(key=lambda tr: np.sum(~np.isnan(tr["boxes"][:, 0])),
                    reverse=True)
        tracks = tracks[:max_tracks]

    # Interpolate, backward-fill, forward-fill, smooth
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

        # Backward-fill: carry first known bbox to start of video
        first = valid_indices[0]
        if first > 0:
            boxes[:first] = boxes[first]

        # Forward-fill: carry last known bbox to end of video
        last = valid_indices[-1]
        if last < n_frames - 1:
            boxes[last + 1:] = boxes[last]

        # Gaussian smooth entire track (0..n)
        if n_frames > 1:
            for col in range(4):
                boxes[:, col] = gaussian_filter1d(
                    boxes[:, col], TEMPORAL_SIGMA_FRAMES)

    result = [[] for _ in range(n_frames)]
    for track in tracks:
        for t in range(n_frames):
            if not np.isnan(track["boxes"][t, 0]):
                box = tuple(np.clip(track["boxes"][t], 0, [w, h, w, h]).astype(int))
                result[t].append(box)

    return result


def _build_hand_mask_from_kps(shape, hand_kps):
    """Build hand exclusion mask from stored (21,2) keypoints."""
    import cv2
    import numpy as np
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if hand_kps is None:
        return mask > 0

    valid = hand_kps[~np.isnan(hand_kps[:, 0])]
    if len(valid) < 3:
        return mask > 0

    hull = cv2.convexHull(valid.astype(np.float32))
    cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)

    if HAND_HULL_DILATION_PX > 0:
        k = 2 * HAND_HULL_DILATION_PX + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, kernel)

    return mask > 0


def _apply_face_blur(frame_half, face_boxes, hand_mask, extra_blur=None):
    """Apply Gaussian blur to face regions, excluding hand region.

    Args:
        extra_blur: optional list of [x1,y1,x2,y2] — additional regions to blur
            unconditionally (e.g., monitors). No hand exclusion applied.
    """
    import cv2
    import numpy as np
    if not face_boxes and not extra_blur:
        return frame_half

    h, w = frame_half.shape[:2]

    face_mask = np.zeros((h, w), dtype=np.uint8)
    for (x1, y1, x2, y2) in face_boxes:
        face_mask[y1:y2, x1:x2] = 255

    face_mask[hand_mask] = 0

    # Add extra blur regions (no hand exclusion)
    if extra_blur:
        for (ex1, ey1, ex2, ey2) in extra_blur:
            ex1, ey1 = max(0, int(ex1)), max(0, int(ey1))
            ex2, ey2 = min(w, int(ex2)), min(h, int(ey2))
            face_mask[ey1:ey2, ex1:ex2] = 255

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


def _extract_hand_kps(hand_results, width, height):
    """Extract hand keypoints as list of (21,2) arrays from MP Hands results."""
    import numpy as np
    hands_kps = []
    if hand_results and hand_results.multi_hand_landmarks:
        for hand_lm in hand_results.multi_hand_landmarks:
            kp = np.array([(lm.x * width, lm.y * height)
                           for lm in hand_lm.landmark])
            hands_kps.append(kp)
    return hands_kps


def deidentify_video(input_path: str, output_path: str,
                     subject_name: str = "",
                     video_pct_base: float = 0,
                     video_pct_span: float = 100,
                     overrides_file: str | None = None):
    """De-identify a single video: blur faces, preserve hands.

    Flow:
      Pass 1: Detect faces AND hands, store per-frame keypoints.
      Apply overrides: hand outlier rejection, hand-overlap face filter,
        inject fixed_faces, smooth with backward-fill + max_tracks.
      Pass 2: Render with stored hand keypoints + extra_blur.
    """
    import cv2
    import numpy as np
    import mediapipe as mp_lib

    video_name = Path(input_path).stem
    overrides = _load_overrides(video_name, overrides_file)

    if overrides["skip"]:
        logger.info(f"Skipping {video_name} (override skip=true)")
        return

    cap = cv2.VideoCapture(input_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    full_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if full_w == 0 or h == 0:
        cap.release()
        raise RuntimeError(f"Cannot read video: {input_path}")

    is_stereo = (full_w / h) > 1.7
    half_w = full_w // 2 if is_stereo else full_w

    logger.info(
        f"Deidentifying {Path(input_path).name}: "
        f"{n_frames} frames, {'stereo' if is_stereo else 'mono'}, {half_w}x{h}"
    )

    face_det = mp_lib.solutions.face_detection.FaceDetection(
        model_selection=FACE_MODEL_SELECTION,
        min_detection_confidence=FACE_CONF_THRESHOLD,
    )
    hands_det = mp_lib.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=2,
        min_detection_confidence=0.3, min_tracking_confidence=0.3,
    )

    # ── Pass 1: Detect faces AND hands ──
    if is_stereo:
        faces_L_raw, faces_R_raw = [], []
        hand_kps_L, hand_kps_R = [], []  # per-frame (21,2) or None
    else:
        faces_raw = []
        hand_kps_all = []

    for t in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            if is_stereo:
                faces_L_raw.append([])
                faces_R_raw.append([])
                hand_kps_L.append(None)
                hand_kps_R.append(None)
            else:
                faces_raw.append([])
                hand_kps_all.append(None)
            continue

        frame = np.ascontiguousarray(frame, dtype=np.uint8)

        if is_stereo:
            left = frame[:, :half_w, :]
            right = frame[:, half_w:, :]
            rgb_l = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
            rgb_r = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)
            faces_L_raw.append(_detect_faces_in_half(rgb_l, face_det, half_w, h))
            faces_R_raw.append(_detect_faces_in_half(rgb_r, face_det, full_w - half_w, h))

            # Detect hands for keypoint storage
            res_l = hands_det.process(rgb_l)
            res_r = hands_det.process(rgb_r)
            kps_l = _extract_hand_kps(res_l, half_w, h)
            kps_r = _extract_hand_kps(res_r, full_w - half_w, h)
            hand_kps_L.append(kps_l[0] if kps_l else None)
            hand_kps_R.append(kps_r[0] if kps_r else None)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces_raw.append(_detect_faces_in_half(rgb, face_det, full_w, h))
            res = hands_det.process(rgb)
            kps = _extract_hand_kps(res, full_w, h)
            hand_kps_all.append(kps[0] if kps else None)

        if subject_name and t % 10 == 0:
            pct = video_pct_base + (t / n_frames * 40 / 100) * video_pct_span
            emit_progress(subject_name, pct)

    face_det.close()
    hands_det.close()

    # ── Apply overrides and filters ──
    if is_stereo:
        # Hand outlier rejection
        if overrides["reject_hand_outliers"]:
            logger.info(f"  Applying hand outlier rejection for {video_name}")
            _reject_hand_outliers(hand_kps_L)
            _reject_hand_outliers(hand_kps_R)

        # Filter false face detections overlapping hand
        _filter_hand_overlapping_faces(faces_L_raw, hand_kps_L)
        _filter_hand_overlapping_faces(faces_R_raw, hand_kps_R)

        # Temporal smoothing with backward-fill + max_tracks
        faces_L = _smooth_face_detections(faces_L_raw, n_frames, half_w, h,
                                           max_tracks=overrides["max_tracks_L"])
        faces_R = _smooth_face_detections(faces_R_raw, n_frames, full_w - half_w, h,
                                           max_tracks=overrides["max_tracks_R"])

        # Inject fixed_faces AFTER smoothing (constant regions that bypass
        # tracking/max_tracks — always present every frame)
        for key, face_list in [("fixed_faces_L", faces_L),
                                ("fixed_faces_R", faces_R)]:
            fixed = overrides[key]
            if fixed and isinstance(fixed, list):
                logger.info(f"  Injecting {len(fixed)} fixed face region(s) ({key})")
                for t in range(n_frames):
                    for box in fixed:
                        face_list[t].append(tuple(box))
    else:
        faces = _smooth_face_detections(faces_raw, n_frames, full_w, h)

    # ── Pass 2: Render with stored hand keypoints ──
    # Reopen video instead of seeking (seeking can be inaccurate with H.264)
    cap.release()
    cap = cv2.VideoCapture(input_path)
    codec = "avc1" if sys.platform == "darwin" else "mp4v"
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (full_w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open: {output_path} (codec={codec})")

    extra_blur_L = overrides["extra_blur_L"] if is_stereo else None
    extra_blur_R = overrides["extra_blur_R"] if is_stereo else None

    for t in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        frame = np.ascontiguousarray(frame, dtype=np.uint8)

        if is_stereo:
            left = frame[:, :half_w, :].copy()
            right = frame[:, half_w:, :].copy()

            # Build hand masks from stored keypoints
            hand_mask_l = _build_hand_mask_from_kps(left.shape, hand_kps_L[t])
            hand_mask_r = _build_hand_mask_from_kps(right.shape, hand_kps_R[t])

            left = _apply_face_blur(left, faces_L[t], hand_mask_l,
                                    extra_blur=extra_blur_L)
            right = _apply_face_blur(right, faces_R[t], hand_mask_r,
                                     extra_blur=extra_blur_R)

            out_frame = np.concatenate([left, right], axis=1)
        else:
            hand_mask = _build_hand_mask_from_kps(frame.shape, hand_kps_all[t])
            frame = _apply_face_blur(frame, faces[t], hand_mask)
            out_frame = frame

        writer.write(out_frame)

        if subject_name and t % 10 == 0:
            pct = video_pct_base + (40 + t / n_frames * 60) / 100 * video_pct_span
            emit_progress(subject_name, pct)

    cap.release()
    writer.release()

    logger.info(f"Deidentified: {Path(input_path).name}")


def run_blur_subject(subject_name: str, videos: list[str], output_dir: str,
                     overrides_file: str | None = None):
    """Run face blur on all videos for a subject."""
    out_subj = Path(output_dir) / subject_name / "deidentified"
    out_subj.mkdir(parents=True, exist_ok=True)

    total = len(videos)
    vid_span = 100.0 / total if total > 0 else 100.0

    for i, vpath in enumerate(videos):
        vid_base = i * vid_span
        out_name = Path(vpath).name
        out_path = str(out_subj / out_name)
        temp_path = out_path + ".tmp.mp4"

        try:
            deidentify_video(
                vpath, temp_path,
                subject_name=subject_name,
                video_pct_base=vid_base,
                video_pct_span=vid_span,
                overrides_file=overrides_file,
            )
            # deidentify_video may skip (override skip=true) — don't rename
            if os.path.exists(temp_path):
                os.replace(temp_path, out_path)
        except Exception as e:
            logger.error(f"Blur failed for {vpath}: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)

    emit_progress(subject_name, 100)


def cmd_blur(args):
    """Blur subcommand handler."""
    subjects = discover_subjects(args.video_dir, args.subjects)
    if not subjects:
        logger.error("No subjects found")
        sys.exit(1)

    overrides_file = getattr(args, "overrides_file", None)
    logger.info(f"Found {len(subjects)} subjects for blur")
    for name, videos in sorted(subjects.items()):
        logger.info(f"Blurring {name} ({len(videos)} videos)")
        run_blur_subject(name, videos, args.output_dir,
                         overrides_file=overrides_file)

    print("BLUR_COMPLETE", flush=True)


# ── Pipeline (detached mode with status.json) ───────────────────────────

def _mp_complete(subject_name: str, output_dir: str) -> bool:
    """Check if MediaPipe output already exists for a subject."""
    npz = os.path.join(output_dir, subject_name, "mediapipe_prelabels.npz")
    return os.path.isfile(npz)


def _blur_complete(subject_name: str, videos: list[str], output_dir: str) -> bool:
    """Check if all blur outputs already exist for a subject."""
    out_subj = os.path.join(output_dir, subject_name, "deidentified")
    if not os.path.isdir(out_subj):
        return False
    for vpath in videos:
        out_path = os.path.join(out_subj, Path(vpath).name)
        if not os.path.isfile(out_path):
            return False
    return True


def cmd_pipeline(args):
    """Pipeline subcommand: run MP + blur as a detached process with status.json."""
    status_file = args.status_file
    overrides_file = getattr(args, "overrides_file", None)
    do_mp = "mp" in args.steps
    do_blur = "blur" in args.steps

    # Progress ranges
    if do_mp and do_blur:
        mp_range = (0.0, 50.0)
        blur_range = (50.0, 100.0)
    elif do_mp:
        mp_range = (0.0, 100.0)
        blur_range = None
    else:
        mp_range = None
        blur_range = (0.0, 100.0)

    _write_status(status_file, "starting", "running", 0.0)
    failed_subjects = []

    if do_mp:
        mp_filter = args.mp_subjects if args.mp_subjects is not None else args.subjects
        mp_subjects = discover_subjects(args.video_dir, mp_filter)
        if mp_subjects:
            # Skip already-completed subjects
            skipped = [n for n in mp_subjects if _mp_complete(n, args.output_dir)]
            if skipped:
                logger.info(f"MediaPipe: skipping {len(skipped)} already-completed: "
                            f"{', '.join(sorted(skipped))}")
                mp_subjects = {k: v for k, v in mp_subjects.items() if k not in skipped}

            pct_start, pct_end = mp_range
            total_subjects = len(mp_subjects)
            if total_subjects == 0:
                logger.info("MediaPipe: all subjects already completed")
            else:
                logger.info(f"=== MediaPipe: {total_subjects} subjects ===")
            sys.stdout.flush()
            sys.stderr.flush()
            _write_status(status_file, "mp", "running", pct_start)

            for si, (name, videos) in enumerate(sorted(mp_subjects.items())):
                subj_pct_start = pct_start + (si / total_subjects) * (pct_end - pct_start)
                subj_pct_end = pct_start + ((si + 1) / total_subjects) * (pct_end - pct_start)
                logger.info(f"MediaPipe: {name} ({len(videos)} videos)")
                sys.stdout.flush()
                sys.stderr.flush()
                _write_status(status_file, "mp", "running", subj_pct_start,
                              current_subject=name)
                try:
                    run_mp_subject(name, videos, args.output_dir)
                except Exception as e:
                    logger.error(f"MediaPipe failed for {name}: {e}\n{traceback.format_exc()}")
                    failed_subjects.append(("mp", name, str(e)))
                _write_status(status_file, "mp", "running", subj_pct_end,
                              current_subject=name)
        else:
            logger.info("No subjects for MediaPipe")

    if do_blur:
        blur_filter = args.blur_subjects if args.blur_subjects is not None else args.subjects
        blur_subjects = discover_subjects(args.video_dir, blur_filter)
        if blur_subjects:
            # Skip already-completed subjects
            skipped = [n for n in blur_subjects
                       if _blur_complete(n, blur_subjects[n], args.output_dir)]
            if skipped:
                logger.info(f"Blur: skipping {len(skipped)} already-completed: "
                            f"{', '.join(sorted(skipped))}")
                blur_subjects = {k: v for k, v in blur_subjects.items() if k not in skipped}

            pct_start, pct_end = blur_range
            total_subjects = len(blur_subjects)
            if total_subjects == 0:
                logger.info("Blur: all subjects already completed")
            else:
                logger.info(f"=== Blur: {total_subjects} subjects ===")
            sys.stdout.flush()
            sys.stderr.flush()
            _write_status(status_file, "blur", "running", pct_start)

            for si, (name, videos) in enumerate(sorted(blur_subjects.items())):
                subj_pct_start = pct_start + (si / total_subjects) * (pct_end - pct_start)
                subj_pct_end = pct_start + ((si + 1) / total_subjects) * (pct_end - pct_start)
                logger.info(f"Blur: {name} ({len(videos)} videos)")
                sys.stdout.flush()
                sys.stderr.flush()
                _write_status(status_file, "blur", "running", subj_pct_start,
                              current_subject=name)
                try:
                    run_blur_subject(name, videos, args.output_dir,
                                     overrides_file=overrides_file)
                except Exception as e:
                    logger.error(f"Blur failed for {name}: {e}\n{traceback.format_exc()}")
                    failed_subjects.append(("blur", name, str(e)))
                _write_status(status_file, "blur", "running", subj_pct_end,
                              current_subject=name)
        else:
            logger.info("No subjects for blur")

    if failed_subjects:
        summary = "; ".join(f"{step}:{subj}" for step, subj, _ in failed_subjects)
        logger.warning(f"=== Completed with {len(failed_subjects)} failures: {summary} ===")
        _write_status(status_file, "done", "completed", 100.0,
                      error=f"{len(failed_subjects)} subjects failed: {summary}")
    else:
        logger.info("=== All phases complete ===")
        _write_status(status_file, "done", "completed", 100.0)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Remote preprocessing for DLC labeler")
    sub = parser.add_subparsers(dest="command", required=True)

    mp_p = sub.add_parser("mp", help="Run MediaPipe hand extraction")
    mp_p.add_argument("video_dir", help="Directory containing subject videos")
    mp_p.add_argument("output_dir", help="Output directory for results")
    mp_p.add_argument("--subjects", nargs="*", help="Process only these subjects")

    blur_p = sub.add_parser("blur", help="Run face de-identification")
    blur_p.add_argument("video_dir", help="Directory containing subject videos")
    blur_p.add_argument("output_dir", help="Output directory for results")
    blur_p.add_argument("--subjects", nargs="*", help="Process only these subjects")
    blur_p.add_argument("--overrides-file", default=None,
                        help="Path to face_blur_overrides.json")

    pipe_p = sub.add_parser("pipeline", help="Run MP + blur as detached process")
    pipe_p.add_argument("video_dir", help="Directory containing subject videos")
    pipe_p.add_argument("output_dir", help="Output directory for results")
    pipe_p.add_argument("--steps", nargs="+", required=True,
                        choices=["mp", "blur"], help="Steps to run")
    pipe_p.add_argument("--subjects", nargs="*", help="Process only these subjects (all steps)")
    pipe_p.add_argument("--mp-subjects", nargs="*", default=None,
                        help="Override: subjects for MediaPipe step only")
    pipe_p.add_argument("--blur-subjects", nargs="*", default=None,
                        help="Override: subjects for blur step only")
    pipe_p.add_argument("--status-file", required=True, help="Path for status.json")
    pipe_p.add_argument("--overrides-file", default=None,
                        help="Path to face_blur_overrides.json")
    pipe_p.add_argument("--log-file", default=None,
                        help="Redirect stdout/stderr to this file")

    args = parser.parse_args()

    if args.command == "mp":
        cmd_mp(args)
    elif args.command == "blur":
        cmd_blur(args)
    elif args.command == "pipeline":
        try:
            cmd_pipeline(args)
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Pipeline failed: {e}\n{tb}")
            try:
                _write_status(args.status_file, "failed", "failed",
                              error=str(e))
            except Exception:
                pass
            sys.exit(1)


if __name__ == "__main__":
    # Redirect stdout/stderr to --log-file BEFORE any heavy imports,
    # so even crash-on-import errors are captured.
    for _i, _arg in enumerate(sys.argv):
        if _arg == "--log-file" and _i + 1 < len(sys.argv):
            _log_fh = open(sys.argv[_i + 1], "a")
            sys.stdout = _log_fh
            sys.stderr = _log_fh
            # Re-point logging handlers to the new stderr (they captured
            # the original stderr at module load time)
            for _h in logging.root.handlers:
                if isinstance(_h, logging.StreamHandler):
                    _h.stream = sys.stderr
            break
    main()
