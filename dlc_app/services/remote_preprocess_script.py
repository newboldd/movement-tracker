#!/usr/bin/env python
"""Remote preprocessing script for DLC labeler.

Self-contained (no dlc_app imports). Uploaded to remote host and executed via SSH.

Subcommands:
    mp    Run MediaPipe hand landmark extraction on stereo videos
    blur  Run face de-identification (blur) on stereo videos

Progress output: PROGRESS:<subject>:<pct> to stdout (parsed by orchestrator).

Usage:
    python remote_preprocess_script.py mp <video_dir> <output_dir> [--subject SUBJ]
    python remote_preprocess_script.py blur <video_dir> <output_dir> [--subject SUBJ]
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np

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
TEMPORAL_SIGMA_FRAMES = 3.0
FEATHER_KERNEL = 15
IOU_MATCH_THRESHOLD = 0.1


# ── Subject discovery ────────────────────────────────────────────────────

def discover_subjects(video_dir: str, subject_filter: str = None) -> dict[str, list[str]]:
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
        subjects = {k: v for k, v in subjects.items() if k == subject_filter}

    return dict(subjects)


def emit_progress(subject: str, pct: float):
    """Print progress line for orchestrator parsing."""
    print(f"PROGRESS:{subject}:{pct:.1f}", flush=True)


# ── MediaPipe extraction ────────────────────────────────────────────────

def _extract_hands(results, width, height):
    """Extract detected hands as list of (21,2) arrays + confidence scores."""
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
        ret, frame0 = cap.read()
        if not ret:
            cap.release()
            logger.warning(f"Cannot read video: {vpath}")
            frames_done += n_frames
            continue
        h, full_w = frame0.shape[:2]
        midline = full_w // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

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
    subjects = discover_subjects(args.video_dir, args.subject)
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


def _build_hand_mask(shape, hand_results):
    """Build hand exclusion mask from MediaPipe Hands results."""
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


def deidentify_video(input_path: str, output_path: str,
                     subject_name: str = "",
                     video_pct_base: float = 0,
                     video_pct_span: float = 100):
    """De-identify a single video: blur faces, preserve hands."""
    import mediapipe as mp_lib

    cap = cv2.VideoCapture(input_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, frame0 = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError(f"Cannot read video: {input_path}")

    h, full_w = frame0.shape[:2]
    is_stereo = (full_w / h) > 1.7
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    half_w = full_w // 2 if is_stereo else full_w

    logger.info(
        f"Deidentifying {Path(input_path).name}: "
        f"{n_frames} frames, {'stereo' if is_stereo else 'mono'}, {half_w}x{h}"
    )

    face_det = mp_lib.solutions.face_detection.FaceDetection(
        model_selection=FACE_MODEL_SELECTION,
        min_detection_confidence=FACE_CONF_THRESHOLD,
    )
    hands = mp_lib.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=2,
        min_detection_confidence=0.3, min_tracking_confidence=0.3,
    )

    # ── Pass 1: Detect faces ──
    if is_stereo:
        faces_L_raw, faces_R_raw = [], []
    else:
        faces_raw = []

    for t in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            if is_stereo:
                faces_L_raw.append([])
                faces_R_raw.append([])
            else:
                faces_raw.append([])
            continue

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

        if subject_name and t % 10 == 0:
            pct = video_pct_base + (t / n_frames * 40 / 100) * video_pct_span
            emit_progress(subject_name, pct)

    face_det.close()

    # ── Temporal smoothing ──
    if is_stereo:
        faces_L = _smooth_face_detections(faces_L_raw, n_frames, half_w, h)
        faces_R = _smooth_face_detections(faces_R_raw, n_frames, full_w - half_w, h)
    else:
        faces = _smooth_face_detections(faces_raw, n_frames, full_w, h)

    # ── Pass 2: Render with hand protection ──
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    codec = "avc1" if sys.platform == "darwin" else "mp4v"
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (full_w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open: {output_path} (codec={codec})")

    for t in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        frame = np.ascontiguousarray(frame, dtype=np.uint8)

        if is_stereo:
            left = frame[:, :half_w, :].copy()
            right = frame[:, half_w:, :].copy()

            rgb_l = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
            rgb_r = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)

            hand_res_l = hands.process(rgb_l)
            hand_res_r = hands.process(rgb_r)

            hand_mask_l = _build_hand_mask(left.shape, hand_res_l)
            hand_mask_r = _build_hand_mask(right.shape, hand_res_r)

            left = _apply_face_blur(left, faces_L[t], hand_mask_l)
            right = _apply_face_blur(right, faces_R[t], hand_mask_r)

            out_frame = np.concatenate([left, right], axis=1)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_res = hands.process(rgb)
            hand_mask = _build_hand_mask(frame.shape, hand_res)

            frame = _apply_face_blur(frame, faces[t], hand_mask)
            out_frame = frame

        writer.write(out_frame)

        if subject_name and t % 10 == 0:
            pct = video_pct_base + (40 + t / n_frames * 60) / 100 * video_pct_span
            emit_progress(subject_name, pct)

    cap.release()
    writer.release()
    hands.close()

    logger.info(f"Deidentified: {Path(input_path).name}")


def run_blur_subject(subject_name: str, videos: list[str], output_dir: str):
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
            )
            os.replace(temp_path, out_path)
        except Exception as e:
            logger.error(f"Blur failed for {vpath}: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)

    emit_progress(subject_name, 100)


def cmd_blur(args):
    """Blur subcommand handler."""
    subjects = discover_subjects(args.video_dir, args.subject)
    if not subjects:
        logger.error("No subjects found")
        sys.exit(1)

    logger.info(f"Found {len(subjects)} subjects for blur")
    for name, videos in sorted(subjects.items()):
        logger.info(f"Blurring {name} ({len(videos)} videos)")
        run_blur_subject(name, videos, args.output_dir)

    print("BLUR_COMPLETE", flush=True)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Remote preprocessing for DLC labeler")
    sub = parser.add_subparsers(dest="command", required=True)

    mp_p = sub.add_parser("mp", help="Run MediaPipe hand extraction")
    mp_p.add_argument("video_dir", help="Directory containing subject videos")
    mp_p.add_argument("output_dir", help="Output directory for results")
    mp_p.add_argument("--subject", help="Process only this subject")

    blur_p = sub.add_parser("blur", help="Run face de-identification")
    blur_p.add_argument("video_dir", help="Directory containing subject videos")
    blur_p.add_argument("output_dir", help="Output directory for results")
    blur_p.add_argument("--subject", help="Process only this subject")

    args = parser.parse_args()

    if args.command == "mp":
        cmd_mp(args)
    elif args.command == "blur":
        cmd_blur(args)


if __name__ == "__main__":
    main()
