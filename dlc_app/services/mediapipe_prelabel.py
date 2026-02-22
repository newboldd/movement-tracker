from __future__ import annotations

"""MediaPipe pre-labeling: extract thumb/index predictions + 3D distance trace.

Runs MediaPipe on stereo videos, extracts thumb tip (joint 4) and index tip
(joint 8) for each camera half, tracks hand identity via wrist proximity,
and computes 3D triangulated distances using stereo calibration.
"""

import logging
import os
import re

import cv2
import numpy as np

from ..config import get_settings, PROJECT_DIR
from .video import get_subject_videos, get_video_info, build_trial_map
from .calibration import get_calibration_for_subject, triangulate_points

logger = logging.getLogger(__name__)

# MediaPipe joint indices for the keypoints we care about
THUMB_TIP = 4
INDEX_TIP = 8
WRIST = 0


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
    """Parse target hand from video name (e.g. 'MSA01_L1' -> 'Right').

    MediaPipe assumes selfie camera (mirrored):
      video _L -> MediaPipe reports 'Right'
      video _R -> MediaPipe reports 'Left'
    """
    m = re.search(r'_([LR])\d', video_name)
    if m:
        return 'Right' if m.group(1) == 'L' else 'Left'
    return None


def run_mediapipe(subject_name: str, progress_callback=None) -> str:
    """Run MediaPipe on all stereo videos for a subject.

    Extracts thumb tip and index tip positions from both camera halves.
    Tracks hand identity via wrist proximity across frames.
    Computes 3D triangulated thumb-index distances when calibration available.

    Args:
        subject_name: Subject identifier
        progress_callback: callable(pct: float) for progress updates (0-100)

    Returns:
        Path to the saved npz file.
    """
    import mediapipe as mp_lib

    settings = get_settings()
    videos = get_subject_videos(subject_name)
    if not videos:
        raise ValueError(f"No videos found for subject {subject_name}")

    # Build trial map for total frame count
    trials = build_trial_map(subject_name)
    total_frames = trials[-1]["end_frame"] + 1 if trials else 0

    # Allocate arrays for all frames across all trials
    # Shape: (total_frames, 2) for each keypoint per camera
    OS_thumb = np.full((total_frames, 2), np.nan)
    OS_index = np.full((total_frames, 2), np.nan)
    OD_thumb = np.full((total_frames, 2), np.nan)
    OD_index = np.full((total_frames, 2), np.nan)
    confidence_OS = np.full(total_frames, np.nan)
    confidence_OD = np.full(total_frames, np.nan)

    cam_names = settings.camera_names  # ['OS', 'OD']

    frames_processed = 0

    for trial in trials:
        video_path = trial["video_path"]
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        start_frame = trial["start_frame"]
        n_frames = trial["frame_count"]

        target_label = _target_hand_label(video_name)

        cap = cv2.VideoCapture(video_path)
        ret, frame0 = cap.read()
        if not ret:
            cap.release()
            logger.warning(f"Cannot read video: {video_path}")
            frames_processed += n_frames
            continue
        h, full_w = frame0.shape[:2]
        midline = full_w // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Create separate MediaPipe instances per camera half
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
                OS_thumb[global_frame] = hands_L[idx][THUMB_TIP]
                OS_index[global_frame] = hands_L[idx][INDEX_TIP]
                confidence_OS[global_frame] = scores_L[idx]
                prev_wrist_L = hands_L[idx][WRIST].copy()

            if hands_R:
                if prev_wrist_R is not None and len(hands_R) > 1:
                    idx = int(np.argmin([
                        np.linalg.norm(hh[WRIST] - prev_wrist_R) for hh in hands_R
                    ]))
                else:
                    idx = 0
                OD_thumb[global_frame] = hands_R[idx][THUMB_TIP]
                OD_index[global_frame] = hands_R[idx][INDEX_TIP]
                confidence_OD[global_frame] = scores_R[idx]
                prev_wrist_R = hands_R[idx][WRIST].copy()

            frames_processed += 1
            if progress_callback and frames_processed % 50 == 0:
                pct = (frames_processed / total_frames) * 100
                progress_callback(pct)

        cap.release()
        det_L.close()
        det_R.close()

    # Compute 3D distances if calibration is available
    distances = np.full(total_frames, np.nan)
    calib = get_calibration_for_subject(subject_name)
    if calib is not None:
        distances = _compute_distances(
            OS_thumb, OS_index, OD_thumb, OD_index, calib
        )
        valid_dist = np.sum(~np.isnan(distances))
        logger.info(f"Computed 3D distances for {valid_dist}/{total_frames} frames")

    # Save to npz
    dlc_path = settings.dlc_path / subject_name
    dlc_path.mkdir(parents=True, exist_ok=True)
    npz_path = str(dlc_path / "mediapipe_prelabels.npz")

    np.savez(
        npz_path,
        OS_thumb=OS_thumb,
        OS_index=OS_index,
        OD_thumb=OD_thumb,
        OD_index=OD_index,
        confidence_OS=confidence_OS,
        confidence_OD=confidence_OD,
        distances=distances,
        total_frames=total_frames,
    )

    valid_OS = np.sum(~np.isnan(OS_thumb[:, 0]))
    valid_OD = np.sum(~np.isnan(OD_thumb[:, 0]))
    logger.info(
        f"MediaPipe prelabels saved: {npz_path} "
        f"(OS: {valid_OS}/{total_frames}, OD: {valid_OD}/{total_frames})"
    )

    if progress_callback:
        progress_callback(100.0)

    return npz_path


def _compute_distances(OS_thumb, OS_index, OD_thumb, OD_index, calib):
    """Compute 3D triangulated thumb-index distance for each frame."""
    n = len(OS_thumb)
    distances = np.full(n, np.nan)

    for i in range(n):
        # Need all 4 points to triangulate
        if (np.any(np.isnan(OS_thumb[i])) or np.any(np.isnan(OD_thumb[i])) or
                np.any(np.isnan(OS_index[i])) or np.any(np.isnan(OD_index[i]))):
            continue

        pts_L = np.array([OS_thumb[i], OS_index[i]])
        pts_R = np.array([OD_thumb[i], OD_index[i]])
        pts_3d = triangulate_points(pts_L, pts_R, calib)

        if not np.any(np.isnan(pts_3d)):
            distances[i] = float(np.linalg.norm(pts_3d[0] - pts_3d[1]))

    return distances


def load_mediapipe_prelabels(subject_name: str) -> dict | None:
    """Load saved MediaPipe prelabels for a subject.

    Returns dict with keys: OS_thumb, OS_index, OD_thumb, OD_index,
    confidence_OS, confidence_OD, distances, total_frames.
    Returns None if file doesn't exist.
    """
    settings = get_settings()
    npz_path = settings.dlc_path / subject_name / "mediapipe_prelabels.npz"
    if not npz_path.exists():
        return None

    data = np.load(str(npz_path))
    return {
        "OS_thumb": data["OS_thumb"],
        "OS_index": data["OS_index"],
        "OD_thumb": data["OD_thumb"],
        "OD_index": data["OD_index"],
        "confidence_OS": data["confidence_OS"],
        "confidence_OD": data["confidence_OD"],
        "distances": data["distances"],
        "total_frames": int(data["total_frames"]),
    }


def get_mediapipe_for_session(subject_name: str) -> dict | None:
    """Get MediaPipe predictions formatted for the labeler API response.

    Returns dict with per-camera thumb/index arrays as lists (JSON-serializable),
    plus distances array. Returns None if no prelabels exist.
    """
    data = load_mediapipe_prelabels(subject_name)
    if data is None:
        return None

    settings = get_settings()
    cam_names = settings.camera_names

    def _to_list(arr):
        """Convert (N,2) array to list of [x,y] or null."""
        result = []
        for i in range(len(arr)):
            if np.isnan(arr[i, 0]):
                result.append(None)
            else:
                result.append([float(arr[i, 0]), float(arr[i, 1])])
        return result

    def _dist_to_list(arr):
        """Convert distances array to list of float or null."""
        result = []
        for v in arr:
            result.append(None if np.isnan(v) else round(float(v), 2))
        return result

    response = {}
    if len(cam_names) >= 1:
        response[cam_names[0]] = {
            "thumb": _to_list(data["OS_thumb"]),
            "index": _to_list(data["OS_index"]),
        }
    if len(cam_names) >= 2:
        response[cam_names[1]] = {
            "thumb": _to_list(data["OD_thumb"]),
            "index": _to_list(data["OD_index"]),
        }

    response["distances"] = _dist_to_list(data["distances"])
    return response


def recompute_distance_for_frame(subject_name: str, frame_num: int,
                                 manual_labels: dict) -> float | None:
    """Recompute 3D distance for a single frame using manual corrections + MP fallback.

    Args:
        subject_name: Subject identifier
        frame_num: Global frame number
        manual_labels: dict of {side: {bodypart: [x,y]}} for this frame

    Returns:
        Distance in mm, or None if cannot compute.
    """
    calib = get_calibration_for_subject(subject_name)
    if calib is None:
        return None

    mp_data = load_mediapipe_prelabels(subject_name)
    settings = get_settings()
    cam_names = settings.camera_names

    # Get coords for each camera: prefer manual, fall back to MP
    def _get_coords(side, bodypart):
        # Check manual labels first
        side_labels = manual_labels.get(side, {})
        coords = side_labels.get(bodypart)
        if coords and coords[0] is not None:
            return np.array(coords, dtype=np.float64)
        # Fall back to MP
        if mp_data is not None:
            key = f"{side}_{bodypart}"
            mp_arr = mp_data.get(key)
            if mp_arr is not None and not np.isnan(mp_arr[frame_num, 0]):
                return mp_arr[frame_num]
        return None

    thumb_L = _get_coords(cam_names[0], "thumb") if len(cam_names) >= 1 else None
    index_L = _get_coords(cam_names[0], "index") if len(cam_names) >= 1 else None
    thumb_R = _get_coords(cam_names[1], "thumb") if len(cam_names) >= 2 else None
    index_R = _get_coords(cam_names[1], "index") if len(cam_names) >= 2 else None

    if any(v is None for v in [thumb_L, index_L, thumb_R, index_R]):
        return None

    pts_L = np.array([thumb_L, index_L])
    pts_R = np.array([thumb_R, index_R])
    pts_3d = triangulate_points(pts_L, pts_R, calib)

    if np.any(np.isnan(pts_3d)):
        return None

    return round(float(np.linalg.norm(pts_3d[0] - pts_3d[1])), 2)


def compute_optimal_crop(subject_name: str) -> dict:
    """Compute optimal crop regions per camera from MP predictions + manual corrections.

    Returns dict with per-camera crop params: {cam_name: (x1, x2, y1, y2)}.
    """
    mp_data = load_mediapipe_prelabels(subject_name)
    settings = get_settings()
    cam_names = settings.camera_names
    margin = 80

    crops = {}

    for i, cam in enumerate(cam_names):
        if mp_data is None:
            crops[cam] = None
            continue

        prefix = cam_names[0] if i == 0 else cam_names[1]
        thumb_key = f"{prefix}_thumb"
        index_key = f"{prefix}_index"

        thumb = mp_data.get(thumb_key)
        index = mp_data.get(index_key)

        if thumb is None or index is None:
            crops[cam] = None
            continue

        # Collect all valid coords
        all_x = []
        all_y = []
        for arr in [thumb, index]:
            valid = ~np.isnan(arr[:, 0])
            all_x.extend(arr[valid, 0].tolist())
            all_y.extend(arr[valid, 1].tolist())

        if not all_x:
            crops[cam] = None
            continue

        # Get frame dimensions from first trial
        trials = build_trial_map(subject_name)
        if trials:
            info = get_video_info(trials[0]["video_path"])
            cam_w = info.midline
            cam_h = info.height
        else:
            cam_w, cam_h = 960, 1080

        x1 = max(0, int(np.min(all_x)) - margin)
        x2 = min(cam_w, int(np.max(all_x)) + margin)
        y1 = max(0, int(np.min(all_y)) - margin)
        y2 = min(cam_h, int(np.max(all_y)) + margin)

        crops[cam] = (x1, x2, y1, y2)

    return crops
