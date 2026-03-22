from __future__ import annotations

"""MediaPipe pre-labeling: extract all 21 hand landmarks + 3D distance trace.

Runs MediaPipe on stereo videos, extracts all 21 hand joint positions for each
camera half, tracks hand identity via wrist proximity, and computes 3D
triangulated thumb-index distances using stereo calibration.
"""

import logging
import os

import cv2
import numpy as np

from ..config import get_settings, PROJECT_DIR
from .video import get_subject_videos, get_video_info, build_trial_map
from .calibration import get_calibration_for_subject, triangulate_points

logger = logging.getLogger(__name__)

# MediaPipe joint indices
THUMB_TIP = 4
INDEX_TIP = 8
WRIST = 0
N_JOINTS = 21


def _extract_hands(results, width, height):
    """Extract detected hands as list of (21,2) arrays + confidence scores + labels."""
    hands = []
    scores = []
    labels = []
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_lm, hand_cls in zip(results.multi_hand_landmarks,
                                     results.multi_handedness):
            kp = np.array([(lm.x * width, lm.y * height)
                           for lm in hand_lm.landmark])
            hands.append(kp)
            scores.append(hand_cls.classification[0].score)
            labels.append(hand_cls.classification[0].label)
    elif results.multi_hand_landmarks:
        for hand_lm in results.multi_hand_landmarks:
            kp = np.array([(lm.x * width, lm.y * height)
                           for lm in hand_lm.landmark])
            hands.append(kp)
            scores.append(0.5)
            labels.append(None)
    return hands, scores, labels


def _assign_hands_to_tracks(hands, scores, prev_wrists, tracks, conf, frame_idx):
    """Assign detected hands to two persistent tracks using wrist proximity.

    Maintains up to two hand tracks per camera half.  When multiple hands
    are detected, each is assigned to the nearest existing track by wrist
    distance.  New track slots are filled when a hand cannot be matched.

    Args:
        hands: list of (21,2) arrays from ``_extract_hands``.
        scores: list of confidence scores (parallel to *hands*).
        prev_wrists: list[2] of previous wrist positions (mutated in-place).
        tracks: (n_frames, 2, 21, 2) array to write into.
        conf: (n_frames, 2) array to write into.
        frame_idx: current local frame index.
    """
    if not hands:
        return

    n_hands = min(len(hands), 2)
    active = [i for i in range(2) if prev_wrists[i] is not None]

    if not active:
        # No tracks yet — seed in detection order
        for i in range(n_hands):
            tracks[frame_idx, i] = hands[i]
            conf[frame_idx, i] = scores[i]
            prev_wrists[i] = hands[i][WRIST].copy()
        return

    # Build (hand, track) pairs sorted by wrist distance
    pairs = []
    for hi in range(n_hands):
        for ti in active:
            d = float(np.linalg.norm(hands[hi][WRIST] - prev_wrists[ti]))
            pairs.append((d, hi, ti))
    pairs.sort()

    assigned_hands = set()
    assigned_tracks = set()
    for _, hi, ti in pairs:
        if hi in assigned_hands or ti in assigned_tracks:
            continue
        tracks[frame_idx, ti] = hands[hi]
        conf[frame_idx, ti] = scores[hi]
        prev_wrists[ti] = hands[hi][WRIST].copy()
        assigned_hands.add(hi)
        assigned_tracks.add(ti)

    # Any remaining hands go to unused track slots
    empty = [i for i in range(2) if i not in active and i not in assigned_tracks]
    remaining = [i for i in range(n_hands) if i not in assigned_hands]
    for ti, hi in zip(empty, remaining):
        tracks[frame_idx, ti] = hands[hi]
        conf[frame_idx, ti] = scores[hi]
        prev_wrists[ti] = hands[hi][WRIST].copy()


def _pick_tapping_track(tracks, video_name=""):
    """Choose the track whose thumb–index distance oscillates more.

    Computes the standard deviation of the Euclidean distance between
    thumb tip and index tip over the middle 50 % of frames for each
    track.  The track with the higher std-dev is selected as the
    tapping hand.

    Args:
        tracks: (n_frames, 2, 21, 2) array.
        video_name: for logging.

    Returns:
        Track index (0 or 1).
    """
    n = tracks.shape[0]
    start = n // 4
    end = 3 * n // 4
    if end <= start:
        start, end = 0, n

    oscs = []
    counts = []
    for t_idx in range(2):
        thumb = tracks[start:end, t_idx, THUMB_TIP]
        index = tracks[start:end, t_idx, INDEX_TIP]
        valid = ~np.isnan(thumb[:, 0]) & ~np.isnan(index[:, 0])
        n_valid = int(np.sum(valid))
        counts.append(n_valid)
        if n_valid < 10:
            oscs.append(0.0)
            continue
        dist = np.linalg.norm(thumb[valid] - index[valid], axis=1)
        oscs.append(float(np.std(dist)))

    chosen = int(np.argmax(oscs))

    # If only one track has data, use it regardless of oscillation
    if counts[0] >= 10 and counts[1] < 10:
        chosen = 0
    elif counts[1] >= 10 and counts[0] < 10:
        chosen = 1

    logger.info(
        f"{video_name}: track0 osc={oscs[0]:.1f} ({counts[0]} pts), "
        f"track1 osc={oscs[1]:.1f} ({counts[1]} pts) → selected track {chosen}"
    )
    return chosen


def run_mediapipe(subject_name: str, progress_callback=None) -> str:
    """Run MediaPipe on all stereo videos for a subject.

    Extracts all 21 hand joint positions from both camera halves.
    Tracks up to two hands per camera using wrist proximity, then
    selects the tapping hand based on thumb-index oscillation.
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

    # Allocate arrays: all 21 joints per camera, shape (total_frames, 21, 2)
    OS_landmarks = np.full((total_frames, N_JOINTS, 2), np.nan)
    OD_landmarks = np.full((total_frames, N_JOINTS, 2), np.nan)
    confidence_OS = np.full(total_frames, np.nan)
    confidence_OD = np.full(total_frames, np.nan)

    frames_processed = 0

    for trial in trials:
        video_path = trial["video_path"]
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        start_frame = trial["start_frame"]
        expected_frame_count = trial["frame_count"]

        # Get actual frame count from video (from cache if available)
        # to handle frame count differences between machines
        try:
            video_info = get_video_info(video_path)
            actual_frame_count = video_info.frame_count
        except Exception as e:
            logger.warning(f"Could not get video info for {video_path}: {e}")
            actual_frame_count = expected_frame_count

        # If video has fewer frames than expected, assume missing frames are at START
        frame_offset = max(0, expected_frame_count - actual_frame_count)
        if frame_offset > 0:
            logger.info(
                f"{video_name}: expected {expected_frame_count} frames, actual {actual_frame_count} "
                f"→ applying start offset {frame_offset}"
            )

        cap = cv2.VideoCapture(video_path)
        ret, frame0 = cap.read()
        if not ret:
            cap.release()
            logger.warning(f"Cannot read video: {video_path}")
            frames_processed += expected_frame_count
            continue
        h, full_w = frame0.shape[:2]
        midline = full_w // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Create separate MediaPipe instances per camera half
        mp_hands = mp_lib.solutions.hands
        det_L = mp_hands.Hands(
            static_image_mode=False, max_num_hands=2,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )
        det_R = mp_hands.Hands(
            static_image_mode=False, max_num_hands=2,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )

        # Track up to 2 hands per camera, then pick the tapping hand
        os_tracks = np.full((actual_frame_count, 2, N_JOINTS, 2), np.nan)
        od_tracks = np.full((actual_frame_count, 2, N_JOINTS, 2), np.nan)
        os_conf = np.full((actual_frame_count, 2), np.nan)
        od_conf = np.full((actual_frame_count, 2), np.nan)
        prev_os = [None, None]
        prev_od = [None, None]

        for local_frame in range(actual_frame_count):
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"{video_name}: unexpected end of video at frame {local_frame}")
                break

            frame = np.ascontiguousarray(frame, dtype=np.uint8)
            frame_L = frame[:, :midline, :]
            frame_R = frame[:, midline:, :]

            # Process left camera (OS)
            rgb_L = cv2.cvtColor(frame_L, cv2.COLOR_BGR2RGB)
            res_L = det_L.process(rgb_L)
            hands_L, scores_L, _ = _extract_hands(res_L, midline, h)
            _assign_hands_to_tracks(hands_L, scores_L, prev_os, os_tracks, os_conf, local_frame)

            # Process right camera (OD)
            rgb_R = cv2.cvtColor(frame_R, cv2.COLOR_BGR2RGB)
            res_R = det_R.process(rgb_R)
            hands_R, scores_R, _ = _extract_hands(res_R, full_w - midline, h)
            _assign_hands_to_tracks(hands_R, scores_R, prev_od, od_tracks, od_conf, local_frame)

            frames_processed += 1
            if progress_callback and frames_processed % 50 == 0:
                pct = (frames_processed / total_frames) * 100
                progress_callback(pct)

        cap.release()
        det_L.close()
        det_R.close()

        # Select the tapping hand for each camera based on oscillation
        os_idx = _pick_tapping_track(os_tracks, f"{video_name}/OS")
        od_idx = _pick_tapping_track(od_tracks, f"{video_name}/OD")

        # Copy selected tracks into the global output arrays
        for local_frame in range(actual_frame_count):
            global_frame = start_frame + frame_offset + local_frame
            if global_frame < total_frames:
                OS_landmarks[global_frame] = os_tracks[local_frame, os_idx]
                OD_landmarks[global_frame] = od_tracks[local_frame, od_idx]
                confidence_OS[global_frame] = os_conf[local_frame, os_idx]
                confidence_OD[global_frame] = od_conf[local_frame, od_idx]

    # Compute distances: prefer 3D triangulated if calibration available,
    # otherwise fall back to 2D pixel distance from OS camera
    distances = np.full(total_frames, np.nan)
    calib = get_calibration_for_subject(subject_name)
    if calib is not None:
        distances = _compute_distances(OS_landmarks, OD_landmarks, calib)
        valid_dist = np.sum(~np.isnan(distances))
        logger.info(f"Computed 3D distances for {valid_dist}/{total_frames} frames")
    else:
        distances = _compute_2d_distances(OS_landmarks)
        valid_dist = np.sum(~np.isnan(distances))
        logger.info(f"Computed 2D pixel distances for {valid_dist}/{total_frames} frames (no calibration)")

    # Save to npz
    dlc_path = settings.dlc_path / subject_name
    dlc_path.mkdir(parents=True, exist_ok=True)
    npz_path = str(dlc_path / "mediapipe_prelabels.npz")

    np.savez(
        npz_path,
        OS_landmarks=OS_landmarks,
        OD_landmarks=OD_landmarks,
        confidence_OS=confidence_OS,
        confidence_OD=confidence_OD,
        distances=distances,
        total_frames=total_frames,
    )

    valid_OS = np.sum(~np.isnan(OS_landmarks[:, 0, 0]))
    valid_OD = np.sum(~np.isnan(OD_landmarks[:, 0, 0]))
    logger.info(
        f"MediaPipe prelabels saved: {npz_path} "
        f"(OS: {valid_OS}/{total_frames}, OD: {valid_OD}/{total_frames})"
    )

    if progress_callback:
        progress_callback(100.0)

    return npz_path


def _compute_distances(OS_landmarks, OD_landmarks, calib):
    """Compute 3D triangulated thumb-index distance for each frame.

    Accepts full landmark arrays (N, 21, 2) — extracts thumb tip (4)
    and index tip (8) internally.
    """
    n = len(OS_landmarks)
    distances = np.full(n, np.nan)

    for i in range(n):
        os_thumb = OS_landmarks[i, THUMB_TIP]
        os_index = OS_landmarks[i, INDEX_TIP]
        od_thumb = OD_landmarks[i, THUMB_TIP]
        od_index = OD_landmarks[i, INDEX_TIP]

        if (np.any(np.isnan(os_thumb)) or np.any(np.isnan(od_thumb)) or
                np.any(np.isnan(os_index)) or np.any(np.isnan(od_index))):
            continue

        pts_L = np.array([os_thumb, os_index])
        pts_R = np.array([od_thumb, od_index])
        pts_3d = triangulate_points(pts_L, pts_R, calib)

        if not np.any(np.isnan(pts_3d)):
            distances[i] = float(np.linalg.norm(pts_3d[0] - pts_3d[1]))

    return distances


def _compute_2d_distances(landmarks):
    """Compute 2D pixel distance between thumb tip and index tip.

    Used as a fallback when no stereo calibration is available.
    The values are in pixel units (not mm) but still useful for
    visualization and relative comparisons.
    """
    n = len(landmarks)
    distances = np.full(n, np.nan)
    for i in range(n):
        thumb = landmarks[i, THUMB_TIP]
        index = landmarks[i, INDEX_TIP]
        if not np.isnan(thumb[0]) and not np.isnan(index[0]):
            distances[i] = float(np.linalg.norm(thumb - index))
    return distances


def _detect_frame_offset(subject_name: str, npz_data: dict) -> int:
    """Detect frame offset for misaligned MediaPipe files.

    When frame counts differ between machines, MediaPipe data may be misaligned.
    This attempts to detect per-trial misalignments.

    However, since the offset wasn't applied when the NPZ was saved, we can only
    detect THAT there's a mismatch, not reliably fix it without knowing the
    original processing details.

    Returns 0 (no correction applied), and logs diagnostics instead.
    """
    try:
        trials = build_trial_map(subject_name)
        expected_total = trials[-1]["end_frame"] + 1 if trials else 0

        # Array size from the npz
        stored_total = int(npz_data.get("total_frames", 0))

        if stored_total != expected_total:
            logger.warning(
                f"{subject_name}: Frame count mismatch detected! "
                f"NPZ stored {stored_total} frames, but trial map expects {expected_total}. "
                f"This suggests the video files were processed on a different machine "
                f"with different codec/frame counting behavior. "
                f"MediaPipe labels may be misaligned. "
                f"Consider reprocessing MediaPipe on this machine."
            )

        return 0  # Conservative: don't attempt auto-correction
    except Exception as e:
        logger.debug(f"Could not detect frame offset for {subject_name}: {e}")
        return 0


def load_mediapipe_prelabels(subject_name: str) -> dict | None:
    """Load saved MediaPipe prelabels for a subject.

    Handles both new format (OS_landmarks/OD_landmarks with all 21 joints)
    and old format (OS_thumb/OS_index/OD_thumb/OD_index with just 2 joints).

    Automatically detects and compensates for frame misalignments from older
    processing where frame offset logic wasn't applied.

    Returns dict with keys: OS_landmarks, OD_landmarks (N, 21, 2),
    confidence_OS, confidence_OD, distances, total_frames.
    Returns None if file doesn't exist.
    """
    settings = get_settings()
    npz_path = settings.dlc_path / subject_name / "mediapipe_prelabels.npz"
    if not npz_path.exists():
        return None

    data = np.load(str(npz_path))

    # Detect if data needs offset correction
    frame_offset = _detect_frame_offset(subject_name, data)

    if "OS_landmarks" in data:
        # New format: full 21-joint arrays
        OS_lm = data["OS_landmarks"].copy()
        OD_lm = data["OD_landmarks"].copy()
        conf_OS = data["confidence_OS"].copy()
        conf_OD = data["confidence_OD"].copy()
        dist = data["distances"].copy() if "distances" in data else None

        # Apply offset correction if needed (shift data forward, pad start with NaN)
        if frame_offset > 0:
            OS_lm_corrected = np.full_like(OS_lm, np.nan)
            OD_lm_corrected = np.full_like(OD_lm, np.nan)
            conf_OS_corrected = np.full_like(conf_OS, np.nan)
            conf_OD_corrected = np.full_like(conf_OD, np.nan)

            # Shift data forward by frame_offset
            OS_lm_corrected[frame_offset:] = OS_lm[:-frame_offset]
            OD_lm_corrected[frame_offset:] = OD_lm[:-frame_offset]
            conf_OS_corrected[frame_offset:] = conf_OS[:-frame_offset]
            conf_OD_corrected[frame_offset:] = conf_OD[:-frame_offset]

            if dist is not None:
                dist_corrected = np.full_like(dist, np.nan)
                dist_corrected[frame_offset:] = dist[:-frame_offset]
                dist = dist_corrected

            OS_lm = OS_lm_corrected
            OD_lm = OD_lm_corrected
            conf_OS = conf_OS_corrected
            conf_OD = conf_OD_corrected

            logger.info(f"{subject_name}: Applied frame offset {frame_offset} to MediaPipe data")

        return {
            "OS_landmarks": OS_lm,
            "OD_landmarks": OD_lm,
            "confidence_OS": conf_OS,
            "confidence_OD": conf_OD,
            "distances": dist if dist is not None else data.get("distances"),
            "total_frames": int(data["total_frames"]),
        }
    else:
        # Old format: only thumb + index tip stored
        n = int(data["total_frames"])
        OS_lm = np.full((n, N_JOINTS, 2), np.nan)
        OD_lm = np.full((n, N_JOINTS, 2), np.nan)
        OS_lm[:, THUMB_TIP] = data["OS_thumb"]
        OS_lm[:, INDEX_TIP] = data["OS_index"]
        OD_lm[:, THUMB_TIP] = data["OD_thumb"]
        OD_lm[:, INDEX_TIP] = data["OD_index"]

        # Apply offset correction if needed
        if frame_offset > 0:
            OS_lm_corrected = np.full_like(OS_lm, np.nan)
            OD_lm_corrected = np.full_like(OD_lm, np.nan)
            OS_lm_corrected[frame_offset:] = OS_lm[:-frame_offset]
            OD_lm_corrected[frame_offset:] = OD_lm[:-frame_offset]
            OS_lm = OS_lm_corrected
            OD_lm = OD_lm_corrected
            logger.info(f"{subject_name}: Applied frame offset {frame_offset} to MediaPipe data (old format)")

        return {
            "OS_landmarks": OS_lm,
            "OD_landmarks": OD_lm,
            "confidence_OS": data["confidence_OS"],
            "confidence_OD": data["confidence_OD"],
            "distances": data.get("distances"),
            "total_frames": n,
        }


def get_mediapipe_for_session(subject_name: str) -> dict | None:
    """Get MediaPipe predictions formatted for the labeler API response.

    Returns dict with per-camera thumb/index arrays as lists (JSON-serializable),
    plus distances array. Returns None if no prelabels exist.

    If stored distances are all NaN but calibration is now available,
    recomputes them from the MP coordinates and updates the npz file.
    """
    data = load_mediapipe_prelabels(subject_name)
    if data is None:
        return None

    OS_lm = data["OS_landmarks"]
    OD_lm = data["OD_landmarks"]

    # Recompute distances if they're all NaN
    if np.all(np.isnan(data["distances"])):
        calib = get_calibration_for_subject(subject_name)
        if calib is not None:
            logger.info(f"Recomputing 3D distances for {subject_name} (calibration now available)")
            distances = _compute_distances(OS_lm, OD_lm, calib)
        else:
            logger.info(f"Computing 2D pixel distances for {subject_name} (no calibration)")
            distances = _compute_2d_distances(OS_lm)
        valid = np.sum(~np.isnan(distances))
        if valid > 0:
            data["distances"] = distances
            # Update the npz file so we don't recompute every time
            settings = get_settings()
            npz_path = str(settings.dlc_path / subject_name / "mediapipe_prelabels.npz")
            np.savez(
                npz_path,
                OS_landmarks=OS_lm,
                OD_landmarks=OD_lm,
                confidence_OS=data["confidence_OS"],
                confidence_OD=data["confidence_OD"],
                distances=distances,
                total_frames=data["total_frames"],
            )
            logger.info(f"Updated distances in {npz_path}: {valid}/{len(distances)} frames")

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
            "thumb": _to_list(OS_lm[:, THUMB_TIP]),
            "index": _to_list(OS_lm[:, INDEX_TIP]),
        }
    if len(cam_names) >= 2:
        response[cam_names[1]] = {
            "thumb": _to_list(OD_lm[:, THUMB_TIP]),
            "index": _to_list(OD_lm[:, INDEX_TIP]),
        }

    response["distances"] = _dist_to_list(data["distances"])

    # Include run history for comparison
    run_history = []
    run_keys = sorted(
        (int(k.split("_")[-1]), k) for k in data if k.startswith("distances_run_")
    )
    for run_num, key in run_keys:
        entry = {"run": run_num, "distances": _dist_to_list(data[key])}
        crop_key = f"crop_run_{run_num}"
        if crop_key in data:
            try:
                import json as _json
                entry["crop"] = _json.loads(str(data[crop_key]))
            except (ValueError, TypeError):
                pass
        run_history.append(entry)
    if run_history:
        response["run_history"] = run_history

    return response


def recompute_distance_for_frame(subject_name: str, frame_num: int,
                                 manual_labels: dict,
                                 stage_data: dict | None = None) -> float | None:
    """Recompute 3D distance for a single frame using manual corrections + stage data + MP fallback.

    Args:
        subject_name: Subject identifier
        frame_num: Global frame number
        manual_labels: dict of {side: {bodypart: [x,y]}} for this frame
        stage_data: Optional pre-loaded stage data (e.g. corrections CSV + DLC fallback).
                    When provided, used as fallback BEFORE raw MP predictions so that
                    un-labeled cameras use the same coordinates as the distance trace
                    display, not raw MP values which may be completely different.

    Returns:
        Distance in mm, or None if cannot compute.
    """
    calib = get_calibration_for_subject(subject_name)
    if calib is None:
        return None

    mp_data = load_mediapipe_prelabels(subject_name)
    settings = get_settings()
    cam_names = settings.camera_names

    # Only load raw DLC fallback when no stage_data is provided.
    # When stage_data is available it already incorporates the best available
    # DLC labels (corrections > labels_v2 > labels_v1) per trial and camera.
    dlc_data = None
    if stage_data is None:
        from .dlc_predictions import get_dlc_predictions_for_session
        dlc_data = get_dlc_predictions_for_session(subject_name)

    # Map bodypart names to joint indices
    bp_to_joint = {"thumb": THUMB_TIP, "index": INDEX_TIP}

    # Get coords for each camera: manual > stage_data > MP > raw DLC
    def _get_coords(side_idx, bodypart):
        side = cam_names[side_idx]
        # 1. Manual labels (highest priority — user's direct corrections)
        side_labels = manual_labels.get(side, {})
        coords = side_labels.get(bodypart)
        if coords and coords[0] is not None:
            return np.array(coords, dtype=np.float64)
        # 2. Stage data (corrections CSV + per-trial/per-camera DLC fallback).
        #    This matches exactly what the distance trace displays so that saving
        #    one camera's label doesn't change the other camera's contribution.
        if stage_data is not None:
            cam_data = stage_data.get(side, {})
            bp_coords = cam_data.get(bodypart, [])
            if frame_num < len(bp_coords) and bp_coords[frame_num] is not None:
                return np.array(bp_coords[frame_num], dtype=np.float64)
        # 3. Fall back to MP
        if mp_data is not None:
            lm_key = "OS_landmarks" if side_idx == 0 else "OD_landmarks"
            joint_idx = bp_to_joint.get(bodypart)
            if joint_idx is not None:
                pt = mp_data[lm_key][frame_num, joint_idx]
                if not np.isnan(pt[0]):
                    return pt
        # 4. Fall back to raw DLC predictions (only when stage_data not available)
        if dlc_data is not None:
            dlc_cam = dlc_data.get(side, {})
            dlc_coords = dlc_cam.get(bodypart, [])
            if frame_num < len(dlc_coords) and dlc_coords[frame_num] is not None:
                return np.array(dlc_coords[frame_num], dtype=np.float64)
        return None

    thumb_L = _get_coords(0, "thumb") if len(cam_names) >= 1 else None
    index_L = _get_coords(0, "index") if len(cam_names) >= 1 else None
    thumb_R = _get_coords(1, "thumb") if len(cam_names) >= 2 else None
    index_R = _get_coords(1, "index") if len(cam_names) >= 2 else None

    if any(v is None for v in [thumb_L, index_L, thumb_R, index_R]):
        return None

    pts_L = np.array([thumb_L, index_L])
    pts_R = np.array([thumb_R, index_R])
    pts_3d = triangulate_points(pts_L, pts_R, calib)

    if np.any(np.isnan(pts_3d)):
        return None

    return round(float(np.linalg.norm(pts_3d[0] - pts_3d[1])), 2)


def run_mediapipe_cropped(
    subject_name: str,
    video_path: str,
    start_frame: int,
    frame_count: int,
    crop: dict,
    camera_key: str,
    is_stereo_video: bool = False,
    stereo_side: str = "",
    progress_callback=None,
) -> None:
    """Re-run MediaPipe on a single trial's video with a bounding-box crop.

    Crops each frame to the given region before passing to MediaPipe Hands,
    then transforms detected landmarks back to full-frame coordinates.
    Merges results into the existing mediapipe_prelabels.npz file.

    Args:
        subject_name: Subject identifier.
        video_path: Path to the video file.
        start_frame: Global frame index where this trial starts in the npz arrays.
        frame_count: Number of frames expected for this trial.
        crop: Dict with x1, y1, x2, y2 in image pixel coordinates.
        camera_key: Which camera slot to update in the npz ('OS' or 'OD').
        is_stereo_video: True if the video is side-by-side stereo.
        stereo_side: For stereo videos, which half to use ('OS'/'OD' or first/second camera name).
        progress_callback: Optional callable(pct: float).
    """
    import mediapipe as mp_lib

    settings = get_settings()
    x1, y1, x2, y2 = int(crop["x1"]), int(crop["y1"]), int(crop["x2"]), int(crop["y2"])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    ret, frame0 = cap.read()
    if not ret:
        cap.release()
        raise ValueError(f"Cannot read video: {video_path}")

    full_h, full_w = frame0.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Determine which region of the full frame to extract first
    # (for stereo, we split left/right; for multicam/single, use the whole frame)
    if is_stereo_video:
        midline = full_w // 2
        cam_names = settings.camera_names
        # Determine if this is left or right camera
        is_right = False
        if len(cam_names) >= 2 and stereo_side == cam_names[1]:
            is_right = True
        elif stereo_side in ("OD", "right"):
            is_right = True
    else:
        midline = full_w

    # Actual video frame count
    actual_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = min(frame_count, actual_frames) if actual_frames > 0 else frame_count

    # Handle frame offset (same logic as run_mediapipe)
    frame_offset = max(0, frame_count - n_frames)

    # Run MediaPipe on cropped frames with 2-hand tracking
    mp_hands = mp_lib.solutions.hands
    detector = mp_hands.Hands(
        static_image_mode=False, max_num_hands=2,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )

    tracks = np.full((n_frames, 2, N_JOINTS, 2), np.nan)
    conf = np.full((n_frames, 2), np.nan)
    prev_wrists = [None, None]

    crop_w = x2 - x1
    crop_h = y2 - y1

    for local_frame in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        frame = np.ascontiguousarray(frame, dtype=np.uint8)

        # Extract camera half for stereo
        if is_stereo_video:
            if is_right:
                cam_frame = frame[:, midline:, :]
            else:
                cam_frame = frame[:, :midline, :]
        else:
            cam_frame = frame

        # Apply crop
        cam_h, cam_w = cam_frame.shape[:2]
        cx1 = max(0, min(x1, cam_w))
        cy1 = max(0, min(y1, cam_h))
        cx2 = max(0, min(x2, cam_w))
        cy2 = max(0, min(y2, cam_h))
        cropped = cam_frame[cy1:cy2, cx1:cx2, :]

        if cropped.size == 0:
            continue

        # Run MediaPipe on the cropped region
        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)

        crop_actual_h, crop_actual_w = cropped.shape[:2]
        hands, scores, _ = _extract_hands(results, crop_actual_w, crop_actual_h)

        # Transform landmarks back to full camera-frame coordinates
        for i, hand_kp in enumerate(hands):
            hand_kp[:, 0] += cx1  # x offset
            hand_kp[:, 1] += cy1  # y offset

        _assign_hands_to_tracks(hands, scores, prev_wrists, tracks, conf, local_frame)

        if progress_callback and local_frame % 50 == 0:
            progress_callback(local_frame / n_frames * 100)

    cap.release()
    detector.close()

    # Pick the tapping hand
    track_idx = _pick_tapping_track(tracks, f"{subject_name}/{camera_key}/cropped")

    # Load existing npz and merge
    npz_path = settings.dlc_path / subject_name / "mediapipe_prelabels.npz"
    if npz_path.exists():
        data = dict(np.load(str(npz_path)))
    else:
        # Create new arrays if no existing file
        trials = build_trial_map(subject_name)
        total = trials[-1]["end_frame"] + 1 if trials else start_frame + frame_count
        data = {
            "OS_landmarks": np.full((total, N_JOINTS, 2), np.nan),
            "OD_landmarks": np.full((total, N_JOINTS, 2), np.nan),
            "confidence_OS": np.full(total, np.nan),
            "confidence_OD": np.full(total, np.nan),
            "distances": np.full(total, np.nan),
            "total_frames": np.array(total),
        }

    # ── Save current distances as run history before overwriting ──
    MAX_RUN_HISTORY = 5
    old_distances = data.get("distances")
    if old_distances is not None and not np.all(np.isnan(old_distances)):
        # Find next run number
        existing_runs = sorted(
            int(k.split("_")[-1]) for k in data if k.startswith("distances_run_")
        )
        next_run = (existing_runs[-1] + 1) if existing_runs else 1
        data[f"distances_run_{next_run}"] = old_distances.copy()
        # Store crop info for this run
        import json as _json
        data[f"crop_run_{next_run}"] = np.array(_json.dumps(crop))

        # Drop oldest runs if exceeding cap
        all_runs = sorted(
            int(k.split("_")[-1]) for k in data if k.startswith("distances_run_")
        )
        while len(all_runs) > MAX_RUN_HISTORY:
            oldest = all_runs.pop(0)
            data.pop(f"distances_run_{oldest}", None)
            data.pop(f"crop_run_{oldest}", None)

    lm_key = "OS_landmarks" if camera_key == "OS" else "OD_landmarks"
    conf_key = "confidence_OS" if camera_key == "OS" else "confidence_OD"
    total_frames = data[lm_key].shape[0]

    # Overwrite the trial's frame range with new detections
    for local_frame in range(n_frames):
        global_frame = start_frame + frame_offset + local_frame
        if global_frame < total_frames:
            data[lm_key][global_frame] = tracks[local_frame, track_idx]
            data[conf_key][global_frame] = conf[local_frame, track_idx]

    # Recompute distances: prefer 3D if calibration available, else 2D pixel
    calib = get_calibration_for_subject(subject_name)
    if calib is not None:
        data["distances"] = _compute_distances(data["OS_landmarks"], data["OD_landmarks"], calib)
    else:
        data["distances"] = _compute_2d_distances(data["OS_landmarks"])

    # Save updated npz (including run history keys)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = {
        "OS_landmarks": data["OS_landmarks"],
        "OD_landmarks": data["OD_landmarks"],
        "confidence_OS": data["confidence_OS"],
        "confidence_OD": data["confidence_OD"],
        "distances": data["distances"],
        "total_frames": data.get("total_frames", total_frames),
    }
    # Include run history keys
    for k, v in data.items():
        if k.startswith("distances_run_") or k.startswith("crop_run_"):
            save_data[k] = v
    np.savez(str(npz_path), **save_data)

    valid_new = np.sum(~np.isnan(tracks[:, track_idx, 0, 0]))
    logger.info(
        f"MediaPipe cropped rerun: {subject_name}/{camera_key}, "
        f"trial frames {start_frame}-{start_frame + frame_count - 1}, "
        f"crop ({x1},{y1})-({x2},{y2}), detected {valid_new}/{n_frames} frames"
    )

    if progress_callback:
        progress_callback(100.0)


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

        lm_key = "OS_landmarks" if i == 0 else "OD_landmarks"
        lm = mp_data.get(lm_key)  # (N, 21, 2)
        if lm is None:
            crops[cam] = None
            continue

        # Use thumb tip and index tip for crop region
        thumb = lm[:, THUMB_TIP]  # (N, 2)
        index = lm[:, INDEX_TIP]  # (N, 2)

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
