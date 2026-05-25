from __future__ import annotations

"""MediaPipe pre-labeling: extract all 21 hand landmarks + 3D distance trace.

Runs MediaPipe on stereo videos, extracts all 21 hand joint positions for each
camera half, tracks hand identity via wrist proximity, and computes 3D
triangulated thumb-index distances using stereo calibration.
"""

import logging
import os
from pathlib import Path

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


def _pick_tapping_track(tracks, video_name="", trial_name=""):
    """Choose the tapping hand track based on power in the distance trace.

    Power = mean(velocity × amplitude) of thumb-index oscillation.
    The tapping hand has rhythmic open-close movements producing high power;
    the resting hand has low-amplitude drift producing low power.

    Args:
        tracks: (n_frames, 2, 21, 2) array.
        video_name: for logging.
        trial_name: unused (kept for API compatibility).

    Returns:
        Track index (0 or 1).
    """
    n = tracks.shape[0]
    # Use full trace for power calculation (not just middle 50%)
    powers = []
    counts = []
    for t_idx in range(2):
        thumb = tracks[:, t_idx, THUMB_TIP]
        index = tracks[:, t_idx, INDEX_TIP]
        valid = ~np.isnan(thumb[:, 0]) & ~np.isnan(index[:, 0])
        n_valid = int(np.sum(valid))
        counts.append(n_valid)
        if n_valid < 20:
            powers.append(0.0)
            continue

        dist = np.linalg.norm(thumb - index, axis=1)
        # Fill NaN gaps with linear interpolation for velocity calc
        valid_idx = np.where(valid)[0]
        dist_clean = np.interp(np.arange(n), valid_idx, dist[valid_idx])

        # Velocity (frame-to-frame change in distance)
        vel = np.abs(np.diff(dist_clean))

        # Find peaks (local maxima in distance = open position)
        # Simple peak detection: frame where dist > both neighbors
        peaks = []
        troughs = []
        for i in range(1, len(dist_clean) - 1):
            if dist_clean[i] > dist_clean[i-1] and dist_clean[i] > dist_clean[i+1]:
                peaks.append(i)
            elif dist_clean[i] < dist_clean[i-1] and dist_clean[i] < dist_clean[i+1]:
                troughs.append(i)

        if len(peaks) < 3 or len(troughs) < 3:
            # Not enough oscillation — use std as fallback
            powers.append(float(np.std(dist_clean)))
            continue

        # Compute amplitude for each peak (distance from nearest preceding trough)
        amplitudes = []
        for pk in peaks:
            preceding = [t for t in troughs if t < pk]
            if preceding:
                amp = dist_clean[pk] - dist_clean[preceding[-1]]
                if amp > 0:
                    amplitudes.append(amp)

        if not amplitudes:
            powers.append(float(np.std(dist_clean)))
            continue

        # Power = mean amplitude × mean velocity at peaks
        mean_amp = float(np.mean(amplitudes))
        # Peak velocities: velocity at each peak frame
        peak_vels = [vel[min(pk, len(vel)-1)] for pk in peaks]
        mean_peak_vel = float(np.mean(peak_vels))
        power = mean_amp * mean_peak_vel
        powers.append(power)

    chosen = int(np.argmax(powers))

    # If only one track has data, use it
    if counts[0] >= 20 and counts[1] < 20:
        chosen = 0
    elif counts[1] >= 20 and counts[0] < 20:
        chosen = 1

    logger.info(
        f"{video_name}: track0 power={powers[0]:.1f} ({counts[0]} pts), "
        f"track1 power={powers[1]:.1f} ({counts[1]} pts) "
        f"→ selected track {chosen}"
    )
    return chosen


def run_mediapipe(subject_name: str, progress_callback=None,
                  crop_boxes: dict | None = None,
                  static_image_mode: bool = False,
                  trial_idx: int | None = None,
                  reverse: bool = False,
                  use_bbox: bool = True) -> str:
    """Run MediaPipe on all stereo videos for a subject.

    Extracts all 21 hand joint positions from both camera halves.
    Tracks up to two hands per camera using wrist proximity, then
    selects the tapping hand based on thumb-index oscillation.
    Computes 3D triangulated thumb-index distances when calibration available.

    Args:
        subject_name: Subject identifier
        progress_callback: callable(pct: float) for progress updates (0-100)
        crop_boxes: Optional per-trial crop boxes, keyed by trial index.
            Each value is a dict with 'OS' and/or 'OD' keys mapping to
            [x1, y1, x2, y2] in camera-half pixel coordinates.
            MediaPipe is run on the cropped region; landmarks are remapped
            back to full half-frame coordinates automatically.
        static_image_mode: When True, every frame runs the full palm
            detector (no between-frame tracker).  3-5x slower but recovers
            detection on hard poses (back-of-hand views, etc.) where the
            tracker fails to lock on and propagates a NaN gap.
        trial_idx: When set, process ONLY this trial — the other trials'
            slices in the saved npz are preserved from the existing file
            (or left NaN if no prior file exists).  Used by the per-trial
            "Run MediaPipe" button on the skeleton page so adjusting one
            trial's bbox doesn't silently re-process the others.
        reverse: When True, feed frames to MediaPipe in reverse temporal
            order so the tracker enters cold-start frames already locked
            on.  Saved to ``mediapipe_reverse_prelabels.npz`` so the
            forward pass output is not overwritten.  Slower than the
            forward pass because it seeks each frame instead of reading
            sequentially.

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
    # All tracks: (total_frames, 2, 21, 2) — both hands saved for later re-selection
    OS_all_tracks = np.full((total_frames, 2, N_JOINTS, 2), np.nan)
    OD_all_tracks = np.full((total_frames, 2, N_JOINTS, 2), np.nan)
    confidence_OS = np.full(total_frames, np.nan)
    confidence_OD = np.full(total_frames, np.nan)

    # Per-trial output layout means single-trial reruns no longer have
    # to preload the subject-wide arrays from a combined npz -- we
    # only WRITE the requested trial's file, so untouched trials'
    # per-trial files stay on disk untouched.  Just filter the trial
    # list and let OS_landmarks et al. stay NaN for the trials we're
    # not processing.
    if trial_idx is not None:
        if 0 <= trial_idx < len(trials):
            _picked = dict(trials[trial_idx])
            _picked["__orig_idx__"] = trial_idx
            trials = [_picked]
        else:
            raise ValueError(f"trial_idx {trial_idx} out of range for {subject_name}")

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

        # Store landmarks at OpenCV frame indices (no offset).
        # Offset compensation for browser playback happens at display time.
        frame_offset = 0

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
            static_image_mode=static_image_mode, max_num_hands=2,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )
        det_R = mp_hands.Hands(
            static_image_mode=static_image_mode, max_num_hands=2,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )

        # Per-trial crop boxes (may be None if not set).  In single-
        # trial mode the filtered list has length 1, so
        # ``trials.index(trial)`` would always be 0; fall back to the
        # original index stashed in ``__orig_idx__`` so we look up the
        # right crop box from the DB-loaded dict.
        _t_idx = trial.get("__orig_idx__", trials.index(trial))
        trial_crop = crop_boxes.get(_t_idx) if crop_boxes else None

        # Track up to 2 hands per camera, then pick the tapping hand
        os_tracks = np.full((actual_frame_count, 2, N_JOINTS, 2), np.nan)
        od_tracks = np.full((actual_frame_count, 2, N_JOINTS, 2), np.nan)
        os_conf = np.full((actual_frame_count, 2), np.nan)
        od_conf = np.full((actual_frame_count, 2), np.nan)
        prev_os = [None, None]
        prev_od = [None, None]

        right_w = full_w - midline

        # Reverse pass: process frames in descending temporal order so
        # the MediaPipe tracker enters a cold-start frame already locked
        # on from a later well-labeled frame.  Sequential cap.read() only
        # walks forward, so we explicitly seek each frame.  This is
        # slower than the forward sequential read but avoids buffering
        # the whole video in memory.
        if reverse:
            frame_indices = range(actual_frame_count - 1, -1, -1)
        else:
            frame_indices = range(actual_frame_count)

        for local_frame in frame_indices:
            if reverse:
                cap.set(cv2.CAP_PROP_POS_FRAMES, local_frame)
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"{video_name}: unexpected read fail at frame {local_frame}")
                if reverse:
                    continue  # try next frame, don't abort the whole trial
                break

            frame = np.ascontiguousarray(frame, dtype=np.uint8)
            frame_L = frame[:, :midline, :]
            frame_R = frame[:, midline:, :]

            # Process left camera (OS) — apply crop if set
            crop_L = trial_crop.get("OS") if trial_crop else None
            if crop_L:
                cx1 = max(0, int(crop_L[0])); cy1 = max(0, int(crop_L[1]))
                cx2 = min(midline, int(crop_L[2])); cy2 = min(h, int(crop_L[3]))
                cropped_L = frame_L[cy1:cy2, cx1:cx2, :]
                if cropped_L.size > 0:
                    cw = cx2 - cx1; ch_c = cy2 - cy1
                    rgb_L = cv2.cvtColor(cropped_L, cv2.COLOR_BGR2RGB)
                    res_L = det_L.process(rgb_L)
                    hands_L, scores_L, _ = _extract_hands(res_L, cw, ch_c)
                    # Offset landmarks back to full half-frame coordinates
                    for hand in hands_L:
                        hand[:, 0] += cx1
                        hand[:, 1] += cy1
                else:
                    hands_L, scores_L = [], []
            else:
                rgb_L = cv2.cvtColor(frame_L, cv2.COLOR_BGR2RGB)
                res_L = det_L.process(rgb_L)
                hands_L, scores_L, _ = _extract_hands(res_L, midline, h)
            _assign_hands_to_tracks(hands_L, scores_L, prev_os, os_tracks, os_conf, local_frame)

            # Process right camera (OD) — apply crop if set
            crop_R = trial_crop.get("OD") if trial_crop else None
            if crop_R:
                cx1 = max(0, int(crop_R[0])); cy1 = max(0, int(crop_R[1]))
                cx2 = min(right_w, int(crop_R[2])); cy2 = min(h, int(crop_R[3]))
                cropped_R = frame_R[cy1:cy2, cx1:cx2, :]
                if cropped_R.size > 0:
                    cw = cx2 - cx1; ch_c = cy2 - cy1
                    rgb_R = cv2.cvtColor(cropped_R, cv2.COLOR_BGR2RGB)
                    res_R = det_R.process(rgb_R)
                    hands_R, scores_R, _ = _extract_hands(res_R, cw, ch_c)
                    # Offset landmarks back to full half-frame coordinates
                    for hand in hands_R:
                        hand[:, 0] += cx1
                        hand[:, 1] += cy1
                else:
                    hands_R, scores_R = [], []
            else:
                rgb_R = cv2.cvtColor(frame_R, cv2.COLOR_BGR2RGB)
                res_R = det_R.process(rgb_R)
                hands_R, scores_R, _ = _extract_hands(res_R, right_w, h)
            _assign_hands_to_tracks(hands_R, scores_R, prev_od, od_tracks, od_conf, local_frame)

            frames_processed += 1
            if progress_callback and frames_processed % 50 == 0:
                pct = (frames_processed / total_frames) * 100
                progress_callback(pct)

        cap.release()
        det_L.close()
        det_R.close()

        # Select the tapping hand for each camera based on oscillation
        trial_name = trial.get("trial_name", video_name)
        os_idx = _pick_tapping_track(os_tracks, f"{video_name}/OS", trial_name=trial_name)
        od_idx = _pick_tapping_track(od_tracks, f"{video_name}/OD", trial_name=trial_name)

        # Copy selected tracks + all tracks into the global output arrays
        for local_frame in range(actual_frame_count):
            global_frame = start_frame + frame_offset + local_frame
            if global_frame < total_frames:
                OS_landmarks[global_frame] = os_tracks[local_frame, os_idx]
                OD_landmarks[global_frame] = od_tracks[local_frame, od_idx]
                OS_all_tracks[global_frame] = os_tracks[local_frame]
                OD_all_tracks[global_frame] = od_tracks[local_frame]
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

    # Per-trial output: one ``mediapipe.npz`` (or ``mediapipe_reverse.npz``
    # for the reverse pass) under ``<dlc>/<subject>/<trial_stem>/`` per
    # trial.  The old combined ``mediapipe_prelabels.npz`` was confusing
    # and made single-trial reruns expensive (the whole file had to be
    # rewritten).  In single-trial mode we only write the requested
    # trial's file; the others stay as-is.
    dlc_path = settings.dlc_path / subject_name
    dlc_path.mkdir(parents=True, exist_ok=True)
    # Per-trial output filename routes by pass.  Reverse and Static
    # always write to their own files regardless of bbox.  Forward
    # splits into two files: the bbox-free baseline (mediapipe.npz)
    # and the bbox-cropped variant (mediapipe_cropped.npz) so the
    # Labels page can compare them side-by-side -- since the user
    # observed that bbox choice helps some frames and hurts others,
    # keeping both is more useful than overwriting one with the other.
    # Per-trial routing for Forward is decided in the loop below,
    # because crop_boxes may have an entry for some trials and not
    # others.
    # Reconstruct the FULL trial list (the run_mediapipe loop above
    # may have filtered to a single trial); we need the start/end for
    # every trial to slice the subject-wide arrays correctly.
    all_trials = build_trial_map(subject_name)
    written_paths: list[str] = []
    target_trial_idxs = {trial_idx} if trial_idx is not None \
        else {i for i in range(len(all_trials))}
    for ti, _trial in enumerate(all_trials):
        if ti not in target_trial_idxs:
            continue
        _sf = int(_trial["start_frame"])
        _ef = _sf + int(_trial["frame_count"]) - 1
        if _ef < _sf or _sf >= total_frames:
            continue
        _ef = min(_ef, total_frames - 1)
        _stem = _trial["trial_name"]
        _trial_dir = dlc_path / _stem
        _trial_dir.mkdir(parents=True, exist_ok=True)
        # Per-trial output filename:
        #   reverse=True                            → mediapipe_reverse.npz
        #   static_image_mode=True                  → mediapipe_static.npz
        #   forward + a crop actually applied here  → mediapipe_cropped.npz
        #   forward, no crop on this trial          → mediapipe.npz
        _trial_crop_lookup = (crop_boxes or {}).get(ti) or {}
        _used_bbox_here = bool(use_bbox and (
            _trial_crop_lookup.get("OS") or _trial_crop_lookup.get("OD")))
        if reverse:
            per_trial_filename = "mediapipe_reverse.npz"
        elif static_image_mode:
            per_trial_filename = "mediapipe_static.npz"
        elif _used_bbox_here:
            per_trial_filename = "mediapipe_cropped.npz"
        else:
            per_trial_filename = "mediapipe.npz"
        _trial_path = _trial_dir / per_trial_filename
        _slice = slice(_sf, _ef + 1)
        _n_trial = _ef - _sf + 1
        np.savez(
            str(_trial_path),
            OS_landmarks=OS_landmarks[_slice],
            OD_landmarks=OD_landmarks[_slice],
            OS_all_tracks=OS_all_tracks[_slice],
            OD_all_tracks=OD_all_tracks[_slice],
            confidence_OS=confidence_OS[_slice],
            confidence_OD=confidence_OD[_slice],
            distances=distances[_slice],
            start_frame=_sf,
            total_frames=_n_trial,
        )
        # ── Sidecar metadata ────────────────────────────────────────
        # Companion ``<stem>.params.json`` next to the npz captures the
        # parameters used to produce this output, so the Labels page can
        # reload them as defaults on the next open + the user can audit
        # what produced a given file independent of the jobs table.
        # File is named after the npz (mediapipe.npz → mediapipe.params
        # .json, mediapipe_reverse.npz → mediapipe_reverse.params.json)
        # so multiple-output subjects keep their metadata granular.
        import json as _json
        from datetime import datetime as _dt
        _params: dict = {
            "job_type": "mediapipe",
            "reverse": bool(reverse),
            "static_image_mode": bool(static_image_mode),
            "use_bbox": bool(use_bbox),
            "trial_idx": int(ti),
            "trial_name": _stem,
            "ran_at": _dt.utcnow().isoformat(timespec="seconds") + "Z",
        }
        # Persist the actual bbox the run consumed (if any) so the
        # Labels page can restore the exact crop next session, even
        # if mp_crop_boxes was edited since.
        if _trial_crop_lookup.get("OS"):
            _params["bbox_os"] = [float(v) for v in _trial_crop_lookup["OS"]]
        if _trial_crop_lookup.get("OD"):
            _params["bbox_od"] = [float(v) for v in _trial_crop_lookup["OD"]]
        _sidecar = _trial_dir / (_trial_path.stem + ".params.json")
        try:
            with open(_sidecar, "w") as _f:
                _json.dump(_params, _f, indent=2)
        except OSError as _e:
            logger.warning(f"Failed to write {_sidecar}: {_e}")
        written_paths.append(str(_trial_path))
        _vO = int(np.sum(~np.isnan(OS_landmarks[_slice, 0, 0])))
        _vD = int(np.sum(~np.isnan(OD_landmarks[_slice, 0, 0])))
        logger.info(
            f"Saved {_trial_path} (OS: {_vO}/{_n_trial}, OD: {_vD}/{_n_trial})"
        )

    if progress_callback:
        progress_callback(100.0)

    # Whenever any per-trial MP source file (forward / cropped /
    # reverse / static) is written, rebuild the Combined fusion for
    # that trial when at least 2 of the 4 sources are now present.
    # Combined is regenerated rather than partially patched -- it's
    # cheap and keeps the file aligned with the latest sources.
    try:
        for _p in written_paths:
            _trial_stem = Path(_p).parent.name
            _td = Path(_p).parent
            _present = sum(1 for _, _fn in COMBINED_SRC_FILES
                            if (_td / _fn).exists())
            if _present >= 2:
                build_combined_mp_npz_for_trial(subject_name, _trial_stem)
    except Exception as _e:
        logger.warning(
            f"Combined-MP rebuild after run_mediapipe failed for "
            f"{subject_name}: {_e}"
        )

    # Return the first per-trial path written so callers (status logs)
    # still get a meaningful string.  Multi-trial runs append all
    # paths to ``written_paths`` in order if the caller wants them.
    return written_paths[0] if written_paths else ""


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


_MAX_PREROLL_TRIM = 10  # Real pre-rolls are typically 1–5 frames; never 30+.


def _detect_frame_offset(subject_name: str, npz_data: dict) -> int:
    """Detect the pre-roll trim offset for MediaPipe NPZ files processed on a
    different machine.

    Some codecs expose negative-PTS pre-roll frames to OpenCV that the browser
    player never sees.  When the NPZ was built on such a machine it has
    ``stored_total = expected_total + pre_roll`` extra frames at the *start*.
    We can correct this deterministically by trimming those frames.

    Hardening:
      * Refuses to trim when the offset is suspiciously large
        (``> _MAX_PREROLL_TRIM``).  Large offsets nearly always come from
        the npz having been built under a different OpenCV decoder that
        agreed with the trial-map cv2 about per-file totals at the time
        but no longer does — trimming the start in that case silently
        corrupts every trial.  Refuse to trim and emit a clear warning
        recommending an MP re-run.
      * Refuses to trim when the offset matches an internal-file decoder
        gap (one of the trial videos reports more frames in metadata
        than cv2 can actually decode) — same reason.

    Returns:
      positive int  – NPZ has this many extra frames at the start; callers
                      should slice ``arr[offset:]`` before use.
      0             – counts match, the offset could not be determined,
                      or trimming was refused as unsafe.
    """
    try:
        trials = build_trial_map(subject_name)
        expected_total = trials[-1]["end_frame"] + 1 if trials else 0
        stored_total   = int(npz_data.get("total_frames", 0))

        if stored_total == expected_total:
            return 0

        offset = stored_total - expected_total

        if offset > 0:
            # Per-trial decoder-gap probe: check the frame-count cache for
            # each trial video and sum any reported-vs-actual gaps.  If the
            # offset is fully explained by internal-file gaps, the diff is
            # NOT pre-roll and trimming the start would corrupt earlier
            # trials.
            import json
            from pathlib import Path
            internal_gap = 0
            for t in trials:
                vp = Path(t["video_path"])
                cache_path = vp.parent / ".frame_counts.json"
                if not cache_path.exists():
                    continue
                try:
                    cache = json.loads(cache_path.read_text())
                except (json.JSONDecodeError, OSError):
                    continue
                entry = cache.get(vp.name)
                if not entry:
                    continue
                # We don't store cv2-reported metadata counts in the cache,
                # so the only proxy we have is the cached `count` (actual
                # decoded).  For the internal-gap check we need to compare
                # against what cv2 reports right now.
                try:
                    import cv2 as _cv2
                    cap = _cv2.VideoCapture(str(vp))
                    reported = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    actual = int(entry.get("count", reported))
                    if reported > actual:
                        internal_gap += (reported - actual)
                except Exception:
                    pass

            if internal_gap >= offset:
                logger.warning(
                    f"{subject_name}: NPZ has {stored_total} frames, trial map expects "
                    f"{expected_total} (offset={offset}). Internal-file decoder gaps sum "
                    f"to {internal_gap} — diff is NOT pre-roll, refusing to trim. "
                    f"Re-run MediaPipe to rebuild the NPZ under the current cv2."
                )
                return 0

            if offset > _MAX_PREROLL_TRIM:
                logger.warning(
                    f"{subject_name}: NPZ offset {offset} frames is too large to be "
                    f"pre-roll (max ≤ {_MAX_PREROLL_TRIM}). Likely a different cv2 decoder "
                    f"built the NPZ. Refusing to trim — re-run MediaPipe to fix."
                )
                return 0

            logger.info(
                f"{subject_name}: NPZ has {stored_total} frames, trial map expects "
                f"{expected_total} — trimming {offset} pre-roll frame(s) from start."
            )
            return offset
        else:
            logger.warning(
                f"{subject_name}: Frame count mismatch — NPZ has {stored_total} frames "
                f"but trial map expects {expected_total} ({-offset} frames missing). "
                f"Consider reprocessing MediaPipe on this machine."
            )
            return 0
    except Exception as e:
        logger.debug(f"Could not detect frame offset for {subject_name}: {e}")
        return 0


def _load_mediapipe_per_trial(subject_name: str, per_trial_filename: str) -> dict | None:
    """Reassemble subject-wide MediaPipe arrays from per-trial npz files.

    For every trial in ``build_trial_map(subject_name)`` we look for
    ``<dlc>/<subject>/<trial_stem>/<per_trial_filename>`` and splice
    its slice into the subject-wide output arrays.  Missing files are
    left as NaN (matching the legacy behaviour of pre-roll trials).
    Returns ``None`` when NO per-trial files exist (so the caller can
    fall back to the legacy combined npz).
    """
    settings = get_settings()
    try:
        trials = build_trial_map(subject_name)
    except Exception:
        return None
    if not trials:
        return None
    total_frames = trials[-1]["end_frame"] + 1
    OS_lm = np.full((total_frames, N_JOINTS, 2), np.nan)
    OD_lm = np.full((total_frames, N_JOINTS, 2), np.nan)
    OS_all = np.full((total_frames, 2, N_JOINTS, 2), np.nan)
    OD_all = np.full((total_frames, 2, N_JOINTS, 2), np.nan)
    conf_OS = np.full(total_frames, np.nan)
    conf_OD = np.full(total_frames, np.nan)
    dist = np.full(total_frames, np.nan)
    any_loaded = False
    for trial in trials:
        stem = trial["trial_name"]
        p = settings.dlc_path / subject_name / stem / per_trial_filename
        if not p.exists():
            continue
        try:
            data = np.load(str(p))
        except (OSError, ValueError) as e:
            logger.warning(f"Failed to load {p}: {e}")
            continue
        sf = int(trial["start_frame"])
        ef = sf + int(trial["frame_count"]) - 1
        ef = min(ef, total_frames - 1)
        # Slice the loaded per-trial arrays to the trial's actual
        # length, then drop them into the subject-wide arrays.
        for key, dst in (
            ("OS_landmarks", OS_lm), ("OD_landmarks", OD_lm),
            ("OS_all_tracks", OS_all), ("OD_all_tracks", OD_all),
            ("confidence_OS", conf_OS), ("confidence_OD", conf_OD),
            ("distances", dist),
        ):
            if key not in data.files:
                continue
            src = data[key]
            n_dst = ef - sf + 1
            n = min(n_dst, src.shape[0])
            if n > 0:
                dst[sf:sf + n] = src[:n]
        any_loaded = True
    if not any_loaded:
        return None
    return {
        "OS_landmarks": OS_lm, "OD_landmarks": OD_lm,
        "OS_all_tracks": OS_all, "OD_all_tracks": OD_all,
        "confidence_OS": conf_OS, "confidence_OD": conf_OD,
        "distances": dist, "total_frames": total_frames,
    }


def _attach_fresh_distances(result: dict | None, subject_name: str) -> dict | None:
    """Always (re)compute the ``distances`` array on the fly from the
    loaded landmarks + the subject's current calibration.

    Triangulation is cheap (~0.1 ms / frame) and recomputing on read
    sidesteps three persistent storage hazards:
      * Remote-produced npz files carry ``distances=NaN`` because the
        remote script has no access to local calibration.
      * Re-calibration silently invalidates any stored distances.
      * Older per-trial layouts may have an empty / placeholder
        distances slice from partial migrations.

    When no calibration is available, falls back to 2D OS-pixel
    distance.  When the loaded ``distances`` already has any non-NaN
    value AND no calibration is found, keeps the stored values rather
    than overwriting with the 2D fallback -- this preserves a
    previously-computed 3D set on uncalibrated subjects.
    """
    if result is None:
        return None
    os_lm = result.get("OS_landmarks")
    od_lm = result.get("OD_landmarks")
    if os_lm is None or od_lm is None:
        return result
    try:
        calib = get_calibration_for_subject(subject_name)
    except Exception:
        calib = None
    if calib is not None:
        result["distances"] = _compute_distances(os_lm, od_lm, calib)
    else:
        existing = result.get("distances")
        if existing is None or not np.any(~np.isnan(existing)):
            result["distances"] = _compute_2d_distances(os_lm)
    return result


def load_mediapipe_prelabels(subject_name: str,
                              filename: str = "mediapipe_prelabels.npz") -> dict | None:
    """Load saved MediaPipe prelabels for a subject.

    Prefers the new per-trial layout (``<dlc>/<subject>/<trial_stem>/
    mediapipe.npz`` / ``mediapipe_reverse.npz``); falls back to the
    legacy combined ``mediapipe_prelabels.npz`` (or
    ``mediapipe_reverse_prelabels.npz``) only when no per-trial files
    exist.  The fallback path also runs the legacy pre-roll-trim
    offset detection so combined files from older machines keep
    working until the startup migration converts them.

    Returns dict with keys: OS_landmarks, OD_landmarks (N, 21, 2),
    confidence_OS, confidence_OD, distances, total_frames.
    Returns None if neither layout has data.
    """
    # Map combined-file name → per-trial-file name.  Forward pass:
    # mediapipe_prelabels.npz → mediapipe.npz.  Reverse pass:
    # mediapipe_reverse_prelabels.npz → mediapipe_reverse.npz.
    per_trial_map = {
        "mediapipe_prelabels.npz":         "mediapipe.npz",
        "mediapipe_reverse_prelabels.npz": "mediapipe_reverse.npz",
        "mediapipe_combined.npz":          "mediapipe_combined.npz",
        "mediapipe_static.npz":            "mediapipe_static.npz",
        "mediapipe_cropped.npz":           "mediapipe_cropped.npz",
    }
    per_trial_filename = per_trial_map.get(filename, "mediapipe.npz")
    per_trial_result = _load_mediapipe_per_trial(subject_name, per_trial_filename)
    if per_trial_result is not None:
        return _attach_fresh_distances(per_trial_result, subject_name)

    # ── Legacy combined-file fallback ──
    settings = get_settings()
    npz_path = settings.dlc_path / subject_name / filename
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

        # Apply pre-roll trim: NPZ stored more frames than the trial map expects
        # because it was processed on a machine that exposed extra pre-roll frames.
        # Trim the first `frame_offset` frames so indices align with the trial map.
        if frame_offset > 0:
            OS_lm    = OS_lm[frame_offset:]
            OD_lm    = OD_lm[frame_offset:]
            conf_OS  = conf_OS[frame_offset:]
            conf_OD  = conf_OD[frame_offset:]
            if dist is not None:
                dist = dist[frame_offset:]

        result = {
            "OS_landmarks": OS_lm,
            "OD_landmarks": OD_lm,
            "confidence_OS": conf_OS,
            "confidence_OD": conf_OD,
            "distances": dist if dist is not None else data.get("distances"),
            "total_frames": len(OS_lm),  # use trimmed length
        }
        # Preserve run history keys
        for k in data.files if hasattr(data, 'files') else data:
            if k.startswith("distances_run_") or k.startswith("crop_run_"):
                result[k] = data[k]
        return _attach_fresh_distances(result, subject_name)
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

        return _attach_fresh_distances({
            "OS_landmarks": OS_lm,
            "OD_landmarks": OD_lm,
            "confidence_OS": data["confidence_OS"],
            "confidence_OD": data["confidence_OD"],
            "distances": data.get("distances"),
            "total_frames": n,
        }, subject_name)


def build_combined_mp_npz_tempfile(subject_name: str,
                                     reverse: bool = False) -> str | None:
    """Aggregate the per-trial MP files for a subject into a single
    temp ``.npz`` and return its path.

    Used by the remote-dispatch paths (HRnet inference, etc.) that
    still expect a combined ``mediapipe_prelabels.npz`` on the remote.
    Callers are responsible for unlinking the returned tempfile.
    Returns ``None`` when no MP data exists.
    """
    import tempfile
    data = load_mediapipe_prelabels(
        subject_name,
        filename=("mediapipe_reverse_prelabels.npz" if reverse
                  else "mediapipe_prelabels.npz"),
    )
    if data is None:
        return None
    fd, tmp = tempfile.mkstemp(prefix=f"mp_combined_{subject_name}_",
                                suffix=".npz")
    import os as _os
    _os.close(fd)
    payload: dict = {}
    for k in ("OS_landmarks", "OD_landmarks", "OS_all_tracks",
              "OD_all_tracks", "confidence_OS", "confidence_OD",
              "distances"):
        v = data.get(k)
        if v is not None:
            payload[k] = v
    payload["total_frames"] = data.get("total_frames", 0)
    np.savez(tmp, **payload)
    return tmp


def has_mediapipe_data(subject_name: str, reverse: bool = False) -> bool:
    """Return True if any trial has saved MediaPipe data on disk.

    Checks the per-trial layout
    (``<dlc>/<subject>/<trial_stem>/mediapipe.npz``, or
    ``mediapipe_reverse.npz`` when ``reverse``) and the legacy
    combined file as a fallback.
    """
    settings = get_settings()
    subj_dir = settings.dlc_path / subject_name
    if not subj_dir.exists():
        return False
    per_trial_name = "mediapipe_reverse.npz" if reverse else "mediapipe.npz"
    try:
        for trial_dir in subj_dir.iterdir():
            if trial_dir.is_dir() and (trial_dir / per_trial_name).exists():
                return True
    except OSError:
        pass
    legacy = "mediapipe_reverse_prelabels.npz" if reverse \
        else "mediapipe_prelabels.npz"
    return (subj_dir / legacy).exists()


# Source codes used by the Combined-MP builder.  Stored per frame in
# the output npz under ``source_OS`` / ``source_OD`` so consumers can
# tell which pass drove each camera-frame.
COMBINED_SRC_NONE     = 0
COMBINED_SRC_FORWARD  = 1
COMBINED_SRC_CROPPED  = 2
COMBINED_SRC_REVERSE  = 3
COMBINED_SRC_STATIC   = 4
COMBINED_SRC_FILES = [
    (COMBINED_SRC_FORWARD, "mediapipe.npz"),
    (COMBINED_SRC_CROPPED, "mediapipe_cropped.npz"),
    (COMBINED_SRC_REVERSE, "mediapipe_reverse.npz"),
    (COMBINED_SRC_STATIC,  "mediapipe_static.npz"),
]
COMBINED_SRC_NAMES = {
    COMBINED_SRC_FORWARD: "forward",
    COMBINED_SRC_CROPPED: "cropped",
    COMBINED_SRC_REVERSE: "reverse",
    COMBINED_SRC_STATIC:  "static",
}



def build_combined_mp_npz_for_trial(subject_name: str, trial_stem: str) -> str | None:
    """Fuse up to four per-trial MP source npzs (forward, cropped,
    reverse, static) into a single ``mediapipe_combined.npz``.

    Per-frame logic (each camera handled independently):

      * No source has labels -> NaN.
      * Exactly one source has labels -> use that source's full
        21-joint set for that camera, that frame.
      * Multiple sources have labels -> consider every
        (OS_source, OD_source) combination where the thumb tip and
        index tip are present on both cameras and triangulate to
        finite 3D points.  Pick the combo with the smallest 3-D
        thumb-tip ↔ index-tip distance.

    Bone-length heuristics were removed because the relevant joint
    pairs (thumb CMC ↔ tip and index MCP ↔ tip) span the finger
    knuckles and flex with the movement, so they aren't a stable
    reference.  Thumb-index aperture alone keeps the choice tied to
    the only quantity the downstream analysis cares about.

    Writes ``<dlc>/<subject>/<trial>/mediapipe_combined.npz`` and
    returns its path on success; returns None if no source exists
    for the trial.
    """
    settings = get_settings()
    trial_dir = settings.dlc_path / subject_name / trial_stem
    sources: dict[int, np.lib.npyio.NpzFile] = {}
    for code, fname in COMBINED_SRC_FILES:
        p = trial_dir / fname
        if p.exists():
            try:
                sources[code] = np.load(str(p))
            except (OSError, ValueError) as e:
                logger.warning(f"Combined: skipping {p} ({e})")
    if not sources:
        return None

    # Size output arrays from any available source (they all should
    # share the same length, but defensively pick the first).
    any_src = next(iter(sources.values()))
    n = int(any_src["total_frames"]) if "total_frames" in any_src.files \
        else int(any_src["OS_landmarks"].shape[0])
    start_frame = int(any_src["start_frame"]) if "start_frame" in any_src.files else 0

    # Per-source landmark + confidence views (None if source absent).
    def _arr(code, key):
        s = sources.get(code)
        if s is None or key not in s.files:
            return None
        return s[key]
    os_lm = {c: _arr(c, "OS_landmarks") for c, _ in COMBINED_SRC_FILES}
    od_lm = {c: _arr(c, "OD_landmarks") for c, _ in COMBINED_SRC_FILES}
    os_cf = {c: _arr(c, "confidence_OS") for c, _ in COMBINED_SRC_FILES}
    od_cf = {c: _arr(c, "confidence_OD") for c, _ in COMBINED_SRC_FILES}

    try:
        calib = get_calibration_for_subject(subject_name)
    except Exception:
        calib = None

    def _valid(arr, f, joint):
        return (arr is not None and f < arr.shape[0]
                and not np.isnan(arr[f, joint, 0]))

    def _tri(os_pt, od_pt):
        """Triangulate a single 2D pair to 3D, returning None on
        failure (no calibration / NaN / cv2 hiccup)."""
        if calib is None:
            return None
        try:
            pts3d = triangulate_points(
                np.array([os_pt], dtype=np.float64),
                np.array([od_pt], dtype=np.float64),
                calib,
            )
        except Exception:
            return None
        if pts3d is None or np.any(np.isnan(pts3d)):
            return None
        return pts3d[0]

    # ── Per-frame: pick the combo with the smallest 3-D thumb-tip ↔
    #    index-tip distance.  Bone-length terms were dropped because
    #    the candidate "bones" (thumb CMC ↔ tip, index MCP ↔ tip)
    #    cross knuckles that bend, so they aren't fixed lengths.
    OS_combined = np.full((n, N_JOINTS, 2), np.nan)
    OD_combined = np.full((n, N_JOINTS, 2), np.nan)
    conf_OS = np.full(n, np.nan)
    conf_OD = np.full(n, np.nan)
    src_OS = np.zeros(n, dtype=np.uint8)
    src_OD = np.zeros(n, dtype=np.uint8)

    for f in range(n):
        os_codes = [c for c in os_lm if (_valid(os_lm[c], f, THUMB_TIP)
                                          and _valid(os_lm[c], f, INDEX_TIP))]
        od_codes = [c for c in od_lm if (_valid(od_lm[c], f, THUMB_TIP)
                                          and _valid(od_lm[c], f, INDEX_TIP))]
        os_pick = od_pick = COMBINED_SRC_NONE
        best_d = float("inf")
        for oc in os_codes:
            for dc in od_codes:
                tip_t = _tri(os_lm[oc][f, THUMB_TIP],
                             od_lm[dc][f, THUMB_TIP])
                tip_i = _tri(os_lm[oc][f, INDEX_TIP],
                             od_lm[dc][f, INDEX_TIP])
                if tip_t is None or tip_i is None:
                    continue
                d = float(np.linalg.norm(tip_t - tip_i))
                if d < best_d:
                    best_d = d
                    os_pick, od_pick = oc, dc

        # When tip-based selection didn't run for a camera (no
        # candidate had both tips), fall back to "use whichever
        # source has any non-NaN joint" so non-tip joints still
        # get populated.  Preference order matches the source
        # code numbering: forward → cropped → reverse → static.
        if os_pick == COMBINED_SRC_NONE:
            for c, _ in COMBINED_SRC_FILES:
                arr = os_lm.get(c)
                if arr is not None and f < arr.shape[0] \
                        and np.any(~np.isnan(arr[f, :, 0])):
                    os_pick = c; break
        if od_pick == COMBINED_SRC_NONE:
            for c, _ in COMBINED_SRC_FILES:
                arr = od_lm.get(c)
                if arr is not None and f < arr.shape[0] \
                        and np.any(~np.isnan(arr[f, :, 0])):
                    od_pick = c; break

        # Copy the chosen source's full 21-joint set per camera.
        if os_pick != COMBINED_SRC_NONE:
            OS_combined[f] = os_lm[os_pick][f]
            cf = os_cf.get(os_pick)
            if cf is not None and f < cf.shape[0]:
                conf_OS[f] = cf[f]
        if od_pick != COMBINED_SRC_NONE:
            OD_combined[f] = od_lm[od_pick][f]
            cf = od_cf.get(od_pick)
            if cf is not None and f < cf.shape[0]:
                conf_OD[f] = cf[f]
        src_OS[f] = os_pick
        src_OD[f] = od_pick

    out_path = trial_dir / "mediapipe_combined.npz"
    try:
        np.savez(
            str(out_path),
            OS_landmarks=OS_combined,
            OD_landmarks=OD_combined,
            confidence_OS=conf_OS,
            confidence_OD=conf_OD,
            distances=np.full(n, np.nan),  # recomputed at load time
            start_frame=start_frame,
            total_frames=n,
            source_OS=src_OS,
            source_OD=src_OD,
        )
    except OSError as e:
        logger.warning(f"Failed to write {out_path}: {e}")
        return None

    def _counts(arr):
        return ", ".join(
            f"{COMBINED_SRC_NAMES[c]}={int(np.sum(arr == c))}"
            for c, _ in COMBINED_SRC_FILES if int(np.sum(arr == c))
        ) or "none"
    logger.info(
        f"Combined MP for {subject_name}/{trial_stem}: "
        f"OS [{_counts(src_OS)}] | OD [{_counts(src_OD)}]"
    )

    # Keep the subject's median hand sizes current — Combined-MP just
    # changed for this trial's hand.
    try:
        update_hand_sizes(subject_name)
    except Exception as e:
        logger.warning(f"Hand-size update failed for {subject_name}: {e}")

    return str(out_path)


def maybe_rebuild_combined_for_subject(subject_name: str) -> int:
    """Walk a subject's per-trial dirs and rebuild
    ``mediapipe_combined.npz`` for every trial where BOTH
    ``mediapipe.npz`` and ``mediapipe_reverse.npz`` exist.  Returns
    the number of combined files (re)written.  Safe to call anytime;
    skipping a trial is a no-op."""
    settings = get_settings()
    subj_dir = settings.dlc_path / subject_name
    if not subj_dir.is_dir():
        return 0
    n_built = 0
    for trial_dir in subj_dir.iterdir():
        if not trial_dir.is_dir():
            continue
        # Build combined as long as at least TWO of the four source
        # files exist (one source alone is just that source -- no
        # fusion to do).  Mirrors the 4-source builder's input set.
        present = sum(1 for _, fname in COMBINED_SRC_FILES
                       if (trial_dir / fname).exists())
        if present >= 2:
            try:
                if build_combined_mp_npz_for_trial(subject_name, trial_dir.name):
                    n_built += 1
            except Exception as e:
                logger.warning(
                    f"Combined rebuild failed for "
                    f"{subject_name}/{trial_dir.name}: {e}")
    return n_built


# MediaPipe hand skeleton (21 landmarks) — the bones whose median
# lengths sum to "hand size".
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                  # palm base → pinky MCP
]


def _hand_letter(trial_name: str) -> str | None:
    """'L'/'R' from a trial name like 'Con03_L1' → 'L'."""
    suffix = str(trial_name).split("_")[-1]
    if suffix[:1].upper() in ("L", "R"):
        return suffix[:1].upper()
    return None


def compute_hand_sizes_for_subject(subject_name: str) -> dict:
    """Median hand size (mm) per hand from the Combined-MP 3D skeleton.

    For each hand (L/R), triangulates all 21 joints on every frame of
    that hand's trials (Combined OS+OD landmarks + stereo calibration),
    takes the median length of each bone across frames, and sums them.
    Returns ``{"left": float|None, "right": float|None}``.  Requires
    calibration (3D); returns None for a hand with no usable frames.
    """
    settings = get_settings()
    out = {"left": None, "right": None}
    calib = get_calibration_for_subject(subject_name)
    if calib is None:
        return out
    try:
        from .video import build_trial_map
        trials = build_trial_map(subject_name)
    except Exception:
        return out

    # Per-hand: list of per-bone length samples across all frames.
    per_hand_bones = {"L": [[] for _ in HAND_CONNECTIONS],
                      "R": [[] for _ in HAND_CONNECTIONS]}
    subj_dir = settings.dlc_path / subject_name

    for t in trials:
        stem = t["trial_name"]
        hand = _hand_letter(stem)
        if hand is None:
            continue
        npz = subj_dir / stem / "mediapipe_combined.npz"
        if not npz.exists():
            continue
        try:
            data = np.load(str(npz))
            os_lm = data["OS_landmarks"]   # (n,21,2)
            od_lm = data["OD_landmarks"]
        except Exception:
            continue
        n = min(len(os_lm), len(od_lm))
        for i in range(n):
            # Triangulate all 21 joints; missing 2D points come back as
            # NaN 3D, so we just skip individual bones rather than the
            # whole frame.
            pts3d = triangulate_points(os_lm[i], od_lm[i], calib)   # (21,3)
            if pts3d is None:
                continue
            for bi, (a, b) in enumerate(HAND_CONNECTIONS):
                if np.isnan(pts3d[a]).any() or np.isnan(pts3d[b]).any():
                    continue
                per_hand_bones[hand][bi].append(
                    float(np.linalg.norm(pts3d[a] - pts3d[b])))

    for hand, key in (("L", "left"), ("R", "right")):
        bones = per_hand_bones[hand]
        if all(len(s) == 0 for s in bones):
            continue
        medians = [float(np.median(s)) for s in bones if len(s) > 0]
        if medians:
            out[key] = round(sum(medians), 3)
    return out


def update_hand_sizes(subject_name: str) -> dict:
    """Recompute and persist a subject's median hand sizes (mm)."""
    sizes = compute_hand_sizes_for_subject(subject_name)
    try:
        from ..db import get_db_ctx
        with get_db_ctx() as db:
            db.execute(
                "UPDATE subjects SET hand_size_left = ?, hand_size_right = ? "
                "WHERE name = ?",
                (sizes["left"], sizes["right"], subject_name),
            )
    except Exception as e:
        logger.warning(f"Failed to store hand sizes for {subject_name}: {e}")
    return sizes


def load_mediapipe_combined_prelabels(subject_name: str) -> dict | None:
    """Load the per-trial combined MediaPipe prelabels for a subject.

    The combined layer is a per-(frame, camera) fusion of up to four
    MP source passes (forward, cropped, reverse, static) -- see
    ``build_combined_mp_npz_for_trial`` for the selection logic.
    Loaded just like the other passes so the Labels page can show
    it as its own source layer.
    """
    return load_mediapipe_prelabels(
        subject_name,
        filename="mediapipe_combined.npz",  # not a real legacy file;
                                              # per_trial_map below maps
                                              # it to the per-trial name.
    )


def load_mediapipe_cropped_prelabels(subject_name: str) -> dict | None:
    """Load the bbox-cropped forward MediaPipe prelabels for a subject.

    Forward MP runs with a non-empty per-trial bbox crop write to
    ``<dlc>/<subject>/<trial>/mediapipe_cropped.npz`` instead of
    ``mediapipe.npz``, so the bbox-free baseline and the cropped
    variant coexist on disk.  Loaded as its own Labels-page layer.
    """
    return load_mediapipe_prelabels(
        subject_name,
        filename="mediapipe_cropped.npz",
    )


def load_mediapipe_static_prelabels(subject_name: str) -> dict | None:
    """Load the static-image-mode MediaPipe prelabels for a subject.

    Static mode runs MediaPipe's full palm detector on every frame
    (no between-frame tracker), giving a fundamentally different
    signal from the forward / reverse tracked passes.  Stored
    per-trial in ``<dlc>/<subject>/<trial>/mediapipe_static.npz``.
    """
    return load_mediapipe_prelabels(
        subject_name,
        filename="mediapipe_static.npz",
    )


def load_mediapipe_reverse_prelabels(subject_name: str) -> dict | None:
    """Load the reverse-pass MediaPipe prelabels for a subject.

    The reverse pass feeds frames to MediaPipe in descending temporal
    order so the tracker enters cold-start frames already locked on.
    Output schema matches the forward pass (OS_landmarks /
    OD_landmarks / confidences / distances / total_frames).
    """
    return load_mediapipe_prelabels(
        subject_name,
        filename="mediapipe_reverse_prelabels.npz",
    )


def get_mediapipe_for_session(subject_name: str,
                                prefer_combined: bool = False) -> dict | None:
    """Get MediaPipe predictions formatted for the labeler API response.

    Returns dict with per-camera thumb/index arrays as lists (JSON-serializable),
    plus distances array. Returns None if no prelabels exist.

    When ``prefer_combined`` is True, prefers the Combined (forward+
    reverse fusion) per-trial output; falls back to the original
    forward MediaPipe when no combined npz exists for any trial.
    Used by the Events page so the distance plot benefits from the
    Combined layer's better tip selection -- forward only is used
    when Combined is unavailable.  Other callers keep the original
    forward-only behaviour by leaving the flag False.

    If stored distances are all NaN but calibration is now available,
    recomputes them from the MP coordinates and updates the npz file.
    """
    data = None
    used_combined = False
    if prefer_combined:
        data = load_mediapipe_combined_prelabels(subject_name)
        if data is not None:
            used_combined = True
    if data is None:
        data = load_mediapipe_prelabels(subject_name)
    if data is None:
        return None

    OS_lm = data["OS_landmarks"]
    OD_lm = data["OD_landmarks"]

    # Recompute distances if they're all NaN.  Note: the loader
    # itself recomputes distances on the fly now (see
    # _attach_fresh_distances), so this block almost never fires
    # anymore -- but we keep it as a safety net for stale callers
    # that build a ``data`` dict directly without going through the
    # loader.  Skip the npz write-back when the data came from the
    # combined layer: the persistent rewrite path is hard-coded to
    # the forward filename and would clobber the forward file with
    # combined values.
    if np.all(np.isnan(data["distances"])) and not used_combined:
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
            save_data = {
                "OS_landmarks": OS_lm,
                "OD_landmarks": OD_lm,
                "confidence_OS": data["confidence_OS"],
                "confidence_OD": data["confidence_OD"],
                "distances": distances,
                "total_frames": data["total_frames"],
            }
            # Preserve run history keys
            for k in data:
                if k.startswith("distances_run_") or k.startswith("crop_run_"):
                    save_data[k] = data[k]
            np.savez(npz_path, **save_data)
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

    # Include all available distance sources for the legend
    available_sources = [{"key": "mediapipe", "label": "MediaPipe", "color": "#4a9eff"}]

    # Vision distances (compute from vision_prelabels.npz if available)
    vision_path = settings.dlc_path / subject_name / "vision_prelabels.npz"
    if vision_path.exists():
        try:
            v_data = np.load(str(vision_path))
            v_os = v_data["OS_landmarks"]
            v_od = v_data.get("OD_landmarks")
            calib = get_calibration_for_subject(subject_name)
            if calib is not None and v_od is not None:
                v_dist = _compute_distances(v_os, v_od, calib)
            else:
                v_dist = _compute_2d_distances(v_os)
            response["vision_distances"] = _dist_to_list(v_dist)
            available_sources.append({"key": "vision", "label": "Apple Vision", "color": "#ff9800"})
        except Exception:
            pass

    # DLC distances (from dlc predictions if available)
    try:
        from .dlc_predictions import load_dlc_predictions
        dlc_data = load_dlc_predictions(subject_name)
        if dlc_data and "distances" in dlc_data:
            response["dlc_distances"] = _dist_to_list(dlc_data["distances"])
            available_sources.append({"key": "dlc", "label": "DLC", "color": "#4caf50"})
    except Exception:
        pass

    response["available_sources"] = available_sources

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

    # Don't apply frame_offset here — landmarks are stored at OpenCV frame indices.
    # The offset compensation (for pre-roll frames browsers skip) happens at display time.
    frame_offset = 0

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

    # Pick the tapping hand — use video filename as trial name hint
    trial_hint = os.path.splitext(os.path.basename(video_path))[0]
    track_idx = _pick_tapping_track(tracks, f"{subject_name}/{camera_key}/cropped", trial_name=trial_hint)

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

    # Expand arrays if video has more frames than npz
    max_needed = start_frame + frame_offset + n_frames
    if max_needed > total_frames:
        new_total = max_needed
        for key in ("OS_landmarks", "OD_landmarks"):
            old = data[key]
            expanded = np.full((new_total, old.shape[1], old.shape[2]), np.nan)
            expanded[:old.shape[0]] = old
            data[key] = expanded
        for key in ("confidence_OS", "confidence_OD"):
            old = data[key]
            expanded = np.full(new_total, np.nan)
            expanded[:old.shape[0]] = old
            data[key] = expanded
        old_dist = data["distances"]
        new_dist = np.full(new_total, np.nan)
        new_dist[:old_dist.shape[0]] = old_dist
        data["distances"] = new_dist
        data["total_frames"] = np.array(new_total)
        total_frames = new_total
        logger.info(f"Expanded npz arrays from {old.shape[0]} to {new_total} frames")

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


# ── MediaPipe Pose prelabeling ────────────────────────────────────────────

# Pose landmark indices (33 total)
POSE_N_LANDMARKS = 33
POSE_NAMES = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
    'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
    'left_index', 'right_index', 'left_thumb', 'right_thumb',
    'left_hip', 'right_hip', 'left_knee', 'right_knee',
    'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
    'left_foot_index', 'right_foot_index',
]


def run_pose_prelabels(subject_name: str, progress_callback=None) -> Path:
    """Run MediaPipe Pose on all videos for a subject.

    Stores results in dlc/{subject}/pose_prelabels.npz with arrays:
        OS_pose: (total_frames, 33, 2)  — per-camera pose landmarks
        OD_pose: (total_frames, 33, 2)
        pose_confidence_OS: (total_frames, 33)
        pose_confidence_OD: (total_frames, 33)
    """
    import mediapipe as mp_lib

    settings = get_settings()
    videos = get_subject_videos(subject_name)
    if not videos:
        raise ValueError(f"No videos found for subject {subject_name}")

    trials = build_trial_map(subject_name)
    total_frames = trials[-1]["end_frame"] + 1 if trials else 0

    OS_pose = np.full((total_frames, POSE_N_LANDMARKS, 2), np.nan)
    OD_pose = np.full((total_frames, POSE_N_LANDMARKS, 2), np.nan)
    conf_OS = np.full((total_frames, POSE_N_LANDMARKS), np.nan)
    conf_OD = np.full((total_frames, POSE_N_LANDMARKS), np.nan)

    pose_det = mp_lib.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    processed = 0
    for trial in trials:
        video_path = trial["video_path"]
        cap = cv2.VideoCapture(video_path)
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        is_stereo = (fw / fh) > 1.7 if fh > 0 else False
        half_w = fw // 2 if is_stereo else fw

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        for local_frame in range(trial["frame_count"]):
            ret, frame = cap.read()
            if not ret:
                break

            global_frame = trial["start_frame"] + local_frame
            if global_frame >= total_frames:
                break

            frame = np.ascontiguousarray(frame, dtype=np.uint8)

            if is_stereo:
                left = frame[:, :half_w, :]
                right = frame[:, half_w:, :]

                rgb_l = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
                res_l = pose_det.process(rgb_l)
                if res_l.pose_landmarks:
                    for j, lm in enumerate(res_l.pose_landmarks.landmark):
                        OS_pose[global_frame, j] = [lm.x * half_w, lm.y * fh]
                        conf_OS[global_frame, j] = lm.visibility

                rgb_r = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)
                res_r = pose_det.process(rgb_r)
                if res_r.pose_landmarks:
                    for j, lm in enumerate(res_r.pose_landmarks.landmark):
                        OD_pose[global_frame, j] = [lm.x * (fw - half_w), lm.y * fh]
                        conf_OD[global_frame, j] = lm.visibility
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose_det.process(rgb)
                if res.pose_landmarks:
                    for j, lm in enumerate(res.pose_landmarks.landmark):
                        OS_pose[global_frame, j] = [lm.x * fw, lm.y * fh]
                        conf_OS[global_frame, j] = lm.visibility

            processed += 1
            if progress_callback and processed % 10 == 0:
                progress_callback(processed / total_frames * 100)

        cap.release()

    pose_det.close()

    # Save
    npz_path = settings.dlc_path / subject_name / "pose_prelabels.npz"
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(npz_path),
        OS_pose=OS_pose,
        OD_pose=OD_pose,
        pose_confidence_OS=conf_OS,
        pose_confidence_OD=conf_OD,
        total_frames=np.array(total_frames),
    )

    valid = np.sum(~np.isnan(OS_pose[:, 0, 0]))
    logger.info(f"Pose prelabels saved: {npz_path} ({valid}/{total_frames} frames with data)")

    if progress_callback:
        progress_callback(100)

    return npz_path


def load_pose_prelabels(subject_name: str) -> dict | None:
    """Load saved pose prelabels for a subject. Returns None if not available.

    Applies the same per-subject pre-roll trim that
    ``load_mediapipe_prelabels`` does -- without it, any subject built
    on a machine whose codec exposed negative-PTS frames (the npz has
    ``offset`` extra leading frames) would have its pose arrays
    misaligned with the trial frame indices, so callers indexing
    ``OS_pose[frame_num]`` would read the wrong frame's elbow.  This
    showed up on the deidentify preview as a missing (or visibly
    wrong) arm-triangle vertex while the live overlay -- which goes
    through ``load_mediapipe_prelabels`` first -- looked correct.
    """
    settings = get_settings()
    npz_path = settings.dlc_path / subject_name / "pose_prelabels.npz"
    if not npz_path.exists():
        return None

    data = dict(np.load(str(npz_path)))
    # Reuse the hand-data offset detector (it inspects the saved npz +
    # the source mp4 to figure out how many pre-roll frames the codec
    # exposed beyond what the browser sees).
    frame_offset = _detect_frame_offset(subject_name, data)
    OS_pose = data["OS_pose"]
    OD_pose = data["OD_pose"]
    conf_OS = data["pose_confidence_OS"]
    conf_OD = data["pose_confidence_OD"]
    total_frames = int(data["total_frames"])
    if frame_offset > 0:
        OS_pose = OS_pose[frame_offset:]
        OD_pose = OD_pose[frame_offset:]
        conf_OS = conf_OS[frame_offset:]
        conf_OD = conf_OD[frame_offset:]
        total_frames = max(0, total_frames - frame_offset)
    return {
        "OS_pose": OS_pose,
        "OD_pose": OD_pose,
        "pose_confidence_OS": conf_OS,
        "pose_confidence_OD": conf_OD,
        "total_frames": total_frames,
    }
