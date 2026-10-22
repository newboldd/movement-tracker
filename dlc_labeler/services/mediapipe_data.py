"""Reading MediaPipe hand landmarks that something else produced.

This app does not run MediaPipe.  Movement Tracker writes
``dlc/{subject}/{trial}/mediapipe.npz`` — trial-local ``(n, 21, 2)``
arrays per camera, with a ``start_frame`` offset — and this module reads
them back so the labeling page can aim its zoom at a fingertip and plot
the thumb-index aperture.

A subject with no npz still labels perfectly well; it is simply not
pre-aimed.  Nothing here imports the ``mediapipe`` package, which is why
it is not a dependency of this app.
"""
from __future__ import annotations

import logging

import numpy as np

from ..config import get_settings
from .calibration import get_calibration_for_subject, triangulate_points
from .video import build_trial_map

logger = logging.getLogger(__name__)

# MediaPipe's hand landmark indices.
THUMB_TIP = 4
INDEX_TIP = 8
WRIST = 0
N_JOINTS = 21

MEDIAPIPE_NPZ = "mediapipe.npz"

# Which landmark stands for each bodypart.  Matched by name so renaming
# or reordering bodyparts cannot quietly point "thumb" at the index.
BODYPART_JOINTS = {"thumb": THUMB_TIP, "index": INDEX_TIP}


def trial_npz_path(subject_name: str, trial_name: str):
    return get_settings().dlc_path / subject_name / trial_name / MEDIAPIPE_NPZ


def load_trial(subject_name: str, trial_name: str) -> dict | None:
    """The forward pass for one trial, or None when there is none."""
    path = trial_npz_path(subject_name, trial_name)
    if not path.is_file():
        return None
    try:
        data = np.load(str(path))
    except (OSError, ValueError) as e:
        logger.warning("Could not read %s: %s", path, e)
        return None
    return {k: data[k] for k in data.files}


def load_subject_landmarks(subject_name: str,
                           camera_mode: str | None = None) -> dict | None:
    """Every trial's landmarks, stitched into subject-wide arrays.

    Indexed by the global frame numbers the label store uses, so a
    caller can look up a frame without knowing which trial it is in.
    Gaps — trials with no npz — stay NaN.
    """
    trials = build_trial_map(subject_name, camera_mode=camera_mode)
    if not trials:
        return None

    total = trials[-1]["end_frame"] + 1
    out = {
        "OS_landmarks": np.full((total, N_JOINTS, 2), np.nan),
        "OD_landmarks": np.full((total, N_JOINTS, 2), np.nan),
    }
    found = False
    for trial in trials:
        data = load_trial(subject_name, trial["trial_name"])
        if not data:
            continue
        start = int(trial["start_frame"])
        for key in ("OS_landmarks", "OD_landmarks"):
            arr = data.get(key)
            if arr is None:
                continue
            n = min(len(arr), total - start)
            if n > 0:
                out[key][start:start + n] = arr[:n]
                found = True
    return out if found else None


def _tip(landmarks, frame_num: int, bodypart: str):
    """One bodypart's landmark on one frame, or None."""
    joint = BODYPART_JOINTS.get(bodypart)
    if joint is None or landmarks is None:
        return None
    if frame_num < 0 or frame_num >= len(landmarks):
        return None
    point = landmarks[frame_num, joint]
    return None if not np.isfinite(point).all() else point


def recompute_distance_for_frame(subject_name: str, frame_num: int,
                                 manual_labels: dict,
                                 stage_data: dict | None = None) -> float | None:
    """The thumb-index aperture on one frame, from labels plus MediaPipe.

    Millimetres when the subject has a stereo calibration, otherwise
    pixels on the first camera.  Labels win where they exist; MediaPipe
    fills in the camera you have not labeled yet, so the number under the
    video means something before a frame is finished.

    Returns None when neither source can place both tips.

    ``stage_data`` is accepted and ignored: it carried DLC predictions,
    which this app no longer reads.
    """
    settings = get_settings()
    cam_names = settings.camera_names
    calib = get_calibration_for_subject(subject_name)
    mp_data = load_subject_landmarks(subject_name)

    def coords(side_idx: int, bodypart: str):
        side = cam_names[side_idx] if side_idx < len(cam_names) else None
        point = (manual_labels.get(side) or {}).get(bodypart) if side else None
        if point and point[0] is not None:
            return np.asarray(point, dtype=float)
        if mp_data is not None:
            key = "OS_landmarks" if side_idx == 0 else "OD_landmarks"
            return _tip(mp_data.get(key), frame_num, bodypart)
        return None

    left_thumb = coords(0, "thumb")
    left_index = coords(0, "index")

    if calib is not None and len(cam_names) >= 2:
        right_thumb = coords(1, "thumb")
        right_index = coords(1, "index")
        if all(p is not None for p in (left_thumb, left_index,
                                       right_thumb, right_index)):
            try:
                pts = triangulate_points(
                    np.array([left_thumb, left_index]),
                    np.array([right_thumb, right_index]), calib)
            except Exception as e:
                logger.warning("Triangulation failed for %s frame %s: %s",
                               subject_name, frame_num, e)
                return None
            if np.isfinite(pts).all():
                return round(float(np.linalg.norm(pts[0] - pts[1])), 2)
        return None

    # No calibration: the pixel distance on the first camera, which is
    # what the stored traces hold in that case too.
    if left_thumb is not None and left_index is not None:
        return round(float(np.linalg.norm(left_thumb - left_index)), 2)
    return None


def has_mediapipe_data(subject_name: str) -> bool:
    """True when at least one of the subject's trials has landmarks."""
    try:
        trials = build_trial_map(subject_name)
    except Exception:
        return False
    return any(trial_npz_path(subject_name, t["trial_name"]).is_file()
               for t in trials)
