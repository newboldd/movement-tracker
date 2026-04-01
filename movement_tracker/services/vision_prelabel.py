"""Apple Vision framework hand/pose detection.

macOS-only. Falls back gracefully when not available.
Stores results in vision_prelabels.npz alongside mediapipe_prelabels.npz.
"""
from __future__ import annotations

import logging
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Callable

from ..config import get_settings

logger = logging.getLogger(__name__)

# Check availability at import time
try:
    import Vision
    import Quartz
    HAS_VISION = True
except ImportError:
    HAS_VISION = False


# Map Vision joint names → MediaPipe-compatible joint indices (0-20)
VISION_TO_MP = {
    "VNHLKWRI": 0,     # wrist
    "VNHLKTCMC": 1,    # thumb CMC
    "VNHLKTMP": 2,     # thumb MCP
    "VNHLKTIP": 3,     # thumb IP
    "VNHLKTTIP": 4,    # thumb tip
    "VNHLKIMCP": 5,    # index MCP
    "VNHLKIPIP": 6,    # index PIP
    "VNHLKIDIP": 7,    # index DIP
    "VNHLKITIP": 8,    # index tip
    "VNHLKMMCP": 9,    # middle MCP
    "VNHLKMPIP": 10,   # middle PIP
    "VNHLKMDIP": 11,   # middle DIP
    "VNHLKMTIP": 12,   # middle tip
    "VNHLKRMCP": 13,   # ring MCP
    "VNHLKRPIP": 14,   # ring PIP
    "VNHLKRDIP": 15,   # ring DIP
    "VNHLKRTIP": 16,   # ring tip
    "VNHLKPMCP": 17,   # pinky MCP
    "VNHLKPPIP": 18,   # pinky PIP
    "VNHLKPDIP": 19,   # pinky DIP
    "VNHLKPTIP": 20,   # pinky tip
}

N_JOINTS = 21


def is_available() -> bool:
    """Check if Apple Vision framework is available."""
    return HAS_VISION


def _detect_hands_single_frame(image_bgr: np.ndarray) -> list[dict]:
    """Run Vision hand pose on one BGR image.

    Returns list of hands, each: {joint_idx: (x_px, y_px, confidence)}
    """
    if not HAS_VISION:
        return []

    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    provider = Quartz.CGDataProviderCreateWithData(None, rgb.tobytes(), rgb.nbytes, None)
    cg_image = Quartz.CGImageCreate(
        w, h, 8, 24, w * 3,
        Quartz.CGColorSpaceCreateDeviceRGB(),
        Quartz.kCGBitmapByteOrderDefault,
        provider, None, False,
        Quartz.kCGRenderingIntentDefault,
    )
    if cg_image is None:
        return []

    request = Vision.VNDetectHumanHandPoseRequest.alloc().init()
    request.setMaximumHandCount_(2)

    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg_image, {})
    success = handler.performRequests_error_([request], None)
    if not success[0]:
        return []

    results = request.results()
    if not results:
        return []

    hands = []
    for obs in results:
        kp = {}
        for joint_name in obs.availableJointNames():
            pt = obs.recognizedPointForJointName_error_(joint_name, None)
            if pt and pt[0]:
                p = pt[0]
                x_px = p.location().x * w
                y_px = (1.0 - p.location().y) * h
                conf = p.confidence()
                mp_idx = VISION_TO_MP.get(str(joint_name))
                if mp_idx is not None:
                    kp[mp_idx] = (x_px, y_px, conf)
        if kp:
            hands.append(kp)

    return hands


def _pick_best_hand(hands, trial_name, frame_width):
    """Pick the best hand from multiple detections using trial name hint.

    For L trials (left hand tapping), the tapping hand appears on the RIGHT
    side of the camera view (higher x). For R trials, LEFT side (lower x).
    Falls back to highest confidence if trial name doesn't indicate a side.
    """
    if len(hands) == 1:
        return hands[0]

    # Determine expected side from trial name
    expect_right = None
    if trial_name:
        tn = trial_name.upper()
        if "_L" in tn:
            expect_right = True
        elif "_R" in tn:
            expect_right = False

    if expect_right is not None:
        # Compute mean x for each hand detection
        def mean_x(h):
            xs = [x for _, (x, _, _) in h.items() if not np.isnan(x)]
            return np.mean(xs) if xs else frame_width / 2

        if expect_right:
            return max(hands, key=lambda h: mean_x(h))
        else:
            return min(hands, key=lambda h: mean_x(h))

    # Fallback: highest total confidence
    return max(hands, key=lambda h: sum(c for _, _, c in h.values()))


def run_vision_hands(
    subject_name: str,
    progress_callback: Optional[Callable] = None,
) -> dict:
    """Run Apple Vision hand pose on all trials for a subject.

    Saves results to {dlc_path}/{subject}/vision_prelabels.npz.
    Same array format as mediapipe_prelabels.npz for easy comparison.
    """
    if not HAS_VISION:
        raise RuntimeError("Apple Vision framework not available (macOS only)")

    from ..services.video import build_trial_map

    settings = get_settings()
    trials = build_trial_map(subject_name)
    if not trials:
        raise ValueError(f"No trials found for {subject_name}")

    cam_names = settings.camera_names

    all_os = []
    all_od = []
    all_conf_os = []
    all_conf_od = []
    total_all = sum(t["frame_count"] for t in trials)
    frames_done = 0

    for trial in trials:
        video_path = trial["video_path"]
        trial_name = trial.get("trial_name", "")
        n_frames = trial["frame_count"]

        cap = cv2.VideoCapture(video_path)
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        is_stereo = (fw / fh) > 1.7 if fh > 0 else False
        half_w = fw // 2 if is_stereo else fw

        os_lm = np.full((n_frames, N_JOINTS, 2), np.nan)
        od_lm = np.full((n_frames, N_JOINTS, 2), np.nan)
        os_conf = np.full(n_frames, np.nan)
        od_conf = np.full(n_frames, np.nan)

        for i in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                break

            # OS (left) side
            if is_stereo:
                left = frame[:, :half_w, :]
                right = frame[:, half_w:, :]
            else:
                left = frame
                right = None

            hands_l = _detect_hands_single_frame(left)
            if hands_l:
                best = _pick_best_hand(hands_l, trial_name, half_w)
                for j, (x, y, c) in best.items():
                    os_lm[i, j, 0] = x
                    os_lm[i, j, 1] = y
                os_conf[i] = np.mean([c for _, _, c in best.values()])

            # OD (right) side
            if right is not None:
                hands_r = _detect_hands_single_frame(right)
                if hands_r:
                    best = _pick_best_hand(hands_r, trial_name, fw - half_w)
                    for j, (x, y, c) in best.items():
                        od_lm[i, j, 0] = x
                        od_lm[i, j, 1] = y
                    od_conf[i] = np.mean([c for _, _, c in best.values()])

            frames_done += 1
            if progress_callback and i % 10 == 0:
                progress_callback(frames_done / total_all * 100)

        cap.release()
        all_os.append(os_lm)
        all_od.append(od_lm)
        all_conf_os.append(os_conf)
        all_conf_od.append(od_conf)

    OS_landmarks = np.concatenate(all_os, axis=0)
    OD_landmarks = np.concatenate(all_od, axis=0)
    conf_OS = np.concatenate(all_conf_os)
    conf_OD = np.concatenate(all_conf_od)
    total = OS_landmarks.shape[0]

    # Save
    out_dir = settings.dlc_path / subject_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "vision_prelabels.npz"

    np.savez(
        str(out_path),
        OS_landmarks=OS_landmarks,
        OD_landmarks=OD_landmarks,
        confidence_OS=conf_OS,
        confidence_OD=conf_OD,
        total_frames=np.array(total),
    )

    valid_os = np.sum(~np.isnan(OS_landmarks[:, 0, 0]))
    valid_od = np.sum(~np.isnan(OD_landmarks[:, 0, 0]))
    logger.info(f"Vision hands: {subject_name} — OS={valid_os}/{total}, OD={valid_od}/{total}")

    if progress_callback:
        progress_callback(100)

    return {"n_frames": total, "valid_os": int(valid_os), "valid_od": int(valid_od)}
