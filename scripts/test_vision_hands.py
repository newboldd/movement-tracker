#!/usr/bin/env python3
"""Test Apple Vision framework hand pose detection on a sample video.

Compares output to MediaPipe hand landmarks for the same frames.
Saves results to a vision_prelabels.npz alongside the existing mediapipe one.

Usage:
    python scripts/test_vision_hands.py [subject_name] [--frames N]
"""
from __future__ import annotations

import sys
import time
import numpy as np
import cv2

# Vision framework (macOS only)
try:
    import Vision
    import Quartz
    HAS_VISION = True
except ImportError:
    HAS_VISION = False
    print("Apple Vision framework not available (need pyobjc-framework-Vision)")
    sys.exit(1)


def detect_hands_vision(image_bgr: np.ndarray) -> list[dict]:
    """Run Apple Vision hand pose detection on a BGR image.

    Returns list of hand detections, each with 21 keypoints:
    [{joint_name: (x_px, y_px, confidence), ...}, ...]

    Coordinates are in pixel space (top-left origin).
    """
    h, w = image_bgr.shape[:2]

    # Convert BGR to RGB bytes for Vision
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Create CGImage from pixel data
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

    # Create and execute hand pose request
    request = Vision.VNDetectHumanHandPoseRequest.alloc().init()
    request.setMaximumHandCount_(2)

    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
        cg_image, {}
    )

    success = handler.performRequests_error_([request], None)
    if not success[0]:
        return []

    results = request.results()
    if not results:
        return []

    # Map Vision joint names to MediaPipe joint indices (0-20)
    # Vision names: VNHLK{finger}{joint} where finger=T/I/M/R/P, joint=CMC/MCP/PIP/DIP/TIP
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

    hands = []
    for observation in results:
        keypoints = {}
        all_joints = observation.availableJointNames()

        for joint_name in all_joints:
            try:
                point = observation.recognizedPointForJointName_error_(joint_name, None)
                if point and point[0]:
                    pt = point[0]
                    # Vision uses normalized coords (0-1) with origin at BOTTOM-LEFT
                    # Convert to pixel coords with TOP-LEFT origin
                    x_px = pt.location().x * w
                    y_px = (1.0 - pt.location().y) * h  # flip Y
                    conf = pt.confidence()

                    # Map to MediaPipe index
                    joint_key = str(joint_name)
                    mp_idx = VISION_TO_MP.get(joint_key)
                    if mp_idx is not None:
                        keypoints[mp_idx] = (x_px, y_px, conf)
            except Exception:
                continue

        if keypoints:
            hands.append(keypoints)

    return hands


def run_on_video(video_path: str, max_frames: int = 0, side: str = "left") -> tuple:
    """Run Vision hand pose on a video, processing one stereo half.

    Returns (landmarks, confidences) arrays shaped (N, 21, 2) and (N,).
    """
    cap = cv2.VideoCapture(video_path)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    is_stereo = (fw / fh) > 1.7 if fh > 0 else False
    half_w = fw // 2 if is_stereo else fw

    if max_frames > 0:
        total = min(total, max_frames)

    print(f"Video: {fw}x{fh} @ {fps:.0f}fps, {total} frames, stereo={is_stereo}")
    print(f"Processing {side} side ({half_w}x{fh})")

    landmarks = np.full((total, 21, 2), np.nan)
    confidences = np.full(total, np.nan)

    t0 = time.time()
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        # Crop to stereo half
        if is_stereo:
            if side == "right":
                frame = frame[:, half_w:, :]
            else:
                frame = frame[:, :half_w, :]

        hands = detect_hands_vision(frame)

        if hands:
            # Pick the hand with most keypoints (or highest avg confidence)
            best = max(hands, key=lambda h: sum(c for _, _, c in h.values()))
            for joint_idx, (x, y, conf) in best.items():
                landmarks[i, joint_idx, 0] = x
                landmarks[i, joint_idx, 1] = y
            confidences[i] = np.mean([c for _, _, c in best.values()])

        if i % 50 == 0:
            elapsed = time.time() - t0
            fps_actual = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  Frame {i}/{total} ({fps_actual:.1f} fps)")

    cap.release()
    elapsed = time.time() - t0
    valid = np.sum(~np.isnan(landmarks[:, 0, 0]))
    print(f"Done: {valid}/{total} frames detected in {elapsed:.1f}s ({total/elapsed:.1f} fps)")

    return landmarks, confidences


def main():
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Set data dir
    os.environ.setdefault("MT_DATA_DIR", os.path.expanduser("~/data/movement-tracker"))

    from movement_tracker.config import get_settings
    from movement_tracker.services.video import build_trial_map

    subject = sys.argv[1] if len(sys.argv) > 1 else "Con01"
    max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    settings = get_settings()
    trials = build_trial_map(subject)
    if not trials:
        print(f"No trials found for {subject}")
        sys.exit(1)

    print(f"Subject: {subject}, {len(trials)} trials")

    # Process all trials
    all_os = []
    all_od = []
    all_conf_os = []
    all_conf_od = []

    for trial in trials:
        print(f"\n=== {trial['trial_name']} ===")
        video_path = trial["video_path"]
        n_frames = trial["frame_count"] if max_frames == 0 else min(trial["frame_count"], max_frames)

        # OS (left) side
        print(f"--- OS (left) ---")
        os_lm, os_conf = run_on_video(video_path, n_frames, "left")
        all_os.append(os_lm)
        all_conf_os.append(os_conf)

        # OD (right) side
        print(f"--- OD (right) ---")
        od_lm, od_conf = run_on_video(video_path, n_frames, "right")
        all_od.append(od_lm)
        all_conf_od.append(od_conf)

    # Concatenate all trials
    OS_landmarks = np.concatenate(all_os, axis=0)
    OD_landmarks = np.concatenate(all_od, axis=0)
    conf_OS = np.concatenate(all_conf_os)
    conf_OD = np.concatenate(all_conf_od)
    total_frames = OS_landmarks.shape[0]

    # Save as vision_prelabels.npz
    out_dir = settings.dlc_path / subject
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "vision_prelabels.npz"

    np.savez(
        str(out_path),
        OS_landmarks=OS_landmarks,
        OD_landmarks=OD_landmarks,
        confidence_OS=conf_OS,
        confidence_OD=conf_OD,
        total_frames=np.array(total_frames),
    )
    print(f"\nSaved: {out_path} ({os.path.getsize(str(out_path)) / 1024:.0f} KB)")

    # Compare with MediaPipe if available
    mp_path = out_dir / "mediapipe_prelabels.npz"
    if mp_path.exists():
        mp = np.load(str(mp_path))
        mp_os = mp["OS_landmarks"]
        n = min(len(OS_landmarks), len(mp_os))

        # Compare thumb tip (joint 4) and index tip (joint 8)
        for joint, name in [(4, "thumb tip"), (8, "index tip")]:
            mp_valid = ~np.isnan(mp_os[:n, joint, 0])
            vi_valid = ~np.isnan(OS_landmarks[:n, joint, 0])
            both = mp_valid & vi_valid

            if both.sum() > 0:
                dx = OS_landmarks[:n, joint, 0][both] - mp_os[:n, joint, 0][both]
                dy = OS_landmarks[:n, joint, 1][both] - mp_os[:n, joint, 1][both]
                dist = np.sqrt(dx**2 + dy**2)
                print(f"\n{name} (OS): {both.sum()} frames with both detections")
                print(f"  Mean distance: {np.mean(dist):.1f} px")
                print(f"  Median: {np.median(dist):.1f} px")
                print(f"  95th percentile: {np.percentile(dist, 95):.1f} px")


if __name__ == "__main__":
    main()
