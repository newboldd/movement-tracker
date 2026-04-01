"""Deidentify API: interactive face blur with hand protection."""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import cv2
from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import FileResponse, Response

from ..config import get_settings, JPEG_QUALITY
from ..db import get_db_ctx
from ..services.jobs import registry
from ..services.video import build_trial_map

router = APIRouter(prefix="/api/deidentify", tags=["deidentify"])
logger = logging.getLogger(__name__)


# ── Trials ────────────────────────────────────────────────────────────────

@router.get("/{subject_id}/trials")
def get_trials(subject_id: int) -> dict:
    """List trials with video metadata for a subject."""
    with get_db_ctx() as db:
        subj = db.execute("SELECT * FROM subjects WHERE id = ?", (subject_id,)).fetchone()
        if not subj:
            raise HTTPException(404, "Subject not found")

    camera_mode = subj.get("camera_mode") or "stereo"
    trials = build_trial_map(subj["name"], camera_mode=camera_mode)

    # Check for mediapipe prelabels
    settings = get_settings()
    npz_path = settings.dlc_path / subj["name"] / "mediapipe_prelabels.npz"
    has_mediapipe = npz_path.exists()

    deident_dir = Path(str(settings.video_path)) / "deidentified"

    # Parse no-face trial list from onboarding
    import json as _json
    no_face_raw = subj.get("no_face_videos")
    try:
        no_face_set = set(_json.loads(no_face_raw)) if no_face_raw else set()
    except (ValueError, TypeError):
        no_face_set = set()

    trial_list = []
    for i, t in enumerate(trials):
        # Check if video is stereo
        cap = cv2.VideoCapture(t["video_path"])
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        is_stereo = (fw / fh) > 1.7 if fh > 0 else False

        # Check if deidentified version exists
        cam = t.get("camera_name")
        if cam:
            deident_name = f"{t['trial_name']}_{cam}.mp4"
        else:
            deident_name = f"{t['trial_name']}.mp4"
        has_blurred = (deident_dir / deident_name).exists()

        entry = {
            "trial_idx": i,
            "trial_name": t["trial_name"],
            "video_path": t["video_path"],
            "start_frame": t["start_frame"],
            "end_frame": t["end_frame"],
            "frame_count": t["frame_count"],
            "fps": t["fps"],
            "frame_offset": t.get("frame_offset", 0),
            "is_stereo": is_stereo,
            "frame_width": fw // 2 if is_stereo else fw,
            "frame_height": fh,
            "has_blurred": has_blurred,
            "has_faces": t["trial_name"] not in no_face_set,
        }
        # Include camera info for multicam trials
        cameras = t.get("cameras", [])
        if len(cameras) > 1:
            entry["cameras"] = [{"name": c["name"], "idx": c["idx"]} for c in cameras]
        trial_list.append(entry)

    return {
        "subject": subj,
        "trials": trial_list,
        "has_mediapipe": has_mediapipe,
        "camera_names": settings.camera_names,
    }


# ── Frame serving ─────────────────────────────────────────────────────────

@router.get("/{subject_id}/frame")
def get_frame(
    subject_id: int,
    trial_idx: int = Query(...),
    frame_num: int = Query(...),
    side: str = Query("full"),
    blurred: bool = Query(False),
    preview: bool = Query(False),
    canvas_w: int = Query(700, description="Frontend canvas width for scaling hand mask params"),
) -> Response:
    """Serve a single JPEG frame.

    blurred=true: serve from pre-rendered deidentified video
    preview=true: apply blur live using the same render pipeline (slower but exact match)
    """
    with get_db_ctx() as db:
        subj = db.execute("SELECT * FROM subjects WHERE id = ?", (subject_id,)).fetchone()
        if not subj:
            raise HTTPException(404, "Subject not found")

    subject_name = subj["name"]
    camera_mode = subj.get("camera_mode") or "stereo"
    trials = build_trial_map(subject_name, camera_mode=camera_mode)
    if trial_idx < 0 or trial_idx >= len(trials):
        raise HTTPException(400, f"Invalid trial_idx {trial_idx}")

    trial = trials[trial_idx]
    settings = get_settings()
    cam_names = settings.camera_names

    # Resolve video path
    video_path = trial["video_path"]

    if blurred:
        # Look for deidentified version (matches render output naming)
        deident_dir = Path(str(settings.video_path)) / "deidentified"
        cam = trial.get("camera_name")
        if cam:
            deident_name = f"{trial['trial_name']}_{cam}.mp4"
        else:
            deident_name = f"{trial['trial_name']}.mp4"
        deident_path = deident_dir / deident_name
        if deident_path.exists():
            video_path = str(deident_path)
        else:
            raise HTTPException(404, f"Deidentified video not found: {deident_name}")

    if camera_mode == "multicam" and side != "full" and not blurred:
        for cam in trial.get("cameras", []):
            if cam["name"] == side:
                video_path = cam["path"]
                break

    cap = cv2.VideoCapture(video_path)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    is_stereo = camera_mode == "stereo" and ((fw / fh) > 1.7 if fh > 0 else False)

    # Convert global frame number to local frame within this trial's video
    local_frame = frame_num - trial.get("start_frame", 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, local_frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise HTTPException(400, f"Cannot read frame {frame_num} (local {local_frame})")

    # Preview mode: apply blur using the same pipeline as the full render
    if preview:
        from ..services.deidentify import _build_blur_mask, _build_hand_mask_from_landmarks, _apply_blur_roi
        import numpy as np

        half_w = fw // 2 if is_stereo else fw

        # Load blur specs for this trial
        with get_db_ctx() as db:
            specs = db.execute(
                "SELECT * FROM blur_specs WHERE subject_id = ? AND trial_idx = ?",
                (subject_id, trial_idx),
            ).fetchall()
            face_rows = db.execute(
                "SELECT frame_num, x1, y1, x2, y2, side FROM face_detections WHERE subject_id = ? AND trial_idx = ?",
                (subject_id, trial_idx),
            ).fetchall()
            hs_row = db.execute(
                "SELECT * FROM blur_hand_settings WHERE subject_id = ? AND trial_idx = ?",
                (subject_id, trial_idx),
            ).fetchone()

        active_specs = [dict(s) for s in specs
                        if s["frame_start"] <= frame_num <= s["frame_end"]]

        # Build face detection lookup
        face_by_frame = {}
        for fd in face_rows:
            fn = fd["frame_num"]
            if fn not in face_by_frame:
                face_by_frame[fn] = []
            face_by_frame[fn].append(dict(fd))

        # Load hand settings
        import json as _json
        hand_mask_radius = 5
        hand_smooth = 7
        forearm_radius_val = 10
        forearm_extent_val = 0.4
        hand_smooth2_val = 5
        dlc_radius_val = 15
        hand_segments = []
        if hs_row:
            hand_mask_radius = hs_row.get("hand_mask_radius", 10) or 10
            hand_smooth = hs_row.get("hand_smooth", 10) or 10
            forearm_radius_val = hs_row.get("forearm_radius", 10) or 10
            forearm_extent_val = hs_row.get("forearm_extent", 0.5) or 0.5
            hand_smooth2_val = hs_row.get("hand_smooth2", 0) or 0
            dlc_radius_val = hs_row.get("dlc_radius", 15) or 15
            seg_json = hs_row.get("segments_json", "[]")
            try:
                hand_segments = _json.loads(seg_json) if isinstance(seg_json, str) else (seg_json or [])
            except (ValueError, TypeError):
                hand_segments = []

        # Check if hand protection is active for this frame
        hand_active = False
        active_radius = hand_mask_radius
        active_smooth = hand_smooth
        for seg in hand_segments:
            if seg.get("start", 0) <= frame_num <= seg.get("end", 0):
                hand_active = True
                active_radius = seg.get("radius", hand_mask_radius)
                active_smooth = seg.get("smooth", hand_smooth)
                break

        # Load hand landmarks from npz
        hand_lm_data = {}
        npz_path = settings.dlc_path / subject_name / "mediapipe_prelabels.npz"
        if npz_path.exists() and hand_active:
            npz = np.load(str(npz_path))
            os_lm = npz.get("OS_landmarks")
            od_lm = npz.get("OD_landmarks")
            if os_lm is not None and frame_num < os_lm.shape[0]:
                for j in range(os_lm.shape[1]):
                    x, y = os_lm[frame_num, j, 0], os_lm[frame_num, j, 1]
                    if not np.isnan(x):
                        if frame_num not in hand_lm_data:
                            hand_lm_data[frame_num] = []
                        hand_lm_data[frame_num].append({"x": float(x), "y": float(y), "side": "left", "type": "hand", "joint": j})
            if od_lm is not None and frame_num < od_lm.shape[0]:
                for j in range(od_lm.shape[1]):
                    x, y = od_lm[frame_num, j, 0], od_lm[frame_num, j, 1]
                    if not np.isnan(x):
                        if frame_num not in hand_lm_data:
                            hand_lm_data[frame_num] = []
                        hand_lm_data[frame_num].append({"x": float(x), "y": float(y), "side": "right", "type": "hand", "joint": j})
            # Also load pose landmarks
            pose_path = settings.dlc_path / subject_name / "pose_prelabels.npz"
            if pose_path.exists():
                pose_npz = np.load(str(pose_path))
                for cam_key, side_label in [("OS_pose", "left"), ("OD_pose", "right")]:
                    plm = pose_npz.get(cam_key)
                    if plm is not None and frame_num < plm.shape[0]:
                        for j in range(plm.shape[1]):
                            x, y = plm[frame_num, j, 0], plm[frame_num, j, 1]
                            if not np.isnan(x):
                                if frame_num not in hand_lm_data:
                                    hand_lm_data[frame_num] = []
                                hand_lm_data[frame_num].append({"x": float(x), "y": float(y), "side": side_label, "type": "pose", "joint": j})

        if active_specs:
            if is_stereo:
                left = frame[:, :half_w, :].copy()
                right = frame[:, half_w:, :].copy()
                left_specs = [s for s in active_specs if s.get("side", "left") == "left"]
                right_specs = [s for s in active_specs if s.get("side", "right") == "right"]
                left_mask = _build_blur_mask(left_specs, half_w, fh, frame_num, face_by_frame, "left")
                right_mask = _build_blur_mask(right_specs, fw - half_w, fh, frame_num, face_by_frame, "right")
                hand_mask_l = np.zeros((fh, half_w), dtype=bool)
                hand_mask_r = np.zeros((fh, fw - half_w), dtype=bool)
                if hand_active and frame_num in hand_lm_data:
                    lms_l = [lm for lm in hand_lm_data[frame_num] if lm["side"] == "left"]
                    lms_r = [lm for lm in hand_lm_data[frame_num] if lm["side"] == "right"]
                    hand_mask_l = _build_hand_mask_from_landmarks(lms_l, half_w, fh, active_radius, active_smooth,
                                                                  forearm_radius_val, forearm_extent_val, hand_smooth2_val,
                                                                  dlc_radius=dlc_radius_val, canvas_w=canvas_w)
                    hand_mask_r = _build_hand_mask_from_landmarks(lms_r, fw - half_w, fh, active_radius, active_smooth,
                                                                  forearm_radius_val, forearm_extent_val, hand_smooth2_val,
                                                                  dlc_radius=dlc_radius_val, canvas_w=canvas_w)
                left = _apply_blur_roi(left, left_mask, hand_mask_l)
                right = _apply_blur_roi(right, right_mask, hand_mask_r)
                frame = np.concatenate([left, right], axis=1)
            else:
                blur_mask = _build_blur_mask(active_specs, fw, fh, frame_num, face_by_frame, "full")
                hand_mask = np.zeros((fh, fw), dtype=bool)
                if hand_active and frame_num in hand_lm_data:
                    hand_mask = _build_hand_mask_from_landmarks(
                        hand_lm_data[frame_num], fw, fh, active_radius, active_smooth,
                        forearm_radius_val, forearm_extent_val, hand_smooth2_val,
                        dlc_radius=dlc_radius_val, canvas_w=canvas_w)
                frame = _apply_blur_roi(frame, blur_mask, hand_mask)

    # Stereo: crop to left or right half based on camera name
    if is_stereo and side != "full":
        half_w = fw // 2
        if len(cam_names) >= 2 and side == cam_names[1]:
            frame = frame[:, half_w:, :]
        else:
            frame = frame[:, :half_w, :]

    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return Response(content=jpeg.tobytes(), media_type="image/jpeg")


@router.get("/{subject_id}/video")
def stream_video(
    subject_id: int,
    trial_idx: int = Query(...),
    blurred: bool = Query(False),
) -> FileResponse:
    """Stream a trial's video file for HTML5 playback."""
    with get_db_ctx() as db:
        subj = db.execute("SELECT * FROM subjects WHERE id = ?", (subject_id,)).fetchone()
        if not subj:
            raise HTTPException(404, "Subject not found")

    subject_name = subj["name"]
    camera_mode = subj.get("camera_mode") or "stereo"
    trials = build_trial_map(subject_name, camera_mode=camera_mode)
    if trial_idx < 0 or trial_idx >= len(trials):
        raise HTTPException(400, f"Invalid trial_idx {trial_idx}")

    trial = trials[trial_idx]
    video_path = trial["video_path"]

    if blurred:
        settings = get_settings()
        deident_dir = Path(str(settings.video_path)) / "deidentified"
        cam = trial.get("camera_name")
        if cam:
            deident_name = f"{trial['trial_name']}_{cam}.mp4"
        else:
            deident_name = f"{trial['trial_name']}.mp4"
        deident_path = deident_dir / deident_name
        if deident_path.exists():
            video_path = str(deident_path)
        else:
            raise HTTPException(404, f"Deidentified video not found: {deident_name}")

    # For multicam, resolve per-camera file
    if camera_mode == "multicam" and not blurred:
        cameras = trial.get("cameras", [])
        if cameras:
            video_path = cameras[0]["path"]

    if not Path(video_path).exists():
        raise HTTPException(404, f"Video not found: {video_path}")

    return FileResponse(video_path, media_type="video/mp4", filename=Path(video_path).name)


# ── Face detection ────────────────────────────────────────────────────────

@router.post("/{subject_id}/detect-faces")
def detect_faces(subject_id: int, body: dict = Body(...)) -> dict:
    """Run face detection on a trial's video frames.

    Body: { trial_idx: int }
    Returns per-frame face bounding boxes with temporal smoothing.
    """
    from ..services.deidentify import detect_faces_in_video

    with get_db_ctx() as db:
        subj = db.execute("SELECT * FROM subjects WHERE id = ?", (subject_id,)).fetchone()
        if not subj:
            raise HTTPException(404, "Subject not found")

    trial_idx = body.get("trial_idx", 0)
    camera_mode = subj.get("camera_mode") or "stereo"
    trials = build_trial_map(subj["name"], camera_mode=camera_mode)
    if trial_idx < 0 or trial_idx >= len(trials):
        raise HTTPException(400, f"Invalid trial_idx {trial_idx}")

    trial = trials[trial_idx]
    result = detect_faces_in_video(
        trial["video_path"],
        start_frame=trial["start_frame"],
        frame_count=trial["frame_count"],
    )

    # Save detections to DB
    faces_list = result.get("faces", [])
    with get_db_ctx() as db:
        db.execute(
            "DELETE FROM face_detections WHERE subject_id = ? AND trial_idx = ?",
            (subject_id, trial_idx),
        )
        for entry in faces_list:
            frame_num = int(entry.get("frame", 0))
            for f in entry.get("faces", []):
                db.execute(
                    """INSERT INTO face_detections
                       (subject_id, trial_idx, frame_num, x1, y1, x2, y2, confidence, side)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (subject_id, trial_idx, frame_num,
                     float(f["x1"]), float(f["y1"]), float(f["x2"]), float(f["y2"]),
                     float(f.get("confidence", 1.0)), f.get("side", "full")),
                )

    return result


# ── Load saved face detections ────────────────────────────────────────────

@router.get("/{subject_id}/face-detections")
def get_face_detections(subject_id: int, trial_idx: int = Query(...)) -> dict:
    """Get saved face detections from DB."""
    with get_db_ctx() as db:
        rows = db.execute(
            "SELECT frame_num, x1, y1, x2, y2, confidence, side FROM face_detections "
            "WHERE subject_id = ? AND trial_idx = ? ORDER BY frame_num",
            (subject_id, trial_idx),
        ).fetchall()

    # Group by frame
    by_frame = {}
    for r in rows:
        fn = r["frame_num"]
        if fn not in by_frame:
            by_frame[fn] = []
        by_frame[fn].append({
            "x1": r["x1"], "y1": r["y1"], "x2": r["x2"], "y2": r["y2"],
            "confidence": r["confidence"], "side": r.get("side", "full"),
        })

    faces = []
    for fn in sorted(by_frame.keys()):
        faces.append({"frame": fn, "faces": by_frame[fn]})

    return {"faces": faces}


# ── Hand coverage (which frames have MP data) ─────────────────────────────

@router.get("/{subject_id}/hand-coverage")
def get_hand_coverage(subject_id: int, trial_idx: int = Query(...)) -> dict:
    """Return frame ranges where MediaPipe hand data exists."""
    import numpy as np
    from ..services.mediapipe_prelabel import load_mediapipe_prelabels

    with get_db_ctx() as db:
        subj = db.execute("SELECT * FROM subjects WHERE id = ?", (subject_id,)).fetchone()
        if not subj:
            raise HTTPException(404, "Subject not found")

    data = load_mediapipe_prelabels(subj["name"])
    if data is None:
        return {"frames": []}

    camera_mode = subj.get("camera_mode") or "stereo"
    trials = build_trial_map(subj["name"], camera_mode=camera_mode)
    if trial_idx < 0 or trial_idx >= len(trials):
        return {"frames": []}

    trial = trials[trial_idx]
    start = trial["start_frame"]
    end = trial["end_frame"]

    # Check which frames have non-NaN landmark data in either camera
    os_lm = data["OS_landmarks"]
    od_lm = data["OD_landmarks"]
    frames_with_data = []
    for f in range(start, min(end + 1, len(os_lm))):
        has_os = not np.all(np.isnan(os_lm[f]))
        has_od = not np.all(np.isnan(od_lm[f])) if f < len(od_lm) else False
        if has_os or has_od:
            frames_with_data.append(int(f))

    return {"frames": frames_with_data}


# ── Bulk hand landmarks (all frames from npz) ─────────────────────────────

@router.get("/{subject_id}/hand-landmarks-bulk")
def get_hand_landmarks_bulk(subject_id: int, trial_idx: int = Query(...)) -> dict:
    """Return all hand landmarks for a trial from MediaPipe npz.

    Returns {landmarks: {frame_num: [{x, y, side}]}} for frames with data.
    Uses stored npz data (fast), not live detection.
    """
    import numpy as np
    from ..services.mediapipe_prelabel import load_mediapipe_prelabels, load_pose_prelabels

    with get_db_ctx() as db:
        subj = db.execute("SELECT * FROM subjects WHERE id = ?", (subject_id,)).fetchone()
        if not subj:
            raise HTTPException(404, "Subject not found")

    hand_data = load_mediapipe_prelabels(subj["name"])
    pose_data = load_pose_prelabels(subj["name"])

    if hand_data is None and pose_data is None:
        return {"landmarks": {}, "has_pose": False}

    camera_mode = subj.get("camera_mode") or "stereo"
    trials = build_trial_map(subj["name"], camera_mode=camera_mode)
    if trial_idx < 0 or trial_idx >= len(trials):
        return {"landmarks": {}, "has_pose": pose_data is not None}

    trial = trials[trial_idx]
    start = trial["start_frame"]
    end = trial["end_frame"]
    is_stereo = camera_mode == "stereo"

    result = {}

    # Hand landmarks (21 keypoints per hand)
    if hand_data:
        os_lm = hand_data["OS_landmarks"]
        od_lm = hand_data["OD_landmarks"]
        for f in range(start, min(end + 1, len(os_lm))):
            pts = result.get(str(f), [])
            if not np.all(np.isnan(os_lm[f])):
                for j in range(os_lm.shape[1]):
                    if not np.isnan(os_lm[f, j, 0]):
                        pts.append({
                            "x": round(float(os_lm[f, j, 0]), 1),
                            "y": round(float(os_lm[f, j, 1]), 1),
                            "side": "left" if is_stereo else "full",
                            "type": "hand",
                            "joint": j,
                        })
            if not np.all(np.isnan(od_lm[f])):
                for j in range(od_lm.shape[1]):
                    if not np.isnan(od_lm[f, j, 0]):
                        pts.append({
                            "x": round(float(od_lm[f, j, 0]), 1),
                            "y": round(float(od_lm[f, j, 1]), 1),
                            "side": "right" if is_stereo else "full",
                            "type": "hand",
                            "joint": j,
                        })
            if pts:
                result[str(f)] = pts

    # Pose landmarks: elbows (13,14) only for forearm triangle.
    # Shoulders excluded — not needed for hand protection.
    POSE_ELBOWS = [13, 14]
    if pose_data:
        os_pose = pose_data["OS_pose"]
        od_pose = pose_data["OD_pose"]
        for f in range(start, min(end + 1, len(os_pose))):
            pts = result.get(str(f), [])
            if not np.all(np.isnan(os_pose[f])):
                for j in POSE_ELBOWS:
                    if j < os_pose.shape[1] and not np.isnan(os_pose[f, j, 0]):
                        pts.append({
                            "x": round(float(os_pose[f, j, 0]), 1),
                            "y": round(float(os_pose[f, j, 1]), 1),
                            "side": "left" if is_stereo else "full",
                            "type": "pose",
                            "joint": j,
                        })
            if not np.all(np.isnan(od_pose[f])):
                for j in POSE_ELBOWS:
                    if j < od_pose.shape[1] and not np.isnan(od_pose[f, j, 0]):
                        pts.append({
                            "x": round(float(od_pose[f, j, 0]), 1),
                            "y": round(float(od_pose[f, j, 1]), 1),
                            "side": "right" if is_stereo else "full",
                            "type": "pose",
                            "joint": j,
                        })
            if pts:
                result[str(f)] = pts

    # DLC labels: thumb and index tip from best available version
    # Priority: corrections > labels_v2 > labels_v1
    has_dlc = False
    try:
        from ..services.dlc_predictions import get_dlc_predictions_for_session
        dlc_data = get_dlc_predictions_for_session(subj["name"])
        if dlc_data:
            has_dlc = True
            settings = get_settings()
            cam_names = settings.camera_names
            # DLC data: {camera: {bodypart: [[x,y]|null, ...]}}
            for cam_idx, cam_name in enumerate(cam_names):
                cam_data = dlc_data.get(cam_name, {})
                side_label = "left" if (is_stereo and cam_idx == 0) else ("right" if is_stereo else "full")
                for bp_name, coords_list in cam_data.items():
                    if bp_name not in ("thumb", "index"):
                        continue
                    # DLC joint IDs: thumb=4 (tip), index=8 (tip) matching MediaPipe convention
                    dlc_joint = 4 if bp_name == "thumb" else 8
                    for f_idx, coord in enumerate(coords_list):
                        if coord is None:
                            continue
                        # DLC coords are globally indexed (all trials concatenated)
                        # Only include frames within this trial's range
                        if f_idx < start or f_idx > end:
                            continue
                        key = str(f_idx)
                        if key not in result:
                            result[key] = []
                        result[key].append({
                            "x": round(float(coord[0]), 1),
                            "y": round(float(coord[1]), 1),
                            "side": side_label,
                            "type": "dlc",
                            "joint": dlc_joint,
                            "bodypart": bp_name,
                        })
    except Exception as e:
        logger.debug(f"Could not load DLC labels for {subj['name']}: {e}")

    return {"landmarks": result, "has_pose": pose_data is not None, "has_dlc": has_dlc}


# ── Hand detection (single frame) ─────────────────────────────────────────

@router.post("/{subject_id}/detect-hands")
def detect_hands(subject_id: int, body: dict = Body(...)) -> dict:
    """Run MediaPipe Hands on a single frame.

    Body: { trial_idx: int, frame_num: int }
    """
    from ..services.deidentify import detect_hands_single_frame

    with get_db_ctx() as db:
        subj = db.execute("SELECT * FROM subjects WHERE id = ?", (subject_id,)).fetchone()
        if not subj:
            raise HTTPException(404, "Subject not found")

    trial_idx = body.get("trial_idx", 0)
    frame_num = body.get("frame_num", 0)
    camera_mode = subj.get("camera_mode") or "stereo"
    trials = build_trial_map(subj["name"], camera_mode=camera_mode)
    if trial_idx < 0 or trial_idx >= len(trials):
        raise HTTPException(400, f"Invalid trial_idx")

    trial = trials[trial_idx]
    cap = cv2.VideoCapture(trial["video_path"])
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    is_stereo = (fw / fh) > 1.7 if fh > 0 else False

    return detect_hands_single_frame(trial["video_path"], frame_num, is_stereo)


# ── Blur specs CRUD ───────────────────────────────────────────────────────

@router.get("/{subject_id}/blur-specs")
def get_blur_specs(subject_id: int, trial_idx: int = Query(None)) -> dict:
    """Get saved blur specs for a subject (optionally filtered by trial)."""
    with get_db_ctx() as db:
        if trial_idx is not None:
            rows = db.execute(
                "SELECT * FROM blur_specs WHERE subject_id = ? AND trial_idx = ?",
                (subject_id, trial_idx),
            ).fetchall()
        else:
            rows = db.execute(
                "SELECT * FROM blur_specs WHERE subject_id = ?", (subject_id,),
            ).fetchall()
    return {"specs": [dict(r) for r in rows]}


@router.put("/{subject_id}/blur-specs")
def save_blur_specs(subject_id: int, body: dict = Body(...)) -> dict:
    """Replace all blur specs for a trial.

    Body: { trial_idx: int, specs: [{spot_type, x, y, radius, frame_start, frame_end}] }
    """
    trial_idx = body.get("trial_idx", 0)
    specs = body.get("specs", [])

    with get_db_ctx() as db:
        db.execute(
            "DELETE FROM blur_specs WHERE subject_id = ? AND trial_idx = ?",
            (subject_id, trial_idx),
        )
        for s in specs:
            db.execute(
                """INSERT INTO blur_specs
                   (subject_id, trial_idx, spot_type, x, y, radius, width, height,
                    offset_x, offset_y, frame_start, frame_end, side, shape)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (subject_id, trial_idx, s.get("spot_type", "face"),
                 s["x"], s["y"], s["radius"],
                 s.get("width"), s.get("height"),
                 s.get("offset_x", 0), s.get("offset_y", 0),
                 s["frame_start"], s["frame_end"],
                 s.get("side", "full"), s.get("shape", "oval")),
            )

    return {"status": "ok", "count": len(specs)}


# ── Hand settings ─────────────────────────────────────────────────────────

@router.get("/{subject_id}/hand-settings")
def get_hand_settings(subject_id: int, trial_idx: int = Query(0)) -> dict:
    """Get hand mask settings for a trial."""
    import json as _json
    with get_db_ctx() as db:
        row = db.execute(
            "SELECT * FROM blur_hand_settings WHERE subject_id = ? AND trial_idx = ?",
            (subject_id, trial_idx),
        ).fetchone()
    if row:
        segments = []
        seg_json = row.get("segments_json")
        if seg_json:
            try:
                segments = _json.loads(seg_json)
            except (ValueError, TypeError):
                pass
        # Backward compat: if no segments but old frame_start/end exist, create one
        if not segments and row.get("hand_frame_start") is not None:
            segments = [{"start": row["hand_frame_start"], "end": row["hand_frame_end"],
                         "radius": row["hand_mask_radius"]}]
        return {
            "enabled": bool(row["hand_mask_enabled"]),
            "mask_radius": row["hand_mask_radius"],
            "hand_smooth": row.get("hand_smooth", 10),
            "forearm_radius": row.get("forearm_radius", 10),
            "forearm_extent": row.get("forearm_extent", 0.5),
            "hand_smooth2": row.get("hand_smooth2", 0),
            "dlc_radius": row.get("dlc_radius", 15),
            "hand_temporal": row.get("hand_temporal", 0),
            "show_landmarks": bool(row.get("show_landmarks", 0)),
            "segments": segments,
        }
    return {"enabled": True, "mask_radius": 10, "hand_smooth": 10,
            "forearm_radius": 10, "forearm_extent": 0.5, "hand_smooth2": 0,
            "dlc_radius": 15, "hand_temporal": 0, "show_landmarks": False,
            "segments": []}


@router.put("/{subject_id}/hand-settings")
def save_hand_settings(subject_id: int, body: dict = Body(...)) -> dict:
    """Save hand mask settings for a trial.

    Body: { trial_idx: int, enabled: bool, mask_radius: int, segments: [{start, end, radius}] }
    """
    import json as _json
    trial_idx = body.get("trial_idx", 0)
    enabled = 1 if body.get("enabled", True) else 0
    mask_radius = body.get("mask_radius", 30)
    hand_smooth = body.get("hand_smooth", 10)
    forearm_radius = body.get("forearm_radius", 10)
    forearm_extent = body.get("forearm_extent", 0.5)
    hand_smooth2 = body.get("hand_smooth2", 0)
    dlc_radius_val = body.get("dlc_radius", 15)
    hand_temporal = body.get("hand_temporal", 0)
    show_landmarks = 1 if body.get("show_landmarks", False) else 0
    segments = body.get("segments", [])
    segments_json = _json.dumps(segments) if segments else None

    with get_db_ctx() as db:
        db.execute(
            """INSERT INTO blur_hand_settings
               (subject_id, trial_idx, hand_mask_enabled, hand_mask_radius, hand_smooth,
                forearm_radius, forearm_extent, hand_smooth2, dlc_radius, hand_temporal,
                show_landmarks, segments_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(subject_id, trial_idx)
               DO UPDATE SET hand_mask_enabled=excluded.hand_mask_enabled,
                             hand_mask_radius=excluded.hand_mask_radius,
                             hand_smooth=excluded.hand_smooth,
                             forearm_radius=excluded.forearm_radius,
                             forearm_extent=excluded.forearm_extent,
                             hand_smooth2=excluded.hand_smooth2,
                             dlc_radius=excluded.dlc_radius,
                             hand_temporal=excluded.hand_temporal,
                             show_landmarks=excluded.show_landmarks,
                             segments_json=excluded.segments_json""",
            (subject_id, trial_idx, enabled, mask_radius, hand_smooth,
             forearm_radius, forearm_extent, hand_smooth2, dlc_radius_val,
             hand_temporal, show_landmarks, segments_json),
        )
    return {"status": "ok"}


# ── Render ────────────────────────────────────────────────────────────────

@router.post("/{subject_id}/render")
def render_deidentified(subject_id: int, body: dict = Body(default={})) -> dict:
    """Render deidentified video(s) using saved blur specs.

    Body: { trial_idx: int (optional) }
    Routes through the queue manager so jobs queue up and appear on
    both the Deidentify page and the Processing page.
    Returns {job_id, queue_id} for SSE tracking.
    """
    from ..services.queue_manager import queue_manager

    with get_db_ctx() as db:
        subj = db.execute("SELECT * FROM subjects WHERE id = ?", (subject_id,)).fetchone()
        if not subj:
            raise HTTPException(404, "Subject not found")

    subject_name = subj["name"]
    settings = get_settings()

    # Create job record upfront so the frontend can track via SSE immediately
    with get_db_ctx() as db:
        log_dir = settings.dlc_path / ".logs"
        log_dir.mkdir(exist_ok=True)
        log_path = str(log_dir / f"job_deidentify_{subject_id}.log")
        db.execute(
            """INSERT INTO jobs (subject_id, job_type, status, log_path)
               VALUES (?, 'deidentify', 'queued', ?)""",
            (subject_id, log_path),
        )
        job = db.execute(
            "SELECT id FROM jobs WHERE subject_id = ? AND job_type = 'deidentify' ORDER BY id DESC LIMIT 1",
            (subject_id,),
        ).fetchone()
    job_id = job["id"]

    # Enqueue via queue manager — jobs appear on Processing page and queue behind others
    result = queue_manager.enqueue(
        "deidentify",
        [subject_id],
        [subject_name],
        execution_target="local-cpu",
    )

    return {"job_id": job_id, "queue_id": result.get("queue_id"), "status": "queued"}
