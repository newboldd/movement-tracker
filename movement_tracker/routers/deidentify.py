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
    from ..services.mediapipe_prelabel import has_mediapipe_data
    has_mediapipe = has_mediapipe_data(subj["name"])
    pose_path = settings.dlc_path / subj["name"] / "pose_prelabels.npz"
    has_pose = pose_path.exists()

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
        deident_path = deident_dir / deident_name
        has_blurred = deident_path.exists() and deident_path.stat().st_size > 10000  # >10KB = valid

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
        "has_pose": has_pose,
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
        if deident_path.exists() and deident_path.stat().st_size > 4096:
            # Don't serve a file that's currently being written by a render job
            with get_db_ctx() as _db2:
                rendering = _db2.execute(
                    "SELECT id FROM jobs WHERE job_type = 'deidentify' "
                    "AND status = 'running' AND subject_id = ?",
                    (subject_id,),
                ).fetchone()
            if not rendering:
                video_path = str(deident_path)
            # else: rendering in progress — fall through and serve original video
        elif not deident_path.exists():
            raise HTTPException(404, f"Deidentified video not found: {deident_name}")
        # else: file exists but too small/corrupt — fall through to original

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
        from ..services.deidentify import (
            _build_blur_mask, _build_hand_mask_from_landmarks, _apply_blur_roi,
            build_hand_mask_from_hrnet_mip,
        )
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

        # Load camera trajectory if any active spot wants motion compensation
        trajectory = None
        if any(s.get("motion_compensate") for s in active_specs):
            from ..services.camera_motion import load_camera_trajectory
            trajectory = load_camera_trajectory(subject_name, trial["trial_name"])
            if trajectory is not None:
                # The H-stack is trial-local (0..n_frames-1). Index by the trial's
                # current authoritative start_frame, not the offset baked in at
                # compute time (which drifts if the subject's trial set changed).
                trajectory["start_frame"] = int(trial.get("start_frame", 0))

        # Build face detection lookup
        face_by_frame = {}
        for fd in face_rows:
            fn = fd["frame_num"]
            if fn not in face_by_frame:
                face_by_frame[fn] = []
            face_by_frame[fn].append(dict(fd))

        # Load hand settings.  Only the 5 user-exposed sliders are
        # honoured -- forearm_radius / hand_smooth2 / dlc_radius were
        # removed (never reachable from the UI; dlc-tip circles now
        # share the MP hand_mask_radius).
        import json as _json
        hand_mask_radius = 5
        hand_smooth = 7
        forearm_extent_val = 0.4
        hand_segments = []
        mask_source = "mediapipe"
        hrnet_mask_thresh = 0.30
        hrnet_mask_smooth = 7
        arm_dorsal_dilate_val = 0
        arm_ventral_dilate_val = 0
        if hs_row:
            hand_mask_radius = hs_row.get("hand_mask_radius", 10) or 10
            hand_smooth = hs_row.get("hand_smooth", 10) or 10
            forearm_extent_val = hs_row.get("forearm_extent", 0.5) or 0.5
            arm_dorsal_dilate_val = hs_row.get("arm_dorsal_dilate", 0) or 0
            arm_ventral_dilate_val = hs_row.get("arm_ventral_dilate", 0) or 0
            seg_json = hs_row.get("segments_json", "[]")
            try:
                hand_segments = _json.loads(seg_json) if isinstance(seg_json, str) else (seg_json or [])
            except (ValueError, TypeError):
                hand_segments = []
            # HRnet alternative hand-mask source — see services.deidentify.
            mask_source = hs_row.get("mask_source") or "mediapipe"
            if mask_source not in ("mediapipe", "hrnet"):
                mask_source = "mediapipe"
            try:
                hrnet_mask_thresh = float(hs_row.get("hrnet_mask_thresh", 0.30) or 0.30)
            except (TypeError, ValueError):
                hrnet_mask_thresh = 0.30
            try:
                hrnet_mask_smooth = int(hs_row.get("hrnet_mask_smooth", 7) or 7)
            except (TypeError, ValueError):
                hrnet_mask_smooth = 7
        else:
            # No DB row — mirror the client-side default behaviour: cover
            # the whole trial with a per-camera segment.  Without this,
            # the preview blur skipped hand protection while the Masks
            # view showed the auto-generated default mask, making the
            # two appear inconsistent for any never-configured subject
            # (e.g. MSA10 — which had no blur_hand_settings row at all).
            _t_start = trial.get("start_frame", 0)
            _t_end = _t_start + trial.get("frame_count", 0) - 1
            if is_stereo:
                hand_segments = [
                    {"start": _t_start, "end": _t_end,
                     "radius": hand_mask_radius, "smooth": hand_smooth,
                     "side": "left"},
                    {"start": _t_start, "end": _t_end,
                     "radius": hand_mask_radius, "smooth": hand_smooth,
                     "side": "right"},
                ]
            else:
                hand_segments = [{
                    "start": _t_start, "end": _t_end,
                    "radius": hand_mask_radius, "smooth": hand_smooth,
                    "side": "full",
                }]

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

        # Load hand landmarks via the SAME helper the bulk-landmarks
        # endpoint uses (``load_mediapipe_prelabels``) so the preview
        # respects the per-subject pre-roll trim from ``_detect_frame_offset``.
        # The previous direct ``np.load`` skipped this trim — for any
        # subject built on a machine that exposed negative-PTS pre-roll
        # frames (e.g. MSA10), ``os_lm[frame_num]`` actually read frame
        # ``frame_num + offset``, so the preview's hand mask was anchored
        # to the wrong frame and looked completely unrelated to the green
        # outline shown in the Masks view.
        hand_lm_data = {}
        from ..services.mediapipe_prelabel import (
            load_mediapipe_prelabels, load_mediapipe_cropped_prelabels,
            load_pose_prelabels,
        )
        # Fall back to the bbox-cropped pass when no plain forward run
        # exists (a Forward MP run with a crop box saves only
        # mediapipe_cropped.npz), so cropped-only subjects still get a mask.
        _hand_npz = (load_mediapipe_prelabels(subject_name)
                     or load_mediapipe_cropped_prelabels(subject_name)) \
            if hand_active else None
        if _hand_npz is not None and hand_active:
            os_lm = _hand_npz.get("OS_landmarks")
            od_lm = _hand_npz.get("OD_landmarks")
            # Interpolate NaN gaps inside this trial's frame range so the
            # hand protection mask stays active on every frame, including
            # frames where MediaPipe lost detection.
            from ..services.deidentify import interpolate_landmarks_inplace
            _t_start = trial.get("start_frame", 0)
            _t_end = _t_start + trial.get("frame_count", 0) - 1
            # Snapshot NaN mask BEFORE interpolation so each landmark dict
            # below can be tagged ``interpolated: True`` when MediaPipe
            # originally lost detection for that (frame, joint).  The
            # DLC tip-disagreement path uses this flag to also fire when
            # the MP tip on this frame was filled in from neighbours.
            _os_was_nan = None
            _od_was_nan = None
            if os_lm is not None:
                os_lm = np.array(os_lm, copy=True)
                _os_was_nan = np.isnan(os_lm[..., 0]).copy()
                interpolate_landmarks_inplace(os_lm, _t_start, _t_end)
            if od_lm is not None:
                od_lm = np.array(od_lm, copy=True)
                _od_was_nan = np.isnan(od_lm[..., 0]).copy()
                interpolate_landmarks_inplace(od_lm, _t_start, _t_end)
            if os_lm is not None and frame_num < os_lm.shape[0]:
                for j in range(os_lm.shape[1]):
                    x, y = os_lm[frame_num, j, 0], os_lm[frame_num, j, 1]
                    if not np.isnan(x):
                        if frame_num not in hand_lm_data:
                            hand_lm_data[frame_num] = []
                        was_interp = bool(_os_was_nan is not None and _os_was_nan[frame_num, j])
                        hand_lm_data[frame_num].append({"x": float(x), "y": float(y), "side": "left", "type": "hand", "joint": j, "interpolated": was_interp})
            if od_lm is not None and frame_num < od_lm.shape[0]:
                for j in range(od_lm.shape[1]):
                    x, y = od_lm[frame_num, j, 0], od_lm[frame_num, j, 1]
                    if not np.isnan(x):
                        if frame_num not in hand_lm_data:
                            hand_lm_data[frame_num] = []
                        was_interp = bool(_od_was_nan is not None and _od_was_nan[frame_num, j])
                        hand_lm_data[frame_num].append({"x": float(x), "y": float(y), "side": "right", "type": "hand", "joint": j, "interpolated": was_interp})
            # Also load pose landmarks via the offset-aware helper, then
            # interpolate the trial's slice so the forearm-triangle
            # elbow stays anchored when MediaPipe Pose drops out.
            pose_npz_data = load_pose_prelabels(subject_name)
            if pose_npz_data is not None:
                _pose_interpolated = {}
                for cam_key in ("OS_pose", "OD_pose"):
                    _src = pose_npz_data.get(cam_key)
                    if _src is not None:
                        _src = np.array(_src, copy=True)
                        interpolate_landmarks_inplace(_src, _t_start, _t_end)
                    _pose_interpolated[cam_key] = _src
                for cam_key, side_label in [("OS_pose", "left"), ("OD_pose", "right")]:
                    plm = _pose_interpolated.get(cam_key)
                    if plm is not None and frame_num < plm.shape[0]:
                        for j in range(plm.shape[1]):
                            x, y = plm[frame_num, j, 0], plm[frame_num, j, 1]
                            if not np.isnan(x):
                                if frame_num not in hand_lm_data:
                                    hand_lm_data[frame_num] = []
                                hand_lm_data[frame_num].append({"x": float(x), "y": float(y), "side": side_label, "type": "pose", "joint": j})
            # Also load DLC labels (thumb tip / index tip, plus any other
            # bodyparts).  ``_build_hand_mask_from_landmarks`` merges
            # type=='dlc' entries: matching joints REPLACE the MP joint
            # coordinates (more accurate fingertip positions); novel
            # joints are added on top.  Without this load, the preview
            # blur would diverge from the live overlay — the overlay
            # uses DLC tips (via a separate /landmarks endpoint), the
            # preview was stuck on raw MP.
            try:
                from ..services.dlc_predictions import get_dlc_predictions_for_session
                _dlc_data = get_dlc_predictions_for_session(subject_name)
            except Exception:
                _dlc_data = None
            if _dlc_data:
                _cam_names = settings.camera_names or []
                # DLC data layout: {camera_name: {bodypart: [[x, y]|None, ...]}}
                for _cam_idx, _cam_name in enumerate(_cam_names):
                    _cam = _dlc_data.get(_cam_name, {})
                    if is_stereo:
                        _side = "left" if _cam_idx == 0 else "right"
                    else:
                        _side = "full"
                    for _bp, _coords in _cam.items():
                        # Map known body parts to MP joint indices.
                        # Extend as DLC dictionary grows (e.g. nailbed,
                        # extra fingertip markers).
                        _joint = None
                        _bp_l = _bp.lower()
                        if _bp_l == "thumb":
                            _joint = 4
                        elif _bp_l == "index":
                            _joint = 8
                        elif _bp_l == "middle":
                            _joint = 12
                        elif _bp_l == "ring":
                            _joint = 16
                        elif _bp_l == "pinky":
                            _joint = 20
                        if _joint is None:
                            continue
                        if frame_num < 0 or frame_num >= len(_coords):
                            continue
                        _c = _coords[frame_num]
                        if _c is None:
                            continue
                        if frame_num not in hand_lm_data:
                            hand_lm_data[frame_num] = []
                        hand_lm_data[frame_num].append({
                            "x": float(_c[0]), "y": float(_c[1]),
                            "side": _side, "type": "dlc", "joint": _joint,
                            "bodypart": _bp,
                        })

        if active_specs:
            if is_stereo:
                left = frame[:, :half_w, :].copy()
                right = frame[:, half_w:, :].copy()
                left_specs = [s for s in active_specs if s.get("side", "left") == "left"]
                right_specs = [s for s in active_specs if s.get("side", "right") == "right"]
                left_mask = _build_blur_mask(left_specs, half_w, fh, frame_num, face_by_frame, "left", trajectory)
                right_mask = _build_blur_mask(right_specs, fw - half_w, fh, frame_num, face_by_frame, "right", trajectory)
                hand_mask_l = np.zeros((fh, half_w), dtype=bool)
                hand_mask_r = np.zeros((fh, fw - half_w), dtype=bool)
                if hand_active:
                    if mask_source == "hrnet":
                        # HRnet MIP path — same code the full renderer
                        # uses.  Reads the per-frame bbox from hand_crop
                        # .json so the mask follows the hand even when MP
                        # labels are wrong or missing.
                        m_l = build_hand_mask_from_hrnet_mip(
                            subject_name, trial_idx, frame_num,
                            "left", half_w, fh,
                            hrnet_mask_thresh, int(round(hrnet_mask_smooth * 2)),
                        )
                        m_r = build_hand_mask_from_hrnet_mip(
                            subject_name, trial_idx, frame_num,
                            "right", fw - half_w, fh,
                            hrnet_mask_thresh, int(round(hrnet_mask_smooth * 2)),
                        )
                        if m_l is not None: hand_mask_l = m_l
                        if m_r is not None: hand_mask_r = m_r
                    elif frame_num in hand_lm_data:
                        lms_l = [lm for lm in hand_lm_data[frame_num] if lm["side"] == "left"]
                        lms_r = [lm for lm in hand_lm_data[frame_num] if lm["side"] == "right"]
                        hand_mask_l = _build_hand_mask_from_landmarks(
                            lms_l, half_w, fh, active_radius, active_smooth,
                            forearm_extent=forearm_extent_val,
                            arm_dorsal_dilate=arm_dorsal_dilate_val,
                            arm_ventral_dilate=arm_ventral_dilate_val)
                        hand_mask_r = _build_hand_mask_from_landmarks(
                            lms_r, fw - half_w, fh, active_radius, active_smooth,
                            forearm_extent=forearm_extent_val,
                            arm_dorsal_dilate=arm_dorsal_dilate_val,
                            arm_ventral_dilate=arm_ventral_dilate_val)
                left = _apply_blur_roi(left, left_mask, hand_mask_l)
                right = _apply_blur_roi(right, right_mask, hand_mask_r)
                frame = np.concatenate([left, right], axis=1)
            else:
                blur_mask = _build_blur_mask(active_specs, fw, fh, frame_num, face_by_frame, "full", trajectory)
                hand_mask = np.zeros((fh, fw), dtype=bool)
                if hand_active:
                    if mask_source == "hrnet":
                        m = build_hand_mask_from_hrnet_mip(
                            subject_name, trial_idx, frame_num,
                            "full", fw, fh,
                            hrnet_mask_thresh, int(round(hrnet_mask_smooth * 2)),
                        )
                        if m is not None: hand_mask = m
                    elif frame_num in hand_lm_data:
                        hand_mask = _build_hand_mask_from_landmarks(
                            hand_lm_data[frame_num], fw, fh, active_radius, active_smooth,
                            forearm_extent=forearm_extent_val,
                            arm_dorsal_dilate=arm_dorsal_dilate_val,
                            arm_ventral_dilate=arm_ventral_dilate_val)
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
        if deident_path.exists() and deident_path.stat().st_size > 4096:
            # Don't serve a file that's currently being written by a render job
            with get_db_ctx() as _db2:
                rendering = _db2.execute(
                    "SELECT id FROM jobs WHERE job_type = 'deidentify' "
                    "AND status = 'running' AND subject_id = ?",
                    (subject_id,),
                ).fetchone()
            if not rendering:
                video_path = str(deident_path)
            # else: rendering in progress — fall through and serve original video
        elif not deident_path.exists():
            raise HTTPException(404, f"Deidentified video not found: {deident_name}")
        # else: file exists but too small/corrupt — fall through to original

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


# ── Camera trajectory (for motion-compensated blur spots) ─────────────────

@router.get("/{subject_id}/camera-trajectory")
def get_camera_trajectory(subject_id: int, trial_idx: int = Query(...)) -> dict:
    """Return per-frame homographies H_to_ref for OS/OD halves.

    When available, the client can warp custom blur spot positions
    (stored in reference-frame coords) into the current frame.
    """
    from ..services.camera_motion import load_camera_trajectory

    with get_db_ctx() as db:
        subj = db.execute("SELECT name FROM subjects WHERE id = ?", (subject_id,)).fetchone()
        if not subj:
            raise HTTPException(404, "Subject not found")
    name = subj["name"]
    trials = build_trial_map(name)
    if trial_idx < 0 or trial_idx >= len(trials):
        raise HTTPException(400, "trial_idx out of range")
    trial_stem = trials[trial_idx]["trial_name"]
    traj = load_camera_trajectory(name, trial_stem)
    if traj is None:
        return {"available": False, "trial_stem": trial_stem}
    return {
        "available": True,
        "trial_stem": trial_stem,
        "reference_frame": int(traj["reference_frame"]),
        "n_frames": int(traj["n_frames"]),
        "is_stereo": bool(traj["is_stereo"]),
        "start_frame": int(traj["start_frame"]),
        "H_to_ref_L": traj["H_to_ref_L"].reshape(int(traj["n_frames"]), 9).tolist(),
        "H_to_ref_R": traj["H_to_ref_R"].reshape(int(traj["n_frames"]), 9).tolist(),
    }


# ── Hand coverage (which frames have MP data) ─────────────────────────────

@router.get("/{subject_id}/hand-coverage")
def get_hand_coverage(subject_id: int, trial_idx: int = Query(...)) -> dict:
    """Return frame ranges where MediaPipe hand data exists."""
    import numpy as np
    from ..services.mediapipe_prelabel import (
        load_mediapipe_prelabels, load_mediapipe_cropped_prelabels,
    )

    with get_db_ctx() as db:
        subj = db.execute("SELECT * FROM subjects WHERE id = ?", (subject_id,)).fetchone()
        if not subj:
            raise HTTPException(404, "Subject not found")

    data = (load_mediapipe_prelabels(subj["name"])
            or load_mediapipe_cropped_prelabels(subj["name"]))
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
def get_hand_landmarks_bulk(subject_id: int, trial_idx: int = Query(...),
                              source: str = Query("all")) -> dict:
    """Return all hand landmarks for a trial from MediaPipe npz(s).

    ``source`` controls which MediaPipe pass(es) feed the protection
    mask.  DLC's thumb/index tips are always added on top regardless:
      - ``all``      → forward + reverse + static
      - ``auto``     → combined if available, else forward
      - ``combined`` → combined (falls back to forward if no combined)
      - ``forward``  → forward only

    Returns {landmarks: {frame_num: [{x, y, side}]}} for frames with data.
    Uses stored npz data (fast), not live detection.
    """
    import numpy as np
    from ..services.mediapipe_prelabel import (
        load_mediapipe_prelabels,
        load_mediapipe_cropped_prelabels,
        load_mediapipe_reverse_prelabels,
        load_mediapipe_static_prelabels,
        load_mediapipe_combined_prelabels,
        load_pose_prelabels,
    )

    with get_db_ctx() as db:
        subj = db.execute("SELECT * FROM subjects WHERE id = ?", (subject_id,)).fetchone()
        if not subj:
            raise HTTPException(404, "Subject not found")

    # Build the list of MP sources to merge into the protection mask.
    # Each entry contributes its own OS/OD hand landmarks; the client's
    # morphological-close step naturally fuses overlapping circles, so
    # duplicate joints across passes are harmless (and additive — they
    # widen the protected region where multiple passes agree).
    src = (source or "all").lower()
    sources: list[dict] = []
    if src == "all":
        for ld in (load_mediapipe_prelabels,
                   load_mediapipe_cropped_prelabels,
                   load_mediapipe_reverse_prelabels,
                   load_mediapipe_static_prelabels):
            d = ld(subj["name"])
            if d is not None:
                sources.append(d)
    elif src == "combined":
        d = load_mediapipe_combined_prelabels(subj["name"]) \
            or load_mediapipe_prelabels(subj["name"]) \
            or load_mediapipe_cropped_prelabels(subj["name"])
        if d is not None:
            sources.append(d)
    elif src == "forward":
        # "forward" means the un-reversed pass; a crop box routes that
        # pass to mediapipe_cropped.npz, so fall back to it.
        d = load_mediapipe_prelabels(subj["name"]) \
            or load_mediapipe_cropped_prelabels(subj["name"])
        if d is not None:
            sources.append(d)
    else:  # auto
        d = load_mediapipe_combined_prelabels(subj["name"]) \
            or load_mediapipe_prelabels(subj["name"]) \
            or load_mediapipe_cropped_prelabels(subj["name"])
        if d is not None:
            sources.append(d)

    # Primary "hand_data" handle kept for downstream pose / has_pose
    # logic and the interpolation pass.  Defaults to the first MP
    # source we found (or None if no MP at all).
    hand_data = sources[0] if sources else None
    pose_data = load_pose_prelabels(subj["name"])

    if hand_data is None and pose_data is None:
        return {"landmarks": {}, "has_pose": False, "has_dlc": False}

    camera_mode = subj.get("camera_mode") or "stereo"
    trials = build_trial_map(subj["name"], camera_mode=camera_mode)
    if trial_idx < 0 or trial_idx >= len(trials):
        return {"landmarks": {}, "has_pose": pose_data is not None}

    trial = trials[trial_idx]
    start = trial["start_frame"]
    end = trial["end_frame"]
    is_stereo = camera_mode == "stereo"

    # Interpolate NaN gaps inside this trial's frame range so every
    # frame has hand-landmark values for the protection mask, even on
    # frames where MediaPipe failed to detect.  Restricted to the trial
    # range so we don't bridge across trial boundaries.
    from ..services.deidentify import interpolate_landmarks_inplace
    for sd in sources:
        interpolate_landmarks_inplace(sd.get("OS_landmarks"), start, end)
        interpolate_landmarks_inplace(sd.get("OD_landmarks"), start, end)
    if pose_data:
        interpolate_landmarks_inplace(pose_data.get("OS_pose"), start, end)
        interpolate_landmarks_inplace(pose_data.get("OD_pose"), start, end)

    result = {}

    # Hand landmarks (21 keypoints per hand).  Iterate every selected
    # MP source and append all its keypoints — overlapping circles are
    # fused by the morphological-close step downstream, so duplicate
    # joints across passes just reinforce the protection region.
    for sd in sources:
        os_lm = sd.get("OS_landmarks")
        od_lm = sd.get("OD_landmarks")
        if os_lm is None or od_lm is None:
            continue
        last_f = min(end + 1, len(os_lm), len(od_lm))
        for f in range(start, last_f):
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


@router.get("/{subject_id}/hrnet-mask-data")
def get_hrnet_mask_data(subject_id: int, trial_idx: int = Query(...)) -> dict:
    """Return HRnet MIP + per-frame bboxes for client-side mask building.

    Used by the deidentify page when ``mask_source = "hrnet"`` so the green-
    outline preview can update instantly when the threshold/smooth sliders
    move (no per-slider HTTP round-trip).

    Returns either ``{"available": true, ...}`` with the MIP + bbox arrays
    serialised as base64-encoded float16 / float32 buffers, or
    ``{"available": false, "reason": "..."}`` when no HRnet output exists
    for the requested trial.
    """
    import base64
    import json as _json
    import numpy as np

    from ..config import get_settings
    from ..services.video import build_trial_map
    from ..services.skeleton_data import _skeleton_dir

    with get_db_ctx() as db:
        subj = db.execute("SELECT name FROM subjects WHERE id=?", (subject_id,)).fetchone()
    if not subj:
        raise HTTPException(404, "Subject not found")
    name = subj["name"]
    trials = build_trial_map(name)
    if trial_idx >= len(trials):
        raise HTTPException(404, "trial_idx out of range")
    stem = trials[trial_idx]["trial_name"]
    n_frames = int(trials[trial_idx]["frame_count"])
    start_frame = int(trials[trial_idx].get("start_frame", 0))

    mip_path = _skeleton_dir(name) / stem / "hrnet_w18_mip.npz"
    crop_path = _skeleton_dir(name) / stem / "hand_crop.json"
    if not mip_path.exists():
        return {"available": False, "reason": f"No HRnet MIP for {stem}"}
    if not crop_path.exists():
        return {"available": False, "reason": f"No hand_crop.json for {stem}"}

    try:
        npz = np.load(mip_path)
        mip_L = npz["heatmaps_L_mip"] if "heatmaps_L_mip" in npz.files else None
        mip_R = npz["heatmaps_R_mip"] if "heatmaps_R_mip" in npz.files else None
    except Exception as e:
        return {"available": False, "reason": f"Failed to load MIP: {e}"}

    try:
        with open(crop_path) as f:
            crop = _json.load(f)
    except Exception as e:
        return {"available": False, "reason": f"Failed to load hand_crop.json: {e}"}

    # Per-frame bboxes — fall back to broadcasting union when missing.
    def _bbox_pf(side_key: str) -> list[list[float]]:
        pf_key = f"{side_key}_perframe"
        arr = crop.get(pf_key)
        if isinstance(arr, list) and arr and isinstance(arr[0], list) and len(arr[0]) == 4:
            return arr
        union = crop.get(side_key)
        if isinstance(union, list) and len(union) == 4:
            return [union] * n_frames
        return []

    bbox_L = _bbox_pf("crop_L")
    bbox_R = _bbox_pf("crop_R")

    def _b64_arr(a) -> str:
        if a is None:
            return ""
        return base64.b64encode(np.ascontiguousarray(a, dtype=np.float16).tobytes()).decode("ascii")

    return {
        "available": True,
        "n_frames": n_frames,
        "start_frame": start_frame,
        "mip_size": 64,
        "mip_L_b64": _b64_arr(mip_L),
        "mip_L_shape": list(mip_L.shape) if mip_L is not None else None,
        "mip_R_b64": _b64_arr(mip_R),
        "mip_R_shape": list(mip_R.shape) if mip_R is not None else None,
        "bbox_L": bbox_L,
        "bbox_R": bbox_R,
    }


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
                    offset_x, offset_y, frame_start, frame_end, side, shape,
                    motion_compensate)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (subject_id, trial_idx, s.get("spot_type", "face"),
                 s["x"], s["y"], s["radius"],
                 s.get("width"), s.get("height"),
                 s.get("offset_x", 0), s.get("offset_y", 0),
                 s["frame_start"], s["frame_end"],
                 s.get("side", "full"), s.get("shape", "oval"),
                 1 if s.get("motion_compensate") else 0),
            )

    return {"status": "ok", "count": len(specs)}


# Same handler for POST (used by navigator.sendBeacon on page unload)
@router.post("/{subject_id}/blur-specs")
def save_blur_specs_post(subject_id: int, body: dict = Body(...)) -> dict:
    return save_blur_specs(subject_id, body)


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
            "has_row": True,
            "enabled": bool(row["hand_mask_enabled"]),
            "mask_radius": row["hand_mask_radius"],
            "hand_smooth": row.get("hand_smooth", 10),
            "forearm_extent": row.get("forearm_extent", 0.5),
            "hand_temporal": row.get("hand_temporal", 0),
            "show_landmarks": bool(row.get("show_landmarks", 0)),
            "mask_source": "mediapipe",
            "arm_dorsal_dilate": row.get("arm_dorsal_dilate", 0) or 0,
            "arm_ventral_dilate": row.get("arm_ventral_dilate", 0) or 0,
            "segments": segments,
        }
    # No DB row — trial has never been configured
    return {"has_row": False, "enabled": True, "mask_radius": 10, "hand_smooth": 10,
            "forearm_extent": 0.5,
            "hand_temporal": 0, "show_landmarks": False,
            "mask_source": "mediapipe", "arm_dorsal_dilate": 0, "arm_ventral_dilate": 0,
            "segments": []}


@router.get("/{subject_id}/render-params")
def get_render_params(subject_id: int, trial_idx: int = Query(...)) -> dict:
    """Return the params sidecar JSON for this trial's deidentified
    output video, or ``{"status": "unknown"}`` when no sidecar exists.

    Looks alongside ``<videos>/deidentified/<trial_name>.mp4`` (single)
    or ``<trial_name>_<cam>.mp4`` (multicam) for a matching
    ``.params.json``.  The Deidentify page reads this on trial load
    to surface a divergence indicator when current slider values
    differ from the values that produced the existing render.
    """
    import json as _json
    settings = get_settings()
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
    deident_dir = Path(str(settings.video_path)) / "deidentified"
    # Try both naming conventions: single (Trial.params.json) and
    # multicam (Trial_<cam>.params.json) -- return the first hit so
    # the page works for both layouts.
    candidates = [deident_dir / f"{trial['trial_name']}.params.json"]
    cam_names = settings.camera_names or []
    for cam in cam_names:
        candidates.append(deident_dir / f"{trial['trial_name']}_{cam}.params.json")
    for sidecar in candidates:
        if sidecar.exists():
            try:
                with open(sidecar) as f:
                    return {"status": "ok", "params": _json.load(f),
                            "path": str(sidecar)}
            except (OSError, ValueError) as e:
                return {"status": "unknown", "error": str(e)}
    return {"status": "unknown"}


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
    forearm_extent = body.get("forearm_extent", 0.5)
    # forearm_radius / hand_smooth2 / dlc_radius are no longer
    # user-controlled.  Keep the columns in the INSERT for back-compat
    # but hardcode constants -- they don't influence the renderer
    # anymore (DLC tips draw at the MP hand_mask_radius now, the
    # 8-stacked threshold replaced any need for a separate smooth2,
    # and the arm-edge dorsal/ventral sliders superseded
    # forearm_radius).
    _forearm_radius_legacy = 10
    _hand_smooth2_legacy = 0
    _dlc_radius_legacy = 0
    hand_temporal = body.get("hand_temporal", 0)
    show_landmarks = 1 if body.get("show_landmarks", False) else 0
    segments = body.get("segments", [])
    segments_json = _json.dumps(segments) if segments else None
    mask_source = "mediapipe"
    try:
        arm_dorsal_dilate = int(body.get("arm_dorsal_dilate", 0) or 0)
    except (TypeError, ValueError):
        arm_dorsal_dilate = 0
    try:
        arm_ventral_dilate = int(body.get("arm_ventral_dilate", 0) or 0)
    except (TypeError, ValueError):
        arm_ventral_dilate = 0

    with get_db_ctx() as db:
        # Lazily add the new columns when this is the first save under
        # the new schema (avoids a startup migration just for these).
        cols = {r["name"] for r in db.execute("PRAGMA table_info(blur_hand_settings)").fetchall()}
        if "mask_source" not in cols:
            db.execute("ALTER TABLE blur_hand_settings ADD COLUMN mask_source TEXT")
        if "arm_dorsal_dilate" not in cols:
            db.execute("ALTER TABLE blur_hand_settings ADD COLUMN arm_dorsal_dilate INTEGER DEFAULT 0")
        if "arm_ventral_dilate" not in cols:
            db.execute("ALTER TABLE blur_hand_settings ADD COLUMN arm_ventral_dilate INTEGER DEFAULT 0")
        db.execute(
            """INSERT INTO blur_hand_settings
               (subject_id, trial_idx, hand_mask_enabled, hand_mask_radius, hand_smooth,
                forearm_radius, forearm_extent, hand_smooth2, dlc_radius, hand_temporal,
                show_landmarks, segments_json, mask_source,
                arm_dorsal_dilate, arm_ventral_dilate)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                             segments_json=excluded.segments_json,
                             mask_source=excluded.mask_source,
                             arm_dorsal_dilate=excluded.arm_dorsal_dilate,
                             arm_ventral_dilate=excluded.arm_ventral_dilate""",
            (subject_id, trial_idx, enabled, mask_radius, hand_smooth,
             _forearm_radius_legacy, forearm_extent, _hand_smooth2_legacy,
             _dlc_radius_legacy,
             hand_temporal, show_landmarks, segments_json,
             mask_source, arm_dorsal_dilate, arm_ventral_dilate),
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
    from ..services.video import build_trial_map

    with get_db_ctx() as db:
        subj = db.execute("SELECT * FROM subjects WHERE id = ?", (subject_id,)).fetchone()
        if not subj:
            raise HTTPException(404, "Subject not found")

    subject_name = subj["name"]
    trial_idx = body.get("trial_idx")
    execution_target = body.get("execution_target", "local-cpu")

    # Resolve trial name for display in job queue
    extra_params: dict | None = None
    if trial_idx is not None:
        try:
            camera_mode = subj.get("camera_mode") or "stereo"
            trials = build_trial_map(subject_name, camera_mode=camera_mode)
            if 0 <= trial_idx < len(trials):
                tn = trials[trial_idx]["trial_name"]
                trial_name = tn.split("_", 1)[1] if "_" in tn else tn
                extra_params = {"trial_idx": trial_idx, "trial_name": trial_name}
        except Exception:
            extra_params = {"trial_idx": trial_idx}

    # Enqueue — the queue manager stores extra_params in job_queue and propagates
    # trial_idx to the worker subprocess so only that trial is rendered.
    result = queue_manager.enqueue(
        "deidentify",
        [subject_id],
        [subject_name],
        execution_target=execution_target,
        extra_params=extra_params,
    )

    # Wait briefly for the queue manager to create the job record
    import time
    job_id = None
    for _ in range(20):
        with get_db_ctx() as db:
            job = db.execute(
                """SELECT id FROM jobs WHERE subject_id = ? AND job_type = 'deidentify'
                   AND status IN ('pending', 'running')
                   ORDER BY id DESC LIMIT 1""",
                (subject_id,),
            ).fetchone()
            if job:
                job_id = job["id"]
                break
        time.sleep(0.3)

    queue_id = result.get("queue_id")
    if not job_id and queue_id:
        return {"job_id": None, "queue_id": queue_id, "status": "queued"}

    return {"job_id": job_id, "queue_id": queue_id, "status": "queued"}
