"""Deidentify API: interactive face blur with hand protection."""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import cv2
from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import Response

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

    trial_list = []
    for i, t in enumerate(trials):
        # Check if video is stereo
        cap = cv2.VideoCapture(t["video_path"])
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        is_stereo = (fw / fh) > 1.7 if fh > 0 else False

        entry = {
            "trial_idx": i,
            "trial_name": t["trial_name"],
            "video_path": t["video_path"],
            "start_frame": t["start_frame"],
            "end_frame": t["end_frame"],
            "frame_count": t["frame_count"],
            "fps": t["fps"],
            "is_stereo": is_stereo,
            "frame_width": fw // 2 if is_stereo else fw,
            "frame_height": fh,
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
) -> Response:
    """Serve a single JPEG frame from the ORIGINAL (un-blurred) video."""
    with get_db_ctx() as db:
        subj = db.execute("SELECT * FROM subjects WHERE id = ?", (subject_id,)).fetchone()
        if not subj:
            raise HTTPException(404, "Subject not found")

    camera_mode = subj.get("camera_mode") or "stereo"
    trials = build_trial_map(subj["name"], camera_mode=camera_mode)
    if trial_idx < 0 or trial_idx >= len(trials):
        raise HTTPException(400, f"Invalid trial_idx {trial_idx}")

    trial = trials[trial_idx]
    settings = get_settings()
    cam_names = settings.camera_names

    # Multicam: resolve to the specific camera file
    video_path = trial["video_path"]
    if camera_mode == "multicam" and side != "full":
        for cam in trial.get("cameras", []):
            if cam["name"] == side:
                video_path = cam["path"]
                break

    cap = cv2.VideoCapture(video_path)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    is_stereo = camera_mode == "stereo" and ((fw / fh) > 1.7 if fh > 0 else False)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise HTTPException(400, f"Cannot read frame {frame_num}")

    # Stereo: crop to left or right half based on camera name
    if is_stereo and side != "full":
        half_w = fw // 2
        if len(cam_names) >= 2 and side == cam_names[1]:
            frame = frame[:, half_w:, :]
        else:
            frame = frame[:, :half_w, :]

    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return Response(content=jpeg.tobytes(), media_type="image/jpeg")


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
                        })
            if not np.all(np.isnan(od_lm[f])):
                for j in range(od_lm.shape[1]):
                    if not np.isnan(od_lm[f, j, 0]):
                        pts.append({
                            "x": round(float(od_lm[f, j, 0]), 1),
                            "y": round(float(od_lm[f, j, 1]), 1),
                            "side": "right" if is_stereo else "full",
                            "type": "hand",
                        })
            if pts:
                result[str(f)] = pts

    # Pose landmarks: shoulders (11,12) and elbows (13,14) only.
    # Wrists (15,16) and hand points (17-22) are excluded because the
    # dedicated Hands model provides more accurate positions for those.
    POSE_UPPER_BODY = [11, 12, 13, 14]
    if pose_data:
        os_pose = pose_data["OS_pose"]
        od_pose = pose_data["OD_pose"]
        for f in range(start, min(end + 1, len(os_pose))):
            pts = result.get(str(f), [])
            if not np.all(np.isnan(os_pose[f])):
                for j in POSE_UPPER_BODY:
                    if j < os_pose.shape[1] and not np.isnan(os_pose[f, j, 0]):
                        pts.append({
                            "x": round(float(os_pose[f, j, 0]), 1),
                            "y": round(float(os_pose[f, j, 1]), 1),
                            "side": "left" if is_stereo else "full",
                            "type": "pose",
                        })
            if not np.all(np.isnan(od_pose[f])):
                for j in POSE_UPPER_BODY:
                    if j < od_pose.shape[1] and not np.isnan(od_pose[f, j, 0]):
                        pts.append({
                            "x": round(float(od_pose[f, j, 0]), 1),
                            "y": round(float(od_pose[f, j, 1]), 1),
                            "side": "right" if is_stereo else "full",
                            "type": "pose",
                        })
            if pts:
                result[str(f)] = pts

    return {"landmarks": result, "has_pose": pose_data is not None}


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
                    offset_x, offset_y, frame_start, frame_end)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (subject_id, trial_idx, s.get("spot_type", "face"),
                 s["x"], s["y"], s["radius"],
                 s.get("width"), s.get("height"),
                 s.get("offset_x", 0), s.get("offset_y", 0),
                 s["frame_start"], s["frame_end"]),
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
            "segments": segments,
        }
    return {"enabled": True, "mask_radius": 30, "segments": []}


@router.put("/{subject_id}/hand-settings")
def save_hand_settings(subject_id: int, body: dict = Body(...)) -> dict:
    """Save hand mask settings for a trial.

    Body: { trial_idx: int, enabled: bool, mask_radius: int, segments: [{start, end, radius}] }
    """
    import json as _json
    trial_idx = body.get("trial_idx", 0)
    enabled = 1 if body.get("enabled", True) else 0
    mask_radius = body.get("mask_radius", 30)
    segments = body.get("segments", [])
    segments_json = _json.dumps(segments) if segments else None

    with get_db_ctx() as db:
        db.execute(
            """INSERT INTO blur_hand_settings
               (subject_id, trial_idx, hand_mask_enabled, hand_mask_radius, segments_json)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(subject_id, trial_idx)
               DO UPDATE SET hand_mask_enabled=excluded.hand_mask_enabled,
                             hand_mask_radius=excluded.hand_mask_radius,
                             segments_json=excluded.segments_json""",
            (subject_id, trial_idx, enabled, mask_radius, segments_json),
        )
    return {"status": "ok"}


# ── Render ────────────────────────────────────────────────────────────────

@router.post("/{subject_id}/render")
def render_deidentified(subject_id: int) -> dict:
    """Render deidentified videos for all trials using saved blur specs.

    Creates a background job. Returns {job_id}.
    """
    from ..services.deidentify import render_with_blur_specs
    import os

    with get_db_ctx() as db:
        subj = db.execute("SELECT * FROM subjects WHERE id = ?", (subject_id,)).fetchone()
        if not subj:
            raise HTTPException(404, "Subject not found")

        all_specs = db.execute(
            "SELECT * FROM blur_specs WHERE subject_id = ?", (subject_id,),
        ).fetchall()

        all_hand_settings = db.execute(
            "SELECT * FROM blur_hand_settings WHERE subject_id = ?", (subject_id,),
        ).fetchall()

    subject_name = subj["name"]
    camera_mode = subj.get("camera_mode") or "stereo"
    trials = build_trial_map(subject_name, camera_mode=camera_mode)

    # Group specs by trial
    specs_by_trial = {}
    for s in all_specs:
        ti = s["trial_idx"]
        if ti not in specs_by_trial:
            specs_by_trial[ti] = []
        specs_by_trial[ti].append(dict(s))

    # Group hand settings by trial
    hand_by_trial = {}
    for hs in all_hand_settings:
        hand_by_trial[hs["trial_idx"]] = {
            "enabled": bool(hs["hand_mask_enabled"]),
            "mask_radius": hs["hand_mask_radius"],
        }

    settings = get_settings()

    # Create job
    with get_db_ctx() as db:
        log_dir = settings.dlc_path / ".logs"
        log_dir.mkdir(exist_ok=True)
        log_path = str(log_dir / f"job_deidentify_{subject_id}.log")

        db.execute(
            """INSERT INTO jobs (subject_id, job_type, status, log_path)
               VALUES (?, 'deidentify', 'pending', ?)""",
            (subject_id, log_path),
        )
        job = db.execute(
            "SELECT * FROM jobs WHERE subject_id = ? ORDER BY id DESC LIMIT 1",
            (subject_id,),
        ).fetchone()

    job_id = job["id"]
    cancel_event = registry.register_cancel_event(job_id)

    def _run():
        try:
            with get_db_ctx() as db:
                db.execute(
                    "UPDATE jobs SET status = 'running', started_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (job_id,),
                )

            output_dir = settings.video_path
            deident_dir = Path(output_dir) / "deidentified"
            deident_dir.mkdir(parents=True, exist_ok=True)

            trials_with_specs = [i for i in range(len(trials)) if i in specs_by_trial]
            n_trials = len(trials_with_specs)

            for ti_idx, trial_i in enumerate(trials_with_specs):
                if cancel_event.is_set():
                    raise InterruptedError("Job cancelled")

                trial = trials[trial_i]
                cam = trial.get("camera_name")
                if cam:
                    output_name = f"{subject_name}_{trial['trial_name']}_{cam}.mp4"
                else:
                    output_name = f"{subject_name}_{trial['trial_name']}.mp4"
                output_path = str(deident_dir / output_name)

                base_pct = ti_idx * (100.0 / max(n_trials, 1))
                span = 100.0 / max(n_trials, 1)

                def progress_cb(pct, _base=base_pct, _span=span):
                    if cancel_event.is_set():
                        raise InterruptedError("Job cancelled")
                    overall = _base + (pct / 100.0) * _span
                    with get_db_ctx() as db:
                        db.execute(
                            "UPDATE jobs SET progress_pct = ? WHERE id = ?",
                            (overall, job_id),
                        )

                # Find the source video (original, not deidentified)
                source_path = trial["video_path"]

                render_with_blur_specs(
                    input_path=source_path,
                    output_path=output_path,
                    blur_specs=specs_by_trial.get(trial_i, []),
                    hand_settings=hand_by_trial.get(trial_i),
                    start_frame=trial["start_frame"],
                    frame_count=trial["frame_count"],
                    progress_callback=progress_cb,
                )

            # Write .deidentified marker
            dlc_path = settings.dlc_path / subject_name
            dlc_path.mkdir(parents=True, exist_ok=True)
            (dlc_path / ".deidentified").write_text("")

            with get_db_ctx() as db:
                db.execute(
                    """UPDATE jobs SET status = 'completed', progress_pct = 100,
                       finished_at = CURRENT_TIMESTAMP WHERE id = ?""",
                    (job_id,),
                )

        except InterruptedError:
            with get_db_ctx() as db:
                db.execute(
                    """UPDATE jobs SET status = 'cancelled',
                       finished_at = CURRENT_TIMESTAMP WHERE id = ?""",
                    (job_id,),
                )
        except Exception as e:
            logger.exception(f"Deidentify job {job_id} failed")
            with get_db_ctx() as db:
                db.execute(
                    """UPDATE jobs SET status = 'failed', error_msg = ?,
                       finished_at = CURRENT_TIMESTAMP WHERE id = ?""",
                    (str(e), job_id),
                )
        finally:
            registry.unregister_cancel_event(job_id)

    threading.Thread(target=_run, daemon=True).start()

    return {"status": "ok", "job_id": job_id}
