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
    return result


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
                   (subject_id, trial_idx, spot_type, x, y, radius, frame_start, frame_end)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (subject_id, trial_idx, s.get("spot_type", "face"),
                 s["x"], s["y"], s["radius"], s["frame_start"], s["frame_end"]),
            )

    return {"status": "ok", "count": len(specs)}


# ── Hand settings ─────────────────────────────────────────────────────────

@router.get("/{subject_id}/hand-settings")
def get_hand_settings(subject_id: int, trial_idx: int = Query(0)) -> dict:
    """Get hand mask settings for a trial."""
    with get_db_ctx() as db:
        row = db.execute(
            "SELECT * FROM blur_hand_settings WHERE subject_id = ? AND trial_idx = ?",
            (subject_id, trial_idx),
        ).fetchone()
    if row:
        return {"enabled": bool(row["hand_mask_enabled"]), "mask_radius": row["hand_mask_radius"]}
    return {"enabled": True, "mask_radius": 30}


@router.put("/{subject_id}/hand-settings")
def save_hand_settings(subject_id: int, body: dict = Body(...)) -> dict:
    """Save hand mask settings for a trial.

    Body: { trial_idx: int, enabled: bool, mask_radius: int }
    """
    trial_idx = body.get("trial_idx", 0)
    enabled = 1 if body.get("enabled", True) else 0
    mask_radius = body.get("mask_radius", 30)

    with get_db_ctx() as db:
        db.execute(
            """INSERT INTO blur_hand_settings (subject_id, trial_idx, hand_mask_enabled, hand_mask_radius)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(subject_id, trial_idx)
               DO UPDATE SET hand_mask_enabled=excluded.hand_mask_enabled,
                             hand_mask_radius=excluded.hand_mask_radius""",
            (subject_id, trial_idx, enabled, mask_radius),
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
