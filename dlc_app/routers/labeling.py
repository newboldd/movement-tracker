"""Labeling session endpoints: frame serving, label CRUD, commit, MediaPipe prelabels."""
from __future__ import annotations

import json
from typing import List

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response

from ..config import get_settings
from ..db import get_db_ctx
from ..models import LabelBatchSave, SessionCreate, SessionResponse, STAGE_INDEX
from ..services.video import (
    extract_frame, build_trial_map, get_total_frames, get_subject_videos,
)
from ..services.labels import commit_labels_to_dlc
from ..services.mediapipe_prelabel import (
    get_mediapipe_for_session,
    recompute_distance_for_frame,
)
from ..services.discovery import _count_labeled_frames
from ..services.dlc_predictions import get_dlc_predictions_for_session

router = APIRouter(prefix="/api/labeling", tags=["labeling"])


@router.post("/{subject_id}/sessions", status_code=201)
def create_session(subject_id: int, req: SessionCreate) -> dict:
    """Create a new labeling session for a subject.

    For 'initial' sessions: copies labels from previous committed session.
    For 'refine' sessions: starts fresh (DLC predictions serve as ghost markers),
    increments iteration, doesn't change subject stage.
    """
    with get_db_ctx() as db:
        subj = db.execute(
            "SELECT * FROM subjects WHERE id = ?", (subject_id,)
        ).fetchone()
        if not subj:
            raise HTTPException(404, "Subject not found")

        is_refine = req.session_type == "refine"
        iteration = subj["iteration"] + 1 if is_refine else subj["iteration"]

        db.execute(
            """INSERT INTO label_sessions (subject_id, iteration, session_type)
               VALUES (?, ?, ?)""",
            (subject_id, iteration, req.session_type),
        )
        session = db.execute(
            "SELECT * FROM label_sessions WHERE subject_id = ? ORDER BY id DESC LIMIT 1",
            (subject_id,),
        ).fetchone()

        if not is_refine:
            # Copy labels from the most recent committed session (if any)
            prev_session = db.execute(
                """SELECT id FROM label_sessions
                   WHERE subject_id = ? AND status = 'committed'
                   ORDER BY committed_at DESC LIMIT 1""",
                (subject_id,),
            ).fetchone()
            if prev_session:
                db.execute(
                    """INSERT INTO frame_labels
                       (session_id, frame_num, trial_idx, side, keypoints, updated_at)
                       SELECT ?, frame_num, trial_idx, side, keypoints, CURRENT_TIMESTAMP
                       FROM frame_labels WHERE session_id = ?""",
                    (session["id"], prev_session["id"]),
                )

            # Only advance to 'labeling' if subject hasn't progressed beyond it
            current_stage = subj.get("stage", "created")
            if STAGE_INDEX.get(current_stage, 0) < STAGE_INDEX["labeling"]:
                db.execute(
                    "UPDATE subjects SET stage = 'labeling', updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (subject_id,),
                )

    return session


@router.get("/sessions/{session_id}/info")
def get_session_info(session_id: int) -> dict:
    """Get session info including subject details, trial map, bodyparts, and camera names."""
    settings = get_settings()

    with get_db_ctx() as db:
        session = db.execute(
            "SELECT * FROM label_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if not session:
            raise HTTPException(404, "Session not found")

        subj = db.execute(
            "SELECT * FROM subjects WHERE id = ?", (session["subject_id"],)
        ).fetchone()

    trials = build_trial_map(subj["name"])
    total_frames = trials[-1]["end_frame"] + 1 if trials else 0

    # Simplify trial info for frontend
    trial_info = [{
        "trial_name": t["trial_name"],
        "start_frame": t["start_frame"],
        "end_frame": t["end_frame"],
        "frame_count": t["frame_count"],
        "fps": t["fps"],
    } for t in trials]

    # Count committed frames in DLC labeled-data/
    committed_frame_count = 0
    if subj.get("dlc_dir"):
        dlc_path = settings.dlc_path / subj["dlc_dir"]
        if dlc_path.exists():
            committed_frame_count = _count_labeled_frames(dlc_path)

    return {
        "session": session,
        "subject": subj,
        "trials": trial_info,
        "total_frames": total_frames,
        "bodyparts": settings.bodyparts,
        "camera_names": settings.camera_names,
        "committed_frame_count": committed_frame_count,
    }


@router.get("/sessions/{session_id}/frame")
def get_frame(
    session_id: int,
    n: int = Query(..., description="Global frame number"),
    side: str = Query(..., description="Camera name"),
) -> Response:
    """Serve a video frame as JPEG."""
    settings = get_settings()
    if side not in settings.camera_names:
        raise HTTPException(400, f"side must be one of {settings.camera_names}")

    with get_db_ctx() as db:
        session = db.execute(
            "SELECT * FROM label_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if not session:
            raise HTTPException(404, "Session not found")
        subj = db.execute(
            "SELECT * FROM subjects WHERE id = ?", (session["subject_id"],)
        ).fetchone()

    try:
        jpeg_bytes = extract_frame(subj["name"], n, side)
    except (ValueError, Exception) as e:
        raise HTTPException(400, str(e))

    return Response(content=jpeg_bytes, media_type="image/jpeg")


@router.get("/sessions/{session_id}/labels")
def get_labels(session_id: int) -> List[dict]:
    """Get all labels for this session."""
    with get_db_ctx() as db:
        labels = db.execute(
            """SELECT frame_num, trial_idx, side, keypoints
               FROM frame_labels WHERE session_id = ? ORDER BY frame_num""",
            (session_id,),
        ).fetchall()

    # Parse keypoints JSON for the response
    for lbl in labels:
        if isinstance(lbl["keypoints"], str):
            lbl["keypoints"] = json.loads(lbl["keypoints"])

    return labels


@router.get("/sessions/{session_id}/video")
def get_video(
    session_id: int,
    trial: int = Query(0, description="Trial index"),
) -> FileResponse:
    """Stream a trial's raw video file for smooth HTML5 playback."""
    with get_db_ctx() as db:
        session = db.execute(
            "SELECT * FROM label_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if not session:
            raise HTTPException(404, "Session not found")
        subj = db.execute(
            "SELECT * FROM subjects WHERE id = ?", (session["subject_id"],)
        ).fetchone()

    trials = build_trial_map(subj["name"])
    if trial < 0 or trial >= len(trials):
        raise HTTPException(400, f"Trial index {trial} out of range (0-{len(trials)-1})")

    video_path = trials[trial]["video_path"]
    return FileResponse(video_path, media_type="video/mp4")


@router.get("/sessions/{session_id}/mediapipe")
def get_mediapipe(session_id: int) -> dict:
    """Return MediaPipe prelabel predictions for this session's subject.

    Response shape:
    {
        "OS": {"thumb": [[x,y], null, ...], "index": [[x,y], ...]},
        "OD": {"thumb": [[x,y], ...], "index": [[x,y], ...]},
        "distances": [d0, null, d2, ...]
    }
    """
    with get_db_ctx() as db:
        session = db.execute(
            "SELECT * FROM label_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if not session:
            raise HTTPException(404, "Session not found")
        subj = db.execute(
            "SELECT * FROM subjects WHERE id = ?", (session["subject_id"],)
        ).fetchone()

    data = get_mediapipe_for_session(subj["name"])
    if data is None:
        return {}

    return data


@router.get("/sessions/{session_id}/dlc_predictions")
def get_dlc_predictions(session_id: int) -> dict:
    """Return DLC analysis predictions for this session's subject.

    Response shape matches mediapipe format:
    {
        "OS": {"thumb": [[x,y], null, ...], "index": [[x,y], ...]},
        "OD": {"thumb": [[x,y], ...], "index": [[x,y], ...]}
    }
    """
    with get_db_ctx() as db:
        session = db.execute(
            "SELECT * FROM label_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if not session:
            raise HTTPException(404, "Session not found")
        subj = db.execute(
            "SELECT * FROM subjects WHERE id = ?", (session["subject_id"],)
        ).fetchone()

    data = get_dlc_predictions_for_session(subj["name"])
    if data is None:
        return {}

    return data


@router.put("/sessions/{session_id}/labels")
def save_labels(session_id: int, req: LabelBatchSave) -> dict:
    """Batch-save labels (upsert). Returns updated distances for affected frames."""
    with get_db_ctx() as db:
        session = db.execute(
            "SELECT * FROM label_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if not session:
            raise HTTPException(404, "Session not found")

        subj = db.execute(
            "SELECT * FROM subjects WHERE id = ?", (session["subject_id"],)
        ).fetchone()

        for label in req.labels:
            kp_json = json.dumps(label.keypoints)
            db.execute(
                """INSERT INTO frame_labels
                   (session_id, frame_num, trial_idx, side, keypoints, updated_at)
                   VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                   ON CONFLICT(session_id, frame_num, trial_idx, side)
                   DO UPDATE SET
                     keypoints = excluded.keypoints,
                     updated_at = CURRENT_TIMESTAMP""",
                (
                    session_id, label.frame_num, label.trial_idx, label.side,
                    kp_json,
                ),
            )

    # Recompute distances for affected frames
    updated_distances = {}
    if subj:
        # Group labels by frame to get both cameras' data
        frame_labels = {}
        for label in req.labels:
            if label.frame_num not in frame_labels:
                frame_labels[label.frame_num] = {}
            frame_labels[label.frame_num][label.side] = label.keypoints

        for frame_num, sides in frame_labels.items():
            dist = recompute_distance_for_frame(subj["name"], frame_num, sides)
            if dist is not None:
                updated_distances[str(frame_num)] = dist

    return {"saved": len(req.labels), "updated_distances": updated_distances}


@router.delete("/sessions/{session_id}/labels/{frame_num}")
def delete_label(session_id: int, frame_num: int, side: str = Query(...)) -> dict:
    """Delete labels for a specific frame."""
    with get_db_ctx() as db:
        db.execute(
            "DELETE FROM frame_labels WHERE session_id = ? AND frame_num = ? AND side = ?",
            (session_id, frame_num, side),
        )
    return {"deleted": True}


@router.post("/sessions/{session_id}/commit")
def commit_session(session_id: int) -> dict:
    """Commit session: extract frames as PNGs, write CSV/H5, create DLC structure.

    For refine sessions: also updates subject.iteration and auto-triggers remote training.
    """
    with get_db_ctx() as db:
        session = db.execute(
            "SELECT * FROM label_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if not session:
            raise HTTPException(404, "Session not found")

        subj = db.execute(
            "SELECT * FROM subjects WHERE id = ?", (session["subject_id"],)
        ).fetchone()

        labels = db.execute(
            """SELECT frame_num, trial_idx, side, keypoints
               FROM frame_labels WHERE session_id = ?""",
            (session_id,),
        ).fetchall()

    if not labels:
        raise HTTPException(400, "No labels to commit")

    result = commit_labels_to_dlc(
        subject_name=subj["name"],
        session_labels=labels,
        iteration=session["iteration"],
    )

    is_refine = session["session_type"] == "refine"

    # Update session and subject status
    with get_db_ctx() as db:
        db.execute(
            "UPDATE label_sessions SET status = 'committed', committed_at = CURRENT_TIMESTAMP WHERE id = ?",
            (session_id,),
        )
        if is_refine:
            db.execute(
                "UPDATE subjects SET iteration = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (session["iteration"], session["subject_id"]),
            )
        else:
            db.execute(
                "UPDATE subjects SET stage = 'committed', updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (session["subject_id"],),
            )

    # For refine sessions, auto-trigger training
    retrain_job_id = None
    if is_refine:
        try:
            from .pipeline import run_step as _run_step
            from ..models import RunStepRequest
            step_result = _run_step(session["subject_id"], RunStepRequest(step="train"))
            retrain_job_id = step_result.get("job_id")
        except Exception:
            pass  # Training trigger is best-effort

    result["retrain_job_id"] = retrain_job_id
    return result
