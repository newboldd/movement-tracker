"""Labeling session endpoints: frame serving, label CRUD, commit."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response

from ..config import get_settings
from ..db import get_db_ctx
from ..models import LabelBatchSave, SessionCreate, SessionResponse
from ..services.video import (
    extract_frame, build_trial_map, get_total_frames, get_subject_videos,
)
from ..services.labels import commit_labels_to_dlc

router = APIRouter(prefix="/api/labeling", tags=["labeling"])


@router.post("/{subject_id}/sessions", status_code=201)
def create_session(subject_id: int, req: SessionCreate) -> dict:
    """Get or create a labeling session for a subject.

    - If an active (non-committed) session exists, reuse it.
    - Otherwise create a new session and copy labels from the most
      recent committed session so the user can refine existing work.
    """
    with get_db_ctx() as db:
        subj = db.execute(
            "SELECT * FROM subjects WHERE id = ?", (subject_id,)
        ).fetchone()
        if not subj:
            raise HTTPException(404, "Subject not found")

        # Check for an existing active session that has labels
        active = db.execute(
            """SELECT ls.* FROM label_sessions ls
               WHERE ls.subject_id = ? AND ls.status != 'committed'
               AND EXISTS (SELECT 1 FROM frame_labels fl WHERE fl.session_id = ls.id)
               ORDER BY ls.id DESC LIMIT 1""",
            (subject_id,),
        ).fetchone()

        if active:
            return active

        # Find the most recent session that has labels (committed or not)
        prev = db.execute(
            """SELECT ls.id FROM label_sessions ls
               WHERE ls.subject_id = ?
               AND EXISTS (SELECT 1 FROM frame_labels fl WHERE fl.session_id = ls.id)
               ORDER BY ls.id DESC LIMIT 1""",
            (subject_id,),
        ).fetchone()

        # Clean up empty active sessions from previous clicks
        db.execute(
            """DELETE FROM label_sessions
               WHERE subject_id = ? AND status != 'committed'
               AND id NOT IN (SELECT DISTINCT session_id FROM frame_labels)""",
            (subject_id,),
        )

        # Create new session
        db.execute(
            """INSERT INTO label_sessions (subject_id, iteration, session_type)
               VALUES (?, ?, ?)""",
            (subject_id, subj["iteration"], req.session_type),
        )
        session = db.execute(
            "SELECT * FROM label_sessions WHERE subject_id = ? ORDER BY id DESC LIMIT 1",
            (subject_id,),
        ).fetchone()

        # Copy labels from previous session
        if prev:
            db.execute(
                """INSERT INTO frame_labels (session_id, frame_num, trial_idx, side, keypoints, updated_at)
                   SELECT ?, frame_num, trial_idx, side, keypoints, CURRENT_TIMESTAMP
                   FROM frame_labels WHERE session_id = ?""",
                (session["id"], prev["id"]),
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

    return {
        "session": session,
        "subject": subj,
        "trials": trial_info,
        "total_frames": total_frames,
        "bodyparts": settings.bodyparts,
        "camera_names": settings.camera_names,
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


@router.get("/sessions/{session_id}/labels")
def get_labels(session_id: int) -> list[dict]:
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


@router.put("/sessions/{session_id}/labels")
def save_labels(session_id: int, req: LabelBatchSave) -> dict:
    """Batch-save labels (upsert)."""
    with get_db_ctx() as db:
        session = db.execute(
            "SELECT * FROM label_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if not session:
            raise HTTPException(404, "Session not found")

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

    return {"saved": len(req.labels)}


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
    """Commit session: extract frames as PNGs, write CSV/H5, create DLC structure."""
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

    # Update session and subject status
    with get_db_ctx() as db:
        db.execute(
            "UPDATE label_sessions SET status = 'committed', committed_at = CURRENT_TIMESTAMP WHERE id = ?",
            (session_id,),
        )
        db.execute(
            "UPDATE subjects SET stage = 'training_dataset_created', updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (session["subject_id"],),
        )

    return result
