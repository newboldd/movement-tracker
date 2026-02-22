"""Labeling session endpoints: frame serving, label CRUD, commit, MediaPipe prelabels."""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from ..config import get_settings
from ..db import get_db_ctx
from ..models import LabelBatchSave, SessionCreate, SessionResponse
from ..services.video import (
    extract_frame, build_trial_map, get_total_frames, get_subject_videos,
)
from ..services.labels import commit_labels_to_dlc
from ..services.mediapipe_prelabel import (
    get_mediapipe_for_session,
    recompute_distance_for_frame,
)

router = APIRouter(prefix="/api/labeling", tags=["labeling"])


@router.post("/{subject_id}/sessions", status_code=201)
def create_session(subject_id: int, req: SessionCreate) -> dict:
    """Create a new labeling session for a subject."""
    with get_db_ctx() as db:
        subj = db.execute(
            "SELECT * FROM subjects WHERE id = ?", (subject_id,)
        ).fetchone()
        if not subj:
            raise HTTPException(404, "Subject not found")

        db.execute(
            """INSERT INTO label_sessions (subject_id, iteration, session_type)
               VALUES (?, ?, ?)""",
            (subject_id, subj["iteration"], req.session_type),
        )
        session = db.execute(
            "SELECT * FROM label_sessions WHERE subject_id = ? ORDER BY id DESC LIMIT 1",
            (subject_id,),
        ).fetchone()

        # Update subject stage
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
            "UPDATE subjects SET stage = 'committed', updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (session["subject_id"],),
        )

    return result
