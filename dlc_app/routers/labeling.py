"""Labeling session endpoints: frame serving, label CRUD, commit, MediaPipe prelabels."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import FileResponse, Response

from ..config import get_settings
from ..db import get_db_ctx
from ..models import CommitRequest, LabelBatchSave, SessionCreate, SessionResponse, STAGE_INDEX
from ..services.video import (
    extract_frame, build_trial_map, get_total_frames, get_subject_videos,
    _deidentified_path, _get_no_face_videos,
)
from ..services.labels import commit_labels_to_dlc, save_corrections_to_csv
from ..services.mediapipe_prelabel import (
    get_mediapipe_for_session,
    recompute_distance_for_frame,
)
from ..services.discovery import _count_labeled_frames, _has_mediapipe
from ..services.dlc_predictions import (
    get_dlc_predictions_for_session,
    get_dlc_predictions_for_stage,
    get_stage_csv_files,
    has_stage_data,
)

router = APIRouter(prefix="/api/labeling", tags=["labeling"])


@router.post("/{subject_id}/sessions", status_code=201)
def create_session(subject_id: int, req: SessionCreate) -> dict:
    """Create a new labeling session for a subject.

    If an active session of the same type already exists, returns it
    (resumes work-in-progress instead of creating a duplicate).

    For 'initial' sessions: copies labels from previous committed session.
    For 'refine' sessions: starts fresh (DLC predictions serve as ghost markers),
    increments iteration, doesn't change subject stage.
    For 'corrections' sessions: starts fresh, doesn't increment iteration or
    change subject stage. Stage data loaded on demand via /stage_data endpoint.
    """
    with get_db_ctx() as db:
        subj = db.execute(
            "SELECT * FROM subjects WHERE id = ?", (subject_id,)
        ).fetchone()
        if not subj:
            raise HTTPException(404, "Subject not found")

        # Resume existing active session of the same type if one exists.
        # Prefer the session with the most labels (handles case where an
        # accidental blank session was created alongside one with work).
        existing = db.execute(
            """SELECT ls.*, COUNT(fl.id) AS label_count
               FROM label_sessions ls
               LEFT JOIN frame_labels fl ON fl.session_id = ls.id
               WHERE ls.subject_id = ? AND ls.session_type = ? AND ls.status = 'active'
               GROUP BY ls.id
               ORDER BY label_count DESC, ls.id DESC
               LIMIT 1""",
            (subject_id, req.session_type),
        ).fetchone()
        if existing:
            result = dict(existing)
            result.pop("label_count", None)
            return result

        is_refine = req.session_type == "refine"
        is_corrections = req.session_type == "corrections"
        is_final = req.session_type == "final"
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

        if not is_refine and not is_corrections and not is_final:
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
        # Corrections sessions: no label copy, no stage change

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

    # Serve deidentified version when enabled
    settings = get_settings()
    if settings.prefer_deidentified:
        stem = Path(video_path).stem
        no_face = _get_no_face_videos(subj["name"])
        if stem not in no_face:
            deident = _deidentified_path(video_path)
            if deident:
                video_path = deident

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


@router.get("/sessions/{session_id}/committed_labels")
def get_committed_labels(session_id: int) -> List[dict]:
    """Get all manually labeled frames from committed sessions for this subject.

    For refine sessions, these represent frames the user already hand-labeled
    in earlier rounds. They should take priority over DLC predictions as ghost
    markers so that human corrections are preserved.

    Returns list of {frame_num, trial_idx, side, keypoints} dicts.
    """
    with get_db_ctx() as db:
        session = db.execute(
            "SELECT * FROM label_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if not session:
            raise HTTPException(404, "Session not found")

        # Find all committed sessions for this subject (excluding current)
        labels = db.execute(
            """SELECT fl.frame_num, fl.trial_idx, fl.side, fl.keypoints
               FROM frame_labels fl
               JOIN label_sessions ls ON ls.id = fl.session_id
               WHERE ls.subject_id = ? AND ls.status = 'committed'
               ORDER BY ls.committed_at DESC, fl.frame_num""",
            (session["subject_id"],),
        ).fetchall()

    # De-duplicate: keep the most recently committed label for each frame/side
    seen = set()
    result = []
    for lbl in labels:
        key = (lbl["frame_num"], lbl["side"])
        if key in seen:
            continue
        seen.add(key)
        if isinstance(lbl["keypoints"], str):
            lbl["keypoints"] = json.loads(lbl["keypoints"])
        result.append(lbl)

    return result


@router.get("/sessions/{session_id}/available_stages")
def get_available_stages(session_id: int) -> dict:
    """Return which processing stages have data for this subject.

    Used by corrections mode to build the stage selector radio buttons.
    Returns {"stages": ["mp", "labels", "dlc", "refine", "corrections"]}
    (only stages that have actual data).
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

    settings = get_settings()
    subject_name = subj["name"]
    dlc_path = settings.dlc_path / subject_name if subject_name else None

    stages = []

    # MP: check for mediapipe_prelabels.npz
    if dlc_path and _has_mediapipe(dlc_path):
        stages.append("mp")

    # Labels: check for committed label sessions in DB
    with get_db_ctx() as db:
        committed = db.execute(
            """SELECT COUNT(*) AS cnt FROM frame_labels fl
               JOIN label_sessions ls ON ls.id = fl.session_id
               WHERE ls.subject_id = ? AND ls.status = 'committed'
               AND ls.session_type != 'corrections'""",
            (subj["id"],),
        ).fetchone()
    if committed and committed["cnt"] > 0:
        stages.append("labels")

    # DLC / Refine / Corrections: check for CSV directories
    stage_files = {}
    for stage_name in ("dlc", "refine", "corrections"):
        csv_files = get_stage_csv_files(subject_name, stage_name)
        if csv_files:
            stages.append(stage_name)
            stage_files[stage_name] = csv_files

    return {"stages": stages, "stage_files": stage_files}


@router.get("/sessions/{session_id}/stage_data")
def get_stage_data(
    session_id: int,
    stage: str = Query(..., description="Processing stage: mp, labels, dlc, refine, corrections"),
) -> dict:
    """Return label data for a specific processing stage.

    Response format matches mediapipe/dlc_predictions:
    {camera: {bodypart: [[x,y]|null, ...]}, distances: [...]}
    """
    valid_stages = ("mp", "labels", "dlc", "refine", "corrections")
    if stage not in valid_stages:
        raise HTTPException(400, f"stage must be one of {valid_stages}")

    with get_db_ctx() as db:
        session = db.execute(
            "SELECT * FROM label_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if not session:
            raise HTTPException(404, "Session not found")
        subj = db.execute(
            "SELECT * FROM subjects WHERE id = ?", (session["subject_id"],)
        ).fetchone()

    subject_name = subj["name"]

    if stage == "mp":
        data = get_mediapipe_for_session(subject_name)
        return data if data else {}

    if stage == "labels":
        data = _committed_labels_to_array(subj)
        return data if data else {}

    # dlc, refine, corrections — load from specific directory
    data = get_dlc_predictions_for_stage(subject_name, stage)
    return data if data else {}


def _committed_labels_to_array(subj: dict) -> dict | None:
    """Convert committed manual labels from DB to array format matching MP/DLC.

    Returns {camera: {bodypart: [[x,y]|null, ...]}, distances: [...]}
    """
    settings = get_settings()
    subject_name = subj["name"]
    cam_names = settings.camera_names

    trials = build_trial_map(subject_name)
    if not trials:
        return None

    total_frames = trials[-1]["end_frame"] + 1

    # Query all committed labels (excluding corrections sessions)
    with get_db_ctx() as db:
        rows = db.execute(
            """SELECT fl.frame_num, fl.side, fl.keypoints
               FROM frame_labels fl
               JOIN label_sessions ls ON ls.id = fl.session_id
               WHERE ls.subject_id = ? AND ls.status = 'committed'
               AND ls.session_type != 'corrections'
               ORDER BY ls.committed_at DESC""",
            (subj["id"],),
        ).fetchall()

    if not rows:
        return None

    # Initialize result structure
    result = {}
    for cam in cam_names:
        result[cam] = {}
        for bp in settings.bodyparts:
            result[cam][bp] = [None] * total_frames

    # De-duplicate: keep most recently committed per (frame, side)
    seen = set()
    for row in rows:
        key = (row["frame_num"], row["side"])
        if key in seen:
            continue
        seen.add(key)

        frame_num = row["frame_num"]
        side = row["side"]
        if side not in cam_names or frame_num >= total_frames:
            continue

        kp = row["keypoints"]
        if isinstance(kp, str):
            kp = json.loads(kp)

        for bp in settings.bodyparts:
            coords = kp.get(bp)
            if coords and coords[0] is not None:
                result[side][bp][frame_num] = coords

    # Check if we have any data
    has_data = any(
        any(c is not None for c in result[cam][bp])
        for cam in cam_names
        for bp in settings.bodyparts
    )
    if not has_data:
        return None

    # Compute distances
    from ..services.dlc_predictions import _compute_dlc_distances
    distances = _compute_dlc_distances(result, cam_names, subject_name, total_frames)
    if distances is not None:
        result["distances"] = distances

    return result


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

    # Recompute distances for affected frames.
    # Merge in any other saved labels for the same frames (e.g. the other
    # camera's correction) so triangulation has both cameras' manual data.
    updated_distances = {}
    if subj:
        affected_frames = {label.frame_num for label in req.labels}

        frame_labels = {}
        with get_db_ctx() as db2:
            for frame_num in affected_frames:
                saved = db2.execute(
                    "SELECT side, keypoints FROM frame_labels "
                    "WHERE session_id = ? AND frame_num = ?",
                    (session_id, frame_num),
                ).fetchall()
                sides = {}
                for row in saved:
                    kp = row["keypoints"]
                    if isinstance(kp, str):
                        kp = json.loads(kp)
                    sides[row["side"]] = kp
                frame_labels[frame_num] = sides

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
def commit_session(
    session_id: int,
    req: CommitRequest = Body(default=None),
) -> dict:
    """Commit session: extract frames as PNGs, write CSV/H5, create DLC structure.

    For refine sessions: saves all labeled frames to corrections CSV, then commits only
    non-excluded frames (those not in req.v2_excludes) to DLC training data, and
    auto-triggers remote training.
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

    is_refine = session["session_type"] == "refine"

    if not labels and not is_refine:
        raise HTTPException(400, "No labels to commit")
    is_corrections = session["session_type"] == "corrections"

    if is_corrections:
        # Corrections: write DLC-format CSVs to corrections/ directory
        result = save_corrections_to_csv(
            subject_name=subj["name"],
            session_labels=labels,
        )
        with get_db_ctx() as db:
            db.execute(
                "UPDATE label_sessions SET status = 'committed', committed_at = CURRENT_TIMESTAMP WHERE id = ?",
                (session_id,),
            )
            db.execute(
                "UPDATE subjects SET stage = 'corrected', updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (session["subject_id"],),
            )
        return result

    if is_refine:
        # Build training labels from corrections stage data for the requested frames.
        # The frontend sends the frames (corrections CSV != DLC CSV) that are NOT excluded.
        import json as _json
        from ..services.dlc_predictions import get_dlc_predictions_for_stage

        # First, persist any manual edits made in refine mode to the corrections CSV
        if labels:
            save_corrections_to_csv(
                subject_name=subj["name"],
                session_labels=labels,
            )

        train_frames = req.v2_train_frames if req is not None else []
        if not train_frames:
            raise HTTPException(400, "No training frames selected (v2_train_frames is empty)")

        # Load corrections stage data to get keypoints for each training frame
        corr_data = get_dlc_predictions_for_stage(subj["name"], "corrections")
        if not corr_data:
            raise HTTPException(400, "No corrections stage data found for this subject")

        settings = get_settings()
        bodyparts = settings.bodyparts
        cam_names = settings.camera_names

        # Build a synthetic session_labels list from corrections data
        from ..services.video import build_trial_map as _build_trial_map
        trials = _build_trial_map(subj["name"])
        frame_to_trial = {}
        for ti, t in enumerate(trials):
            for f in range(t["start_frame"], t["end_frame"] + 1):
                frame_to_trial[f] = ti

        training_labels = []
        for vf in train_frames:
            frame_num = vf.frame_num
            side = vf.side
            if side not in cam_names:
                continue
            cam_data = corr_data.get(side, {})
            kp = {}
            for bp in bodyparts:
                arr = cam_data.get(bp)
                if arr and frame_num < len(arr) and arr[frame_num] is not None:
                    kp[bp] = arr[frame_num]
            if not kp:
                continue
            training_labels.append({
                "frame_num": frame_num,
                "trial_idx": frame_to_trial.get(frame_num, 0),
                "side": side,
                "keypoints": kp,
            })

        if not training_labels:
            raise HTTPException(400, "No usable labels found in corrections data for selected frames")

        result = commit_labels_to_dlc(
            subject_name=subj["name"],
            session_labels=training_labels,
            iteration=session["iteration"],
        )
    else:
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
