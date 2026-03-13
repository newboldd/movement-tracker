"""Labeling session endpoints: frame serving, label CRUD, commit, MediaPipe prelabels."""
from __future__ import annotations

import csv as _csv
import json
from pathlib import Path
from typing import List

from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import FileResponse, Response

from ..config import get_settings
from ..db import get_db_ctx
from ..models import CommitRequest, LabelBatchSave, SessionCreate, SessionResponse, STAGE_INDEX, STAGES
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
    get_corrections_with_dlc_fallback,
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
        is_final = req.session_type in ("final", "events")
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

    # corrections: load corrections where available, DLC fallback for uncovered trials
    if stage == "corrections":
        data = get_corrections_with_dlc_fallback(subject_name)
        return data if data else {}

    # dlc, refine — load from specific directory
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

        # Load stage data so un-labeled cameras use the same coordinates as the
        # distance trace display (corrections CSV + per-trial DLC fallback),
        # rather than raw MP which may differ significantly.
        stage_data = None
        session_type = session["session_type"] if session else None
        if session_type in ("corrections", "refine", "initial"):
            stage_data = get_corrections_with_dlc_fallback(subj["name"])

        for frame_num, sides in frame_labels.items():
            dist = recompute_distance_for_frame(subj["name"], frame_num, sides,
                                                stage_data=stage_data)
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

    return result


EVENT_TYPES = ("open", "peak", "close")


def _events_csv_path(subject_name: str) -> Path:
    settings = get_settings()
    return settings.dlc_path / subject_name / "events.csv"


def _read_events_csv(subject_name: str) -> dict:
    result = {t: [] for t in EVENT_TYPES}
    path = _events_csv_path(subject_name)
    if not path.exists():
        return result
    with open(path, newline="") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            et = row.get("event_type", "").strip()
            fn = row.get("frame_num", "").strip()
            if et in result and fn.isdigit():
                result[et].append(int(fn))
    return result


def _write_events_csv(subject_name: str, events: dict) -> int:
    path = _events_csv_path(subject_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", newline="") as f:
        writer = _csv.writer(f)
        writer.writerow(["event_type", "frame_num"])
        for et in EVENT_TYPES:
            for fn in sorted(events.get(et, [])):
                writer.writerow([et, fn])
                count += 1
    return count


def _get_subject_name_for_session(session_id: int) -> str:
    with get_db_ctx() as db:
        session = db.execute(
            "SELECT * FROM label_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if not session:
            raise HTTPException(404, "Session not found")
        subj = db.execute(
            "SELECT name FROM subjects WHERE id = ?", (session["subject_id"],)
        ).fetchone()
    return subj["name"]


@router.get("/sessions/{session_id}/events")
def get_events(session_id: int) -> dict:
    """Get all tapping events for the session's subject from CSV."""
    subject_name = _get_subject_name_for_session(session_id)
    return _read_events_csv(subject_name)


@router.put("/sessions/{session_id}/events")
def save_events(session_id: int, body: dict = Body(...)) -> dict:
    """Update events for provided types only; preserve other types from existing CSV."""
    subject_name = _get_subject_name_for_session(session_id)
    existing = _read_events_csv(subject_name)
    for et in EVENT_TYPES:
        if et in body:
            existing[et] = [int(f) for f in body[et]]
    count = _write_events_csv(subject_name, existing)

    # Update subject stage based on event completeness
    types_with_data = [et for et in EVENT_TYPES if len(existing.get(et, [])) > 0]
    if types_with_data:
        new_stage = "events_complete" if len(types_with_data) == len(EVENT_TYPES) else "events_partial"
        with get_db_ctx() as db:
            session = db.execute(
                "SELECT subject_id FROM label_sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if session:
                current = db.execute(
                    "SELECT stage FROM subjects WHERE id = ?", (session["subject_id"],)
                ).fetchone()
                cur_stage = current["stage"] if current else "created"
                cur_idx = STAGE_INDEX.get(cur_stage, 0)
                new_idx = STAGE_INDEX.get(new_stage, 0)
                # Advance stage (or update partial→complete)
                if new_idx >= cur_idx:
                    db.execute(
                        "UPDATE subjects SET stage = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                        (new_stage, session["subject_id"]),
                    )

    return {"saved": count}


@router.post("/sessions/{session_id}/detect_events")
def detect_events(session_id: int, body: dict = Body(default={})) -> dict:
    """Auto-detect tapping events from the distance trace.

    Uses threshold crossing on 4-frame velocity derivative to find opening/closing
    movements, then searches backward/forward for precise start/stop frames.
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

    subject_name = subj["name"]

    # Load distance data — prefer corrections stage (triangulated from full manual corrections,
    # clean data with no tracking artifacts). Fall back to mediapipe if corrections unavailable.
    dist_raw = None
    for stage in ("corrections", "refine", "dlc"):
        stage_data = get_dlc_predictions_for_stage(subject_name, stage)
        if stage_data and stage_data.get("distances"):
            candidate = stage_data["distances"]
            valid_count = sum(1 for d in candidate if d is not None)
            if valid_count > len(candidate) * 0.5:  # need >50% coverage
                dist_raw = candidate
                break

    if not dist_raw:
        # Fallback: mediapipe (full coverage but may have tracking outliers)
        data = get_mediapipe_for_session(subject_name)
        dist_raw = data.get("distances") if data else None

    if not dist_raw:
        raise HTTPException(400, "No distance data available for auto-detection")

    # Parameters (can be overridden via request body)
    # Minimum distance (mm) for a local maximum to count as a real peak opening
    min_peak_height = float(body.get("min_peak_height", 15.0))
    # 1-frame velocity threshold for pinpointing the open onset (backtrack from peak)
    open_start_thresh = float(body.get("open_start_thresh", 5))
    nback = int(body.get("nback", 60))
    # Minimum frames between consecutive open events (prevents double-counting).
    # 10 frames @ 60 fps = 167 ms = 6 Hz max tapping rate.
    min_event_gap = int(body.get("min_event_gap", 10))
    # Distances above this are tracking artifacts; excluded from derivatives
    max_valid_dist = float(body.get("max_valid_dist", 200.0))

    # Clean raw distances: mark outliers as None, then linearly interpolate gaps.
    # This prevents derivative spikes caused by NaN→0 substitution or tracking artifacts.
    dist_clean: list[float | None] = []
    for d in dist_raw:
        if d is None or not (0 <= float(d) <= max_valid_dist):
            dist_clean.append(None)
        else:
            dist_clean.append(float(d))

    # Track which frames were originally missing (before interpolation)
    # Used later to skip peak candidates flanked by genuine missing data.
    nan_mask = [v is None for v in dist_clean]

    # Linear interpolation over None runs
    n = len(dist_clean)
    for i in range(n):
        if dist_clean[i] is not None:
            continue
        left_v = left_i = None
        for j in range(i - 1, -1, -1):
            if dist_clean[j] is not None:
                left_v, left_i = dist_clean[j], j
                break
        right_v = right_i = None
        for j in range(i + 1, n):
            if dist_clean[j] is not None:
                right_v, right_i = dist_clean[j], j
                break
        if left_v is not None and right_v is not None:
            t = (i - left_i) / (right_i - left_i)
            dist_clean[i] = left_v + t * (right_v - left_v)
        elif left_v is not None:
            dist_clean[i] = left_v
        elif right_v is not None:
            dist_clean[i] = right_v
        else:
            dist_clean[i] = 0.0

    dist = [d if d is not None else 0.0 for d in dist_clean]

    # 4-frame velocity (opening/closing speed)
    ddist = [0.0] * n
    for i in range(4, n):
        ddist[i] = dist[i] - dist[i - 4]

    # 1-frame velocity (fine-grained direction)
    ddist1 = [0.0] * n
    for i in range(1, n):
        ddist1[i] = dist[i] - dist[i - 1]

    # --- Detect peaks (local maxima), then derive opens and closes from them ---
    # Peaks are the most geometrically stable feature: local maxima in the distance trace
    # are unambiguous regardless of opening amplitude, so we find them first and derive
    # open onsets by backtracking and closes by looking forward.
    half_win = max(min_event_gap // 2, 10)

    # Find local-maximum candidates above min_peak_height
    peak_candidates: list[int] = []
    for i in range(half_win, n - half_win):
        if dist[i] < min_peak_height:
            continue
        # Skip candidates flanked by originally-missing data (interpolated values
        # can produce spurious peaks at the boundary of tracked/untracked regions)
        if any(nan_mask[i - half_win: i + half_win + 1]):
            continue
        if dist[i] >= max(dist[i - half_win: i + half_win + 1]):
            peak_candidates.append(i)

    # Merge nearby candidates: keep the highest frame within each cluster
    merged_peaks: list[int] = []
    if peak_candidates:
        group = [peak_candidates[0]]
        for k in range(1, len(peak_candidates)):
            if peak_candidates[k] - peak_candidates[k - 1] < min_event_gap:
                group.append(peak_candidates[k])
            else:
                merged_peaks.append(max(group, key=lambda f: dist[f]))
                group = [peak_candidates[k]]
        merged_peaks.append(max(group, key=lambda f: dist[f]))

    # Valley filter: consecutive peaks with no clear close between them are part of the
    # same opening — keep only the higher one.  Real distinct taps always have the finger
    # return below valley_thresh between them.
    valley_thresh = float(body.get("valley_thresh", 25.0))
    filtered: list[int] = []
    for pk in merged_peaks:
        if not filtered:
            filtered.append(pk)
            continue
        valley_min = min(dist[filtered[-1]: pk + 1])
        if valley_min < valley_thresh:
            filtered.append(pk)          # clear close separates these → distinct peaks
        elif dist[pk] > dist[filtered[-1]]:
            filtered[-1] = pk            # same opening, keep the higher peak
    merged_peaks = filtered

    # For each detected peak, find the open onset and close.
    # open onset: find the valley (local minimum) in [search_start, pk], then advance
    #             forward from the valley until velocity first exceeds open_start_thresh.
    #             This reliably finds the true onset even when the peak is approached slowly.
    # close: minimum distance between this peak and the next peak.
    opens_raw: list[int] = []
    closes_raw: list[int] = []
    for idx, pk in enumerate(merged_peaks):
        search_start = closes_raw[-1] if closes_raw else max(0, pk - nback)
        valley = min(range(search_start, pk + 1), key=lambda f: dist[f])
        open_frame = valley  # fallback: the valley bottom itself
        for k in range(valley, pk):
            if ddist1[k] > open_start_thresh:
                open_frame = k - 1  # last frame before the rise began
                break
        opens_raw.append(max(search_start, open_frame))

        next_pk = merged_peaks[idx + 1] if idx + 1 < len(merged_peaks) else n
        closes_raw.append(min(range(pk, min(next_pk, n)), key=lambda f: dist[f]))

    # Merge detected opens with saved events; apply minimum inter-event gap.
    # Saved opens take priority: drop detected opens within min_event_gap of a saved one.
    saved_events = _read_events_csv(subject_name)
    saved_opens = sorted(saved_events.get("open", []))
    detected_opens = [f for f in sorted(opens_raw)
                      if not any(abs(f - s) < min_event_gap for s in saved_opens)]
    all_open = sorted(set(detected_opens + saved_opens))
    opens: list[int] = []
    last_f = -min_event_gap
    for f in all_open:
        if f - last_f >= min_event_gap:
            opens.append(f)
            last_f = f

    # --- Detect peak and close events (anchor-based, per video) ---
    # Use saved opens as anchors; fall back to detected opens if none saved.
    anchor_opens = sorted(saved_events.get("open", [])) or opens

    # Minimum peak distance for edge-interval detection (before first / after last anchor).
    edge_min_peak = float(body.get("edge_min_peak", 15.0))

    def _detect_edge_interval(t0: int, t1: int):
        """Detect up to one peak and close in an edge interval.

        Returns (peak_frame, close_frame); either may be None if not clearly detected.
        """
        if t1 - t0 < 5:
            return None, None
        peak_frame = max(range(t0, t1), key=lambda f: dist[f])
        if dist[peak_frame] < edge_min_peak:
            return None, None
        search_end = min(t1, n)
        close_frame = min(range(peak_frame, search_end), key=lambda f: dist[f])
        return peak_frame, close_frame

    # Load trial map to process videos independently.
    trials = build_trial_map(subject_name)
    frame_to_trial: dict[int, int] = {}
    for ti, t in enumerate(trials):
        for f in range(t["start_frame"], t["end_frame"] + 1):
            frame_to_trial[f] = ti

    anchor_by_trial: dict[int, list[int]] = {}
    for f in anchor_opens:
        ti = frame_to_trial.get(f, 0)
        anchor_by_trial.setdefault(ti, []).append(f)

    peaks: list[int] = []
    closes: list[int] = []

    for ti, t in enumerate(trials):
        vid_anchors = sorted(anchor_by_trial.get(ti, []))
        if not vid_anchors:
            continue

        vid_start = t["start_frame"]
        vid_end = t["end_frame"] + 1  # exclusive

        # Before first anchor: optional detection
        pk_f, cl_f = _detect_edge_interval(vid_start, vid_anchors[0])
        if pk_f is not None:
            peaks.append(pk_f)
        if cl_f is not None:
            closes.append(cl_f)

        # Between consecutive anchors: exactly one peak and one close
        for idx in range(len(vid_anchors) - 1):
            t0 = vid_anchors[idx]
            t1 = vid_anchors[idx + 1]
            if t1 - t0 < 5:
                continue
            peak_frame = max(range(t0, t1), key=lambda f: dist[f])
            peaks.append(peak_frame)
            search_end = min(t1, n)
            closes.append(min(range(peak_frame, search_end), key=lambda f: dist[f]))

        # After last anchor: optional detection
        pk_f, cl_f = _detect_edge_interval(vid_anchors[-1], vid_end)
        if pk_f is not None:
            peaks.append(pk_f)
        if cl_f is not None:
            closes.append(cl_f)

    def _merge_with_saved(detected: list[int], saved: list[int], gap: int) -> list[int]:
        """Merge detected events with saved events; saved events always survive.

        Detected events within `gap` frames of any saved event are discarded.
        Remaining detected events are filtered by the gap constraint relative
        to their neighbors; saved events always pass through regardless of spacing.
        """
        saved_set = set(saved)
        # Remove detected events that are within gap of any saved event
        filtered_detected = [f for f in detected if not any(abs(f - s) < gap for s in saved_set)]
        merged = sorted(saved_set | set(filtered_detected))
        out: list[int] = []
        last = -gap
        for f in merged:
            if f in saved_set:
                # Saved events always survive
                out.append(f)
                last = f
            elif f - last >= gap:
                out.append(f)
                last = f
        return out

    peaks  = _merge_with_saved(peaks,  saved_events.get("peak",  []), min_event_gap)
    closes = _merge_with_saved(closes, saved_events.get("close", []), min_event_gap)

    # Filter out events at video edges (first/last 30 frames per trial = ~0.5 sec @ 60fps)
    edge_margin = 30
    valid_frames = set()
    for t in trials:
        for f in range(t["start_frame"] + edge_margin, t["end_frame"] - edge_margin + 1):
            valid_frames.add(f)

    opens  = [f for f in opens if f in valid_frames]
    peaks  = [f for f in peaks if f in valid_frames]
    closes = [f for f in closes if f in valid_frames]

    # Validate event sequence: must follow open > peak > close > (repeat) pattern.
    # Combine all events with their types and sort by frame.
    all_events = []
    for f in opens:
        all_events.append((f, 'open'))
    for f in peaks:
        all_events.append((f, 'peak'))
    for f in closes:
        all_events.append((f, 'close'))
    all_events.sort()

    # Filter to valid sequence: open, then peak, then close, repeat.
    # Peaks can be missing (open > close is valid). Valid transitions:
    # open -> peak, open -> close, peak -> close, close -> open
    valid_events = {'open': [], 'peak': [], 'close': []}
    expected_next = 'open'  # Start by expecting an open

    for frame, etype in all_events:
        if expected_next == 'open':
            if etype == 'open':
                valid_events['open'].append(frame)
                expected_next = 'peak'  # After open, expect peak or close
            # Skip peaks/closes that appear before any open
        elif expected_next == 'peak':
            if etype == 'peak':
                valid_events['peak'].append(frame)
                expected_next = 'close'
            elif etype == 'close':
                # Peak is missing, transition directly to close (valid)
                valid_events['close'].append(frame)
                expected_next = 'open'
            # Skip other opens before close
        elif expected_next == 'close':
            if etype == 'close':
                valid_events['close'].append(frame)
                expected_next = 'open'
            elif etype == 'open':
                # Close is missing? Skip this open (we expect close before next open)
                pass
            # Skip peaks after peak

    opens  = sorted(valid_events['open'])
    peaks  = sorted(valid_events['peak'])
    closes = sorted(valid_events['close'])

    return {
        "open":  opens,
        "peak":  peaks,
        "close": closes,
    }


@router.post("/sessions/{session_id}/save_corrections")
def save_corrections_only(session_id: int) -> dict:
    """Persist session labels to corrections CSV without marking the session committed.

    Used by the refine view's 'Save Corrections' button so the user can save
    mid-session without committing to DLC training data.
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
        raise HTTPException(400, "No labels to save")

    return save_corrections_to_csv(subject_name=subj["name"], session_labels=labels)
