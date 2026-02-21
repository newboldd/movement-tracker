"""Subject CRUD and dashboard data endpoints."""

from pathlib import Path
from fastapi import APIRouter, HTTPException

from ..config import get_settings
from ..db import get_db_ctx
from ..models import (
    SubjectCreate, SubjectResponse, SubjectDetail,
    STAGE_INDEX,
)
from ..services.discovery import scan_all_subjects, infer_stage, _find_videos, _has_snapshots, _has_labeled_data

router = APIRouter(prefix="/api/subjects", tags=["subjects"])


def _resolve_dlc_path(dlc_dir_value: str | None) -> Path | None:
    """Resolve a dlc_dir value (subject name) to absolute path."""
    if not dlc_dir_value:
        return None
    settings = get_settings()
    return settings.dlc_path / dlc_dir_value


def _subject_row_to_response(row: dict) -> dict:
    """Convert a DB row to SubjectResponse fields."""
    dlc_path = _resolve_dlc_path(row.get("dlc_dir"))
    videos = _find_videos(row["name"]) if row.get("name") else []
    return {
        **row,
        "stage_idx": STAGE_INDEX.get(row.get("stage", "created"), 0),
        "video_count": len(videos),
        "has_snapshots": _has_snapshots(dlc_path) if dlc_path and dlc_path.exists() else False,
        "has_labels": _has_labeled_data(dlc_path) if dlc_path and dlc_path.exists() else False,
    }


@router.get("")
def list_subjects() -> list[dict]:
    """List all subjects with stage info."""
    with get_db_ctx() as db:
        rows = db.execute(
            "SELECT * FROM subjects ORDER BY name"
        ).fetchall()
    return [_subject_row_to_response(r) for r in rows]


@router.post("", status_code=201)
def create_subject(req: SubjectCreate) -> dict:
    """Create a new subject entry."""
    # Store just the subject name as dlc_dir (relative)
    dlc_dir = req.name
    with get_db_ctx() as db:
        # Check if already exists
        existing = db.execute(
            "SELECT id FROM subjects WHERE name = ?", (req.name,)
        ).fetchone()
        if existing:
            raise HTTPException(400, f"Subject '{req.name}' already exists")

        db.execute(
            """INSERT INTO subjects (name, stage, dlc_dir, video_pattern)
               VALUES (?, 'created', ?, ?)""",
            (req.name, dlc_dir, req.video_pattern),
        )
        row = db.execute(
            "SELECT * FROM subjects WHERE name = ?", (req.name,)
        ).fetchone()
    return _subject_row_to_response(row)


@router.get("/{subject_id}")
def get_subject(subject_id: int) -> dict:
    """Get full subject detail including jobs and sessions."""
    with get_db_ctx() as db:
        row = db.execute(
            "SELECT * FROM subjects WHERE id = ?", (subject_id,)
        ).fetchone()
        if not row:
            raise HTTPException(404, "Subject not found")

        jobs = db.execute(
            "SELECT * FROM jobs WHERE subject_id = ? ORDER BY created_at DESC",
            (subject_id,),
        ).fetchall()

        sessions = db.execute(
            "SELECT * FROM label_sessions WHERE subject_id = ? ORDER BY created_at DESC",
            (subject_id,),
        ).fetchall()

    resp = _subject_row_to_response(row)
    videos = _find_videos(row["name"])
    # Extract trial names from videos (e.g. "Con07_L1" from "Con07_L1.mp4")
    trials = [Path(v).stem for v in videos]

    resp["videos"] = videos
    resp["trials"] = trials
    resp["jobs"] = jobs
    resp["label_sessions"] = sessions
    return resp


@router.post("/sync")
def sync_from_filesystem() -> dict:
    """Re-scan dlc/ directories and sync subjects table."""
    discovered = scan_all_subjects()
    created = 0
    updated = 0

    with get_db_ctx() as db:
        for subj in discovered:
            existing = db.execute(
                "SELECT id, stage FROM subjects WHERE name = ?", (subj["name"],)
            ).fetchone()

            if existing:
                # Update stage if filesystem shows more progress
                fs_stage_idx = STAGE_INDEX.get(subj["stage"], 0)
                db_stage_idx = STAGE_INDEX.get(existing["stage"], 0)
                if fs_stage_idx > db_stage_idx:
                    db.execute(
                        """UPDATE subjects SET stage = ?, dlc_dir = ?,
                           updated_at = CURRENT_TIMESTAMP WHERE id = ?""",
                        (subj["stage"], subj["dlc_dir"], existing["id"]),
                    )
                    updated += 1
            else:
                db.execute(
                    """INSERT INTO subjects (name, stage, dlc_dir, camera_name)
                       VALUES (?, ?, ?, ?)""",
                    (subj["name"], subj["stage"], subj["dlc_dir"], subj["camera_name"]),
                )
                created += 1

    return {"created": created, "updated": updated, "total": len(discovered)}
