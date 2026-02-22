"""Subject CRUD and dashboard data endpoints."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, HTTPException

from ..config import get_settings
from ..db import get_db_ctx
from ..models import (
    SubjectCreate, SubjectUpdate, SubjectResponse, SubjectDetail,
    STAGE_INDEX,
)
from ..services.discovery import (
    scan_all_subjects, infer_stage, _find_videos, _find_deidentified_videos,
    _has_snapshots, _has_labeled_data, _has_mediapipe, _has_deidentified,
)

router = APIRouter(prefix="/api/subjects", tags=["subjects"])


def _resolve_dlc_path(dlc_dir_value: Optional[str]) -> Optional[Path]:
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
        "has_mediapipe": _has_mediapipe(dlc_path) if dlc_path and dlc_path.exists() else False,
        "has_blur": _has_deidentified(dlc_path) if dlc_path and dlc_path.exists() else False,
    }


@router.get("")
def list_subjects() -> List[dict]:
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


@router.patch("/{subject_id}")
def update_subject(subject_id: int, req: SubjectUpdate) -> dict:
    """Update subject fields (e.g. camera_name)."""
    with get_db_ctx() as db:
        row = db.execute("SELECT * FROM subjects WHERE id = ?", (subject_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Subject not found")
        # camera_name: empty string → NULL
        camera_val = req.camera_name if req.camera_name else None
        db.execute(
            "UPDATE subjects SET camera_name = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (camera_val, subject_id),
        )
        row = db.execute("SELECT * FROM subjects WHERE id = ?", (subject_id,)).fetchone()
    return _subject_row_to_response(row)


@router.delete("/{subject_id}")
def delete_subject(subject_id: int) -> dict:
    """Remove a subject: delete DLC dir, purge from DB if no videos remain.

    Always deletes the DLC directory (config, models, labels — not trial/deidentified videos).
    Only removes the DB record if no trial or deidentified videos exist.
    """
    with get_db_ctx() as db:
        row = db.execute("SELECT * FROM subjects WHERE id = ?", (subject_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Subject not found")

        subject_name = row["name"]
        settings = get_settings()

        # Delete DLC directory
        dlc_path = _resolve_dlc_path(row.get("dlc_dir"))
        dlc_deleted = False
        if dlc_path and dlc_path.exists():
            shutil.rmtree(dlc_path)
            dlc_deleted = True

        # Check for remaining videos
        trial_videos = _find_videos(subject_name)
        deident_videos = _find_deidentified_videos(subject_name)
        has_videos = bool(trial_videos or deident_videos)

        if has_videos:
            # Keep DB entry but clear dlc_dir
            db.execute(
                "UPDATE subjects SET dlc_dir = NULL, stage = 'created', updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (subject_id,),
            )
            return {
                "deleted_from_db": False,
                "dlc_deleted": dlc_deleted,
                "remaining_videos": len(trial_videos) + len(deident_videos),
                "message": f"DLC dir removed. Subject kept — {len(trial_videos)} trial + {len(deident_videos)} deidentified videos remain.",
            }

        # No videos — full purge from DB (cascade)
        session_ids = [r["id"] for r in db.execute(
            "SELECT id FROM label_sessions WHERE subject_id = ?", (subject_id,)
        ).fetchall()]
        if session_ids:
            placeholders = ",".join("?" * len(session_ids))
            db.execute(f"DELETE FROM frame_labels WHERE session_id IN ({placeholders})", session_ids)
        db.execute("DELETE FROM label_sessions WHERE subject_id = ?", (subject_id,))
        db.execute("DELETE FROM jobs WHERE subject_id = ?", (subject_id,))
        db.execute("DELETE FROM subjects WHERE id = ?", (subject_id,))

        return {
            "deleted_from_db": True,
            "dlc_deleted": dlc_deleted,
            "remaining_videos": 0,
            "message": f"Subject '{subject_name}' fully removed.",
        }


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
    removed = 0

    discovered_names = {s["name"] for s in discovered}

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

        # Remove stale subjects: in DB but no DLC dir AND no trial/deidentified videos
        all_db = db.execute("SELECT id, name FROM subjects").fetchall()
        for row in all_db:
            if row["name"] in discovered_names:
                continue
            # Subject has no DLC dir on disk — check for videos
            trial_vids = _find_videos(row["name"])
            deident_vids = _find_deidentified_videos(row["name"])
            if not trial_vids and not deident_vids:
                # Full purge
                session_ids = [r["id"] for r in db.execute(
                    "SELECT id FROM label_sessions WHERE subject_id = ?", (row["id"],)
                ).fetchall()]
                if session_ids:
                    placeholders = ",".join("?" * len(session_ids))
                    db.execute(f"DELETE FROM frame_labels WHERE session_id IN ({placeholders})", session_ids)
                db.execute("DELETE FROM label_sessions WHERE subject_id = ?", (row["id"],))
                db.execute("DELETE FROM jobs WHERE subject_id = ?", (row["id"],))
                db.execute("DELETE FROM subjects WHERE id = ?", (row["id"],))
                removed += 1

    return {"created": created, "updated": updated, "removed": removed, "total": len(discovered)}
