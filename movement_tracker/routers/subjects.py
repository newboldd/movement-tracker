"""Subject CRUD and dashboard data endpoints."""
from __future__ import annotations

import json
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


def _parse_no_face_videos(raw: str | None) -> list[str]:
    """Parse no_face_videos JSON column (NULL → empty list)."""
    if not raw:
        return []
    try:
        val = json.loads(raw)
        return val if isinstance(val, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _all_no_face(row: dict) -> bool:
    """True if every video for this subject is marked as no-face."""
    no_face = _parse_no_face_videos(row.get("no_face_videos"))
    if not no_face:
        return False
    videos = _find_videos(row["name"]) if row.get("name") else []
    if not videos:
        return False
    video_stems = {Path(v).stem for v in videos}
    return video_stems.issubset(set(no_face))


def _has_blur_complete(dlc_path: Path, row: dict) -> bool:
    """True if blur is done: either has .deidentified marker or all videos are no-face."""
    if _has_deidentified(dlc_path):
        return True
    return _all_no_face(row)


def _delete_subject_deps(db, subject_id: int):
    """Delete all dependent records for a subject. Safe if tables don't exist."""
    # Delete frame_labels via label_sessions
    session_ids = [r["id"] for r in db.execute(
        "SELECT id FROM label_sessions WHERE subject_id = ?", (subject_id,)
    ).fetchall()]
    if session_ids:
        placeholders = ",".join("?" * len(session_ids))
        db.execute(f"DELETE FROM frame_labels WHERE session_id IN ({placeholders})", session_ids)
    db.execute("DELETE FROM label_sessions WHERE subject_id = ?", (subject_id,))
    db.execute("DELETE FROM segments WHERE subject_id = ?", (subject_id,))
    db.execute("DELETE FROM jobs WHERE subject_id = ?", (subject_id,))
    # Optional tables that may not exist in all schema versions
    for table in ("subject_events", "mp_crop_boxes", "blur_specs",
                  "blur_hand_settings", "hand_protection_segments",
                  "face_detections"):
        try:
            db.execute(f"DELETE FROM {table} WHERE subject_id = ?", (subject_id,))
        except Exception:
            pass


def _subject_row_to_response(row: dict) -> dict:
    """Convert a DB row to SubjectResponse fields."""
    dlc_path = _resolve_dlc_path(row.get("dlc_dir"))
    videos = _find_videos(row["name"]) if row.get("name") else []
    resp = {
        **row,
        "stage_idx": STAGE_INDEX.get(row.get("stage", "created"), 0),
        "video_count": len(videos),
        "has_snapshots": _has_snapshots(dlc_path) if dlc_path and dlc_path.exists() else False,
        "has_labels": _has_labeled_data(dlc_path) if dlc_path and dlc_path.exists() else False,
        "has_mediapipe": _has_mediapipe(dlc_path) if dlc_path and dlc_path.exists() else False,
        "has_blur": _has_blur_complete(dlc_path, row) if dlc_path and dlc_path.exists() else _all_no_face(row),
    }
    resp["no_face_videos"] = _parse_no_face_videos(row.get("no_face_videos"))
    return resp


@router.get("")
def list_subjects() -> List[dict]:
    """List all subjects with stage info."""
    from ..config import get_settings
    settings = get_settings()

    with get_db_ctx() as db:
        rows = db.execute(
            "SELECT * FROM subjects ORDER BY name"
        ).fetchall()

    # Hide Example subject if disabled in settings
    if not settings.show_example_subject:
        rows = [r for r in rows if r["name"] != "Example"]

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
            """INSERT INTO subjects (name, stage, dlc_dir, video_pattern, diagnosis)
               VALUES (?, 'created', ?, ?, ?)""",
            (req.name, dlc_dir, req.video_pattern, req.diagnosis),
        )
        row = db.execute(
            "SELECT * FROM subjects WHERE name = ?", (req.name,)
        ).fetchone()
    return _subject_row_to_response(row)


@router.patch("/{subject_id}")
def update_subject(subject_id: int, req: SubjectUpdate) -> dict:
    """Update subject fields (e.g. name, camera_name, diagnosis)."""
    with get_db_ctx() as db:
        row = db.execute("SELECT * FROM subjects WHERE id = ?", (subject_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Subject not found")

        if req.name is not None and req.name != row["name"]:
            _rename_subject(db, row, req.name)

        if req.camera_mode is not None:
            if req.camera_mode not in ("single", "stereo", "multicam"):
                raise HTTPException(400, "camera_mode must be single, stereo, or multicam")
            db.execute(
                "UPDATE subjects SET camera_mode = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (req.camera_mode, subject_id),
            )

        if req.camera_name is not None:
            # camera_name: empty string → NULL
            camera_val = req.camera_name if req.camera_name else None
            db.execute(
                "UPDATE subjects SET camera_name = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (camera_val, subject_id),
            )
        if req.no_face_videos is not None:
            nfv_json = json.dumps(req.no_face_videos) if req.no_face_videos else None
            db.execute(
                "UPDATE subjects SET no_face_videos = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (nfv_json, subject_id),
            )
            # Clear video.py cache for this subject
            from ..services.video import _no_face_cache
            _no_face_cache.pop(row["name"], None)
        if req.diagnosis is not None:
            db.execute(
                "UPDATE subjects SET diagnosis = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (req.diagnosis, subject_id),
            )
        row = db.execute("SELECT * FROM subjects WHERE id = ?", (subject_id,)).fetchone()
    return _subject_row_to_response(row)


def _rename_subject(db, row, new_name: str):
    """Rename a subject: update DB, rename video files, rename DLC dir."""
    import re
    old_name = row["name"]
    subject_id = row["id"]

    if not re.match(r'^[A-Za-z0-9_]+$', new_name):
        raise HTTPException(400, "Name must be alphanumeric (letters, numbers, underscores)")

    # Check no name collision
    existing = db.execute("SELECT id FROM subjects WHERE name = ? AND id != ?",
                          (new_name, subject_id)).fetchone()
    if existing:
        raise HTTPException(400, f"Subject '{new_name}' already exists")

    settings = get_settings()

    # Rename video files: {old_name}_{trial}.mp4 → {new_name}_{trial}.mp4
    video_dir = settings.video_path
    if video_dir.is_dir():
        for vf in video_dir.iterdir():
            if vf.is_file() and vf.stem.startswith(old_name + "_"):
                suffix = vf.stem[len(old_name):]  # e.g. "_L1" or "_R1_cam0"
                new_path = vf.parent / f"{new_name}{suffix}{vf.suffix}"
                vf.rename(new_path)
        # Also rename deidentified videos
        deident_dir = video_dir / "deidentified"
        if deident_dir.is_dir():
            for vf in deident_dir.iterdir():
                if vf.is_file() and vf.stem.startswith(old_name + "_"):
                    suffix = vf.stem[len(old_name):]
                    new_path = vf.parent / f"{new_name}{suffix}{vf.suffix}"
                    vf.rename(new_path)

    # Rename DLC directory
    dlc_old = settings.dlc_path / old_name
    dlc_new = settings.dlc_path / new_name
    if dlc_old.is_dir() and not dlc_new.exists():
        dlc_old.rename(dlc_new)

    # Update DB
    db.execute(
        "UPDATE subjects SET name = ?, dlc_dir = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (new_name, new_name, subject_id),
    )


@router.delete("/{subject_id}/trials/{trial_label}")
def delete_trial(subject_id: int, trial_label: str) -> dict:
    """Delete a single trial: remove video file(s) and segment record."""
    with get_db_ctx() as db:
        row = db.execute("SELECT * FROM subjects WHERE id = ?", (subject_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Subject not found")

        subject_name = row["name"]
        settings = get_settings()
        video_dir = settings.video_path
        deleted_files = []

        # Find and delete matching video files
        # Trial videos are named {subject}_{trial}.mp4 or {subject}_{trial}_{camera}.mp4
        if video_dir.is_dir():
            for vf in video_dir.iterdir():
                stem = vf.stem
                if vf.is_file() and (
                    stem == f"{subject_name}_{trial_label}" or
                    stem.startswith(f"{subject_name}_{trial_label}_")
                ):
                    vf.unlink()
                    deleted_files.append(str(vf))
            # Also remove deidentified versions
            deident_dir = video_dir / "deidentified"
            if deident_dir.is_dir():
                for vf in deident_dir.iterdir():
                    stem = vf.stem
                    if vf.is_file() and (
                        stem == f"{subject_name}_{trial_label}" or
                        stem.startswith(f"{subject_name}_{trial_label}_")
                    ):
                        vf.unlink()
                        deleted_files.append(str(vf))

        # Remove segment record(s)
        db.execute(
            "DELETE FROM segments WHERE subject_id = ? AND trial_label = ?",
            (subject_id, trial_label),
        )

    return {"deleted_files": deleted_files, "trial_label": trial_label}


@router.delete("/{subject_id}")
def delete_subject(subject_id: int) -> dict:
    """Fully remove a subject: delete all videos, DLC directory, and DB records."""
    with get_db_ctx() as db:
        row = db.execute("SELECT * FROM subjects WHERE id = ?", (subject_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Subject not found")

        subject_name = row["name"]
        settings = get_settings()
        video_dir = settings.video_path
        deleted_files = []

        # Delete DLC directory
        dlc_path = _resolve_dlc_path(row.get("dlc_dir"))
        dlc_deleted = False
        if dlc_path and dlc_path.exists():
            shutil.rmtree(dlc_path)
            dlc_deleted = True

        # Delete trial videos ({subject}_{trial}.mp4 or {subject}_{trial}_{camera}.mp4)
        if video_dir.is_dir():
            prefix_lower = (subject_name + "_").lower()
            for vf in video_dir.iterdir():
                if vf.is_file() and vf.name.lower().startswith(prefix_lower):
                    vf.unlink()
                    deleted_files.append(str(vf))

        # Delete deidentified videos
        deident_dir = video_dir / "deidentified"
        if deident_dir.is_dir():
            prefix_lower = (subject_name + "_").lower()
            for vf in deident_dir.iterdir():
                if vf.is_file() and vf.name.lower().startswith(prefix_lower):
                    vf.unlink()
                    deleted_files.append(str(vf))

        # Full purge from DB
        _delete_subject_deps(db, subject_id)
        db.execute("DELETE FROM subjects WHERE id = ?", (subject_id,))

        return {
            "deleted_from_db": True,
            "dlc_deleted": dlc_deleted,
            "deleted_videos": len(deleted_files),
            "message": f"Subject '{subject_name}' fully removed ({len(deleted_files)} video files, DLC dir {'deleted' if dlc_deleted else 'N/A'}).",
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

        segments = db.execute(
            "SELECT * FROM segments WHERE subject_id = ? ORDER BY created_at",
            (subject_id,),
        ).fetchall()

    resp = _subject_row_to_response(row)
    videos = _find_videos(row["name"])
    # Extract trial names from videos (e.g. "Con07_L1" from "Con07_L1.mp4")
    trials = [Path(v).stem for v in videos]

    resp["videos"] = videos
    resp["trials"] = trials
    resp["segments"] = segments
    resp["jobs"] = jobs
    resp["label_sessions"] = sessions
    return resp


@router.post("/auto-assign-diagnosis")
def auto_assign_diagnosis() -> dict:
    """Auto-assign diagnosis to all subjects based on name patterns."""
    from ..config import get_settings

    settings = get_settings()
    diagnosis_groups = settings.diagnosis_groups or ["Control", "MSA", "PD", "PSP"]

    def infer_diagnosis(subject_name: str) -> str:
        """Infer diagnosis from subject name using first 3 characters or keyword matching."""
        name_upper = subject_name.upper()
        name_lower = subject_name.lower()

        # Try exact keyword matching first
        for diagnosis in diagnosis_groups:
            diag_upper = diagnosis.upper()
            if diag_upper in name_upper:
                return diagnosis

        # Try first 3 characters matching
        if len(subject_name) >= 3:
            prefix = subject_name[:3].upper()
            for diagnosis in diagnosis_groups:
                if diagnosis.upper().startswith(prefix):
                    return diagnosis

        # Default to first diagnosis group
        return diagnosis_groups[0] if diagnosis_groups else "Control"

    with get_db_ctx() as db:
        subjects = db.execute("SELECT id, name, diagnosis FROM subjects").fetchall()
        updated = 0
        assignments = {}

        for subject in subjects:
            new_diagnosis = infer_diagnosis(subject["name"])
            assignments[subject["name"]] = new_diagnosis

            if subject["diagnosis"] != new_diagnosis:
                db.execute(
                    "UPDATE subjects SET diagnosis = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (new_diagnosis, subject["id"]),
                )
                updated += 1

    return {
        "updated": updated,
        "assignments": assignments,
        "total_subjects": len(assignments),
    }


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
                try:
                    _delete_subject_deps(db, row["id"])
                    db.execute("DELETE FROM subjects WHERE id = ?", (row["id"],))
                    removed += 1
                except Exception as e:
                    logger.warning(f"Could not remove stale subject {row['name']}: {e}")

    return {"created": created, "updated": updated, "removed": removed, "total": len(discovered)}
