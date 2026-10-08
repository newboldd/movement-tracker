"""Subject CRUD and discovery.

A subject is one person's set of trial videos plus the DeepLabCut
project built from them.  Both live on disk under the data directory, so
``/sync`` can rebuild the whole subject list from the filesystem — the
database is a cache, not the record.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from ..config import get_settings
from ..db import get_db_ctx
from ..models import STAGE_INDEX, SubjectCreate, SubjectUpdate
from ..services.discovery import (
    _count_labeled_frames, _find_deidentified_videos, _find_videos,
    _has_labeled_data, _has_mediapipe, _has_snapshots, infer_stage,
    scan_all_subjects,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/subjects", tags=["subjects"])

# Trial videos are named {Subject}_{Trial}.mp4, optionally with a camera
# suffix for multicam rigs: Con01_L1.mp4, Con01_L1_OS.mp4.
_VIDEO_NAME_RE = re.compile(r"^(?P<subject>.+?)_(?P<trial>[A-Za-z]+\d*)"
                            r"(?:_(?P<camera>[A-Za-z0-9]+))?$")


def _resolve_dlc_path(dlc_dir_value: Optional[str]) -> Optional[Path]:
    """Resolve a stored dlc_dir (a subject name) to an absolute path."""
    if not dlc_dir_value:
        return None
    return get_settings().dlc_path / dlc_dir_value


def _parse_no_face_videos(raw: str | None) -> list[str]:
    """Parse the no_face_videos JSON column (NULL → empty list)."""
    if not raw:
        return []
    try:
        val = json.loads(raw)
        return val if isinstance(val, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _subject_row_to_response(row: dict) -> dict:
    """Decorate a DB row with what is actually on disk."""
    dlc_path = _resolve_dlc_path(row.get("dlc_dir") or row.get("name"))
    exists = bool(dlc_path and dlc_path.exists())
    videos = _find_videos(row["name"]) if row.get("name") else []

    return {
        **row,
        "stage_idx": STAGE_INDEX.get(row.get("stage", "created"), 0),
        "video_count": len(videos),
        "has_project": exists and (dlc_path / "config.yaml").exists(),
        "has_snapshots": _has_snapshots(dlc_path) if exists else False,
        "has_labels": _has_labeled_data(dlc_path) if exists else False,
        "has_mediapipe": _has_mediapipe(dlc_path) if exists else False,
        "labeled_frame_count": _count_labeled_frames(dlc_path) if exists else 0,
        "no_face_videos": _parse_no_face_videos(row.get("no_face_videos")),
    }


def _delete_subject_deps(db, subject_id: int):
    """Delete a subject's dependent rows. Tolerates tables we don't own.

    The database may be shared with the full Movement Tracker app, whose
    tables this fork never creates; deleting from them when they happen
    to exist keeps a shared database from accumulating orphans.
    """
    session_ids = [r["id"] for r in db.execute(
        "SELECT id FROM label_sessions WHERE subject_id = ?",
        (subject_id,)).fetchall()]
    if session_ids:
        placeholders = ",".join("?" * len(session_ids))
        db.execute(
            f"DELETE FROM frame_labels WHERE session_id IN ({placeholders})",
            session_ids)
    db.execute("DELETE FROM label_sessions WHERE subject_id = ?", (subject_id,))
    db.execute("DELETE FROM jobs WHERE subject_id = ?", (subject_id,))
    for table in ("segments", "mp_crop_boxes", "subject_events", "blur_specs",
                  "blur_hand_settings", "face_detections"):
        try:
            db.execute(f"DELETE FROM {table} WHERE subject_id = ?",
                       (subject_id,))
        except Exception:
            pass


# ── Read ───────────────────────────────────────────────────────────────

@router.get("")
def list_subjects() -> List[dict]:
    """List all subjects with their stage and on-disk status."""
    settings = get_settings()
    with get_db_ctx() as db:
        rows = db.execute("SELECT * FROM subjects ORDER BY name").fetchall()
    if not settings.show_example_subject:
        rows = [r for r in rows if r["name"] != "Example"]
    return [_subject_row_to_response(r) for r in rows]


@router.get("/{subject_id}")
def get_subject(subject_id: int) -> dict:
    """Full subject detail: trials, per-trial progress, jobs, sessions."""
    settings = get_settings()
    with get_db_ctx() as db:
        row = db.execute("SELECT * FROM subjects WHERE id = ?",
                         (subject_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Subject not found")
        jobs = db.execute(
            "SELECT * FROM jobs WHERE subject_id = ? "
            "ORDER BY created_at DESC LIMIT 50", (subject_id,)).fetchall()
        sessions = db.execute(
            "SELECT ls.*, COUNT(fl.id) AS label_count FROM label_sessions ls "
            "LEFT JOIN frame_labels fl ON fl.session_id = ls.id "
            "WHERE ls.subject_id = ? GROUP BY ls.id "
            "ORDER BY ls.created_at DESC", (subject_id,)).fetchall()

    resp = _subject_row_to_response(row)
    subject_name = row["name"]
    dlc_path = settings.dlc_path / subject_name

    from ..services.video import build_trial_map
    try:
        trials = build_trial_map(subject_name,
                                 camera_mode=row.get("camera_mode"))
    except Exception as e:
        logger.warning("Could not build trial map for %s: %s", subject_name, e)
        trials = []

    resp["videos"] = _find_videos(subject_name)
    resp["trials"] = [t["trial_name"] for t in trials]
    resp["jobs"] = jobs
    resp["label_sessions"] = sessions

    # Per-trial progress, read off the filesystem so it stays true even
    # for a project copied in from another machine.
    has_corrections = (dlc_path / "corrections").exists()
    analysis_dirs = [d for d in ("labels_v2", "labels_v1")
                     if (dlc_path / d).exists()]
    trial_status = []
    for t in trials:
        stem = t["trial_name"]
        trial_dir = dlc_path / stem
        trial_status.append({
            "name": stem,
            "frame_count": t["frame_count"],
            "status": {
                "mp_forward": (trial_dir / "mediapipe.npz").exists(),
                "mp_reverse": (trial_dir / "mediapipe_reverse.npz").exists(),
                "mp_static": (trial_dir / "mediapipe_static.npz").exists(),
                "mp_best": (trial_dir / "mediapipe_combined.npz").exists(),
                "analysis": any(
                    list((dlc_path / d).glob(f"{stem}_*.csv"))
                    for d in analysis_dirs),
                "corrections": has_corrections and bool(
                    list((dlc_path / "corrections").glob(f"{stem}_*.csv"))),
            },
        })
    resp["trial_status"] = trial_status
    return resp


# ── Write ──────────────────────────────────────────────────────────────

@router.post("", status_code=201)
def create_subject(req: SubjectCreate) -> dict:
    """Register a subject. Its videos are found by the {Subject}_{Trial} pattern."""
    settings = get_settings()
    with get_db_ctx() as db:
        if db.execute("SELECT id FROM subjects WHERE name = ?",
                      (req.name,)).fetchone():
            raise HTTPException(400, f"Subject '{req.name}' already exists")
        cur = db.execute(
            """INSERT INTO subjects (name, stage, dlc_dir, video_pattern, camera_mode)
               VALUES (?, 'created', ?, ?, ?)""",
            (req.name, req.name, req.video_pattern,
             req.camera_mode or settings.default_camera_mode),
        )
        row = db.execute("SELECT * FROM subjects WHERE id = ?",
                         (cur.lastrowid,)).fetchone()
    return _subject_row_to_response(row)


@router.patch("/{subject_id}")
def update_subject(subject_id: int, req: SubjectUpdate) -> dict:
    """Update a subject's camera mode, calibration assignment or notes."""
    with get_db_ctx() as db:
        row = db.execute("SELECT * FROM subjects WHERE id = ?",
                         (subject_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Subject not found")

        if req.camera_mode is not None:
            if req.camera_mode not in ("single", "stereo", "multicam"):
                raise HTTPException(
                    400, "camera_mode must be single, stereo or multicam")
            db.execute("UPDATE subjects SET camera_mode = ?, "
                       "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                       (req.camera_mode, subject_id))

        if req.camera_name is not None:
            db.execute("UPDATE subjects SET camera_name = ?, "
                       "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                       (req.camera_name or None, subject_id))
            # The calibration cache is keyed by camera name, so a changed
            # assignment has to invalidate it or 3D stays wrong until restart.
            from ..services.calibration import clear_calibration_cache
            clear_calibration_cache()

        if req.no_face_videos is not None:
            db.execute("UPDATE subjects SET no_face_videos = ?, "
                       "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                       (json.dumps(req.no_face_videos)
                        if req.no_face_videos else None, subject_id))

        if req.notes is not None:
            db.execute("UPDATE subjects SET notes = ?, "
                       "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                       (req.notes or None, subject_id))

        updated = db.execute("SELECT * FROM subjects WHERE id = ?",
                             (subject_id,)).fetchone()
    return _subject_row_to_response(updated)


@router.delete("/{subject_id}")
def delete_subject(subject_id: int,
                   purge: bool = Query(False, description="Also delete the "
                                       "DLC project directory on disk")) -> dict:
    """Remove a subject from the database.

    Videos are never deleted: they are the irreplaceable input, and a
    mis-click here would otherwise destroy a recording session.  The DLC
    project directory — labels, snapshots, predictions — is deleted only
    with ``purge=true``, and a later ``/sync`` re-registers the subject
    from the videos either way.
    """
    with get_db_ctx() as db:
        row = db.execute("SELECT * FROM subjects WHERE id = ?",
                         (subject_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Subject not found")

        dlc_deleted = False
        if purge:
            dlc_path = _resolve_dlc_path(row.get("dlc_dir") or row["name"])
            if dlc_path and dlc_path.exists():
                shutil.rmtree(dlc_path)
                dlc_deleted = True

        _delete_subject_deps(db, subject_id)
        db.execute("DELETE FROM subjects WHERE id = ?", (subject_id,))

    return {
        "deleted_from_db": True,
        "dlc_deleted": dlc_deleted,
        "videos_kept": True,
        "message": (f"Subject '{row['name']}' removed"
                    + (" with its DLC project" if dlc_deleted else
                       " (DLC project and videos left on disk)")),
    }


# ── Discovery ──────────────────────────────────────────────────────────

def _subjects_from_videos() -> dict[str, set[str]]:
    """Group the video directory by subject name.

    Returns ``{subject: {trial, ...}}``.  Discovering from the videos —
    not only from existing DLC projects — is what lets someone drop a new
    recording into the folder, press Sync, and start labeling.
    """
    settings = get_settings()
    video_dir = settings.video_path
    if not video_dir.is_dir():
        return {}

    found: dict[str, set[str]] = {}
    for vf in sorted(video_dir.iterdir()):
        if not vf.is_file() or vf.suffix.lower() not in (".mp4", ".mov", ".avi"):
            continue
        m = _VIDEO_NAME_RE.match(vf.stem)
        if not m:
            continue
        found.setdefault(m.group("subject"), set()).add(m.group("trial"))
    return found


@router.post("/sync")
def sync_from_filesystem() -> dict:
    """Rebuild the subject list from the data directory.

    Two sources, in this order: every DLC project under ``dlc/``, and
    every ``{Subject}_{Trial}`` video under the video directory.  A
    subject's stage is re-read from its project artifacts and only ever
    moved forward, so a database that has fallen behind what is on disk
    catches up without losing a manually set stage.
    """
    settings = get_settings()
    discovered = {s["name"]: s for s in scan_all_subjects()}

    # Videos without a DLC project yet — a new recording, ready to label.
    for name, trials in _subjects_from_videos().items():
        if name in discovered:
            continue
        dlc_path = settings.dlc_path / name
        discovered[name] = {
            "name": name,
            "stage": infer_stage(dlc_path) if dlc_path.exists() else "created",
            "dlc_dir": name,
            "camera_name": None,
            "video_count": len(trials),
        }

    created = updated = 0
    with get_db_ctx() as db:
        for name, subj in discovered.items():
            existing = db.execute(
                "SELECT id, stage FROM subjects WHERE name = ?",
                (name,)).fetchone()
            if existing:
                if STAGE_INDEX.get(subj["stage"], 0) > \
                        STAGE_INDEX.get(existing["stage"], 0):
                    db.execute(
                        "UPDATE subjects SET stage = ?, dlc_dir = ?, "
                        "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                        (subj["stage"], subj["dlc_dir"], existing["id"]))
                    updated += 1
            else:
                db.execute(
                    """INSERT INTO subjects
                           (name, stage, dlc_dir, camera_name, camera_mode)
                       VALUES (?, ?, ?, ?, ?)""",
                    (name, subj["stage"], subj["dlc_dir"],
                     subj.get("camera_name"), settings.default_camera_mode))
                created += 1

        # Drop rows whose subject has neither a project nor any video
        # left on disk.  Anything still on disk stays listed.
        stale = 0
        for row in db.execute("SELECT id, name FROM subjects").fetchall():
            if row["name"] in discovered:
                continue
            if _find_videos(row["name"]) or _find_deidentified_videos(row["name"]):
                continue
            try:
                _delete_subject_deps(db, row["id"])
                db.execute("DELETE FROM subjects WHERE id = ?", (row["id"],))
                stale += 1
            except Exception as e:
                logger.warning("Could not remove stale subject %s: %s",
                               row["name"], e)

    return {"created": created, "updated": updated, "removed": stale,
            "total": len(discovered)}
