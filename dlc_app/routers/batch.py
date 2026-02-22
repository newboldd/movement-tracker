"""Batch operations: run preprocessing steps on all/selected subjects."""
from __future__ import annotations

import threading
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..config import get_settings
from ..db import get_db_ctx
from ..services.jobs import registry
from ..services.remote import remote_preprocess_batch

router = APIRouter(prefix="/api/batch", tags=["batch"])


class BatchPreprocessRequest(BaseModel):
    steps: List[str]       # "mediapipe", "blur"
    subjects: List[str] = []  # empty = all subjects with videos


VALID_PREPROCESS_STEPS = {"mediapipe", "blur"}


@router.post("/preprocess")
def batch_preprocess(req: BatchPreprocessRequest) -> dict:
    """Launch batch remote preprocessing (MediaPipe and/or blur) for subjects.

    Requires remote SSH to be configured in settings.
    """
    settings = get_settings()
    remote_cfg = settings.get_remote_config()
    if not remote_cfg:
        raise HTTPException(
            400,
            "Remote SSH not configured. Set remote host, python, and work dir in Settings.",
        )

    invalid = set(req.steps) - VALID_PREPROCESS_STEPS
    if invalid:
        raise HTTPException(400, f"Invalid steps: {invalid}. Valid: {VALID_PREPROCESS_STEPS}")
    if not req.steps:
        raise HTTPException(400, "At least one step required")

    # Create job record (not tied to a specific subject)
    with get_db_ctx() as db:
        # Use subject_id=0 for batch jobs (no single subject)
        # First ensure a placeholder exists or use the first subject
        log_dir = settings.dlc_path / ".logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        step_names = "+".join(req.steps)
        log_path = str(log_dir / f"job_batch_{step_names}.log")

        # For batch jobs, use subject_id of the first subject or create a sentinel
        if req.subjects:
            subj = db.execute(
                "SELECT id FROM subjects WHERE name = ?", (req.subjects[0],)
            ).fetchone()
            subject_id = subj["id"] if subj else 1
        else:
            # Use the first subject's ID as a placeholder
            subj = db.execute("SELECT id FROM subjects ORDER BY id LIMIT 1").fetchone()
            subject_id = subj["id"] if subj else 1

        db.execute(
            """INSERT INTO jobs (subject_id, job_type, status, remote_host, log_path)
               VALUES (?, ?, 'pending', ?, ?)""",
            (subject_id, f"batch_{step_names}", remote_cfg.host, log_path),
        )
        job = db.execute(
            "SELECT * FROM jobs ORDER BY id DESC LIMIT 1"
        ).fetchone()

    # Launch in background thread
    thread = threading.Thread(
        target=remote_preprocess_batch,
        kwargs=dict(
            job_id=job["id"],
            cfg=remote_cfg,
            steps=req.steps,
            subjects=req.subjects,
            log_path=log_path,
            registry=registry,
        ),
        daemon=True,
    )
    thread.start()
    registry._threads[job["id"]] = thread

    with get_db_ctx() as db:
        db.execute(
            "UPDATE jobs SET status = 'running', started_at = CURRENT_TIMESTAMP WHERE id = ?",
            (job["id"],),
        )

    return {
        "job_id": job["id"],
        "status": "running",
        "steps": req.steps,
        "subjects": req.subjects or "all",
        "remote": True,
    }
