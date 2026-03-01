"""Remote job queue API: launch, monitor, cancel, redownload."""
from __future__ import annotations

import asyncio
import json
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..db import get_db_ctx
from ..services.queue_manager import queue_manager, STEP_DEFINITIONS, RESOURCE_MAP

router = APIRouter(prefix="/api/remote", tags=["remote"])


class LaunchRequest(BaseModel):
    job_type: str
    subject_ids: List[int] = []
    subjects: List[str] = []


class RedownloadRequest(BaseModel):
    job_type: str
    subject_ids: List[int] = []
    subjects: List[str] = []


@router.get("/steps")
def list_steps() -> list[dict]:
    """Return available remote step definitions."""
    return STEP_DEFINITIONS


@router.post("/launch")
def launch_job(req: LaunchRequest) -> dict:
    """Enqueue a new remote job."""
    if req.job_type not in RESOURCE_MAP:
        raise HTTPException(400, f"Unknown job type: {req.job_type}")

    # Resolve subject names from IDs if needed
    subject_names = list(req.subjects)
    if req.subject_ids and not subject_names:
        with get_db_ctx() as db:
            placeholders = ",".join("?" * len(req.subject_ids))
            rows = db.execute(
                f"SELECT name FROM subjects WHERE id IN ({placeholders})",
                req.subject_ids,
            ).fetchall()
            subject_names = [r["name"] for r in rows]

    if not subject_names:
        raise HTTPException(400, "No subjects specified")

    result = queue_manager.enqueue(req.job_type, req.subject_ids, subject_names)
    return result


@router.get("/queue")
def get_queue() -> dict:
    """Get current queue state."""
    return queue_manager.get_state()


@router.post("/cancel/{queue_id}")
def cancel_queue_item(queue_id: int) -> dict:
    """Cancel a queued or running item."""
    success = queue_manager.cancel(queue_id)
    if not success:
        raise HTTPException(404, "Queue item not found or not cancellable")
    return {"cancelled": True}


@router.post("/redownload")
def redownload_results(req: RedownloadRequest) -> dict:
    """Re-download results for completed jobs (download-only, no re-compute)."""
    from ..config import get_settings
    from ..services.remote import remote_train_monitor, remote_preprocess_download
    from ..services.jobs import registry
    import threading

    settings = get_settings()
    remote_cfg = settings.get_remote_config()
    if not remote_cfg:
        raise HTTPException(400, "Remote not configured")

    # Resolve subject names
    subject_names = list(req.subjects)
    if req.subject_ids and not subject_names:
        with get_db_ctx() as db:
            placeholders = ",".join("?" * len(req.subject_ids))
            rows = db.execute(
                f"SELECT name FROM subjects WHERE id IN ({placeholders})",
                req.subject_ids,
            ).fetchall()
            subject_names = [r["name"] for r in rows]

    if not subject_names:
        raise HTTPException(400, "No subjects specified")

    # Resolve a subject_id for the jobs record
    with get_db_ctx() as db:
        subj = db.execute(
            "SELECT id FROM subjects WHERE name = ?", (subject_names[0],)
        ).fetchone()
    if not subj:
        raise HTTPException(404, f"Subject {subject_names[0]} not found")

    log_dir = settings.dlc_path / ".logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    subj_label = "_".join(subject_names[:3])
    if len(subject_names) > 3:
        subj_label += f"_+{len(subject_names) - 3}"
    log_path = str(log_dir / f"job_redownload_{req.job_type}_{subj_label}.log")

    with get_db_ctx() as db:
        db.execute(
            """INSERT INTO jobs (subject_id, job_type, status, remote_host, log_path)
               VALUES (?, ?, 'running', ?, ?)""",
            (subj["id"], f"redownload_{req.job_type}", remote_cfg.host, log_path),
        )
        job = db.execute("SELECT * FROM jobs ORDER BY id DESC LIMIT 1").fetchone()

    if req.job_type in ("mediapipe", "blur", "mediapipe+blur"):
        # CPU: re-download preprocessing results
        steps = []
        if req.job_type in ("mediapipe", "mediapipe+blur"):
            steps.append("mediapipe")
        if req.job_type in ("blur", "mediapipe+blur"):
            steps.append("blur")

        thread = threading.Thread(
            target=remote_preprocess_download,
            kwargs=dict(
                job_id=job["id"],
                cfg=remote_cfg,
                steps=steps,
                subjects=subject_names,
                log_path=log_path,
                registry=registry,
            ),
            daemon=True,
        )
        thread.start()
        registry._threads[job["id"]] = thread

        return {"job_id": job["id"], "status": "running"}

    if req.job_type in ("train", "analyze_v1", "analyze_v2"):
        # GPU: re-download by resuming monitor (download-only phase)
        subject_name = subject_names[0]
        is_v1 = req.job_type == "analyze_v1"
        labels_dir_name = "labels_v1" if is_v1 else "labels_v2"
        iteration = 0 if is_v1 else None

        thread = threading.Thread(
            target=remote_train_monitor,
            kwargs=dict(
                job_id=job["id"],
                cfg=remote_cfg,
                local_dlc_dir=settings.dlc_path / subject_name,
                subject_name=subject_name,
                log_path=log_path,
                progress_parser=None,
                on_complete=None,
                registry=registry,
                resume=True,
                labels_dir_name=labels_dir_name,
                iteration=iteration,
            ),
            daemon=True,
        )
        thread.start()
        registry._threads[job["id"]] = thread

        return {"job_id": job["id"], "status": "running"}

    raise HTTPException(400, f"Re-download not supported for job type: {req.job_type}")


@router.get("/stream")
async def stream_queue() -> StreamingResponse:
    """SSE stream that emits queue state changes every 2s."""
    async def event_generator():
        last_state = None
        while True:
            state = queue_manager.get_state()
            state_json = json.dumps(state, default=str)
            if state_json != last_state:
                last_state = state_json
                yield f"data: {state_json}\n\n"
            await asyncio.sleep(2)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
