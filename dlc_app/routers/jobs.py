"""Job listing, log tailing, and SSE progress streaming."""

import asyncio
import json
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from ..db import get_db_ctx
from ..services.jobs import registry

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("")
def list_jobs(
    subject_id: Optional[int] = Query(None),
    status: Optional[str] = Query(None),
) -> List[dict]:
    """List jobs, optionally filtered by subject or status."""
    with get_db_ctx() as db:
        query = "SELECT * FROM jobs WHERE 1=1"
        params = []
        if subject_id is not None:
            query += " AND subject_id = ?"
            params.append(subject_id)
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC"
        jobs = db.execute(query, params).fetchall()
    return jobs


@router.get("/{job_id}")
def get_job(job_id: int) -> dict:
    """Get a single job with log tail."""
    with get_db_ctx() as db:
        job = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not job:
        raise HTTPException(404, "Job not found")

    # Add log tail
    if job.get("log_path"):
        job["log_tail"] = registry.get_log_tail(job["log_path"], 50)
    return job


@router.get("/{job_id}/stream")
async def stream_job(job_id: int) -> StreamingResponse:
    """SSE stream for job progress updates."""
    with get_db_ctx() as db:
        job = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not job:
        raise HTTPException(404, "Job not found")

    async def event_generator():
        while True:
            with get_db_ctx() as db:
                current = db.execute(
                    "SELECT status, progress_pct, error_msg FROM jobs WHERE id = ?",
                    (job_id,),
                ).fetchone()

            if not current:
                break

            data = json.dumps(current)
            yield f"data: {data}\n\n"

            if current["status"] in ("completed", "failed", "cancelled"):
                break

            await asyncio.sleep(1)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/{job_id}/cancel")
def cancel_job(job_id: int) -> dict:
    """Cancel a running job."""
    with get_db_ctx() as db:
        job = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not job:
        raise HTTPException(404, "Job not found")

    if job["status"] != "running":
        raise HTTPException(400, "Job is not running")

    success = registry.cancel(job_id)
    return {"cancelled": success}
