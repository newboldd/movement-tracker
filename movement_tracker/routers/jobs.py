"""Job listing, log tailing, and SSE progress streaming."""
from __future__ import annotations

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
                    """SELECT j.status, j.progress_pct, j.error_msg, j.remote_host,
                              j.job_type, s.name AS subject_name
                       FROM jobs j LEFT JOIN subjects s ON j.subject_id = s.id
                       WHERE j.id = ?""",
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


@router.get("/{job_id}/log-stream")
async def stream_job_log(job_id: int) -> StreamingResponse:
    """SSE stream that tails the job log file in real time."""
    with get_db_ctx() as db:
        job = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not job:
        raise HTTPException(404, "Job not found")
    log_path = job.get("log_path")
    if not log_path:
        raise HTTPException(400, "No log file for this job")

    async def event_generator():
        offset = 0
        while True:
            # Read any new content from the log file
            try:
                with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                    f.seek(offset)
                    new_text = f.read()
                    if new_text:
                        offset = f.tell()
                        yield f"data: {json.dumps({'text': new_text})}\n\n"
            except FileNotFoundError:
                pass  # Log file not created yet

            # Check if job is still active
            with get_db_ctx() as db:
                current = db.execute(
                    "SELECT status FROM jobs WHERE id = ?", (job_id,)
                ).fetchone()
            if not current or current["status"] in ("completed", "failed", "cancelled"):
                # Flush any remaining content
                try:
                    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                        f.seek(offset)
                        remaining = f.read()
                        if remaining:
                            yield f"data: {json.dumps({'text': remaining})}\n\n"
                except FileNotFoundError:
                    pass
                status = current["status"] if current else "unknown"
                yield f"data: {json.dumps({'done': True, 'status': status})}\n\n"
                break

            await asyncio.sleep(0.5)

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
