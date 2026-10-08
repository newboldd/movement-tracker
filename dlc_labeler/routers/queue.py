"""Job queue endpoints: what the Jobs page launches, watches and cancels."""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import List

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from ..config import PROJECT_DIR, get_settings
from ..db import get_db_ctx
from ..models import JobLaunch
from ..services.job_history import read_history, summary
from ..services.queue_manager import RESOURCE_MAP, STEP_DEFINITIONS, queue_manager
from ..services.video import build_trial_map

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/queue", tags=["queue"])


@router.get("/steps")
def get_steps() -> dict:
    """The jobs this app can run, plus what the machine can actually do.

    ``dlc_installed`` is what the page uses to decide whether to offer
    the training steps or the "Install DeepLabCut" button — the base
    install deliberately ships without it.
    """
    settings = get_settings()
    return {
        "steps": STEP_DEFINITIONS,
        "gpus": settings.get_available_gpus(),
        "gpu_available": settings.local_gpu_available,
        "dlc_installed": settings.dlc_installed(),
        "dlc_python": settings.python_executable or sys.executable,
        "dlc_requirements": str(PROJECT_DIR / "requirements-dlc.txt"),
    }


@router.post("/launch")
def launch(req: JobLaunch) -> dict:
    """Enqueue a job per subject and return the queue items created.

    One item per subject rather than one for the batch: each gets its own
    log, progress bar and cancel button, and a failure on subject three
    doesn't take the other four with it.
    """
    if req.job_type not in RESOURCE_MAP:
        raise HTTPException(400, f"Unknown job type: {req.job_type}")

    settings = get_settings()

    if req.job_type == "install-dlc":
        if settings.dlc_installed():
            raise HTTPException(400, "DeepLabCut is already installed")
        return {"queued": [queue_manager.enqueue("install-dlc", [])]}

    if not req.subjects:
        raise HTTPException(400, "Select at least one subject")

    if req.job_type in ("train", "refine", "analyze") and not settings.dlc_installed():
        raise HTTPException(
            400, "DeepLabCut is not installed in this environment. Run the "
                 "install step first (or ./setup.sh --with-dlc).")

    extra: dict = {}
    if req.job_type == "mediapipe":
        extra = {
            "reverse": req.reverse,
            "static_image_mode": req.static_image_mode,
            "use_bbox": req.use_bbox,
        }
        if req.trial_idx is not None:
            extra["trial_idx"] = req.trial_idx
        # Name the pass in the queue row so the page can label it.
        extra["pass"] = ("reverse" if req.reverse else
                         "frame-by-frame" if req.static_image_mode else "forward")

    queued = []
    for name in req.subjects:
        try:
            queued.append({
                "subject": name,
                **queue_manager.enqueue(req.job_type, [name],
                                        gpu_index=req.gpu_index,
                                        extra_params=extra),
            })
        except ValueError as e:
            raise HTTPException(400, str(e))

    return {"queued": queued}


@router.get("/state")
def get_state() -> dict:
    """Current lanes, running jobs and recent history."""
    state = queue_manager.get_state()
    # subject_ids is stored as a JSON array; decode it for the page.
    for key in ("cpu_queue", "gpu_queue", "running", "history"):
        for row in state.get(key, []):
            raw = row.get("subject_ids")
            if isinstance(raw, str):
                try:
                    row["subjects"] = json.loads(raw)
                except (ValueError, TypeError):
                    row["subjects"] = []
            params = row.get("extra_params_json") or row.get("params_json")
            if isinstance(params, str):
                try:
                    row["params"] = json.loads(params)
                except (ValueError, TypeError):
                    row["params"] = {}
    return state


@router.post("/cancel/{queue_id}")
def cancel(queue_id: int) -> dict:
    """Cancel a queued or running queue item."""
    if not queue_manager.cancel(queue_id):
        raise HTTPException(400, "Queue item is not cancellable")
    return {"cancelled": True}


@router.get("/subjects")
def queue_subjects() -> List[dict]:
    """Subjects with the per-trial facts the Jobs page needs to choose work.

    Which MediaPipe passes a trial already has, and whether it has been
    analysed, is the difference between re-running everything and running
    only what is missing.
    """
    settings = get_settings()
    with get_db_ctx() as db:
        rows = db.execute(
            "SELECT id, name, stage, camera_mode FROM subjects ORDER BY name"
        ).fetchall()

    out = []
    for row in rows:
        dlc_dir = settings.dlc_path / row["name"]
        try:
            trials = build_trial_map(row["name"],
                                     camera_mode=row.get("camera_mode"))
        except Exception:
            trials = []

        analysis_dirs = [d for d in ("labels_v2", "labels_v1")
                         if (dlc_dir / d).exists()]
        trial_rows = []
        for ti, t in enumerate(trials):
            td = dlc_dir / t["trial_name"]
            trial_rows.append({
                "trial_idx": ti,
                "trial_name": t["trial_name"],
                "frame_count": t["frame_count"],
                "has_forward": (td / "mediapipe.npz").exists(),
                "has_reverse": (td / "mediapipe_reverse.npz").exists(),
                "has_static": (td / "mediapipe_static.npz").exists(),
                "has_best": (td / "mediapipe_combined.npz").exists(),
                "has_analysis": any(
                    list((dlc_dir / d).glob(f"{t['trial_name']}_*.csv"))
                    for d in analysis_dirs),
            })

        out.append({
            "id": row["id"],
            "name": row["name"],
            "stage": row["stage"],
            "trial_count": len(trials),
            "has_project": (dlc_dir / "config.yaml").exists(),
            "has_labeled_data": (dlc_dir / "labeled-data").is_dir() and any(
                (dlc_dir / "labeled-data").glob("round*/CollectedData_*.csv")),
            "has_snapshots": bool(
                list((dlc_dir / "dlc-models-pytorch").rglob("snapshot-*.pt"))
                if (dlc_dir / "dlc-models-pytorch").exists() else []),
            "trials": trial_rows,
        })
    return out


@router.get("/history")
def lifetime_history(
    limit: int = Query(200, ge=1, le=2000),
    job_type: str | None = Query(None),
    status: str | None = Query(None),
) -> dict:
    """The append-only lifetime job history, newest first.

    Separate from the jobs table: this file records every run with the
    app's git version and its stage timings, and it survives the database
    being deleted — which is what makes a past analysis reproducible.
    """
    return {
        "records": read_history(limit=limit, job_type=job_type, status=status),
        "summary": summary(job_type=job_type),
    }


@router.get("/stream")
async def stream_state() -> StreamingResponse:
    """Server-sent stream of the queue state, for live lane/progress updates."""
    async def event_generator():
        last = None
        try:
            while True:
                state = get_state()
                body = json.dumps(state, default=str)
                # Only push on change, so an idle Jobs page costs nothing.
                if body != last:
                    last = body
                    yield f"data: {body}\n\n"
                else:
                    yield ": keep-alive\n\n"
                await asyncio.sleep(1.5)
        except asyncio.CancelledError:
            return

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
