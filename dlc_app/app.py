"""FastAPI application: mount routers, serve static files, startup discovery."""

import logging
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .config import get_settings
from .db import init_db
from .routers import subjects, labeling, pipeline, jobs, results, settings, filebrowser, video_tools, batch, remote_jobs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="DLC Labeler", version="0.2.0")


# Disable browser caching for all responses (dev tool — always serve fresh)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class NoCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static") or request.url.path in (
            "/", "/labeling", "/results", "/settings", "/onboarding", "/remote"
        ):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return response


app.add_middleware(NoCacheMiddleware)

# Mount routers
app.include_router(subjects.router)
app.include_router(labeling.router)
app.include_router(pipeline.router)
app.include_router(jobs.router)
app.include_router(results.router)
app.include_router(settings.router)
app.include_router(filebrowser.router)
app.include_router(video_tools.router)
app.include_router(batch.router)
app.include_router(remote_jobs.router)

# Mount static files
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
def startup():
    """Initialize database and sync subjects from filesystem."""
    logger.info("Initializing database...")
    init_db()

    s = get_settings()
    if not s.is_configured:
        logger.info("App not configured. Visit /settings to set up paths.")
        return

    # Handle stale jobs from prior server sessions
    from .db import get_db_ctx
    with get_db_ctx() as db:
        stale_jobs = db.execute(
            "SELECT id, tmux_session, remote_host, subject_id, log_path FROM jobs "
            "WHERE status IN ('running', 'pending')"
        ).fetchall()

    if stale_jobs:
        remote_cfg = s.get_remote_config()
        resumed = 0
        failed = 0

        for job in stale_jobs:
            session_info = job.get("tmux_session") or ""
            remote_host = job.get("remote_host")

            # Parse PID from session_info (format: "pid:12345" or legacy "dlc_job_N")
            remote_pid = None
            if session_info.startswith("pid:"):
                try:
                    remote_pid = int(session_info.split(":")[1])
                except (ValueError, IndexError):
                    pass

            if session_info and remote_host and remote_cfg:
                # Check if remote process is still alive
                from .services.remote import _check_remote_pid_alive, _read_remote_status, remote_train_monitor
                from .services.jobs import registry

                if remote_pid and _check_remote_pid_alive(remote_cfg, remote_pid):
                    # Process alive — resume monitoring
                    logger.info(f"Job {job['id']}: remote PID {remote_pid} alive, resuming monitor")

                    # Look up subject name for this job
                    with get_db_ctx() as db:
                        subj = db.execute(
                            "SELECT name FROM subjects WHERE id = ?",
                            (job["subject_id"],),
                        ).fetchone()

                    if subj:
                        subject_name = subj["name"]
                        local_dlc_dir = s.dlc_path / subject_name
                        log_path = job["log_path"] or str(s.dlc_path / ".logs" / f"job_train_{job['subject_id']}.log")

                        def _on_resume_complete(jid, returncode, _subj=subject_name):
                            if returncode == 0:
                                from .services.dlc_wrapper import fix_project_path
                                try:
                                    fix_project_path(_subj)
                                except Exception:
                                    pass
                                with get_db_ctx() as db2:
                                    db2.execute(
                                        "UPDATE subjects SET stage = 'trained', updated_at = CURRENT_TIMESTAMP "
                                        "WHERE name = ?", (_subj,),
                                    )

                        thread = threading.Thread(
                            target=remote_train_monitor,
                            kwargs=dict(
                                job_id=job["id"],
                                cfg=remote_cfg,
                                local_dlc_dir=local_dlc_dir,
                                subject_name=subject_name,
                                log_path=log_path,
                                progress_parser=None,
                                on_complete=_on_resume_complete,
                                registry=registry,
                                resume=True,
                            ),
                            daemon=True,
                        )
                        thread.start()
                        registry._threads[job["id"]] = thread
                        resumed += 1
                        continue
                else:
                    # Process dead or PID unknown — check final status.json
                    with get_db_ctx() as db:
                        subj = db.execute(
                            "SELECT name FROM subjects WHERE id = ?",
                            (job["subject_id"],),
                        ).fetchone()
                    if subj:
                        status_file = f"{remote_cfg.work_dir}/{subj['name']}/status.json"
                        remote_status = _read_remote_status(remote_cfg, status_file)
                        if remote_status and remote_status.get("status") == "completed":
                            # Completed while we were down — spawn download-only thread
                            logger.info(f"Job {job['id']}: process exited but status=completed, spawning download")
                            subject_name = subj["name"]
                            local_dlc_dir = s.dlc_path / subject_name
                            log_path = job["log_path"] or str(s.dlc_path / ".logs" / f"job_train_{job['subject_id']}.log")

                            def _on_dl_complete(jid, returncode, _subj=subject_name):
                                if returncode == 0:
                                    from .services.dlc_wrapper import fix_project_path
                                    try:
                                        fix_project_path(_subj)
                                    except Exception:
                                        pass
                                    with get_db_ctx() as db2:
                                        db2.execute(
                                            "UPDATE subjects SET stage = 'trained', updated_at = CURRENT_TIMESTAMP "
                                            "WHERE name = ?", (_subj,),
                                        )

                            thread = threading.Thread(
                                target=remote_train_monitor,
                                kwargs=dict(
                                    job_id=job["id"],
                                    cfg=remote_cfg,
                                    local_dlc_dir=local_dlc_dir,
                                    subject_name=subject_name,
                                    log_path=log_path,
                                    progress_parser=None,
                                    on_complete=_on_dl_complete,
                                    registry=registry,
                                    resume=True,
                                ),
                                daemon=True,
                            )
                            thread.start()
                            registry._threads[job["id"]] = thread
                            resumed += 1
                            continue

            # No remote session or process dead with no good status — mark failed
            with get_db_ctx() as db:
                db.execute(
                    "UPDATE jobs SET status = 'failed', error_msg = 'Server restarted', "
                    "finished_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (job["id"],),
                )
            failed += 1

        logger.info(f"Startup recovery: {resumed} resumed, {failed} marked failed")

    logger.info("Syncing subjects from filesystem...")
    from .routers.subjects import sync_from_filesystem
    result = sync_from_filesystem()
    removed = result.get('removed', 0)
    logger.info(f"Discovery: {result['created']} new, {result['updated']} updated, {removed} removed, {result['total']} total")

    # Start queue manager
    from .services.queue_manager import queue_manager
    queue_manager.recover()
    queue_manager.start()


@app.get("/")
def index():
    """Serve the dashboard page."""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/labeling")
def labeling_page():
    """Serve the labeling page."""
    return FileResponse(str(STATIC_DIR / "labeling.html"))


@app.get("/labeling-select")
def labeling_select_page(subject: Optional[int] = None, mode: Optional[str] = None):
    """Smart redirect to labeling page.

    If ``subject`` is given, create a session for that subject with the
    requested ``mode`` (or auto-detect the best mode from the subject's
    pipeline stage).  Otherwise fall back to the most recent session or
    create one for the first subject.
    """
    from fastapi.responses import RedirectResponse
    from .db import get_db_ctx

    # Stages that indicate no DLC predictions yet → default to 'initial'
    _PRE_DLC_STAGES = {
        "created", "videos_linked", "prelabeled", "labeling", "labeled",
        "committed", "training", "training_dataset_created", "trained",
    }
    # Stages that have DLC predictions but no corrections → default to 'corrections'
    _PRE_CORRECTIONS_STAGES = {"analyzed", "refined"}
    # Valid session types
    _VALID_MODES = {"initial", "refine", "corrections", "events", "final"}

    def _create_session(db, sid: int, stype: str) -> Optional[int]:
        db.execute(
            "INSERT INTO label_sessions (subject_id, session_type) "
            "VALUES (?, ?)",
            (sid, stype),
        )
        db.commit()
        row = db.execute(
            "SELECT id FROM label_sessions WHERE subject_id = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (sid,),
        ).fetchone()
        return row["id"] if row else None

    def _smart_mode(stage: str) -> str:
        """Pick the best labeling mode based on the subject's pipeline stage."""
        if stage in _PRE_DLC_STAGES:
            return "initial"
        if stage in _PRE_CORRECTIONS_STAGES:
            return "corrections"
        # corrected, events_partial, events_complete, complete, etc.
        return "events"

    with get_db_ctx() as db:
        # ── Subject explicitly requested ──────────────────────────
        if subject is not None:
            row = db.execute(
                "SELECT * FROM subjects WHERE id = ?", (subject,)
            ).fetchone()
            if not row:
                return RedirectResponse(url="/", status_code=302)

            # Determine session type
            if mode and mode in _VALID_MODES:
                session_type = mode
            else:
                session_type = _smart_mode(row["stage"] or "created")

            try:
                sid = _create_session(db, subject, session_type)
                if sid:
                    return RedirectResponse(
                        url=f"/labeling?session={sid}", status_code=302
                    )
                else:
                    logger.warning("labeling-select: _create_session returned None for subject=%s type=%s", subject, session_type)
            except Exception as exc:
                logger.exception("labeling-select: session creation failed for subject=%s type=%s: %s", subject, session_type, exc)
            return RedirectResponse(url="/", status_code=302)

        # ── No subject param — fall back to most recent session ───
        recent_session = db.execute(
            "SELECT id FROM label_sessions "
            "ORDER BY created_at DESC LIMIT 1"
        ).fetchone()

        if recent_session:
            return RedirectResponse(
                url=f"/labeling?session={recent_session['id']}", status_code=302
            )

        # No session found — get first subject
        first_subject = db.execute(
            "SELECT * FROM subjects ORDER BY id LIMIT 1"
        ).fetchone()

        if not first_subject:
            return RedirectResponse(url="/", status_code=302)

        subject_id = first_subject["id"]
        session_type = _smart_mode(first_subject["stage"] or "created")

        try:
            sid = _create_session(db, subject_id, session_type)
            if sid:
                return RedirectResponse(
                    url=f"/labeling?session={sid}", status_code=302
                )
        except Exception:
            pass

        return RedirectResponse(url="/", status_code=302)


@app.get("/results")
def results_page():
    """Serve the results page."""
    return FileResponse(str(STATIC_DIR / "results.html"))


@app.get("/settings")
def settings_page():
    """Serve the settings page."""
    return FileResponse(str(STATIC_DIR / "settings.html"))


@app.get("/remote")
def remote_page():
    """Serve the remote jobs page."""
    return FileResponse(str(STATIC_DIR / "remote.html"))


@app.get("/onboarding")
def onboarding_page():
    """Serve the subject onboarding page."""
    return FileResponse(str(STATIC_DIR / "onboarding.html"))


if __name__ == "__main__":
    import uvicorn
    s = get_settings()
    uvicorn.run("dlc_app.app:app", host=s.host, port=s.port, reload=True)
