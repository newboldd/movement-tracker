"""FastAPI application: mount routers, serve static files, startup discovery."""

import logging
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .config import get_settings
from .db import init_db
from .routers import subjects, labeling, pipeline, jobs, results, settings, filebrowser, video_tools, batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="DLC Labeler", version="0.2.0")

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
            tmux_session = job.get("tmux_session")
            remote_host = job.get("remote_host")

            if tmux_session and remote_host and remote_cfg:
                # Check if tmux session is still alive on remote
                from .services.remote import _check_tmux_alive, _read_remote_status, remote_train_monitor
                from .services.jobs import registry

                if _check_tmux_alive(remote_cfg, tmux_session):
                    # Tmux alive — resume monitoring
                    logger.info(f"Job {job['id']}: tmux '{tmux_session}' alive, resuming monitor")

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
                    # Tmux dead — check final status.json
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
                            logger.info(f"Job {job['id']}: tmux exited but status=completed, spawning download")
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

            # No tmux or tmux dead with no good status — mark failed
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


@app.get("/")
def index():
    """Serve the dashboard page."""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/labeling")
def labeling_page():
    """Serve the labeling page."""
    return FileResponse(str(STATIC_DIR / "labeling.html"))


@app.get("/results")
def results_page():
    """Serve the results page."""
    return FileResponse(str(STATIC_DIR / "results.html"))


@app.get("/settings")
def settings_page():
    """Serve the settings page."""
    return FileResponse(str(STATIC_DIR / "settings.html"))


@app.get("/onboarding")
def onboarding_page():
    """Serve the subject onboarding page."""
    return FileResponse(str(STATIC_DIR / "onboarding.html"))


if __name__ == "__main__":
    import uvicorn
    s = get_settings()
    uvicorn.run("dlc_app.app:app", host=s.host, port=s.port, reload=True)
