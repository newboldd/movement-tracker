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
from .routers import subjects, labeling, pipeline, jobs, results, settings, filebrowser, video_tools, batch, remote_jobs, mano, export, camera_setups

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Movement Tracker", version="0.2.0")


# Disable browser caching for all responses (dev tool — always serve fresh)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class NoCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static") or request.url.path in (
            "/", "/labeling", "/results", "/settings", "/onboarding", "/remote", "/mano", "/videos", "/calibration"
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
app.include_router(mano.router)
app.include_router(export.router)
app.include_router(camera_setups.router)

# Mount static files
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _recover_stale_jobs(s):
    """Handle stale jobs from prior server sessions.

    Key design: when a remote process has already finished, update the DB
    **synchronously** so the status is correct even if a subsequent --reload
    kills any background download thread.  Downloads are best-effort and
    will be re-attempted on the next server restart if interrupted.
    """
    from .db import get_db_ctx
    with get_db_ctx() as db:
        stale_jobs = db.execute(
            "SELECT id, job_type, tmux_session, remote_host, subject_id, log_path FROM jobs "
            "WHERE status IN ('running', 'pending')"
        ).fetchall()

    if not stale_jobs:
        return

    remote_cfg = s.get_remote_config()
    resumed = 0
    failed = 0

    TRAIN_JOB_TYPES = {"train", "analyze_v1", "analyze_v2"}
    PREPROCESS_JOB_TYPES = {"mediapipe", "blur", "mediapipe+blur"}

    for job in stale_jobs:
        session_info = job.get("tmux_session") or ""
        remote_host = job.get("remote_host")
        job_type = job.get("job_type", "")

        # Parse PID from session_info (format: "pid:12345")
        remote_pid = None
        if session_info.startswith("pid:"):
            try:
                remote_pid = int(session_info.split(":")[1])
            except (ValueError, IndexError):
                pass

        if not (session_info and remote_host and remote_cfg):
            # No remote info — mark failed
            with get_db_ctx() as db:
                db.execute(
                    "UPDATE jobs SET status = 'failed', error_msg = 'Server restarted (no remote config)', "
                    "finished_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (job["id"],),
                )
            failed += 1
            continue

        from .services.remote import (
            _check_remote_pid_alive, _read_remote_status,
            remote_train_monitor, resume_preprocess_monitor,
        )
        from .services.jobs import registry

        log_path = job["log_path"] or str(s.dlc_path / ".logs" / f"job_{job['id']}.log")
        proc_alive = bool(remote_pid) and _check_remote_pid_alive(remote_cfg, remote_pid)

        # ── Read remote status (works for both alive and dead processes) ──
        if job_type in PREPROCESS_JOB_TYPES:
            status_file = f"{remote_cfg.work_dir}/preprocess_{job['id']}_status.json"
        elif job_type in TRAIN_JOB_TYPES:
            with get_db_ctx() as db:
                subj = db.execute(
                    "SELECT name FROM subjects WHERE id = ?",
                    (job["subject_id"],),
                ).fetchone()
            status_file = f"{remote_cfg.work_dir}/{subj['name']}/status.json" if subj else None
        else:
            status_file = None

        remote_status = _read_remote_status(remote_cfg, status_file) if status_file else None
        remote_done = remote_status and remote_status.get("status") == "completed"
        remote_failed = remote_status and remote_status.get("status") == "failed"

        # ── Case 1: Remote job completed ──────────────────────────
        if remote_done:
            # Mark completed SYNCHRONOUSLY so DB is correct even if
            # --reload kills the download thread immediately after.
            logger.info(f"Job {job['id']} ({job_type}): remote completed — marking done synchronously")
            with get_db_ctx() as db:
                db.execute(
                    """UPDATE jobs SET status = 'completed', progress_pct = 100,
                       finished_at = CURRENT_TIMESTAMP WHERE id = ?""",
                    (job["id"],),
                )

            # Spawn best-effort background download (if killed, next startup will retry)
            if job_type in PREPROCESS_JOB_TYPES:
                _spawn_preprocess_download(job, remote_cfg, log_path, registry)
            elif job_type in TRAIN_JOB_TYPES and subj:
                _spawn_train_download(job, subj, s, remote_cfg, log_path, registry)

            resumed += 1
            continue

        # ── Case 2: Remote job explicitly failed ──────────────────
        if remote_failed:
            error = remote_status.get("error", "Unknown error")
            logger.info(f"Job {job['id']} ({job_type}): remote failed — {error}")
            with get_db_ctx() as db:
                db.execute(
                    "UPDATE jobs SET status = 'failed', error_msg = ?, "
                    "finished_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (f"Remote: {error}", job["id"]),
                )
            failed += 1
            continue

        # ── Case 3: Remote process still alive ────────────────────
        if proc_alive:
            logger.info(f"Job {job['id']} ({job_type}): PID {remote_pid} alive, resuming monitor")
            if job_type in PREPROCESS_JOB_TYPES:
                thread = threading.Thread(
                    target=resume_preprocess_monitor,
                    kwargs=dict(
                        job_id=job["id"], cfg=remote_cfg,
                        log_path=log_path, registry=registry,
                    ),
                    daemon=True,
                )
                thread.start()
                registry._threads[job["id"]] = thread
            elif job_type in TRAIN_JOB_TYPES and subj:
                subject_name = subj["name"]
                local_dlc_dir = s.dlc_path / subject_name

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
                        job_id=job["id"], cfg=remote_cfg,
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

        # ── Case 4: Process dead + no completed/failed status ─────
        with get_db_ctx() as db:
            db.execute(
                "UPDATE jobs SET status = 'failed', error_msg = 'Remote process exited without status', "
                "finished_at = CURRENT_TIMESTAMP WHERE id = ?",
                (job["id"],),
            )
        failed += 1

    logger.info(f"Startup recovery: {resumed} resumed, {failed} marked failed")

    # ── Resume any pending downloads for previously-completed jobs ──
    _resume_pending_downloads(s, remote_cfg)


def _spawn_preprocess_download(job, remote_cfg, log_path, registry):
    """Spawn a best-effort background download for a completed preprocess job."""
    from .services.remote import resume_preprocess_monitor
    thread = threading.Thread(
        target=resume_preprocess_monitor,
        kwargs=dict(
            job_id=job["id"], cfg=remote_cfg,
            log_path=log_path, registry=registry,
        ),
        daemon=True,
    )
    thread.start()
    registry._threads[job["id"]] = thread


def _spawn_train_download(job, subj, settings, remote_cfg, log_path, registry):
    """Spawn a best-effort background download for a completed train job."""
    from .services.remote import remote_train_monitor
    subject_name = subj["name"]
    local_dlc_dir = settings.dlc_path / subject_name

    def _on_dl_complete(jid, returncode, _subj=subject_name):
        if returncode == 0:
            from .services.dlc_wrapper import fix_project_path
            try:
                fix_project_path(_subj)
            except Exception:
                pass
            from .db import get_db_ctx
            with get_db_ctx() as db2:
                db2.execute(
                    "UPDATE subjects SET stage = 'trained', updated_at = CURRENT_TIMESTAMP "
                    "WHERE name = ?", (_subj,),
                )

    thread = threading.Thread(
        target=remote_train_monitor,
        kwargs=dict(
            job_id=job["id"], cfg=remote_cfg,
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


def _resume_pending_downloads(s, remote_cfg):
    """Check for completed preprocess jobs whose results weren't fully downloaded.

    Compares each job's subject list against local .deidentified markers.
    If any subjects are missing their blurred videos, spawns a download thread.
    This ensures downloads eventually complete even after repeated --reload kills.
    """
    if not remote_cfg:
        return

    import json as _json
    from .db import get_db_ctx
    from .services.jobs import registry

    with get_db_ctx() as db:
        # Find completed preprocess jobs that have a queue entry with subject list
        rows = db.execute(
            """SELECT j.id, j.job_type, j.log_path, jq.subject_ids
               FROM jobs j
               JOIN job_queue jq ON jq.job_id = j.id
               WHERE j.status = 'completed'
               AND j.job_type IN ('blur', 'mediapipe', 'mediapipe+blur')
               AND j.finished_at > datetime('now', '-7 days')"""
        ).fetchall()

    for row in rows:
        try:
            subjects = _json.loads(row["subject_ids"])
        except (TypeError, _json.JSONDecodeError):
            continue

        job_type = row["job_type"]
        has_blur = job_type in ("blur", "mediapipe+blur")

        if has_blur:
            # Check which subjects are missing the .deidentified marker
            missing = [
                subj for subj in subjects
                if not (s.dlc_path / subj / ".deidentified").exists()
            ]
            if not missing:
                continue

            logger.info(
                f"Job {row['id']} ({job_type}): {len(missing)}/{len(subjects)} "
                f"subjects missing downloads, resuming"
            )
            log_path = row["log_path"] or str(s.dlc_path / ".logs" / f"job_{row['id']}.log")

            from .services.remote import resume_preprocess_monitor
            thread = threading.Thread(
                target=resume_preprocess_monitor,
                kwargs=dict(
                    job_id=row["id"], cfg=remote_cfg,
                    log_path=log_path, registry=registry,
                ),
                daemon=True,
            )
            thread.start()
            registry._threads[row["id"]] = thread
            # Only resume one download at a time to avoid hammering SSH
            break


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
    _recover_stale_jobs(s)

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

    def _get_or_create_session(db, sid: int, stype: str) -> Optional[int]:
        """Return an existing active session of the same type, or create one."""
        # Reuse existing active session (prefer one with labels)
        existing = db.execute(
            """SELECT ls.id, COUNT(fl.id) AS label_count
               FROM label_sessions ls
               LEFT JOIN frame_labels fl ON fl.session_id = ls.id
               WHERE ls.subject_id = ? AND ls.session_type = ? AND ls.status = 'active'
               GROUP BY ls.id
               ORDER BY label_count DESC, ls.id DESC
               LIMIT 1""",
            (sid, stype),
        ).fetchone()
        if existing:
            return existing["id"]
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
        """Pick the best labeling mode based on the subject's pipeline stage.

        Default to events for all subjects past the initial labeling stages.
        """
        if stage in _PRE_DLC_STAGES:
            return "initial"
        # All post-analysis stages default to events mode
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
                sid = _get_or_create_session(db, subject, session_type)
                if sid:
                    return RedirectResponse(
                        url=f"/labeling?session={sid}", status_code=302
                    )
                else:
                    logger.warning("labeling-select: _create_session returned None for subject=%s type=%s", subject, session_type)
            except Exception as exc:
                logger.exception("labeling-select: session creation failed for subject=%s type=%s: %s", subject, session_type, exc)
            return RedirectResponse(url="/", status_code=302)

        # ── No subject param — fall back to most recent active session ──
        # Prefer sessions with labels (actual work), then most recent
        recent_session = db.execute(
            """SELECT ls.id, COUNT(fl.id) AS label_count
               FROM label_sessions ls
               LEFT JOIN frame_labels fl ON fl.session_id = ls.id
               WHERE ls.status = 'active'
               GROUP BY ls.id
               ORDER BY label_count DESC, ls.created_at DESC
               LIMIT 1"""
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
            sid = _get_or_create_session(db, subject_id, session_type)
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


@app.get("/videos")
def videos_page():
    """Serve the videos viewer page."""
    return FileResponse(str(STATIC_DIR / "videos.html"))


@app.get("/mano")
def mano_page():
    """Serve the MANO 3D hand model viewer page."""
    return FileResponse(str(STATIC_DIR / "mano.html"))


@app.get("/calibration")
def calibration_page():
    """Serve the camera calibration page."""
    return FileResponse(str(STATIC_DIR / "calibration.html"))


if __name__ == "__main__":
    import uvicorn
    s = get_settings()
    uvicorn.run("movement_tracker.app:app", host=s.host, port=s.port, reload=True)
