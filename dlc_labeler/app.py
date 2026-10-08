"""DLC Labeler — FastAPI app: routes, static files, startup."""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from .config import get_settings
from .db import get_db_ctx, init_db
from .routers import (
    jobs, labeling, packages, queue, settings as settings_router, subjects,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="DLC Labeler", version="1.0.0")

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Pages, in the order a project moves through them.
PAGES = {
    "/": "index.html",
    "/subjects": "index.html",
    "/label": "label.html",
    "/jobs": "jobs.html",
    "/settings": "settings.html",
}


class _QuietPollFilter(logging.Filter):
    """Keep the polling endpoints out of the access log.

    The Jobs page and the label page poll a few endpoints every second
    or two; left alone they bury every message worth reading.
    """

    _QUIET = ("/api/jobs", "/api/queue/state", "/api/queue/stream")

    def filter(self, record):
        msg = record.getMessage()
        return not any(p in msg for p in self._QUIET)


logging.getLogger("uvicorn.access").addFilter(_QuietPollFilter())


class NoCacheMiddleware(BaseHTTPMiddleware):
    """Never let a browser cache the app's own pages or assets.

    There is no build step and no content hashing, so a cached JS file
    after an update is a page that behaves like the old version with no
    obvious cause.  The cost is one revalidation per asset on a local
    server, which is nothing.
    """

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static") or request.url.path in PAGES:
            response.headers["Cache-Control"] = \
                "no-store, no-cache, must-revalidate, max-age=0"
        return response


app.add_middleware(NoCacheMiddleware)

app.include_router(subjects.router)
app.include_router(labeling.router)
app.include_router(jobs.router)
app.include_router(queue.router)
app.include_router(packages.router)
app.include_router(settings_router.router)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _pid_alive(pid) -> bool:
    import os
    if not pid or int(pid) <= 0:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _recover_orphan_jobs():
    """Fail jobs left 'running' by a previous process, except live ones.

    The server restarts on every code edit under ``--reload``.  Queue-
    backed jobs are reconciled by the queue manager, which can re-attach
    to a surviving subprocess; what is left here are jobs that ran in a
    thread inside the old process (the per-trial cropped MediaPipe run).
    Those cannot survive the restart, and leaving them 'running' shows a
    progress bar that will never move.
    """
    with get_db_ctx() as db:
        stale = db.execute(
            "SELECT j.id, j.job_type, j.pid FROM jobs j "
            "LEFT JOIN job_queue q ON q.job_id = j.id "
            "WHERE j.status IN ('running', 'pending') AND q.id IS NULL"
        ).fetchall()

        failed = 0
        for job in stale:
            if _pid_alive(job.get("pid")):
                continue
            db.execute(
                "UPDATE jobs SET status = 'failed', "
                "error_msg = 'Interrupted by app restart', "
                "finished_at = CURRENT_TIMESTAMP WHERE id = ?", (job["id"],))
            failed += 1

    if failed:
        logger.info("Marked %d interrupted job(s) as failed", failed)


@app.on_event("startup")
def startup():
    """Initialise the database, sync subjects from disk, start the queue."""
    logger.info("Initializing database...")
    init_db()

    s = get_settings()
    logger.info("Data directory: %s", s.dlc_path.parent)
    logger.info("Videos:         %s", s.video_path)

    _recover_orphan_jobs()

    logger.info("Syncing subjects from the data directory...")
    try:
        from .routers.subjects import sync_from_filesystem
        result = sync_from_filesystem()
        logger.info("Discovery: %d new, %d updated, %d removed, %d total",
                    result["created"], result["updated"], result["removed"],
                    result["total"])
    except Exception:
        logger.exception("Subject sync failed; the app will still start")

    from .services.queue_manager import queue_manager
    queue_manager.recover()
    queue_manager.start()


@app.get("/favicon.ico")
def favicon():
    """204 rather than a 404 in the log on every page load."""
    return Response(status_code=204)


@app.get("/.well-known/appspecific/com.chrome.devtools.json")
def chrome_devtools_probe():
    """Chrome DevTools probes this path whenever it is open."""
    return Response(status_code=204)


def _page(name: str) -> FileResponse:
    return FileResponse(str(STATIC_DIR / name))


@app.get("/")
def index():
    return _page("index.html")


@app.get("/subjects")
def subjects_page():
    return _page("index.html")


@app.get("/label")
def label_page():
    return _page("label.html")


@app.get("/jobs")
def jobs_page():
    return _page("jobs.html")


@app.get("/settings")
def settings_page():
    return _page("settings.html")
