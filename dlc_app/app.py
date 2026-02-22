"""FastAPI application: mount routers, serve static files, startup discovery."""

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .config import get_settings
from .db import init_db
from .routers import subjects, labeling, pipeline, jobs, results, settings, filebrowser, video_tools

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="DLC Finger-Tapping Pipeline", version="0.2.0")

# Mount routers
app.include_router(subjects.router)
app.include_router(labeling.router)
app.include_router(pipeline.router)
app.include_router(jobs.router)
app.include_router(results.router)
app.include_router(settings.router)
app.include_router(filebrowser.router)
app.include_router(video_tools.router)

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

    # Mark orphaned "running" jobs as failed (from prior server sessions)
    from .db import get_db_ctx
    with get_db_ctx() as db:
        stale = db.execute(
            "UPDATE jobs SET status = 'failed', error_msg = 'Server restarted', "
            "finished_at = CURRENT_TIMESTAMP WHERE status IN ('running', 'pending')"
        )
        if stale.rowcount:
            logger.info(f"Cleaned up {stale.rowcount} stale jobs from prior session")

    logger.info("Syncing subjects from filesystem...")
    from .routers.subjects import sync_from_filesystem
    result = sync_from_filesystem()
    logger.info(f"Discovery: {result['created']} new, {result['updated']} updated, {result['total']} total")


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
