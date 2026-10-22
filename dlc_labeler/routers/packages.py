"""Labeling packages: list what is on disk, and bring labels home.

A package is a folder of frames that left this machine to be labeled
somewhere else.  The Label page opens one directly (see
``services/labelpack``); this router is the other half — what the person
who sent it out does when the folder comes back.
"""
from __future__ import annotations

import logging
import threading

from fastapi import APIRouter, Body, HTTPException, Query

from ..config import get_settings
from ..db import get_db_ctx
from ..services.labelpack import (
    import_package_labels, list_packages, read_frames_csv,
    read_package_labels,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/packages", tags=["packages"])


@router.get("")
def list_all() -> list[dict]:
    """Every package in the packages directory, with how far along it is.

    ``labeled`` against ``image_count`` is the number someone actually
    wants: how much of what was sent out has come back done.
    """
    settings = get_settings()
    out = []
    for pkg in list_packages():
        manifest = pkg["manifest"]
        subject = manifest.get("subject")
        with get_db_ctx() as db:
            known = bool(subject and db.execute(
                "SELECT 1 FROM subjects WHERE name = ?", (subject,)).fetchone())
        out.append({
            "name": pkg["name"],
            "dir": pkg["dir"],
            "subject": subject,
            "subject_exists": known,
            "created_at": manifest.get("created_at"),
            "image_count": manifest.get("image_count") or len(
                read_frames_csv(pkg["dir"])),
            "labeled": len(read_package_labels(pkg["dir"])),
            "bodyparts": manifest.get("bodyparts") or settings.bodyparts,
            "cameras": manifest.get("cameras") or [],
            "directories": manifest.get("directories") or [],
        })
    return out


@router.post("/{name}/import")
def import_labels(
    name: str,
    overwrite: bool = Query(False, description="Replace labels that differ"),
    dry_run: bool = Query(False, description="Report what would happen, "
                                             "changing nothing"),
    subject: str | None = Query(None, description="Import into this subject "
                                                  "instead of the one the "
                                                  "package names"),
) -> dict:
    """Merge a returned package's labels into its subject's label session.

    Run it with ``dry_run`` first: the report says how many frames would
    land, which ones conflict with labels already there, and which
    images never came back — all before anything is written.
    """
    try:
        return import_package_labels(name, target_subject=subject,
                                     overwrite=overwrite, dry_run=dry_run)
    except ValueError as e:
        raise HTTPException(400, str(e))


# Export runs on a thread and reports here; the page polls it.  In
# memory because it is live state of this process, meaningless after a
# restart and not worth a table.
_export_lock = threading.Lock()
_export_progress: dict[str, dict] = {}


@router.post("/export")
def export_frames(body: dict = Body(default=None)) -> dict:
    """Write a subject's selected frames as a package to send out.

    The frames are the ones the Label page's k-means already chose, so
    what a collaborator labels is exactly what you would have labeled.
    Runs in the background: it seeks every trial and writes an image per
    frame per camera.
    """
    from ..services.frameselect import load_selection
    from ..services.labelpack import export_package

    body = body or {}
    subject = (body.get("subject") or "").strip()
    if not subject:
        raise HTTPException(400, "subject is required")

    with get_db_ctx() as db:
        row = db.execute("SELECT id FROM subjects WHERE name = ?",
                         (subject,)).fetchone()
    if not row:
        raise HTTPException(404, f"No subject named {subject}")
    if not load_selection(subject):
        raise HTTPException(
            400, "This subject has no frame selection yet — open it on the "
                 "Label page and pick frames first.")

    with _export_lock:
        running = _export_progress.get(subject)
        if running and running.get("status") == "running":
            return {"status": "already-running", **running}
        _export_progress[subject] = {"status": "running", "pct": 0.0,
                                     "phase": "Starting", "error": None}

    def _set(**kw):
        with _export_lock:
            _export_progress.setdefault(subject, {}).update(kw)

    def _run():
        try:
            result = export_package(
                subject, get_settings().packages_path,
                package_name=body.get("package_name") or None,
                seed=body.get("seed"),
                progress=lambda pct, phase="": _set(
                    pct=min(float(pct), 99.0), phase=phase or "Writing frames"),
            )
            _set(status="done", pct=100.0, phase="Done", error=None,
                 package_dir=result["package_dir"],
                 image_count=result["image_count"])
        except Exception as e:
            logger.exception("Package export failed for %s", subject)
            _set(status="failed", phase="Failed", error=str(e))

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "running", "subject": subject}


@router.get("/export/progress")
def export_progress(subject: str = Query(...)) -> dict:
    """How far a running export has got, for the page to poll."""
    with _export_lock:
        state = _export_progress.get(subject)
    return state or {"status": "idle", "pct": 0.0, "phase": "", "error": None}
