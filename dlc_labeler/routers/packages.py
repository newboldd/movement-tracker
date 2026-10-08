"""Labeling packages: list what is on disk, and bring labels home.

A package is a folder of frames that left this machine to be labeled
somewhere else.  The Label page opens one directly (see
``services/labelpack``); this router is the other half — what the person
who sent it out does when the folder comes back.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

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
