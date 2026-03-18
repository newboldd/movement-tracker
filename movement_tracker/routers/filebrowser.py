"""File browser API for navigating the filesystem to select source videos."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException, Query

from ..config import get_settings

router = APIRouter(prefix="/api/files", tags=["filebrowser"])

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

# Default starting locations for the file browser
def _get_start_locations() -> list[dict]:
    """Get common starting locations for browsing."""
    settings = get_settings()
    locations = []

    if settings.video_dir:
        locations.append({"name": "Videos (configured)", "path": settings.video_dir})

    # Common user locations
    home = Path.home()
    for name, subdir in [
        ("Desktop", "Desktop"),
        ("Videos", "Videos"),
        ("Downloads", "Downloads"),
    ]:
        p = home / subdir
        if p.exists():
            locations.append({"name": name, "path": str(p)})

    return locations


@router.get("")
def list_directory(
    path: str = Query("", description="Directory path to list. Empty = show start locations."),
) -> dict:
    """List directory contents: subdirectories and video files.

    When path is empty, returns a list of starting locations instead.
    """
    if not path:
        return {"locations": _get_start_locations(), "items": [], "path": ""}

    dir_path = Path(path)
    if not dir_path.exists():
        raise HTTPException(404, f"Path not found: {path}")
    if not dir_path.is_dir():
        raise HTTPException(400, f"Not a directory: {path}")

    dirs = []
    videos = []
    try:
        for entry in dir_path.iterdir():
            if entry.name.startswith("."):
                continue

            if entry.is_dir():
                dirs.append({
                    "name": entry.name,
                    "path": str(entry),
                    "type": "dir",
                })
            elif entry.is_file() and entry.suffix.lower() in VIDEO_EXTENSIONS:
                try:
                    stat = entry.stat()
                    size = stat.st_size
                    ctime = stat.st_ctime
                except OSError:
                    size = 0
                    ctime = 0.0
                videos.append({
                    "name": entry.name,
                    "path": str(entry),
                    "type": "video",
                    "size": size,
                    "size_mb": round(size / (1024 * 1024), 1),
                    "created": datetime.fromtimestamp(ctime, tz=timezone.utc).isoformat(),
                    "created_ts": ctime,
                })
    except PermissionError:
        raise HTTPException(403, f"Permission denied: {path}")

    # Dirs alphabetical, videos by creation date (newest first)
    dirs.sort(key=lambda d: d["name"].lower())
    videos.sort(key=lambda v: v["created_ts"], reverse=True)
    items = dirs + videos

    # Build breadcrumbs
    parts = dir_path.parts
    breadcrumbs = []
    for i in range(len(parts)):
        crumb_path = str(Path(*parts[:i + 1]))
        # On Windows, the first part is like 'C:\', join properly
        if i == 0:
            crumb_path = parts[0]
        breadcrumbs.append({"name": parts[i], "path": crumb_path})

    return {
        "path": str(dir_path),
        "parent": str(dir_path.parent) if dir_path.parent != dir_path else "",
        "breadcrumbs": breadcrumbs,
        "items": items,
    }
