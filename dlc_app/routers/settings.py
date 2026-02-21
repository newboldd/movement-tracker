"""Settings API: read, update, and status check."""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import List, Optional

from ..config import get_settings, PROJECT_DIR

router = APIRouter(prefix="/api/settings", tags=["settings"])

# DLC install state (module-level, survives across requests)
_dlc_install = {"running": False, "log": [], "status": None, "error": None}


class SettingsUpdate(BaseModel):
    video_dir: Optional[str] = None
    dlc_dir: Optional[str] = None
    calibration_3d_config: Optional[str] = None
    python_executable: Optional[str] = None
    camera_names: Optional[List[str]] = None
    bodyparts: Optional[List[str]] = None
    dlc_scorer: Optional[str] = None
    dlc_date: Optional[str] = None
    dlc_net_type: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None


@router.get("")
def get_all_settings() -> dict:
    """Get current settings."""
    return get_settings().to_dict()


@router.put("")
def update_settings(req: SettingsUpdate) -> dict:
    """Update settings and save to disk."""
    settings = get_settings()
    data = req.model_dump(exclude_none=True)
    settings.update(data)
    return settings.to_dict()


@router.get("/status")
def settings_status() -> dict:
    """Check if the app is configured."""
    settings = get_settings()
    issues = []

    if not settings.video_dir:
        issues.append("video_dir not set")
    elif not Path(settings.video_dir).exists():
        issues.append(f"video_dir does not exist: {settings.video_dir}")

    if not settings.dlc_dir:
        issues.append("dlc_dir not set")
    elif not Path(settings.dlc_dir).exists():
        issues.append(f"dlc_dir does not exist: {settings.dlc_dir}")

    if not settings.python_executable:
        issues.append("python_executable not set")

    return {
        "configured": settings.is_configured,
        "has_calibration": bool(settings.calibration_3d_config),
        "issues": issues,
    }


@router.get("/browse")
def browse_directory(path: Optional[str] = Query(None)) -> dict:
    """List directories at the given path for a folder picker.

    Returns the current path and a list of child directories.
    Starts at the project directory if no path is given.
    """
    if not path:
        base = PROJECT_DIR
    else:
        base = Path(path)

    base = base.resolve()

    if not base.exists() or not base.is_dir():
        return {"path": str(base), "dirs": [], "error": "Directory not found"}

    dirs = []
    try:
        for entry in sorted(base.iterdir()):
            if entry.is_dir() and not entry.name.startswith("."):
                dirs.append(entry.name)
    except PermissionError:
        return {"path": str(base), "dirs": [], "error": "Permission denied"}

    # Parent directory (unless at filesystem root)
    parent = str(base.parent) if base.parent != base else None

    return {
        "path": str(base),
        "parent": parent,
        "dirs": dirs,
    }


@router.get("/dlc-status")
def dlc_install_status() -> dict:
    """Check if DeepLabCut is installed and get install progress."""
    settings = get_settings()
    python = settings.python_executable
    installed = False
    version = None
    try:
        # Use importlib.metadata instead of importing deeplabcut — DLC import
        # is very slow (loads PyTorch etc.) and can exceed subprocess timeout.
        result = subprocess.run(
            [python, "-c",
             "from importlib.metadata import version; print(version('deeplabcut'))"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            installed = True
            version = result.stdout.strip()
    except Exception:
        pass

    return {
        "installed": installed,
        "version": version,
        "install_running": _dlc_install["running"],
        "install_status": _dlc_install["status"],
        "install_error": _dlc_install["error"],
        "install_log": _dlc_install["log"][-20:],  # last 20 lines
    }


@router.post("/install-dlc")
def install_dlc() -> dict:
    """Start installing DeepLabCut[pytorch] in a background thread."""
    if _dlc_install["running"]:
        return {"status": "already_running"}

    settings = get_settings()
    python = settings.python_executable

    _dlc_install["running"] = True
    _dlc_install["log"] = []
    _dlc_install["status"] = "installing"
    _dlc_install["error"] = None

    thread = threading.Thread(target=_run_dlc_install, args=(python,), daemon=True)
    thread.start()

    return {"status": "started"}


def _run_dlc_install(python: str):
    """Run pip install deeplabcut[pytorch] and capture output."""
    try:
        proc = subprocess.Popen(
            [python, "-m", "pip", "install", "deeplabcut[pytorch]", "tensorflow", "tensorpack"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        for line in proc.stdout:
            _dlc_install["log"].append(line.rstrip())
            # Keep log bounded
            if len(_dlc_install["log"]) > 500:
                _dlc_install["log"] = _dlc_install["log"][-200:]

        proc.wait()

        if proc.returncode == 0:
            _dlc_install["status"] = "completed"
        else:
            _dlc_install["status"] = "failed"
            _dlc_install["error"] = f"pip exited with code {proc.returncode}"

    except Exception as e:
        _dlc_install["status"] = "failed"
        _dlc_install["error"] = str(e)
    finally:
        _dlc_install["running"] = False
