"""Settings API: read, update, and status check."""

import os
import sys
import threading
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional

from ..config import get_settings, DATA_DIR, BOOTSTRAP_DATA_DIR_FILE

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/settings", tags=["settings"])


class SettingsUpdate(BaseModel):
    video_dir: Optional[str] = None
    dlc_dir: Optional[str] = None
    calibration_3d_config: Optional[str] = None
    python_executable: Optional[str] = None
    default_camera_mode: Optional[str] = None
    camera_names: Optional[list[str]] = None
    bodyparts: Optional[list[str]] = None
    dlc_scorer: Optional[str] = None
    dlc_date: Optional[str] = None
    dlc_net_type: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    remote_host: Optional[str] = None
    remote_python: Optional[str] = None
    remote_work_dir: Optional[str] = None
    remote_ssh_key: Optional[str] = None
    remote_ssh_port: Optional[int] = None
    calibrations: Optional[Dict[str, str]] = None
    prefer_deidentified: Optional[bool] = None
    show_tutorials: Optional[bool] = None
    show_example_subject: Optional[bool] = None
    diagnosis_groups: Optional[list[str]] = None
    event_types: Optional[list[dict]] = None


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
    # Clear calibration cache so new paths take effect
    from ..services.calibration import clear_calibration_cache
    clear_calibration_cache()
    return settings.to_dict()


@router.get("/status")
def settings_status() -> dict:
    """Check if the app is configured and return system capabilities."""
    settings = get_settings()
    issues = []

    if not settings.python_executable:
        issues.append("python_executable not set")

    # Ensure DLC directory exists
    settings.dlc_path.mkdir(parents=True, exist_ok=True)

    # Get GPU information
    local_gpu_available = settings.local_gpu_available
    gpus = settings.get_available_gpus() if local_gpu_available else []

    # Try to get CUDA version if available
    cuda_version = None
    try:
        import torch
        if torch.version.cuda:
            cuda_version = torch.version.cuda
    except (ImportError, AttributeError):
        pass

    return {
        "configured": settings.is_configured,
        "has_calibration": bool(settings.calibrations) or bool(settings.calibration_3d_config),
        "remote_enabled": settings.remote_enabled,
        "issues": issues,
        "local_gpu_available": local_gpu_available,
        "gpus": gpus,
        "cuda_version": cuda_version,
    }


class CalibrationValidate(BaseModel):
    path: str


@router.post("/validate-calibration")
def validate_calibration(req: CalibrationValidate) -> dict:
    """Check that a calibration YAML file exists and contains K1."""
    import cv2
    p = Path(req.path)
    if not p.exists():
        return {"valid": False, "error": f"File not found: {req.path}"}
    try:
        fs = cv2.FileStorage(str(p), cv2.FILE_STORAGE_READ)
        k1 = fs.getNode("K1").mat()
        fs.release()
        if k1 is None:
            return {"valid": False, "error": "No K1 matrix found in file"}
        return {"valid": True, "error": None}
    except Exception as e:
        return {"valid": False, "error": str(e)}


# ── Data directory (DB lives here) ─────────────────────────────────────────

class DataDirReq(BaseModel):
    path: str


@router.get("/data-dir")
def get_data_dir() -> dict:
    """Return the current data dir + the override source.

    Source is ``env`` (MT_DATA_DIR set), ``bootstrap`` (saved here),
    or ``default`` (PROJECT_DIR fallback).
    """
    env_val = os.environ.get("MT_DATA_DIR") or None
    boot_val = None
    if BOOTSTRAP_DATA_DIR_FILE.is_file():
        try:
            boot_val = BOOTSTRAP_DATA_DIR_FILE.read_text().strip() or None
        except OSError:
            boot_val = None
    if env_val:
        source = "env"
    elif boot_val:
        source = "bootstrap"
    else:
        source = "default"
    return {
        "current": str(DATA_DIR),
        "bootstrap": boot_val,
        "env_override": env_val,
        "source": source,
        "bootstrap_file": str(BOOTSTRAP_DATA_DIR_FILE),
    }


@router.post("/data-dir")
def set_data_dir(req: DataDirReq) -> dict:
    """Persist a new data directory and re-launch the server.

    The path is written to ``~/.movement_tracker/data_dir``; on the
    next process start ``config.DATA_DIR`` resolves to it.  We
    immediately re-exec the running interpreter so the DB switches
    to the new path -- a hot rebind would leave dozens of modules
    holding the OLD DATA_DIR through their top-level imports.
    """
    raw = (req.path or "").strip()
    if not raw:
        raise HTTPException(400, "Path is required.")
    new_path = Path(raw).expanduser()
    if not new_path.is_absolute():
        raise HTTPException(400, "Path must be absolute.")
    try:
        new_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise HTTPException(400, f"Cannot create directory: {e}")
    if not new_path.is_dir():
        raise HTTPException(400, "Path is not a directory.")

    try:
        BOOTSTRAP_DATA_DIR_FILE.parent.mkdir(parents=True, exist_ok=True)
        BOOTSTRAP_DATA_DIR_FILE.write_text(str(new_path))
    except OSError as e:
        raise HTTPException(500, f"Failed to write bootstrap file: {e}")

    # Re-exec on a background thread so the HTTP response gets sent first.
    def _restart():
        import time
        time.sleep(0.5)
        logger.info(f"Restarting process to switch DATA_DIR -> {new_path}")
        try:
            os.execv(sys.executable, [sys.executable, *sys.argv])
        except Exception as e:
            logger.error(f"os.execv failed: {e}; exiting so launcher can restart")
            os._exit(1)

    threading.Thread(target=_restart, daemon=True).start()
    return {
        "saved": str(new_path),
        "bootstrap_file": str(BOOTSTRAP_DATA_DIR_FILE),
        "restarting": True,
    }


@router.post("/test-remote")
def test_remote_connection() -> dict:
    """Test SSH connection to the configured remote training host."""
    settings = get_settings()
    cfg = settings.get_remote_config()
    if cfg is None:
        return {
            "ok": False,
            "message": "Remote training not configured. Set host, python, and work directory.",
            "details": {},
        }

    from ..services.remote import test_connection
    return test_connection(cfg)
