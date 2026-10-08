"""Settings API: read, update, status, data directory."""
from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..config import BOOTSTRAP_DATA_DIR_FILE, DATA_DIR, get_settings

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
    calibrations: Optional[Dict[str, str]] = None
    prefer_deidentified: Optional[bool] = None
    show_example_subject: Optional[bool] = None


@router.get("")
def get_all_settings() -> dict:
    """Current settings."""
    return get_settings().to_dict()


@router.put("")
def update_settings(req: SettingsUpdate) -> dict:
    """Update settings and save to disk."""
    settings = get_settings()
    settings.update(req.model_dump(exclude_none=True))
    # Calibration is cached by camera name; new paths need the cache gone.
    from ..services.calibration import clear_calibration_cache
    clear_calibration_cache()
    return settings.to_dict()


@router.get("/status")
def settings_status() -> dict:
    """What this machine can do: GPU, CUDA, DeepLabCut, calibration."""
    settings = get_settings()
    issues = []

    if not settings.python_executable:
        issues.append("python_executable not set")

    settings.dlc_path.mkdir(parents=True, exist_ok=True)
    if not settings.video_path.is_dir():
        issues.append(f"Video directory not found: {settings.video_path}")

    local_gpu_available = settings.local_gpu_available
    cuda_version = None
    try:
        import torch
        cuda_version = torch.version.cuda
    except Exception:
        pass

    dlc_installed = settings.dlc_installed()
    if not dlc_installed:
        issues.append("DeepLabCut is not installed — labeling works, "
                      "training and analysis do not")

    return {
        "configured": settings.is_configured,
        "data_dir": str(DATA_DIR),
        "video_dir": str(settings.video_path),
        "dlc_dir": str(settings.dlc_path),
        "has_calibration": bool(settings.calibrations)
        or bool(settings.calibration_3d_config),
        "dlc_installed": dlc_installed,
        "dlc_python": settings.python_executable or sys.executable,
        "local_gpu_available": local_gpu_available,
        "gpus": settings.get_available_gpus() if local_gpu_available else [],
        "cuda_version": cuda_version,
        "issues": issues,
    }


class CalibrationValidate(BaseModel):
    path: str


@router.post("/validate-calibration")
def validate_calibration(req: CalibrationValidate) -> dict:
    """Check that a calibration YAML exists and carries a K1 matrix."""
    import cv2
    p = Path(req.path).expanduser()
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


# ── Data directory (videos, DLC projects and the database live here) ──────

class DataDirReq(BaseModel):
    path: str


@router.get("/data-dir")
def get_data_dir() -> dict:
    """The active data directory and where that choice came from."""
    env_val = (os.environ.get("DLC_DATA_DIR")
               or os.environ.get("MT_DATA_DIR") or None)
    boot_val = None
    if BOOTSTRAP_DATA_DIR_FILE.is_file():
        try:
            boot_val = BOOTSTRAP_DATA_DIR_FILE.read_text().strip() or None
        except OSError:
            boot_val = None

    source = "env" if env_val else ("bootstrap" if boot_val else "default")
    return {
        "current": str(DATA_DIR),
        "bootstrap": boot_val,
        "env_override": env_val,
        "source": source,
        "bootstrap_file": str(BOOTSTRAP_DATA_DIR_FILE),
    }


@router.post("/data-dir")
def set_data_dir(req: DataDirReq) -> dict:
    """Persist a new data directory, then re-exec to pick it up.

    DATA_DIR is resolved at import time and baked into module-level
    constants across the package (the database path, the calibration
    directory).  Re-binding it live would leave half the app pointing at
    the old location, so the process restarts instead — the launcher
    scripts expect that and bring it straight back up.
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

    def _restart():
        import time
        time.sleep(0.5)
        logger.info("Restarting process to switch DATA_DIR -> %s", new_path)
        try:
            os.execv(sys.executable, [sys.executable, *sys.argv])
        except Exception as e:
            logger.error("os.execv failed: %s; exiting so the launcher "
                         "can restart us", e)
            os._exit(1)

    threading.Thread(target=_restart, daemon=True).start()
    return {
        "saved": str(new_path),
        "bootstrap_file": str(BOOTSTRAP_DATA_DIR_FILE),
        "restarting": True,
    }
