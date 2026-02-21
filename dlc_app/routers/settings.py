"""Settings API: read, update, and status check."""

from pathlib import Path
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from ..config import get_settings

router = APIRouter(prefix="/api/settings", tags=["settings"])


class SettingsUpdate(BaseModel):
    video_dir: Optional[str] = None
    dlc_dir: Optional[str] = None
    calibration_3d_config: Optional[str] = None
    python_executable: Optional[str] = None
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
        "remote_enabled": settings.remote_enabled,
        "issues": issues,
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
