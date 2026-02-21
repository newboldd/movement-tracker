"""Configuration for DLC web app — Settings singleton with JSON persistence."""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Fixed paths (not configurable) ────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
DB_PATH = APP_DIR / "dlc_app.db"
SETTINGS_PATH = APP_DIR / "settings.json"

# ── Constants (not worth making configurable) ─────────────────────────────
JPEG_QUALITY = 85
FRAME_CACHE_SIZE = 256


class Settings:
    """App settings loaded from settings.json with env var overrides."""

    _instance: Optional["Settings"] = None

    def __init__(self):
        self.video_dir: str = ""
        self.dlc_dir: str = ""
        self.calibration_3d_config: str = ""
        self.python_executable: str = sys.executable
        self.camera_names: list[str] = ["OS", "OD"]
        self.bodyparts: list[str] = ["thumb", "index"]
        self.dlc_scorer: str = "labels"
        self.dlc_date: str = "Sep16"
        self.dlc_net_type: str = "resnet_50"
        self.host: str = "127.0.0.1"
        self.port: int = 8080

        # Remote training (optional)
        self.remote_host: str = ""       # e.g. user@192.168.1.50
        self.remote_python: str = ""     # e.g. /home/user/miniconda3/envs/dlc/bin/python
        self.remote_work_dir: str = ""   # e.g. /home/user/dlc_training
        self.remote_ssh_key: str = ""    # optional, e.g. ~/.ssh/id_ed25519
        self.remote_ssh_port: int = 22

        self._load()

    @property
    def is_configured(self) -> bool:
        return bool(self.video_dir) and bool(self.dlc_dir)

    @property
    def video_path(self) -> Path:
        return Path(self.video_dir) if self.video_dir else PROJECT_DIR / "videos"

    @property
    def dlc_path(self) -> Path:
        return Path(self.dlc_dir) if self.dlc_dir else PROJECT_DIR / "dlc"

    @property
    def remote_enabled(self) -> bool:
        """True when remote training is configured (host + python + work_dir all set)."""
        return bool(self.remote_host) and bool(self.remote_python) and bool(self.remote_work_dir)

    def get_remote_config(self):
        """Return a RemoteConfig if remote training is configured, else None."""
        if not self.remote_enabled:
            return None
        from .services.remote import RemoteConfig
        return RemoteConfig(
            host=self.remote_host,
            python_executable=self.remote_python,
            work_dir=self.remote_work_dir,
            ssh_key_path=self.remote_ssh_key,
            port=self.remote_ssh_port,
        )

    @property
    def data_path(self) -> Path:
        return PROJECT_DIR / "data"

    def _load(self):
        """Load from settings.json, then apply env var overrides."""
        if SETTINGS_PATH.exists():
            try:
                data = json.loads(SETTINGS_PATH.read_text())
                self._apply_dict(data)
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Failed to load settings.json: {e}")
        else:
            self._auto_detect()

        self._apply_env_overrides()

    def _auto_detect(self):
        """On first run, detect existing paths from project structure."""
        video_dir = PROJECT_DIR / "videos"
        dlc_dir = PROJECT_DIR / "dlc"

        if video_dir.exists():
            self.video_dir = str(video_dir)
        if dlc_dir.exists():
            self.dlc_dir = str(dlc_dir)

        if self.is_configured:
            logger.info("Auto-detected existing paths, saving settings.json")
            self.save()

    def _apply_dict(self, data: dict):
        """Apply a dict of settings values."""
        for key in [
            "video_dir", "dlc_dir", "calibration_3d_config",
            "python_executable", "dlc_scorer", "dlc_date", "dlc_net_type",
            "host",
            "remote_host", "remote_python", "remote_work_dir", "remote_ssh_key",
        ]:
            if key in data and data[key] is not None:
                setattr(self, key, data[key])

        if "camera_names" in data and isinstance(data["camera_names"], list):
            self.camera_names = data["camera_names"]
        if "bodyparts" in data and isinstance(data["bodyparts"], list):
            self.bodyparts = data["bodyparts"]
        if "port" in data and data["port"] is not None:
            self.port = int(data["port"])
        if "remote_ssh_port" in data and data["remote_ssh_port"] is not None:
            self.remote_ssh_port = int(data["remote_ssh_port"])

    def _apply_env_overrides(self):
        """Override settings from DLC_APP_* environment variables."""
        env_map = {
            "DLC_APP_VIDEO_DIR": "video_dir",
            "DLC_APP_DLC_DIR": "dlc_dir",
            "DLC_APP_CALIBRATION_3D_CONFIG": "calibration_3d_config",
            "DLC_APP_PYTHON_EXECUTABLE": "python_executable",
            "DLC_APP_DLC_SCORER": "dlc_scorer",
            "DLC_APP_DLC_DATE": "dlc_date",
            "DLC_APP_DLC_NET_TYPE": "dlc_net_type",
            "DLC_APP_HOST": "host",
            "DLC_APP_REMOTE_HOST": "remote_host",
            "DLC_APP_REMOTE_PYTHON": "remote_python",
            "DLC_APP_REMOTE_WORK_DIR": "remote_work_dir",
            "DLC_APP_REMOTE_SSH_KEY": "remote_ssh_key",
        }
        for env_key, attr in env_map.items():
            val = os.environ.get(env_key)
            if val is not None:
                setattr(self, attr, val)

        port = os.environ.get("DLC_APP_PORT")
        if port is not None:
            self.port = int(port)

        remote_port = os.environ.get("DLC_APP_REMOTE_SSH_PORT")
        if remote_port is not None:
            self.remote_ssh_port = int(remote_port)

        cam = os.environ.get("DLC_APP_CAMERA_NAMES")
        if cam:
            self.camera_names = [c.strip() for c in cam.split(",")]

        bp = os.environ.get("DLC_APP_BODYPARTS")
        if bp:
            self.bodyparts = [b.strip() for b in bp.split(",")]

    def to_dict(self) -> dict:
        return {
            "video_dir": self.video_dir,
            "dlc_dir": self.dlc_dir,
            "calibration_3d_config": self.calibration_3d_config,
            "python_executable": self.python_executable,
            "camera_names": self.camera_names,
            "bodyparts": self.bodyparts,
            "dlc_scorer": self.dlc_scorer,
            "dlc_date": self.dlc_date,
            "dlc_net_type": self.dlc_net_type,
            "host": self.host,
            "port": self.port,
            "remote_host": self.remote_host,
            "remote_python": self.remote_python,
            "remote_work_dir": self.remote_work_dir,
            "remote_ssh_key": self.remote_ssh_key,
            "remote_ssh_port": self.remote_ssh_port,
        }

    def save(self):
        """Persist current settings to settings.json."""
        SETTINGS_PATH.write_text(json.dumps(self.to_dict(), indent=2))

    def update(self, data: dict):
        """Update settings from dict and save."""
        self._apply_dict(data)
        self.save()

    @classmethod
    def _reset(cls):
        """Reset singleton (for testing)."""
        cls._instance = None


def get_settings() -> Settings:
    """Get the Settings singleton."""
    if Settings._instance is None:
        Settings._instance = Settings()
    return Settings._instance
