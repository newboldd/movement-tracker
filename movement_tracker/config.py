"""Configuration for DLC web app — Settings singleton with JSON persistence."""
from __future__ import annotations

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
CALIBRATION_DIR = PROJECT_DIR / "calibration"

# Default calibrations shipped with the app (camera name -> YAML path)
DEFAULT_CALIBRATIONS = {
    "camera1": str(CALIBRATION_DIR / "camera1_recalibrated.yaml"),
    "dysaut_cam1": str(CALIBRATION_DIR / "cam1_calibration.yaml"),
    "dysaut_cam2": str(CALIBRATION_DIR / "cam2_calibration.yaml"),
}


class Settings:
    """App settings loaded from settings.json with env var overrides."""

    _instance: Optional["Settings"] = None

    def __init__(self):
        self.video_dir: str = ""
        self.dlc_dir: str = ""
        self.calibration_3d_config: str = ""
        self.python_executable: str = sys.executable
        self.default_camera_mode: str = "stereo"  # "single", "stereo", or "multicam"
        self.camera_names: list[str] = ["OS", "OD"]
        self.bodyparts: list[str] = ["thumb", "index"]
        self.dlc_scorer: str = "labels"
        self.dlc_date: str = "Sep16"
        self.dlc_net_type: str = "resnet_50"
        self.calibration_dir: str = ""  # path to calibration/ with camera_assignments.yaml
        self.calibrations: dict = dict(DEFAULT_CALIBRATIONS)  # camera_name -> YAML path
        self.host: str = "127.0.0.1"
        self.port: int = 8080

        # Display preferences
        self.prefer_deidentified: bool = True  # show deidentified videos in labeling UI
        self.show_tutorials: bool = True  # show Tutorials link in nav bar
        self.show_example_subject: bool = True  # show the built-in Example subject

        # Diagnosis/Group settings (for dashboard organization)
        self.diagnosis_groups: list[str] = ["Control", "MSA", "PD", "PSP"]

        # Event types for labeling (name, color, shortcut key)
        # open/peak/close are "special" — auto-detection targets them specifically
        self.event_types: list[dict] = [
            {"name": "open",  "color": "#00cc44", "shortcut": "1"},
            {"name": "peak",  "color": "#ffcc00", "shortcut": "2"},
            {"name": "close", "color": "#ff4444", "shortcut": "3"},
            {"name": "pause", "color": "#cc66ff", "shortcut": "4"},
        ]

        # Remote training (optional)
        self.remote_host: str = ""       # e.g. user@192.168.1.50
        self.remote_python: str = ""     # e.g. /home/user/miniconda3/envs/dlc/bin/python
        self.remote_work_dir: str = ""   # e.g. /home/user/dlc_training
        self.remote_ssh_key: str = ""    # optional, e.g. ~/.ssh/id_ed25519
        self.remote_ssh_port: int = 22

        self._load()

    @property
    def is_configured(self) -> bool:
        return True  # No required user configuration; dlc dir is hard-coded

    @property
    def video_path(self) -> Path:
        return Path(self.video_dir) if self.video_dir else PROJECT_DIR / "videos"

    @property
    def dlc_path(self) -> Path:
        """Always use dlc/ at the project root as the DLC project directory."""
        return PROJECT_DIR / "dlc"

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
    def local_gpu_available(self) -> bool:
        """Check if GPU is available locally."""
        try:
            import torch
            return torch.cuda.is_available()
        except (ImportError, Exception):
            return False

    def get_available_gpus(self) -> list[dict]:
        """Get list of available GPUs with details.

        Returns list of dicts with keys: index, name, memory_mb
        Empty list if GPU not available or detection fails.
        """
        gpus = []
        try:
            import torch
            if not torch.cuda.is_available():
                return gpus

            count = torch.cuda.device_count()
            for i in range(count):
                try:
                    name = torch.cuda.get_device_name(i)
                    # Get memory in MB
                    memory_bytes = torch.cuda.get_device_properties(i).total_memory
                    memory_mb = int(memory_bytes / (1024 * 1024))
                    gpus.append({
                        "index": i,
                        "name": name,
                        "memory_mb": memory_mb
                    })
                except Exception as e:
                    logger.warning(f"Could not get details for GPU {i}: {e}")
                    gpus.append({
                        "index": i,
                        "name": f"GPU {i}",
                        "memory_mb": 0
                    })
        except (ImportError, Exception) as e:
            logger.debug(f"Could not detect GPUs: {e}")

        return gpus

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
        """On first run, set sensible defaults from project structure.

        Priority for video directory:
          1. sample_data/  (ships with the repo for new users)
          2. videos/       (existing installs that have a real video library)

        DLC directory defaults to dlc/ inside the repo; created if absent.
        """
        sample_dir = PROJECT_DIR / "sample_data"
        video_dir  = PROJECT_DIR / "videos"
        dlc_dir    = PROJECT_DIR / "dlc"

        if sample_dir.exists():
            self.video_dir = str(sample_dir)
        elif video_dir.exists():
            self.video_dir = str(video_dir)

        # Always default dlc_dir; create the directory so it's ready to use
        dlc_dir.mkdir(exist_ok=True)
        self.dlc_dir = str(dlc_dir)

        logger.info("Auto-detected default paths, saving settings.json")
        self.save()

    def _apply_dict(self, data: dict):
        """Apply a dict of settings values."""
        for key in [
            "video_dir", "dlc_dir", "calibration_3d_config",
            "python_executable", "default_camera_mode",
            "dlc_scorer", "dlc_date", "dlc_net_type",
            "calibration_dir",
            "host",
            "remote_host", "remote_python", "remote_work_dir", "remote_ssh_key",
        ]:
            if key in data and data[key] is not None:
                setattr(self, key, data[key])

        # Backward compat: old "camera_mode" key → "default_camera_mode"
        if "camera_mode" in data and "default_camera_mode" not in data:
            self.default_camera_mode = data["camera_mode"]
        if "camera_names" in data and isinstance(data["camera_names"], list):
            self.camera_names = data["camera_names"]
        if "bodyparts" in data and isinstance(data["bodyparts"], list):
            self.bodyparts = data["bodyparts"]
        if "calibrations" in data and isinstance(data["calibrations"], dict):
            self.calibrations = data["calibrations"]
        if "diagnosis_groups" in data and isinstance(data["diagnosis_groups"], list):
            self.diagnosis_groups = data["diagnosis_groups"]
        if "event_types" in data and isinstance(data["event_types"], list):
            self.event_types = data["event_types"]
        if "port" in data and data["port"] is not None:
            self.port = int(data["port"])
        if "remote_ssh_port" in data and data["remote_ssh_port"] is not None:
            self.remote_ssh_port = int(data["remote_ssh_port"])
        if "prefer_deidentified" in data and data["prefer_deidentified"] is not None:
            self.prefer_deidentified = bool(data["prefer_deidentified"])
        if "show_tutorials" in data and data["show_tutorials"] is not None:
            self.show_tutorials = bool(data["show_tutorials"])
        if "show_example_subject" in data and data["show_example_subject"] is not None:
            self.show_example_subject = bool(data["show_example_subject"])

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
            "DLC_APP_CALIBRATION_DIR": "calibration_dir",
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
            "default_camera_mode": self.default_camera_mode,
            "camera_names": self.camera_names,
            "bodyparts": self.bodyparts,
            "dlc_scorer": self.dlc_scorer,
            "dlc_date": self.dlc_date,
            "dlc_net_type": self.dlc_net_type,
            "calibration_dir": self.calibration_dir,
            "calibrations": self.calibrations,
            "diagnosis_groups": self.diagnosis_groups,
            "host": self.host,
            "port": self.port,
            "remote_host": self.remote_host,
            "remote_python": self.remote_python,
            "remote_work_dir": self.remote_work_dir,
            "remote_ssh_key": self.remote_ssh_key,
            "remote_ssh_port": self.remote_ssh_port,
            "prefer_deidentified": self.prefer_deidentified,
            "show_tutorials": self.show_tutorials,
            "show_example_subject": self.show_example_subject,
            "event_types": self.event_types,
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
