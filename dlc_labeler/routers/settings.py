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

    _schedule_restart(f"DATA_DIR -> {new_path}")
    return {
        "saved": str(new_path),
        "bootstrap_file": str(BOOTSTRAP_DATA_DIR_FILE),
        "restarting": True,
    }


# ── Updating the app in place ──────────────────────────────────────────
#
# The people this is built for install by cloning and run by
# double-clicking.  Asking them to open a terminal to pick up a fix is
# the step that does not happen, so the app pulls its own updates.
#
# Deliberately narrow: a fast-forward pull of the checkout the app is
# running from, and nothing else.  It never merges, never rewrites
# history, never discards local work, and never installs packages — each
# of those can leave someone with a broken install and no way to tell
# what happened.

GIT_TIMEOUT = 120

# A change to any of these means the running process is now stale: Python
# is read once at import, and the launcher scripts are what started it.
_RESTART_SUFFIXES = (".py",)
_RESTART_NAMES = ("setup.sh", "run.bat")
_DEPS_NAMES = ("requirements.txt", "requirements-dlc.txt")


def _git(*args: str, timeout: int = GIT_TIMEOUT):
    """Run git in the app's own directory. Returns (ok, stdout, stderr)."""
    import subprocess

    from ..config import PROJECT_DIR

    try:
        out = subprocess.run(
            ["git", "-C", str(PROJECT_DIR), *args],
            capture_output=True, text=True, timeout=timeout,
        )
        return out.returncode == 0, out.stdout.strip(), out.stderr.strip()
    except FileNotFoundError:
        return False, "", "git is not installed"
    except subprocess.SubprocessError as e:
        return False, "", str(e)


def _git_state() -> dict:
    """What the checkout looks like right now, without touching the network."""
    from ..config import PROJECT_DIR

    ok, head, _ = _git("rev-parse", "HEAD", timeout=10)
    if not ok:
        return {"available": False,
                "reason": f"{PROJECT_DIR} is not a git checkout — update by "
                          f"downloading the app again."}

    has_origin, _, _ = _git("remote", "get-url", "origin", timeout=10)
    if not has_origin:
        return {"available": False,
                "reason": "This checkout has no 'origin' remote to update from."}

    _, branch, _ = _git("rev-parse", "--abbrev-ref", "HEAD", timeout=10)

    # Asked as two commands that print plain paths rather than parsing
    # `status --porcelain`, whose two-column prefix is partly spaces and
    # does not survive being stripped.
    _, tracked, _ = _git("diff", "--name-only", "HEAD", timeout=15)
    _, untracked, _ = _git("ls-files", "--others", "--exclude-standard",
                           timeout=15)
    return {
        "available": True,
        "commit": head[:8],
        "branch": branch,
        # Edits to files the app ships: a pull could overwrite these, so
        # they stop it.
        "dirty": [f for f in tracked.splitlines() if f.strip()],
        # New files that are not ignored.  Reported but not a blocker:
        # one only matters if the update adds a file at the same path,
        # and git refuses that itself — blocking every update because
        # somebody left a notes file in the folder would mean nobody
        # updates again.
        "untracked": [f for f in untracked.splitlines() if f.strip()],
        "project_dir": str(PROJECT_DIR),
    }


@router.get("/update/status")
def update_status() -> dict:
    """Whether this install can update itself, and what it is running.

    Local only — no fetch.  The nav asks this on every page load, and a
    network round trip there would stall the header on a machine that is
    offline, which these often are.
    """
    return _git_state()


@router.post("/update")
def run_update() -> dict:
    """Fast-forward the checkout, and restart if the code changed.

    Refuses rather than resolves anything: uncommitted changes stop the
    update with the files named, and a branch that has diverged from the
    remote is reported instead of merged.  Someone who has edited the app
    knows what they did; this should not be what tells them it is gone.
    """
    state = _git_state()
    if not state.get("available"):
        raise HTTPException(400, state.get("reason", "Cannot update."))
    if state["dirty"]:
        raise HTTPException(
            409, "This checkout has uncommitted changes, so an update would "
                 "risk losing them: "
                 + ", ".join(state["dirty"][:8])
                 + (f" and {len(state['dirty']) - 8} more" if len(state["dirty"]) > 8 else "")
                 + ". Commit or discard them, then update again.")

    before = state["commit"]
    ok, out, err = _git("pull", "--ff-only")
    if not ok:
        detail = (err or out or "git pull failed").splitlines()
        message = detail[-1] if detail else "git pull failed"
        blob = (err + " " + out).lower()
        if "fast-forward" in blob or "diverge" in blob:
            message = ("This copy has commits the remote does not, so it "
                       "cannot be fast-forwarded. Sort it out with git, then "
                       "update again.")
        elif "would be overwritten" in blob:
            message = ("Local files would be overwritten by the update. "
                       "Move or remove them, then update again: " + message)
        raise HTTPException(400, f"Update failed: {message}")

    _, after, _ = _git("rev-parse", "HEAD", timeout=10)
    after = after[:8]
    if after == before:
        return {"updated": False, "commit": after, "restarting": False,
                "message": "Already up to date."}

    _, names, _ = _git("diff", "--name-only", before, after, timeout=30)
    files = [f for f in names.splitlines() if f.strip()]
    deps = [f for f in files if Path(f).name in _DEPS_NAMES]
    needs_restart = any(
        f.endswith(_RESTART_SUFFIXES) or Path(f).name in _RESTART_NAMES
        for f in files)

    if needs_restart:
        _schedule_restart(f"update {before} -> {after}")

    return {
        "updated": True,
        "from": before,
        "commit": after,
        "file_count": len(files),
        "files": files[:40],
        "restarting": needs_restart,
        "deps_changed": [Path(f).name for f in deps],
        "message": (f"Updated to {after} ({len(files)} file(s) changed)."),
    }


def _restart_argv() -> list[str]:
    """The command that would start this process again.

    ``sys.argv`` alone is not it.  Under ``python -m uvicorn`` — which is
    how the launchers start the app — ``sys.argv[0]`` is uvicorn's
    ``__main__.py``, and re-execing that as a *script* puts uvicorn's own
    package directory first on ``sys.path``.  The next ``import logging``
    then finds ``uvicorn/logging.py`` instead of the standard library and
    the process dies on startup.  ``__main__.__spec__`` is what says the
    process was started with ``-m``, so rebuild it that way.
    """
    import __main__

    spec = getattr(__main__, "__spec__", None)
    if spec is not None and spec.name:
        return [sys.executable, "-m", spec.name, *sys.argv[1:]]
    return [sys.executable, *sys.argv]


def _schedule_restart(reason: str):
    """Bring the server back, after the HTTP response has gone out.

    Prefers handing the job to the launcher: setup.sh and run.bat loop on
    exit code 42 and start the app exactly the way they did the first
    time, which beats any guess made from inside the process.  Re-exec is
    the fallback for someone running uvicorn themselves.
    """
    def _restart():
        import time
        time.sleep(0.5)
        if os.environ.get("DLC_LAUNCHER"):
            logger.info("Exiting 42 after %s; the launcher restarts us", reason)
            os._exit(42)

        argv = _restart_argv()
        logger.info("Restarting after %s: %s", reason, " ".join(argv))
        try:
            os.execv(argv[0], argv)
        except Exception as e:
            # Nothing is supervising us and the re-exec failed, so 42 is
            # a guess — but a launcher that is there will catch it, and
            # staying up on stale code is not better.
            logger.error("Re-exec failed: %s; exiting 42", e)
            os._exit(42)

    threading.Thread(target=_restart, daemon=True).start()
