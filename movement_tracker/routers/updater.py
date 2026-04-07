"""In-app update: check for new versions on GitHub and apply updates."""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..config import PROJECT_DIR

router = APIRouter(prefix="/api/update", tags=["update"])
logger = logging.getLogger(__name__)

GITHUB_REPO = "newboldd/movement-tracker"
GITHUB_API = f"https://api.github.com/repos/{GITHUB_REPO}/commits/master"
GITHUB_ZIP = f"https://github.com/{GITHUB_REPO}/archive/refs/heads/master.zip"
VERSION_FILE = PROJECT_DIR / "VERSION"

# Paths to preserve during update (relative to project root)
PRESERVE = {
    "movement_tracker/dlc_app.db",
    "movement_tracker/settings.json",
    "videos",
    "dlc",
    "sample_data",
    ".python",
    ".venv",
    "data",
}


def _read_local_sha() -> str | None:
    """Read the local VERSION file."""
    try:
        return VERSION_FILE.read_text().strip()
    except FileNotFoundError:
        return None


@router.get("/check")
def check_for_updates() -> dict:
    """Compare local version against latest GitHub commit."""
    local_sha = _read_local_sha()

    try:
        req = urllib.request.Request(GITHUB_API, headers={
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "MovementTracker-Updater",
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        raise HTTPException(502, f"Cannot reach GitHub: {e}")

    latest_sha = data.get("sha", "")
    latest_date = data.get("commit", {}).get("committer", {}).get("date", "")
    latest_message = data.get("commit", {}).get("message", "").split("\n")[0]

    return {
        "current_sha": local_sha or "unknown",
        "current_short": (local_sha or "unknown")[:7],
        "latest_sha": latest_sha,
        "latest_short": latest_sha[:7],
        "latest_date": latest_date,
        "latest_message": latest_message,
        "update_available": bool(local_sha and latest_sha and local_sha != latest_sha),
        "first_install": local_sha is None,
    }


@router.post("/apply")
def apply_update() -> dict:
    """Download latest code from GitHub and apply it, preserving user data.

    After copying new files, writes the new SHA to VERSION and exits
    with code 42 to trigger a restart from the launcher script.
    """
    # 1. Get latest SHA
    try:
        req = urllib.request.Request(GITHUB_API, headers={
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "MovementTracker-Updater",
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        latest_sha = data["sha"]
    except Exception as e:
        raise HTTPException(502, f"Cannot check GitHub: {e}")

    # 2. Download zip
    tmp_dir = tempfile.mkdtemp(prefix="mt_update_")
    zip_path = os.path.join(tmp_dir, "update.zip")
    extract_dir = os.path.join(tmp_dir, "extracted")

    try:
        logger.info(f"Downloading update from {GITHUB_ZIP}...")
        urllib.request.urlretrieve(GITHUB_ZIP, zip_path)

        # 3. Extract
        logger.info("Extracting update...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)

        # GitHub zips contain a top-level dir like "movement-tracker-master/"
        contents = os.listdir(extract_dir)
        if len(contents) == 1 and os.path.isdir(os.path.join(extract_dir, contents[0])):
            source_dir = os.path.join(extract_dir, contents[0])
        else:
            source_dir = extract_dir

        # 4. Copy new files, skipping preserved paths
        project_root = str(PROJECT_DIR)
        copied = 0
        skipped = 0

        for root, dirs, files in os.walk(source_dir):
            rel_root = os.path.relpath(root, source_dir)

            # Skip preserved directories entirely
            dirs_to_remove = []
            for d in dirs:
                rel_path = os.path.join(rel_root, d) if rel_root != "." else d
                rel_path = rel_path.replace("\\", "/")
                if rel_path in PRESERVE:
                    dirs_to_remove.append(d)
                    skipped += 1
            for d in dirs_to_remove:
                dirs.remove(d)

            for f in files:
                rel_path = os.path.join(rel_root, f) if rel_root != "." else f
                rel_path = rel_path.replace("\\", "/")

                # Skip preserved files
                if rel_path in PRESERVE:
                    skipped += 1
                    continue

                src = os.path.join(root, f)
                dst = os.path.join(project_root, rel_path)

                # Ensure destination directory exists
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
                copied += 1

        # 5. Install any new dependencies from updated requirements.txt
        req_file = os.path.join(project_root, "requirements.txt")
        if os.path.exists(req_file):
            import sys, subprocess
            logger.info("Installing updated dependencies...")
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q",
                     "--disable-pip-version-check", "-r", req_file],
                    timeout=120, capture_output=True,
                )
            except Exception as e:
                logger.warning(f"Dependency install failed (non-fatal): {e}")

        # 6. Restore execute permissions on shell scripts
        for script in ["setup.sh", "Movement Tracker.command"]:
            script_path = os.path.join(project_root, script)
            if os.path.exists(script_path):
                os.chmod(script_path, 0o755)

        # 7. Write new VERSION
        VERSION_FILE.write_text(latest_sha + "\n")

        logger.info(f"Update applied: {copied} files copied, {skipped} preserved. SHA: {latest_sha[:7]}")

    except Exception as e:
        logger.exception("Update failed")
        raise HTTPException(500, f"Update failed: {e}")
    finally:
        # Clean up temp files
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # 8. Schedule restart — exit with code 42 after responding
    import threading
    def _restart():
        import time
        time.sleep(1)  # Let the response complete
        logger.info("Restarting server after update (exit code 42)...")
        os._exit(42)
    threading.Thread(target=_restart, daemon=True).start()

    return {
        "status": "ok",
        "sha": latest_sha,
        "short": latest_sha[:7],
        "files_copied": copied,
        "files_preserved": skipped,
        "message": "Update applied. Server restarting...",
    }
