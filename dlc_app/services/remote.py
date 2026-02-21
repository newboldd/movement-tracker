"""Remote DLC training via SSH: upload project, run training, download results."""

import logging
import os
import re
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

from ..db import get_db_ctx

logger = logging.getLogger(__name__)


@dataclass
class RemoteConfig:
    host: str               # user@hostname
    python_executable: str  # remote python with DLC installed
    work_dir: str           # remote directory for DLC projects
    ssh_key_path: str = ""  # optional path to SSH private key
    port: int = 22


def _ssh_base_args(cfg: RemoteConfig) -> list[str]:
    """Build base SSH command args with BatchMode (no password prompts)."""
    args = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]
    if cfg.port != 22:
        args += ["-p", str(cfg.port)]
    if cfg.ssh_key_path:
        args += ["-i", cfg.ssh_key_path]
    return args


def _scp_base_args(cfg: RemoteConfig) -> list[str]:
    """Build base SCP command args."""
    args = ["scp", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-r"]
    if cfg.port != 22:
        args += ["-P", str(cfg.port)]
    if cfg.ssh_key_path:
        args += ["-i", cfg.ssh_key_path]
    return args


def test_connection(cfg: RemoteConfig) -> dict:
    """Test SSH connection with 4 checks: connect, mkdir, DLC version, GPU.

    Returns dict with 'ok' (bool), 'message' (str), 'details' (dict of check results).
    """
    details = {}

    # Check 1: SSH connection
    try:
        result = subprocess.run(
            _ssh_base_args(cfg) + [cfg.host, "echo ok"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            return {
                "ok": False,
                "message": f"SSH connection failed: {result.stderr.strip()}",
                "details": {"ssh": False},
            }
        details["ssh"] = True
    except subprocess.TimeoutExpired:
        return {"ok": False, "message": "SSH connection timed out", "details": {"ssh": False}}
    except FileNotFoundError:
        return {"ok": False, "message": "ssh command not found on this machine", "details": {"ssh": False}}

    # Check 2: Create work directory
    result = subprocess.run(
        _ssh_base_args(cfg) + [cfg.host, f"mkdir -p {cfg.work_dir} && echo ok"],
        capture_output=True, text=True, timeout=15,
    )
    details["work_dir"] = result.returncode == 0
    if not details["work_dir"]:
        return {
            "ok": False,
            "message": f"Cannot create work directory: {result.stderr.strip()}",
            "details": details,
        }

    # Check 3: DLC version
    result = subprocess.run(
        _ssh_base_args(cfg) + [
            cfg.host,
            f"{cfg.python_executable} -c \"import deeplabcut; print(deeplabcut.__version__)\"",
        ],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode == 0:
        details["dlc_version"] = result.stdout.strip()
    else:
        return {
            "ok": False,
            "message": f"DeepLabCut not found on remote: {result.stderr.strip()[:200]}",
            "details": details,
        }

    # Check 4: GPU availability
    result = subprocess.run(
        _ssh_base_args(cfg) + [
            cfg.host,
            f"{cfg.python_executable} -c \""
            "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU')\"",
        ],
        capture_output=True, text=True, timeout=30,
    )
    details["gpu"] = result.stdout.strip() if result.returncode == 0 else "check failed"

    return {
        "ok": True,
        "message": f"Connected. DLC {details['dlc_version']}, {details['gpu']}",
        "details": details,
    }


def remote_train_monitor(
    job_id: int,
    cfg: RemoteConfig,
    local_dlc_dir: Path,
    subject_name: str,
    log_path: str,
    progress_parser,
    on_complete,
    registry,
):
    """3-phase remote training lifecycle. Runs in a daemon thread.

    Phase 1 (0-5%):   scp local DLC dir to remote
    Phase 2 (5-95%):  ssh training command, parse stdout for epoch progress
    Phase 3 (95-100%): scp trained model back to local

    Args:
        job_id: Database job ID
        cfg: RemoteConfig with SSH details
        local_dlc_dir: Local path to subject's DLC project directory
        subject_name: Subject name (used for remote path)
        log_path: Path for log file
        progress_parser: callable(line) -> float|None for training progress
        on_complete: callable(job_id, returncode) for post-completion
        registry: JobRegistry instance (to register subprocesses for cancel)
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    remote_project_dir = f"{cfg.work_dir}/{subject_name}"

    def _update_progress(pct):
        with get_db_ctx() as db:
            db.execute("UPDATE jobs SET progress_pct = ? WHERE id = ?", (pct, job_id))

    def _fail(msg):
        logger.error(f"Job {job_id} remote training failed: {msg}")
        with get_db_ctx() as db:
            db.execute(
                """UPDATE jobs SET status = 'failed', error_msg = ?,
                   finished_at = CURRENT_TIMESTAMP WHERE id = ?""",
                (msg, job_id),
            )

    try:
        with open(log_path, "w") as logfile:
            # ── Phase 1: Upload ──────────────────────────────────────
            logfile.write(f"=== Phase 1: Uploading {local_dlc_dir} to {cfg.host}:{remote_project_dir} ===\n")
            logfile.flush()

            # Ensure remote dir exists
            subprocess.run(
                _ssh_base_args(cfg) + [cfg.host, f"mkdir -p {cfg.work_dir}"],
                capture_output=True, timeout=15,
            )

            upload_cmd = _scp_base_args(cfg) + [
                str(local_dlc_dir),
                f"{cfg.host}:{cfg.work_dir}/",
            ]
            proc = subprocess.Popen(
                upload_cmd,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            registry._processes[job_id] = proc

            for line in proc.stdout:
                logfile.write(line)
                logfile.flush()

            proc.wait()
            if proc.returncode != 0:
                _fail(f"Upload failed (exit {proc.returncode})")
                if on_complete:
                    on_complete(job_id, proc.returncode)
                return

            _update_progress(5.0)
            logfile.write("=== Upload complete ===\n")
            logfile.flush()

            # ── Fix remote config.yaml project_path ──────────────────
            fix_cmd = (
                f"sed -i 's|^project_path:.*|project_path: {remote_project_dir}|' "
                f"{remote_project_dir}/config.yaml"
            )
            subprocess.run(
                _ssh_base_args(cfg) + [cfg.host, fix_cmd],
                capture_output=True, timeout=15,
            )

            # ── Phase 2: Training ────────────────────────────────────
            remote_config = f"{remote_project_dir}/config.yaml"
            train_script = (
                f"import deeplabcut; "
                f"deeplabcut.train_network(r'{remote_config}')"
            )
            train_cmd = _ssh_base_args(cfg) + [
                cfg.host,
                f"{cfg.python_executable} -u -c \"{train_script}\"",
            ]

            logfile.write(f"=== Phase 2: Training on {cfg.host} ===\n")
            logfile.flush()

            proc = subprocess.Popen(
                train_cmd,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            registry._processes[job_id] = proc

            for line in proc.stdout:
                logfile.write(line)
                logfile.flush()

                if progress_parser:
                    raw_pct = progress_parser(line)
                    if raw_pct is not None:
                        # Scale 0-100% training progress to 5-95% overall
                        scaled = 5.0 + (raw_pct / 100.0) * 90.0
                        _update_progress(scaled)

            proc.wait()
            if proc.returncode != 0:
                _fail(f"Training failed (exit {proc.returncode})")
                if on_complete:
                    on_complete(job_id, proc.returncode)
                return

            _update_progress(95.0)
            logfile.write("=== Training complete ===\n")
            logfile.flush()

            # ── Phase 3: Download trained model ──────────────────────
            logfile.write(f"=== Phase 3: Downloading model from {cfg.host} ===\n")
            logfile.flush()

            # Download dlc-models directory back to local
            remote_models = f"{cfg.host}:{remote_project_dir}/dlc-models"
            local_models = local_dlc_dir / "dlc-models"

            download_cmd = _scp_base_args(cfg) + [
                remote_models,
                str(local_dlc_dir) + "/",
            ]
            proc = subprocess.Popen(
                download_cmd,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            registry._processes[job_id] = proc

            for line in proc.stdout:
                logfile.write(line)
                logfile.flush()

            proc.wait()
            if proc.returncode != 0:
                _fail(f"Download failed (exit {proc.returncode})")
                if on_complete:
                    on_complete(job_id, proc.returncode)
                return

            _update_progress(100.0)
            logfile.write("=== Download complete ===\n")
            logfile.flush()

        # ── Success ──────────────────────────────────────────────
        with get_db_ctx() as db:
            db.execute(
                """UPDATE jobs SET status = 'completed', progress_pct = 100,
                   finished_at = CURRENT_TIMESTAMP WHERE id = ?""",
                (job_id,),
            )

        if on_complete:
            on_complete(job_id, 0)

    except Exception as e:
        logger.exception(f"Job {job_id} remote training error")
        _fail(str(e))
    finally:
        registry._processes.pop(job_id, None)
        registry._threads.pop(job_id, None)
