"""Remote DLC training and preprocessing via SSH."""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import threading
import time
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
    """Build base SSH command args with BatchMode (no password prompts).

    ClearAllForwardings suppresses port-forwarding noise from ~/.ssh/config.
    """
    args = [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ClearAllForwardings=yes",
    ]
    if cfg.port != 22:
        args += ["-p", str(cfg.port)]
    if cfg.ssh_key_path:
        args += ["-i", cfg.ssh_key_path]
    return args


def _scp_base_args(cfg: RemoteConfig) -> list[str]:
    """Build base SCP command args."""
    args = [
        "scp",
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ClearAllForwardings=yes",
        "-r",
    ]
    if cfg.port != 22:
        args += ["-P", str(cfg.port)]
    if cfg.ssh_key_path:
        args += ["-i", cfg.ssh_key_path]
    return args


def _py_cmd(cfg: RemoteConfig, script: str) -> list[str]:
    """Build SSH + remote python one-liner. Shell-agnostic (works on PowerShell, bash, etc.)."""
    return _ssh_base_args(cfg) + [cfg.host, cfg.python_executable, "-u", "-c", script]


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

    # Check 2: Create work directory (use Python — shell-agnostic)
    try:
        result = subprocess.run(
            _py_cmd(cfg, f"\"import os; os.makedirs(r'{cfg.work_dir}', exist_ok=True); print('ok')\""),
            capture_output=True, text=True, timeout=15,
        )
        details["work_dir"] = result.returncode == 0 and "ok" in result.stdout
    except subprocess.TimeoutExpired:
        details["work_dir"] = False
    if not details["work_dir"]:
        return {
            "ok": False,
            "message": f"Cannot create work directory (timed out or failed)",
            "details": details,
        }

    # Check 3: DLC version (generous timeout — DLC import loads PyTorch/CUDA)
    try:
        result = subprocess.run(
            _py_cmd(cfg, "\"import deeplabcut; print(deeplabcut.__version__)\""),
            capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "message": "DLC import timed out (>120s). Check the remote Python path.",
            "details": details,
        }
    if result.returncode == 0 and result.stdout.strip():
        # DLC prints "Loading DLC X.Y.Z..." to stderr; version is on stdout
        details["dlc_version"] = result.stdout.strip().splitlines()[-1]
    else:
        return {
            "ok": False,
            "message": f"DeepLabCut not found on remote: {(result.stderr or result.stdout).strip()[:200]}",
            "details": details,
        }

    # Check 4: GPU availability
    try:
        result = subprocess.run(
            _py_cmd(cfg, "\"import os; os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'; import torch; print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU')\""),
            capture_output=True, text=True, timeout=60,
        )
        details["gpu"] = result.stdout.strip() if result.returncode == 0 else "check failed"
    except subprocess.TimeoutExpired:
        details["gpu"] = "check timed out"

    return {
        "ok": True,
        "message": f"Connected. DLC {details['dlc_version']}, {details['gpu']}",
        "details": details,
    }


def _check_remote_pid_alive(cfg: RemoteConfig, pid: int) -> bool:
    """Check if a process with the given PID is running on the remote host."""
    try:
        script = (
            f"\"import psutil; print(psutil.pid_exists({pid}))\""
            if False else  # psutil may not be installed; use os-level check
            f"\"import os; "
            f"alive = False; "
            f"exec('try:\\n import ctypes\\n k = ctypes.windll.kernel32\\n h = k.OpenProcess(0x100000, False, {pid})\\n alive = h != 0\\n if h: k.CloseHandle(h)\\nexcept: pass') "
            f"if os.name == 'nt' else "
            f"exec('try:\\n os.kill({pid}, 0); alive = True\\nexcept (ProcessLookupError, PermissionError): pass'); "
            f"print(alive)\""
        )
        result = subprocess.run(
            _py_cmd(cfg, script),
            capture_output=True, text=True, timeout=15,
        )
        return result.returncode == 0 and "True" in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def _read_remote_status(cfg: RemoteConfig, status_file: str) -> dict | None:
    """Read status.json from remote host. Returns parsed dict or None."""
    try:
        result = subprocess.run(
            _py_cmd(cfg, f"\"import pathlib; p = pathlib.Path(r'{status_file}'); print(p.read_text()) if p.exists() else None\""),
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0 and result.stdout.strip() and result.stdout.strip() != "None":
            return json.loads(result.stdout.strip())
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
        pass
    return None


def _tail_remote_log(cfg: RemoteConfig, log_file: str, offset: int) -> tuple[str, int]:
    """Read new log content from remote file starting at byte offset.

    Returns (new_content, new_offset).
    """
    try:
        script = (
            f"\"import os; "
            f"p = r'{log_file}'; "
            f"sz = os.path.getsize(p) if os.path.exists(p) else 0; "
            f"f = open(p, 'r') if sz > {offset} else None; "
            f"f.seek({offset}) if f else None; "
            f"data = f.read() if f else ''; "
            f"f.close() if f else None; "
            f"print(f'OFFSET:{{sz}}'); "
            f"print(data, end='')\""
        )
        result = subprocess.run(
            _py_cmd(cfg, script),
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0 and result.stdout:
            lines = result.stdout.split("\n", 1)
            if lines[0].startswith("OFFSET:"):
                new_offset = int(lines[0].split(":")[1])
                content = lines[1] if len(lines) > 1 else ""
                return content, new_offset
    except (subprocess.TimeoutExpired, ValueError, OSError):
        pass
    return "", offset


def _kill_remote_pid(cfg: RemoteConfig, pid: int):
    """Kill a process on the remote host by PID (best-effort)."""
    try:
        script = (
            f"\"import os, signal; "
            f"exec('try:\\n import ctypes\\n k = ctypes.windll.kernel32\\n h = k.OpenProcess(1, False, {pid})\\n k.TerminateProcess(h, 1) if h else None\\n k.CloseHandle(h) if h else None\\nexcept: pass') "
            f"if os.name == 'nt' else "
            f"exec('try:\\n os.kill({pid}, signal.SIGTERM)\\nexcept: pass')\""
        )
        subprocess.run(
            _py_cmd(cfg, script),
            capture_output=True, timeout=15,
        )
    except (subprocess.TimeoutExpired, OSError):
        pass


def remote_train_monitor(
    job_id: int,
    cfg: RemoteConfig,
    local_dlc_dir: Path,
    subject_name: str,
    log_path: str,
    progress_parser,
    on_complete,
    registry,
    cam_names: list[str] | None = None,
    resume: bool = False,
    skip_train: bool = False,
    labels_dir_name: str = "labels_v1",
    iteration: int | None = None,
):
    """Remote training lifecycle. Runs in a daemon thread.

    Uploads the DLC project and a self-contained training script to the remote,
    launches it as a detached process, then polls status.json for progress.
    The remote process survives SSH disconnections and app restarts.

    Phase 1 (0-3%):    scp local DLC dir to remote (skip if resume)
    Phase 1b (3-5%):   fix config, detect shuffle, create dataset (skip if resume)
    Phase 1c (5-7%):   upload subject videos for cropping (skip if resume)
    Phase 1d (7-8%):   upload training script, launch detached (skip if resume)
    Monitor (8-90%):   poll status.json + remote log every 10s
    Phase 5 (90-100%): download model + CSV results to local

    Args:
        job_id: Database job ID
        cfg: RemoteConfig with SSH details
        local_dlc_dir: Local path to subject's DLC project directory
        subject_name: Subject name (used for remote path)
        log_path: Path for log file
        progress_parser: callable(line) -> float|None (unused, kept for API compat)
        on_complete: callable(job_id, returncode) for post-completion
        registry: JobRegistry instance
        cam_names: Camera names for cropping (e.g. ['OS', 'OD'])
        resume: If True, skip upload phases and go straight to monitoring
        skip_train: If True, pass --skip-train to remote script (crop+analyze only)
        labels_dir_name: Output directory name for labels (e.g. 'labels_v1', 'labels_v2')
        iteration: If set, override iteration in remote config.yaml (0 for v1 model)
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    remote_project_dir = f"{cfg.work_dir}/{subject_name}"
    remote_config = f"{remote_project_dir}/config.yaml"
    remote_labels_dir = f"{remote_project_dir}/{labels_dir_name}"
    remote_log_file = f"{remote_project_dir}/train.log"
    remote_status_file = f"{remote_project_dir}/status.json"
    remote_pid = None

    cancel_event = registry.register_cancel_event(job_id)

    if cam_names is None:
        from ..config import get_settings
        cam_names = get_settings().camera_names

    def _update_progress(pct):
        with get_db_ctx() as db:
            db.execute("UPDATE jobs SET progress_pct = ? WHERE id = ?",
                       (round(pct, 1), job_id))

    def _fail(msg):
        logger.error(f"Job {job_id} remote training failed: {msg}")
        with get_db_ctx() as db:
            db.execute(
                """UPDATE jobs SET status = 'failed', error_msg = ?,
                   finished_at = CURRENT_TIMESTAMP WHERE id = ?""",
                (msg, job_id),
            )

    def _check_cancel():
        if cancel_event.is_set():
            raise InterruptedError("Job cancelled")

    def _run_remote_proc(cmd, logfile, phase_name):
        """Run a remote command, log output, return process."""
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        registry._processes[job_id] = proc

        for line in proc.stdout:
            logfile.write(line)
            logfile.flush()

        proc.wait()
        registry._processes.pop(job_id, None)
        return proc

    try:
        with open(log_path, "a" if resume else "w") as logfile:

            if not resume:
                # ── Pre-upload: backfill label metadata ──────────────
                try:
                    from .labels import backfill_label_metadata
                    backfill_label_metadata(subject_name)
                except Exception as e:
                    logfile.write(f"Warning: metadata backfill failed: {e}\n")
                    logfile.flush()

                # ── Phase 1: Upload DLC dir ──────────────────────────
                logfile.write(f"=== Phase 1: Uploading {local_dlc_dir} to {cfg.host}:{remote_project_dir} ===\n")
                logfile.flush()
                _check_cancel()

                subprocess.run(
                    _py_cmd(cfg, f"\"import os; os.makedirs(r'{cfg.work_dir}', exist_ok=True)\""),
                    capture_output=True, timeout=15,
                )

                upload_cmd = _scp_base_args(cfg) + [
                    str(local_dlc_dir),
                    f"{cfg.host}:{cfg.work_dir}/",
                ]
                proc = _run_remote_proc(upload_cmd, logfile, "Upload")

                if proc.returncode != 0:
                    _fail(f"Upload failed (exit {proc.returncode})")
                    if on_complete:
                        on_complete(job_id, proc.returncode)
                    return

                _update_progress(3.0)
                logfile.write("=== Upload complete ===\n")
                logfile.flush()

                # ── Fix remote config.yaml project_path ──────────────
                fix_fields = [
                    f"t = re.sub(r'(?m)^project_path:.*', r'project_path: {remote_project_dir}', t)",
                ]
                if iteration is not None:
                    fix_fields.append(
                        f"t = re.sub(r'(?m)^iteration:.*', r'iteration: {iteration}', t)"
                    )
                fix_subs = "; ".join(fix_fields)
                fix_script = (
                    f"\"import re, pathlib; "
                    f"p = pathlib.Path(r'{remote_project_dir}/config.yaml'); "
                    f"t = p.read_text(); "
                    f"{fix_subs}; "
                    f"p.write_text(t)\""
                )
                subprocess.run(
                    _py_cmd(cfg, fix_script),
                    capture_output=True, timeout=15,
                )

                # ── Convert CSV labels to H5 ─────────────────────────
                csv_to_h5_script = (
                    f"\"import pandas as pd, glob; "
                    f"csvs = glob.glob(r'{remote_project_dir}/labeled-data/*/CollectedData_*.csv'); "
                    f"[pd.read_csv(c, header=[0,1,2], index_col=[0,1,2])"
                    f".to_hdf(c.replace('.csv','.h5'), key='df_with_missing', mode='w') "
                    f"for c in csvs]; "
                    f"print(f'Converted {{len(csvs)}} CSV(s) to H5')\""
                )
                result = subprocess.run(
                    _py_cmd(cfg, csv_to_h5_script),
                    capture_output=True, text=True, timeout=30,
                )
                if result.returncode == 0:
                    logfile.write(f"=== {result.stdout.strip()} ===\n")
                else:
                    logfile.write(f"Warning: CSV to H5 conversion failed: {result.stderr.strip()[:200]}\n")
                logfile.flush()

                # ── Phase 1b: Detect shuffle ─────────────────────────
                # Read iteration from config.yaml so we look for the right
                # iteration-N directory (a refine bumps iteration).
                _check_cancel()
                iter_script = (
                    f"\"import yaml; "
                    f"cfg = yaml.safe_load(open(r'{remote_config}')); "
                    f"print(cfg.get('iteration', 0))\""
                )
                iter_result = subprocess.run(
                    _py_cmd(cfg, iter_script),
                    capture_output=True, text=True, timeout=15,
                )
                remote_iteration = int(iter_result.stdout.strip()) if iter_result.returncode == 0 else 0

                detect_script = (
                    f"\"import glob, re, sys; "
                    f"hits = glob.glob(r'{remote_project_dir}/dlc-models-pytorch/iteration-{remote_iteration}/*/train/pytorch_config.yaml'); "
                    f"shuffles = [int(m.group(1)) for h in hits if (m := re.search(r'shuffle(\\d+)', h))]; "
                    f"print(max(shuffles)) if shuffles else sys.exit(1)\""
                )
                result = subprocess.run(
                    _py_cmd(cfg, detect_script),
                    capture_output=True, text=True, timeout=30,
                )

                if result.returncode == 0:
                    shuffle = int(result.stdout.strip())
                    logfile.write(f"=== Detected existing pytorch config (shuffle {shuffle}) ===\n")
                    logfile.flush()
                else:
                    logfile.write(f"=== Creating training dataset on {cfg.host} ===\n")
                    logfile.flush()

                    create_ds_cmd = _py_cmd(
                        cfg,
                        f"\"from deeplabcut.core.engine import Engine; import deeplabcut; deeplabcut.create_training_dataset(r'{remote_config}', engine=Engine.PYTORCH)\"",
                    )
                    proc = _run_remote_proc(create_ds_cmd, logfile, "Create dataset")

                    if proc.returncode != 0:
                        _fail(f"Create training dataset failed (exit {proc.returncode})")
                        if on_complete:
                            on_complete(job_id, proc.returncode)
                        return

                    logfile.write("=== Training dataset created ===\n")
                    logfile.flush()

                    result = subprocess.run(
                        _py_cmd(cfg, detect_script),
                        capture_output=True, text=True, timeout=30,
                    )
                    if result.returncode != 0:
                        _fail("Could not detect shuffle number after creating training dataset")
                        if on_complete:
                            on_complete(job_id, 1)
                        return
                    shuffle = int(result.stdout.strip())

                _update_progress(5.0)

                # ── Phase 1c: Upload subject videos for cropping ─────
                _check_cancel()
                logfile.write("=== Phase 1c: Uploading subject videos ===\n")
                logfile.flush()

                from .video import get_subject_videos
                local_videos = get_subject_videos(subject_name)

                if local_videos:
                    # Check which videos already exist on remote
                    result = subprocess.run(
                        _py_cmd(cfg, f"\"import os; print('\\n'.join(os.listdir(r'{cfg.work_dir}')))\""),
                        capture_output=True, text=True, timeout=15,
                    )
                    remote_files = set(result.stdout.strip().splitlines()) if result.returncode == 0 else set()

                    to_upload = [v for v in local_videos if Path(v).name not in remote_files]
                    if len(to_upload) < len(local_videos):
                        logfile.write(f"  Skipping {len(local_videos) - len(to_upload)} already-uploaded videos\n")

                    for vid_path in to_upload:
                        upload_vid_cmd = _scp_base_args(cfg) + [
                            vid_path,
                            f"{cfg.host}:{cfg.work_dir}/",
                        ]
                        proc = _run_remote_proc(upload_vid_cmd, logfile, "Upload video")
                        if proc.returncode != 0:
                            logfile.write(f"Warning: Failed to upload {vid_path}\n")

                _update_progress(7.0)
                logfile.write("=== Video upload complete ===\n")
                logfile.flush()

                # ── Phase 1d: Upload training script, launch detached ──
                _check_cancel()
                logfile.write("=== Phase 1d: Starting remote training process ===\n")
                logfile.flush()

                # Upload remote_train_script.py
                train_script_path = Path(__file__).parent / "remote_train_script.py"
                upload_script_cmd = _scp_base_args(cfg) + [
                    str(train_script_path),
                    f"{cfg.host}:{cfg.work_dir}/",
                ]
                proc = _run_remote_proc(upload_script_cmd, logfile, "Upload training script")
                if proc.returncode != 0:
                    _fail("Failed to upload training script")
                    if on_complete:
                        on_complete(job_id, 1)
                    return

                # Launch the training script as a detached process on remote.
                # Works on both Windows (PowerShell) and Linux (bash) remotes.
                # The script writes its PID to status.json; we poll that + the log.
                remote_script = f"{cfg.work_dir}/remote_train_script.py"
                # Build cam_names as separate list items for argparse nargs="+"
                cam_list_str = ", ".join(f"'{c}'" for c in cam_names)
                # CREATE_BREAKAWAY_FROM_JOB (0x01000000): escape sshd's Job
                # Object so the process survives SSH disconnect.
                # CREATE_NEW_PROCESS_GROUP (0x00000200): no Ctrl+C propagation.
                # Redirect stdout/stderr to log file; launcher sleeps 3s so
                # inherited handles are valid when child starts writing.
                skip_train_arg = "'--skip-train', " if skip_train else ""
                launch_script = (
                    f"\"import subprocess, os, time; "
                    f"log_fh = open(r'{remote_log_file}', 'w'); "
                    f"args = [r'{cfg.python_executable}', '-u', r'{remote_script}', "
                    f"'--config-path', r'{remote_config}', "
                    f"'--shuffle', '{shuffle}', "
                    f"'--labels-dir', r'{remote_labels_dir}', "
                    f"'--video-dir', r'{cfg.work_dir}', "
                    f"'--subject-name', '{subject_name}', "
                    f"'--cam-names', {cam_list_str}, "
                    f"'--status-file', r'{remote_status_file}', "
                    f"{skip_train_arg}]; "
                    f"flags = 0x01000200 if os.name == 'nt' else 0; "
                    f"p = subprocess.Popen(args, creationflags=flags, "
                    f"stdin=subprocess.DEVNULL, stdout=log_fh, stderr=log_fh); "
                    f"print(p.pid); time.sleep(3)\""
                )

                result = subprocess.run(
                    _py_cmd(cfg, launch_script),
                    capture_output=True, text=True, timeout=30,
                )
                if result.returncode != 0:
                    _fail(f"Failed to start remote training: {result.stderr.strip()[:200]}")
                    if on_complete:
                        on_complete(job_id, 1)
                    return

                # Parse the remote PID
                try:
                    remote_pid = int(result.stdout.strip().splitlines()[-1])
                except (ValueError, IndexError):
                    _fail(f"Could not parse remote PID from: {result.stdout.strip()[:100]}")
                    if on_complete:
                        on_complete(job_id, 1)
                    return

                # Store session info in DB (reuse tmux_session column for PID)
                with get_db_ctx() as db:
                    db.execute(
                        "UPDATE jobs SET tmux_session = ? WHERE id = ?",
                        (f"pid:{remote_pid}", job_id),
                    )

                _update_progress(8.0)
                logfile.write(f"=== Remote process started (PID {remote_pid}) ===\n")
                logfile.flush()

                # Quick health check — fail fast if process dies on startup
                time.sleep(5)
                status = _read_remote_status(cfg, remote_status_file)
                if not status:
                    proc_alive = _check_remote_pid_alive(cfg, remote_pid)
                    if not proc_alive:
                        error_log, _ = _tail_remote_log(cfg, remote_log_file, 0)
                        if error_log:
                            logfile.write(f"=== Remote process crashed on startup ===\n{error_log}\n")
                        else:
                            logfile.write("=== Remote process crashed on startup (no log output) ===\n")
                        logfile.flush()
                        error_msg = error_log.strip()[-500:] if error_log else "no output in train.log"
                        _fail(f"Remote process died immediately: {error_msg}")
                        if on_complete:
                            on_complete(job_id, 1)
                        return

            # ── Monitor loop (8-90%) ─────────────────────────────────
            # Both fresh and resume paths converge here
            if resume:
                logfile.write(f"\n=== Resuming monitoring (PID {remote_pid}) ===\n")
                logfile.flush()

            log_offset = 0
            ssh_fail_count = 0
            max_ssh_failures = 30  # 30 × 10s = 5min tolerance
            poll_interval = 10  # seconds

            while True:
                _check_cancel()
                time.sleep(poll_interval)
                _check_cancel()

                # Read status.json (also gives us latest PID if it changed)
                status = _read_remote_status(cfg, remote_status_file)

                # Update remote_pid from status if available
                if status and status.get("pid"):
                    remote_pid = status["pid"]

                # Check remote process alive
                proc_alive = _check_remote_pid_alive(cfg, remote_pid) if remote_pid else False

                # Tail remote log → append to local log
                new_log, log_offset = _tail_remote_log(cfg, remote_log_file, log_offset)
                if new_log:
                    logfile.write(new_log)
                    logfile.flush()
                    ssh_fail_count = 0  # successful SSH = reset counter

                # Update progress from status
                if status:
                    ssh_fail_count = 0
                    pct = status.get("progress_pct", 0)
                    # Scale remote 0-100% to local 8-90%
                    scaled = 8.0 + (pct / 100.0) * 82.0
                    _update_progress(scaled)

                    remote_status_val = status.get("status", "")
                    if remote_status_val == "completed":
                        logfile.write("=== Remote pipeline completed ===\n")
                        logfile.flush()
                        break
                    elif remote_status_val == "failed":
                        error = status.get("error", "Unknown error")
                        _fail(f"Remote pipeline failed: {error}")
                        if on_complete:
                            on_complete(job_id, 1)
                        return
                    elif not proc_alive:
                        # Status exists but not completed/failed, yet process died
                        error = status.get("error", "Remote process exited unexpectedly")
                        phase = status.get("phase", "unknown")
                        _fail(f"Process exited during phase '{phase}': {error}")
                        if on_complete:
                            on_complete(job_id, 1)
                        return

                elif not proc_alive:
                    # No status.json and process dead — SSH connectivity issue?
                    ssh_fail_count += 1
                    logfile.write(f"Warning: process not found and no status.json, attempt {ssh_fail_count}/{max_ssh_failures}\n")
                    logfile.flush()

                    if ssh_fail_count >= max_ssh_failures:
                        _fail("Remote process died and no status.json found")
                        if on_complete:
                            on_complete(job_id, 1)
                        return

            # ── Phase 5: Download model + CSV results ────────────────
            logfile.write(f"=== Phase 5: Downloading results from {cfg.host} ===\n")
            logfile.flush()
            _update_progress(90.0)

            # Download dlc-models-pytorch directory
            remote_models = f"{cfg.host}:{remote_project_dir}/dlc-models-pytorch"
            download_cmd = _scp_base_args(cfg) + [
                remote_models,
                str(local_dlc_dir) + "/",
            ]
            proc = _run_remote_proc(download_cmd, logfile, "Download models")
            if proc.returncode != 0:
                logfile.write(f"Warning: Model download failed (exit {proc.returncode})\n")

            # Download analysis results only (CSV, H5, pickle) — skip cropped videos
            remote_labels_path = f"{remote_project_dir}/{labels_dir_name}"
            local_labels_dir = local_dlc_dir / labels_dir_name
            local_labels_dir.mkdir(exist_ok=True)

            # List files in remote labels dir, filter to analysis outputs
            list_script = (
                f"\"import os; d = r'{remote_labels_path}'; "
                f"files = [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)) "
                f"and any(f.endswith(e) for e in ('.h5', '.csv', '.pickle'))] "
                f"if os.path.isdir(d) else []; print('\\n'.join(files))\""
            )
            result = subprocess.run(
                _py_cmd(cfg, list_script),
                capture_output=True, text=True, timeout=15,
            )
            analysis_files = [f for f in result.stdout.strip().splitlines() if f]

            if analysis_files:
                logfile.write(f"  Downloading {len(analysis_files)} analysis files from {labels_dir_name}/\n")
                for af in analysis_files:
                    dl_cmd = _scp_base_args(cfg) + [
                        f"{cfg.host}:{remote_labels_path}/{af}",
                        str(local_labels_dir / af),
                    ]
                    proc = _run_remote_proc(dl_cmd, logfile, f"Download {af}")
                    if proc.returncode == 0:
                        logfile.write(f"  Downloaded {labels_dir_name}/{af}\n")
                    else:
                        logfile.write(f"  Warning: download failed for {af}\n")
            else:
                logfile.write(f"  Warning: no analysis files found in {labels_dir_name}/\n")

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

    except InterruptedError:
        logger.info(f"Job {job_id} cancelled — killing remote process")
        if remote_pid:
            _kill_remote_pid(cfg, remote_pid)
        with get_db_ctx() as db:
            db.execute(
                """UPDATE jobs SET status = 'cancelled',
                   finished_at = CURRENT_TIMESTAMP WHERE id = ?""",
                (job_id,),
            )
    except Exception as e:
        logger.exception(f"Job {job_id} remote training error")
        _fail(str(e))
    finally:
        registry._processes.pop(job_id, None)
        registry._threads.pop(job_id, None)
        registry.unregister_cancel_event(job_id)


def remote_train_download(
    job_id: int,
    cfg: RemoteConfig,
    local_dlc_dir: Path,
    subject_name: str,
    log_path: str,
    registry,
    labels_dir_name: str = "labels_v1",
):
    """Download-only: SCP model + analysis outputs from a completed remote training run.

    No monitoring, no polling — just downloads results and marks the job done.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    remote_project_dir = f"{cfg.work_dir}/{subject_name}"

    cancel_event = registry.register_cancel_event(job_id)

    def _update_progress(pct):
        with get_db_ctx() as db:
            db.execute("UPDATE jobs SET progress_pct = ? WHERE id = ?",
                       (round(pct, 1), job_id))

    def _fail(msg):
        logger.error(f"Job {job_id} download failed: {msg}")
        with get_db_ctx() as db:
            db.execute(
                """UPDATE jobs SET status = 'failed', error_msg = ?,
                   finished_at = CURRENT_TIMESTAMP WHERE id = ?""",
                (msg, job_id),
            )

    def _run_scp(cmd, logfile, label):
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        registry._processes[job_id] = proc
        for line in proc.stdout:
            logfile.write(line)
            logfile.flush()
        proc.wait()
        registry._processes.pop(job_id, None)
        return proc

    try:
        with open(log_path, "w") as logfile:
            logfile.write(f"=== Downloading results for {subject_name} from {cfg.host} ===\n")
            logfile.flush()
            _update_progress(10.0)

            if cancel_event.is_set():
                raise InterruptedError("Job cancelled")

            # Download dlc-models-pytorch directory
            remote_models = f"{cfg.host}:{remote_project_dir}/dlc-models-pytorch"
            logfile.write(f"  Downloading dlc-models-pytorch/\n")
            logfile.flush()
            dl_cmd = _scp_base_args(cfg) + [remote_models, str(local_dlc_dir) + "/"]
            proc = _run_scp(dl_cmd, logfile, "Download models")
            if proc.returncode != 0:
                logfile.write(f"  Warning: model download failed (exit {proc.returncode})\n")
            _update_progress(40.0)

            if cancel_event.is_set():
                raise InterruptedError("Job cancelled")

            # Download analysis results (CSV, H5, pickle) from labels dir
            remote_labels_path = f"{remote_project_dir}/{labels_dir_name}"
            local_labels_dir = local_dlc_dir / labels_dir_name
            local_labels_dir.mkdir(exist_ok=True)

            list_script = (
                f"\"import os; d = r'{remote_labels_path}'; "
                f"files = [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)) "
                f"and any(f.endswith(e) for e in ('.h5', '.csv', '.pickle'))] "
                f"if os.path.isdir(d) else []; print('\\n'.join(files))\""
            )
            result = subprocess.run(
                _py_cmd(cfg, list_script),
                capture_output=True, text=True, timeout=15,
            )
            analysis_files = [f for f in result.stdout.strip().splitlines() if f]

            if analysis_files:
                logfile.write(f"  Downloading {len(analysis_files)} analysis files from {labels_dir_name}/\n")
                for i, af in enumerate(analysis_files):
                    dl_cmd = _scp_base_args(cfg) + [
                        f"{cfg.host}:{remote_labels_path}/{af}",
                        str(local_labels_dir / af),
                    ]
                    proc = _run_scp(dl_cmd, logfile, f"Download {af}")
                    if proc.returncode == 0:
                        logfile.write(f"  Downloaded {labels_dir_name}/{af}\n")
                    else:
                        logfile.write(f"  Warning: download failed for {af}\n")
                    _update_progress(40.0 + (i + 1) / len(analysis_files) * 50.0)
            else:
                logfile.write(f"  Warning: no analysis files found in {labels_dir_name}/\n")

            _update_progress(100.0)
            logfile.write("=== Download complete ===\n")
            logfile.flush()

        with get_db_ctx() as db:
            db.execute(
                """UPDATE jobs SET status = 'completed', progress_pct = 100,
                   finished_at = CURRENT_TIMESTAMP WHERE id = ?""",
                (job_id,),
            )

    except InterruptedError:
        logger.info(f"Job {job_id} download cancelled")
        with get_db_ctx() as db:
            db.execute(
                """UPDATE jobs SET status = 'cancelled',
                   finished_at = CURRENT_TIMESTAMP WHERE id = ?""",
                (job_id,),
            )
    except Exception as e:
        logger.exception(f"Job {job_id} download error")
        _fail(str(e))
    finally:
        registry._processes.pop(job_id, None)
        registry._threads.pop(job_id, None)
        registry.unregister_cancel_event(job_id)


def remote_preprocess_batch(
    job_id: int,
    cfg: RemoteConfig,
    steps: list[str],
    subjects: list[str],
    log_path: str,
    registry,
    force: bool = False,
):
    """Batch remote preprocessing: upload videos, run MP/blur, download results.

    Runs in a daemon thread. Steps can include 'mediapipe' and/or 'blur'.
    The remote processing is launched as a detached process (like training),
    so it survives SSH disconnections and local app restarts.

    Phases:
      1. Upload videos to remote (only new ones)
      2. Upload preprocessing script
      3. Launch detached preprocessing process on remote
      Monitor: Poll status.json + remote log every 10s
      5. Download results (npz files, blurred videos)

    Args:
        job_id: Database job ID
        cfg: RemoteConfig with SSH details
        steps: List of steps to run ('mediapipe', 'blur')
        subjects: List of subject names (empty = all discovered)
        log_path: Path for log file
        registry: JobRegistry instance
        force: If True, skip already-done checks and re-run for all subjects
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    from ..config import get_settings
    settings = get_settings()

    remote_video_dir = f"{cfg.work_dir}/videos"
    remote_output_dir = f"{cfg.work_dir}/preprocess_output"
    script_path = Path(__file__).parent / "remote_preprocess_script.py"

    cancel_event = registry.register_cancel_event(job_id)

    do_mp = "mediapipe" in steps
    do_blur = "blur" in steps
    download_start = 85

    def _update_progress(pct):
        with get_db_ctx() as db:
            db.execute("UPDATE jobs SET progress_pct = ? WHERE id = ?",
                        (round(pct, 1), job_id))

    def _fail(msg):
        logger.error(f"Job {job_id} remote preprocess failed: {msg}")
        with get_db_ctx() as db:
            db.execute(
                """UPDATE jobs SET status = 'failed', error_msg = ?,
                   finished_at = CURRENT_TIMESTAMP WHERE id = ?""",
                (msg, job_id),
            )

    def _check_cancel():
        if cancel_event.is_set():
            raise InterruptedError("Job cancelled")

    def _run_remote_proc(cmd, logfile, phase_name):
        """Run a remote command, log output, return process."""
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        registry._processes[job_id] = proc

        for line in proc.stdout:
            logfile.write(line)
            logfile.flush()

        proc.wait()
        registry._processes.pop(job_id, None)
        return proc

    try:
        with open(log_path, "w") as logfile:
            # ── Phase 1: Upload videos ────────────────────────────
            logfile.write(f"=== Phase 1: Uploading videos to {cfg.host}:{remote_video_dir} ===\n")
            logfile.flush()
            _check_cancel()

            # Ensure remote dirs exist
            subprocess.run(
                _py_cmd(cfg, f"\"import os; os.makedirs(r'{remote_video_dir}', exist_ok=True); os.makedirs(r'{remote_output_dir}', exist_ok=True)\""),
                capture_output=True, timeout=15,
            )

            # List remote videos to skip already-uploaded ones
            result = subprocess.run(
                _py_cmd(cfg, f"\"import os; print('\\n'.join(os.listdir(r'{remote_video_dir}')))\""),
                capture_output=True, text=True, timeout=15,
            )
            remote_files = set(result.stdout.strip().splitlines()) if result.returncode == 0 else set()

            # Find local videos to upload
            from .video import get_subject_videos
            from .discovery import _has_mediapipe, _has_deidentified

            all_videos = []
            if subjects:
                for subj_name in subjects:
                    all_videos.extend(get_subject_videos(subj_name))
            else:
                # All subjects — list all .mp4 in video dir
                import glob
                all_videos = sorted(glob.glob(str(settings.video_path / "*.mp4")))

            # Extract subject names from video filenames
            all_subject_names = set()
            for v in all_videos:
                m = re.match(r'^(.+?)_[LR]\d', Path(v).stem)
                if m:
                    all_subject_names.add(m.group(1))

            # Filter to subjects that still need each step (unless force=True)
            mp_subjects = []
            blur_subjects = []
            for subj_name in sorted(all_subject_names):
                dlc_path = settings.dlc_path / subj_name
                if do_mp and (force or not _has_mediapipe(dlc_path)):
                    mp_subjects.append(subj_name)
                if do_blur and (force or not _has_deidentified(dlc_path)):
                    blur_subjects.append(subj_name)

            needed_subjects = set(mp_subjects) | set(blur_subjects)
            skipped = all_subject_names - needed_subjects

            if skipped:
                logfile.write(f"  Skipping {len(skipped)} already-completed: {', '.join(sorted(skipped))}\n")
                logfile.flush()

            if not needed_subjects:
                logfile.write("=== All subjects already completed, nothing to do ===\n")
                logfile.flush()
                _update_progress(100.0)
                with get_db_ctx() as db:
                    db.execute(
                        """UPDATE jobs SET status = 'completed', progress_pct = 100,
                           finished_at = CURRENT_TIMESTAMP WHERE id = ?""",
                        (job_id,),
                    )
                return

            # Only upload videos for subjects that need processing
            all_videos = [v for v in all_videos
                          if any(Path(v).stem.startswith(s + '_') for s in needed_subjects)]

            to_upload = [v for v in all_videos if Path(v).name not in remote_files]
            logfile.write(f"  {len(all_videos)} videos for {len(needed_subjects)} subjects, {len(to_upload)} new to upload\n")
            logfile.flush()

            for i, vid_path in enumerate(to_upload):
                _check_cancel()
                logfile.write(f"  Uploading {Path(vid_path).name}...\n")
                logfile.flush()

                upload_cmd = _scp_base_args(cfg) + [
                    vid_path,
                    f"{cfg.host}:{remote_video_dir}/",
                ]
                proc = _run_remote_proc(upload_cmd, logfile, "Upload")
                if proc.returncode != 0:
                    logfile.write(f"  Warning: Upload failed for {Path(vid_path).name}\n")

                pct = (i + 1) / max(len(to_upload), 1) * 3.0
                _update_progress(pct)

            _update_progress(3.0)
            logfile.write("=== Upload complete ===\n")
            logfile.flush()

            # ── Phase 2: Upload script ────────────────────────────
            _check_cancel()
            logfile.write(f"=== Phase 2: Uploading preprocessing script ===\n")
            logfile.flush()

            upload_script_cmd = _scp_base_args(cfg) + [
                str(script_path),
                f"{cfg.host}:{cfg.work_dir}/",
            ]
            proc = _run_remote_proc(upload_script_cmd, logfile, "Upload script")
            if proc.returncode != 0:
                _fail("Failed to upload preprocessing script")
                return

            # Upload face_blur_overrides.json if it exists (near video dir)
            # Merge in skip=true for videos marked as no-face in the DB
            remote_overrides_file = None
            overrides_candidates = [
                settings.video_path.parent.parent / "face_blur_overrides.json",
                settings.video_path.parent / "face_blur_overrides.json",
                settings.video_path / "face_blur_overrides.json",
            ]
            overrides_data = {}
            for cand in overrides_candidates:
                if cand.is_file():
                    try:
                        overrides_data = json.loads(cand.read_text())
                    except (json.JSONDecodeError, OSError):
                        pass
                    break

            # Add skip=true for no-face videos from DB
            no_face_stems = set()
            with get_db_ctx() as nf_db:
                for subj_name in sorted(needed_subjects):
                    row = nf_db.execute(
                        "SELECT no_face_videos FROM subjects WHERE name = ?",
                        (subj_name,),
                    ).fetchone()
                    if row and row["no_face_videos"]:
                        try:
                            stems = json.loads(row["no_face_videos"])
                            no_face_stems.update(stems)
                        except (json.JSONDecodeError, TypeError):
                            pass
            for stem in sorted(no_face_stems):
                overrides_data.setdefault(stem, {})["skip"] = True
                logfile.write(f"  Override: {stem} skip=true (no faces)\n")

            if overrides_data:
                import tempfile
                tmp = tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", prefix="face_blur_overrides_",
                    delete=False,
                )
                json.dump(overrides_data, tmp, indent=2)
                tmp.close()
                upload_ov_cmd = _scp_base_args(cfg) + [
                    tmp.name,
                    f"{cfg.host}:{cfg.work_dir}/face_blur_overrides.json",
                ]
                proc = _run_remote_proc(upload_ov_cmd, logfile, "Upload overrides")
                os.unlink(tmp.name)
                if proc.returncode == 0:
                    remote_overrides_file = f"{cfg.work_dir}/face_blur_overrides.json"
                    logfile.write(f"  Uploaded merged face_blur_overrides.json\n")
                else:
                    logfile.write(f"  Warning: failed to upload face_blur_overrides.json\n")
                logfile.flush()

            _update_progress(5.0)

            # ── Phase 3: Launch detached preprocessing process ────
            _check_cancel()

            # Determine which steps to run (subject lists are passed per-step)
            pipeline_steps = []
            if do_mp and mp_subjects:
                pipeline_steps.append("mp")
            elif do_mp:
                logfile.write("=== Skipping MediaPipe (all subjects already have MP) ===\n")
                logfile.flush()
            if do_blur and blur_subjects:
                pipeline_steps.append("blur")
            elif do_blur:
                logfile.write("=== Skipping blur (all subjects already deidentified) ===\n")
                logfile.flush()

            if not pipeline_steps:
                logfile.write("=== No processing steps needed ===\n")
                logfile.flush()
            else:
                remote_script = f"{cfg.work_dir}/remote_preprocess_script.py"
                remote_log_file = f"{cfg.work_dir}/preprocess_{job_id}.log"
                remote_status_file = f"{cfg.work_dir}/preprocess_{job_id}_status.json"
                remote_pid = None

                step_summary = []
                if "mp" in pipeline_steps:
                    step_summary.append(f"mp ({len(mp_subjects)} subjects)")
                if "blur" in pipeline_steps:
                    step_summary.append(f"blur ({len(blur_subjects)} subjects)")
                logfile.write(f"=== Phase 3: Launching detached process for {', '.join(step_summary)} ===\n")
                logfile.flush()

                # Sanity check: run script import test via SSH to catch
                # errors before detached launch (where stderr is lost)
                test_result = subprocess.run(
                    _ssh_base_args(cfg) + [
                        cfg.host, cfg.python_executable, "-u", "-c",
                        f"\"import importlib.util, sys; "
                        f"spec = importlib.util.spec_from_file_location('test', r'{remote_script}'); "
                        f"print('OK')\"",
                    ],
                    capture_output=True, text=True, timeout=60,
                )
                if test_result.returncode != 0 or "OK" not in test_result.stdout:
                    error_detail = (test_result.stderr or test_result.stdout).strip()[-500:]
                    logfile.write(f"=== Script sanity check failed ===\n{error_detail}\n")
                    logfile.flush()
                    _fail(f"Remote script failed to load: {error_detail}")
                    return
                logfile.write("  Script sanity check passed\n")
                logfile.flush()

                # Build launch command (same detached pattern as training).
                # CREATE_BREAKAWAY_FROM_JOB (0x01000000): escape sshd's Job
                # Object so the process survives SSH disconnect.
                # CREATE_NEW_PROCESS_GROUP (0x00000200): no Ctrl+C propagation.
                # Launcher sleeps 3s so inherited handles are valid at startup.
                steps_list_str = ", ".join(f"'{s}'" for s in pipeline_steps)
                # Build per-step subject lists so MP doesn't re-run on
                # subjects that already have local mediapipe labels.
                launch_args_parts = [
                    f"'pipeline', r'{remote_video_dir}', r'{remote_output_dir}', "
                    f"'--steps', {steps_list_str}",
                ]
                if mp_subjects:
                    mp_list_str = ", ".join(f"'{s}'" for s in sorted(mp_subjects))
                    launch_args_parts.append(f"'--mp-subjects', {mp_list_str}")
                if blur_subjects:
                    blur_list_str = ", ".join(f"'{s}'" for s in sorted(blur_subjects))
                    launch_args_parts.append(f"'--blur-subjects', {blur_list_str}")
                if remote_overrides_file:
                    launch_args_parts.append(
                        f"'--overrides-file', r'{remote_overrides_file}'")
                if force:
                    launch_args_parts.append("'--force'")
                launch_args_str = ", ".join(launch_args_parts)
                # Popen stdout/stderr handles C-level AND Python-level output.
                # Do NOT also pass --log-file (two handles to the same file
                # causes C-level output to overwrite Python logger output).
                launch_script = (
                    f"\"import subprocess, os, time; "
                    f"log_fh = open(r'{remote_log_file}', 'w'); "
                    f"args = [r'{cfg.python_executable}', '-u', r'{remote_script}', "
                    f"{launch_args_str}, "
                    f"'--status-file', r'{remote_status_file}']; "
                    f"flags = 0x01000200 if os.name == 'nt' else 0; "
                    f"p = subprocess.Popen(args, creationflags=flags, "
                    f"stdin=subprocess.DEVNULL, stdout=log_fh, stderr=log_fh); "
                    f"print(p.pid); time.sleep(3)\""
                )

                result = subprocess.run(
                    _py_cmd(cfg, launch_script),
                    capture_output=True, text=True, timeout=30,
                )
                if result.returncode != 0:
                    _fail(f"Failed to start remote preprocessing: {result.stderr.strip()[:200]}")
                    return

                try:
                    remote_pid = int(result.stdout.strip().splitlines()[-1])
                except (ValueError, IndexError):
                    _fail(f"Could not parse remote PID from: {result.stdout.strip()[:100]}")
                    return

                with get_db_ctx() as db:
                    db.execute(
                        "UPDATE jobs SET tmux_session = ? WHERE id = ?",
                        (f"pid:{remote_pid}", job_id),
                    )

                _update_progress(7.0)
                logfile.write(f"=== Remote process started (PID {remote_pid}) ===\n")
                logfile.flush()

                # Quick health check
                time.sleep(5)
                status = _read_remote_status(cfg, remote_status_file)
                if not status:
                    proc_alive = _check_remote_pid_alive(cfg, remote_pid)
                    if not proc_alive:
                        error_log, _ = _tail_remote_log(cfg, remote_log_file, 0)
                        if error_log:
                            logfile.write(f"=== Remote process crashed on startup ===\n{error_log}\n")
                        else:
                            logfile.write("=== Remote process crashed on startup (no log output) ===\n")
                        logfile.flush()
                        error_msg = error_log.strip()[-500:] if error_log else "no output in preprocess.log"
                        _fail(f"Remote process died immediately: {error_msg}")
                        return

                # ── Monitor loop (7-85%) ──────────────────────────
                log_offset = 0
                ssh_fail_count = 0
                max_ssh_failures = 30
                poll_interval = 10

                while True:
                    _check_cancel()
                    time.sleep(poll_interval)
                    _check_cancel()

                    status = _read_remote_status(cfg, remote_status_file)

                    if status and status.get("pid"):
                        remote_pid = status["pid"]

                    proc_alive = _check_remote_pid_alive(cfg, remote_pid) if remote_pid else False

                    new_log, log_offset = _tail_remote_log(cfg, remote_log_file, log_offset)
                    if new_log:
                        logfile.write(new_log)
                        logfile.flush()
                        ssh_fail_count = 0

                    if status:
                        ssh_fail_count = 0
                        pct = status.get("progress_pct", 0)
                        # Scale remote 0-100% to local 7-85%
                        scaled = 7.0 + (pct / 100.0) * 78.0
                        _update_progress(scaled)

                        remote_status_val = status.get("status", "")
                        if remote_status_val == "completed":
                            error_info = status.get("error")
                            if error_info:
                                logfile.write(f"=== Remote preprocessing completed with errors: {error_info} ===\n")
                            else:
                                logfile.write("=== Remote preprocessing completed ===\n")
                            logfile.flush()
                            break
                        elif remote_status_val == "failed":
                            error = status.get("error", "Unknown error")
                            _fail(f"Remote preprocessing failed: {error}")
                            return
                        elif not proc_alive:
                            error = status.get("error", "Remote process exited unexpectedly")
                            phase = status.get("phase", "unknown")
                            _fail(f"Process exited during phase '{phase}': {error}")
                            return

                    elif not proc_alive:
                        ssh_fail_count += 1
                        logfile.write(f"Warning: process not found and no status.json, attempt {ssh_fail_count}/{max_ssh_failures}\n")
                        logfile.flush()

                        if ssh_fail_count >= max_ssh_failures:
                            _fail("Remote process died and no status.json found")
                            return

            # ── Phase 5: Download results ─────────────────────────
            _check_cancel()
            logfile.write(f"=== Phase 5: Downloading results from {cfg.host} ===\n")
            logfile.flush()
            _update_progress(download_start)

            # List subject dirs in remote output
            result = subprocess.run(
                _py_cmd(cfg, f"\"import os; dirs = [d for d in os.listdir(r'{remote_output_dir}') if os.path.isdir(os.path.join(r'{remote_output_dir}', d))]; print('\\n'.join(dirs))\""),
                capture_output=True, text=True, timeout=15,
            )
            remote_subjects = result.stdout.strip().splitlines() if result.returncode == 0 else []
            remote_subjects = [s for s in remote_subjects if s]

            logfile.write(f"  Found {len(remote_subjects)} subject outputs\n")
            logfile.flush()

            for i, subj_name in enumerate(remote_subjects):
                _check_cancel()

                # Download mediapipe_prelabels.npz
                if do_mp:
                    remote_npz = f"{cfg.host}:{remote_output_dir}/{subj_name}/mediapipe_prelabels.npz"
                    local_dlc_dir = settings.dlc_path / subj_name
                    local_dlc_dir.mkdir(parents=True, exist_ok=True)
                    local_npz = str(local_dlc_dir / "mediapipe_prelabels.npz")

                    dl_cmd = _scp_base_args(cfg) + [remote_npz, local_npz]
                    proc = _run_remote_proc(dl_cmd, logfile, f"Download npz {subj_name}")
                    if proc.returncode == 0:
                        logfile.write(f"  Downloaded {subj_name}/mediapipe_prelabels.npz\n")
                    else:
                        logfile.write(f"  Warning: npz download failed for {subj_name}\n")

                # Download deidentified videos
                if do_blur:
                    remote_deident = f"{cfg.host}:{remote_output_dir}/{subj_name}/deidentified"
                    local_deident_dir = settings.video_path / "deidentified"
                    local_deident_dir.mkdir(parents=True, exist_ok=True)

                    # SCP individual deidentified videos (not the whole dir)
                    result = subprocess.run(
                        _py_cmd(cfg, f"\"import os; d = r'{remote_output_dir}/{subj_name}/deidentified'; print('\\n'.join(os.listdir(d))) if os.path.isdir(d) else print('')\""),
                        capture_output=True, text=True, timeout=15,
                    )
                    deident_files = [f for f in result.stdout.strip().splitlines() if f.endswith(".mp4")]

                    for df in deident_files:
                        dl_cmd = _scp_base_args(cfg) + [
                            f"{cfg.host}:{remote_output_dir}/{subj_name}/deidentified/{df}",
                            str(local_deident_dir / df),
                        ]
                        proc = _run_remote_proc(dl_cmd, logfile, f"Download blur {df}")
                        if proc.returncode == 0:
                            logfile.write(f"  Downloaded deidentified/{df}\n")

                    # Write .deidentified marker
                    if deident_files:
                        local_dlc_dir = settings.dlc_path / subj_name
                        local_dlc_dir.mkdir(parents=True, exist_ok=True)
                        (local_dlc_dir / ".deidentified").write_text("")
                        logfile.write(f"  Wrote .deidentified marker for {subj_name}\n")

                logfile.flush()
                pct = download_start + (i + 1) / max(len(remote_subjects), 1) * (100 - download_start)
                _update_progress(pct)

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

    except InterruptedError:
        logger.info(f"Job {job_id} remote preprocess cancelled")
        # Kill remote process if we launched one
        try:
            with get_db_ctx() as db:
                row = db.execute("SELECT tmux_session FROM jobs WHERE id = ?",
                                 (job_id,)).fetchone()
                if row and row[0] and row[0].startswith("pid:"):
                    pid = int(row[0].split(":")[1])
                    _kill_remote_pid(cfg, pid)
        except Exception:
            pass
        with get_db_ctx() as db:
            db.execute(
                """UPDATE jobs SET status = 'cancelled',
                   finished_at = CURRENT_TIMESTAMP WHERE id = ?""",
                (job_id,),
            )
    except Exception as e:
        logger.exception(f"Job {job_id} remote preprocess error")
        _fail(str(e))
    finally:
        registry._processes.pop(job_id, None)
        registry._threads.pop(job_id, None)
        registry.unregister_cancel_event(job_id)


def remote_preprocess_download(
    job_id: int,
    cfg: RemoteConfig,
    steps: list[str],
    subjects: list[str],
    log_path: str,
    registry,
):
    """Download-only variant of remote_preprocess_batch.

    Skips upload/launch/monitor — just downloads results from the remote
    preprocess_output directory for the given subjects. Runs in a daemon thread.

    Args:
        job_id: Database job ID
        cfg: RemoteConfig with SSH details
        steps: List of steps whose results to download ('mediapipe', 'blur')
        subjects: List of subject names to download for
        log_path: Path for log file
        registry: JobRegistry instance
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    from ..config import get_settings
    settings = get_settings()

    remote_output_dir = f"{cfg.work_dir}/preprocess_output"
    do_mp = "mediapipe" in steps
    do_blur = "blur" in steps

    cancel_event = registry.register_cancel_event(job_id)

    def _update_progress(pct):
        with get_db_ctx() as db:
            db.execute("UPDATE jobs SET progress_pct = ? WHERE id = ?",
                       (round(pct, 1), job_id))

    def _fail(msg):
        logger.error(f"Job {job_id} redownload preprocess failed: {msg}")
        with get_db_ctx() as db:
            db.execute(
                """UPDATE jobs SET status = 'failed', error_msg = ?,
                   finished_at = CURRENT_TIMESTAMP WHERE id = ?""",
                (msg, job_id),
            )

    def _check_cancel():
        if cancel_event.is_set():
            raise InterruptedError("Job cancelled")

    def _run_scp(cmd, logfile, label):
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        registry._processes[job_id] = proc
        for line in proc.stdout:
            logfile.write(line)
            logfile.flush()
        proc.wait()
        registry._processes.pop(job_id, None)
        return proc

    try:
        with open(log_path, "w") as logfile:
            # Determine which subjects have remote output
            download_subjects = list(subjects)
            if not download_subjects:
                # List all subject dirs in remote output
                result = subprocess.run(
                    _py_cmd(cfg, f"\"import os; dirs = [d for d in os.listdir(r'{remote_output_dir}') if os.path.isdir(os.path.join(r'{remote_output_dir}', d))]; print('\\n'.join(dirs))\""),
                    capture_output=True, text=True, timeout=15,
                )
                download_subjects = [s for s in result.stdout.strip().splitlines() if s]

            # Check which subjects actually have output on the remote
            logfile.write(f"=== Checking remote output for {len(download_subjects)} subjects ===\n")
            logfile.write(f"  Steps: {', '.join(steps)}\n")
            logfile.flush()

            result = subprocess.run(
                _py_cmd(cfg, f"\"import os; d = r'{remote_output_dir}'; dirs = [x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))] if os.path.isdir(d) else []; print('\\n'.join(dirs))\""),
                capture_output=True, text=True, timeout=15,
            )
            available_remote = set(result.stdout.strip().splitlines()) if result.returncode == 0 else set()

            missing = [s for s in download_subjects if s not in available_remote]
            if missing:
                logfile.write(f"  Warning: no remote output for: {', '.join(missing)}\n")
                logfile.flush()

            download_subjects = [s for s in download_subjects if s in available_remote]
            if not download_subjects:
                msg = "No remote output found for any of the requested subjects"
                logfile.write(f"=== {msg} ===\n")
                logfile.flush()
                _fail(msg)
                return

            logfile.write(f"  Found remote output for {len(download_subjects)} subjects\n")
            logfile.flush()
            _update_progress(5.0)

            total_downloaded = 0

            for i, subj_name in enumerate(download_subjects):
                _check_cancel()

                if do_mp:
                    remote_npz = f"{cfg.host}:{remote_output_dir}/{subj_name}/mediapipe_prelabels.npz"
                    local_dlc_dir = settings.dlc_path / subj_name
                    local_dlc_dir.mkdir(parents=True, exist_ok=True)
                    local_npz = str(local_dlc_dir / "mediapipe_prelabels.npz")

                    dl_cmd = _scp_base_args(cfg) + [remote_npz, local_npz]
                    proc = _run_scp(dl_cmd, logfile, f"Download npz {subj_name}")
                    if proc.returncode == 0:
                        logfile.write(f"  Downloaded {subj_name}/mediapipe_prelabels.npz\n")
                        total_downloaded += 1
                    else:
                        logfile.write(f"  Warning: npz not found for {subj_name}\n")

                if do_blur:
                    # List deidentified video files on remote
                    result = subprocess.run(
                        _py_cmd(cfg, f"\"import os; d = r'{remote_output_dir}/{subj_name}/deidentified'; print('\\n'.join(os.listdir(d))) if os.path.isdir(d) else print('NO_DIR')\""),
                        capture_output=True, text=True, timeout=15,
                    )
                    raw_lines = result.stdout.strip().splitlines()
                    if raw_lines == ["NO_DIR"]:
                        logfile.write(f"  Warning: no deidentified/ dir for {subj_name}\n")
                        deident_files = []
                    else:
                        deident_files = [f for f in raw_lines if f.endswith(".mp4")]

                    local_deident_dir = settings.video_path / "deidentified"
                    local_deident_dir.mkdir(parents=True, exist_ok=True)

                    for df in deident_files:
                        dl_cmd = _scp_base_args(cfg) + [
                            f"{cfg.host}:{remote_output_dir}/{subj_name}/deidentified/{df}",
                            str(local_deident_dir / df),
                        ]
                        proc = _run_scp(dl_cmd, logfile, f"Download blur {df}")
                        if proc.returncode == 0:
                            logfile.write(f"  Downloaded deidentified/{df}\n")
                            total_downloaded += 1

                    if deident_files:
                        local_dlc_dir = settings.dlc_path / subj_name
                        local_dlc_dir.mkdir(parents=True, exist_ok=True)
                        (local_dlc_dir / ".deidentified").write_text("")
                        logfile.write(f"  Wrote .deidentified marker for {subj_name}\n")

                logfile.flush()
                pct = 5.0 + (i + 1) / max(len(download_subjects), 1) * 90.0
                _update_progress(pct)

            _update_progress(100.0)
            if total_downloaded == 0:
                logfile.write("=== Warning: completed but no files were actually downloaded ===\n")
            else:
                logfile.write(f"=== Re-download complete: {total_downloaded} files ===\n")
            logfile.flush()

        with get_db_ctx() as db:
            db.execute(
                """UPDATE jobs SET status = 'completed', progress_pct = 100,
                   finished_at = CURRENT_TIMESTAMP WHERE id = ?""",
                (job_id,),
            )

    except InterruptedError:
        logger.info(f"Job {job_id} redownload cancelled")
        with get_db_ctx() as db:
            db.execute(
                """UPDATE jobs SET status = 'cancelled',
                   finished_at = CURRENT_TIMESTAMP WHERE id = ?""",
                (job_id,),
            )
    except Exception as e:
        logger.exception(f"Job {job_id} redownload error")
        _fail(str(e))
    finally:
        registry._processes.pop(job_id, None)
        registry._threads.pop(job_id, None)
        registry.unregister_cancel_event(job_id)
