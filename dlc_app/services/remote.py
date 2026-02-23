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


def _check_tmux_alive(cfg: RemoteConfig, session_name: str) -> bool:
    """Check if a tmux session exists on the remote host."""
    try:
        result = subprocess.run(
            _ssh_base_args(cfg) + [cfg.host, "tmux", "has-session", "-t", session_name],
            capture_output=True, text=True, timeout=15,
        )
        return result.returncode == 0
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


def _kill_tmux_session(cfg: RemoteConfig, session_name: str):
    """Kill a tmux session on the remote host (best-effort)."""
    try:
        subprocess.run(
            _ssh_base_args(cfg) + [cfg.host, "tmux", "kill-session", "-t", session_name],
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
):
    """Remote training lifecycle via tmux. Runs in a daemon thread.

    Uploads the DLC project and a self-contained training script to the remote,
    launches it inside a tmux session, then polls status.json for progress.
    The remote process survives SSH disconnections and app restarts.

    Phase 1 (0-3%):    scp local DLC dir to remote (skip if resume)
    Phase 1b (3-5%):   fix config, detect shuffle, create dataset (skip if resume)
    Phase 1c (5-7%):   upload subject videos for cropping (skip if resume)
    Phase 1d (7-8%):   upload training script, start tmux (skip if resume)
    Monitor (8-90%):   poll tmux + status.json every 10s
    Phase 5 (90-100%): download model + CSV results to local

    Args:
        job_id: Database job ID
        cfg: RemoteConfig with SSH details
        local_dlc_dir: Local path to subject's DLC project directory
        subject_name: Subject name (used for remote path)
        log_path: Path for log file
        progress_parser: callable(line) -> float|None (unused in tmux mode, kept for API compat)
        on_complete: callable(job_id, returncode) for post-completion
        registry: JobRegistry instance
        cam_names: Camera names for cropping (e.g. ['OS', 'OD'])
        resume: If True, skip upload phases and go straight to monitoring
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    remote_project_dir = f"{cfg.work_dir}/{subject_name}"
    remote_config = f"{remote_project_dir}/config.yaml"
    remote_labels_dir = f"{remote_project_dir}/labels_v1"
    remote_log_file = f"{remote_project_dir}/train.log"
    remote_status_file = f"{remote_project_dir}/status.json"
    session_name = f"dlc_job_{job_id}"

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
                fix_script = (
                    f"\"import re, pathlib; "
                    f"p = pathlib.Path(r'{remote_project_dir}/config.yaml'); "
                    f"t = p.read_text(); "
                    f"t = re.sub(r'(?m)^project_path:.*', r'project_path: {remote_project_dir}', t); "
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
                _check_cancel()
                detect_script = (
                    f"\"import glob, re, sys; "
                    f"hits = glob.glob(r'{remote_project_dir}/dlc-models-pytorch/iteration-*/*/train/pytorch_config.yaml'); "
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
                    for vid_path in local_videos:
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

                # ── Phase 1d: Upload training script, start tmux ─────
                _check_cancel()
                logfile.write("=== Phase 1d: Starting tmux session ===\n")
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

                # Build the tmux command:
                # tmux new-session -d -s <name> '<python> -u <script> <args> 2>&1 | tee <log>'
                cam_args = " ".join(cam_names)
                remote_script = f"{cfg.work_dir}/remote_train_script.py"
                train_cmd_str = (
                    f"{cfg.python_executable} -u {remote_script}"
                    f" --config-path {remote_config}"
                    f" --shuffle {shuffle}"
                    f" --labels-dir {remote_labels_dir}"
                    f" --video-dir {cfg.work_dir}"
                    f" --subject-name {subject_name}"
                    f" --cam-names {cam_args}"
                    f" --status-file {remote_status_file}"
                    f" 2>&1 | tee {remote_log_file}"
                )

                tmux_cmd = _ssh_base_args(cfg) + [
                    cfg.host,
                    "tmux", "new-session", "-d", "-s", session_name,
                    train_cmd_str,
                ]

                result = subprocess.run(
                    tmux_cmd, capture_output=True, text=True, timeout=30,
                )
                if result.returncode != 0:
                    _fail(f"Failed to start tmux session: {result.stderr.strip()[:200]}")
                    if on_complete:
                        on_complete(job_id, 1)
                    return

                # Store tmux session name in DB
                with get_db_ctx() as db:
                    db.execute(
                        "UPDATE jobs SET tmux_session = ? WHERE id = ?",
                        (session_name, job_id),
                    )

                _update_progress(8.0)
                logfile.write(f"=== tmux session '{session_name}' started ===\n")
                logfile.flush()

            # ── Monitor loop (8-90%) ─────────────────────────────────
            # Both fresh and resume paths converge here
            if resume:
                logfile.write(f"\n=== Resuming monitoring of tmux session '{session_name}' ===\n")
                logfile.flush()

            log_offset = 0
            ssh_fail_count = 0
            max_ssh_failures = 30  # 30 × 10s = 5min tolerance
            poll_interval = 10  # seconds

            while True:
                _check_cancel()
                time.sleep(poll_interval)
                _check_cancel()

                # Check tmux alive
                tmux_alive = _check_tmux_alive(cfg, session_name)

                # Read status.json
                status = _read_remote_status(cfg, remote_status_file)

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
                    elif not tmux_alive:
                        # Status exists but not completed/failed, yet tmux died
                        error = status.get("error", "tmux session exited unexpectedly")
                        phase = status.get("phase", "unknown")
                        _fail(f"tmux exited during phase '{phase}': {error}")
                        if on_complete:
                            on_complete(job_id, 1)
                        return

                elif not tmux_alive:
                    # No status.json and tmux dead — SSH connectivity issue?
                    ssh_fail_count += 1
                    logfile.write(f"Warning: tmux not found and no status.json, attempt {ssh_fail_count}/{max_ssh_failures}\n")
                    logfile.flush()

                    if ssh_fail_count >= max_ssh_failures:
                        _fail("tmux session died and no status.json found")
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

            # Download labels_v1 directory
            remote_labels = f"{cfg.host}:{remote_project_dir}/labels_v1"
            local_labels_dir = local_dlc_dir / "labels_v1"
            local_labels_dir.mkdir(exist_ok=True)

            download_labels_cmd = _scp_base_args(cfg) + [
                remote_labels,
                str(local_dlc_dir) + "/",
            ]
            proc = _run_remote_proc(download_labels_cmd, logfile, "Download labels_v1")
            if proc.returncode != 0:
                logfile.write(f"Warning: labels_v1 download failed\n")

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
        logger.info(f"Job {job_id} cancelled — killing tmux session")
        _kill_tmux_session(cfg, session_name)
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


def remote_preprocess_batch(
    job_id: int,
    cfg: RemoteConfig,
    steps: list[str],
    subjects: list[str],
    log_path: str,
    registry,
):
    """Batch remote preprocessing: upload videos, run MP/blur, download results.

    Runs in a daemon thread. Steps can include 'mediapipe' and/or 'blur'.

    Phases:
      1. Upload videos to remote (only new ones)
      2. Upload preprocessing script
      3. Run MediaPipe (if requested)
      4. Run blur (if requested)
      5. Download results (npz files, blurred videos)
      6. Write local markers

    Args:
        job_id: Database job ID
        cfg: RemoteConfig with SSH details
        steps: List of steps to run ('mediapipe', 'blur')
        subjects: List of subject names (empty = all discovered)
        log_path: Path for log file
        registry: JobRegistry instance
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    from ..config import get_settings
    settings = get_settings()

    remote_video_dir = f"{cfg.work_dir}/videos"
    remote_output_dir = f"{cfg.work_dir}/preprocess_output"
    script_path = Path(__file__).parent / "remote_preprocess_script.py"

    cancel_event = registry.register_cancel_event(job_id)

    # Progress allocation across phases
    do_mp = "mediapipe" in steps
    do_blur = "blur" in steps
    # Upload: 0-5%, MP: 5-50%, Blur: 50-90%, Download: 90-100%
    # Adjust ranges based on which steps are requested
    if do_mp and do_blur:
        mp_range = (5, 45)
        blur_range = (45, 85)
    elif do_mp:
        mp_range = (5, 85)
        blur_range = None
    else:
        mp_range = None
        blur_range = (5, 85)
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

    def _run_remote_proc_with_progress(cmd, logfile, phase_name,
                                        pct_start, pct_end):
        """Run remote command, parse PROGRESS: lines, scale to pct range."""
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        registry._processes[job_id] = proc

        for line in proc.stdout:
            logfile.write(line)
            logfile.flush()

            # Parse PROGRESS:subject:pct
            m = re.match(r"PROGRESS:([^:]+):([\d.]+)", line.strip())
            if m:
                raw_pct = float(m.group(2))
                # Scale per-subject progress to overall range
                # (crude: treats all subjects as equal weight)
                scaled = pct_start + (raw_pct / 100.0) * (pct_end - pct_start)
                _update_progress(scaled)

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
            all_videos = []
            if subjects:
                for subj_name in subjects:
                    all_videos.extend(get_subject_videos(subj_name))
            else:
                # All subjects — list all .mp4 in video dir
                import glob
                all_videos = sorted(glob.glob(str(settings.video_path / "*.mp4")))

            to_upload = [v for v in all_videos if Path(v).name not in remote_files]
            logfile.write(f"  {len(all_videos)} videos total, {len(to_upload)} new to upload\n")
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

            _update_progress(5.0)

            # Build subject filter arg
            subject_arg = ""
            if subjects and len(subjects) == 1:
                subject_arg = f" --subject {subjects[0]}"

            # ── Phase 3: MediaPipe ────────────────────────────────
            if do_mp:
                _check_cancel()
                logfile.write(f"=== Phase 3: Running MediaPipe on {cfg.host} ===\n")
                logfile.flush()

                mp_cmd = _ssh_base_args(cfg) + [
                    cfg.host,
                    cfg.python_executable, "-u",
                    f"{cfg.work_dir}/remote_preprocess_script.py",
                    "mp", remote_video_dir, remote_output_dir,
                ] + (["--subject", subjects[0]] if subjects and len(subjects) == 1 else [])

                proc = _run_remote_proc_with_progress(
                    mp_cmd, logfile, "MediaPipe",
                    mp_range[0], mp_range[1],
                )

                if proc.returncode != 0:
                    _fail(f"MediaPipe failed (exit {proc.returncode})")
                    return

                _update_progress(mp_range[1])
                logfile.write("=== MediaPipe complete ===\n")
                logfile.flush()

            # ── Phase 4: Blur ─────────────────────────────────────
            if do_blur:
                _check_cancel()
                logfile.write(f"=== Phase 4: Running blur on {cfg.host} ===\n")
                logfile.flush()

                blur_cmd = _ssh_base_args(cfg) + [
                    cfg.host,
                    cfg.python_executable, "-u",
                    f"{cfg.work_dir}/remote_preprocess_script.py",
                    "blur", remote_video_dir, remote_output_dir,
                ] + (["--subject", subjects[0]] if subjects and len(subjects) == 1 else [])

                proc = _run_remote_proc_with_progress(
                    blur_cmd, logfile, "Blur",
                    blur_range[0], blur_range[1],
                )

                if proc.returncode != 0:
                    _fail(f"Blur failed (exit {proc.returncode})")
                    return

                _update_progress(blur_range[1])
                logfile.write("=== Blur complete ===\n")
                logfile.flush()

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
        registry.unregister_cancel_event(job_id)
