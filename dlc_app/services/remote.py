"""Remote DLC training via SSH: upload project, train, crop, analyze, download results."""
from __future__ import annotations

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
    """7-phase remote training lifecycle. Runs in a daemon thread.

    Phase 1 (0-3%):    scp local DLC dir to remote
    Phase 1b (3-5%):   fix config, detect shuffle, create dataset if needed
    Phase 2 (5-75%):   ssh training command, parse stdout for epoch progress
    Phase 3 (75-80%):  crop stereo videos into L/R halves on remote
    Phase 4 (80-90%):  run DLC analyze_videos + convert H5 to CSV on remote
    Phase 5 (90-100%): download model + CSV results to local

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
        return proc

    try:
        with open(log_path, "w") as logfile:
            # ── Phase 1: Upload ──────────────────────────────────────
            logfile.write(f"=== Phase 1: Uploading {local_dlc_dir} to {cfg.host}:{remote_project_dir} ===\n")
            logfile.flush()

            # Ensure remote dir exists (Python — shell-agnostic)
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

            # ── Fix remote config.yaml project_path (Python — shell-agnostic) ──
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

            # ── Phase 1b: Detect shuffle and ensure pytorch config exists ─
            remote_config = f"{remote_project_dir}/config.yaml"

            # Detect shuffle number from dlc-models-pytorch, or create dataset if missing
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
                # No pytorch config found — create training dataset on remote
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

                # Re-detect shuffle after creation
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

            # ── Phase 2: Training ────────────────────────────────────
            train_cmd = _py_cmd(
                cfg,
                f"\"from deeplabcut.core.engine import Engine; import deeplabcut; deeplabcut.train_network(r'{remote_config}', shuffle={shuffle}, engine=Engine.PYTORCH)\"",
            )

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
                        # Scale 0-100% training progress to 5-75% overall
                        scaled = 5.0 + (raw_pct / 100.0) * 70.0
                        _update_progress(scaled)

            proc.wait()
            if proc.returncode != 0:
                _fail(f"Training failed (exit {proc.returncode})")
                if on_complete:
                    on_complete(job_id, proc.returncode)
                return

            _update_progress(75.0)
            logfile.write("=== Training complete ===\n")
            logfile.flush()

            # ── Phase 3: Crop stereo videos on remote ────────────────
            logfile.write("=== Phase 3: Cropping stereo videos on remote ===\n")
            logfile.flush()

            from ..config import get_settings
            settings = get_settings()

            # Find local videos for this subject
            from .video import get_subject_videos
            local_videos = get_subject_videos(subject_name)

            if local_videos:
                # Create remote output directory
                remote_labels_dir = f"{remote_project_dir}/labels_v1"
                subprocess.run(
                    _py_cmd(cfg, f"\"import os; os.makedirs(r'{remote_labels_dir}', exist_ok=True)\""),
                    capture_output=True, timeout=15,
                )

                cam_names = settings.camera_names
                cam_names_str = repr(cam_names)

                # Upload source videos first
                for vid_path in local_videos:
                    upload_vid_cmd = _scp_base_args(cfg) + [
                        vid_path,
                        f"{cfg.host}:{cfg.work_dir}/",
                    ]
                    proc = _run_remote_proc(upload_vid_cmd, logfile, "Upload video")
                    if proc.returncode != 0:
                        logfile.write(f"Warning: Failed to upload {vid_path}\n")

                # Build crop script
                crop_script = (
                    f"\"import cv2, os, pathlib; "
                    f"cam_names = {cam_names_str}; "
                    f"video_dir = r'{cfg.work_dir}'; "
                    f"out_dir = pathlib.Path(r'{remote_labels_dir}'); "
                    f"out_dir.mkdir(exist_ok=True); "
                    f"import glob; "
                    f"videos = sorted(glob.glob(os.path.join(video_dir, '{subject_name}_*.mp4'))); "
                    f"print(f'Found {{len(videos)}} videos'); "
                    f"[exec('"
                    f"cap = cv2.VideoCapture(v);\\n"
                    f"fps = int(cap.get(cv2.CAP_PROP_FPS));\\n"
                    f"n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT));\\n"
                    f"ret, f0 = cap.read();\\n"
                    f"h, w = f0.shape[:2];\\n"
                    f"mid = w // 2;\\n"
                    f"cap.set(cv2.CAP_PROP_POS_FRAMES, 0);\\n"
                    f"stem = pathlib.Path(v).stem;\\n"
                    f"ext = pathlib.Path(v).suffix;\\n"
                    f"wL = cv2.VideoWriter(str(out_dir / f\\'{{stem}}_{{cam_names[0]}}{{ext}}\\'), cv2.VideoWriter_fourcc(*\\'avc1\\'), fps, (mid, h));\\n"
                    f"wR = cv2.VideoWriter(str(out_dir / f\\'{{stem}}_{{cam_names[1]}}{{ext}}\\'), cv2.VideoWriter_fourcc(*\\'avc1\\'), fps, (w-mid, h)) if len(cam_names) > 1 else None;\\n"
                    f"[exec(\\'ret, fr = cap.read();\\\\nwL.write(fr[:, :mid]) if ret else None;\\\\nwR.write(fr[:, mid:]) if ret and wR else None\\') for _ in range(n)];\\n"
                    f"cap.release(); wL.release();\\n"
                    f"wR.release() if wR else None;\\n"
                    f"print(f\\'Cropped {{stem}}: {{n}} frames\\')"
                    f"') for v in videos]; "
                    f"print('Crop complete')\""
                )

                # Run crop on remote
                proc = _run_remote_proc(_py_cmd(cfg, crop_script), logfile, "Crop")
                if proc.returncode != 0:
                    logfile.write(f"Warning: Remote crop failed (exit {proc.returncode}), continuing...\n")

            _update_progress(80.0)
            logfile.write("=== Crop complete ===\n")
            logfile.flush()

            # ── Phase 4: Analyze videos on remote ────────────────────
            logfile.write("=== Phase 4: Analyzing videos on remote ===\n")
            logfile.flush()

            remote_labels_dir = f"{remote_project_dir}/labels_v1"
            analyze_script = (
                f"\"from deeplabcut.core.engine import Engine; "
                f"import deeplabcut; "
                f"deeplabcut.analyze_videos(r'{remote_config}', r'{remote_labels_dir}', shuffle={shuffle}, engine=Engine.PYTORCH); "
                f"print('Analysis complete'); "
                f"deeplabcut.analyze_videos_converth5_to_csv(r'{remote_labels_dir}'); "
                f"print('H5 to CSV conversion complete')\""
            )

            proc = _run_remote_proc(_py_cmd(cfg, analyze_script), logfile, "Analyze")
            if proc.returncode != 0:
                logfile.write(f"Warning: Remote analysis failed (exit {proc.returncode}), continuing to download...\n")

            _update_progress(90.0)
            logfile.write("=== Analysis complete ===\n")
            logfile.flush()

            # ── Phase 5: Download model + CSV results ────────────────
            logfile.write(f"=== Phase 5: Downloading results from {cfg.host} ===\n")
            logfile.flush()

            # Download dlc-models-pytorch directory back to local (PyTorch engine)
            remote_models = f"{cfg.host}:{remote_project_dir}/dlc-models-pytorch"
            download_cmd = _scp_base_args(cfg) + [
                remote_models,
                str(local_dlc_dir) + "/",
            ]
            proc = _run_remote_proc(download_cmd, logfile, "Download models")
            if proc.returncode != 0:
                logfile.write(f"Warning: Model download failed (exit {proc.returncode})\n")

            # Download labels_v1 directory (cropped videos + CSV analysis results)
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

    except Exception as e:
        logger.exception(f"Job {job_id} remote training error")
        _fail(str(e))
    finally:
        registry._processes.pop(job_id, None)
        registry._threads.pop(job_id, None)
