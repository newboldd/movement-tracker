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


def _remote_file_size(cfg: RemoteConfig, remote_path: str, timeout: int = 10) -> int | None:
    """Return remote file size in bytes, or None if missing / unreachable.

    Used by upload helpers to skip re-transferring files that are already
    on the host with the expected size — turns batched upload phases into
    idempotent, restart-tolerant operations.
    """
    try:
        result = subprocess.run(
            _py_cmd(cfg,
                    f"\"import os; p=r'{remote_path}'; print(os.path.getsize(p) if os.path.isfile(p) else -1)\""),
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode == 0:
            try:
                v = int(result.stdout.strip().splitlines()[-1])
                return v if v >= 0 else None
            except (ValueError, IndexError):
                pass
    except (subprocess.TimeoutExpired, OSError):
        pass
    return None


def _scp_if_changed(cfg: RemoteConfig, local_path: str, remote_path: str,
                    timeout: int, logfile=None, label: str = "") -> tuple[bool, str]:
    """Idempotent SCP: skips upload when the remote already has a same-size
    copy.  Returns (success, action) where action ∈ {"uploaded", "skipped",
    "failed"}.  ``label`` is a short tag printed to the log on each line.
    """
    try:
        local_size = os.path.getsize(local_path)
    except OSError:
        return False, "failed"
    remote_size = _remote_file_size(cfg, remote_path)
    if remote_size is not None and remote_size == local_size:
        if logfile:
            logfile.write(f"  {label}skipped upload (already on remote, {local_size} bytes)\n")
            logfile.flush()
        return True, "skipped"
    proc = subprocess.run(
        _scp_base_args(cfg) + [local_path, f"{cfg.host}:{remote_path}"],
        capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        if logfile:
            logfile.write(f"  {label}upload FAILED: {proc.stderr[-200:]}\n")
            logfile.flush()
        return False, "failed"
    if logfile:
        logfile.write(f"  {label}uploaded ({local_size} bytes)\n")
        logfile.flush()
    return True, "uploaded"


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
        # Simple approach: ctypes on Windows, /proc on Linux, os.kill fallback
        script = (
            f"\"import os, ctypes; "
            f"alive = False; "
            f"k = getattr(getattr(ctypes, 'windll', None), 'kernel32', None) if os.name == 'nt' else None; "
            f"h = k.OpenProcess(0x100000, False, {pid}) if k else 0; "
            f"alive = bool(h) if k else os.path.exists('/proc/{pid}'); "
            f"k.CloseHandle(h) if k and h else None; "
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
    force_create_dataset: bool = False,
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
        force_create_dataset: If True, always re-run create_training_dataset even
            when a model already exists (needed for refinement to pick up new labels)
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

    _first_epoch_at = [None]  # mutable for closure

    def _update_progress(pct, epoch=None, total_epochs=None):
        from datetime import datetime, timezone
        with get_db_ctx() as db:
            if epoch is not None:
                if epoch >= 1 and _first_epoch_at[0] is None:
                    _first_epoch_at[0] = datetime.now(timezone.utc).isoformat()
                epoch_json = json.dumps({
                    "epoch": epoch,
                    "total": total_epochs or 200,
                    "first_epoch_at": _first_epoch_at[0],
                })
                db.execute(
                    "UPDATE jobs SET progress_pct = ?, epoch_info = ? WHERE id = ?",
                    (round(pct, 1), epoch_json, job_id),
                )
            else:
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
                _check_cancel()

                subprocess.run(
                    _py_cmd(cfg, f"\"import os; os.makedirs(r'{cfg.work_dir}', exist_ok=True)\""),
                    capture_output=True, timeout=15,
                )

                if force_create_dataset:
                    # Refinement: only upload label CSVs/metadata + config.yaml.
                    # Model checkpoints already exist on remote from initial training.
                    # PNGs are regenerated on the remote by ensure_labeled_data_pngs().
                    logfile.write(
                        f"=== Phase 1: Uploading label data + config to "
                        f"{cfg.host}:{remote_project_dir} (refine) ===\n"
                    )
                    logfile.flush()

                    labeled_data_dir = local_dlc_dir / "labeled-data"
                    for round_dir in sorted(labeled_data_dir.iterdir()) if labeled_data_dir.exists() else []:
                        if not round_dir.is_dir():
                            continue
                        # Upload only CSVs, H5s, and metadata JSONs (not PNGs)
                        data_files = (
                            list(round_dir.glob("*.csv"))
                            + list(round_dir.glob("*.h5"))
                            + list(round_dir.glob("*.json"))
                        )
                        if not data_files:
                            continue

                        remote_round = f"{remote_project_dir}/labeled-data/{round_dir.name}"
                        subprocess.run(
                            _py_cmd(cfg, f"\"import os; os.makedirs(r'{remote_round}', exist_ok=True)\""),
                            capture_output=True, timeout=15,
                        )
                        upload_cmd = _scp_base_args(cfg) + [
                            str(f) for f in data_files
                        ] + [f"{cfg.host}:{remote_round}/"]
                        proc = _run_remote_proc(upload_cmd, logfile, f"Upload {round_dir.name}")
                        if proc.returncode != 0:
                            _fail(f"Upload {round_dir.name} failed (exit {proc.returncode})")
                            if on_complete:
                                on_complete(job_id, proc.returncode)
                            return

                    # Upload config.yaml — patched so video_sets covers all rounds
                    config_file = local_dlc_dir / "config.yaml"
                    if config_file.exists():
                        config_text = config_file.read_text()
                        # Ensure every labeled-data/roundN/ has a matching video_sets entry
                        for rd in sorted(labeled_data_dir.iterdir()) if labeled_data_dir.exists() else []:
                            if not rd.is_dir():
                                continue
                            rn = rd.name
                            # Skip if already in video_sets (forward or backslash)
                            if f"/{rn}.mp4:" in config_text or f"\\{rn}.mp4:" in config_text:
                                continue
                            # Clone the first video_sets entry with the new round name
                            m = re.search(
                                r'(  .+[\\/])(\w+)(\.mp4:\s*\n\s+crop:\s*[^\n]+)',
                                config_text,
                            )
                            if m:
                                new_entry = f"{m.group(1)}{rn}{m.group(3)}"
                                config_text = config_text.replace(
                                    m.group(0), m.group(0) + "\n" + new_entry, 1
                                )
                                logfile.write(f"  Added video_sets entry for {rn}\n")

                        import tempfile as _tmp
                        with _tmp.NamedTemporaryFile(
                            mode="w", suffix=".yaml", delete=False
                        ) as _tf:
                            _tf.write(config_text)
                            _tmp_path = _tf.name

                        upload_cmd = _scp_base_args(cfg) + [
                            _tmp_path,
                            f"{cfg.host}:{remote_project_dir}/config.yaml",
                        ]
                        proc = _run_remote_proc(upload_cmd, logfile, "Upload config")
                        os.unlink(_tmp_path)
                        if proc.returncode != 0:
                            _fail(f"Upload config failed (exit {proc.returncode})")
                            if on_complete:
                                on_complete(job_id, proc.returncode)
                            return
                else:
                    # Full upload for initial training
                    logfile.write(
                        f"=== Phase 1: Uploading {local_dlc_dir} to "
                        f"{cfg.host}:{remote_project_dir} ===\n"
                    )
                    logfile.flush()

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

                need_create = result.returncode != 0

                if result.returncode == 0:
                    shuffle = int(result.stdout.strip())
                    logfile.write(f"=== Detected existing pytorch config (shuffle {shuffle}) ===\n")
                    logfile.flush()
                    if force_create_dataset:
                        need_create = True

                if need_create:
                    label = "Recreating" if force_create_dataset else "Creating"
                    logfile.write(f"=== {label} training dataset on {cfg.host} ===\n")
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
            _epoch_re = re.compile(r"Epoch\s+(\d+)/(\d+)")

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
                    cur_epoch_val = None
                    max_epoch_val = None

                    # Supplement with epoch-level progress from log (more granular
                    # than status.json which only updates at phase boundaries)
                    if new_log and status.get("phase") == "train":
                        epoch_matches = _epoch_re.findall(new_log)
                        if epoch_matches:
                            cur_epoch_val, max_epoch_val = int(epoch_matches[-1][0]), int(epoch_matches[-1][1])
                            if max_epoch_val > 0:
                                # Training is 5-75% of remote progress
                                epoch_pct = 5.0 + (cur_epoch_val / max_epoch_val) * 70.0
                                pct = max(pct, epoch_pct)

                    # Scale remote 0-100% to local 8-90%
                    scaled = 8.0 + (pct / 100.0) * 82.0
                    _update_progress(scaled, epoch=cur_epoch_val, total_epochs=max_epoch_val)

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
    reverse: bool = False,
    use_bbox: bool = False,
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

    # Normalize the work_dir to forward slashes — Windows OpenSSH's SCP
    # rejects mixed-slash paths like ``C:\Users\dilla\...\foo/bar/baz``
    # with "No such file or directory" even when the file exists, because
    # the shell escapes the literal backslashes (``C:\\Users\\dilla...``)
    # and the remote scp then can't resolve them.  poll_remote_batch
    # already does this for HRnet jobs; we missed it here.
    _work_dir = cfg.work_dir.replace("\\", "/")
    remote_video_dir = f"{_work_dir}/videos"
    remote_output_dir = f"{_work_dir}/preprocess_output"
    remote_bbox_dir = f"{_work_dir}/bboxes"
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
            # ── Phase 0: Seed params_json.trials so the in-progress
            # trial-detail modal can render per-trial chips for this
            # MP batch (HRnet batches already do this; MP didn't).
            # Each entry: {subject_name, trial_idx, trial_name, outcome:
            # None}.  Outcomes flip to "ok" / "remote_only" / "failed"
            # in _try_running_download as per-subject downloads land.
            if do_mp and subjects:
                from .video import build_trial_map as _btm
                import json as _seed_json
                _seed_trials: list[dict] = []
                for _sub in subjects:
                    try:
                        _tmap = _btm(_sub)
                    except Exception:
                        _tmap = []
                    for _ti, _t in enumerate(_tmap):
                        _seed_trials.append({
                            "subject_name": _sub,
                            "trial_idx": _ti,
                            "trial_name": _t.get("trial_name", ""),
                            "uploaded": True,  # remote MP reads videos from
                                                # shared video dir; treat as
                                                # "ready" so chips render
                                                # blue (pending) instead of
                                                # grey (uploading) until the
                                                # download flips them green.
                        })
                if _seed_trials:
                    with get_db_ctx() as db:
                        _row = db.execute(
                            "SELECT params_json FROM jobs WHERE id=?", (job_id,)
                        ).fetchone()
                        try:
                            _ep = _seed_json.loads(_row["params_json"]) \
                                if _row and _row["params_json"] else {}
                        except (ValueError, TypeError):
                            _ep = {}
                        _ep["trials"] = _seed_trials
                        _ep["subjects"] = list(subjects)
                        db.execute(
                            "UPDATE jobs SET params_json=? WHERE id=?",
                            (_seed_json.dumps(_ep), job_id),
                        )

            # ── Phase 1: Upload videos ────────────────────────────
            logfile.write(f"=== Phase 1: Uploading videos to {cfg.host}:{remote_video_dir} ===\n")
            logfile.flush()
            _check_cancel()

            # Ensure remote dirs exist (videos + outputs + bbox JSON dir).
            subprocess.run(
                _py_cmd(cfg,
                    f"\"import os; "
                    f"os.makedirs(r'{remote_video_dir}', exist_ok=True); "
                    f"os.makedirs(r'{remote_output_dir}', exist_ok=True); "
                    f"os.makedirs(r'{remote_bbox_dir}', exist_ok=True)\""),
                capture_output=True, timeout=15,
            )

            # ── Phase 1a: build + upload per-subject bbox JSON ─────
            # Reads ``mp_crop_boxes`` (model_name='run-mediapipe') for
            # every subject in this run, packages into one JSON per
            # subject keyed by trial stem, and SCPs to
            # ``<work>/bboxes/<subject>_bboxes.json``.  The remote
            # script reads these when --use-bbox is set; absent files
            # transparently fall through to the no-crop path.
            if do_mp and use_bbox and subjects:
                import json as _bbox_json
                import tempfile as _bbox_tmp
                from .video import build_trial_map
                cam_names = settings.camera_names or ["OS", "OD"]
                cam_OS = cam_names[0]
                cam_OD = cam_names[1] if len(cam_names) > 1 else cam_names[0]
                uploaded = 0
                with get_db_ctx() as _db:
                    for sub in subjects:
                        try:
                            tmap = build_trial_map(sub)
                        except Exception:
                            tmap = []
                        if not tmap:
                            continue
                        srow = _db.execute(
                            "SELECT id FROM subjects WHERE name = ?", (sub,)
                        ).fetchone()
                        if not srow:
                            continue
                        rows = _db.execute(
                            "SELECT trial_idx, camera_name, x1, y1, x2, y2 "
                            "FROM mp_crop_boxes "
                            "WHERE subject_id = ? AND model_name = 'run-mediapipe'",
                            (srow["id"],),
                        ).fetchall()
                        if not rows:
                            continue
                        by_trial: dict = {}
                        for r in rows:
                            stem = tmap[r["trial_idx"]]["trial_name"] \
                                if 0 <= r["trial_idx"] < len(tmap) else None
                            if not stem:
                                continue
                            d = by_trial.setdefault(stem, {})
                            box = [r["x1"], r["y1"], r["x2"], r["y2"]]
                            if r["camera_name"] == cam_OS:
                                d["OS"] = box
                            elif r["camera_name"] == cam_OD:
                                d["OD"] = box
                        if not by_trial:
                            continue
                        with _bbox_tmp.NamedTemporaryFile(
                                mode="w", suffix=".json", delete=False) as fh:
                            _bbox_json.dump(by_trial, fh)
                            tmp_path = fh.name
                        proc = subprocess.run(
                            _scp_base_args(cfg) + [
                                tmp_path,
                                f"{cfg.host}:{remote_bbox_dir}/{sub}_bboxes.json",
                            ],
                            capture_output=True, text=True, timeout=30,
                        )
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass
                        if proc.returncode == 0:
                            uploaded += 1
                logfile.write(f"  Uploaded bbox JSON for {uploaded}/{len(subjects)} subject(s)\n")
                logfile.flush()

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

            # No-face overrides only matter for the face-blur (deidentify)
            # path; MediaPipe / pose / vision processing ignores them.  Skip
            # collecting + logging them when the current job isn't doing
            # blur — otherwise users see misleading "Override: X skip=true"
            # lines on a pure-MP job.
            blur_in_steps = any(
                s in steps for s in ("blur", "deidentify", "mediapipe+blur")
            )
            if blur_in_steps:
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
                if reverse:
                    launch_args_parts.append("'--reverse'")
                if use_bbox:
                    launch_args_parts.append(
                        f"'--bbox-dir', r'{remote_bbox_dir}'")
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
                # Per-subject "MP done" markers.  Two patterns supported:
                #   * NEW (per-trial output):  "MP_SUBJECT_DONE <subj>"
                #     emitted once when every trial for the subject has
                #     been written.
                #   * LEGACY (combined npz):   "Saved <path>/<subj>/
                #     mediapipe_prelabels.npz: ..." (older remote
                #     installs that still write a combined file).
                # Either pattern triggers the same per-subject SCP pull.
                import re as _re
                _mp_done_re = _re.compile(
                    r"(?:MP_SUBJECT_DONE\s+(\S+))"
                    r"|(?:Saved\s+\S*?/(\S+?)/mediapipe_prelabels\.npz:)"
                )
                downloaded_subjects: set[str] = set()

                _dl_lock = threading.Lock()
                _params_write_lock = threading.Lock()

                def _mark_mp_subject_done(subj_name: str):
                    """Mark every params_json.trials entry for ``subj_name``
                    with ``outcome='ok'`` so the trial-detail modal turns
                    its chips green.  Re-reads params_json fresh under a
                    lock to avoid clobbering concurrent writes."""
                    import json as _pj_json
                    with _params_write_lock:
                        try:
                            with get_db_ctx() as _db:
                                _r = _db.execute(
                                    "SELECT params_json FROM jobs WHERE id=?",
                                    (job_id,),
                                ).fetchone()
                                try:
                                    _pj = _pj_json.loads(_r["params_json"]) \
                                        if _r and _r["params_json"] else {}
                                except (ValueError, TypeError):
                                    _pj = {}
                                _tlist = _pj.get("trials") or []
                                _changed = False
                                for _t in _tlist:
                                    if _t.get("subject_name") == subj_name \
                                            and _t.get("outcome") != "ok":
                                        _t["outcome"] = "ok"
                                        _changed = True
                                if _changed:
                                    _pj["trials"] = _tlist
                                    _db.execute(
                                        "UPDATE jobs SET params_json=? WHERE id=?",
                                        (_pj_json.dumps(_pj), job_id),
                                    )
                        except Exception as _e:
                            try:
                                logfile.write(
                                    f"  Warning: failed to mark {subj_name} "
                                    f"outcome=ok in params_json: {_e}\n")
                                logfile.flush()
                            except Exception:
                                pass

                def _try_running_download(subj_name: str):
                    """Pull this subject's MP npz now that the remote has
                    finished it.  Runs in a daemon thread so the SCP +
                    stable-size probe don't block the monitor loop (which
                    would otherwise stop tailing the remote log while a
                    download is in flight)."""
                    with _dl_lock:
                        if subj_name in downloaded_subjects:
                            return
                        downloaded_subjects.add(subj_name)   # claim it now
                    def _worker():
                        try:
                            from .remote_results import (
                                output_specs as _outputs,
                                download_one as _dl_one,
                            )
                            with get_db_ctx() as _db:
                                srow = _db.execute(
                                    "SELECT id FROM subjects WHERE name = ?",
                                    (subj_name,),
                                ).fetchone()
                            if not srow:
                                return
                            any_ok = False
                            for spec in _outputs("mediapipe", subj_name, remote_output_dir):
                                if _dl_one(spec, cfg, srow["id"], job_id):
                                    any_ok = True
                            if any_ok:
                                logfile.write(
                                    f"  Downloaded (running): {subj_name}/mediapipe_prelabels.npz\n")
                                logfile.flush()
                                # Flip outcome=ok on this subject's trial
                                # entries in params_json so the in-progress
                                # trial-detail modal turns the chips green
                                # the moment files land locally.  Mirrors
                                # the HRnet poller's per-trial outcome
                                # writes -- the chip renderer reads outcome
                                # from params_json.trials[i].outcome.
                                _mark_mp_subject_done(subj_name)
                            else:
                                # File wasn't ready/stable — let the next
                                # detection (or Phase 5) try again.
                                with _dl_lock:
                                    downloaded_subjects.discard(subj_name)
                        except Exception as _e:
                            logfile.write(f"  Warning: running-basis download for "
                                          f"{subj_name} failed: {_e}\n")
                            logfile.flush()
                            with _dl_lock:
                                downloaded_subjects.discard(subj_name)
                    threading.Thread(target=_worker, daemon=True,
                                     name=f"running-dl-{subj_name}").start()

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
                        # Detect per-subject MP completion lines and pull
                        # those results immediately, instead of waiting for
                        # the batch's Phase-5 download at the end.
                        for m in _mp_done_re.finditer(new_log):
                            # Group 1 = new MP_SUBJECT_DONE pattern;
                            # group 2 = legacy combined-npz pattern.
                            # Either present means the subject's MP
                            # outputs are stable on disk.
                            subj_done = (m.group(1) or m.group(2) or "").strip()
                            if subj_done:
                                _try_running_download(subj_done)

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

            # Only iterate subjects we actually processed in THIS job —
            # not whatever leftover dirs happen to live under
            # preprocess_output/ from prior runs.  Otherwise we'd attempt
            # (and fail) SCPs for every old subject left on disk.
            mp_set = set(mp_subjects) if do_mp else set()
            blur_set = set(blur_subjects) if do_blur else set()
            requested_subjects = sorted(mp_set | blur_set)

            logfile.write(
                f"  Downloading results for {len(requested_subjects)} subject(s): "
                f"{', '.join(requested_subjects)}\n"
            )
            logfile.flush()

            for i, subj_name in enumerate(requested_subjects):
                _check_cancel()

                # Download MediaPipe outputs (only if this subject was
                # actually in the mp run — not for blur-only subjects).
                # Uses output_specs so the per-trial layout is honored;
                # legacy combined falls back automatically and is split
                # into per-trial files post-download.
                if do_mp and subj_name in mp_set:
                    from .remote_results import (
                        output_specs as _outputs,
                        download_one as _dl_one,
                    )
                    with get_db_ctx() as _db:
                        srow = _db.execute(
                            "SELECT id FROM subjects WHERE name = ?",
                            (subj_name,),
                        ).fetchone()
                    if srow:
                        any_ok = False
                        for spec in _outputs("mediapipe", subj_name, remote_output_dir):
                            if _dl_one(spec, cfg, srow["id"], job_id):
                                any_ok = True
                        if any_ok:
                            logfile.write(f"  Downloaded MediaPipe outputs for {subj_name}\n")
                        else:
                            logfile.write(f"  Warning: npz download failed for {subj_name}\n")
                    else:
                        logfile.write(f"  Warning: subject {subj_name} not in DB\n")

                # Download deidentified videos (only for subjects in the
                # blur run — not for mp-only subjects).
                if do_blur and subj_name in blur_set:
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
                pct = download_start + (i + 1) / max(len(requested_subjects), 1) * (100 - download_start)
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
                sess = row.get("tmux_session") if row else None
                if sess and sess.startswith("pid:"):
                    pid = int(sess.split(":")[1])
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


def remote_hrnet_job(
    job_id: int,
    cfg: RemoteConfig,
    subject_name: str,
    extra_params: dict,
    log_path: str,
    registry,
):
    """Run HRNet inference on a remote GPU server.

    Phases: upload video + MP npz + script → launch → monitor → download results.
    Runs in the calling thread (queue_manager already runs in a worker thread).
    """
    import time

    from ..config import get_settings
    from ..db import get_db_ctx

    settings = get_settings()
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    trial_idx = extra_params.get("trial_idx", 0)
    from .video import build_trial_map
    trials = build_trial_map(subject_name)
    if trial_idx >= len(trials):
        with get_db_ctx() as db:
            db.execute("UPDATE jobs SET status='failed', error_msg='Trial not found', finished_at=CURRENT_TIMESTAMP WHERE id=?", (job_id,))
        return {"remote_done": False, "downloaded": False, "error": "Trial not found"}
    trial = trials[trial_idx]
    trial_name = trial["trial_name"]
    video_path = trial["video_path"]
    n_frames = trial["frame_count"]
    start_frame = trial.get("start_frame", 0)
    camera_mode = "stereo"  # default
    with get_db_ctx() as db:
        row = db.execute("SELECT camera_mode FROM subjects WHERE name = ?", (subject_name,)).fetchone()
    if row and row["camera_mode"]:
        camera_mode = row["camera_mode"]

    # Find / build a combined MediaPipe npz to upload to the remote
    # HRnet runner.  The remote script consumes one ``mediapipe_prelabels
    # .npz`` per subject; locally we now store per-trial files, so we
    # aggregate them into a temp npz on the fly.  Falls back to the
    # legacy combined file if it happens to still exist.
    from .mediapipe_prelabel import (
        build_combined_mp_npz_tempfile, has_mediapipe_data,
    )
    if not has_mediapipe_data(subject_name):
        with get_db_ctx() as db:
            db.execute("UPDATE jobs SET status='failed', error_msg='MediaPipe data not found (run MediaPipe first)', finished_at=CURRENT_TIMESTAMP WHERE id=?", (job_id,))
        return {"remote_done": False, "downloaded": False, "error": "MediaPipe data not found"}
    legacy_combined = settings.dlc_path / subject_name / "mediapipe_prelabels.npz"
    if legacy_combined.exists():
        mp_npz_local = legacy_combined
        _mp_tempfile_to_unlink: str | None = None
    else:
        _mp_tempfile_to_unlink = build_combined_mp_npz_tempfile(subject_name)
        if not _mp_tempfile_to_unlink:
            with get_db_ctx() as db:
                db.execute("UPDATE jobs SET status='failed', error_msg='Could not assemble combined MediaPipe npz for HRnet upload', finished_at=CURRENT_TIMESTAMP WHERE id=?", (job_id,))
            return {"remote_done": False, "downloaded": False, "error": "MP aggregate failed"}
        mp_npz_local = Path(_mp_tempfile_to_unlink)

    remote_work = f"{cfg.work_dir}/hrnet_jobs/{subject_name}_{trial_name}"
    remote_video = f"{remote_work}/video.mp4"
    remote_mp_npz = f"{remote_work}/mediapipe_prelabels.npz"
    remote_script = f"{remote_work}/remote_hrnet_script.py"
    remote_output = f"{remote_work}/output"
    remote_status = f"{remote_work}/status.json"
    remote_log = f"{remote_work}/hrnet.log"

    # Batch-mode progress scaling: when this trial is one of several in a
    # parent job, the queue_manager passes ``_batch_index`` and
    # ``_batch_total`` so we can map our local 0-100 progress into the
    # parent's slice.  Otherwise behave as if we own the whole bar.
    _batch_idx = int(extra_params.get("_batch_index", 0))
    _batch_total = max(1, int(extra_params.get("_batch_total", 1)))
    _is_batched = _batch_total > 1

    def _update_progress(pct):
        global_pct = 100.0 * (_batch_idx + (pct or 0) / 100.0) / _batch_total
        with get_db_ctx() as db:
            db.execute("UPDATE jobs SET progress_pct=? WHERE id=?",
                       (round(global_pct, 1), job_id))

    def _fail(msg):
        logger.error(f"Job {job_id} HRNet remote failed: {msg}")
        if _is_batched:
            # Don't mark the parent job failed — the queue_manager loop
            # owns the final status; per-trial failures are logged but
            # let other trials run.
            try:
                with open(log_path, "a") as _lf:
                    _lf.write(f"  HRnet trial failed: {msg}\n")
            except OSError:
                pass
            return
        with get_db_ctx() as db:
            db.execute("UPDATE jobs SET status='failed', error_msg=?, "
                       "finished_at=CURRENT_TIMESTAMP WHERE id=?",
                       (msg[:500], job_id))

    # Outcome tracking — caller (queue_manager batch loop) reads this to
    # color trial chips and compute the final batch status badge.
    # ``remote_done`` flips when the inference phase reports completion;
    # ``downloaded`` flips after the heatmaps SCP succeeds.
    _outcome = {"remote_done": False, "downloaded": False, "error": None}

    # In batch mode the queue_manager has already opened the log and
    # written a "[i/N] subj trial" header.  Append rather than truncate
    # so per-trial output accumulates in one log.
    logfile = open(log_path, "a" if _is_batched else "w")
    try:
        # Phase 0: Verify remote is reachable
        logfile.write(f"Checking remote connection to {cfg.host}...\n")
        logfile.flush()
        try:
            probe = subprocess.run(
                _ssh_base_args(cfg) + [cfg.host, "echo", "ok"],
                capture_output=True, text=True, timeout=15,
            )
            if probe.returncode != 0 or "ok" not in probe.stdout:
                _fail(f"Remote machine unreachable (may be asleep or offline): {probe.stderr[:200]}")
                return _outcome
        except subprocess.TimeoutExpired:
            _fail("Remote machine unreachable — connection timed out (may be asleep or offline)")
            return _outcome
        logfile.write("  Connected.\n")
        logfile.flush()

        # Phase 1: Create remote dir + upload files
        _update_progress(2)
        logfile.write(f"Creating remote directory: {remote_work}\n")
        logfile.flush()
        try:
            subprocess.run(
                _py_cmd(cfg, f"\"import os; os.makedirs(r'{remote_work}', exist_ok=True); os.makedirs(r'{remote_output}', exist_ok=True)\""),
                capture_output=True, timeout=30,
            )
        except subprocess.TimeoutExpired:
            logfile.write("  mkdir timed out (dirs may already exist, continuing)\n")
            logfile.flush()

        # Upload video — idempotent so partial-batch resume + restart loops
        # don't re-transfer the same hundreds of megabytes per trial.
        logfile.write(f"Uploading video: {video_path}\n")
        logfile.flush()
        ok, _ = _scp_if_changed(cfg, video_path, remote_video,
                                  timeout=600, logfile=logfile, label="video: ")
        if not ok:
            _fail("Failed to upload video")
            return _outcome
        _update_progress(10)

        # Upload mediapipe npz
        logfile.write(f"Uploading MediaPipe data\n")
        logfile.flush()
        ok, _ = _scp_if_changed(cfg, str(mp_npz_local), remote_mp_npz,
                                  timeout=120, logfile=logfile, label="mp: ")
        if not ok:
            _fail("Failed to upload MediaPipe data")
            return _outcome
        _update_progress(15)

        # HRNet script — always re-uploaded (may have changed between
        # batches; size compare alone isn't enough to detect content edits
        # of the same length).  Tiny file so cost is negligible.
        script_local = Path(__file__).parent / "remote_hrnet_script.py"
        proc = subprocess.run(
            _scp_base_args(cfg) + [str(script_local), f"{cfg.host}:{remote_script}"],
            capture_output=True, text=True, timeout=30,
        )
        if proc.returncode != 0:
            _fail(f"Failed to upload HRNet script: {proc.stderr[-200:]}")
            return _outcome
        _update_progress(18)

        # Phase 2: Write a launcher script and upload it, then execute
        bbox_os_arg = ",".join(str(x) for x in extra_params["bbox_os"]) if extra_params.get("bbox_os") else ""
        bbox_od_arg = ",".join(str(x) for x in extra_params["bbox_od"]) if extra_params.get("bbox_od") else ""

        # Build the launcher as a Python file (avoids SSH command-line length limits)
        launcher_lines = [
            "import subprocess, os, sys, time",
            f"args = [r'{cfg.python_executable}', '-u', r'{remote_script}',",
            f"    r'{remote_video}', r'{remote_output}', r'{remote_mp_npz}',",
            f"    '--trial-name', '{trial_name}',",
            f"    '--trial-idx', '{trial_idx}',",
            f"    '--start-frame', '{start_frame}',",
            f"    '--n-frames', '{n_frames}',",
            f"    '--camera-mode', '{camera_mode}',",
            f"    '--status-file', r'{remote_status}',",
        ]
        if bbox_os_arg:
            launcher_lines.append(f"    '--bbox-os', '{bbox_os_arg}',")
        if bbox_od_arg:
            launcher_lines.append(f"    '--bbox-od', '{bbox_od_arg}',")
        launcher_lines += [
            "]",
            "# Redirect child stdout/stderr to log file.",
            "# On Windows, the child inherits the file handle and keeps it open",
            "# even after this launcher exits — the OS reference-counts handles.",
            f"log_fh = open(r'{remote_log}', 'w', buffering=1)",  # line-buffered
            # Detach child from this SSH session on Windows.  Without these flags,
            # closing the launcher's SSH connection sends a CTRL_CLOSE_EVENT down the
            # process tree and kills the still-importing child before it produces any
            # output (symptom: hrnet.log stops at 'Phase: ensure dependencies').
            "creationflags = 0",
            "if sys.platform == 'win32':",
            "    creationflags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_BREAKAWAY_FROM_JOB",
            # close_fds=True: drop the launcher's own stdin/stdout/stderr handles
            # (which are pipes back to the SSH client) so that when the detached child
            # inherits the file table, it does NOT keep the SSH stdio open.  Otherwise
            # ssh waits indefinitely for those handles to close and the launch ssh
            # call times out even though the child is happily running.  log_fh is
            # explicitly passed as stdout/stderr, so it still propagates regardless.
            "p = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=log_fh, stderr=log_fh,"
            "                     close_fds=True, creationflags=creationflags)",
            "time.sleep(3)",
            "log_fh.flush()",
            "print(p.pid)",
        ]
        import tempfile
        launcher_local = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
        launcher_local.write("\n".join(launcher_lines))
        launcher_local.close()
        remote_launcher = f"{remote_work}/launcher.py"

        # Upload launcher
        proc = subprocess.run(
            _scp_base_args(cfg) + [launcher_local.name, f"{cfg.host}:{remote_launcher}"],
            capture_output=True, text=True, timeout=30,
        )
        os.unlink(launcher_local.name)
        if proc.returncode != 0:
            _fail(f"Failed to upload launcher: {proc.stderr[-200:]}")
            return _outcome

        logfile.write(f"Launching remote HRNet process\n")
        logfile.flush()
        result = subprocess.run(
            _ssh_base_args(cfg) + [cfg.host, cfg.python_executable, "-u", remote_launcher],
            capture_output=True, text=True, timeout=30,
        )
        logfile.write(f"Launch stdout: {result.stdout[:500]}\n")
        logfile.write(f"Launch stderr: {result.stderr[:500]}\n")
        logfile.write(f"Launch returncode: {result.returncode}\n")
        logfile.flush()
        if result.returncode != 0:
            _fail(f"Failed to launch: {result.stderr[-300:] or result.stdout[-300:]}")
            return _outcome

        remote_pid = None
        try:
            remote_pid = int(result.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            pass
        logfile.write(f"Remote PID: {remote_pid}\n")
        logfile.flush()

        # Store remote info in DB so restart can find it
        with get_db_ctx() as db:
            db.execute(
                "UPDATE jobs SET tmux_session=?, remote_host=?, status='running' WHERE id=?",
                (f"pid:{remote_pid}" if remote_pid else "", cfg.host, job_id),
            )
        _update_progress(20)

        # Phase 3: Monitor via status.json polling
        poll_interval = 10
        consecutive_dead = 0  # require multiple dead checks before giving up
        log_offset = 0
        logfile.write("Entering monitor loop...\n")
        logfile.flush()
        while True:
            time.sleep(poll_interval)

            # Check cancel
            cancel_ev = registry._cancel_events.get(job_id)
            if cancel_ev and cancel_ev.is_set():
                _fail("Cancelled")
                return _outcome

            # Read remote status
            status = _read_remote_status(cfg, remote_status)

            # Tail remote log
            new_log, log_offset = _tail_remote_log(cfg, remote_log, log_offset)
            if new_log:
                logfile.write(new_log)
                logfile.flush()

            # If we got new log output, the process is alive — reset dead counter
            if new_log:
                consecutive_dead = 0

            if status:
                pct = status.get("progress_pct", 0)
                scaled = 20 + (pct / 100.0) * 65  # 20-85%
                _update_progress(scaled)
                consecutive_dead = 0  # status.json exists → alive

                if status.get("status") == "completed":
                    logfile.write("Remote HRNet completed\n")
                    _outcome["remote_done"] = True
                    break
                elif status.get("status") == "failed":
                    _outcome["error"] = status.get('error', 'unknown')
                    _fail(f"Remote HRNet failed: {_outcome['error']}")
                    return _outcome
            elif not new_log:
                # No status file AND no new log output — check if process died
                consecutive_dead += 1
                logfile.write(f"[poll] no status, no new log (dead_count={consecutive_dead})\n")
                logfile.flush()
                if consecutive_dead >= 60:  # 10 minutes of silence (pip installs can be slow)
                    # Confirm process is actually dead before failing
                    if remote_pid and not _check_remote_pid_alive(cfg, remote_pid):
                        full_log, _ = _tail_remote_log(cfg, remote_log, 0)
                        if full_log:
                            logfile.write(full_log)
                            logfile.flush()
                            lines = [l.strip() for l in full_log.strip().splitlines() if l.strip()]
                            tail = lines[-1] if lines else "no output"
                        else:
                            tail = "no remote log after 2 min silence"
                        _outcome["error"] = f"process died: {tail}"
                        _fail(f"Remote process died: {tail}")
                        return _outcome

        _update_progress(87)

        # Phase 4: Download results
        logfile.write("Downloading results\n")
        logfile.flush()
        from .skeleton_data import _skeleton_dir
        local_out_dir = _skeleton_dir(subject_name) / trial_name
        local_out_dir.mkdir(parents=True, exist_ok=True)

        # Normalize backslashes → forward slashes for SCP source URIs.
        # OpenSSH on Windows occasionally fails to resolve mixed-slash
        # paths after the "host:" prefix even when the file exists.
        _trial_dir_remote = f"{remote_output}/{trial_name}".replace("\\", "/")

        # Download heatmaps npz
        remote_npz = f"{cfg.host}:{_trial_dir_remote}/hrnet_w18_heatmaps.npz"
        proc = subprocess.run(
            _scp_base_args(cfg) + [remote_npz, str(local_out_dir / "hrnet_w18_heatmaps.npz")],
            capture_output=True, text=True, timeout=300,
        )
        if proc.returncode != 0:
            _outcome["error"] = f"download failed: {proc.stderr[-200:]}"
            _fail(f"Failed to download heatmaps: {proc.stderr[-200:]}")
            return _outcome
        _outcome["downloaded"] = True

        # Download hand_crop.json
        remote_crop = f"{cfg.host}:{_trial_dir_remote}/hand_crop.json"
        subprocess.run(
            _scp_base_args(cfg) + [remote_crop, str(local_out_dir / "hand_crop.json")],
            capture_output=True, text=True, timeout=30,
        )

        # Download pre-computed MIP npz (optional — present on newer runs)
        remote_mip = f"{cfg.host}:{_trial_dir_remote}/hrnet_w18_mip.npz"
        subprocess.run(
            _scp_base_args(cfg) + [remote_mip, str(local_out_dir / "hrnet_w18_mip.npz")],
            capture_output=True, text=True, timeout=60,
        )

        _update_progress(100)
        if not _is_batched:
            with get_db_ctx() as db:
                db.execute(
                    "UPDATE jobs SET status='completed', progress_pct=100, finished_at=CURRENT_TIMESTAMP WHERE id=?",
                    (job_id,),
                )
        # In batch mode the queue_manager owns the final completed/failed
        # status; this trial just logs success and returns.
        logfile.write("HRNet remote job completed successfully\n")
        return _outcome

    except Exception as e:
        logger.exception(f"Job {job_id} remote HRNet error")
        _outcome["error"] = str(e)
        _fail(str(e))
        return _outcome
    finally:
        logfile.close()
        if _mp_tempfile_to_unlink:
            try:
                os.unlink(_mp_tempfile_to_unlink)
            except OSError:
                pass
        registry._processes.pop(job_id, None)
        registry._threads.pop(job_id, None)
        registry.unregister_cancel_event(job_id)


def remote_hrnet_redownload(
    job_id: int,
    cfg: RemoteConfig,
    subject_names: list[str],
    log_path: str,
    registry,
    parent_job_id: int | None = None,
) -> None:
    """Re-download HRnet outputs without re-running.

    When ``parent_job_id`` is given, the helper targets only the trials in
    that batch's ``params_json.trials`` list:
      * If any trial has ``outcome="remote_only"`` → re-download those only,
        and flip their outcome to ``"ok"`` on success.
      * Otherwise → re-download every ``outcome="ok"`` trial (force re-pull).
      * If no outcomes are present (legacy job) → re-download every trial in
        the list.
    The parent's ``params_json`` and overall status badge are updated to
    reflect the new outcome distribution.

    When ``parent_job_id`` is None, the legacy "scan remote dirs" behavior
    is used (mirrors every ``hrnet_jobs/{subject}_*`` directory locally).
    """
    from .skeleton_data import _skeleton_dir
    import json as _json

    logfile = open(log_path, "a", buffering=1)

    def _fail(msg: str):
        logfile.write(f"ERROR: {msg}\n")
        logfile.flush()
        with get_db_ctx() as db:
            db.execute(
                "UPDATE jobs SET status='failed', error_msg=?, finished_at=CURRENT_TIMESTAMP WHERE id=?",
                (msg[:500], job_id),
            )

    # Build the per-trial work list.  Each entry: dict(subject, trial_name,
    # short_trial, idx_in_parent_trials).  For the parent-aware path we also
    # remember which entries to flip to "ok" on success.
    work = []
    parent_trials = None
    update_outcomes = False
    if parent_job_id is not None:
        with get_db_ctx() as db:
            row = db.execute(
                "SELECT params_json FROM jobs WHERE id = ?", (parent_job_id,)
            ).fetchone()
        try:
            _p = _json.loads(row["params_json"]) if row and row["params_json"] else {}
            parent_trials = _p.get("trials") if isinstance(_p, dict) else None
        except (ValueError, TypeError):
            parent_trials = None
    # Each work item resolves to:
    #   remote_outer_dir  = "<sub>_<full_stem>"   (e.g. Con04_Con04_L1)
    #   remote_inner_dir  = "<full_stem>"          (e.g. Con04_L1)
    #   local_dir_name    = "<full_stem>"          (skeleton/<sub>/<full_stem>/)
    if parent_trials and isinstance(parent_trials, list):
        update_outcomes = True
        any_outcomes = any(t.get("outcome") for t in parent_trials)
        any_remote_only = any(t.get("outcome") == "remote_only" for t in parent_trials)
        for ti, t in enumerate(parent_trials):
            sub = t.get("subject_name")
            tn = t.get("trial_name")
            if not sub or not tn:
                continue
            if any_outcomes:
                if any_remote_only:
                    if t.get("outcome") != "remote_only":
                        continue
                else:
                    if t.get("outcome") != "ok":
                        continue
            # Frontend stores the short trial code ("L1"); build_trial_map
            # produces the full stem ("Con04_L1").  Normalize to full_stem.
            full_stem = tn if "_" in tn else f"{sub}_{tn}"
            work.append(dict(
                sub=sub,
                outer_dir=f"{sub}_{full_stem}",  # Con04_Con04_L1
                inner_dir=full_stem,             # Con04_L1
                short=full_stem.split("_", 1)[1] if "_" in full_stem else full_stem,
                parent_idx=ti,
            ))

    try:
        logfile.write(f"=== HRnet re-download: {len(subject_names)} subject(s) ===\n")
        logfile.flush()
        work_dir_norm = cfg.work_dir.replace("\\", "/")
        hrnet_root = f"{work_dir_norm}/hrnet_jobs"

        if not work:
            # Legacy path: scan every remote dir for the listed subjects.
            list_cmd = _py_cmd(
                cfg,
                f"\"import os; root = r'{cfg.work_dir}/hrnet_jobs'; print('\\n'.join(os.listdir(root)) if os.path.isdir(root) else '')\"",
            )
            proc = subprocess.run(list_cmd, capture_output=True, text=True, timeout=30)
            if proc.returncode != 0:
                _fail(f"Failed to list remote hrnet_jobs: {proc.stderr[-200:]}")
                return
            all_dirs = [d.strip() for d in proc.stdout.splitlines() if d.strip()]
            for subject_name in subject_names:
                for dname in [d for d in all_dirs if d.startswith(f"{subject_name}_")]:
                    # dname is the OUTER dir name "<sub>_<full_stem>";
                    # the inner dir is "<full_stem>" (the trial stem).
                    inner = dname[len(subject_name) + 1:]   # Con04_L1
                    work.append(dict(
                        sub=subject_name,
                        outer_dir=dname,
                        inner_dir=inner,
                        short=inner.split("_", 1)[1] if "_" in inner else inner,
                        parent_idx=None,
                    ))

        if not work:
            logfile.write("Nothing to re-download.\n")
            with get_db_ctx() as db:
                db.execute(
                    "UPDATE jobs SET status='completed', progress_pct=100, "
                    "finished_at=CURRENT_TIMESTAMP WHERE id=?",
                    (job_id,),
                )
            return

        logfile.write(f"  Will pull {len(work)} trial dir(s).\n")
        n_files_total = 0
        n_files_ok = 0
        success_indices = []
        for i, w in enumerate(work):
            sub = w["sub"]
            outer = w["outer_dir"]   # Con04_Con04_L1
            inner = w["inner_dir"]   # Con04_L1
            short = w["short"]       # L1 (display only)
            # Remote layout (mirrors remote_hrnet_job's save path):
            #   hrnet_jobs/<outer>/output/<inner>/<file>
            remote_dir = f"{hrnet_root}/{outer}/output/{inner}".replace("\\", "/")
            local_trial_dir = _skeleton_dir(sub) / inner
            local_trial_dir.mkdir(parents=True, exist_ok=True)
            logfile.write(f"  [{i+1}/{len(work)}] {sub} {short}\n")

            files = [
                ("hrnet_w18_heatmaps.npz", 300, True),  # required
                ("hand_crop.json",          30, False),
                ("hrnet_w18_mip.npz",       60, False),
            ]
            heatmaps_ok = False
            for fname, timeout_s, required in files:
                src = f"{cfg.host}:{remote_dir}/{fname}"
                dst = str(local_trial_dir / fname)
                n_files_total += 1
                p = subprocess.run(
                    _scp_base_args(cfg) + [src, dst],
                    capture_output=True, text=True, timeout=timeout_s,
                )
                if p.returncode == 0:
                    n_files_ok += 1
                    logfile.write(f"    ok: {fname}\n")
                    if fname == "hrnet_w18_heatmaps.npz":
                        heatmaps_ok = True
                else:
                    if required:
                        logfile.write(f"    FAIL: {fname}: {p.stderr[-160:]}\n")
                    else:
                        logfile.write(f"    skip: {fname} (not present)\n")
                logfile.flush()
            if heatmaps_ok and w["parent_idx"] is not None:
                success_indices.append(w["parent_idx"])

            pct = 100.0 * (i + 1) / max(1, len(work))
            with get_db_ctx() as db:
                db.execute(
                    "UPDATE jobs SET progress_pct = ? WHERE id = ?",
                    (round(pct, 1), job_id),
                )

        # Update parent batch's per-trial outcomes + final status.
        if update_outcomes and parent_trials is not None and success_indices:
            for idx in success_indices:
                if 0 <= idx < len(parent_trials):
                    parent_trials[idx]["outcome"] = "ok"
            with get_db_ctx() as db:
                row = db.execute(
                    "SELECT params_json FROM jobs WHERE id = ?", (parent_job_id,)
                ).fetchone()
                _p = _json.loads(row["params_json"]) if row and row["params_json"] else {}
                _p["trials"] = parent_trials
                # Recompute final status badge.
                n_ok = sum(1 for t in parent_trials if t.get("outcome") == "ok")
                n_rem = sum(1 for t in parent_trials if t.get("outcome") == "remote_only")
                n_fail = sum(1 for t in parent_trials if t.get("outcome") == "failed")
                N = len(parent_trials)
                if n_fail > 0:
                    parent_status = "failed"
                    parent_err = f"Ran {n_ok + n_rem}/{N} trials"
                elif n_rem > 0:
                    parent_status = "completed"
                    parent_err = f"Download incomplete: {n_ok}/{N} downloaded"
                else:
                    parent_status = "completed"
                    parent_err = None
                db.execute(
                    "UPDATE jobs SET params_json=?, status=?, error_msg=? WHERE id=?",
                    (_json.dumps(_p), parent_status, parent_err, parent_job_id),
                )

        logfile.write(f"=== Re-download done: {n_files_ok}/{n_files_total} files ===\n")
        with get_db_ctx() as db:
            db.execute(
                "UPDATE jobs SET status='completed', progress_pct=100, "
                "finished_at=CURRENT_TIMESTAMP WHERE id=?",
                (job_id,),
            )

    except Exception as e:
        logger.exception(f"Job {job_id} HRnet re-download error")
        _fail(str(e))
    finally:
        logfile.close()
        registry._threads.pop(job_id, None)


# ──────────────────────────────────────────────────────────────────────
# Remote batch runner — survives local app restarts.
# ──────────────────────────────────────────────────────────────────────
#
# A single long-lived process on the remote owns the trial loop, writing
# per-trial done.flag markers and a batch_status.json heartbeat.  The
# local poller below reads those files; the remote runner doesn't depend
# on the local poller in any way, so closing the local app doesn't
# interrupt remote work.
#
# Layout under ``<work_dir>/batches/<batch_id>/``::
#
#     batch.json
#     videos/<stem>.mp4
#     mp/<subject>.npz
#     remote_hrnet_script.py
#     remote_batch_runner.py
#     output/<stem>/
#         hrnet_w18_heatmaps.npz, hand_crop.json, hrnet_w18_mip.npz, done.flag
#     batch_status.json, runner.log
#
# Local DB stores ``params_json["_batch_id"]`` so recovery + poller can
# reattach to the right remote dir after a restart.

_BATCH_HRNET_FILES = [
    ("hrnet_w18_heatmaps.npz", 300, True),
    ("hand_crop.json",          30, False),
    ("hrnet_w18_mip.npz",       60, False),
]

# Serializes read-modify-write of params_json across the dispatch upload-tail
# thread and the poller thread.  Both update trials[i].uploaded / .outcome,
# and without a lock the two threads can clobber each other's edits — most
# visibly the "chip went green then back to blue" flicker when an outcome
# write was overwritten by a stale upload-tail snapshot.  Acquire around the
# whole read-modify-write block — duration is dominated by SCPs the poller
# does between read and write, but contention is rare in practice.
_params_lock = threading.Lock()


def dispatch_remote_batch(
    job_id: int,
    cfg: RemoteConfig,
    subject_name: str,            # used for the foreign key only
    extra_params: dict,
    log_path: str,
    registry,
) -> dict:
    """Upload only the missing per-trial files + a tiny batch state dir,
    then launch the long-lived runner detached on the remote.

    Reuses each trial's existing ``hrnet_jobs/<sub>_<stem>/`` dir for
    video, MP npz, inference script, and outputs (no batch-wide video
    duplication).  The batch state — ``batch.json``, ``batch_status.json``,
    ``runner.log`` — lives under ``batch_runs/<batch_id>/`` so a single
    poller has one place to read.

    Idempotent: every SCP skips when the remote already has a same-size
    copy, so re-submitting the same batch (or running another batch over
    the same trials) is essentially free.

    Returns ``{"batch_id", "remote_pid", "state_dir"}`` on success.
    """
    import json as _json
    import tempfile
    import time as _time
    import uuid as _uuid

    from ..config import get_settings
    from ..db import get_db_ctx
    from .video import build_trial_map

    settings = get_settings()
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logfile = open(log_path, "a", buffering=1)
    cancel_event = registry.register_cancel_event(job_id)

    def _fail(msg: str) -> dict:
        logfile.write(f"ERROR: {msg}\n"); logfile.flush()
        with get_db_ctx() as db:
            db.execute(
                "UPDATE jobs SET status='failed', error_msg=?, "
                "finished_at=CURRENT_TIMESTAMP WHERE id=?",
                (msg[:500], job_id),
            )
        return {"error": msg}

    trials_in = extra_params.get("trials") or []
    if not trials_in:
        return _fail("No trials in batch")
    batch_id = extra_params.get("_batch_id") or f"b{int(_time.time())}_{_uuid.uuid4().hex[:6]}"
    work_dir = cfg.work_dir.replace("\\", "/")
    work_root = f"{work_dir}/hrnet_jobs"
    state_dir = f"{work_dir}/batch_runs/{batch_id}"
    logfile.write(f"=== Remote batch {batch_id}: {len(trials_in)} trial(s) ===\n")
    logfile.flush()

    try:
        # Reachability probe.
        probe = subprocess.run(
            _ssh_base_args(cfg) + [cfg.host, "echo", "ok"],
            capture_output=True, text=True, timeout=15,
        )
        if probe.returncode != 0 or "ok" not in probe.stdout:
            return _fail(f"Remote unreachable: {probe.stderr[:200]}")

        # Mint state dir (small — just batch.json + heartbeat + log).
        mk = subprocess.run(
            _py_cmd(cfg,
                    f"\"import os; os.makedirs(r'{cfg.work_dir}/batch_runs/{batch_id}', exist_ok=True)\""),
            capture_output=True, text=True, timeout=30,
        )
        if mk.returncode != 0:
            return _fail(f"mkdir state_dir failed: {mk.stderr[:200]}")

        # ── Phase A: resolve per-trial metadata (no uploads yet) ──
        # Build the spec list so we can write batch.json + launch the
        # runner BEFORE all videos finish uploading.  The runner waits
        # up to UPLOAD_WAIT_S per trial for its files to land, so trial 2's
        # video can SCP up while trial 1 is already inferencing on the GPU.
        local_script = Path(__file__).parent / "remote_hrnet_script.py"
        local_runner = Path(__file__).parent / "remote_batch_runner.py"
        # Shared per-subject MP-npz cache on remote — avoids re-uploading
        # the same 3-MB MediaPipe file into each trial's dir.  Filled
        # lazily by ``_upload_one`` (first trial of a subject uploads MP
        # to the shared path; subsequent trials skip).
        mp_cache_remote = f"{work_dir}/mp_cache"
        subprocess.run(
            _py_cmd(cfg, f"\"import os; os.makedirs(r'{cfg.work_dir}/mp_cache', exist_ok=True)\""),
            capture_output=True, text=True, timeout=15,
        )
        per_trial = []  # list of {t, sub, stem, outer, trial_remote, video_local, mp_npz_local, spec}
        _mp_uploaded_subjects: set[str] = set()
        # Per-subject MP combined npz to upload — uses legacy combined file
        # if present, otherwise builds an aggregated temp npz from the
        # per-trial layout.  Tracked here so we can clean up the tempfiles
        # in the finally block.
        from .mediapipe_prelabel import (
            build_combined_mp_npz_tempfile, has_mediapipe_data,
        )
        _mp_local_paths: dict[str, str] = {}
        _mp_tempfiles_to_unlink: list[str] = []
        for t in trials_in:
            if cancel_event.is_set():
                return _fail("Cancelled during upload")
            sub = t.get("subject_name") or subject_name
            tidx = int(t.get("trial_idx", 0))
            try:
                tmap = build_trial_map(sub)
                if tidx >= len(tmap):
                    raise ValueError(f"trial_idx {tidx} out of range for {sub}")
                tdef = tmap[tidx]
            except Exception as e:
                return _fail(f"build_trial_map({sub}): {e}")
            stem = tdef["trial_name"]                 # "Con04_L1"
            outer = f"{sub}_{stem}"                    # "Con04_Con04_L1"
            trial_remote = f"{work_root}/{outer}"
            if sub not in _mp_local_paths:
                legacy = settings.dlc_path / sub / "mediapipe_prelabels.npz"
                if legacy.exists():
                    _mp_local_paths[sub] = str(legacy)
                elif has_mediapipe_data(sub):
                    tmp = build_combined_mp_npz_tempfile(sub)
                    if not tmp:
                        return _fail(f"Could not assemble combined MediaPipe npz for {sub}")
                    _mp_local_paths[sub] = tmp
                    _mp_tempfiles_to_unlink.append(tmp)
                else:
                    return _fail(f"MediaPipe data missing for {sub}")
            mp_npz_local = _mp_local_paths[sub]
            per_trial.append(dict(
                t=t, sub=sub, stem=stem, outer=outer,
                trial_remote=trial_remote,
                video_local=tdef["video_path"],
                mp_npz_local=mp_npz_local,
                spec={
                    "stem":        stem,
                    "subject":     sub,
                    "trial_idx":   tidx,
                    "n_frames":    int(tdef["frame_count"]),
                    "start_frame": int(tdef.get("start_frame", 0)),
                    "camera_mode": "stereo",
                    "bbox_os":     t.get("bbox_os"),
                    "bbox_od":     t.get("bbox_od"),
                    # Shared per-subject MP cache path — runner reads from
                    # here instead of <trial_dir>/mediapipe_prelabels.npz.
                    "mp_path":     f"{mp_cache_remote}/{sub}.npz",
                },
            ))
        spec_trials = [p["spec"] for p in per_trial]

        # ── Phase B: clear any leftover done.flag files for these trials ──
        # The poller's done.flag scan greps every hrnet_jobs/*/output/*/
        # dir on the remote, so a stale done.flag from a prior run would
        # be misinterpreted as "this batch's trial just finished" — chips
        # would flash green immediately and the log would mention trials
        # that aren't part of the current batch (e.g. an old Con03 run).
        # Build one Python-on-remote script that deletes only the
        # done.flags for trials in *this* batch, leaving other batches'
        # flags intact.
        flag_paths = [
            f"{cfg.work_dir}/hrnet_jobs/{p['outer']}/output/{p['stem']}/done.flag"
            for p in per_trial
        ]
        if flag_paths:
            _flag_list = "[" + ",".join(f"r'{fp}'" for fp in flag_paths) + "]"
            subprocess.run(
                _py_cmd(cfg, f"\"import os\nfor _f in {_flag_list}:\n    "
                              "try: os.unlink(_f)\n    except Exception: pass\""),
                capture_output=True, text=True, timeout=30,
            )

        # Upload the runner + write batch.json.
        proc = subprocess.run(
            _scp_base_args(cfg) + [str(local_runner), f"{cfg.host}:{state_dir}/remote_batch_runner.py"],
            capture_output=True, text=True, timeout=30,
        )
        if proc.returncode != 0:
            return _fail(f"Failed to upload runner: {proc.stderr[-200:]}")

        spec = {
            "batch_id":       batch_id,
            "work_root":      work_root,
            "trials":         spec_trials,
            "created":        _time.time(),
            "upload_wait_s":  900,   # runner polls up to 15 min for late
                                      # uploads before declaring a trial failed
        }
        spec_json_local = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        spec_json_local.write(_json.dumps(spec))
        spec_json_local.close()
        proc = subprocess.run(
            _scp_base_args(cfg) + [spec_json_local.name, f"{cfg.host}:{state_dir}/batch.json"],
            capture_output=True, text=True, timeout=30,
        )
        os.unlink(spec_json_local.name)
        if proc.returncode != 0:
            return _fail(f"Failed to upload batch.json: {proc.stderr[-200:]}")

        # ── Phase C: upload trial 1's files, then launch ──
        # We launch the runner AFTER trial 1 is uploaded so the GPU starts
        # working immediately; uploads for trials 2..N continue in this
        # thread while inference runs on the remote.  The runner waits for
        # later trials' files to appear.
        def _upload_one(idx: int, log=None) -> bool:
            """Upload video + MP + script for per_trial[idx].  Returns
            True on success.  Idempotent — skips files already at the
            expected size on the remote.

            ``log`` is the file handle to write progress lines to.
            Defaults to the outer ``logfile`` so calls before the
            dispatcher returns work unchanged.  The background
            upload-tail thread passes its OWN file handle (the outer
            handle is closed by dispatch_remote_batch's finally block
            once the function returns) — without that, writes from the
            tail thread would error with "I/O operation on closed file".

            Resume optimisation: if the trial entry already has
            ``uploaded=True`` in params_json (carried over from a prior
            successful dispatch), skip the upload + size-checks entirely.

            MP de-duplication: MP npz uploads to a shared
            ``mp_cache/<subject>.npz`` path, once per subject.
            Subsequent trials of the same subject skip the MP upload.
            """
            log = log if log is not None else logfile
            pt = per_trial[idx]
            if pt["t"].get("uploaded"):
                try:
                    log.write(f"  {pt['stem']}: already uploaded — skipping checks\n")
                    log.flush()
                except Exception:
                    pass
                return True
            mk = subprocess.run(
                _py_cmd(cfg,
                        f"\"import os; os.makedirs(r'{cfg.work_dir}/hrnet_jobs/{pt['outer']}', exist_ok=True);"
                        f" os.makedirs(r'{cfg.work_dir}/hrnet_jobs/{pt['outer']}/output/{pt['stem']}', exist_ok=True)\""),
                capture_output=True, text=True, timeout=30,
            )
            if mk.returncode != 0:
                try:
                    log.write(f"  mkdir {pt['outer']} failed: {mk.stderr[:200]}\n")
                    log.flush()
                except Exception:
                    pass
                return False
            ok, _ = _scp_if_changed(cfg, pt["video_local"],
                                     f"{pt['trial_remote']}/video.mp4",
                                     timeout=600, logfile=log,
                                     label=f"video {pt['stem']}: ")
            if not ok:
                return False
            # MP npz: shared per-subject path; upload once, then skip
            # for subsequent trials of the same subject within this batch.
            if pt["sub"] not in _mp_uploaded_subjects:
                ok, _ = _scp_if_changed(cfg, pt["mp_npz_local"],
                                         f"{mp_cache_remote}/{pt['sub']}.npz",
                                         timeout=120, logfile=log,
                                         label=f"mp {pt['sub']}: ")
                if not ok:
                    return False
                _mp_uploaded_subjects.add(pt["sub"])
            proc = subprocess.run(
                _scp_base_args(cfg) + [str(local_script),
                                       f"{cfg.host}:{pt['trial_remote']}/remote_hrnet_script.py"],
                capture_output=True, text=True, timeout=30,
            )
            if proc.returncode != 0:
                try:
                    log.write(f"  script upload failed for {pt['stem']}: {proc.stderr[-200:]}\n")
                    log.flush()
                except Exception:
                    pass
                return False
            # Mark uploaded on parent params_json so the modal can un-dim.
            # IMPORTANT: do NOT blast the entire trials list from our stale
            # in-memory snapshot — the poller writes outcomes to the DB
            # concurrently and our snapshot would clobber them (chip flickers
            # green→blue).  Instead, re-read fresh from DB and merge by
            # full stem.
            pt["t"]["uploaded"] = True
            try:
              with _params_lock:
                with get_db_ctx() as _db_up:
                    _row = _db_up.execute(
                        "SELECT params_json FROM jobs WHERE id=?", (job_id,)
                    ).fetchone()
                    _existing = {}
                    try:
                        _existing = _json.loads(_row["params_json"]) if _row and _row["params_json"] else {}
                    except (ValueError, TypeError):
                        pass
                    _fresh_trials = _existing.get("trials") or []
                    # Locate the matching trial in the fresh list by full
                    # stem (subject + trial_name).  Falls back to short
                    # name if subject isn't on the trial dict.
                    target_full = pt.get("full_stem") or pt["stem"]
                    target_short = pt["stem"]
                    matched = False
                    for _ft in _fresh_trials:
                        _name = (_ft.get("trial_name") or "") or ""
                        _sub = (_ft.get("subject_name") or "") or ""
                        _fs = _name if "_" in _name else (f"{_sub}_{_name}" if _sub else _name)
                        if _fs == target_full or _name == target_short:
                            _ft["uploaded"] = True
                            matched = True
                            break
                    if not matched:
                        # Nothing matched — don't write anything rather than
                        # risk clobbering DB state.
                        return True
                    _existing["trials"] = _fresh_trials
                    _db_up.execute(
                        "UPDATE jobs SET params_json=? WHERE id=?",
                        (_json.dumps(_existing), job_id),
                    )
            except Exception:
                pass
            return True

        if per_trial:
            if cancel_event.is_set():
                return _fail("Cancelled during upload")
            if not _upload_one(0):
                return _fail(f"Failed to upload trial 1 ({per_trial[0]['stem']})")

        # Detached launcher (same pattern as HRnet single-trial launcher).
        runner_log = f"{state_dir}/runner.log"
        launcher_lines = [
            "import subprocess, os, sys",
            f"args = [r'{cfg.python_executable}', '-u',",
            f"        r'{state_dir}/remote_batch_runner.py',",
            f"        r'{state_dir}']",
            f"log_fh = open(r'{runner_log}', 'a', buffering=1)",
            "creationflags = 0",
            "if sys.platform == 'win32':",
            "    creationflags = (subprocess.DETACHED_PROCESS"
            " | subprocess.CREATE_NEW_PROCESS_GROUP"
            " | subprocess.CREATE_BREAKAWAY_FROM_JOB)",
            "p = subprocess.Popen(args, stdin=subprocess.DEVNULL,",
            "                     stdout=log_fh, stderr=log_fh,",
            "                     close_fds=True, creationflags=creationflags)",
            "print(p.pid)",
        ]
        launcher_local = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
        launcher_local.write("\n".join(launcher_lines))
        launcher_local.close()
        remote_launcher = f"{state_dir}/launcher.py"
        proc = subprocess.run(
            _scp_base_args(cfg) + [launcher_local.name, f"{cfg.host}:{remote_launcher}"],
            capture_output=True, text=True, timeout=30,
        )
        os.unlink(launcher_local.name)
        if proc.returncode != 0:
            return _fail(f"Failed to upload launcher: {proc.stderr[-200:]}")

        # Spawn the runner (one-shot — launcher exits after Popen).
        result = subprocess.run(
            _ssh_base_args(cfg) + [cfg.host, cfg.python_executable, "-u", remote_launcher],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return _fail(f"Launcher failed: {result.stderr[-300:] or result.stdout[-300:]}")
        try:
            remote_pid = int(result.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            remote_pid = None

        logfile.write(f"Runner launched, pid={remote_pid}, state_dir={state_dir}\n")
        logfile.flush()

        # Persist batch_id + remote_pid on the parent job for recovery.
        with get_db_ctx() as db:
            row = db.execute("SELECT params_json FROM jobs WHERE id=?", (job_id,)).fetchone()
            existing = {}
            try:
                existing = _json.loads(row["params_json"]) if row and row["params_json"] else {}
            except (ValueError, TypeError):
                pass
            existing["_batch_id"] = batch_id
            existing["_remote_pid"] = remote_pid
            db.execute(
                "UPDATE jobs SET params_json=?, tmux_session=? WHERE id=?",
                (_json.dumps(existing), f"pid:{remote_pid}" if remote_pid else "",
                 job_id),
            )

        # ── Phase E: spawn a background upload-tail thread ──
        # Runner is alive and starting trial 1 (already uploaded).  We
        # don't block the queue-manager dispatch thread on the remaining
        # uploads — they happen in a daemon thread so the GPU keeps
        # working while videos for trials 2..N continue to SCP up, and
        # the queue can dispatch the next job behind this one without
        # waiting for the full N-video upload phase.
        if len(per_trial) > 1:
            import threading as _th
            def _upload_tail(per_trial=per_trial, cancel_event=cancel_event,
                             log_path=log_path,
                             _tempfiles=_mp_tempfiles_to_unlink):
                tail_log = None
                try:
                    tail_log = open(log_path, "a", buffering=1)
                    tail_log.write(f"=== Background upload of {len(per_trial)-1} trial(s) (parallel with inference) ===\n")
                    tail_log.flush()
                    for i in range(1, len(per_trial)):
                        if cancel_event.is_set():
                            tail_log.write("Cancelled during background uploads\n")
                            tail_log.flush()
                            break
                        # Pass the thread-owned log handle so writes go to
                        # this still-open file, not the dispatcher's
                        # already-closed handle.
                        if not _upload_one(i, log=tail_log):
                            tail_log.write(f"  Warning: upload failed for {per_trial[i]['stem']}\n")
                            tail_log.flush()
                            # Continue with the rest — the runner marks
                            # missing-files trials as failed once its
                            # upload_wait_s timeout elapses.
                    tail_log.write("=== Background uploads complete ===\n")
                    tail_log.flush()
                except Exception as _e:
                    logger.exception(f"Job {job_id} upload-tail error: {_e}")
                finally:
                    try:
                        if tail_log is not None:
                            tail_log.close()
                    except Exception:
                        pass
                    for _tmp in _tempfiles:
                        try:
                            os.unlink(_tmp)
                        except OSError:
                            pass
            _th.Thread(target=_upload_tail, daemon=True).start()

        return {"batch_id": batch_id, "remote_pid": remote_pid, "state_dir": state_dir}
    except Exception as e:
        logger.exception(f"Job {job_id} dispatch_remote_batch error")
        return _fail(str(e))
    finally:
        logfile.close()
        # If the background upload-tail thread was started, it owns
        # tempfile cleanup so the files survive until later trials are
        # uploaded.  Only clean up here when no tail thread will run.
        try:
            if len(per_trial) <= 1:
                for _tmp in _mp_tempfiles_to_unlink:
                    try:
                        os.unlink(_tmp)
                    except OSError:
                        pass
        except NameError:
            pass


def poll_remote_batch(
    job_id: int,
    cfg: RemoteConfig,
    batch_id: str,
    log_path: str,
    registry,
    parent_extra: dict | None = None,
) -> None:
    """Poll the remote batch's status and download finished trials as they go.

    Runs in a daemon thread.  Exits when ``batch_status.json`` reports
    ``status="completed"`` (or ``"failed"`` / ``"cancelled"``) AND every
    trial's outputs have been pulled locally.

    Updates per-trial ``outcome`` in the parent job's params_json:
        ok          — done.flag = "ok" + heatmaps SCPed locally
        remote_only — done.flag = "ok" but SCP failed
        failed      — done.flag = "failed: ..."
    """
    # Defensive logging — if anything in the setup below raises in a
    # daemon thread, the thread dies silently without leaving any trace
    # in the user-facing log.  Open log_path FIRST so even import or
    # config errors get recorded.
    import traceback as _tb
    try:
        logfile = open(log_path, "a", buffering=1)
    except Exception as e:
        # Last-resort console print.  If log_path is broken there's
        # nothing user-visible we can do.
        print(f"[poll_remote_batch] cannot open log {log_path}: {e}", flush=True)
        return
    logfile.write(f"=== Polling batch {batch_id} (job_id={job_id}) ===\n"); logfile.flush()

    try:
        import json as _json
        from .skeleton_data import _skeleton_dir
        from ..db import get_db_ctx

        work_dir = cfg.work_dir.replace("\\", "/")
        work_root = f"{work_dir}/hrnet_jobs"
        state_dir = f"{work_dir}/batch_runs/{batch_id}"
        status_file = f"{state_dir}/batch_status.json"
        cancel_event = registry.register_cancel_event(job_id)
    except Exception as e:
        logfile.write(f"FATAL during setup: {e}\n{_tb.format_exc()}\n")
        logfile.flush()
        logfile.close()
        return

    pulled: set[str] = set()      # stems we've already SCPed locally
    poll_interval = 8
    consecutive_no_status = 0

    try:
        while True:
            if cancel_event.is_set():
                logfile.write("Local cancel — leaving runner alive on remote\n")
                logfile.flush()
                break

            time.sleep(poll_interval)
            status = _read_remote_status(cfg, status_file)
            if status is None:
                consecutive_no_status += 1
                # 5 min of nothing → assume runner crashed before writing
                # any status.  A later restart can re-launch the batch.
                if consecutive_no_status >= 40:
                    logfile.write("No batch_status.json after 5 min — giving up\n")
                    break
                continue
            consecutive_no_status = 0

            n_done = int(status.get("n_done", 0))
            n_total = max(1, int(status.get("n_total", 1)))
            current = status.get("current") or ""
            # Fractional progress: read the inference script's per-trial
            # status file (updated continuously while a trial runs) and
            # add (sub_pct / 100) to n_done before scaling.  Without this,
            # a 4-trial batch sits at 0% for the first quarter of its
            # wall time.
            sub_pct = 0.0
            cur_status = _read_remote_status(cfg, f"{state_dir}/current_trial.json")
            if cur_status and cur_status.get("status") == "running":
                try:
                    sub_pct = float(cur_status.get("progress_pct") or 0)
                except (TypeError, ValueError):
                    sub_pct = 0.0
            global_pct = 100.0 * (n_done + sub_pct / 100.0) / n_total
            with get_db_ctx() as db:
                db.execute(
                    "UPDATE jobs SET progress_pct=? WHERE id=?",
                    (round(global_pct, 1), job_id),
                )

            # Poll done.flags across every per-trial output dir.  One SSH
            # call walks ``hrnet_jobs/*/output/*/done.flag`` and prints
            # "<stem>\t<contents>" lines.  Stem is the dirname of done.flag
            # (e.g. ``Con04_L1``).
            ls = subprocess.run(
                _py_cmd(cfg,
                        f"\"import os, glob; root=r'{cfg.work_dir}/hrnet_jobs';"
                        f" [print(os.path.basename(os.path.dirname(p)) + '\\t' + open(p).read().strip())"
                        f" for p in glob.glob(os.path.join(root,'*','output','*','done.flag'))]\""),
                capture_output=True, text=True, timeout=30,
            )
            new_outcomes: dict[str, str] = {}
            if ls.returncode == 0:
                for line in ls.stdout.splitlines():
                    line = line.strip()
                    if "\t" not in line:
                        continue
                    stem, outcome_text = line.split("\t", 1)
                    new_outcomes[stem] = outcome_text or "ok"

            # Pull any trial whose flag is "ok" and hasn't been pulled yet.
            params = parent_extra or {}
            try:
                with get_db_ctx() as db:
                    row = db.execute(
                        "SELECT params_json FROM jobs WHERE id=?", (job_id,)
                    ).fetchone()
                params = _json.loads(row["params_json"]) if row and row["params_json"] else {}
            except (ValueError, TypeError):
                pass
            trials_meta = params.get("trials") or []
            # Index trials by their FULL stem (subject prefix + trial
            # name).  The done.flag scan returns full stems like
            # "Con03_L1" / "MSA08_R1"; matching on those uniquely
            # identifies a trial.  Earlier code fell back to a
            # subject-prefix strip ("Con03_L1" → "L1") and looked up by
            # short name, which collides whenever multiple subjects in
            # one batch share short names (every batch with >1 subject
            # has multiple "L1" / "R1") and silently mis-attributes
            # cross-batch leftovers like Con03 to whichever batch trial
            # happened to land last in the dict.
            def _full_stem(t):
                name = t.get("trial_name", "") or ""
                sub = t.get("subject_name", "") or ""
                if not name:
                    return None
                return name if "_" in name else f"{sub}_{name}" if sub else name
            stem_to_idx = {}
            for i, t in enumerate(trials_meta):
                fs = _full_stem(t)
                if fs:
                    stem_to_idx[fs] = i

            for stem, outcome_text in new_outcomes.items():
                if stem in pulled:
                    continue
                ok = outcome_text.startswith("ok")
                parent_idx = stem_to_idx.get(stem)

                # The done.flag glob scans every hrnet_jobs/*/output/*/
                # directory on the remote, so it sees flags from prior
                # batches and other concurrently-running batches.  Skip
                # any flag that doesn't belong to *this* batch's trial
                # list — otherwise we'd "download" Con03 leftovers into
                # the current MSA-only batch.
                if parent_idx is None:
                    continue

                if not ok:
                    # Trial failed remotely.
                    if parent_idx is not None and parent_idx < len(trials_meta):
                        trials_meta[parent_idx]["outcome"] = "failed"
                        trials_meta[parent_idx]["outcome_error"] = outcome_text[:300]
                    pulled.add(stem)
                    logfile.write(f"  {stem}: FAILED ({outcome_text})\n")
                    logfile.flush()
                    continue

                # Trial succeeded — pull the outputs.  Stem is "Con04_L1";
                # subject is the first underscore-segment.  Per-trial
                # outputs live at hrnet_jobs/<sub>_<stem>/output/<stem>/.
                sub = stem.split("_", 1)[0] if "_" in stem else stem
                local_dir = _skeleton_dir(sub) / stem
                local_dir.mkdir(parents=True, exist_ok=True)
                remote_dir = f"{work_root}/{sub}_{stem}/output/{stem}"
                fail_msg = None
                for fname, timeout_s, required in _BATCH_HRNET_FILES:
                    p = subprocess.run(
                        _scp_base_args(cfg)
                        + [f"{cfg.host}:{remote_dir}/{fname}", str(local_dir / fname)],
                        capture_output=True, text=True, timeout=timeout_s,
                    )
                    if p.returncode != 0 and required:
                        fail_msg = p.stderr[-200:]
                        break
                if parent_idx is not None and parent_idx < len(trials_meta):
                    if fail_msg is None:
                        trials_meta[parent_idx]["outcome"] = "ok"
                    else:
                        trials_meta[parent_idx]["outcome"] = "remote_only"
                        trials_meta[parent_idx]["outcome_error"] = fail_msg[:300]
                pulled.add(stem)
                logfile.write(f"  {stem}: {'ok' if fail_msg is None else 'remote_only'}\n")
                logfile.flush()

            # Persist updated outcomes back to params_json so the UI shows
            # progress mid-batch.  Re-read fresh under the lock and merge
            # ONLY the per-trial fields we touched (outcome, outcome_error) —
            # don't blast our older `trials_meta` snapshot over fields the
            # upload-tail may have written (e.g. `uploaded=True`) during the
            # SCPs we just did between reading and writing.
            with _params_lock:
                with get_db_ctx() as db:
                    _row_w = db.execute(
                        "SELECT params_json FROM jobs WHERE id=?", (job_id,)
                    ).fetchone()
                    try:
                        _fresh = _json.loads(_row_w["params_json"]) if _row_w and _row_w["params_json"] else {}
                    except (ValueError, TypeError):
                        _fresh = {}
                    _fresh_trials = _fresh.get("trials") or []
                    # Build full-stem → index for the fresh list.
                    _fresh_idx = {}
                    for _i, _ft in enumerate(_fresh_trials):
                        _n = (_ft.get("trial_name") or "") or ""
                        _s = (_ft.get("subject_name") or "") or ""
                        _fs = _n if "_" in _n else (f"{_s}_{_n}" if _s else _n)
                        if _fs:
                            _fresh_idx[_fs] = _i
                    # Walk our in-memory trials_meta and copy outcome fields
                    # into the fresh list.
                    for _src in trials_meta:
                        if "outcome" not in _src and "outcome_error" not in _src:
                            continue
                        _n = (_src.get("trial_name") or "") or ""
                        _s = (_src.get("subject_name") or "") or ""
                        _fs = _n if "_" in _n else (f"{_s}_{_n}" if _s else _n)
                        _idx = _fresh_idx.get(_fs)
                        if _idx is None:
                            continue
                        if "outcome" in _src:
                            _fresh_trials[_idx]["outcome"] = _src["outcome"]
                        if "outcome_error" in _src:
                            _fresh_trials[_idx]["outcome_error"] = _src["outcome_error"]
                    _fresh["trials"] = _fresh_trials
                    db.execute(
                        "UPDATE jobs SET params_json=? WHERE id=?",
                        (_json.dumps(_fresh), job_id),
                    )

            # Are we done?
            if status.get("status") in ("completed", "failed", "cancelled"):
                # One more pass to make sure we didn't miss any trial that
                # finished between the last ls and the status flip.
                if all(t.get("outcome") for t in trials_meta if t.get("trial_name")):
                    break
                # Else loop one more time — there are still trials we
                # haven't processed locally.

        # Final status: derived from per-trial outcomes (same logic as
        # queue_manager batch finalization).
        with get_db_ctx() as db:
            row = db.execute("SELECT params_json FROM jobs WHERE id=?", (job_id,)).fetchone()
            params = _json.loads(row["params_json"]) if row and row["params_json"] else {}
            ts = params.get("trials") or []
            n_ok   = sum(1 for t in ts if t.get("outcome") == "ok")
            n_rem  = sum(1 for t in ts if t.get("outcome") == "remote_only")
            n_fail = sum(1 for t in ts if t.get("outcome") == "failed")
            N = max(1, len(ts))
            if n_fail > 0:
                final_status, err = "failed", f"Ran {n_ok + n_rem}/{N} trials"
            elif n_rem > 0:
                final_status, err = "completed", f"Download incomplete: {n_ok}/{N} downloaded"
            else:
                final_status, err = "completed", None
            db.execute(
                "UPDATE jobs SET status=?, error_msg=?, progress_pct=100, "
                "finished_at=CURRENT_TIMESTAMP WHERE id=?",
                (final_status, err, job_id),
            )
        logfile.write("=== Polling complete ===\n"); logfile.flush()
    except Exception as e:
        logger.exception(f"Job {job_id} poll_remote_batch error")
    finally:
        logfile.close()
        registry._threads.pop(job_id, None)


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

    # Normalize work_dir to forward slashes — see comment in
    # remote_preprocess_batch.  Windows OpenSSH SCP rejects mixed-slash
    # paths.
    _work_dir = cfg.work_dir.replace("\\", "/")
    remote_output_dir = f"{_work_dir}/preprocess_output"
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
                    from .remote_results import (
                        output_specs as _outputs,
                        download_one as _dl_one,
                    )
                    with get_db_ctx() as _db:
                        srow = _db.execute(
                            "SELECT id FROM subjects WHERE name = ?",
                            (subj_name,),
                        ).fetchone()
                    if srow:
                        any_ok = False
                        for spec in _outputs("mediapipe", subj_name, remote_output_dir):
                            if _dl_one(spec, cfg, srow["id"], job_id):
                                any_ok = True
                        if any_ok:
                            logfile.write(f"  Downloaded MediaPipe outputs for {subj_name}\n")
                            total_downloaded += 1
                        else:
                            logfile.write(f"  Warning: npz not found for {subj_name}\n")
                    else:
                        logfile.write(f"  Warning: subject {subj_name} not in DB\n")

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


def resume_preprocess_monitor(
    job_id: int,
    cfg: RemoteConfig,
    log_path: str,
    registry,
):
    """Resume monitoring a detached remote preprocessing job after server restart.

    Similar to remote_train_monitor(resume=True) but for preprocessing jobs.
    Monitors the preprocess-specific status.json and log files, then handles
    result downloads on completion.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    from ..config import get_settings
    settings = get_settings()

    # Forward-slash normalization for SCP (see remote_preprocess_batch).
    _work_dir = cfg.work_dir.replace("\\", "/")
    remote_output_dir = f"{_work_dir}/preprocess_output"
    remote_log_file = f"{_work_dir}/preprocess_{job_id}.log"
    remote_status_file = f"{_work_dir}/preprocess_{job_id}_status.json"

    cancel_event = registry.register_cancel_event(job_id)

    # Read PID from jobs table.  ``get_db_ctx`` uses a dict row factory, so
    # rows are dicts keyed by column name (not tuples).
    with get_db_ctx() as db:
        row = db.execute("SELECT tmux_session FROM jobs WHERE id = ?", (job_id,)).fetchone()
    remote_pid = None
    sess = row.get("tmux_session") if row else None
    if sess and sess.startswith("pid:"):
        try:
            remote_pid = int(sess.split(":")[1])
        except (ValueError, IndexError):
            pass

    def _update_progress(pct):
        with get_db_ctx() as db:
            db.execute("UPDATE jobs SET progress_pct = ? WHERE id = ?",
                        (round(pct, 1), job_id))

    def _fail(msg):
        logger.error(f"Job {job_id} resume preprocess monitor failed: {msg}")
        # Don't overwrite a job already marked completed (startup may have
        # set it synchronously before spawning this thread).
        with get_db_ctx() as db:
            db.execute(
                """UPDATE jobs SET status = 'failed', error_msg = ?,
                   finished_at = CURRENT_TIMESTAMP
                   WHERE id = ? AND status != 'completed'""",
                (msg, job_id),
            )

    def _check_cancel():
        if cancel_event.is_set():
            raise InterruptedError("Job cancelled")

    def _run_remote_proc(cmd, logfile, phase_name):
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
        with open(log_path, "a") as logfile:
            logfile.write(f"\n=== Resuming preprocessing monitor (PID {remote_pid}) ===\n")
            logfile.flush()

            # ── Check if already completed (skip monitor loop) ───
            status = _read_remote_status(cfg, remote_status_file)
            if status and status.get("status") == "completed":
                logfile.write("=== Remote already completed, skipping to download ===\n")
                logfile.flush()
            else:
                # ── Monitor loop ─────────────────────────────────
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

            # ── Download results ──────────────────────────────────
            logfile.write(f"=== Downloading results from {cfg.host} ===\n")
            logfile.flush()
            _update_progress(85.0)

            # Determine what steps were run from the status
            do_mp = True  # conservative: try to download both
            do_blur = True

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

                # Download MediaPipe outputs (per-trial + legacy fallback)
                from .remote_results import (
                    output_specs as _outputs,
                    download_one as _dl_one,
                )
                with get_db_ctx() as _db:
                    srow = _db.execute(
                        "SELECT id FROM subjects WHERE name = ?",
                        (subj_name,),
                    ).fetchone()
                if srow:
                    for spec in _outputs("mediapipe", subj_name, remote_output_dir):
                        if _dl_one(spec, cfg, srow["id"], job_id):
                            logfile.write(f"  Downloaded {spec.label} for {subj_name}\n")
                # (download failures are OK — subject might not have MP results)

                # Download deidentified videos
                remote_deident = f"{cfg.host}:{remote_output_dir}/{subj_name}/deidentified"
                local_deident_dir = settings.video_path / "deidentified"
                local_deident_dir.mkdir(parents=True, exist_ok=True)

                result2 = subprocess.run(
                    _py_cmd(cfg, f"\"import os; d = r'{remote_output_dir}/{subj_name}/deidentified'; print('\\n'.join(os.listdir(d))) if os.path.isdir(d) else print('')\""),
                    capture_output=True, text=True, timeout=15,
                )
                deident_files = [f for f in result2.stdout.strip().splitlines() if f.endswith(".mp4")]

                for df in deident_files:
                    dl_cmd = _scp_base_args(cfg) + [
                        f"{cfg.host}:{remote_output_dir}/{subj_name}/deidentified/{df}",
                        str(local_deident_dir / df),
                    ]
                    proc = _run_remote_proc(dl_cmd, logfile, f"Download blur {df}")
                    if proc.returncode == 0:
                        logfile.write(f"  Downloaded deidentified/{df}\n")

                if deident_files:
                    local_dlc_dir = settings.dlc_path / subj_name
                    local_dlc_dir.mkdir(parents=True, exist_ok=True)
                    (local_dlc_dir / ".deidentified").write_text("")

                logfile.flush()
                pct = 85.0 + (i + 1) / max(len(remote_subjects), 1) * 15.0
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
        logger.info(f"Job {job_id} resume preprocess monitor cancelled")
        with get_db_ctx() as db:
            db.execute(
                """UPDATE jobs SET status = 'cancelled',
                   finished_at = CURRENT_TIMESTAMP WHERE id = ?""",
                (job_id,),
            )
    except Exception as e:
        logger.exception(f"Job {job_id} resume preprocess monitor error")
        _fail(str(e))
    finally:
        registry._processes.pop(job_id, None)
        registry._threads.pop(job_id, None)
        registry.unregister_cancel_event(job_id)


def remote_deidentify(
    job_id: int,
    cfg: RemoteConfig,
    subject_name: str,
    log_path: str,
    registry,
    trial_idx: int | None = None,
    batch_idx: int = 0,
    batch_total: int = 1,
):
    """Remote deidentify: upload video + saved blur specs, render on remote, download result.

    Uses locally saved blur specs, face detections, and hand settings from the DB.
    Uploads them as a JSON bundle, along with the video and mediapipe/pose npz files.
    Runs the render on the remote host using a standalone script, then downloads the result.

    If trial_idx is specified, only that trial is rendered (not all trials).

    ``batch_idx`` / ``batch_total`` scale the local 0-100 progress into the
    parent batch's global 0-100 so multi-trial deidentify shows monotonic
    progress and a meaningful ETA across the entire batch.  Failures don't
    mark the parent job failed in batch mode — the queue manager owns final
    status (mirrors the ``_is_batched`` pattern in ``remote_hrnet_job``).
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    from ..config import get_settings
    from .video import build_trial_map

    settings = get_settings()
    # Use subject-scoped work dir (no job_id) so previously-uploaded videos and
    # npz files are reused across runs for the same subject.
    remote_work = f"{cfg.work_dir}/deidentify_{subject_name}"

    cancel_event = registry.register_cancel_event(job_id)

    _is_batched = batch_total > 1

    def _update_progress(pct):
        global_pct = (100.0 * (batch_idx + (pct or 0) / 100.0) / batch_total
                      if _is_batched else pct)
        with get_db_ctx() as db:
            db.execute("UPDATE jobs SET progress_pct = ? WHERE id = ?",
                       (round(global_pct, 1), job_id))

    def _fail(msg):
        """Log + raise.  The queue_manager wrapper catches the exception
        and computes the final outcome; if it's not batched, it writes
        status='failed' itself.  Previously this only wrote the DB row
        without raising -- which got OVERWRITTEN by the queue manager's
        post-batch ``outcome='remote_only' -> status='completed'`` path
        whenever the worker died before producing an output file (e.g.
        ffmpeg pip-install fails on the remote)."""
        logger.error(f"Job {job_id} remote deidentify failed: {msg}")
        raise RuntimeError(msg)

    def _check_cancel():
        if cancel_event.is_set():
            raise InterruptedError("Job cancelled")

    try:
        with open(log_path, "w") as logfile:
            # ── Gather local data ──────────────────────────────
            logfile.write(f"=== Preparing deidentify data for {subject_name} ===\n")
            logfile.flush()

            with get_db_ctx() as db:
                subj = db.execute("SELECT * FROM subjects WHERE name = ?", (subject_name,)).fetchone()
                if not subj:
                    _fail(f"Subject not found: {subject_name}")
                    return
                subj_id = subj["id"]

            camera_mode = subj.get("camera_mode") or "stereo"
            trials = build_trial_map(subject_name, camera_mode=camera_mode)
            if not trials:
                _fail("No trials found")
                return

            # Export blur data from DB as JSON bundle.
            # Always compute total_frame_count across ALL trials (needed for npz offset detection).
            # If trial_idx is specified, only include that trial in the bundle for rendering.
            trial_indices = [trial_idx] if trial_idx is not None else list(range(len(trials)))
            total_frame_count = sum(t.get("frame_count", 0) for t in trials if t.get("frame_count"))

            bundle = {"subject_name": subject_name, "trials": [], "total_frame_count": total_frame_count}
            for ti in trial_indices:
                trial = trials[ti]
                with get_db_ctx() as db:
                    specs = [dict(s) for s in db.execute(
                        "SELECT * FROM blur_specs WHERE subject_id = ? AND trial_idx = ?",
                        (subj_id, ti)).fetchall()]
                    faces = [dict(f) for f in db.execute(
                        "SELECT frame_num, x1, y1, x2, y2, side FROM face_detections WHERE subject_id = ? AND trial_idx = ?",
                        (subj_id, ti)).fetchall()]
                    hs = db.execute(
                        "SELECT * FROM blur_hand_settings WHERE subject_id = ? AND trial_idx = ?",
                        (subj_id, ti)).fetchone()

                bundle["trials"].append({
                    "trial_idx": ti,
                    "trial_name": trial["trial_name"],
                    "video_name": Path(trial["video_path"]).name,
                    "start_frame": trial.get("start_frame", 0),
                    "frame_count": trial["frame_count"],
                    "fps": trial.get("fps", 30),
                    "blur_specs": specs,
                    "face_detections": faces,
                    "hand_settings": dict(hs) if hs else None,
                })

            # Write bundle to temp file
            import tempfile
            bundle_path = os.path.join(tempfile.gettempdir(), f"deid_bundle_{job_id}.json")
            with open(bundle_path, "w") as bf:
                json.dump(bundle, bf)

            _update_progress(5)
            _check_cancel()

            # ── Phase 1: Create remote dir and upload files ────
            logfile.write(f"=== Uploading to {cfg.host}:{remote_work} ===\n")
            logfile.flush()

            # Create the shared subject work directory
            subprocess.run(
                _py_cmd(cfg, f"\"import os; os.makedirs(r'{remote_work}', exist_ok=True)\""),
                capture_output=True, timeout=15,
            )

            # Check which files already exist in remote_work to avoid re-uploading
            result = subprocess.run(
                _py_cmd(cfg, f"\"import os; d=r'{remote_work}'; print('\\n'.join(os.listdir(d)) if os.path.isdir(d) else '')\""),
                capture_output=True, text=True, timeout=15,
            )
            remote_existing = set(result.stdout.strip().splitlines()) if result.returncode == 0 else set()
            if remote_existing:
                logfile.write(f"  {len(remote_existing)} file(s) already on remote — skipping those\n")
                logfile.flush()

            # Upload video files (skip those already present)
            video_files = set()
            for t in bundle["trials"]:
                if t["blur_specs"]:  # only upload videos that have specs
                    vpath = str(settings.video_path / t["video_name"])
                    if os.path.exists(vpath):
                        video_files.add(vpath)

            video_list = sorted(video_files)
            to_upload = [vp for vp in video_list if Path(vp).name not in remote_existing]
            if len(to_upload) < len(video_list):
                logfile.write(f"  Skipping {len(video_list) - len(to_upload)} already-uploaded video(s)\n")
                logfile.flush()

            for i, vpath in enumerate(to_upload):
                _check_cancel()
                vname = Path(vpath).name
                logfile.write(f"  Uploading {vname}...\n")
                logfile.flush()
                proc = subprocess.run(
                    _scp_base_args(cfg) + [vpath, f"{cfg.host}:{remote_work}/"],
                    capture_output=True, text=True, timeout=600,
                )
                if proc.returncode != 0:
                    logfile.write(f"  Warning: failed to upload {vname}: {proc.stderr[:100]}\n")
                pct = 5 + (i + 1) / max(len(to_upload), 1) * 15
                _update_progress(pct)

            # Upload mediapipe + pose prelabels.  Always force-upload
            # so a freshly re-run npz locally is what the remote uses
            # -- the previous behaviour skipped when the file was
            # already on the remote, which silently let stale
            # landmarks drive the hand mask after a Labels-page MP
            # rerun.  These are small (low single-digit MB) so the
            # bandwidth cost of always uploading is negligible.
            #
            # MP locally is per-trial now (<dlc>/<subj>/<stem>/
            # mediapipe.npz), but the deidentify worker still reads
            # the legacy combined ``mediapipe_prelabels.npz`` -- so
            # aggregate the per-trial slices into a temp combined
            # file and upload that.  Pose stays subject-wide so
            # uploads from its on-disk path directly.
            dlc_dir = settings.dlc_path / subject_name
            from .mediapipe_prelabel import (
                build_combined_mp_npz_tempfile, has_mediapipe_data,
            )
            # 1) MediaPipe combined npz (built fresh from per-trial files
            #    when no legacy combined file exists locally).
            _mp_tempfile_to_unlink: str | None = None
            mp_local: str | None = None
            legacy_mp = dlc_dir / "mediapipe_prelabels.npz"
            if legacy_mp.exists():
                mp_local = str(legacy_mp)
            elif has_mediapipe_data(subject_name):
                _mp_tempfile_to_unlink = build_combined_mp_npz_tempfile(subject_name)
                mp_local = _mp_tempfile_to_unlink
            if mp_local:
                proc = subprocess.run(
                    _scp_base_args(cfg) + [mp_local,
                        f"{cfg.host}:{remote_work}/mediapipe_prelabels.npz"],
                    capture_output=True, timeout=120,
                )
                if proc.returncode == 0:
                    logfile.write("  Uploaded mediapipe_prelabels.npz (force-fresh)\n")
                else:
                    logfile.write("  Warning: mediapipe_prelabels.npz upload failed\n")
            if _mp_tempfile_to_unlink:
                try:
                    os.unlink(_mp_tempfile_to_unlink)
                except OSError:
                    pass
            # 2) Pose npz (still subject-wide on disk).
            pose_path = dlc_dir / "pose_prelabels.npz"
            if pose_path.exists():
                proc = subprocess.run(
                    _scp_base_args(cfg) + [str(pose_path),
                        f"{cfg.host}:{remote_work}/pose_prelabels.npz"],
                    capture_output=True, timeout=120,
                )
                if proc.returncode == 0:
                    logfile.write("  Uploaded pose_prelabels.npz (force-fresh)\n")
                else:
                    logfile.write("  Warning: pose_prelabels.npz upload failed\n")
            logfile.flush()

            _update_progress(25)
            _check_cancel()

            # ── Phase 2: Upload and launch render script ───────
            # Use a job-id-specific run subdir for bundle/status/log/output so
            # concurrent or sequential runs for the same subject don't clobber each other.
            remote_run = f"{remote_work}/run_{job_id}"
            remote_status_file = f"{remote_run}/status.json"
            remote_log_file = f"{remote_run}/render.log"
            remote_output_dir = f"{remote_run}/output"

            # Upload the deidentify module files needed for rendering (always fresh).
            # Previously this silently swallowed scp failures, so a flaky
            # ssh handshake or a timeout-during-transfer could leave the
            # remote running an older deidentify.py and the user would
            # see stale-render bugs that don't reproduce locally.  Now
            # we check returncode + remote file size after upload and
            # fail the job loudly if either is wrong.
            service_dir = Path(__file__).parent
            for mod_file in ("deidentify.py", "ffmpeg.py"):
                mod_path = service_dir / mod_file
                if not mod_path.exists():
                    continue
                local_size = mod_path.stat().st_size
                up = subprocess.run(
                    _scp_base_args(cfg) + [str(mod_path), f"{cfg.host}:{remote_work}/"],
                    capture_output=True, text=True, timeout=60,
                )
                if up.returncode != 0:
                    _fail(f"scp {mod_file} failed: {up.stderr[:200]}")
                    return
                probe = subprocess.run(
                    _py_cmd(cfg,
                        f"\"import os; "
                        f"p=r'{remote_work}/{mod_file}'; "
                        f"print(os.path.getsize(p) if os.path.exists(p) else -1)\""),
                    capture_output=True, text=True, timeout=15,
                )
                try:
                    remote_size = int((probe.stdout or "-1").strip())
                except ValueError:
                    remote_size = -1
                if remote_size != local_size:
                    _fail(f"Module upload size mismatch for {mod_file}: "
                          f"local={local_size}, remote={remote_size}")
                    return
                logfile.write(f"  Uploaded {mod_file} ({local_size} bytes)\n")

            # Create the per-job run subdirectory and upload bundle there
            subprocess.run(
                _py_cmd(cfg, f"\"import os; os.makedirs(r'{remote_run}', exist_ok=True)\""),
                capture_output=True, timeout=15,
            )
            proc = subprocess.run(
                _scp_base_args(cfg) + [bundle_path, f"{cfg.host}:{remote_run}/bundle.json"],
                capture_output=True, text=True, timeout=30,
            )
            if proc.returncode != 0:
                _fail(f"Failed to upload bundle to run dir: {proc.stderr[:200]}")
                return

            logfile.write("=== Launching remote render ===\n")
            logfile.flush()

            # Write and upload worker script (always refresh to pick up local changes)
            worker_script_content = _REMOTE_DEIDENTIFY_WORKER
            worker_path = os.path.join(tempfile.gettempdir(), f"remote_deidentify_worker_{job_id}.py")
            with open(worker_path, "w") as wf:
                wf.write(worker_script_content)

            subprocess.run(
                _scp_base_args(cfg) + [worker_path, f"{cfg.host}:{remote_work}/remote_deidentify_worker.py"],
                capture_output=True, timeout=30,
            )
            logfile.write("  Uploaded remote_deidentify_worker.py\n")
            logfile.flush()

            # Launch the worker:
            # - bundle_path → remote_run/bundle.json
            # - work_dir    → remote_work (where deidentify.py, npz, videos live)
            # - output_dir  → remote_run/output
            # - status_file → remote_run/status.json
            launch_script = (
                f"\"import subprocess, os, time; "
                f"os.makedirs(r'{remote_output_dir}', exist_ok=True); "
                f"log_fh = open(r'{remote_log_file}', 'w'); "
                f"args = [r'{cfg.python_executable}', '-u', "
                f"r'{remote_work}/remote_deidentify_worker.py', "
                f"r'{remote_run}/bundle.json', "
                f"r'{remote_work}', "
                f"r'{remote_output_dir}', "
                f"r'{remote_status_file}']; "
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
                _fail(f"Failed to start remote render: {result.stderr.strip()[:200]}")
                return

            try:
                remote_pid = int(result.stdout.strip().splitlines()[-1])
            except (ValueError, IndexError):
                _fail(f"Could not parse remote PID: {result.stdout.strip()[:100]}")
                return

            _update_progress(30)
            logfile.write(f"=== Remote render started (PID {remote_pid}) ===\n")
            logfile.flush()

            # ── Phase 3: Monitor ───────────────────────────────
            log_offset = 0
            ssh_fail_count = 0

            while True:
                _check_cancel()
                time.sleep(10)
                _check_cancel()

                status = _read_remote_status(cfg, remote_status_file)
                proc_alive = _check_remote_pid_alive(cfg, remote_pid)

                new_log, log_offset = _tail_remote_log(cfg, remote_log_file, log_offset)
                if new_log:
                    logfile.write(new_log)
                    logfile.flush()
                    ssh_fail_count = 0

                if status:
                    ssh_fail_count = 0
                    pct = status.get("progress_pct", 0)
                    scaled = 30 + (pct / 100.0) * 55  # 30-85%
                    _update_progress(scaled)

                    if status.get("status") == "completed":
                        logfile.write("=== Remote render completed ===\n")
                        logfile.flush()
                        break
                    elif status.get("status") == "failed":
                        _fail(f"Remote render failed: {status.get('error', 'unknown')}")
                        return
                    elif not proc_alive:
                        # Process died — re-read status one more time (it may have
                        # written "completed" between our last check and death)
                        time.sleep(2)
                        final_status = _read_remote_status(cfg, remote_status_file)
                        if final_status and final_status.get("status") == "completed":
                            logfile.write("=== Remote render completed (detected after process exit) ===\n")
                            logfile.flush()
                            break
                        _fail(f"Remote render process died: {(final_status or status).get('error', 'unknown')}")
                        return
                elif not proc_alive:
                    ssh_fail_count += 1
                    # Before giving up, check if status.json appeared
                    if ssh_fail_count > 5:
                        late_status = _read_remote_status(cfg, remote_status_file)
                        if late_status and late_status.get("status") == "completed":
                            logfile.write("=== Remote render completed (late status detection) ===\n")
                            logfile.flush()
                            break
                    if ssh_fail_count > 20:
                        _fail("Remote process not found and no status updates")
                        return
                else:
                    ssh_fail_count += 1
                    if ssh_fail_count > 30:
                        _fail("Lost contact with remote host")
                        return

            # ── Phase 4: Download results ──────────────────────
            _update_progress(87)
            logfile.write("=== Downloading rendered videos ===\n")
            logfile.flush()

            local_deident_dir = settings.video_path / "deidentified"
            os.makedirs(str(local_deident_dir), exist_ok=True)

            # List remote output files
            result = subprocess.run(
                _py_cmd(cfg, f"\"import os; d=r'{remote_output_dir}'; print('\\n'.join(os.listdir(d)) if os.path.isdir(d) else '')\""),
                capture_output=True, text=True, timeout=15,
            )
            remote_files = [f for f in result.stdout.strip().splitlines() if f.endswith('.mp4')]
            logfile.write(f"  Found {len(remote_files)} mp4 files in {remote_output_dir}\n")
            if result.stderr:
                logfile.write(f"  ls stderr: {result.stderr.strip()}\n")
            logfile.flush()

            for i, fname in enumerate(remote_files):
                _check_cancel()
                logfile.write(f"  Downloading {fname}...\n")
                logfile.flush()
                # Normalize path separators for SCP (forward slashes work on all platforms over SSH)
                remote_path = f"{remote_output_dir}/{fname}".replace('\\', '/')
                proc = subprocess.run(
                    _scp_base_args(cfg) + [f"{cfg.host}:{remote_path}", str(local_deident_dir / fname)],
                    capture_output=True, text=True, timeout=600,
                )
                if proc.returncode != 0:
                    logfile.write(f"  Warning: failed to download {fname}\n")
                    if proc.stderr:
                        logfile.write(f"    scp error: {proc.stderr.strip()}\n")
                    logfile.flush()
                pct = 87 + (i + 1) / max(len(remote_files), 1) * 12
                _update_progress(pct)

            _update_progress(100)
            logfile.write("=== Remote deidentify complete ===\n")
            logfile.flush()

            # In batch mode the queue manager owns the final completed/
            # failed status across all trials; skip the per-trial completion
            # write so trial 1 doesn't mark the whole batch finished.
            if not _is_batched:
                with get_db_ctx() as db:
                    db.execute(
                        """UPDATE jobs SET status = 'completed', progress_pct = 100,
                           finished_at = CURRENT_TIMESTAMP WHERE id = ?""",
                        (job_id,),
                    )

            # Clean up temp files
            try:
                os.unlink(bundle_path)
                os.unlink(worker_path)
            except OSError:
                pass

    except InterruptedError:
        logger.info(f"Job {job_id} cancelled")
        with get_db_ctx() as db:
            db.execute(
                """UPDATE jobs SET status = 'cancelled',
                   finished_at = CURRENT_TIMESTAMP WHERE id = ?""",
                (job_id,),
            )
    except Exception as e:
        logger.exception(f"Job {job_id} remote deidentify error")
        _fail(str(e))
    finally:
        registry._processes.pop(job_id, None)
        registry._threads.pop(job_id, None)
        registry.unregister_cancel_event(job_id)


def resume_remote_deidentify_monitor(
    job_id: int,
    cfg: RemoteConfig,
    subject_name: str,
    remote_pid: int,
    trial_idx: int | None,
    log_path: str,
    registry,
) -> None:
    """Reattach to a remote deidentify render after a local server restart.

    The remote worker is launched with ``DETACHED_PROCESS`` so it survives
    the local SSH session ending — what dies on local restart is just the
    poll-and-download loop that *watches* it.  This function reproduces
    the Phase 3 (monitor) + Phase 4 (download) of ``remote_deidentify``
    so a fresh thread can pick up the in-flight render and finish it.

    Path layout matches what ``remote_deidentify`` creates::

        <work>/deidentify_<subject>/run_<job_id>/status.json
                                                /render.log
                                                /output/<trial>.mp4
    """
    from ..config import get_settings

    settings = get_settings()
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logfile = open(log_path, "a", buffering=1)
    cancel_event = registry.register_cancel_event(job_id)

    remote_work = f"{cfg.work_dir}/deidentify_{subject_name}"
    remote_run = f"{remote_work}/run_{job_id}"
    remote_status_file = f"{remote_run}/status.json"
    remote_log_file = f"{remote_run}/render.log"
    remote_output_dir = f"{remote_run}/output"

    def _fail(msg: str) -> None:
        logfile.write(f"ERROR: {msg}\n"); logfile.flush()
        with get_db_ctx() as db:
            db.execute(
                "UPDATE jobs SET status='failed', error_msg=?, "
                "finished_at=CURRENT_TIMESTAMP WHERE id=?",
                (msg[:500], job_id),
            )

    def _update_progress(pct: float) -> None:
        with get_db_ctx() as db:
            db.execute(
                "UPDATE jobs SET progress_pct=? WHERE id=?",
                (round(pct, 1), job_id),
            )

    try:
        logfile.write(f"=== Resuming remote deidentify monitor (PID {remote_pid}) ===\n")
        logfile.flush()

        # Phase 3: Monitor (mirrors remote_deidentify's loop).
        log_offset = 0
        ssh_fail_count = 0
        completed = False
        while True:
            if cancel_event.is_set():
                logfile.write("Local cancel — leaving remote alive\n")
                logfile.flush()
                return
            time.sleep(8)
            status = _read_remote_status(cfg, remote_status_file)
            proc_alive = _check_remote_pid_alive(cfg, remote_pid) if remote_pid else False
            new_log, log_offset = _tail_remote_log(cfg, remote_log_file, log_offset)
            if new_log:
                logfile.write(new_log); logfile.flush()
                ssh_fail_count = 0

            if status:
                ssh_fail_count = 0
                pct = float(status.get("progress_pct", 0))
                _update_progress(30 + (pct / 100.0) * 55)
                if status.get("status") == "completed":
                    completed = True
                    break
                if status.get("status") == "failed":
                    _fail(f"Remote: {status.get('error', 'unknown')}")
                    return
                if not proc_alive:
                    time.sleep(2)
                    final_status = _read_remote_status(cfg, remote_status_file)
                    if final_status and final_status.get("status") == "completed":
                        completed = True
                        break
                    _fail(f"Remote render died: {(final_status or status).get('error', 'unknown')}")
                    return
            elif not proc_alive:
                ssh_fail_count += 1
                if ssh_fail_count > 20:
                    _fail("Remote process not found and no status updates")
                    return
            else:
                ssh_fail_count += 1
                if ssh_fail_count > 30:
                    _fail("Lost contact with remote host")
                    return

        if not completed:
            return

        # Phase 4: Download.
        _update_progress(87)
        local_deident_dir = settings.video_path / "deidentified"
        os.makedirs(str(local_deident_dir), exist_ok=True)
        result = subprocess.run(
            _py_cmd(cfg,
                    f"\"import os; d=r'{remote_output_dir}'; print('\\n'.join(os.listdir(d)) if os.path.isdir(d) else '')\""),
            capture_output=True, text=True, timeout=15,
        )
        remote_files = [f for f in result.stdout.strip().splitlines() if f.endswith(".mp4")]
        logfile.write(f"=== Downloading {len(remote_files)} rendered file(s) ===\n")
        logfile.flush()
        for i, fname in enumerate(remote_files):
            remote_path = f"{remote_output_dir}/{fname}".replace("\\", "/")
            proc = subprocess.run(
                _scp_base_args(cfg) + [f"{cfg.host}:{remote_path}",
                                         str(local_deident_dir / fname)],
                capture_output=True, text=True, timeout=600,
            )
            if proc.returncode != 0:
                logfile.write(f"  Warning: download failed for {fname}: {proc.stderr[-200:]}\n")
                logfile.flush()
            _update_progress(87 + (i + 1) / max(len(remote_files), 1) * 12)

        _update_progress(100)
        logfile.write("=== Resume monitor complete ===\n"); logfile.flush()
        with get_db_ctx() as db:
            db.execute(
                "UPDATE jobs SET status='completed', progress_pct=100, "
                "finished_at=CURRENT_TIMESTAMP WHERE id=?",
                (job_id,),
            )
    except Exception as e:
        logger.exception(f"Job {job_id} resume_remote_deidentify_monitor error")
        _fail(str(e))
    finally:
        logfile.close()
        registry._threads.pop(job_id, None)


def remote_deidentify_download(
    job_id: int,
    cfg,
    subject_name: str,
    log_path: str,
    registry,
):
    """Re-download deidentified videos from the most recent remote run for a subject."""
    import os
    from ..db import get_db_ctx
    from ..config import get_settings

    settings = get_settings()

    try:
        with open(log_path, "w") as logfile:
            logfile.write(f"=== Re-downloading deidentify results for {subject_name} ===\n")

            remote_work = f"{cfg.work_dir}/deidentify_{subject_name}"

            # Find the latest run directory
            result = subprocess.run(
                _py_cmd(cfg, f"\"import os; d=r'{remote_work}'; dirs=[x for x in os.listdir(d) if x.startswith('run_')] if os.path.isdir(d) else []; dirs.sort(); print('\\n'.join(dirs))\""),
                capture_output=True, text=True, timeout=15,
            )
            runs = [r for r in result.stdout.strip().splitlines() if r]
            if not runs:
                raise ValueError(f"No run directories found in {remote_work}")

            latest_run = runs[-1]
            remote_output_dir = f"{remote_work}/{latest_run}/output"
            logfile.write(f"  Latest run: {latest_run}\n")
            logfile.write(f"  Output dir: {remote_output_dir}\n")

            # List mp4 files
            result = subprocess.run(
                _py_cmd(cfg, f"\"import os; d=r'{remote_output_dir}'; print('\\n'.join(os.listdir(d)) if os.path.isdir(d) else '')\""),
                capture_output=True, text=True, timeout=15,
            )
            remote_files = [f for f in result.stdout.strip().splitlines() if f.endswith('.mp4')]
            logfile.write(f"  Found {len(remote_files)} mp4 files\n")
            logfile.flush()

            if not remote_files:
                raise ValueError("No mp4 files found in remote output directory")

            local_deident_dir = settings.video_path / "deidentified"
            os.makedirs(str(local_deident_dir), exist_ok=True)

            for i, fname in enumerate(remote_files):
                logfile.write(f"  Downloading {fname}...\n")
                logfile.flush()
                remote_path = f"{remote_output_dir}/{fname}".replace('\\', '/')
                proc = subprocess.run(
                    _scp_base_args(cfg) + [f"{cfg.host}:{remote_path}", str(local_deident_dir / fname)],
                    capture_output=True, text=True, timeout=600,
                )
                if proc.returncode != 0:
                    logfile.write(f"  Warning: failed to download {fname}\n")
                    if proc.stderr:
                        logfile.write(f"    scp error: {proc.stderr.strip()}\n")
                else:
                    logfile.write(f"  OK: {fname}\n")
                logfile.flush()

                with get_db_ctx() as db:
                    pct = (i + 1) / len(remote_files) * 100
                    db.execute("UPDATE jobs SET progress_pct = ? WHERE id = ?", (pct, job_id))

            logfile.write("=== Re-download complete ===\n")
            with get_db_ctx() as db:
                db.execute(
                    "UPDATE jobs SET status='completed', progress_pct=100, finished_at=CURRENT_TIMESTAMP WHERE id=?",
                    (job_id,),
                )

    except Exception as e:
        logger.exception(f"Re-download deidentify job {job_id} failed")
        with get_db_ctx() as db:
            db.execute(
                "UPDATE jobs SET status='failed', error_msg=?, finished_at=CURRENT_TIMESTAMP WHERE id=?",
                (str(e)[:500], job_id),
            )
    finally:
        registry._threads.pop(job_id, None)


# Standalone remote worker script for deidentify rendering.
# Uploaded to the remote host and executed there.
_REMOTE_DEIDENTIFY_WORKER = r'''#!/usr/bin/env python3
"""Remote deidentify worker — reads a JSON bundle and renders blurred videos.

Usage: python remote_deidentify_worker.py bundle.json work_dir output_dir status_file
"""
import json
import os
import shutil
import subprocess
import sys
import traceback
import types

import cv2
import numpy as np


def _write_status(path, **kwargs):
    """Atomically write status JSON.

    On Windows, ``os.replace`` raises PermissionError [WinError 5] if the
    destination has an open handle — which happens routinely when the local
    poller is fetching ``status.json`` via SCP/SFTP at the same moment the
    remote worker is replacing it.  Retry a few times with a small backoff;
    a stale status.json one tick old is far better than the worker
    crash-halting an entire trial.

    Also normalize the destination path so mixed forward/back slashes
    (which leak in when callers concatenate ``work_dir`` with a
    forward-slash literal) don't confuse Windows handle tracking.
    """
    import tempfile
    import time as _time
    path = os.path.normpath(path)
    d = os.path.dirname(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=d, suffix=".tmp", delete=False) as f:
        json.dump({**kwargs, "pid": os.getpid()}, f)
        tmp = f.name
    # Retry os.replace on Windows file-locking errors.  Up to ~2s total
    # (10 attempts, 50–500 ms exponential-ish backoff).
    last_exc = None
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError as e:
            last_exc = e
            _time.sleep(min(0.05 * (1 + attempt), 0.5))
        except OSError as e:
            # WinError 32 (sharing violation) shows up as OSError sometimes.
            last_exc = e
            _time.sleep(min(0.05 * (1 + attempt), 0.5))
    # Final attempt as last resort — if it still fails we silently drop
    # this status update rather than tearing the worker down.  The next
    # _write_status call (one tick later) will succeed.
    try:
        os.replace(tmp, path)
    except (PermissionError, OSError):
        try:
            os.unlink(tmp)
        except OSError:
            pass
        print(f"[worker] _write_status: giving up after retries: {last_exc}",
              flush=True)


def _patch_deidentify_imports(work_dir):
    """Create a fake ffmpeg module so deidentify.py's relative import works."""
    # Create a stub 'ffmpeg' module with get_ffmpeg_path
    ffmpeg_mod = types.ModuleType("ffmpeg")

    def _try_pip_install_imageio_ffmpeg(force_reinstall: bool = False):
        """One-shot self-install on the remote when ``imageio_ffmpeg`` is
        missing or its cached binary disappeared.  The package ships a
        small (~20 MB) ffmpeg binary -- far less friction than asking
        the user to put ffmpeg on PATH manually.

        ``force_reinstall=True`` runs ``pip install --force-reinstall``
        so the bundled binary is re-extracted under AppData even when
        the Python package is technically already importable.  Use
        this when ``get_ffmpeg_exe()`` returned a path that no longer
        exists (e.g. AV quarantined the file).

        Returns the module on success, None on failure.  Captures pip's
        stderr / stdout into the job log on failure so Windows-specific
        DLL-init crashes (e.g. exit code 0xC0000142 from Defender
        intercepting the child python) are debuggable.
        """
        import subprocess as _sp
        try:
            label = "force-reinstalling" if force_reinstall else "pip-installing"
            print(f"[worker] imageio_ffmpeg not found; {label} now...",
                  flush=True)
            cmd = [
                sys.executable, "-m", "pip", "install",
                "--disable-pip-version-check",
            ]
            if force_reinstall:
                cmd += ["--force-reinstall", "--no-deps"]
            cmd.append("imageio-ffmpeg")
            proc = _sp.run(cmd, capture_output=True, text=True, timeout=180)
            if proc.returncode != 0:
                _hex = f"0x{proc.returncode & 0xFFFFFFFF:08X}" if proc.returncode < 0 \
                    or proc.returncode > 0xFFFF else str(proc.returncode)
                print(f"[worker] pip exited {proc.returncode} ({_hex})",
                      flush=True)
                if proc.stdout:
                    print(f"[worker] pip stdout: {proc.stdout[-400:]}",
                          flush=True)
                if proc.stderr:
                    print(f"[worker] pip stderr: {proc.stderr[-400:]}",
                          flush=True)
                return None
            if force_reinstall and "imageio_ffmpeg" in sys.modules:
                del sys.modules["imageio_ffmpeg"]
            import imageio_ffmpeg
            print(f"[worker] imageio_ffmpeg installed at {imageio_ffmpeg.__file__}",
                  flush=True)
            return imageio_ffmpeg
        except Exception as _e:
            print(f"[worker] pip install imageio-ffmpeg failed: "
                  f"{type(_e).__name__}: {_e}", flush=True)
            return None

    def _ffmpeg_actually_works(path):
        """Smoke-test the binary by running ``ffmpeg -version`` and
        checking we got recognisable output AND a real-sized binary
        (a missing-DLL silent failure on Windows can leave us with
        an exit-0 stub that produced no output).  Returns True on
        success, prints a [worker] diagnostic + returns False on
        failure so the caller falls through to the next candidate."""
        try:
            st = os.stat(path)
        except OSError as _e:
            print(f"[worker] ffmpeg stat failed for {path}: {_e}", flush=True)
            return False
        # A 360 KB ffmpeg.exe wrapped a missing-DLL stub for the user
        # in May 2026.  Real ffmpeg builds are all >1 MB.  Use 1 MB as
        # a low-confidence floor.
        if st.st_size < 1_000_000:
            print(f"[worker] ffmpeg at {path} is suspiciously small "
                  f"({st.st_size} bytes) -- likely a broken stub.",
                  flush=True)
            return False
        try:
            proc = subprocess.run([path, "-version"], capture_output=True,
                                  timeout=10)
        except (OSError, subprocess.TimeoutExpired) as _e:
            print(f"[worker] {path} -version failed: {_e}", flush=True)
            return False
        out = (proc.stdout + proc.stderr).decode("utf-8", errors="replace")
        if "ffmpeg version" not in out.lower():
            print(f"[worker] {path} -version rc={proc.returncode} but "
                  f"output looks wrong ({len(out)} chars). First 200: "
                  f"{out[:200]!r}", flush=True)
            return False
        return True

    def _imageio_ffmpeg_env_override():
        """Honour ``IMAGEIO_FFMPEG_EXE`` if the user / admin set it on the
        remote.  Most reliable escape hatch when pip + imageio_ffmpeg's
        auto-download are both broken on a locked-down host."""
        exe = os.environ.get("IMAGEIO_FFMPEG_EXE")
        if exe and os.path.exists(exe):
            print(f"[worker] using IMAGEIO_FFMPEG_EXE={exe}", flush=True)
            return exe
        return None

    def _conda_env_ffmpeg():
        """Probe ``<sys.prefix>/Library/bin/ffmpeg.exe`` (Windows) or
        ``<sys.prefix>/bin/ffmpeg`` (Linux/macOS).  ``conda install
        ffmpeg`` puts the binary there, but the env's PATH isn't always
        active when the worker is launched as a detached subprocess --
        so ``shutil.which`` misses it.  Probing the prefix directly is
        the reliable recovery when the user has run ``conda install -n
        <env> ffmpeg`` on the host."""
        candidates = []
        if os.name == "nt":
            candidates.append(os.path.join(sys.prefix, "Library", "bin", "ffmpeg.exe"))
        else:
            candidates.append(os.path.join(sys.prefix, "bin", "ffmpeg"))
        for c in candidates:
            if os.path.exists(c):
                print(f"[worker] found ffmpeg via conda env: {c}", flush=True)
                return c
        return None

    def _imageio_ffmpeg_exe():
        """Return ``imageio_ffmpeg.get_ffmpeg_exe()`` only if the binary
        actually exists on disk.  Catches broader exceptions than
        ``(ImportError, RuntimeError)`` -- imageio_ffmpeg can raise
        ``OSError`` / ``PermissionError`` on concurrent first-use
        extraction, or return a path whose file was since deleted /
        AV-quarantined.  Returns ``None`` on any failure and prints a
        one-line diagnostic so intermittent failures are debuggable
        from the job log."""
        try:
            import imageio_ffmpeg
        except Exception as _e:
            print(f"[worker] imageio_ffmpeg import failed: "
                  f"{type(_e).__name__}: {_e}", flush=True)
            return None
        try:
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as _e:
            print(f"[worker] imageio_ffmpeg.get_ffmpeg_exe() raised: "
                  f"{type(_e).__name__}: {_e}", flush=True)
            return None
        if not exe or not os.path.exists(exe):
            print(f"[worker] imageio_ffmpeg returned {exe!r} but the "
                  f"path doesn't exist on disk", flush=True)
            return None
        return exe

    def get_ffmpeg_path():
        # Try 1: IMAGEIO_FFMPEG_EXE env var (explicit user override).
        env_exe = _imageio_ffmpeg_env_override()
        if env_exe and _ffmpeg_actually_works(env_exe):
            return env_exe
        # Try 2: ffmpeg on PATH (after a working check).
        p = shutil.which("ffmpeg")
        if p and os.path.exists(p) and _ffmpeg_actually_works(p):
            return p
        # Try 3: conda env's Library/bin/ffmpeg.exe -- present after
        # ``conda install -n <env> ffmpeg`` even when not on PATH.
        conda_exe = _conda_env_ffmpeg()
        if conda_exe and _ffmpeg_actually_works(conda_exe):
            return conda_exe
        # Try 4: imageio_ffmpeg's bundled / downloaded binary.
        exe = _imageio_ffmpeg_exe()
        if exe and _ffmpeg_actually_works(exe):
            return exe
        # Try 5: pip-install imageio_ffmpeg + retry.
        if _try_pip_install_imageio_ffmpeg() is not None:
            exe = _imageio_ffmpeg_exe()
            if exe and _ffmpeg_actually_works(exe):
                return exe
        # Try 6: --force-reinstall so the bundled binary is re-extracted.
        # Covers AV-quarantined-binary + importable-package case.
        if _try_pip_install_imageio_ffmpeg(force_reinstall=True) is not None:
            exe = _imageio_ffmpeg_exe()
            if exe and _ffmpeg_actually_works(exe):
                return exe
        raise FileNotFoundError(
            "ffmpeg not found on remote and all auto-recovery attempts failed.  "
            "Easiest fix on this Windows host: run "
            f"'conda install -n <env> -y ffmpeg' (the env containing "
            f"'{sys.executable}'), or set IMAGEIO_FFMPEG_EXE to an existing "
            "ffmpeg.exe.  See the [worker] lines above for the underlying "
            "cause (pip exit code, get_ffmpeg_exe() error, etc.)."
        )

    def get_ffprobe_path():
        # Try 1: ffprobe on PATH.
        p = shutil.which("ffprobe")
        if p and os.path.exists(p):
            return p
        from pathlib import Path as _Path
        # Try 1b: conda env's Library/bin/ffprobe.exe.
        if os.name == "nt":
            _conda_probe = os.path.join(sys.prefix, "Library", "bin", "ffprobe.exe")
        else:
            _conda_probe = os.path.join(sys.prefix, "bin", "ffprobe")
        if os.path.exists(_conda_probe):
            return _conda_probe
        # Try 2: derive ffprobe from imageio_ffmpeg's ffmpeg path.
        exe = _imageio_ffmpeg_exe()
        if exe:
            fp = _Path(exe)
            probe = fp.parent / fp.name.replace("ffmpeg", "ffprobe")
            if probe.exists():
                return str(probe)
        # Try 3: pip-install + retry.
        if _try_pip_install_imageio_ffmpeg() is not None:
            exe = _imageio_ffmpeg_exe()
            if exe:
                fp = _Path(exe)
                probe = fp.parent / fp.name.replace("ffmpeg", "ffprobe")
                if probe.exists():
                    return str(probe)
        # Final fallback: --force-reinstall to re-extract the bundled
        # binary if AV / file-lock / manual delete removed it.
        if _try_pip_install_imageio_ffmpeg(force_reinstall=True) is not None:
            exe = _imageio_ffmpeg_exe()
            if exe:
                fp = _Path(exe)
                probe = fp.parent / fp.name.replace("ffmpeg", "ffprobe")
                if probe.exists():
                    return str(probe)
        raise FileNotFoundError(
            "ffprobe not found on remote.  Put ffprobe on PATH or run "
            f"'{sys.executable}' -m pip install imageio-ffmpeg"
        )

    ffmpeg_mod.get_ffmpeg_path = get_ffmpeg_path
    ffmpeg_mod.get_ffprobe_path = get_ffprobe_path
    sys.modules["ffmpeg"] = ffmpeg_mod

    # Also patch so relative import from deidentify works
    # deidentify.py does "from .ffmpeg import get_ffmpeg_path"
    # We need to make deidentify think it's in a package
    # Simplest: rewrite the import in deidentify.py before importing
    deid_path = os.path.join(work_dir, "deidentify.py")
    if os.path.exists(deid_path):
        with open(deid_path, "r") as f:
            src = f.read()
        # Replace relative imports with absolute
        src = src.replace("from .ffmpeg import", "from ffmpeg import")
        src = src.replace("from .calibration import", "# from .calibration import")
        with open(deid_path, "w") as f:
            f.write(src)


def main(bundle_path, work_dir, output_dir, status_file):
    _write_status(status_file, status="running", phase="init", progress_pct=0)

    try:
        # Patch imports before loading deidentify module
        _patch_deidentify_imports(work_dir)
        sys.path.insert(0, work_dir)

        with open(bundle_path) as f:
            bundle = json.load(f)

        subject_name = bundle["subject_name"]
        trial_list = bundle["trials"]

        # Load mediapipe + pose landmarks if available.
        # Compute frame offset: npz may have extra pre-roll frames if built on a
        # machine whose codec exposed negative-PTS frames that browsers skip.
        mp_data = None
        mp_path = os.path.join(work_dir, "mediapipe_prelabels.npz")
        if os.path.exists(mp_path):
            mp_data = dict(np.load(mp_path, allow_pickle=True))
            expected_total = bundle.get("total_frame_count", 0)
            if expected_total > 0:
                stored_total = int(mp_data.get("total_frames", 0))
                mp_offset = max(0, stored_total - expected_total)
                if mp_offset > 0:
                    print(f"Trimming {mp_offset} pre-roll frame(s) from mediapipe landmarks", flush=True)
                    for key in ("OS_landmarks", "OD_landmarks"):
                        if key in mp_data and hasattr(mp_data[key], '__len__'):
                            mp_data[key] = mp_data[key][mp_offset:]

        pose_data = None
        pose_path = os.path.join(work_dir, "pose_prelabels.npz")
        if os.path.exists(pose_path):
            pose_data = dict(np.load(pose_path, allow_pickle=True))
            expected_total = bundle.get("total_frame_count", 0)
            if expected_total > 0:
                stored_total = int(pose_data.get("total_frames", 0))
                pose_offset = max(0, stored_total - expected_total)
                if pose_offset > 0:
                    for key in ("OS_pose", "OD_pose"):
                        if key in pose_data and hasattr(pose_data[key], '__len__'):
                            pose_data[key] = pose_data[key][pose_offset:]

        trials_with_specs = [t for t in trial_list if t.get("blur_specs")]
        n_trials = len(trials_with_specs)
        if n_trials == 0:
            _write_status(status_file, status="completed", progress_pct=100)
            return

        from deidentify import render_with_blur_specs

        for ti_pos, trial in enumerate(trials_with_specs):
            trial_name = trial["trial_name"]
            video_name = trial["video_name"]
            video_path = os.path.join(work_dir, video_name)

            if not os.path.exists(video_path):
                print(f"Warning: video not found: {video_path}", flush=True)
                continue

            output_path = os.path.join(output_dir, f"{trial_name}.mp4")

            _write_status(status_file, status="running", phase=f"render_{trial_name}",
                         progress_pct=ti_pos / n_trials * 100,
                         current_trial=trial_name)

            base_pct = ti_pos / n_trials * 100
            span = 100.0 / n_trials

            def progress_cb(pct, _base=base_pct, _span=span):
                overall = _base + (pct / 100.0) * _span
                _write_status(status_file, status="running", phase=f"render_{trial_name}",
                             progress_pct=overall, current_trial=trial_name)
                print(f"PROGRESS:{overall:.1f}", flush=True)

            render_with_blur_specs(
                input_path=video_path,
                output_path=output_path,
                blur_specs=trial["blur_specs"],
                hand_settings=trial.get("hand_settings"),
                face_detections=trial.get("face_detections", []),
                subject_name=subject_name,
                start_frame=trial.get("start_frame", 0),
                frame_count=trial.get("frame_count"),
                progress_callback=progress_cb,
                mp_data=mp_data,
                pose_data=pose_data,
            )

            print(f"Rendered {trial_name}", flush=True)

        _write_status(status_file, status="completed", progress_pct=100)
        print("=== All trials rendered ===", flush=True)

    except Exception as e:
        traceback.print_exc()
        _write_status(status_file, status="failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(f"Usage: {sys.argv[0]} bundle.json work_dir output_dir status_file")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
'''


# ─── Remote preproc (camera trajectory + background/mask bake) ─────────


def remote_preproc(
    job_id: int,
    cfg: RemoteConfig,
    subject_name: str,
    log_path: str,
    registry,
    trials: list[dict],
) -> None:
    """Remote preproc: per-trial camera-trajectory + background/mask bake.

    Layout under ``cfg.work_dir/preproc_<subject>/``:
      videos/                 — source mp4s (skip-if-already-uploaded)
      <subject>/mediapipe_prelabels.npz  — copied from local
      modules/                — uploaded source files (camera_motion.py,
                                 background.py, ffmpeg.py, plus stubs)
      run_<job_id>/bundle.json, status.json, render.log

    Outputs the worker writes (each trial):
      <subject>/preproc/<trial_stem>/{camera_trajectory.npz, background.npz,
        stable.mp4, fg.mp4, outline.mp4, background_*.png, mad_*.png}

    Each output dir is downloaded back to the local
    ``<dlc>/<subject>/preproc/<trial_stem>/``.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    from ..config import get_settings
    from .video import build_trial_map

    settings = get_settings()
    remote_work = f"{cfg.work_dir}/preproc_{subject_name}"
    remote_run  = f"{remote_work}/run_{job_id}"
    remote_status = f"{remote_run}/status.json"
    remote_log    = f"{remote_run}/render.log"

    cancel_event = registry.register_cancel_event(job_id)

    def _check_cancel():
        if cancel_event.is_set():
            raise InterruptedError("Job cancelled")

    def _set_progress(pct):
        with get_db_ctx() as db:
            db.execute("UPDATE jobs SET progress_pct=? WHERE id=?",
                       (round(float(pct), 1), job_id))

    from .job_history import stage_timer, add_stage, finalize_job_record

    def _fail(msg):
        logger.error(f"Job {job_id} remote preproc failed: {msg}")
        with get_db_ctx() as db:
            db.execute(
                "UPDATE jobs SET status='failed', error_msg=?, "
                "finished_at=CURRENT_TIMESTAMP WHERE id=?",
                (msg, job_id),
            )
        finalize_job_record(job_id)

    try:
        local_dlc_dir = settings.dlc_path / subject_name
        all_trials = build_trial_map(subject_name)
        if not all_trials:
            _fail("No trials found locally")
            return

        # Filter to the trial indices the caller wants.  Each entry in
        # ``trials`` is ``{trial_idx, trial_name?, subject_name?}``.
        wanted_idxs = []
        for t in trials:
            ti = t.get("trial_idx")
            if ti is None:
                continue
            ti = int(ti)
            if 0 <= ti < len(all_trials):
                wanted_idxs.append(ti)
        if not wanted_idxs:
            _fail("No valid trial indices in request")
            return

        # Build the trial entries the remote worker needs.  We bake the
        # video filename (not the local absolute path) because the file
        # lives under the remote work dir.
        bundle_trials = []
        for ti in wanted_idxs:
            tr = all_trials[ti]
            bundle_trials.append({
                "trial_idx": int(ti),
                "trial_name": tr["trial_name"],
                "video_name": Path(tr["video_path"]).name,
                "frame_count": int(tr["frame_count"]),
                "start_frame": int(tr.get("start_frame", 0)),
                "fps": float(tr.get("fps", 30.0)),
            })

        with open(log_path, "w") as logfile:
            logfile.write(f"=== Remote preproc for {subject_name} "
                          f"({len(bundle_trials)} trials) ===\n")
            logfile.flush()

            # ── Phase 1: create remote dirs, probe existing files ────
            subprocess.run(
                _py_cmd(cfg, f"\"import os; os.makedirs(r'{remote_run}', exist_ok=True); "
                              f"os.makedirs(r'{remote_work}/{subject_name}', exist_ok=True); "
                              f"os.makedirs(r'{remote_work}/modules', exist_ok=True)\""),
                capture_output=True, timeout=20,
            )
            # List existing files under remote_work for skip-if-present.
            existing = set()
            res = subprocess.run(
                _py_cmd(cfg,
                    f"\"import os; d=r'{remote_work}'; "
                    f"out=[]; \\\n"
                    f"  [out.extend([f'{{root}}/{{f}}' for f in files]) "
                    f"for root,_,files in os.walk(d)]; print('\\n'.join(out))\""),
                capture_output=True, text=True, timeout=30,
            )
            if res.returncode == 0:
                existing = set(res.stdout.strip().splitlines())
            logfile.write(f"  remote dir scan: {len(existing)} files already present\n")
            logfile.flush()
            _check_cancel()
            _set_progress(2)

            # ── Phase 2: upload source videos that aren't already there ──
            video_path_map = {}   # video_name → remote absolute path
            videos_dir_remote = f"{remote_work}/videos"
            subprocess.run(
                _py_cmd(cfg, f"\"import os; os.makedirs(r'{videos_dir_remote}', exist_ok=True)\""),
                capture_output=True, timeout=15,
            )
            _videos_t0 = time.perf_counter()
            _bytes_uploaded = 0
            _videos_uploaded = 0
            _videos_reused = 0
            for bt in bundle_trials:
                vname = bt["video_name"]
                remote_video = f"{videos_dir_remote}/{vname}"
                video_path_map[vname] = remote_video
                # Also check the legacy deidentify_<subject> work dir —
                # that's where Render/Deidentify jobs have already
                # uploaded videos for the same subject.
                deidentify_video = f"{cfg.work_dir}/deidentify_{subject_name}/{vname}"
                # Probe both candidate locations on the remote.
                probe = subprocess.run(
                    _py_cmd(cfg,
                        f"\"import os; "
                        f"print('A' if os.path.exists(r'{remote_video}') else "
                        f"('B' if os.path.exists(r'{deidentify_video}') else 'N'))\""),
                    capture_output=True, text=True, timeout=15,
                )
                where = (probe.stdout or "").strip()
                if where == "A":
                    logfile.write(f"  video {vname}: already in preproc dir\n")
                    _videos_reused += 1
                elif where == "B":
                    # Cross-link to avoid duplicating: just point the
                    # bundle at the deidentify location.
                    video_path_map[vname] = deidentify_video
                    logfile.write(f"  video {vname}: reusing deidentify upload\n")
                    _videos_reused += 1
                else:
                    # Upload from local
                    local_vp = str(settings.video_path / vname)
                    if not os.path.exists(local_vp):
                        _fail(f"Local video not found: {local_vp}")
                        return
                    _file_size = os.path.getsize(local_vp)
                    logfile.write(f"  uploading {vname} ({_file_size/1e6:.1f} MB)...\n")
                    logfile.flush()
                    up = subprocess.run(
                        _scp_base_args(cfg) + [local_vp, f"{cfg.host}:{remote_video}"],
                        capture_output=True, text=True, timeout=600,
                    )
                    if up.returncode != 0:
                        _fail(f"Video upload failed: {up.stderr[:200]}")
                        return
                    _bytes_uploaded += _file_size
                    _videos_uploaded += 1
                _check_cancel()
            add_stage(job_id, "upload_videos",
                       time.perf_counter() - _videos_t0,
                       outcome="ok", uploaded=_videos_uploaded,
                       reused=_videos_reused, bytes=int(_bytes_uploaded))
            _set_progress(15)

            # ── Phase 3: upload mediapipe + pose npz if available ────
            _npz_t0 = time.perf_counter()
            _npz_bytes = 0
            _npz_uploaded = 0
            for npz_name in ("mediapipe_prelabels.npz", "pose_prelabels.npz"):
                local_npz = local_dlc_dir / npz_name
                if not local_npz.exists():
                    logfile.write(f"  {npz_name} not present locally — skipping\n")
                    continue
                remote_npz = f"{remote_work}/{subject_name}/{npz_name}"
                # Probe; skip if present and the local file is the same size.
                probe = subprocess.run(
                    _py_cmd(cfg,
                        f"\"import os; "
                        f"print(os.path.getsize(r'{remote_npz}') if "
                        f"os.path.exists(r'{remote_npz}') else 0)\""),
                    capture_output=True, text=True, timeout=15,
                )
                remote_size = 0
                try: remote_size = int((probe.stdout or "0").strip())
                except ValueError: remote_size = 0
                local_size = local_npz.stat().st_size
                if remote_size == local_size:
                    logfile.write(f"  {npz_name}: already up to date on remote\n")
                else:
                    logfile.write(f"  uploading {npz_name} ({local_size/1e6:.1f} MB)...\n")
                    logfile.flush()
                    up = subprocess.run(
                        _scp_base_args(cfg) + [str(local_npz), f"{cfg.host}:{remote_npz}"],
                        capture_output=True, text=True, timeout=600,
                    )
                    if up.returncode != 0:
                        logfile.write(f"  WARN: {npz_name} upload failed: {up.stderr[:200]}\n")
                    else:
                        _npz_bytes += local_size
                        _npz_uploaded += 1
                _check_cancel()
            add_stage(job_id, "upload_npz",
                       time.perf_counter() - _npz_t0,
                       outcome="ok", uploaded=_npz_uploaded,
                       bytes=int(_npz_bytes))
            _set_progress(25)

            # ── Phase 4: upload source modules (always refresh) ──────
            _mods_t0 = time.perf_counter()
            service_dir = Path(__file__).parent
            modules_dir = f"{remote_work}/modules"
            for mod_file in ("camera_motion.py", "background.py", "ffmpeg.py"):
                mp = service_dir / mod_file
                if not mp.exists():
                    continue
                up = subprocess.run(
                    _scp_base_args(cfg) + [str(mp), f"{cfg.host}:{modules_dir}/"],
                    capture_output=True, text=True, timeout=60,
                )
                if up.returncode != 0:
                    logfile.write(f"  WARN: module {mod_file} upload failed\n")
                else:
                    logfile.write(f"  uploaded module {mod_file}\n")
                logfile.flush()
            _check_cancel()
            add_stage(job_id, "upload_modules",
                       time.perf_counter() - _mods_t0, outcome="ok")
            _set_progress(30)

            # ── Phase 5: write + upload bundle and worker script ─────
            import tempfile as _tf
            bundle = {
                "subject_name": subject_name,
                "modules_dir":  modules_dir,
                "data_dir":     remote_work,           # holds <subject>/mediapipe_prelabels.npz
                "trials": [
                    {**bt, "video_remote_path": video_path_map[bt["video_name"]]}
                    for bt in bundle_trials
                ],
            }
            bundle_path = os.path.join(_tf.gettempdir(), f"preproc_bundle_{job_id}.json")
            with open(bundle_path, "w") as bf:
                json.dump(bundle, bf)
            up = subprocess.run(
                _scp_base_args(cfg) + [bundle_path, f"{cfg.host}:{remote_run}/bundle.json"],
                capture_output=True, text=True, timeout=30,
            )
            if up.returncode != 0:
                _fail(f"Bundle upload failed: {up.stderr[:200]}")
                return

            worker_path = os.path.join(_tf.gettempdir(), f"remote_preproc_worker_{job_id}.py")
            with open(worker_path, "w") as wf:
                wf.write(_REMOTE_PREPROC_WORKER)
            subprocess.run(
                _scp_base_args(cfg) + [worker_path,
                                        f"{cfg.host}:{remote_work}/remote_preproc_worker.py"],
                capture_output=True, timeout=30,
            )
            logfile.write("  uploaded worker script\n"); logfile.flush()
            _check_cancel()
            _set_progress(35)

            # ── Phase 6: launch the worker as a detached process ─────
            launch = (
                f"\"import subprocess, os, time; "
                f"log_fh = open(r'{remote_log}', 'w'); "
                f"args = [r'{cfg.python_executable}', '-u', "
                f"r'{remote_work}/remote_preproc_worker.py', "
                f"r'{remote_run}/bundle.json', "
                f"r'{remote_status}']; "
                f"flags = 0x01000200 if os.name == 'nt' else 0; "
                f"p = subprocess.Popen(args, creationflags=flags, "
                f"stdin=subprocess.DEVNULL, stdout=log_fh, stderr=log_fh); "
                f"print(p.pid); time.sleep(2)\""
            )
            launch_res = subprocess.run(_py_cmd(cfg, launch),
                                         capture_output=True, text=True, timeout=30)
            if launch_res.returncode != 0:
                _fail(f"Worker launch failed: {launch_res.stderr[:200]}")
                return
            logfile.write(f"  worker started (pid {launch_res.stdout.strip()})\n")
            logfile.flush()

            # ── Phase 7: poll status.json ────────────────────────────
            import time as _t
            _bake_t0 = time.perf_counter()
            _seen_trials: set[str] = set()
            _trial_t0_by_name: dict[str, float] = {}
            _trial_phase_by_name: dict[str, str] = {}
            poll_iv = 3.0
            last_pct = 35.0
            while True:
                _check_cancel()
                _t.sleep(poll_iv)
                # Pull status.json
                local_status = os.path.join(_tf.gettempdir(),
                                              f"preproc_status_{job_id}.json")
                _dl = subprocess.run(
                    _scp_base_args(cfg) + [f"{cfg.host}:{remote_status}", local_status],
                    capture_output=True, text=True, timeout=30,
                )
                if _dl.returncode != 0 or not os.path.exists(local_status):
                    continue   # status.json not written yet
                try:
                    with open(local_status) as f:
                        st = json.load(f)
                except (OSError, json.JSONDecodeError):
                    continue
                phase = st.get("status", "running")
                pct = float(st.get("progress_pct", last_pct))
                # Map worker 0–100 into our 35–95 window so upload + download
                # phases get visible progress around it.
                shown = 35.0 + 0.60 * pct
                if abs(shown - last_pct) > 0.5:
                    _set_progress(shown)
                    last_pct = shown
                cur = st.get("current_trial")
                cur_phase = st.get("phase")    # "trajectory" | "background" | "setup"
                if cur and cur_phase:
                    # Track per-(trial, phase) timings: stamp a t0 on
                    # first sight, close out on transition.
                    key = f"{cur}::{cur_phase}"
                    prev_phase = _trial_phase_by_name.get(cur)
                    if prev_phase and prev_phase != cur_phase:
                        # Previous phase wrapped — emit a stage record.
                        prev_key = f"{cur}::{prev_phase}"
                        if prev_key in _trial_t0_by_name:
                            add_stage(job_id,
                                       f"compute_{prev_phase}",
                                       time.perf_counter() - _trial_t0_by_name.pop(prev_key),
                                       outcome="ok", trial=cur, target="remote")
                    if key not in _trial_t0_by_name:
                        _trial_t0_by_name[key] = time.perf_counter()
                        _seen_trials.add(cur)
                    _trial_phase_by_name[cur] = cur_phase
                if cur:
                    logfile.write(f"  progress: trial={cur} {pct:.1f}%\n")
                    logfile.flush()
                if phase == "completed":
                    # Flush whichever stage was still open at the moment
                    # the worker called itself done.
                    for cur, last_phase in _trial_phase_by_name.items():
                        key = f"{cur}::{last_phase}"
                        if key in _trial_t0_by_name:
                            add_stage(job_id, f"compute_{last_phase}",
                                       time.perf_counter() - _trial_t0_by_name.pop(key),
                                       outcome="ok", trial=cur, target="remote")
                    add_stage(job_id, "remote_bake_total",
                               time.perf_counter() - _bake_t0,
                               outcome="ok", n_trials=len(_seen_trials))
                    logfile.write("  worker completed\n")
                    break
                if phase == "failed":
                    add_stage(job_id, "remote_bake_total",
                               time.perf_counter() - _bake_t0,
                               outcome="failed", n_trials=len(_seen_trials))
                    _fail(f"Worker failed: {st.get('error', 'unknown')}")
                    return
                if phase == "cancelled":
                    add_stage(job_id, "remote_bake_total",
                               time.perf_counter() - _bake_t0,
                               outcome="cancelled", n_trials=len(_seen_trials))
                    _fail("Worker cancelled")
                    return

            _set_progress(95)

            # ── Phase 8: download outputs back ───────────────────────
            _dl_t0 = time.perf_counter()
            local_preproc_root = settings.dlc_path / subject_name / "preproc"
            local_preproc_root.mkdir(parents=True, exist_ok=True)
            n_ok = 0
            for bt in bundle_trials:
                stem = bt["trial_name"]
                remote_stem_dir = f"{remote_work}/{subject_name}/preproc/{stem}/"
                local_stem_dir  = local_preproc_root / stem
                local_stem_dir.mkdir(parents=True, exist_ok=True)
                _trial_dl_t0 = time.perf_counter()
                logfile.write(f"  downloading {stem}/\n"); logfile.flush()
                _dl = subprocess.run(
                    _scp_base_args(cfg) + ["-r", f"{cfg.host}:{remote_stem_dir}",
                                            str(local_stem_dir.parent) + os.sep],
                    capture_output=True, text=True, timeout=900,
                )
                if _dl.returncode != 0:
                    logfile.write(f"    WARN: download failed: {_dl.stderr[:200]}\n")
                    add_stage(job_id, "download_trial",
                               time.perf_counter() - _trial_dl_t0,
                               outcome="failed", trial=stem)
                else:
                    n_ok += 1
                    # Sum downloaded bytes for the trial dir (best-effort).
                    _dl_bytes = 0
                    try:
                        for f in local_stem_dir.iterdir():
                            if f.is_file():
                                _dl_bytes += f.stat().st_size
                    except OSError:
                        pass
                    add_stage(job_id, "download_trial",
                               time.perf_counter() - _trial_dl_t0,
                               outcome="ok", trial=stem,
                               bytes=int(_dl_bytes))
                _check_cancel()
            add_stage(job_id, "download_outputs",
                       time.perf_counter() - _dl_t0,
                       outcome="ok", n_trials=n_ok)
            logfile.write(f"  downloaded {n_ok}/{len(bundle_trials)} trial dirs\n")
            logfile.flush()

            with get_db_ctx() as db:
                db.execute(
                    "UPDATE jobs SET status='completed', progress_pct=100, "
                    "finished_at=CURRENT_TIMESTAMP WHERE id=?",
                    (job_id,),
                )
            finalize_job_record(job_id)

    except InterruptedError:
        logger.info(f"Job {job_id} remote preproc cancelled")
        with get_db_ctx() as db:
            db.execute(
                "UPDATE jobs SET status='cancelled', "
                "finished_at=CURRENT_TIMESTAMP WHERE id=?",
                (job_id,),
            )
        finalize_job_record(job_id)
    except Exception as e:
        logger.exception(f"Job {job_id} remote preproc exception: {e}")
        _fail(str(e)[:500])


# Worker script — uploaded to the remote and executed there.
_REMOTE_PREPROC_WORKER = r'''#!/usr/bin/env python3
"""Remote preproc worker — runs camera trajectory + background bake for trials.

Reads bundle.json (subject_name, modules_dir, data_dir, trials).  Imports
the uploaded camera_motion.py + background.py + ffmpeg.py modules from
modules_dir, rewriting their relative imports to flat imports as needed,
and runs the compute functions per trial.

Stubs ``config.get_settings`` to point dlc_path/video_path at data_dir, and
``video.build_trial_map`` to return the bundled trial map — that's enough
for compute_camera_trajectory and compute_background to resolve their
paths without ever reaching out to the original package.
"""
import json
import os
import sys
import time as _time
import traceback
import types
from pathlib import Path


def _write_status(path, **kwargs):
    """Atomic status write, retrying on Windows file-locking errors."""
    import tempfile
    path = os.path.normpath(path)
    d = os.path.dirname(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=d, suffix=".tmp", delete=False) as f:
        json.dump({**kwargs, "pid": os.getpid()}, f)
        tmp = f.name
    for attempt in range(10):
        try:
            os.replace(tmp, path); return
        except (PermissionError, OSError):
            _time.sleep(min(0.05 * (1 + attempt), 0.5))
    try: os.replace(tmp, path)
    except (PermissionError, OSError):
        try: os.unlink(tmp)
        except OSError: pass


def _rewrite_relative_imports(modules_dir):
    """In-place rewrite of ``from ..config / .video / ...`` to flat
    imports so the modules load when placed at sys.path[0]."""
    rewrites = [
        ("from ..config import",            "from config import"),
        ("from .config import",             "from config import"),
        ("from .ffmpeg import",             "from ffmpeg import"),
        ("from .video import",              "from video import"),
        ("from .camera_motion import",      "from camera_motion import"),
        ("from .background import",         "from background import"),
        ("from .mediapipe_prelabel import", "from mediapipe_prelabel import"),
        ("from .calibration import",        "from calibration import"),
        ("from ..services.",                "from "),
    ]
    for fname in os.listdir(modules_dir):
        if not fname.endswith(".py"):
            continue
        p = os.path.join(modules_dir, fname)
        try:
            with open(p, "r", encoding="utf-8") as f:
                src = f.read()
            new = src
            for old, repl in rewrites:
                new = new.replace(old, repl)
            if new != src:
                with open(p, "w", encoding="utf-8") as f:
                    f.write(new)
        except OSError:
            pass


def _install_stubs(data_dir, bundled_trials):
    """Install stub ``config`` + ``video`` + ``mediapipe_prelabel``
    modules in sys.modules so compute_X can resolve their imports.

    config.get_settings returns dlc_path = video_path = data_dir.  The
    compute functions read settings.dlc_path / <subject>/<file> to find
    the MP npz and to write outputs under preproc/<stem>/.

    video.build_trial_map returns the pre-baked trial map from the
    bundle (so compute_camera_trajectory doesn't need ffprobe on the
    remote).
    """
    data_path = Path(data_dir)
    settings = types.SimpleNamespace(
        dlc_path=data_path,
        video_path=data_path,
        calibration_path=data_path,
        camera_names=["OS", "OD"],
        default_camera_mode="stereo",
    )
    config_mod = types.ModuleType("config")
    config_mod.get_settings = lambda: settings
    sys.modules["config"] = config_mod

    # video.build_trial_map: return the bundle's pre-computed trial info
    # but with absolute remote video paths so cv2.VideoCapture finds them.
    # build_trial_map indexes by trial position; we re-build the full list
    # in order, dropping in the bundled entries by trial_idx and leaving
    # the rest as plausible stubs (compute functions only ever access the
    # indices we passed in).
    trial_by_idx = {int(t["trial_idx"]): t for t in bundled_trials}
    max_idx = max(trial_by_idx) if trial_by_idx else -1
    tmap = []
    for i in range(max_idx + 1):
        if i in trial_by_idx:
            t = trial_by_idx[i]
            tmap.append({
                "trial_name":  t["trial_name"],
                "video_path":  t["video_remote_path"],
                "frame_count": int(t["frame_count"]),
                "start_frame": int(t.get("start_frame", 0)),
                "fps":         float(t.get("fps", 30.0)),
            })
        else:
            tmap.append({"trial_name": f"trial_{i}", "video_path": "",
                          "frame_count": 0, "start_frame": 0, "fps": 30.0})
    video_mod = types.ModuleType("video")
    video_mod.build_trial_map = lambda subject_name, **_kw: tmap
    sys.modules["video"] = video_mod

    # mediapipe_prelabel.load_mediapipe_prelabels: minimal loader.
    import numpy as _np
    def _load_mp(subject_name):
        npz_path = data_path / subject_name / "mediapipe_prelabels.npz"
        if not npz_path.exists():
            return None
        d = _np.load(str(npz_path))
        out = {}
        for k in ("OS_landmarks", "OD_landmarks", "confidence_OS",
                  "confidence_OD", "total_frames"):
            if k in d.files:
                out[k] = d[k] if k != "total_frames" else int(d[k])
        return out
    mp_mod = types.ModuleType("mediapipe_prelabel")
    mp_mod.load_mediapipe_prelabels = _load_mp
    sys.modules["mediapipe_prelabel"] = mp_mod


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} bundle.json status.json", flush=True)
        sys.exit(1)
    bundle_path, status_file = sys.argv[1], sys.argv[2]
    with open(bundle_path) as f:
        bundle = json.load(f)

    subject_name = bundle["subject_name"]
    modules_dir  = bundle["modules_dir"]
    data_dir     = bundle["data_dir"]
    trials       = bundle["trials"]

    try:
        _write_status(status_file, status="running", phase="setup", progress_pct=0)

        _rewrite_relative_imports(modules_dir)
        sys.path.insert(0, modules_dir)
        _install_stubs(data_dir, trials)

        # Now safe to import the real compute modules.
        from camera_motion import compute_camera_trajectory
        from background    import compute_stable, compute_background

        n_total = len(trials)
        for i, t in enumerate(trials):
            tname = t["trial_name"]
            ti = int(t["trial_idx"])
            print(f"=== [{i+1}/{n_total}] {tname}: trajectory ===", flush=True)

            # Per-trial split: trajectory ~20%, stable ~60%, background ~20%.
            # Hand boundary is computed on demand from the UI; no extra bake.
            def _on_traj(pct, _i=i, _n=n_total):
                local = 100.0 * (_i + (pct / 100.0) * 0.20) / _n
                _write_status(status_file, status="running", phase="trajectory",
                              current_trial=tname, progress_pct=local)
            compute_camera_trajectory(
                subject_name, ti, progress_callback=_on_traj)

            print(f"=== [{i+1}/{n_total}] {tname}: stable ===", flush=True)
            def _on_stable(pct, _i=i, _n=n_total):
                local = 100.0 * (_i + 0.20 + (pct / 100.0) * 0.60) / _n
                _write_status(status_file, status="running", phase="stable",
                              current_trial=tname, progress_pct=local)
            compute_stable(
                subject_name, ti, progress_callback=_on_stable)

            print(f"=== [{i+1}/{n_total}] {tname}: background ===", flush=True)
            def _on_bg(pct, _i=i, _n=n_total):
                local = 100.0 * (_i + 0.80 + (pct / 100.0) * 0.20) / _n
                _write_status(status_file, status="running", phase="background",
                              current_trial=tname, progress_pct=local)
            compute_background(
                subject_name, ti, progress_callback=_on_bg)

        _write_status(status_file, status="completed", progress_pct=100)
        print("=== All trials done ===", flush=True)

    except Exception as e:
        traceback.print_exc()
        _write_status(status_file, status="failed",
                      error=f"{type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
'''
