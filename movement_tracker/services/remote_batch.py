"""Generalised remote batch-job framework.

Same architectural pattern as ``dispatch_remote_preproc_batch`` /
``poll_remote_preproc_batch`` / ``resume_remote_preproc_batch`` /
``_kill_remote_preproc_runner``, but factored so each job type only
has to supply a :class:`RemoteBatchJobSpec` describing its uploads,
worker script, and downloads — the dispatcher / poller / resume /
kill are job-type-agnostic.

REMOTE LAYOUT
-------------

The framework standardises on this directory structure on the
remote (one working dir per app deployment):

    <cfg.work_dir>/
        shared/
            modules/<job_type>/             ← Python sources the
                                              worker imports
            scripts/runner.py               ← one generic runner
                                              for ALL job types
            scripts/<job_type>_worker.py    ← per-job-type worker
        subjects/
            <subject>/
                videos/<v>.mp4
                mediapipe_prelabels.npz
                pose_prelabels.npz
                preproc/<trial>/{outputs}   ← outputs colocated
                                              with inputs per
                                              subject; job-type
                                              subdirs keep things
                                              tidy
                hrnet/<trial>/{outputs}     ← future job types
                                              extend here
        batches/<batch_id>/
            batch.json
            batch_status.json
            runner.log
            cancel.flag                     ← sentinel; runner
                                              checks between items

The legacy ``preproc_<subject>/`` / ``deidentify_<subject>/`` /
``hrnet_jobs/<sub>_<stem>/`` layouts coexist — old code paths keep
working until each job type is migrated.

PER-JOB-TYPE PLUG-INS
---------------------

A job type defines one :class:`RemoteBatchJobSpec` with:

* ``job_type``                    — string used in paths + DB
* ``worker_script_source``        — Python source for the per-item
                                    worker (replaces the legacy
                                    job-type-specific ``_REMOTE_*_WORKER``)
* ``shared_module_paths``         — local Path objects to ship into
                                    ``shared/modules/<job_type>/``
* ``item_uploads(item, cfg, settings)``   — yields (local, remote)
                                              pairs to scp
* ``item_spec(item, cfg, settings)``      — dict written into the
                                              bundle that the worker
                                              reads (per-item)
* ``item_id(item)``                       — opaque string used as
                                              batch_status key
* ``item_downloads(item, cfg, settings)`` — yields (remote, local)
                                              pairs to scp back when
                                              the item completes

The framework owns:

* ``dispatch_remote_batch(spec, job_id, cfg, items, log_path, registry)``
* ``poll_remote_batch(spec, job_id, cfg, batch_id, log_path, registry)``
* ``resume_remote_batch(spec, job_id, cfg, log_path, registry)``
* ``kill_remote_batch_runner(cfg, runner_pid, batch_id, log_path)``
* The generic runner script (writes ``batch_status.json``, calls the
  per-job-type worker for each item, respects ``cancel.flag``).

NOTE: this module is brand-new and not yet wired to any UI button
or queue-manager dispatch path.  Currently exists for future use;
the legacy ``dispatch_remote_preproc_batch`` etc. in remote.py
remain the active code path.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from ..db import get_db_ctx
from .remote import (
    RemoteConfig,
    _scp_base_args,
    _scp_if_changed,
    _ssh_base_args,
    _py_cmd,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Layout helpers — one work_dir / subjects / shared / batches tree.
# ──────────────────────────────────────────────────────────────────

def _work_dir(cfg: RemoteConfig) -> str:
    """Backslash-normalised work_dir.  SCP rejects mixed-slash paths
    on Windows (``C:\\Users\\...\\workdir/subjects/...`` silently
    returns "No such file"), so every path builder funnels through
    here.  See poll_remote_preproc_batch story for the bug this
    caught.
    """
    return cfg.work_dir.replace("\\", "/")


def shared_modules_remote(cfg: RemoteConfig, job_type: str) -> str:
    return f"{_work_dir(cfg)}/shared/modules/{job_type}"


def shared_scripts_remote(cfg: RemoteConfig) -> str:
    return f"{_work_dir(cfg)}/shared/scripts"


def subject_remote(cfg: RemoteConfig, subject: str) -> str:
    return f"{_work_dir(cfg)}/subjects/{subject}"


def subject_videos_remote(cfg: RemoteConfig, subject: str) -> str:
    return f"{subject_remote(cfg, subject)}/videos"


def batch_state_remote(cfg: RemoteConfig, batch_id: str) -> str:
    return f"{_work_dir(cfg)}/batches/{batch_id}"


# ──────────────────────────────────────────────────────────────────
# Job-spec dataclass
# ──────────────────────────────────────────────────────────────────

@dataclass
class RemoteBatchJobSpec:
    """One per-job-type configuration; the dispatcher / poller /
    resume / kill all take this as their first argument.
    """
    job_type: str                            # 'preproc', 'mediapipe', etc.
    worker_script_source: str                # Python source for the worker
    # Local Path objects whose contents go into shared/modules/<job_type>/.
    # These get import-flattened on the remote (relative → flat) so the
    # worker can import them from a flat sys.path.
    shared_module_paths: tuple = ()
    # Per-item callbacks.  Each receives the item dict (whatever the
    # caller passed into dispatch), plus cfg + settings for path
    # resolution.  Yield (local_path, remote_path) for uploads or
    # (remote_path, local_path) for downloads.
    item_uploads_fn: Callable[[dict, RemoteConfig, Any], Iterable] = lambda *_: ()
    item_spec_fn:    Callable[[dict, RemoteConfig, Any], dict]     = lambda *_: {}
    item_id_fn:      Callable[[dict], str]                          = lambda i: str(i)
    item_downloads_fn: Callable[[dict, RemoteConfig, Any], Iterable] = lambda *_: ()

    # Display label used in log lines + error messages.
    display_label: str = ""

    def __post_init__(self):
        if not self.display_label:
            self.display_label = self.job_type


# ──────────────────────────────────────────────────────────────────
# Generic remote runner script (one for all job types)
# ──────────────────────────────────────────────────────────────────
#
# Reads batch.json which lists items + per-item bundle paths.  For
# each item, spawns the job-type-specific worker as a subprocess
# (its source is at shared/scripts/<job_type>_worker.py).  Writes
# batch_status.json with per-item state so the local poller can
# stream chips + downloads.  Respects cancel.flag (checked between
# items) so a local kill is graceful.

_GENERIC_RUNNER_SOURCE = r'''#!/usr/bin/env python3
"""Generic remote batch runner.  Spawned detached by the local
dispatcher; outlives the SSH session.  Reads batch.json, loops
items, spawns the per-job-type worker for each, writes a heartbeat'd
batch_status.json.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback


def _atomic_write(path, payload):
    path = os.path.normpath(path)
    d = os.path.dirname(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=d, suffix=".tmp", delete=False) as f:
        json.dump(payload, f)
        tmp = f.name
    for attempt in range(10):
        try:
            os.replace(tmp, path); return
        except OSError:
            time.sleep(min(0.05 * (1 + attempt), 0.5))


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} batch.json batch_status.json", flush=True)
        sys.exit(1)
    batch_spec_path, status_path = sys.argv[1], sys.argv[2]
    with open(batch_spec_path) as f:
        spec = json.load(f)

    items        = spec["items"]                # list of {id, bundle_path, status_path}
    worker_py    = spec["worker_path"]
    python_exe   = spec["python"]
    cancel_flag  = spec.get("cancel_flag", "")  # path to sentinel file
    batch_id     = spec.get("batch_id", "?")

    state = {it["id"]: {"status": "pending"} for it in items}
    started = time.time()

    def _flush(current_id=None, extra_log=None):
        payload = {
            "batch_id": batch_id,
            "status": "running",
            "items": state,
            "started_at": started,
            "heartbeat": time.time(),
            "pid": os.getpid(),
        }
        if current_id is not None:
            payload["current_item"] = current_id
        if extra_log:
            payload["last_log"] = extra_log
        _atomic_write(status_path, payload)

    def _cancelled():
        return bool(cancel_flag and os.path.exists(cancel_flag))

    _flush()

    for it in items:
        if _cancelled():
            print(f"[runner] cancel.flag present → stopping early", flush=True)
            for k, v in state.items():
                if v.get("status") == "pending":
                    state[k] = {"status": "cancelled"}
            break

        item_id = it["id"]
        bundle  = it["bundle_path"]
        item_st = it["status_path"]

        # Skip-if-completed: a per-item status file from an earlier
        # runner attempt that succeeded means we don't re-do work.
        try:
            if os.path.exists(item_st):
                with open(item_st) as f:
                    prev = json.load(f)
                if prev.get("status") == "completed":
                    state[item_id] = {"status": "completed", "skipped": True}
                    _flush(extra_log=f"{item_id}: already completed, skipping")
                    continue
        except Exception:
            pass

        state[item_id] = {"status": "running"}
        _flush(current_id=item_id, extra_log=f"starting {item_id}")
        try:
            rc = subprocess.run(
                [python_exe, "-u", worker_py, bundle, item_st],
                cwd=os.path.dirname(worker_py),
            ).returncode
            state[item_id] = (
                {"status": "completed"} if rc == 0
                else {"status": "failed", "rc": rc}
            )
        except Exception as e:
            state[item_id] = {"status": "failed", "error": str(e)}
        _flush(extra_log=f"finished {item_id}: {state[item_id]['status']}")

    final_status = "cancelled" if _cancelled() else "completed"
    _atomic_write(status_path, {
        "batch_id": batch_id,
        "status": final_status,
        "items": state,
        "started_at": started,
        "finished_at": time.time(),
        "heartbeat": time.time(),
        "pid": os.getpid(),
    })


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
'''


# Import-flattening rewrites — same set the legacy preproc worker
# applies to its modules dir.  Kept here so any per-job-type worker
# that imports the shared modules sees a flat import surface.
_IMPORT_REWRITES = (
    ("from ..config import",            "from config import"),
    ("from .config import",             "from config import"),
    ("from .ffmpeg import",             "from ffmpeg import"),
    ("from .video import",              "from video import"),
    ("from .camera_motion import",      "from camera_motion import"),
    ("from .background import",         "from background import"),
    ("from .mediapipe_prelabel import", "from mediapipe_prelabel import"),
    ("from .calibration import",        "from calibration import"),
    ("from ..services.",                "from "),
)


def _flatten_imports_to_tempfile(src_path: Path) -> Path:
    """Read a local .py module, apply the import-flattening rewrites,
    write to a temp file, return its path.  We do this LOCALLY so the
    file we scp up has flat imports and ``_scp_if_changed`` size-match
    is stable across batches (the legacy in-place rewrite on the
    remote made the file size drift between batches).
    """
    with open(src_path, "r", encoding="utf-8") as f:
        src = f.read()
    new = src
    for old, repl in _IMPORT_REWRITES:
        new = new.replace(old, repl)
    fd, tmp = tempfile.mkstemp(suffix="_" + src_path.name)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(new)
    return Path(tmp)


# ──────────────────────────────────────────────────────────────────
# Kill helper (generic — independent of job type)
# ──────────────────────────────────────────────────────────────────

def kill_remote_batch_runner(
    cfg: RemoteConfig,
    runner_pid,
    batch_id: str | None = None,
    log_path: str | None = None,
):
    """Best-effort kill of the runner process + process tree.  Drops
    a ``cancel.flag`` sentinel into the batch state dir so a runner
    that hasn't yet started the next item bails on its next iteration
    even if the signal didn't land.
    """
    if not runner_pid:
        return
    try:
        runner_pid = int(runner_pid)
    except (TypeError, ValueError):
        return
    script_parts = [
        f"import os, subprocess, sys",
        f"pid={runner_pid}",
        # taskkill /T walks the process tree on Windows; os.kill is
        # the cross-platform fallback for POSIX.
        (f"_ = subprocess.run(['taskkill','/F','/T','/PID',str(pid)], "
         f"capture_output=True) if os.name=='nt' "
         f"else os.kill(pid, 15) if pid > 0 else None"),
    ]
    if batch_id:
        sentinel = f"{batch_state_remote(cfg, batch_id)}/cancel.flag"
        script_parts.append(f"open(r'{sentinel}', 'w').write('1')")
    try:
        subprocess.run(
            _py_cmd(cfg, f"\"{'; '.join(script_parts)}\""),
            capture_output=True, timeout=15,
        )
        if log_path:
            try:
                with open(log_path, "a") as f:
                    f.write(f"  killed remote batch runner pid {runner_pid}\n")
            except OSError:
                pass
    except Exception as e:
        if log_path:
            try:
                with open(log_path, "a") as f:
                    f.write(f"  WARN: kill_remote_batch_runner failed: {e}\n")
            except OSError:
                pass


# ──────────────────────────────────────────────────────────────────
# Dispatcher
# ──────────────────────────────────────────────────────────────────

def dispatch_remote_batch(
    spec: RemoteBatchJobSpec,
    job_id: int,
    cfg: RemoteConfig,
    items: list[dict],
    log_path: str,
    registry,
    upload_only: bool = False,
) -> dict:
    """Upload shared scripts + modules + per-item inputs, write the
    batch spec, launch the detached runner.  Returns
    ``{"batch_id", "remote_pid"}`` on success or ``{"error": ...}``
    on failure.

    Idempotent: re-running with the same ``job_id`` reuses the
    persisted ``_batch_id`` from the job's ``params_json`` and only
    re-uploads files whose size differs from the remote.  The
    ``uploads_complete.json`` sentinel in the batch state dir lets
    a re-dispatch skip the upload phase entirely.
    """
    from ..config import get_settings
    settings = get_settings()

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with get_db_ctx() as db:
        row = db.execute(
            "SELECT params_json FROM jobs WHERE id=?", (job_id,),
        ).fetchone()
    parent_params = {}
    if row:
        try:
            parent_params = json.loads(row["params_json"] or "{}")
        except Exception:
            parent_params = {}

    batch_id = (parent_params.get("_batch_id")
                or f"{spec.job_type[:1]}{int(time.time())}_{uuid.uuid4().hex[:6]}")
    state_dir = batch_state_remote(cfg, batch_id)
    shared_modules = shared_modules_remote(cfg, spec.job_type)
    shared_scripts = shared_scripts_remote(cfg)

    logfile = open(log_path, "a", buffering=1)
    logfile.write(
        f"\n=== Remote {spec.display_label} batch {batch_id} "
        f"({len(items)} item(s)) ===\n"
    )
    logfile.flush()

    cancel_event = registry.register_cancel_event(job_id)

    def _check_cancel():
        if cancel_event.is_set():
            raise InterruptedError("cancelled")

    def _fail(msg: str) -> dict:
        logfile.write(f"ERROR: {msg}\n"); logfile.flush()
        kill_remote_batch_runner(cfg, parent_params.get("_remote_pid"),
                                  batch_id, log_path)
        with get_db_ctx() as db:
            db.execute(
                "UPDATE jobs SET status='failed', error_msg=?, "
                "finished_at=CURRENT_TIMESTAMP WHERE id=?",
                (msg[:500], job_id),
            )
        return {"error": msg}

    try:
        # Reachability probe.
        probe = subprocess.run(
            _ssh_base_args(cfg) + [cfg.host, "echo", "ok"],
            capture_output=True, text=True, timeout=15,
        )
        if probe.returncode != 0 or "ok" not in probe.stdout:
            return _fail(f"remote unreachable: {probe.stderr[:200]}")

        # Mint dirs.
        subprocess.run(
            _py_cmd(cfg, f"\"import os; "
                          f"os.makedirs(r'{state_dir}', exist_ok=True); "
                          f"os.makedirs(r'{shared_modules}', exist_ok=True); "
                          f"os.makedirs(r'{shared_scripts}', exist_ok=True)\""),
            capture_output=True, timeout=20,
        )

        # Sentinel check.
        uploads_complete = f"{state_dir}/uploads_complete.json"
        sc = subprocess.run(
            _py_cmd(cfg, f"\"import os; "
                          f"print('yes' if os.path.exists(r'{uploads_complete}') else 'no')\""),
            capture_output=True, text=True, timeout=15,
        )
        uploads_done = (sc.returncode == 0 and "yes" in (sc.stdout or ""))

        if uploads_done:
            logfile.write("  uploads_complete sentinel — skipping upload phase\n")
            logfile.flush()
        else:
            logfile.write(f"  uploading inputs for {len(items)} item(s)...\n")
            logfile.flush()
            # Shared modules (flatten locally, then scp with size check).
            for mp in spec.shared_module_paths:
                if not Path(mp).exists():
                    continue
                flat = _flatten_imports_to_tempfile(Path(mp))
                _scp_if_changed(
                    cfg, str(flat), f"{shared_modules}/{Path(mp).name}",
                    timeout=60, logfile=logfile,
                    label=f"module {Path(mp).name}: ",
                )
                try:
                    os.unlink(flat)
                except OSError:
                    pass
            # Runner script (shared across job types).
            runner_local = Path(tempfile.gettempdir()) / f"remote_batch_runner_{job_id}.py"
            runner_local.write_text(_GENERIC_RUNNER_SOURCE)
            _scp_if_changed(cfg, str(runner_local),
                             f"{shared_scripts}/runner.py",
                             timeout=30, logfile=logfile, label="runner: ")
            # Worker script (per-job-type).
            worker_local = Path(tempfile.gettempdir()) / f"{spec.job_type}_worker_{job_id}.py"
            worker_local.write_text(spec.worker_script_source)
            worker_remote = f"{shared_scripts}/{spec.job_type}_worker.py"
            _scp_if_changed(cfg, str(worker_local), worker_remote,
                             timeout=30, logfile=logfile,
                             label=f"{spec.job_type} worker: ")

            # Per-item uploads.
            for item in items:
                _check_cancel()
                for local_p, remote_p in spec.item_uploads_fn(item, cfg, settings):
                    if not os.path.exists(str(local_p)):
                        continue
                    _scp_if_changed(
                        cfg, str(local_p), str(remote_p),
                        timeout=600, logfile=logfile,
                        label=f"{spec.item_id_fn(item)}/{Path(str(local_p)).name}: ",
                    )
            # Mark uploads complete.
            subprocess.run(
                _py_cmd(cfg,
                    f"\"import json,time; "
                    f"open(r'{uploads_complete}','w').write(json.dumps({{'at':time.time()}}))\""),
                capture_output=True, timeout=15,
            )
            logfile.write("  uploads_complete sentinel written\n"); logfile.flush()

        if upload_only:
            return {"batch_id": batch_id}

        # ── Per-item bundle.json (always re-write; cheap) ──
        item_specs = []
        for item in items:
            item_id = spec.item_id_fn(item)
            item_state_dir = f"{state_dir}/items/{item_id}"
            subprocess.run(
                _py_cmd(cfg, f"\"import os; os.makedirs(r'{item_state_dir}', exist_ok=True)\""),
                capture_output=True, timeout=15,
            )
            bundle = spec.item_spec_fn(item, cfg, settings)
            # Make sure the bundle carries the shared paths the
            # worker will need (modules dir, etc.).
            bundle.setdefault("modules_dir", shared_modules)
            bundle.setdefault("item_id", item_id)
            bundle_local = Path(tempfile.gettempdir()) / f"bundle_{job_id}_{item_id}.json"
            bundle_local.write_text(json.dumps(bundle))
            bundle_remote = f"{item_state_dir}/bundle.json"
            subprocess.run(
                _scp_base_args(cfg) + [str(bundle_local),
                                        f"{cfg.host}:{bundle_remote}"],
                capture_output=True, text=True, timeout=30,
            )
            item_specs.append({
                "id":          item_id,
                "bundle_path": bundle_remote,
                "status_path": f"{item_state_dir}/status.json",
            })

        # batch.json
        batch_spec = {
            "job_id":      job_id,
            "batch_id":    batch_id,
            "worker_path": f"{shared_scripts}/{spec.job_type}_worker.py",
            "python":      cfg.python_executable,
            "items":       item_specs,
            "cancel_flag": f"{state_dir}/cancel.flag",
        }
        batch_local = Path(tempfile.gettempdir()) / f"batch_{job_id}.json"
        batch_local.write_text(json.dumps(batch_spec))
        batch_remote = f"{state_dir}/batch.json"
        up = subprocess.run(
            _scp_base_args(cfg) + [str(batch_local), f"{cfg.host}:{batch_remote}"],
            capture_output=True, text=True, timeout=30,
        )
        if up.returncode != 0:
            return _fail(f"batch.json upload failed: {up.stderr[:200]}")

        # Launch detached runner.
        runner_remote = f"{shared_scripts}/runner.py"
        status_remote = f"{state_dir}/batch_status.json"
        runner_log    = f"{state_dir}/runner.log"
        launch = (
            f"\"import subprocess, os, time; "
            f"log_fh = open(r'{runner_log}', 'w'); "
            f"args = [r'{cfg.python_executable}', '-u', "
            f"r'{runner_remote}', r'{batch_remote}', r'{status_remote}']; "
            f"flags = 0x01000200 if os.name == 'nt' else 0; "
            f"p = subprocess.Popen(args, creationflags=flags, "
            f"stdin=subprocess.DEVNULL, stdout=log_fh, stderr=log_fh); "
            f"print(p.pid); time.sleep(2)\""
        )
        launch_res = subprocess.run(_py_cmd(cfg, launch),
                                     capture_output=True, text=True, timeout=30)
        if launch_res.returncode != 0:
            return _fail(f"runner launch failed: {launch_res.stderr[:200]}")
        try:
            remote_pid = int(launch_res.stdout.strip().splitlines()[-1])
        except Exception:
            remote_pid = None
        logfile.write(f"  runner started (pid {remote_pid})\n"); logfile.flush()

        # Persist on the job row.
        parent_params["_batch_id"]   = batch_id
        parent_params["_job_type"]   = spec.job_type
        if remote_pid is not None:
            parent_params["_remote_pid"] = remote_pid
        with get_db_ctx() as db:
            db.execute(
                "UPDATE jobs SET params_json=?, tmux_session=? WHERE id=?",
                (json.dumps(parent_params),
                 f"pid:{remote_pid}" if remote_pid else None,
                 job_id),
            )
        return {"batch_id": batch_id, "remote_pid": remote_pid}

    except InterruptedError:
        logfile.write("  dispatch cancelled\n"); logfile.flush()
        kill_remote_batch_runner(cfg, parent_params.get("_remote_pid"),
                                  batch_id, log_path)
        with get_db_ctx() as db:
            db.execute(
                "UPDATE jobs SET status='cancelled', "
                "finished_at=CURRENT_TIMESTAMP WHERE id=?", (job_id,),
            )
        return {"error": "cancelled"}
    except Exception as e:
        logger.exception(f"dispatch_remote_batch {job_id} ({spec.job_type}): {e}")
        return _fail(str(e)[:500])
    finally:
        try: logfile.close()
        except Exception: pass


# ──────────────────────────────────────────────────────────────────
# Poller
# ──────────────────────────────────────────────────────────────────

def poll_remote_batch(
    spec: RemoteBatchJobSpec,
    job_id: int,
    cfg: RemoteConfig,
    batch_id: str,
    items: list[dict],
    log_path: str,
    registry,
    on_item_outcome: Callable | None = None,
) -> None:
    """Stream batch_status.json updates from the remote; when an item
    completes, scp back the files ``spec.item_downloads_fn`` lists
    and fire ``on_item_outcome(item, outcome)``.

    Exits when the runner reports ``completed`` or ``cancelled``.
    Handles transient SSH failures (laptop sleep, network blip) by
    just skipping a poll and retrying on the next tick.
    """
    from ..config import get_settings
    settings = get_settings()
    state_dir = batch_state_remote(cfg, batch_id)
    status_remote = f"{state_dir}/batch_status.json"
    runner_log    = f"{state_dir}/runner.log"

    cancel_event = registry.register_cancel_event(job_id)

    def _check_cancel():
        if cancel_event.is_set():
            raise InterruptedError("cancelled")

    items_by_id = {spec.item_id_fn(it): it for it in items}
    downloaded: set[str] = set()
    last_log_size = 0

    logfile = open(log_path, "a", buffering=1)
    logfile.write(f"\n=== Poller attached to {spec.display_label} batch {batch_id} ===\n")
    logfile.flush()

    def _download_item(item):
        item_id = spec.item_id_fn(item)
        any_ok = False
        for remote_p, local_p in spec.item_downloads_fn(item, cfg, settings):
            os.makedirs(os.path.dirname(str(local_p)), exist_ok=True)
            r = subprocess.run(
                _scp_base_args(cfg) + [f"{cfg.host}:{remote_p}", str(local_p)],
                capture_output=True, text=True, timeout=300,
            )
            if r.returncode == 0 and os.path.exists(str(local_p)):
                any_ok = True
        return any_ok

    try:
        while True:
            _check_cancel()
            time.sleep(3.0)
            # Pull batch_status.
            local_status = Path(tempfile.gettempdir()) / f"batch_status_{job_id}.json"
            r = subprocess.run(
                _scp_base_args(cfg) + [f"{cfg.host}:{status_remote}", str(local_status)],
                capture_output=True, text=True, timeout=30,
            )
            if r.returncode != 0:
                continue
            try:
                with open(local_status) as f:
                    st = json.load(f)
            except Exception:
                continue
            item_state = st.get("items", {})
            phase = st.get("status", "")

            # Pull per-item outputs for any newly-completed item.
            for item_id, info in item_state.items():
                if item_id in downloaded:
                    continue
                if info.get("status") not in ("completed", "failed"):
                    continue
                item = items_by_id.get(item_id)
                if not item:
                    downloaded.add(item_id)
                    continue
                outcome = "ok" if info.get("status") == "completed" else "failed"
                if outcome == "ok":
                    logfile.write(f"  downloading outputs for {item_id}\n")
                    logfile.flush()
                    ok = _download_item(item)
                    if not ok:
                        outcome = "failed"
                downloaded.add(item_id)
                if on_item_outcome:
                    try:
                        on_item_outcome(item, outcome)
                    except Exception:
                        pass

            # Update parent job's progress_pct (count of done items).
            n_total = max(1, len(item_state))
            n_done = sum(1 for v in item_state.values()
                         if v.get("status") in ("completed", "failed"))
            pct = round(100.0 * n_done / n_total, 1)
            with get_db_ctx() as db:
                db.execute("UPDATE jobs SET progress_pct=? WHERE id=?",
                            (pct, job_id))

            # Stream new runner-log lines into the parent log.
            try:
                tail = subprocess.run(
                    _py_cmd(cfg, f"\"import os; p=r'{runner_log}'; "
                                  f"print(os.path.getsize(p) if os.path.exists(p) else 0)\""),
                    capture_output=True, text=True, timeout=15,
                )
                cur_sz = int((tail.stdout or "0").strip())
                if cur_sz > last_log_size:
                    pull = subprocess.run(
                        _scp_base_args(cfg) + [f"{cfg.host}:{runner_log}",
                                                str(local_status) + ".log"],
                        capture_output=True, text=True, timeout=30,
                    )
                    if pull.returncode == 0:
                        try:
                            with open(str(local_status) + ".log") as f:
                                f.seek(last_log_size)
                                new = f.read()
                            if new.strip():
                                logfile.write(new); logfile.flush()
                        except OSError:
                            pass
                    last_log_size = cur_sz
            except Exception:
                pass

            if phase in ("completed", "cancelled", "failed"):
                logfile.write(f"=== Batch {phase} ===\n"); logfile.flush()
                break

        # Final job status flush.
        n_ok = sum(1 for v in item_state.values()
                   if v.get("status") == "completed")
        n_total = len(item_state)
        err = None if n_ok == n_total else f"{n_ok}/{n_total} succeeded"
        with get_db_ctx() as db:
            db.execute(
                "UPDATE jobs SET status=?, error_msg=?, progress_pct=100, "
                "finished_at=CURRENT_TIMESTAMP WHERE id=?",
                ("completed", err, job_id),
            )
    except InterruptedError:
        logfile.write("  poller cancelled — killing remote runner\n")
        logfile.flush()
        kill_remote_batch_runner(cfg, _fetch_runner_pid(job_id),
                                  batch_id, log_path)
        with get_db_ctx() as db:
            db.execute(
                "UPDATE jobs SET status='cancelled', "
                "finished_at=CURRENT_TIMESTAMP WHERE id=?", (job_id,),
            )
    except Exception as e:
        logger.exception(f"poll_remote_batch {job_id} ({spec.job_type}): {e}")
        kill_remote_batch_runner(cfg, _fetch_runner_pid(job_id),
                                  batch_id, log_path)
        with get_db_ctx() as db:
            db.execute(
                "UPDATE jobs SET status='failed', error_msg=?, "
                "finished_at=CURRENT_TIMESTAMP WHERE id=?",
                (f"poller: {str(e)[:400]}", job_id),
            )
    finally:
        try: logfile.close()
        except Exception: pass


# ──────────────────────────────────────────────────────────────────
# Resume
# ──────────────────────────────────────────────────────────────────

def resume_remote_batch(
    spec: RemoteBatchJobSpec,
    job_id: int,
    cfg: RemoteConfig,
    items: list[dict],
    log_path: str,
    registry,
    on_item_outcome: Callable | None = None,
) -> None:
    """Attach to an existing batch by probing the remote heartbeat;
    fall back to re-dispatch if the runner died.  Uploads are
    skip-if-present so re-dispatch is cheap.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    try:
        with open(log_path, "a") as f:
            f.write(f"\n=== Resuming {spec.display_label} job {job_id} ===\n")
    except OSError:
        pass

    try:
        with get_db_ctx() as db:
            row = db.execute(
                "SELECT params_json FROM jobs WHERE id=?", (job_id,),
            ).fetchone()
        if not row:
            return
        params = json.loads(row["params_json"] or "{}")
        batch_id = params.get("_batch_id")
        runner_alive = False
        if batch_id:
            status_remote = f"{batch_state_remote(cfg, batch_id)}/batch_status.json"
            probe = subprocess.run(
                _py_cmd(cfg,
                    f"\"import os, json, time; p=r'{status_remote}'; "
                    f"d=json.load(open(p)) if os.path.exists(p) else None; "
                    f"print(json.dumps({{'has':d is not None,"
                    f"'hb':(d or {{}}).get('heartbeat'),"
                    f"'now':time.time()}}))\""),
                capture_output=True, text=True, timeout=20,
            )
            try:
                info = json.loads((probe.stdout or "").strip().splitlines()[-1])
                if info.get("has"):
                    hb = info.get("hb") or 0
                    now = info.get("now") or 0
                    runner_alive = (now - hb) < 60
            except Exception:
                pass

        if runner_alive and batch_id:
            try:
                with open(log_path, "a") as f:
                    f.write(f"  runner alive (batch {batch_id}) — reattaching poller\n")
            except OSError:
                pass
            poll_remote_batch(
                spec, job_id=job_id, cfg=cfg, batch_id=batch_id,
                items=items, log_path=log_path, registry=registry,
                on_item_outcome=on_item_outcome,
            )
            return

        # Re-dispatch.
        try:
            with open(log_path, "a") as f:
                f.write("  runner not alive — re-dispatching\n")
        except OSError:
            pass
        result = dispatch_remote_batch(
            spec, job_id=job_id, cfg=cfg, items=items,
            log_path=log_path, registry=registry,
        )
        if result.get("error"):
            return
        poll_remote_batch(
            spec, job_id=job_id, cfg=cfg, batch_id=result["batch_id"],
            items=items, log_path=log_path, registry=registry,
            on_item_outcome=on_item_outcome,
        )
    except Exception as e:
        logger.exception(f"resume_remote_batch {job_id} ({spec.job_type}): {e}")
        kill_remote_batch_runner(cfg, _fetch_runner_pid(job_id), None, log_path)
        with get_db_ctx() as db:
            db.execute(
                "UPDATE jobs SET status='failed', error_msg=?, "
                "finished_at=CURRENT_TIMESTAMP WHERE id=?",
                (f"resume: {str(e)[:400]}", job_id),
            )


def _fetch_runner_pid(job_id: int):
    """Mirror of remote.py's _fetch_runner_pid — duplicated here to
    avoid a circular import.  Reads the stored PID from either the
    ``tmux_session`` column ('pid:N') or ``params_json._remote_pid``.
    """
    try:
        with get_db_ctx() as db:
            row = db.execute(
                "SELECT params_json, tmux_session FROM jobs WHERE id=?",
                (job_id,),
            ).fetchone()
        if not row:
            return None
        ts = (row["tmux_session"] or "").strip()
        if ts.startswith("pid:"):
            try:
                return int(ts.split(":", 1)[1])
            except (ValueError, IndexError):
                pass
        try:
            p = json.loads(row["params_json"] or "{}")
            v = p.get("_remote_pid")
            if v is not None:
                return int(v)
        except Exception:
            pass
    except Exception:
        pass
    return None
