"""Job registry: subprocess management and progress tracking."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from datetime import datetime, timezone

from ..config import PROJECT_DIR
from ..db import get_db_ctx

logger = logging.getLogger(__name__)


class JobRegistry:
    """Track running subprocesses and their progress."""

    def __init__(self):
        self._processes: dict[int, subprocess.Popen] = {}
        self._threads: dict[int, threading.Thread] = {}
        self._cancel_events: dict[int, threading.Event] = {}

    def launch(self, job_id: int, cmd: list[str], log_path: str,
               progress_parser=None, on_complete=None, env=None) -> int:
        """Launch a subprocess and track it.

        Args:
            job_id: Database job ID
            cmd: Command list for subprocess
            log_path: Path to write stdout/stderr
            progress_parser: callable(line) -> float|None for progress extraction
            on_complete: callable(job_id, returncode) for post-completion actions
            env: Optional environment variables dict (merged with os.environ)
        """
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        # Merge provided env with current environment
        popen_env = os.environ.copy() if env is None else dict(env)

        # Embeddable/portable Python on Windows ships with a
        # ``python3XX._pth`` file that overrides ``sys.path`` and
        # does NOT include ``.`` (cwd).  Passing ``cwd=PROJECT_DIR``
        # to Popen sets the subprocess's cwd but the ``_pth`` still
        # wins, so ``python -m dlc_labeler.services.worker``
        # fails with ``ModuleNotFoundError: No module named
        # 'dlc_labeler'``.  Prepending PROJECT_DIR to
        # PYTHONPATH does work — Python honors PYTHONPATH even
        # under a ``_pth`` override.
        _existing_pp = popen_env.get("PYTHONPATH", "")
        popen_env["PYTHONPATH"] = (str(PROJECT_DIR)
                                    + (os.pathsep + _existing_pp
                                        if _existing_pp else ""))

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(PROJECT_DIR),
            env=popen_env,
        )

        self._processes[job_id] = proc

        # Update DB with PID.  Preserve any existing started_at so a job
        # that launches several subprocesses in sequence keeps its original
        # start time — the ETA math extrapolates from total-elapsed × pct.
        with get_db_ctx() as db:
            db.execute(
                """UPDATE jobs SET status = 'running', pid = ?,
                   started_at = COALESCE(started_at, CURRENT_TIMESTAMP) WHERE id = ?""",
                (proc.pid, job_id),
            )

        # Start monitoring thread
        thread = threading.Thread(
            target=self._monitor,
            args=(job_id, proc, log_path, progress_parser, on_complete),
            daemon=True,
        )
        thread.start()
        self._threads[job_id] = thread

        return proc.pid

    def reattach(self, job_id: int, pid: int, log_path: str,
                 progress_parser=None, on_complete=None):
        """Re-attach to a running subprocess after app restart.

        Opens the process by PID and starts monitoring it.
        The log file is appended to (not overwritten).
        """
        import psutil  # type: ignore

        try:
            proc = psutil.Process(pid)
            if not proc.is_running():
                raise ProcessLookupError(f"PID {pid} not running")
        except (ImportError, ProcessLookupError):
            # psutil not available or process dead — try basic approach
            # We can't get stdout from an existing process, but we can
            # poll for completion by checking if PID is alive
            logger.info(f"Reattach job {job_id}: monitoring PID {pid} by polling")
            self._poll_pid(job_id, pid, on_complete)
            return

        logger.info(f"Reattach job {job_id}: monitoring PID {pid}")
        self._poll_pid(job_id, pid, on_complete)

    def _poll_pid(self, job_id: int, pid: int, on_complete=None):
        """Monitor an existing process by polling its PID."""
        def _poll():
            while True:
                try:
                    os.kill(pid, 0)  # Check if alive
                except (OSError, ProcessLookupError):
                    # Process exited
                    with get_db_ctx() as db:
                        job = db.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
                        if job and job["status"] == "running":
                            # Worker updates DB directly, so check if it set a final status
                            # If still "running" after process death → it crashed
                            db.execute(
                                """UPDATE jobs SET status = 'failed',
                                   error_msg = 'Process exited unexpectedly',
                                   finished_at = CURRENT_TIMESTAMP WHERE id = ?""",
                                (job_id,),
                            )
                    self._write_history(job_id)
                    if on_complete:
                        on_complete(job_id, 1)
                    return
                time.sleep(3)

        thread = threading.Thread(target=_poll, daemon=True)
        thread.start()
        self._threads[job_id] = thread

    @staticmethod
    def _write_history(job_id: int):
        """Append this job to the durable lifetime history file.

        The jobs table is pruned by the UI and lives in the same database
        a student might delete; the history file is append-only JSONL
        with the app's git version, so "which script ran, on what, when,
        with which code" stays answerable months later.
        """
        try:
            from .job_history import finalize_job_record
            finalize_job_record(job_id)
        except Exception:
            logger.exception("job_history flush failed for job %s", job_id)

    def _monitor(self, job_id, proc, log_path, progress_parser, on_complete):
        """Monitor subprocess output, update progress, write log."""
        try:
            first_epoch_at = None
            with open(log_path, "w") as logfile:
                for line in proc.stdout:
                    logfile.write(line)
                    logfile.flush()

                    if progress_parser:
                        result = progress_parser(line)
                        if result is not None:
                            pct, epoch, total = result
                            # Track first epoch timestamp
                            if epoch is not None and epoch >= 1 and first_epoch_at is None:
                                first_epoch_at = datetime.now(timezone.utc).isoformat()
                            epoch_json = None
                            if epoch is not None:
                                epoch_json = json.dumps({
                                    "epoch": epoch,
                                    "total": total,
                                    "first_epoch_at": first_epoch_at,
                                })
                            with get_db_ctx() as db:
                                db.execute(
                                    "UPDATE jobs SET progress_pct = ?, epoch_info = COALESCE(?, epoch_info) WHERE id = ?",
                                    (pct, epoch_json, job_id),
                                )

            proc.wait()
            status = "completed" if proc.returncode == 0 else "failed"
            error_msg = None if proc.returncode == 0 else f"Exit code {proc.returncode}"

            with get_db_ctx() as db:
                db.execute(
                    """UPDATE jobs SET status = ?, error_msg = ?, progress_pct = ?,
                       finished_at = CURRENT_TIMESTAMP WHERE id = ?""",
                    (status, error_msg, 100.0 if status == "completed" else None, job_id),
                )

            self._write_history(job_id)

            if on_complete:
                on_complete(job_id, proc.returncode)

        except Exception as e:
            logger.exception(f"Job {job_id} monitor error")
            with get_db_ctx() as db:
                db.execute(
                    """UPDATE jobs SET status = 'failed', error_msg = ?,
                       finished_at = CURRENT_TIMESTAMP WHERE id = ?""",
                    (str(e), job_id),
                )
            self._write_history(job_id)
        finally:
            self._processes.pop(job_id, None)
            self._threads.pop(job_id, None)

    def register_cancel_event(self, job_id: int) -> threading.Event:
        """Create and register a cancel event for a thread-based job."""
        evt = threading.Event()
        self._cancel_events[job_id] = evt
        return evt

    def unregister_cancel_event(self, job_id: int):
        """Remove cancel event after job finishes."""
        self._cancel_events.pop(job_id, None)

    def cancel(self, job_id: int) -> bool:
        """Cancel a running job (subprocess or thread-based)."""
        # Try subprocess first
        proc = self._processes.get(job_id)
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

            with get_db_ctx() as db:
                db.execute(
                    """UPDATE jobs SET status = 'cancelled',
                       finished_at = CURRENT_TIMESTAMP WHERE id = ?""",
                    (job_id,),
                )
            self._write_history(job_id)
            return True

        # Try cancel event (thread-based jobs)
        evt = self._cancel_events.get(job_id)
        if evt is not None:
            evt.set()
            # DB update will be handled by the thread itself when it checks the event
            return True

        return False

    def is_running(self, job_id: int) -> bool:
        return job_id in self._processes or job_id in self._cancel_events

    def get_log_tail(self, log_path: str, n_lines: int = 50) -> list[str]:
        """Read last N lines of a log file."""
        try:
            with open(log_path, "r") as f:
                lines = f.readlines()
                return lines[-n_lines:]
        except FileNotFoundError:
            return []


# Global singleton
registry = JobRegistry()
