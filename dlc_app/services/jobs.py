"""Job registry: subprocess management and progress tracking."""
from __future__ import annotations

import logging
import os
import re
import subprocess
import threading
from pathlib import Path

from ..config import get_settings, PROJECT_DIR
from ..db import get_db_ctx

logger = logging.getLogger(__name__)


class JobRegistry:
    """Track running subprocesses and their progress."""

    def __init__(self):
        self._processes: dict[int, subprocess.Popen] = {}
        self._threads: dict[int, threading.Thread] = {}
        self._cancel_events: dict[int, threading.Event] = {}

    def launch(self, job_id: int, cmd: list[str], log_path: str,
               progress_parser=None, on_complete=None) -> int:
        """Launch a subprocess and track it.

        Args:
            job_id: Database job ID
            cmd: Command list for subprocess
            log_path: Path to write stdout/stderr
            progress_parser: callable(line) -> float|None for progress extraction
            on_complete: callable(job_id, returncode) for post-completion actions
        """
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(PROJECT_DIR),
        )

        self._processes[job_id] = proc

        # Update DB with PID
        with get_db_ctx() as db:
            db.execute(
                """UPDATE jobs SET status = 'running', pid = ?,
                   started_at = CURRENT_TIMESTAMP WHERE id = ?""",
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

    def _monitor(self, job_id, proc, log_path, progress_parser, on_complete):
        """Monitor subprocess output, update progress, write log."""
        try:
            with open(log_path, "w") as logfile:
                for line in proc.stdout:
                    logfile.write(line)
                    logfile.flush()

                    if progress_parser:
                        pct = progress_parser(line)
                        if pct is not None:
                            with get_db_ctx() as db:
                                db.execute(
                                    "UPDATE jobs SET progress_pct = ? WHERE id = ?",
                                    (pct, job_id),
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
        """Cancel a running job (subprocess, tmux remote, or thread-based)."""
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
            return True

        # Kill remote process if one exists
        with get_db_ctx() as db:
            job = db.execute(
                "SELECT tmux_session, remote_host FROM jobs WHERE id = ?",
                (job_id,),
            ).fetchone()

        if job and job.get("tmux_session") and job.get("remote_host"):
            session_info = job["tmux_session"]
            if session_info.startswith("pid:"):
                try:
                    remote_pid = int(session_info.split(":")[1])
                    from ..config import get_settings
                    from .remote import _kill_remote_pid
                    remote_cfg = get_settings().get_remote_config()
                    if remote_cfg:
                        _kill_remote_pid(remote_cfg, remote_pid)
                        # The monitoring thread will detect process death on its next poll
                except (ValueError, IndexError):
                    pass

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


def parse_dlc_training_progress(line: str) -> float | None:
    """Parse DLC training log output for epoch progress."""
    # PyTorch DLC format: "Epoch 5/200"
    match = re.search(r"Epoch\s+(\d+)/(\d+)", line)
    if match:
        current = int(match.group(1))
        total = int(match.group(2))
        return (current / total) * 100.0
    return None


# Global singleton
registry = JobRegistry()
