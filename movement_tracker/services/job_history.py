"""Lifetime job-history JSONL log.

Persists every completed job (success / failure / cancelled) into
``<DATA_DIR>/job_history.jsonl`` -- one JSON record per line, appended
forever.  Designed for later analysis: pair each record with a git
checkout to correlate code changes to job-runtime impact.

Each record carries:
- All the metadata the DB ``jobs`` table tracks (type, status, subject,
  timestamps, error message)
- Total wall-clock duration
- Per-stage timings recorded via :func:`add_stage` / :class:`stage_timer`
  -- e.g. video upload time, per-trial compute time, download time
- The git commit hash currently running (read from the VERSION file)
- The hostname so multi-machine setups stay disambiguated

The DB only keeps recent jobs (job-history view); this file accumulates
indefinitely and survives DB resets / app reinstalls.
"""
from __future__ import annotations

import json
import logging
import socket
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Per-job in-memory stage timings, populated as the job runs.  The list
# is flushed to the JSONL file by :func:`finalize_job_record` at the
# dispatcher's success / failure / cancel exit points.
_job_stages: dict[int, list[dict[str, Any]]] = {}
_lock = threading.Lock()


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _history_path() -> Path:
    from ..config import DATA_DIR
    return DATA_DIR / "job_history.jsonl"


def _git_version() -> str | None:
    """Read the latest commit hash from the VERSION file written at
    startup.  Returns None if the file is missing or empty."""
    try:
        from ..config import PROJECT_DIR
        vp = Path(PROJECT_DIR) / "VERSION"
        if vp.exists():
            v = vp.read_text().strip()
            return v or None
    except Exception:
        pass
    return None


def add_stage(job_id: int, name: str, duration_sec: float, **extra) -> None:
    """Record one completed stage for a job.

    Call at the end of each instrumented sub-step (one upload phase, one
    trial of a batch, one compute call inside a compound job, etc.).
    Stages accumulate in memory until :func:`finalize_job_record` flushes
    them to disk.

    Extra keyword args are stored verbatim in the stage record -- common
    ones: ``trial`` (str), ``outcome`` (``"ok"|"failed"``), ``bytes``
    (int).
    """
    rec = {"name": name, "duration_sec": round(float(duration_sec), 3), **extra}
    with _lock:
        _job_stages.setdefault(int(job_id), []).append(rec)


class stage_timer:
    """Context manager that times a block and records it via :func:`add_stage`.

    Use it to wrap any meaningful unit of work::

        with stage_timer(job_id, "upload_videos", n_files=12):
            for ...: ...

        with stage_timer(job_id, "compute_trajectory", trial="Con02_L1"):
            compute_camera_trajectory(...)

    The ``outcome`` field is set automatically to ``"ok"`` or ``"failed"``
    depending on whether the block exits normally.  Exceptions are
    propagated unchanged.
    """

    def __init__(self, job_id: int, name: str, **extra):
        self.job_id = int(job_id)
        self.name = name
        self.extra = extra
        self._t0 = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        dur = time.perf_counter() - self._t0
        outcome = "failed" if exc_type is not None else "ok"
        add_stage(self.job_id, self.name, dur, outcome=outcome, **self.extra)
        return False    # never suppress exceptions


def finalize_job_record(job_id: int, *, status_override: str | None = None,
                          error_override: str | None = None) -> None:
    """Append a final record for ``job_id`` to the history file and clear
    its in-memory stage list.

    Call from every dispatcher exit point (success, failure, cancelled).
    Reads the canonical job metadata from the DB at flush time so the
    record reflects the same status the Jobs page will show.

    ``status_override`` / ``error_override`` let a caller stamp values
    when the DB hasn't been updated yet (e.g. an early-fail before the
    final UPDATE) or when the caller knows more than the DB does.
    """
    from ..db import get_db_ctx
    try:
        with get_db_ctx() as db:
            row = db.execute(
                "SELECT * FROM jobs WHERE id = ?", (int(job_id),)
            ).fetchone()
        job = dict(row) if row else {}

        params: dict = {}
        if job.get("params_json"):
            try:
                params = json.loads(job["params_json"])
            except (ValueError, TypeError):
                params = {}

        with _lock:
            stages = _job_stages.pop(int(job_id), [])

        # Wall-clock duration from started_at -> finished_at if both are
        # set; otherwise fall back to the sum of stage durations.
        duration: float | None = None
        st = job.get("started_at"); fn = job.get("finished_at")
        if st and fn:
            try:
                t0 = datetime.fromisoformat(str(st).replace(" ", "T"))
                t1 = datetime.fromisoformat(str(fn).replace(" ", "T"))
                duration = (t1 - t0).total_seconds()
            except (ValueError, TypeError):
                duration = None
        if duration is None and stages:
            duration = sum(float(s.get("duration_sec", 0.0)) for s in stages)

        record = {
            "ts":              _now_utc_iso(),
            "git_version":     _git_version(),
            "host":            socket.gethostname(),
            "job_id":          int(job_id),
            "job_type":        job.get("job_type"),
            "status":          status_override or job.get("status"),
            "subject_id":      job.get("subject_id"),
            "remote_host":     job.get("remote_host"),
            "params":          params,
            "error_msg":       error_override or job.get("error_msg"),
            "created_at":      str(job.get("created_at"))  if job.get("created_at")  else None,
            "started_at":      str(job.get("started_at"))  if job.get("started_at")  else None,
            "finished_at":     str(job.get("finished_at")) if job.get("finished_at") else None,
            "duration_sec":    (round(duration, 3) if duration is not None else None),
            "n_stages":        len(stages),
            "stages":          stages,
        }

        path = _history_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
        logger.debug(
            f"job_history: wrote job {job_id} ({record['job_type']}, "
            f"{record['status']}, {record['duration_sec']}s, "
            f"{len(stages)} stages)"
        )
    except Exception as e:
        logger.exception(f"job_history.finalize_job_record({job_id}) failed: {e}")
        # Always clear the in-memory entry so a failure here doesn't
        # leak stage timings across job ids.
        with _lock:
            _job_stages.pop(int(job_id), None)


def read_history(
    limit: int = 500,
    job_type: str | None = None,
    subject_id: int | None = None,
    status: str | None = None,
) -> list[dict]:
    """Read recent records from the history file (newest first).

    Streams the file line-by-line so the memory cost stays bounded even
    for very long histories.  Filters are applied before the limit, so
    ``limit=500`` with ``job_type='preproc'`` always returns the 500
    newest preproc records (if that many exist).
    """
    path = _history_path()
    if not path.exists():
        return []
    keep: list[dict] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if job_type and rec.get("job_type") != job_type:
                    continue
                if subject_id is not None and int(rec.get("subject_id") or -1) != int(subject_id):
                    continue
                if status and rec.get("status") != status:
                    continue
                keep.append(rec)
        # Most recent last in the file -> reverse to newest-first.
        return list(reversed(keep[-int(limit):]))
    except OSError as e:
        logger.warning(f"job_history.read_history: {e}")
        return []


def summary(job_type: str | None = None) -> dict:
    """Aggregate stats from the full history: counts, mean/median
    duration, recent failure rate.  Useful for at-a-glance performance
    comparison after a refactor."""
    path = _history_path()
    if not path.exists():
        return {"count": 0}
    durations: list[float] = []
    n_ok = n_fail = n_cancel = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if job_type and rec.get("job_type") != job_type:
                    continue
                st = rec.get("status")
                if st == "completed": n_ok += 1
                elif st == "failed":  n_fail += 1
                elif st == "cancelled": n_cancel += 1
                d = rec.get("duration_sec")
                if d is not None:
                    durations.append(float(d))
    except OSError:
        return {"count": 0}
    durations.sort()
    n = len(durations)
    median = durations[n // 2] if n else None
    mean = (sum(durations) / n) if n else None
    return {
        "count":    n_ok + n_fail + n_cancel,
        "ok":       n_ok,
        "failed":   n_fail,
        "cancelled": n_cancel,
        "median_duration_sec": (round(median, 3) if median is not None else None),
        "mean_duration_sec":   (round(mean,   3) if mean   is not None else None),
    }
