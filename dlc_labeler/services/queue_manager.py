"""Job queue: one CPU lane and one GPU lane, FIFO within each.

MediaPipe is CPU-bound and DeepLabCut is GPU-bound, so the two lanes run
concurrently — a MediaPipe pass on one subject does not have to wait for
a training run to finish.  Within a lane jobs are serialised, because two
training runs on one GPU just make each other slower.

Everything executes locally.  Each queue item gets a ``jobs`` row (the
unit the log viewer and progress bar read) and a subprocess whose PID is
recorded, so a job outlives the web app restarting.
"""
from __future__ import annotations

import json
import logging
import os
import threading

from ..db import get_db_ctx
from .jobs import registry

logger = logging.getLogger(__name__)


def _pid_alive(pid: int) -> bool:
    """Check whether a process with this PID is still running."""
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)  # signal 0 = existence check
        return True
    except (OSError, ProcessLookupError):
        return False


# Which lane each job type needs.
RESOURCE_MAP = {
    "mediapipe": "cpu",
    "export-frames": "cpu",
    "install-dlc": "cpu",
    "train": "gpu",
    "refine": "gpu",
    "analyze": "gpu",
}

# What the Jobs page offers, in the order a project runs them.
STEP_DEFINITIONS = [
    {"name": "mediapipe", "resource": "cpu", "label": "MediaPipe (Hands)",
     "help": "Detect 21 hand landmarks per camera. Run this first — it "
             "drives the default zoom while you label."},
    {"name": "export-frames", "resource": "cpu", "label": "Export frames to label",
     "help": "Pick frames with DeepLabCut's k-means, cropped around the "
             "MediaPipe hand, and write them as a folder you can send to "
             "someone else to label."},
    {"name": "train", "resource": "gpu", "label": "Train (round 1)",
     "help": "Create the training dataset from committed labels, train "
             "from scratch, then analyze every trial into labels_v1/."},
    {"name": "refine", "resource": "gpu", "label": "Refine (round 2+)",
     "help": "Pick up newly committed correction rounds, resume training, "
             "then analyze into labels_v2/."},
    {"name": "analyze", "resource": "gpu", "label": "Analyze only",
     "help": "Re-run inference with the existing trained model. No training."},
]

_TERMINAL = ("completed", "failed", "cancelled")


class QueueManager:
    """Singleton queue manager with independent CPU and GPU lanes."""

    def __init__(self):
        self._lock = threading.Lock()
        self._drain_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._running: dict[str, int | None] = {"cpu": None, "gpu": None}
        self._job_threads: dict[int, threading.Thread] = {}

    def start(self):
        """Start the drain loop thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._drain_loop, daemon=True)
        self._thread.start()
        logger.info("QueueManager drain loop started")

    # ── Public API ──────────────────────────────────────────────────────

    def enqueue(self, job_type: str, subject_names: list[str],
                gpu_index: int = 0, extra_params: dict | None = None) -> dict:
        """Add a job to its lane. Returns {queue_id, position}.

        One queue item per subject is the caller's job — a MediaPipe run
        over five subjects enqueues five items so each gets its own log,
        its own progress bar and can be cancelled on its own.
        """
        resource = RESOURCE_MAP.get(job_type)
        if not resource:
            raise ValueError(f"Unknown job type: {job_type}")

        target = "local-gpu" if resource == "gpu" else "local-cpu"
        params = dict(extra_params or {})
        if resource == "gpu":
            params.setdefault("gpu_index", int(gpu_index))

        with get_db_ctx() as db:
            row = db.execute(
                "SELECT COALESCE(MAX(position), 0) + 1 AS next_pos FROM job_queue "
                "WHERE resource = ? AND status IN ('queued', 'running')",
                (resource,),
            ).fetchone()
            position = row["next_pos"]

            cur = db.execute(
                """INSERT INTO job_queue
                   (job_type, subject_ids, resource, status, position,
                    execution_target, extra_params_json)
                   VALUES (?, ?, ?, 'queued', ?, ?, ?)""",
                (job_type, json.dumps(subject_names), resource, position,
                 target, json.dumps(params) if params else None),
            )
            queue_id = cur.lastrowid

        self._drain_event.set()
        return {"queue_id": queue_id, "position": position}

    def cancel(self, queue_id: int) -> bool:
        """Cancel a queued or running item."""
        with get_db_ctx() as db:
            item = db.execute("SELECT * FROM job_queue WHERE id = ?",
                              (queue_id,)).fetchone()
        if not item:
            return False
        if item["status"] not in ("queued", "running"):
            return False

        with get_db_ctx() as db:
            db.execute(
                "UPDATE job_queue SET status = 'cancelled', "
                "finished_at = CURRENT_TIMESTAMP WHERE id = ?", (queue_id,))

        if item["status"] == "running":
            with self._lock:
                for lane, running_id in self._running.items():
                    if running_id == queue_id:
                        self._running[lane] = None
            job_id = item.get("job_id")
            if job_id:
                try:
                    registry.cancel(job_id)
                except Exception:
                    logger.warning("Cancel of job %s failed", job_id, exc_info=True)

        self._drain_event.set()
        return True

    def get_state(self) -> dict:
        """Return the queue state the Jobs page renders.

        Scoped to this app's job types throughout: a shared data directory
        means the tables can also hold the full Movement Tracker app's
        work, and showing someone a queued job they cannot cancel or a
        failed job they never started is worse than not showing it.
        """
        owned = sorted(RESOURCE_MAP)
        ph = ",".join("?" * len(owned))
        with get_db_ctx() as db:
            queued = db.execute(
                f"SELECT * FROM job_queue WHERE status = 'queued' "
                f"AND job_type IN ({ph}) "
                f"ORDER BY resource, position", owned).fetchall()
            running = db.execute(
                # COALESCE so a queue row whose jobs row hasn't reported
                # yet still shows the queue-level progress.
                f"SELECT q.*, "
                f"COALESCE(j.progress_pct, q.progress_pct, 0) AS progress_pct, "
                f"j.epoch_info, j.log_path, j.params_json "
                f"FROM job_queue q LEFT JOIN jobs j ON q.job_id = j.id "
                f"WHERE q.status = 'running' AND q.job_type IN ({ph}) "
                f"ORDER BY q.started_at", owned).fetchall()
            history = db.execute(
                f"""SELECT
                       j.id AS job_id,
                       COALESCE(q.job_type, j.job_type) AS job_type,
                       COALESCE(q.subject_ids, json_array(COALESCE(s.name, ''))) AS subject_ids,
                       COALESCE(q.resource,
                           CASE WHEN j.job_type IN ('train','refine','analyze')
                                THEN 'gpu' ELSE 'cpu' END) AS resource,
                       j.status,
                       j.log_path,
                       j.progress_pct,
                       j.params_json,
                       j.started_at,
                       COALESCE(q.finished_at, j.finished_at) AS finished_at,
                       COALESCE(q.error_msg, j.error_msg) AS error_msg
                   FROM jobs j
                   LEFT JOIN subjects s ON j.subject_id = s.id
                   LEFT JOIN job_queue q ON q.job_id = j.id
                   WHERE j.status IN ('completed', 'failed', 'cancelled')
                     AND j.job_type IN ({ph})
                   ORDER BY COALESCE(q.finished_at, j.finished_at) DESC
                   LIMIT 50""", owned).fetchall()

        return {
            "cpu_queue": [q for q in queued if q["resource"] == "cpu"],
            "gpu_queue": [q for q in queued if q["resource"] == "gpu"],
            "running": running,
            "history": history,
        }

    def recover(self):
        """Reconcile queue rows left 'running' by a previous app process.

        The app restarts on every code edit under ``--reload``, and a
        two-hour training run must not be lost to that.  A subprocess
        whose PID is still alive is re-tracked and keeps reporting
        progress; one whose process is gone is marked failed so the lane
        frees up instead of blocking the queue forever.
        """
        owned = sorted(RESOURCE_MAP)
        placeholders = ",".join("?" * len(owned))
        with get_db_ctx() as db:
            stale = db.execute(
                f"SELECT * FROM job_queue WHERE status = 'running' "
                f"AND job_type IN ({placeholders})", owned).fetchall()

        for item in stale:
            queue_id = item["id"]
            job_id = item.get("job_id")
            job = None
            if job_id:
                with get_db_ctx() as db:
                    job = db.execute(
                        "SELECT status, pid, log_path FROM jobs WHERE id = ?",
                        (job_id,)).fetchone()

            if job and job["status"] in _TERMINAL:
                with get_db_ctx() as db:
                    db.execute(
                        "UPDATE job_queue SET status = ?, "
                        "finished_at = CURRENT_TIMESTAMP WHERE id = ?",
                        (job["status"], queue_id))
                logger.info("Queue item %s: job %s had already %s",
                            queue_id, job_id, job["status"])
                continue

            pid = (job or {}).get("pid")
            if job and _pid_alive(pid):
                lane = "gpu" if item["resource"] == "gpu" else "cpu"
                with self._lock:
                    self._running[lane] = queue_id
                try:
                    registry.reattach(job_id, pid, job.get("log_path") or "")
                except Exception as e:
                    logger.warning("Could not reattach to PID %s: %s", pid, e)
                logger.info("Queue item %s: subprocess %s still alive, re-tracking",
                            queue_id, pid)
                continue

            msg = "App restarted (process exited)"
            with get_db_ctx() as db:
                if job_id:
                    db.execute(
                        "UPDATE jobs SET status = 'failed', error_msg = ?, "
                        "finished_at = CURRENT_TIMESTAMP WHERE id = ?",
                        (msg, job_id))
                db.execute(
                    "UPDATE job_queue SET status = 'failed', error_msg = ?, "
                    "finished_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (msg, queue_id))
            logger.info("Queue item %s: marked failed (%s)", queue_id, msg)

        self._drain_event.set()

    # ── Drain loop ──────────────────────────────────────────────────────

    def _drain_loop(self):
        while True:
            self._drain_event.wait(timeout=3)
            self._drain_event.clear()
            for lane in ("gpu", "cpu"):
                try:
                    self._try_launch_lane(lane)
                except Exception:
                    logger.exception("Error draining %s lane", lane)

    def _try_launch_lane(self, resource: str):
        """If the lane is free, pop and launch the next queued item."""
        with self._lock:
            running_id = self._running[resource]
            if running_id is not None:
                if self._lane_still_busy(running_id):
                    return
                self._running[resource] = None

        # Only ever pick up job types this app owns.  The queue table lives
        # in the shared database, and a data directory can legitimately be
        # shared with the full Movement Tracker app — which queues job types
        # this fork has no executor for (hrnet, skeleton_*, deidentify,
        # preproc).  Without this filter, whichever drain loop reached the
        # row first would claim the other app's job and fail it.
        owned = sorted(RESOURCE_MAP)
        placeholders = ",".join("?" * len(owned))
        with get_db_ctx() as db:
            next_item = db.execute(
                f"SELECT * FROM job_queue WHERE resource = ? AND status = 'queued' "
                f"AND job_type IN ({placeholders}) "
                f"ORDER BY position LIMIT 1", (resource, *owned)).fetchone()

        if next_item:
            self._launch_item(next_item)

    def _lane_still_busy(self, running_id: int) -> bool:
        """True while ``running_id`` is genuinely occupying its lane.

        A queue row can be left reading 'running' after its underlying
        job finished; syncing the row here is what frees the lane.
        """
        with get_db_ctx() as db:
            item = db.execute(
                "SELECT status, job_id FROM job_queue WHERE id = ?",
                (running_id,)).fetchone()
        if not item or item["status"] != "running":
            return False

        job_id = item.get("job_id")
        if not job_id:
            return False

        with get_db_ctx() as db:
            job = db.execute(
                "SELECT status, error_msg FROM jobs WHERE id = ?",
                (job_id,)).fetchone()
        if job and job["status"] in _TERMINAL:
            with get_db_ctx() as db:
                db.execute(
                    "UPDATE job_queue SET status = ?, error_msg = ?, "
                    "finished_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (job["status"], job.get("error_msg"), running_id))
            logger.info("Queue item %s: synced to '%s' from job %s",
                        running_id, job["status"], job_id)
            return False
        return True

    def _launch_item(self, queue_item: dict):
        """Create the jobs row for a queue item and dispatch it."""
        from ..config import get_settings

        queue_id = queue_item["id"]
        job_type = queue_item["job_type"]
        resource = queue_item["resource"]
        subject_names = json.loads(queue_item["subject_ids"] or "[]")

        extra_params: dict = {}
        if queue_item.get("extra_params_json"):
            try:
                extra_params = json.loads(queue_item["extra_params_json"])
            except (ValueError, TypeError):
                pass

        settings = get_settings()

        # install-dlc has no subject; everything else needs one.
        subject_id = None
        if job_type != "install-dlc":
            if not subject_names:
                self._fail_queue_item(queue_id, "No subject specified")
                return
            with get_db_ctx() as db:
                subj = db.execute("SELECT id FROM subjects WHERE name = ?",
                                  (subject_names[0],)).fetchone()
            if not subj:
                self._fail_queue_item(
                    queue_id, f"Subject not found: {subject_names[0]}")
                return
            subject_id = subj["id"]
        else:
            # jobs.subject_id is NOT NULL, so borrow any row; the Jobs
            # page keys install-dlc off job_type, not its subject.
            with get_db_ctx() as db:
                any_subj = db.execute(
                    "SELECT id FROM subjects ORDER BY id LIMIT 1").fetchone()
            if not any_subj:
                self._fail_queue_item(
                    queue_id, "Add a subject before installing DeepLabCut")
                return
            subject_id = any_subj["id"]

        log_dir = settings.dlc_path / ".logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        label = subject_names[0] if subject_names else "app"
        log_path = str(log_dir / f"{job_type}_{label}_{queue_id}.log")

        with get_db_ctx() as db:
            cur = db.execute(
                """INSERT INTO jobs
                       (subject_id, job_type, status, log_path, params_json)
                   VALUES (?, ?, 'pending', ?, ?)""",
                (subject_id, job_type, log_path,
                 json.dumps(extra_params) if extra_params else None),
            )
            job_id = cur.lastrowid
            db.execute(
                "UPDATE job_queue SET status = 'running', job_id = ?, "
                "started_at = CURRENT_TIMESTAMP WHERE id = ?",
                (job_id, queue_id))

        with self._lock:
            self._running[resource] = queue_id

        thread = threading.Thread(
            target=self._run_job,
            args=(queue_id, job_id, job_type, subject_names, log_path,
                  extra_params),
            daemon=True,
        )
        thread.start()
        self._job_threads[queue_id] = thread

    def _run_job(self, queue_id, job_id, job_type, subject_names, log_path,
                 extra_params):
        """Dispatch one job, then make sure its lane is released.

        The executor returns as soon as the subprocess is launched;
        ``registry`` owns the jobs row from then on and the drain loop
        notices the terminal status on its next pass.  So the only
        failures handled here are failures to launch at all.
        """
        from .local_executor import local_executor

        subject_name = subject_names[0] if subject_names else ""
        try:
            if job_type == "mediapipe":
                local_executor.execute_mediapipe(
                    subject_name, job_id, log_path,
                    static_image_mode=bool(extra_params.get("static_image_mode")),
                    trial_idx=extra_params.get("trial_idx"),
                    reverse=bool(extra_params.get("reverse")),
                    use_bbox=extra_params.get("use_bbox", True),
                )
            elif job_type == "export-frames":
                local_executor.execute_export_frames(
                    subject_name, job_id, log_path,
                    n_per_trial=int(extra_params.get("n_per_trial", 20)),
                    cameras=extra_params.get("cameras") or None,
                    dest_dir=extra_params.get("dest_dir") or None,
                    package_name=extra_params.get("package_name") or None,
                    step=int(extra_params.get("step", 1)),
                    seed=extra_params.get("seed"),
                )
            elif job_type in ("train", "refine", "analyze"):
                local_executor.execute_dlc(
                    job_type, subject_name, job_id, log_path,
                    gpu_index=int(extra_params.get("gpu_index", 0)),
                )
            elif job_type == "install-dlc":
                local_executor.execute_install_dlc(job_id, log_path)
            else:
                raise ValueError(f"Unhandled job type: {job_type}")
        except Exception as e:
            logger.exception("Launch of %s (job %s) failed", job_type, job_id)
            with get_db_ctx() as db:
                db.execute(
                    "UPDATE jobs SET status = 'failed', error_msg = ?, "
                    "finished_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (str(e), job_id))
                db.execute(
                    "UPDATE job_queue SET status = 'failed', error_msg = ?, "
                    "finished_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (str(e), queue_id))
            with self._lock:
                for lane, running_id in self._running.items():
                    if running_id == queue_id:
                        self._running[lane] = None
        finally:
            self._job_threads.pop(queue_id, None)
            self._drain_event.set()

    @staticmethod
    def _fail_queue_item(queue_id: int, message: str):
        logger.error("Queue item %s: %s", queue_id, message)
        with get_db_ctx() as db:
            db.execute(
                "UPDATE job_queue SET status = 'failed', error_msg = ?, "
                "finished_at = CURRENT_TIMESTAMP WHERE id = ?",
                (message, queue_id))


# Global singleton
queue_manager = QueueManager()
