"""Job queue manager: GPU and CPU lanes with FIFO scheduling."""
from __future__ import annotations

import json
import logging
import os
import signal
import threading
import time
from pathlib import Path

from ..db import get_db_ctx

logger = logging.getLogger(__name__)


def _pid_alive(pid: int) -> bool:
    """Check if a process with given PID is still running."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)  # signal 0 = existence check
        return True
    except (OSError, ProcessLookupError):
        return False

# Map job types to resource lanes
RESOURCE_MAP = {
    "train": "gpu",
    "refine": "gpu",
    "analyze": "gpu",
    "analyze_v1": "gpu",
    "analyze_v2": "gpu",
    "mediapipe": "cpu",
    "blur": "cpu",
    "mediapipe+blur": "cpu",
    "pose": "cpu",
    "deidentify": "cpu",
}

STEP_DEFINITIONS = [
    {"name": "mediapipe", "resource": "cpu", "label": "MediaPipe (Hands)"},
    {"name": "pose", "resource": "cpu", "label": "Pose Detection"},
    {"name": "deidentify", "resource": "cpu", "label": "Deidentify (Render)"},
    {"name": "blur", "resource": "cpu", "label": "Face Blur"},
    {"name": "mediapipe+blur", "resource": "cpu", "label": "MediaPipe + Blur"},
    {"name": "train", "resource": "gpu", "label": "Train"},
    {"name": "refine", "resource": "gpu", "label": "Refine"},
    {"name": "analyze_v1", "resource": "gpu", "label": "Analyze v1"},
    {"name": "analyze_v2", "resource": "gpu", "label": "Analyze v2"},
]


class QueueManager:
    """Singleton queue manager with independent GPU and CPU lanes.

    GPU and CPU jobs can run concurrently (different lanes), but within
    each lane jobs are serialized in FIFO order.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._drain_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._running_gpu: int | None = None  # queue_id
        self._running_cpu: int | None = None  # queue_id
        self._job_threads: dict[int, threading.Thread] = {}

    def start(self):
        """Start the drain loop thread."""
        self._thread = threading.Thread(target=self._drain_loop, daemon=True)
        self._thread.start()
        logger.info("QueueManager drain loop started")

    def enqueue(self, job_type: str, subject_ids: list[int], subject_names: list[str],
                execution_target: str = "remote", gpu_index: int = 0) -> dict:
        """Add a job to the queue. Returns {queue_id, position}.

        Args:
            job_type: Type of job (mediapipe, blur, train, analyze_v1, analyze_v2, etc.)
            subject_ids: List of subject IDs from database
            subject_names: List of subject names
            execution_target: Where to execute - "local-cpu", "local-gpu", or "remote"
            gpu_index: Which GPU to use if execution_target is "local-gpu"
        """
        resource = RESOURCE_MAP.get(job_type)
        if not resource:
            raise ValueError(f"Unknown job type: {job_type}")

        with get_db_ctx() as db:
            # Get next position for this resource
            row = db.execute(
                "SELECT COALESCE(MAX(position), 0) + 1 AS next_pos FROM job_queue "
                "WHERE resource = ? AND status IN ('queued', 'running')",
                (resource,),
            ).fetchone()
            position = row["next_pos"]

            db.execute(
                """INSERT INTO job_queue (job_type, subject_ids, resource, status, position, execution_target)
                   VALUES (?, ?, ?, 'queued', ?, ?)""",
                (job_type, json.dumps(subject_names), resource, position, execution_target),
            )
            queue_id = db.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]

            # Store gpu_index in the job_queue record if using local GPU
            if execution_target == "local-gpu":
                db.execute(
                    "UPDATE job_queue SET resource = ? WHERE id = ?",
                    (f"gpu_{gpu_index}", queue_id),
                )

        # Signal the drain thread
        self._drain_event.set()

        return {"queue_id": queue_id, "position": position}

    def cancel(self, queue_id: int) -> bool:
        """Cancel a queued or running item."""
        with get_db_ctx() as db:
            item = db.execute(
                "SELECT * FROM job_queue WHERE id = ?", (queue_id,)
            ).fetchone()

        if not item:
            return False

        if item["status"] == "queued":
            with get_db_ctx() as db:
                db.execute(
                    "UPDATE job_queue SET status = 'cancelled', finished_at = CURRENT_TIMESTAMP "
                    "WHERE id = ?",
                    (queue_id,),
                )
            return True

        if item["status"] == "running":
            # Mark cancelled in DB immediately (non-blocking)
            with get_db_ctx() as db:
                db.execute(
                    "UPDATE job_queue SET status = 'cancelled', finished_at = CURRENT_TIMESTAMP "
                    "WHERE id = ?",
                    (queue_id,),
                )

            with self._lock:
                if self._running_gpu == queue_id:
                    self._running_gpu = None
                if self._running_cpu == queue_id:
                    self._running_cpu = None

            # Cancel the underlying job in background (SSH kill can block)
            job_id = item.get("job_id")
            if job_id:
                def _bg_cancel(jid):
                    try:
                        from .jobs import registry
                        registry.cancel(jid)
                    except Exception:
                        logger.warning("Background cancel of job %s failed", jid, exc_info=True)
                threading.Thread(target=_bg_cancel, args=(job_id,), daemon=True).start()

            # Trigger drain to pick up next item
            self._drain_event.set()
            return True

        return False

    def get_state(self) -> dict:
        """Return current queue state for the frontend."""
        with get_db_ctx() as db:
            gpu_queue = db.execute(
                "SELECT * FROM job_queue WHERE resource = 'gpu' AND status = 'queued' "
                "ORDER BY position",
            ).fetchall()
            cpu_queue = db.execute(
                "SELECT * FROM job_queue WHERE resource = 'cpu' AND status = 'queued' "
                "ORDER BY position",
            ).fetchall()
            running = db.execute(
                "SELECT q.*, j.progress_pct, j.epoch_info, j.log_path FROM job_queue q "
                "LEFT JOIN jobs j ON q.job_id = j.id "
                "WHERE q.status = 'running' ORDER BY q.started_at",
            ).fetchall()
            history = db.execute(
                "SELECT q.*, j.progress_pct FROM job_queue q "
                "LEFT JOIN jobs j ON q.job_id = j.id "
                "WHERE q.status IN ('completed', 'failed', 'cancelled') "
                "ORDER BY q.finished_at DESC LIMIT 50",
            ).fetchall()

        return {
            "gpu_queue": gpu_queue,
            "cpu_queue": cpu_queue,
            "running": running,
            "history": history,
        }

    def recover(self):
        """On startup, handle stale running items from a prior server session."""
        with get_db_ctx() as db:
            stale = db.execute(
                "SELECT * FROM job_queue WHERE status = 'running'"
            ).fetchall()

        for item in stale:
            job_id = item.get("job_id")
            if job_id:
                # Check if the underlying job is still alive
                with get_db_ctx() as db:
                    job = db.execute(
                        "SELECT status FROM jobs WHERE id = ?", (job_id,)
                    ).fetchone()
                if job and job["status"] in ("completed", "failed", "cancelled"):
                    # Underlying job finished — sync queue status
                    new_status = job["status"]
                    with get_db_ctx() as db:
                        db.execute(
                            "UPDATE job_queue SET status = ?, finished_at = CURRENT_TIMESTAMP "
                            "WHERE id = ?",
                            (new_status, item["id"]),
                        )
                    logger.info(f"Queue item {item['id']}: underlying job {job_id} was {new_status}")
                    continue
                if job and job["status"] in ("running", "pending"):
                    exec_target = item.get("execution_target", "remote")
                    if exec_target.startswith("local"):
                        # Local jobs now run as subprocesses — check if PID is still alive
                        pid = job.get("pid") if job else None
                        if pid and _pid_alive(pid):
                            # Subprocess survived! Re-track it and resume monitoring
                            resource = item.get("resource", "cpu")
                            with self._lock:
                                if resource == "gpu":
                                    self._running_gpu = item["id"]
                                else:
                                    self._running_cpu = item["id"]
                            # Re-attach the registry monitor to the running process
                            try:
                                from .worker import parse_worker_progress
                                registry.reattach(job_id, pid, job.get("log_path", ""),
                                                  progress_parser=parse_worker_progress)
                            except Exception as e:
                                logger.warning(f"Could not reattach to PID {pid}: {e}")
                            logger.info(f"Queue item {item['id']}: local subprocess {pid} still alive, re-tracking")
                            continue
                        else:
                            # Process is dead — mark as failed
                            with get_db_ctx() as db:
                                db.execute(
                                    "UPDATE jobs SET status = 'failed', error_msg = 'Server restarted (process exited)', "
                                    "finished_at = CURRENT_TIMESTAMP WHERE id = ?",
                                    (job_id,),
                                )
                                db.execute(
                                    "UPDATE job_queue SET status = 'failed', error_msg = 'Server restarted (process exited)', "
                                    "finished_at = CURRENT_TIMESTAMP WHERE id = ?",
                                    (item["id"],),
                                )
                            logger.info(f"Queue item {item['id']}: local job {job_id} (PID {pid}) not alive, marked failed")
                            continue
                    else:
                        # Remote job: might still be alive on the remote host — re-track
                        resource = item.get("resource", "gpu")
                        with self._lock:
                            if resource == "gpu":
                                self._running_gpu = item["id"]
                            else:
                                self._running_cpu = item["id"]
                        logger.info(f"Queue item {item['id']}: remote job {job_id} still running, re-tracking")
                        continue

            # If we can't determine state, mark failed
            with get_db_ctx() as db:
                db.execute(
                    "UPDATE job_queue SET status = 'failed', error_msg = 'Server restarted', "
                    "finished_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (item["id"],),
                )
            logger.info(f"Queue item {item['id']}: marked failed (server restarted)")

    def _drain_loop(self):
        """Main loop: check lanes and launch next queued items."""
        while True:
            self._drain_event.wait(timeout=3)
            self._drain_event.clear()

            try:
                self._try_launch_lane("gpu")
                self._try_launch_lane("cpu")
            except Exception:
                logger.exception("Error in drain loop")

    def _try_launch_lane(self, resource: str):
        """If the lane is free, pop and launch the next queued item."""
        with self._lock:
            running_id = self._running_gpu if resource == "gpu" else self._running_cpu
            if running_id is not None:
                # Check if still actually running
                with get_db_ctx() as db:
                    item = db.execute(
                        "SELECT status, job_id FROM job_queue WHERE id = ?", (running_id,)
                    ).fetchone()
                if item and item["status"] == "running":
                    # Queue item thinks it's running — verify the underlying job
                    job_id = item.get("job_id")
                    if job_id:
                        with get_db_ctx() as db:
                            job = db.execute(
                                "SELECT status, error_msg FROM jobs WHERE id = ?", (job_id,)
                            ).fetchone()
                        if job and job["status"] in ("completed", "failed", "cancelled"):
                            # Underlying job finished (e.g. app.py resume thread) — sync queue
                            with get_db_ctx() as db:
                                db.execute(
                                    "UPDATE job_queue SET status = ?, finished_at = CURRENT_TIMESTAMP, "
                                    "error_msg = ? WHERE id = ?",
                                    (job["status"], job.get("error_msg"), running_id),
                                )
                            logger.info(
                                f"Queue item {running_id}: synced status to '{job['status']}' from job {job_id}"
                            )
                        else:
                            return  # Lane busy — job still running
                    else:
                        return  # Lane busy — no job to check
                # Lane freed — clear it
                if resource == "gpu":
                    self._running_gpu = None
                else:
                    self._running_cpu = None

        # Find next queued item
        with get_db_ctx() as db:
            next_item = db.execute(
                "SELECT * FROM job_queue WHERE resource = ? AND status = 'queued' "
                "ORDER BY position LIMIT 1",
                (resource,),
            ).fetchone()

        if not next_item:
            return

        # Launch it
        self._launch_item(next_item)

    def _launch_item(self, queue_item: dict):
        """Create a jobs row and launch the appropriate local or remote function."""
        queue_id = queue_item["id"]
        job_type = queue_item["job_type"]
        subject_names = json.loads(queue_item["subject_ids"])
        resource = queue_item["resource"]
        execution_target = queue_item.get("execution_target", "remote")

        from ..config import get_settings
        from .jobs import registry

        settings = get_settings()

        # Parse GPU index from resource if using local GPU
        gpu_index = 0
        if resource.startswith("gpu_"):
            gpu_index = int(resource.split("_", 1)[1])

        # For remote execution, require remote config
        remote_cfg = None
        if execution_target == "remote":
            remote_cfg = settings.get_remote_config()
            if not remote_cfg:
                with get_db_ctx() as db:
                    db.execute(
                        "UPDATE job_queue SET status = 'failed', error_msg = 'Remote not configured', "
                        "finished_at = CURRENT_TIMESTAMP WHERE id = ?",
                        (queue_id,),
                    )
                self._drain_event.set()
                return

        # Resolve subject IDs from names
        with get_db_ctx() as db:
            if subject_names:
                placeholders = ",".join("?" * len(subject_names))
                subjects = db.execute(
                    f"SELECT id, name FROM subjects WHERE name IN ({placeholders})",
                    subject_names,
                ).fetchall()
            else:
                subjects = db.execute("SELECT id, name FROM subjects").fetchall()

        if not subjects:
            with get_db_ctx() as db:
                db.execute(
                    "UPDATE job_queue SET status = 'failed', error_msg = 'No subjects found', "
                    "finished_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (queue_id,),
                )
            self._drain_event.set()
            return

        first_subject_id = subjects[0]["id"]

        # Create the jobs record
        log_dir = settings.dlc_path / ".logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        subj_label = "_".join(s["name"] for s in subjects[:3])
        if len(subjects) > 3:
            subj_label += f"_+{len(subjects) - 3}"
        log_path = str(log_dir / f"job_queue_{job_type}_{subj_label}_{queue_id}.log")

        # Set remote_host based on execution target
        remote_host = None
        if execution_target == "remote":
            remote_host = remote_cfg.host if remote_cfg else None
        elif execution_target in ("local-cpu", "local-gpu"):
            remote_host = "localhost"

        with get_db_ctx() as db:
            db.execute(
                """INSERT INTO jobs (subject_id, job_type, status, remote_host, log_path)
                   VALUES (?, ?, 'pending', ?, ?)""",
                (first_subject_id, job_type, remote_host, log_path),
            )
            job = db.execute("SELECT * FROM jobs ORDER BY id DESC LIMIT 1").fetchone()

        job_id = job["id"]

        # Update queue item with job_id and status
        with get_db_ctx() as db:
            db.execute(
                "UPDATE job_queue SET status = 'running', job_id = ?, started_at = CURRENT_TIMESTAMP "
                "WHERE id = ?",
                (job_id, queue_id),
            )
            db.execute(
                "UPDATE jobs SET status = 'running', started_at = CURRENT_TIMESTAMP WHERE id = ?",
                (job_id,),
            )

        with self._lock:
            if resource.startswith("gpu"):
                self._running_gpu = queue_id
            else:
                self._running_cpu = queue_id

        # Launch in a thread
        thread = threading.Thread(
            target=self._run_job,
            args=(queue_id, job_id, job_type, subjects, remote_cfg, settings, log_path,
                  execution_target, gpu_index),
            daemon=True,
        )
        thread.start()
        self._job_threads[queue_id] = thread

    def _run_job(self, queue_id, job_id, job_type, subjects, remote_cfg, settings, log_path,
                  execution_target="remote", gpu_index=0):
        """Run a job and update queue status on completion.

        Args:
            execution_target: "local-cpu", "local-gpu", or "remote"
            gpu_index: GPU device index if execution_target is "local-gpu"
        """
        from .jobs import registry
        from .remote import remote_preprocess_batch, remote_train_monitor
        from ..services.dlc_wrapper import fix_project_path
        from .local_executor import local_executor

        try:
            subject_names = [s["name"] for s in subjects]

            # Route based on execution target
            if execution_target in ("local-cpu", "local-gpu"):
                # Local execution
                if job_type in ("mediapipe", "blur", "mediapipe+blur", "pose", "deidentify"):
                    # CPU lane: preprocessing
                    if job_type == "mediapipe":
                        local_executor.execute_mediapipe(subject_names[0], job_id, log_path)
                    elif job_type == "pose":
                        local_executor.execute_pose(subject_names[0], job_id, log_path)
                    elif job_type == "blur":
                        local_executor.execute_blur(subject_names, job_id, log_path)
                    elif job_type == "deidentify":
                        local_executor.execute_deidentify(subject_names[0], job_id, log_path)
                    elif job_type == "mediapipe+blur":
                        # Run both sequentially
                        local_executor.execute_mediapipe(subject_names[0], job_id, log_path)
                        # Wait a bit for first to complete
                        import time
                        time.sleep(1)
                        local_executor.execute_blur(subject_names, job_id, log_path)

                elif job_type == "train":
                    # GPU lane: training on local GPU
                    local_executor.execute_train(subject_names[0], gpu_index, job_id, log_path)

                elif job_type == "refine":
                    # GPU lane: recreate dataset + resume training on local GPU
                    local_executor.execute_refine(subject_names[0], gpu_index, job_id, log_path)

                elif job_type in ("analyze_v1", "analyze_v2"):
                    # GPU lane: analyze on local GPU
                    local_executor.execute_analyze(subject_names[0], gpu_index, job_id, log_path)

                # Wait for the subprocess to finish (registry._monitor thread)
                # All local jobs now run as subprocesses tracked by the registry
                monitor_thread = registry._threads.get(job_id)
                if monitor_thread:
                    monitor_thread.join()  # blocks until subprocess completes

            else:
                # Remote execution (existing logic)
                if job_type in ("mediapipe", "blur", "mediapipe+blur"):
                    # CPU lane: preprocessing
                    steps = []
                    if job_type in ("mediapipe", "mediapipe+blur"):
                        steps.append("mediapipe")
                    if job_type in ("blur", "mediapipe+blur"):
                        steps.append("blur")

                    remote_preprocess_batch(
                        job_id=job_id,
                        cfg=remote_cfg,
                        steps=steps,
                        subjects=subject_names,
                        log_path=log_path,
                        registry=registry,
                        force=True,
                    )

                elif job_type in ("train", "refine"):
                    # GPU lane: training (one subject at a time through the queue)
                    subject = subjects[0]
                    subject_name = subject["name"]
                    cam_names = settings.camera_names

                    def on_complete(jid, returncode):
                        if returncode == 0:
                            try:
                                fix_project_path(subject_name)
                            except Exception:
                                pass
                            with get_db_ctx() as db:
                                db.execute(
                                    "UPDATE subjects SET stage = 'trained', "
                                    "updated_at = CURRENT_TIMESTAMP WHERE name = ?",
                                    (subject_name,),
                                )

                    remote_train_monitor(
                        job_id=job_id,
                        cfg=remote_cfg,
                        local_dlc_dir=settings.dlc_path / subject_name,
                        subject_name=subject_name,
                        log_path=log_path,
                        progress_parser=None,
                        on_complete=on_complete,
                        registry=registry,
                        cam_names=cam_names,
                        force_create_dataset=(job_type == "refine"),
                        labels_dir_name="labels_v2" if job_type == "refine" else "labels_v1",
                    )

                elif job_type in ("analyze_v1", "analyze_v2"):
                    # GPU lane: analyze
                    subject = subjects[0]
                    subject_name = subject["name"]
                    cam_names = settings.camera_names
                    is_v1 = job_type == "analyze_v1"
                    labels_dir_name = "labels_v1" if is_v1 else "labels_v2"
                    analyze_iteration = 0 if is_v1 else None

                    def on_complete(jid, returncode):
                        if returncode == 0:
                            try:
                                fix_project_path(subject_name)
                            except Exception:
                                pass
                            with get_db_ctx() as db:
                                db.execute(
                                    "UPDATE subjects SET stage = 'analyzed', "
                                    "updated_at = CURRENT_TIMESTAMP WHERE name = ?",
                                    (subject_name,),
                                )

                    remote_train_monitor(
                        job_id=job_id,
                        cfg=remote_cfg,
                        local_dlc_dir=settings.dlc_path / subject_name,
                        subject_name=subject_name,
                        log_path=log_path,
                        progress_parser=None,
                        on_complete=on_complete,
                        registry=registry,
                        cam_names=cam_names,
                        skip_train=True,
                        labels_dir_name=labels_dir_name,
                        iteration=analyze_iteration,
                    )

            # Check final job status
            with get_db_ctx() as db:
                job = db.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()

            final_status = job["status"] if job else "failed"
            with get_db_ctx() as db:
                db.execute(
                    "UPDATE job_queue SET status = ?, finished_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (final_status, queue_id),
                )

        except Exception as e:
            logger.exception(f"Queue item {queue_id} failed")
            with get_db_ctx() as db:
                db.execute(
                    "UPDATE job_queue SET status = 'failed', error_msg = ?, "
                    "finished_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (str(e)[:500], queue_id),
                )
        finally:
            with self._lock:
                if self._running_gpu == queue_id:
                    self._running_gpu = None
                if self._running_cpu == queue_id:
                    self._running_cpu = None
            self._job_threads.pop(queue_id, None)
            # Signal drain thread to check for next item
            self._drain_event.set()


# Global singleton
queue_manager = QueueManager()
