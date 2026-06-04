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
    "vision": "cpu",
    "blur": "cpu",
    "mediapipe+blur": "cpu",
    "pose": "cpu",
    "deidentify": "cpu",
    "hrnet": "gpu",
    "skeleton_v1": "cpu",
    "skeleton_v2": "cpu",
    "skeleton_v3": "cpu",
    "stereo_correct": "cpu",
    # Per-trial pre-processing: camera trajectory + background/mask bake.
    # Pure CPU (OpenCV + numpy + ffmpeg).  No GPU needed.
    "preproc": "cpu",
}

STEP_DEFINITIONS = [
    {"name": "mediapipe", "resource": "cpu", "label": "MediaPipe (Hands)"},
    {"name": "vision", "resource": "cpu", "label": "Vision (Hands)"},
    {"name": "pose", "resource": "cpu", "label": "Pose Detection"},
    {"name": "preproc",    "resource": "cpu", "label": "Preproc (Trajectory + Mask)"},
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
        self._running_cpu: int | None = None  # queue_id (local CPU)
        self._running_remote_cpu: int | None = None  # queue_id (remote CPU)
        self._job_threads: dict[int, threading.Thread] = {}

    def start(self):
        """Start the drain loop thread."""
        self._thread = threading.Thread(target=self._drain_loop, daemon=True)
        self._thread.start()
        logger.info("QueueManager drain loop started")

    def enqueue(self, job_type: str, subject_ids: list[int], subject_names: list[str],
                execution_target: str = "remote", gpu_index: int = 0,
                extra_params: dict | None = None) -> dict:
        """Add a job to the queue. Returns {queue_id, position}.

        Args:
            job_type: Type of job (mediapipe, blur, train, analyze_v1, analyze_v2, etc.)
            subject_ids: List of subject IDs from database
            subject_names: List of subject names
            execution_target: Where to execute - "local-cpu", "local-gpu", or "remote"
            gpu_index: Which GPU to use if execution_target is "local-gpu"
            extra_params: Optional per-job params (e.g. {"trial_idx": 0, "trial_name": "L1"})
        """
        resource = RESOURCE_MAP.get(job_type)
        if not resource:
            raise ValueError(f"Unknown job type: {job_type}")

        extra_params_json = json.dumps(extra_params) if extra_params else None

        with get_db_ctx() as db:
            # Get next position for this resource
            row = db.execute(
                "SELECT COALESCE(MAX(position), 0) + 1 AS next_pos FROM job_queue "
                "WHERE resource = ? AND status IN ('queued', 'running')",
                (resource,),
            ).fetchone()
            position = row["next_pos"]

            db.execute(
                """INSERT INTO job_queue
                   (job_type, subject_ids, resource, status, position, execution_target, extra_params_json)
                   VALUES (?, ?, ?, 'queued', ?, ?, ?)""",
                (job_type, json.dumps(subject_names), resource, position, execution_target,
                 extra_params_json),
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
                if self._running_remote_cpu == queue_id:
                    self._running_remote_cpu = None

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
                "SELECT q.*, j.progress_pct, j.epoch_info, j.log_path, j.params_json, "
                "j.tmux_session FROM job_queue q "
                "LEFT JOIN jobs j ON q.job_id = j.id "
                "WHERE q.status = 'running' ORDER BY q.started_at",
            ).fetchall()
            # History: start from jobs table so jobs submitted outside the queue
            # (e.g. from the Analyze page) also appear. LEFT JOIN job_queue for
            # execution_target, subject_ids, and queue-level metadata.
            history = db.execute(
                """SELECT
                    j.id AS job_id,
                    COALESCE(q.job_type, j.job_type) AS job_type,
                    COALESCE(q.subject_ids, json_array(COALESCE(s.name, ''))) AS subject_ids,
                    COALESCE(q.resource,
                        CASE WHEN j.job_type IN ('train', 'analyze_v1', 'analyze_v2') THEN 'gpu'
                             ELSE 'cpu' END
                    ) AS resource,
                    j.status,
                    COALESCE(q.execution_target,
                        CASE WHEN j.remote_host IS NOT NULL AND j.remote_host != 'localhost'
                             THEN 'remote' ELSE 'local-cpu' END
                    ) AS execution_target,
                    COALESCE(q.finished_at, j.finished_at) AS finished_at,
                    COALESCE(q.error_msg, j.error_msg) AS error_msg,
                    j.progress_pct,
                    j.params_json,
                    j.started_at
                FROM jobs j
                LEFT JOIN subjects s ON j.subject_id = s.id
                LEFT JOIN job_queue q ON q.job_id = j.id
                WHERE j.status IN ('completed', 'failed', 'cancelled')
                ORDER BY COALESCE(q.finished_at, j.finished_at) DESC LIMIT 50""",
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
                                elif item.get("execution_target") == "remote":
                                    self._running_remote_cpu = item["id"]
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
                            elif item.get("execution_target") == "remote":
                                self._running_remote_cpu = item["id"]
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
                self._try_launch_lane("cpu", execution_target="local-cpu")
                self._try_launch_lane("cpu", execution_target="remote")
            except Exception:
                logger.exception("Error in drain loop")

    def _try_launch_lane(self, resource: str, execution_target: str | None = None):
        """If the lane is free, pop and launch the next queued item."""
        with self._lock:
            if execution_target == "remote":
                running_id = self._running_remote_cpu
            elif resource == "gpu":
                running_id = self._running_gpu
            else:
                running_id = self._running_cpu
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
                if execution_target == "remote":
                    self._running_remote_cpu = None
                elif resource == "gpu":
                    self._running_gpu = None
                else:
                    self._running_cpu = None

        # Find next queued item (filter by execution_target for CPU)
        with get_db_ctx() as db:
            if execution_target:
                next_item = db.execute(
                    "SELECT * FROM job_queue WHERE resource = ? AND execution_target = ? AND status = 'queued' "
                    "ORDER BY position LIMIT 1",
                    (resource, execution_target),
                ).fetchone()
            else:
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

        # Parse per-job params (trial_idx, trial_name, etc.)
        extra_params = {}
        raw_ep = queue_item.get("extra_params_json")
        if raw_ep:
            try:
                extra_params = json.loads(raw_ep)
            except (ValueError, TypeError):
                pass

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

        # Propagate display params (trial_name etc.) to jobs.params_json
        params_json_val = json.dumps(extra_params) if extra_params else None

        with get_db_ctx() as db:
            db.execute(
                """INSERT INTO jobs (subject_id, job_type, status, remote_host, log_path, params_json)
                   VALUES (?, ?, 'pending', ?, ?, ?)""",
                (first_subject_id, job_type, remote_host, log_path, params_json_val),
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
            elif execution_target == "remote":
                self._running_remote_cpu = queue_id
            else:
                self._running_cpu = queue_id

        # Launch in a thread
        thread = threading.Thread(
            target=self._run_job,
            args=(queue_id, job_id, job_type, subjects, remote_cfg, settings, log_path,
                  execution_target, gpu_index, extra_params),
            daemon=True,
        )
        thread.start()
        self._job_threads[queue_id] = thread

    def _run_job(self, queue_id, job_id, job_type, subjects, remote_cfg, settings, log_path,
                  execution_target="remote", gpu_index=0, extra_params=None):
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
                if job_type in ("mediapipe", "vision", "blur", "mediapipe+blur", "pose", "hrnet", "deidentify", "preproc"):
                    # CPU lane: preprocessing
                    if job_type == "mediapipe":
                        _sim = bool((extra_params or {}).get("static_image_mode"))
                        _rev = bool((extra_params or {}).get("reverse"))
                        # ``use_bbox`` defaults True for backward compat
                        # with launches that don't include the flag.
                        _ub = bool((extra_params or {}).get("use_bbox", True))
                        _ti = (extra_params or {}).get("trial_idx")
                        try:
                            _ti = int(_ti) if _ti is not None else None
                        except (TypeError, ValueError):
                            _ti = None
                        local_executor.execute_mediapipe(subject_names[0], job_id, log_path,
                                                          static_image_mode=_sim,
                                                          trial_idx=_ti,
                                                          reverse=_rev,
                                                          use_bbox=_ub)
                    elif job_type == "vision":
                        local_executor.execute_vision(subject_names[0], job_id, log_path)
                    elif job_type == "pose":
                        local_executor.execute_pose(subject_names[0], job_id, log_path)
                    elif job_type == "blur":
                        local_executor.execute_blur(subject_names, job_id, log_path)
                    elif job_type == "hrnet":
                        from ..services.hrnet import run_hrnet_trial
                        from ..services.jobs import registry as job_registry
                        ep = extra_params or {}
                        cancel_event = threading.Event()
                        job_registry._cancel_events[job_id] = cancel_event

                        # Batch mode: ``extra_params.trials`` is a list of
                        # ``{subject_name, trial_idx, trial_name, bbox_os,
                        # bbox_od, use_bbox}``.  Otherwise treat the
                        # legacy single-trial fields as a one-element batch.
                        trials_batch = ep.get("trials")
                        if not trials_batch:
                            trials_batch = [{
                                "subject_name": subject_names[0],
                                "trial_idx": int(ep.get("trial_idx", 0)),
                                "trial_name": ep.get("trial_name"),
                                "bbox_os": ep.get("bbox_os"),
                                "bbox_od": ep.get("bbox_od"),
                                "use_bbox": ep.get("use_bbox"),
                            }]
                        n_total = max(1, len(trials_batch))
                        completed_trials = [0]

                        def _make_progress(idx):
                            def _cb(pct):
                                global_pct = 100.0 * (idx + pct / 100.0) / n_total
                                with get_db_ctx() as _db:
                                    _db.execute(
                                        "UPDATE jobs SET progress_pct = ? WHERE id = ?",
                                        (round(global_pct, 1), job_id),
                                    )
                            return _cb

                        for i, t in enumerate(trials_batch):
                            if cancel_event.is_set():
                                break
                            sub = t.get("subject_name") or subject_names[0]
                            tidx = int(t.get("trial_idx", 0))
                            bbox_os = t.get("bbox_os")
                            bbox_od = t.get("bbox_od")
                            use_bbox = t.get("use_bbox")
                            if use_bbox is None:
                                use_bbox = ep.get("use_bbox")
                            # Full-frame bbox when use_bbox is False.
                            if use_bbox is False:
                                try:
                                    from ..services.video import build_trial_map, get_video_info
                                    vtm = build_trial_map(sub)
                                    if 0 <= tidx < len(vtm):
                                        vp = vtm[tidx]["video_path"]
                                        vi = get_video_info(vp)
                                        is_stereo = vi.width > vi.height * 1.7
                                        if is_stereo:
                                            bbox_os = [0, 0, vi.midline, vi.height]
                                            bbox_od = [0, 0, vi.width - vi.midline, vi.height]
                                        else:
                                            bbox_os = [0, 0, vi.width, vi.height]
                                            bbox_od = [0, 0, vi.width, vi.height]
                                except Exception as _e:
                                    logger.warning(f"Full-frame bbox derive failed: {_e}")

                            tname = t.get("trial_name") or f"trial_{tidx}"
                            logger.info(f"HRnet batch [{i+1}/{n_total}]: {sub} {tname}")

                            def _record_device(dev_str: str, _jid=job_id):
                                # Stash actual torch device in jobs.tmux_session so
                                # the Jobs page badge can show "CPU"/"MPS"/"CUDA"
                                # instead of just the lane label.
                                try:
                                    with get_db_ctx() as _db:
                                        _db.execute(
                                            "UPDATE jobs SET tmux_session = ? WHERE id = ?",
                                            (f"device:{dev_str}", _jid),
                                        )
                                except Exception:
                                    pass

                            try:
                                run_hrnet_trial(
                                    sub, trial_idx=tidx,
                                    bbox_os=bbox_os, bbox_od=bbox_od,
                                    cancel_event=cancel_event,
                                    progress_callback=_make_progress(i),
                                    device_callback=_record_device,
                                )
                                completed_trials[0] += 1
                            except Exception as _e:
                                logger.warning(f"HRnet failed on {sub} {tname}: {_e}")
                                # Continue to next trial — partial success
                                # is better than aborting.
                        job_registry._cancel_events.pop(job_id, None)
                        with get_db_ctx() as _db:
                            _db.execute(
                                "UPDATE jobs SET status='completed', progress_pct=100, "
                                "finished_at=CURRENT_TIMESTAMP WHERE id=?",
                                (job_id,),
                            )
                    elif job_type == "deidentify":
                        # Batch mode: ``extra_params.trials`` is a list of
                        # ``{subject_name, trial_idx, trial_name}`` from
                        # the Jobs-page Multi-trial submit.  We iterate
                        # them under one parent job_id with a single log
                        # so the queue + history show one row instead of
                        # N — same pattern as HRnet.  When no list is
                        # present, fall back to the legacy single-trial
                        # form (``trial_idx`` only).
                        ep = extra_params or {}
                        trials_batch = ep.get("trials")
                        if not trials_batch:
                            trials_batch = [{
                                "subject_name": subject_names[0],
                                "trial_idx": ep.get("trial_idx"),
                            }]
                        n_total = max(1, len(trials_batch))
                        try:
                            with open(log_path, "a") as _lf:
                                _lf.write(f"\n=== Deidentify batch: {n_total} trial(s) ===\n")
                        except OSError:
                            pass
                        for i, t in enumerate(trials_batch):
                            sub = t.get("subject_name") or subject_names[0]
                            tname = t.get("trial_name") or f"trial_{t.get('trial_idx', '?')}"
                            try:
                                with open(log_path, "a") as _lf:
                                    _lf.write(f"\n--- [{i+1}/{n_total}] {sub} {tname} ---\n")
                            except OSError:
                                pass
                            local_executor.execute_deidentify(
                                sub, job_id, log_path,
                                trial_idx=t.get("trial_idx"),
                                batch_idx=i, batch_total=n_total,
                            )
                            # Wait for this trial's subprocess to finish
                            # before launching the next one.  Without the
                            # join, multiple worker subprocesses would
                            # contend for the same job_id and clobber
                            # each other's progress writes.
                            mt = registry._threads.get(job_id)
                            if mt:
                                mt.join()
                            # No need for an explicit milestone snap — the
                            # scaled parser already wrote (i+1)/N×100 as the
                            # last value of this trial.
                        # Mark final completion (per-trial subprocess
                        # writes are sub-100 between trials).
                        with get_db_ctx() as _db:
                            _db.execute(
                                "UPDATE jobs SET status='completed', progress_pct=100, "
                                "finished_at=CURRENT_TIMESTAMP WHERE id=?",
                                (job_id,),
                            )
                    elif job_type == "mediapipe+blur":
                        # Run both sequentially
                        local_executor.execute_mediapipe(subject_names[0], job_id, log_path)
                        # Wait a bit for first to complete
                        import time
                        time.sleep(1)
                        local_executor.execute_blur(subject_names, job_id, log_path)

                    elif job_type == "preproc":
                        # Per-trial preproc: trajectory -> Stabilize
                        # (stable.mp4) -> Background (background.npz)
                        # -> Refine bg (Remove stump) -> Outlines for
                        # every frame (outlines.json).  Matches the
                        # five Compute buttons on the Pre-proc page.
                        from ..services.camera_motion import compute_camera_trajectory
                        from ..services.background import (
                            compute_stable, compute_background, refine_background,
                            compute_outlines_all,
                        )
                        from ..services.jobs import registry as job_registry
                        from ..services.video import build_trial_map
                        from ..services.job_history import stage_timer, finalize_job_record
                        import traceback as _tb
                        import datetime as _dt

                        ep = extra_params or {}
                        cancel_event = threading.Event()
                        job_registry._cancel_events[job_id] = cancel_event

                        trials_batch = ep.get("trials")
                        if not trials_batch:
                            trials_batch = [{
                                "subject_name": subject_names[0],
                                "trial_idx": int(ep.get("trial_idx", 0)),
                                "trial_name": ep.get("trial_name"),
                            }]
                        n_total = max(1, len(trials_batch))

                        # ── Log file: open it up-front and write progress
                        # / errors so the Jobs-page log viewer has
                        # something to show.  Without this every local
                        # preproc job looked silent in the UI even when
                        # backend exceptions were being thrown.
                        os.makedirs(os.path.dirname(log_path), exist_ok=True)
                        log_fh = open(log_path, "w", buffering=1)   # line-buffered

                        def _log(msg: str) -> None:
                            stamp = _dt.datetime.now().strftime("%H:%M:%S")
                            try:
                                log_fh.write(f"[{stamp}] {msg}\n")
                                log_fh.flush()
                            except (OSError, ValueError):
                                pass

                        _log(f"=== Preproc job {job_id} ({n_total} trial(s)) ===")
                        for t in trials_batch:
                            _log(f"  - {t.get('subject_name', subject_names[0])} "
                                 f"trial_idx={t.get('trial_idx')} "
                                 f"trial_name={t.get('trial_name')}")

                        def _preproc_progress(trial_i, sub_pct, phase_weight):
                            # Map (trial_i + phase_local_pct) into the 0–100
                            # range scaled by the batch size.  phase_weight is
                            # the fraction of one trial's work the current
                            # sub-step (trajectory vs background) accounts for.
                            global_pct = 100.0 * (trial_i + sub_pct / 100.0 * 1.0) / n_total
                            with get_db_ctx() as _db:
                                _db.execute(
                                    "UPDATE jobs SET progress_pct = ? WHERE id = ?",
                                    (round(global_pct, 1), job_id),
                                )

                        had_failure = False
                        was_cancelled = False
                        try:
                            for i, t in enumerate(trials_batch):
                                if cancel_event.is_set():
                                    was_cancelled = True
                                    _log("  cancelled by user")
                                    break
                                sub = t.get("subject_name") or subject_names[0]
                                tidx = int(t.get("trial_idx", 0))
                                tname = t.get("trial_name") or f"trial_{tidx}"
                                _log(f"[{i+1}/{n_total}] {sub} {tname} (idx {tidx})")
                                try:
                                    # Phase weights — reapportioned to make room
                                    # for the new outlines bake.  Roughly:
                                    #   trajectory  15%   (camera motion estimate)
                                    #   stable      45%   (full-res warp + encode)
                                    #   background  10%   (sample + temporal median)
                                    #   refine       5%   (skin colour pass on the bg)
                                    #   outlines    25%   (per-frame Otsu+contour)
                                    # Cumulative anchors: 0, 15, 60, 70, 75, 100.
                                    def _on_traj(pct, _i=i):
                                        _preproc_progress(_i, pct * 0.15, 0.15)
                                    _log("  compute_camera_trajectory...")
                                    with stage_timer(job_id, "compute_trajectory",
                                                      subject=sub, trial=tname,
                                                      target="local"):
                                        compute_camera_trajectory(
                                            sub, tidx,
                                            progress_callback=_on_traj,
                                            cancel_event=cancel_event,
                                        )
                                    _log("  compute_camera_trajectory done")
                                    def _on_stable(pct, _i=i):
                                        _preproc_progress(_i, 15 + pct * 0.45, 0.45)
                                    _log("  compute_stable...")
                                    with stage_timer(job_id, "compute_stable",
                                                      subject=sub, trial=tname,
                                                      target="local"):
                                        compute_stable(
                                            sub, tidx,
                                            progress_callback=_on_stable,
                                            cancel_event=cancel_event,
                                        )
                                    _log("  compute_stable done")
                                    def _on_bg(pct, _i=i):
                                        _preproc_progress(_i, 60 + pct * 0.10, 0.10)
                                    _log("  compute_background...")
                                    with stage_timer(job_id, "compute_background",
                                                      subject=sub, trial=tname,
                                                      target="local"):
                                        compute_background(
                                            sub, tidx,
                                            progress_callback=_on_bg,
                                            cancel_event=cancel_event,
                                        )
                                    _log("  compute_background done")
                                    def _on_refine(pct, _i=i):
                                        _preproc_progress(_i, 70 + pct * 0.05, 0.05)
                                    _log("  refine_background...")
                                    with stage_timer(job_id, "refine_background",
                                                      subject=sub, trial=tname,
                                                      target="local"):
                                        refine_background(
                                            sub, tidx,
                                            progress_callback=_on_refine,
                                            cancel_event=cancel_event,
                                        )
                                    _log("  refine_background done")
                                    def _on_outlines(pct, _i=i):
                                        _preproc_progress(_i, 75 + pct * 0.25, 0.25)
                                    _log("  compute_outlines_all...")
                                    with stage_timer(job_id, "compute_outlines",
                                                      subject=sub, trial=tname,
                                                      target="local"):
                                        compute_outlines_all(
                                            sub, tidx,
                                            progress_callback=_on_outlines,
                                            cancel_event=cancel_event,
                                        )
                                    _log("  compute_outlines_all done")
                                except InterruptedError:
                                    was_cancelled = True
                                    _log(f"  CANCELLED at {sub}/{tname}")
                                    logger.info(f"preproc cancelled at trial {sub}/{tname}")
                                    break
                                except Exception as e:
                                    had_failure = True
                                    _log(f"  ERROR at {sub}/{tname}: {type(e).__name__}: {e}")
                                    _log(_tb.format_exc())
                                    logger.exception(f"preproc trial {sub}/{tname} failed: {e}")
                                    with get_db_ctx() as _db:
                                        _db.execute(
                                            "UPDATE jobs SET error_msg = ? WHERE id = ?",
                                            (f"{type(e).__name__}: {e}"[:500], job_id),
                                        )
                        finally:
                            job_registry._cancel_events.pop(job_id, None)
                            final_status = ("cancelled" if was_cancelled else
                                             "failed"    if had_failure else
                                             "completed")
                            _log(f"=== finished: {final_status} ===")
                            try:
                                log_fh.close()
                            except (OSError, ValueError):
                                pass
                            with get_db_ctx() as _db:
                                _db.execute(
                                    f"UPDATE jobs SET status='{final_status}', progress_pct=100, "
                                    "finished_at=CURRENT_TIMESTAMP WHERE id=?",
                                    (job_id,),
                                )
                            finalize_job_record(job_id)

                elif job_type == "train":
                    # GPU lane: training on local GPU
                    local_executor.execute_train(subject_names[0], gpu_index, job_id, log_path)

                elif job_type == "refine":
                    # GPU lane: recreate dataset + resume training on local GPU
                    local_executor.execute_refine(subject_names[0], gpu_index, job_id, log_path)

                elif job_type in ("analyze_v1", "analyze_v2"):
                    # GPU lane: analyze on local GPU
                    local_executor.execute_analyze(subject_names[0], gpu_index, job_id, log_path)

                elif job_type in ("skeleton_v1", "skeleton_v2", "skeleton_v3"):
                    # Skeleton fitting — runs in-process (PyTorch CPU)
                    from ..services.jobs import registry as job_registry
                    from ..services.video import build_trial_map
                    cancel_event = threading.Event()
                    job_registry._cancel_events[job_id] = cancel_event
                    def on_progress(pct):
                        with get_db_ctx() as _db:
                            _db.execute("UPDATE jobs SET progress_pct = ? WHERE id = ?", (pct, job_id))
                    ep = extra_params or {}

                    # Batch mode: when the frontend submits
                    # ``extra_params.trials = [{subject_name, trial_idx,
                    # trial_name}, ...]`` we iterate the list under one
                    # parent job_id so the user sees a single row in the
                    # queue with per-trial outcome chips (matches the
                    # hrnet / deidentify batch pattern).  Falls back to
                    # the legacy single-trial path otherwise.
                    trials_batch = ep.get("trials") if job_type == "skeleton_v1" else None
                    if trials_batch:
                        from ..services.skeleton_v1 import run_skeleton_v1_fit
                        from ..services.skeleton_data import _skeleton_dir
                        n_total = max(1, len(trials_batch))
                        # Per-trial outcome map lives inside extra_params
                        # so the JS queue renderer can color the chips.
                        # Initialize and persist before starting.
                        outcomes = [None] * n_total
                        ep_state = dict(ep)
                        ep_state["trials"] = [dict(t) for t in trials_batch]
                        for i, t in enumerate(ep_state["trials"]):
                            t.setdefault("subject_name", subject_names[0])
                            t["outcome"] = None
                            outcomes[i] = t
                        with get_db_ctx() as _db:
                            _db.execute("UPDATE jobs SET params_json = ? WHERE id = ?",
                                        (json.dumps(ep_state), job_id))

                        # Open the per-job log file and write a settings
                        # header so the Jobs-page History panel actually
                        # has something to show.  Append mode so a
                        # restarted worker doesn't clobber prior runs
                        # at the same log_path.
                        def _log_line(msg: str) -> None:
                            try:
                                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                                with open(log_path, "a") as _lf:
                                    _lf.write(msg.rstrip() + "\n")
                            except Exception:
                                pass
                        _log_line(f"\n=== Skeleton fit v1 batch: {n_total} trial(s) ===")
                        _log_line(
                            f"  Settings: w_reproj={ep.get('w_reproj', 1.0)}"
                            f" w_bone={ep.get('w_bone', 5.0)}"
                            f" w_smooth={ep.get('w_smooth', 1.0)}"
                            f" snap_bones={bool(ep.get('snap_bones', False))}"
                            f" w_angle={ep.get('w_angle', 0.0)}")
                        _log_line(
                            f"  Outlier pre-filter:"
                            f" accel_k={ep.get('accel_k', 6.0)}"
                            f" bone_k={ep.get('bone_k', 6.0)}"
                            f" k_max={ep.get('k_max', 30)}")
                        for i, t in enumerate(ep_state["trials"]):
                            if cancel_event.is_set():
                                t["outcome"] = "cancelled"
                                with get_db_ctx() as _db:
                                    _db.execute("UPDATE jobs SET params_json = ? WHERE id = ?",
                                                (json.dumps(ep_state), job_id))
                                break
                            sub = t.get("subject_name") or subject_names[0]
                            try:
                                vtm = build_trial_map(sub)
                            except Exception as exc:
                                t["outcome"] = "failed"
                                t["outcome_error"] = str(exc)
                                with get_db_ctx() as _db:
                                    _db.execute("UPDATE jobs SET params_json = ? WHERE id = ?",
                                                (json.dumps(ep_state), job_id))
                                continue
                            ti = int(t.get("trial_idx", -1))
                            if ti < 0 or ti >= len(vtm):
                                t["outcome"] = "failed"
                                t["outcome_error"] = f"trial_idx {ti} out of range"
                                with get_db_ctx() as _db:
                                    _db.execute("UPDATE jobs SET params_json = ? WHERE id = ?",
                                                (json.dumps(ep_state), job_id))
                                continue
                            trial_stem = vtm[ti]["trial_name"]
                            def _per_trial_progress(pct, _i=i, _n=n_total):
                                # Map per-trial 0..100 → batch 0..100.
                                batch_pct = ((_i + pct / 100.0) / _n) * 100.0
                                on_progress(batch_pct)
                            try:
                                run_skeleton_v1_fit(
                                    sub, trial_stem,
                                    cancel_event=cancel_event,
                                    progress_callback=_per_trial_progress,
                                    w_reproj=ep.get("w_reproj", 1.0),
                                    w_bone=ep.get("w_bone", 5.0),
                                    w_smooth=ep.get("w_smooth", 1.0),
                                    snap_bones=ep.get("snap_bones", False),
                                    w_angle=ep.get("w_angle", 0.0),
                                    accel_k=ep.get("accel_k", 6.0),
                                    bone_k=ep.get("bone_k", 6.0),
                                    k_max=ep.get("k_max", 30),
                                )
                                # Detect outcome by file presence: if
                                # skeleton_v1.npz now exists, mark ok.
                                root = _skeleton_dir(sub)
                                npz = root / trial_stem / "skeleton_v1.npz" if root else None
                                t["outcome"] = "ok" if (npz and npz.exists()) else "failed"
                            except Exception as exc:
                                t["outcome"] = "failed"
                                t["outcome_error"] = str(exc)
                            with get_db_ctx() as _db:
                                _db.execute("UPDATE jobs SET params_json = ? WHERE id = ?",
                                            (json.dumps(ep_state), job_id))
                            # Per-trial log line.  Reads skeleton_v1_params.json
                            # if the fit produced one — that's where the
                            # outlier-detector counts live (n_outliers_masked,
                            # n_frames_with_any_mask, n_cam_L_masked,
                            # n_cam_R_masked) and the final mean reproj errors.
                            _sidecar = None
                            try:
                                _root = _skeleton_dir(sub)
                                _sidecar_path = _root / trial_stem / "skeleton_v1_params.json"
                                if _sidecar_path.exists():
                                    _sidecar = json.loads(_sidecar_path.read_text())
                            except Exception:
                                _sidecar = None
                            _res = (_sidecar or {}).get("results", {}) if _sidecar else {}
                            _outline = (
                                f"  [{i + 1}/{n_total}] {sub} {trial_stem}: "
                                f"outcome={t.get('outcome')}"
                            )
                            if _res:
                                _outline += (
                                    f"  reproj_L={_res.get('mean_error_L', float('nan')):.2f}px"
                                    f" reproj_R={_res.get('mean_error_R', float('nan')):.2f}px"
                                    f"  masked cells={_res.get('n_outliers_masked', 0)}"
                                    f" frames={_res.get('n_frames_with_any_mask', 0)}"
                                    f" L={_res.get('n_cam_L_masked', 0)}"
                                    f" R={_res.get('n_cam_R_masked', 0)}"
                                )
                            if t.get("outcome_error"):
                                _outline += f"  ERROR: {t['outcome_error']}"
                            _log_line(_outline)
                        _log_line(f"=== batch finished ===\n")
                        # End of batch — leave the loop; outer code will
                        # mark the parent job completed/failed based on
                        # whether any exception escaped.
                        continue_after_batch = True
                    else:
                        continue_after_batch = False
                    if not continue_after_batch:
                        # Legacy single-trial path — works for v1/v2/v3.
                        trial_idx = ep.get("trial_idx", 0)
                        vtm = build_trial_map(subject_names[0])
                        if trial_idx >= len(vtm):
                            raise ValueError(f"Trial index {trial_idx} out of range")
                        trial_stem = vtm[trial_idx]["trial_name"]
                        if job_type == "skeleton_v1":
                            # v1 = original FK skeleton fit (writes skeleton_v1.npz)
                            from ..services.skeleton_v1 import run_skeleton_v1_fit
                            from ..services.skeleton_data import _skeleton_dir as _sk_dir
                            # Settings header to the per-job log.
                            try:
                                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                                with open(log_path, "a") as _lf:
                                    _lf.write(
                                        f"\n=== Skeleton fit v1 (single trial) ===\n"
                                        f"  {subject_names[0]} {trial_stem}\n"
                                        f"  Settings: w_reproj={ep.get('w_reproj', 1.0)}"
                                        f" w_bone={ep.get('w_bone', 5.0)}"
                                        f" w_smooth={ep.get('w_smooth', 1.0)}"
                                        f" snap_bones={bool(ep.get('snap_bones', False))}"
                                        f" w_angle={ep.get('w_angle', 0.0)}\n"
                                        f"  Outlier pre-filter: accel_k={ep.get('accel_k', 6.0)}"
                                        f" bone_k={ep.get('bone_k', 6.0)}"
                                        f" k_max={ep.get('k_max', 30)}\n"
                                    )
                            except Exception:
                                pass
                            run_skeleton_v1_fit(
                                subject_names[0], trial_stem,
                                cancel_event=cancel_event, progress_callback=on_progress,
                                w_reproj=ep.get("w_reproj", 1.0), w_bone=ep.get("w_bone", 5.0),
                                w_smooth=ep.get("w_smooth", 1.0), snap_bones=ep.get("snap_bones", False),
                                # Joint-angle regularization is no longer
                                # exposed in the UI; default to 0.
                                w_angle=ep.get("w_angle", 0.0),
                                accel_k=ep.get("accel_k", 6.0),
                                bone_k=ep.get("bone_k", 6.0),
                                k_max=ep.get("k_max", 30),
                            )
                            # Outcome line from the JSON sidecar.
                            try:
                                _sp = _sk_dir(subject_names[0]) / trial_stem / "skeleton_v1_params.json"
                                if _sp.exists():
                                    _res = (json.loads(_sp.read_text()) or {}).get("results", {})
                                    with open(log_path, "a") as _lf:
                                        _lf.write(
                                            f"  Result: reproj_L={_res.get('mean_error_L', float('nan')):.2f}px"
                                            f" reproj_R={_res.get('mean_error_R', float('nan')):.2f}px"
                                            f"  masked cells={_res.get('n_outliers_masked', 0)}"
                                            f" frames={_res.get('n_frames_with_any_mask', 0)}"
                                            f" L={_res.get('n_cam_L_masked', 0)}"
                                            f" R={_res.get('n_cam_R_masked', 0)}\n"
                                            f"=== finished ===\n\n"
                                        )
                            except Exception:
                                pass
                        elif job_type == "skeleton_v2":
                            # v2 = legacy smoothing fit (writes skeleton_v2.npz)
                            from ..services.skeleton_v2 import run_skeleton_v2_fit
                            run_skeleton_v2_fit(
                                subject_names[0], trial_stem,
                                cancel_event=cancel_event, progress_callback=on_progress,
                            )
                        else:
                            # v3 = corrections pipeline (writes skeleton_v3.npz).
                            # skeleton_v3.py's public entry point now wraps
                            # mp_error_detection.run_correction_pipeline + save_errors
                            # so the queue manager only knows one function per version.
                            from ..services.skeleton_v3 import run_skeleton_v3_fit
                            det = dict(ep.get("detection") or {})
                            attr = dict(ep.get("attribution") or {})
                            run_skeleton_v3_fit(
                                subject_names[0], trial_stem,
                                detection=det, attribution=attr,
                                progress_callback=on_progress,
                                cancel_event=cancel_event,
                                hrnet_source=ep.get("hrnet_source", "auto"),
                                stereo_mode=ep.get("stereo_mode", "image"),
                                stereo_mask_dilate_px=int(ep.get("mask_dilate_px", 10)),
                                stereo_gauss_center_weight=float(ep.get("gauss_center_weight", 0.0)),
                                stereo_conf=float(ep.get("stereo_conf", 0.0)),
                                stereo_dist_px=float(ep.get("stereo_dist_px", 0.0)),
                                stereo_occlusion_px=float(ep.get("stereo_occlusion_px", 0.0)),
                            )
                    job_registry._cancel_events.pop(job_id, None)
                    with get_db_ctx() as _db:
                        _db.execute("UPDATE jobs SET status='completed', progress_pct=100, finished_at=CURRENT_TIMESTAMP WHERE id=?", (job_id,))

                elif job_type == "stereo_correct":
                    # Stereo Correct — batched: one parent job row, one
                    # log file, per-trial outcome chips.  Same shape as
                    # skeleton_v1's batched mode but calls run_stereo_align
                    # per trial instead of the FK fit.  Local CPU only;
                    # remote execution isn't wired (the bake is video +
                    # numpy, fits the local pattern).
                    from ..services.jobs import registry as job_registry
                    from ..services.video import build_trial_map
                    from ..services.stereo_align import run_stereo_align
                    from ..services.skeleton_data import _skeleton_dir
                    cancel_event = threading.Event()
                    job_registry._cancel_events[job_id] = cancel_event
                    def on_progress(pct):
                        with get_db_ctx() as _db:
                            _db.execute(
                                "UPDATE jobs SET progress_pct = ? WHERE id = ?",
                                (pct, job_id),
                            )
                    ep = extra_params or {}
                    trials_batch = ep.get("trials") or [{
                        "subject_name": subject_names[0],
                        "trial_idx": ep.get("trial_idx"),
                        "trial_name": ep.get("trial_name"),
                    }]
                    _mode = (ep.get("mode") or "image").lower()
                    if _mode not in ("image", "outline"):
                        _mode = "image"
                    _dilate = int(ep.get("mask_dilate_px", 10))
                    _gauss  = float(ep.get("gauss_center_weight", 0.0))
                    _expected_fname = (
                        "stereo_align.npz" if _mode == "image"
                        else "stereo_align_outline.npz"
                    )

                    n_total = max(1, len(trials_batch))
                    ep_state = dict(ep)
                    ep_state["trials"] = [dict(t) for t in trials_batch]
                    for t in ep_state["trials"]:
                        t.setdefault("subject_name", subject_names[0])
                        t["outcome"] = None
                    with get_db_ctx() as _db:
                        _db.execute("UPDATE jobs SET params_json = ? WHERE id = ?",
                                    (json.dumps(ep_state), job_id))

                    def _log_line(msg: str) -> None:
                        try:
                            os.makedirs(os.path.dirname(log_path), exist_ok=True)
                            with open(log_path, "a") as _lf:
                                _lf.write(msg.rstrip() + "\n")
                        except Exception:
                            pass
                    _log_line(f"\n=== Stereo Correct batch: {n_total} trial(s) ===")
                    _log_line(
                        f"  Settings: mode={_mode}"
                        f" mask_dilate_px={_dilate}"
                        f" gauss_center_weight={_gauss:.2f}"
                    )
                    for i, t in enumerate(ep_state["trials"]):
                        if cancel_event.is_set():
                            t["outcome"] = "cancelled"
                            with get_db_ctx() as _db:
                                _db.execute("UPDATE jobs SET params_json = ? WHERE id = ?",
                                            (json.dumps(ep_state), job_id))
                            break
                        sub = t.get("subject_name") or subject_names[0]
                        try:
                            vtm = build_trial_map(sub)
                        except Exception as exc:
                            t["outcome"] = "failed"
                            t["outcome_error"] = str(exc)
                            with get_db_ctx() as _db:
                                _db.execute("UPDATE jobs SET params_json = ? WHERE id = ?",
                                            (json.dumps(ep_state), job_id))
                            continue
                        ti = int(t.get("trial_idx", -1))
                        if ti < 0 or ti >= len(vtm):
                            t["outcome"] = "failed"
                            t["outcome_error"] = f"trial_idx {ti} out of range"
                            with get_db_ctx() as _db:
                                _db.execute("UPDATE jobs SET params_json = ? WHERE id = ?",
                                            (json.dumps(ep_state), job_id))
                            continue
                        trial_stem = vtm[ti]["trial_name"]
                        def _per_trial_progress(pct, _i=i, _n=n_total):
                            batch_pct = ((_i + pct / 100.0) / _n) * 100.0
                            on_progress(batch_pct)
                        try:
                            run_stereo_align(
                                sub, ti,
                                progress_callback=_per_trial_progress,
                                cancel_event=cancel_event,
                                mode=_mode,
                                mask_dilate_px=_dilate,
                                gauss_center_weight=_gauss,
                            )
                            root = _skeleton_dir(sub)
                            npz = (root / trial_stem / _expected_fname) if root else None
                            t["outcome"] = "ok" if (npz and npz.exists()) else "failed"
                        except Exception as exc:
                            t["outcome"] = "failed"
                            t["outcome_error"] = str(exc)
                        with get_db_ctx() as _db:
                            _db.execute("UPDATE jobs SET params_json = ? WHERE id = ?",
                                        (json.dumps(ep_state), job_id))
                        _outline = (
                            f"  [{i + 1}/{n_total}] {sub} {trial_stem}: "
                            f"outcome={t.get('outcome')}"
                        )
                        if t.get("outcome_error"):
                            _outline += f"  ERROR: {t['outcome_error']}"
                        _log_line(_outline)
                    _log_line(f"=== batch finished ===\n")
                    job_registry._cancel_events.pop(job_id, None)
                    # Mark job done.  If any trial failed, set status=failed
                    # but keep all the per-trial outcome chips intact.
                    _any_ok = any(t.get("outcome") == "ok"
                                  for t in ep_state["trials"])
                    _all_ok = all(t.get("outcome") == "ok"
                                  for t in ep_state["trials"])
                    if _all_ok:
                        _final_status, _err = "completed", None
                    elif _any_ok:
                        _n_ok = sum(1 for t in ep_state["trials"]
                                    if t.get("outcome") == "ok")
                        _final_status = "completed"
                        _err = f"{_n_ok}/{n_total} succeeded"
                    else:
                        _final_status = "failed"
                        _err = f"0/{n_total} succeeded"
                    with get_db_ctx() as _db:
                        _db.execute(
                            "UPDATE jobs SET status=?, error_msg=?, "
                            "progress_pct=100, finished_at=CURRENT_TIMESTAMP "
                            "WHERE id=?",
                            (_final_status, _err, job_id),
                        )

                # Wait for the subprocess to finish (registry._monitor thread)
                # All local jobs now run as subprocesses tracked by the registry
                monitor_thread = registry._threads.get(job_id)
                if monitor_thread:
                    monitor_thread.join()  # blocks until subprocess completes

            else:
                # Remote execution
                if job_type == "deidentify":
                    # Batch-aware remote deidentify.  When the frontend
                    # submits ``extra_params.trials = [...]`` we iterate
                    # them under one parent job_id so the user sees a
                    # single row in the queue/history — same pattern as
                    # HRnet.  Falls back to the legacy single-trial form
                    # for older callers.
                    from .remote import remote_deidentify
                    ep = extra_params or {}
                    trials_batch = ep.get("trials")
                    if not trials_batch:
                        trials_batch = [{
                            "subject_name": subject_names[0],
                            "trial_idx": ep.get("trial_idx"),
                        }]
                    n_total = max(1, len(trials_batch))
                    try:
                        with open(log_path, "a") as _lf:
                            _lf.write(f"\n=== Deidentify batch: {n_total} trial(s) ===\n")
                    except OSError:
                        pass
                    from ..config import get_settings as _get_settings
                    _deident_dir = _get_settings().video_path / "deidentified"
                    for i, t in enumerate(trials_batch):
                        sub = t.get("subject_name") or subject_names[0]
                        tname = t.get("trial_name") or f"trial_{t.get('trial_idx', '?')}"
                        try:
                            with open(log_path, "a") as _lf:
                                _lf.write(f"\n--- [{i+1}/{n_total}] {sub} {tname} ---\n")
                        except OSError:
                            pass
                        _exc = None
                        try:
                            remote_deidentify(
                                job_id=job_id,
                                cfg=remote_cfg,
                                subject_name=sub,
                                log_path=log_path,
                                registry=registry,
                                trial_idx=t.get("trial_idx"),
                                batch_idx=i, batch_total=n_total,
                            )
                        except Exception as _e:
                            _exc = _e
                            try:
                                with open(log_path, "a") as _lf:
                                    _lf.write(f"  ERROR: {_e}\n")
                            except OSError:
                                pass
                        # Outcome from local file presence: deidentify's
                        # only artifact is the rendered .mp4 in
                        # videos/deidentified/.  Present + sized → ok.
                        # Absent and we caught an exception → failed.
                        # Absent without exception → remote_only (rare for
                        # deidentify; would mean the remote wrote it but
                        # SCP-back failed silently).
                        #
                        # The remote worker saves files as
                        # ``f"{trial_name}.mp4"`` where ``trial_name`` comes
                        # from ``build_trial_map`` (i.e. ``Path(video).stem``,
                        # typically the full subject-prefixed stem like
                        # ``MSA07_L1``).  The frontend passes a *short*
                        # trial_name (``L1``), so naively checking
                        # ``f"{sub}_{tname}.mp4"`` only works when the
                        # subject prefix happens to round-trip cleanly.
                        # Probe several plausible names to be robust against
                        # callers that send the full stem in trial_name, or
                        # videos whose stem doesn't include the subject
                        # prefix.
                        _candidates = []
                        # 1. Source-of-truth: look up the actual stem the
                        #    worker would have used.
                        try:
                            from .video import build_trial_map as _btm
                            _tmap = _btm(sub)
                            _ti = t.get("trial_idx")
                            if _ti is not None and 0 <= int(_ti) < len(_tmap):
                                _stem = _tmap[int(_ti)].get("trial_name") or ""
                                if _stem:
                                    _candidates.append(f"{_stem}.mp4")
                        except Exception:
                            pass
                        # 2. Frontend short-stem convention.
                        _candidates.append(f"{sub}_{tname}.mp4")
                        # 3. Caller may have already passed a full stem.
                        _candidates.append(f"{tname}.mp4")
                        _present = False
                        _out_path = None
                        for _cand in _candidates:
                            _p = _deident_dir / _cand
                            if _p.exists() and _p.stat().st_size > 4096:
                                _out_path = _p
                                _present = True
                                break
                        if _out_path is None:
                            _out_path = _deident_dir / _candidates[0]
                        if _present:
                            t["outcome"] = "ok"
                        elif _exc:
                            t["outcome"] = "failed"
                            t["outcome_error"] = str(_exc)[:300]
                        else:
                            t["outcome"] = "remote_only"
                        # Live mid-batch persist for the trial-detail modal.
                        try:
                            import json as _json2
                            _live_params = dict(ep); _live_params["trials"] = trials_batch
                            with get_db_ctx() as _db_live:
                                _db_live.execute(
                                    "UPDATE jobs SET params_json=? WHERE id=?",
                                    (_json2.dumps(_live_params), job_id),
                                )
                        except Exception:
                            pass
                    # Persist + final status decision (same logic as HRnet).
                    import json as _json
                    _new_params = dict(ep); _new_params["trials"] = trials_batch
                    n_ok = sum(1 for _t in trials_batch if _t.get("outcome") == "ok")
                    n_rem = sum(1 for _t in trials_batch if _t.get("outcome") == "remote_only")
                    n_fail = sum(1 for _t in trials_batch if _t.get("outcome") == "failed")
                    if n_fail > 0:
                        _final_status = "failed"
                        # Surface the first per-trial error so the UI
                        # shows the actual cause (e.g. "ffmpeg not
                        # found on remote") instead of just a count.
                        _first_err = next(
                            (str(_t.get("outcome_error") or "").strip()
                             for _t in trials_batch
                             if _t.get("outcome") == "failed"
                                and _t.get("outcome_error")),
                            "")
                        _msg = f"{n_ok + n_rem}/{n_total} trial(s) succeeded"
                        _final_err = f"{_msg}: {_first_err}" if _first_err else _msg
                    elif n_rem > 0:
                        _final_status = "completed"
                        _final_err = f"Download incomplete: {n_ok}/{n_total} downloaded"
                    else:
                        _final_status = "completed"
                        _final_err = None
                    with get_db_ctx() as _db:
                        _db.execute(
                            "UPDATE jobs SET params_json=?, error_msg=? WHERE id=?",
                            (_json.dumps(_new_params), _final_err, job_id),
                        )
                        _db.execute(
                            "UPDATE jobs SET status=?, progress_pct=100, "
                            "finished_at=CURRENT_TIMESTAMP WHERE id=?",
                            (_final_status, job_id),
                        )

                elif job_type == "hrnet":
                    # HRNet remote dispatch.  Two paths:
                    #   * extra_params._use_batch_runner=True  → upload all
                    #     videos+MP, fire a single long-lived runner on the
                    #     remote, return immediately; a poller thread tracks
                    #     completion + downloads.  Survives local app
                    #     restarts (the new "Submit Batch" button uses this).
                    #   * default → legacy per-trial dispatch (current
                    #     "Submit Processing Job" behavior).
                    from .remote import remote_hrnet_job
                    ep = extra_params or {}
                    if ep.get("_use_batch_runner"):
                        from .remote import dispatch_remote_batch, poll_remote_batch
                        # Normalize the trials list (the dispatcher needs
                        # subject_name + trial_idx for each).
                        _tb = ep.get("trials") or [{
                            "subject_name": subject_names[0],
                            "trial_idx": int(ep.get("trial_idx", 0)),
                            "trial_name": ep.get("trial_name"),
                            "bbox_os": ep.get("bbox_os"),
                            "bbox_od": ep.get("bbox_od"),
                        }]
                        ep_norm = dict(ep); ep_norm["trials"] = _tb
                        result = dispatch_remote_batch(
                            job_id=job_id, cfg=remote_cfg,
                            subject_name=subject_names[0],
                            extra_params=ep_norm,
                            log_path=log_path, registry=registry,
                        )
                        if result.get("error"):
                            # dispatch_remote_batch already wrote the
                            # failed status; just bail.
                            return
                        # Spawn the poller in a daemon thread so this
                        # worker thread can return + free the dispatch slot.
                        import threading as _th
                        poller = _th.Thread(
                            target=poll_remote_batch,
                            kwargs=dict(
                                job_id=job_id, cfg=remote_cfg,
                                batch_id=result["batch_id"],
                                log_path=log_path, registry=registry,
                                parent_extra=ep_norm,
                            ),
                            daemon=True,
                        )
                        poller.start()
                        registry._threads[job_id] = poller
                        # The remote runner now owns the work; this thread
                        # returns and the queue can dispatch the next item.
                        return
                    trials_batch = ep.get("trials")
                    if not trials_batch:
                        trials_batch = [{
                            "subject_name": subject_names[0],
                            "trial_idx": int(ep.get("trial_idx", 0)),
                            "trial_name": ep.get("trial_name"),
                            "bbox_os": ep.get("bbox_os"),
                            "bbox_od": ep.get("bbox_od"),
                            "use_bbox": ep.get("use_bbox"),
                        }]
                    n_total = max(1, len(trials_batch))

                    # Append a header to the log so the user sees the batch.
                    try:
                        with open(log_path, "a") as _lf:
                            _lf.write(f"\n=== HRnet batch: {n_total} trial(s) ===\n")
                    except OSError:
                        pass

                    for i, t in enumerate(trials_batch):
                        sub = t.get("subject_name") or subject_names[0]
                        tname = t.get("trial_name") or f"trial_{t.get('trial_idx', 0)}"
                        try:
                            with open(log_path, "a") as _lf:
                                _lf.write(f"\n--- [{i+1}/{n_total}] {sub} {tname} ---\n")
                        except OSError:
                            pass
                        per_trial_ep = {
                            "trial_idx": int(t.get("trial_idx", 0)),
                            "trial_name": tname,
                            "bbox_os": t.get("bbox_os"),
                            "bbox_od": t.get("bbox_od"),
                            "use_bbox": t.get("use_bbox", ep.get("use_bbox")),
                            # Pass batch position so the remote handler can
                            # scale its 0-100 progress into the global slice.
                            "_batch_index": i,
                            "_batch_total": n_total,
                        }
                        try:
                            _res = remote_hrnet_job(
                                job_id=job_id,
                                cfg=remote_cfg,
                                subject_name=sub,
                                extra_params=per_trial_ep,
                                log_path=log_path,
                                registry=registry,
                            ) or {}
                        except Exception as _e:
                            logger.warning(f"HRnet remote failed on {sub} {tname}: {_e}")
                            try:
                                with open(log_path, "a") as _lf:
                                    _lf.write(f"  ERROR: {_e}\n")
                            except OSError:
                                pass
                            _res = {"remote_done": False, "downloaded": False, "error": str(_e)}
                        # Translate to the on-disk outcome tag used by the
                        # frontend trial-chip color and parent status badge.
                        if _res.get("remote_done") and _res.get("downloaded"):
                            t["outcome"] = "ok"
                        elif _res.get("remote_done"):
                            t["outcome"] = "remote_only"
                        else:
                            t["outcome"] = "failed"
                        if _res.get("error"):
                            t["outcome_error"] = str(_res["error"])[:300]
                        # Persist mid-batch so the active-job modal can show
                        # per-trial colors live while the loop is still
                        # iterating (otherwise outcomes only land at end).
                        try:
                            import json as _json2
                            _live_params = dict(ep); _live_params["trials"] = trials_batch
                            with get_db_ctx() as _db_live:
                                _db_live.execute(
                                    "UPDATE jobs SET params_json=? WHERE id=?",
                                    (_json2.dumps(_live_params), job_id),
                                )
                        except Exception:
                            pass
                    # Persist per-trial outcomes back to params_json so the
                    # Jobs page can color chips and pick a correct status.
                    import json as _json
                    _new_params = dict(ep); _new_params["trials"] = trials_batch
                    n_ok = sum(1 for _t in trials_batch if _t.get("outcome") == "ok")
                    n_rem = sum(1 for _t in trials_batch if _t.get("outcome") == "remote_only")
                    n_fail = sum(1 for _t in trials_batch if _t.get("outcome") == "failed")
                    if n_fail > 0:
                        _final_status = "failed"
                        # Surface the first per-trial error so the UI
                        # shows the actual cause (e.g. "ffmpeg not
                        # found on remote") instead of just a count.
                        _first_err = next(
                            (str(_t.get("outcome_error") or "").strip()
                             for _t in trials_batch
                             if _t.get("outcome") == "failed"
                                and _t.get("outcome_error")),
                            "")
                        _msg = f"{n_ok + n_rem}/{n_total} trial(s) succeeded"
                        _final_err = f"{_msg}: {_first_err}" if _first_err else _msg
                    elif n_rem > 0:
                        _final_status = "completed"
                        _final_err = f"Download incomplete: {n_ok}/{n_total} downloaded"
                    else:
                        _final_status = "completed"
                        _final_err = None
                    with get_db_ctx() as _db:
                        _db.execute(
                            "UPDATE jobs SET params_json=?, error_msg=? WHERE id=?",
                            (_json.dumps(_new_params), _final_err, job_id),
                        )
                        _db.execute(
                            "UPDATE jobs SET status=?, progress_pct=100, "
                            "finished_at=CURRENT_TIMESTAMP WHERE id=?",
                            (_final_status, job_id),
                        )

                elif job_type == "preproc":
                    # Remote preproc batch: dispatcher uploads all
                    # subjects' videos + modules + per-subject bundle
                    # specs, then launches a single long-lived
                    # ``remote_preproc_batch_runner.py`` on the remote
                    # that loops subjects sequentially.  Local side
                    # spawns a poller thread, then frees this worker
                    # slot so the queue can dispatch the next item.
                    # Survives server restart via resume_remote_preproc_batch.
                    from .remote import (
                        dispatch_remote_preproc_batch,
                        poll_remote_preproc_batch,
                    )
                    import json as _json
                    import threading as _th
                    ep = extra_params or {}
                    trials_batch = ep.get("trials") or [{
                        "subject_name": subject_names[0],
                        "trial_idx": ep.get("trial_idx"),
                        "trial_name": ep.get("trial_name"),
                    }]
                    # Per-trial chip state init.  Trials start with no
                    # outcome and no `uploaded` flag → chips render as
                    # dim blue ("uploading").  Once dispatch_remote_
                    # preproc_batch's upload phase finishes, we mark
                    # them uploaded=True so chips flip to accent blue.
                    _job_params = dict(ep); _job_params["trials"] = trials_batch
                    _job_params["phase"] = "uploading"
                    with get_db_ctx() as _db:
                        _db.execute(
                            "UPDATE jobs SET params_json=?, progress_pct=0 WHERE id=?",
                            (_json.dumps(_job_params), job_id),
                        )

                    # Dispatch: uploads (idempotent skip-if-present) +
                    # writes batch.json + launches remote runner as a
                    # detached process.  Returns immediately.
                    try:
                        result = dispatch_remote_preproc_batch(
                            job_id=job_id, cfg=remote_cfg,
                            trials_batch=trials_batch,
                            log_path=log_path, registry=registry,
                        )
                    except Exception as _de:
                        logger.exception(f"preproc dispatch {job_id} failed: {_de}")
                        with get_db_ctx() as _db:
                            _db.execute(
                                "UPDATE jobs SET status='failed', error_msg=?, "
                                "finished_at=CURRENT_TIMESTAMP WHERE id=?",
                                (f"dispatch: {str(_de)[:400]}", job_id),
                            )
                        return
                    if result.get("error"):
                        # dispatch already wrote failed status
                        return
                    # Uploads complete: flip phase to "running" so the
                    # UI swaps "Uploading..." for the % progress bar,
                    # and mark every trial uploaded=True so chips turn
                    # accent blue.  Carry forward _batch_id (written
                    # by the dispatcher into the DB) so the resume
                    # function can find it after a restart.
                    _job_params["phase"] = "running"
                    if result.get("batch_id"):
                        _job_params["_batch_id"] = result["batch_id"]
                    if result.get("remote_pid"):
                        _job_params["_remote_pid"] = result["remote_pid"]
                    for _t in trials_batch:
                        _t["uploaded"] = True
                    with get_db_ctx() as _db:
                        _db.execute(
                            "UPDATE jobs SET params_json=? WHERE id=?",
                            (_json.dumps(_job_params), job_id),
                        )

                    # Spawn the poller in a daemon thread so THIS
                    # worker thread returns and frees the dispatch
                    # slot.  Poller's lifecycle is now decoupled from
                    # the queue manager — it owns the job until the
                    # remote batch runner finishes (or the server
                    # restarts and resume_remote_preproc_batch takes
                    # over).
                    poller = _th.Thread(
                        target=poll_remote_preproc_batch,
                        kwargs=dict(
                            job_id=job_id, cfg=remote_cfg,
                            batch_id=result["batch_id"],
                            log_path=log_path, registry=registry,
                            parent_extra=_job_params,
                        ),
                        daemon=True,
                    )
                    poller.start()
                    registry._threads[job_id] = poller
                    # Hand lifecycle off to the poller — DO NOT fall
                    # through to the queue worker's final-status flush
                    # below, which would otherwise read jobs.status
                    # (still 'running' here) and write that back to
                    # job_queue.status, stranding the row at 'running'
                    # forever. The poller writes BOTH jobs and
                    # job_queue terminal status when it exits.
                    return

                elif job_type in ("vision", "pose", "skeleton_v1", "skeleton_v2"):
                    # These don't have remote handlers yet — run locally
                    logger.warning(f"Job {job_id}: {job_type} not supported remotely, running locally")
                    if job_type == "vision":
                        local_executor.execute_vision(subject_names[0], job_id, log_path)
                        monitor_thread = registry._threads.get(job_id)
                        if monitor_thread: monitor_thread.join()
                    elif job_type == "pose":
                        local_executor.execute_pose(subject_names[0], job_id, log_path)
                        monitor_thread = registry._threads.get(job_id)
                        if monitor_thread: monitor_thread.join()
                    elif job_type in ("skeleton_v1", "skeleton_v2"):
                        # Re-use the local-cpu in-process dispatch
                        from ..services.jobs import registry as job_registry
                        from ..services.video import build_trial_map as _btm
                        cancel_event = threading.Event()
                        job_registry._cancel_events[job_id] = cancel_event
                        ep = extra_params or {}
                        def _fallback_progress(pct):
                            with get_db_ctx() as _db:
                                _db.execute("UPDATE jobs SET progress_pct = ? WHERE id = ?", (pct, job_id))
                        if job_type == "skeleton_v1":
                            from ..services.skeleton_v1 import run_skeleton_v1_fit
                            vtm = _btm(subject_names[0])
                            trial_stem = vtm[ep.get("trial_idx", 0)]["trial_name"]
                            run_skeleton_v1_fit(subject_names[0], trial_stem, cancel_event=cancel_event,
                                               progress_callback=_fallback_progress,
                                               w_reproj=ep.get("w_reproj", 1.0), w_bone=ep.get("w_bone", 5.0),
                                               w_smooth=ep.get("w_smooth", 1.0), snap_bones=ep.get("snap_bones", False),
                                               w_angle=ep.get("w_angle", 2.0))
                        elif job_type == "skeleton_v2":
                            from ..services.skeleton_v2 import run_skeleton_v2_fit
                            vtm = _btm(subject_names[0])
                            trial_stem = vtm[ep.get("trial_idx", 0)]["trial_name"]
                            run_skeleton_v2_fit(subject_names[0], trial_stem,
                                                cancel_event=cancel_event,
                                                progress_callback=_fallback_progress)
                        job_registry._cancel_events.pop(job_id, None)
                        with get_db_ctx() as _db:
                            _db.execute("UPDATE jobs SET status='completed', progress_pct=100, finished_at=CURRENT_TIMESTAMP WHERE id=?", (job_id,))

                elif job_type in ("mediapipe", "blur", "mediapipe+blur"):
                    # CPU lane: preprocessing.
                    #
                    # Single-trial MP requests (i.e. the Auto-page per-
                    # trial Run-MediaPipe button) carry ``trial_idx`` in
                    # extra_params.  ``remote_preprocess_batch`` has no
                    # per-trial mode — it always runs MP on every video
                    # for the subject and overwrites the full npz on
                    # download, silently destroying the other trials'
                    # data.  Fall back to LOCAL CPU for single-trial MP
                    # so the per-trial merge path in ``run_mediapipe``
                    # is preserved.  Multi-trial / "Run on all" jobs
                    # still go to the remote host as before.
                    _ti = (extra_params or {}).get("trial_idx") if job_type == "mediapipe" else None
                    if _ti is not None:
                        try:
                            _ti = int(_ti)
                        except (TypeError, ValueError):
                            _ti = None
                    if _ti is not None and job_type == "mediapipe":
                        try:
                            with open(log_path, "a") as _lf:
                                _lf.write(
                                    f"\n[queue] Single-trial MP (trial_idx={_ti}) — "
                                    f"routing to LOCAL CPU to preserve other trials' "
                                    f"npz data.  Remote MP path lacks per-trial mode.\n"
                                )
                        except OSError:
                            pass
                        _sim = bool((extra_params or {}).get("static_image_mode"))
                        _rev = bool((extra_params or {}).get("reverse"))
                        _ub  = bool((extra_params or {}).get("use_bbox", True))
                        local_executor.execute_mediapipe(
                            subject_names[0], job_id, log_path,
                            static_image_mode=_sim, trial_idx=_ti,
                            reverse=_rev, use_bbox=_ub,
                        )
                        # Wait for the subprocess to finish (it writes
                        # progress + final status via the registry monitor).
                        mt = registry._threads.get(job_id)
                        if mt:
                            mt.join()
                    else:
                        steps = []
                        if job_type in ("mediapipe", "mediapipe+blur"):
                            steps.append("mediapipe")
                        if job_type in ("blur", "mediapipe+blur"):
                            steps.append("blur")
                        _rev = bool((extra_params or {}).get("reverse"))
                        # use_bbox defaults to True (matches the local
                        # behaviour); only forward False explicitly.
                        _ub  = bool((extra_params or {}).get("use_bbox", True))
                        remote_preprocess_batch(
                            job_id=job_id,
                            cfg=remote_cfg,
                            steps=steps,
                            subjects=subject_names,
                            log_path=log_path,
                            registry=registry,
                            force=True,
                            reverse=_rev,
                            use_bbox=_ub,
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
                db.execute(
                    "UPDATE jobs SET status = 'failed', error_msg = ?, "
                    "finished_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (str(e)[:500], job_id),
                )
        finally:
            with self._lock:
                if self._running_gpu == queue_id:
                    self._running_gpu = None
                if self._running_cpu == queue_id:
                    self._running_cpu = None
                if self._running_remote_cpu == queue_id:
                    self._running_remote_cpu = None
            self._job_threads.pop(queue_id, None)
            # Lifetime job-history flush.  Runs once per job no matter
            # which branch the dispatcher took; some branches also call
            # finalize_job_record themselves (preproc local + remote),
            # which is fine — the second call here is a no-op because
            # the in-memory stage list was already cleared.  Catches
            # any branch we haven't instrumented yet (deidentify, hrnet,
            # mediapipe, train, etc.) so every job lands in the file
            # with at least its basic metadata + total duration.
            try:
                from .job_history import finalize_job_record
                finalize_job_record(job_id)
            except Exception:
                logger.exception(f"job_history flush failed for job {job_id}")
            # Signal drain thread to check for next item
            self._drain_event.set()


# Global singleton
queue_manager = QueueManager()
