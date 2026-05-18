"""FastAPI application: mount routers, serve static files, startup discovery."""

import logging
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .config import get_settings
from .db import init_db
from .routers import subjects, labeling, pipeline, jobs, results, settings, filebrowser, video_tools, batch, remote_jobs, skeleton, export, camera_setups, updater, deidentify, analyze, preproc

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Movement Tracker", version="0.2.0")


# Disable browser caching for all responses (dev tool — always serve fresh)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


# Suppress noisy access logs for polling endpoints
class _QuietPollFilter(logging.Filter):
    """Filter out high-frequency polling requests from uvicorn access log."""
    _QUIET = (
        "/api/jobs",
        "/api/remote/queue",
        "/api/remote/pending-downloads",
        "/api/remote/download-progress",
        "/api/remote/stream",
    )
    def filter(self, record):
        msg = record.getMessage()
        return not any(p in msg for p in self._QUIET)

logging.getLogger("uvicorn.access").addFilter(_QuietPollFilter())


class NoCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static") or request.url.path in (
            "/", "/labeling", "/labeling-select", "/mediapipe", "/deidentify", "/labels", "/preproc", "/oscillations", "/results", "/settings", "/onboarding", "/remote", "/videos", "/calibration", "/tutorials", "/tutorial", "/events"
        ):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return response


app.add_middleware(NoCacheMiddleware)

# Mount routers
app.include_router(subjects.router)
app.include_router(labeling.router)
app.include_router(pipeline.router)
app.include_router(jobs.router)
app.include_router(results.router)
app.include_router(settings.router)
app.include_router(filebrowser.router)
app.include_router(video_tools.router)
app.include_router(batch.router)
app.include_router(remote_jobs.router)

app.include_router(skeleton.router)  # API endpoints used by videos.js (page removed)
app.include_router(export.router)
app.include_router(camera_setups.router)
app.include_router(updater.router)
app.include_router(deidentify.router)
app.include_router(analyze.router)
app.include_router(preproc.router)

# Mount static files
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _recover_stale_jobs(s):
    """Handle stale jobs from prior server sessions.

    Key design: when a remote process has already finished, update the DB
    **synchronously** so the status is correct even if a subsequent --reload
    kills any background download thread.  Downloads are best-effort and
    will be re-attempted on the next server restart if interrupted.
    """
    from .db import get_db_ctx
    with get_db_ctx() as db:
        stale_jobs = db.execute(
            "SELECT id, job_type, tmux_session, remote_host, subject_id, log_path FROM jobs "
            "WHERE status IN ('running', 'pending')"
        ).fetchall()

    if not stale_jobs:
        return

    remote_cfg = s.get_remote_config()
    resumed = 0
    failed = 0

    TRAIN_JOB_TYPES = {"train", "analyze_v1", "analyze_v2"}
    PREPROCESS_JOB_TYPES = {"mediapipe", "blur", "mediapipe+blur"}

    for job in stale_jobs:
        session_info = job.get("tmux_session") or ""
        remote_host = job.get("remote_host")
        job_type = job.get("job_type", "")

        # Parse PID from session_info (format: "pid:12345")
        remote_pid = None
        if session_info.startswith("pid:"):
            try:
                remote_pid = int(session_info.split(":")[1])
            except (ValueError, IndexError):
                pass

        LOCAL_JOB_TYPES = {"hrnet", "skeleton_v1", "skeleton_v2", "skeleton_v3", "vision", "pose", "deidentify"}
        if not remote_host and job_type in LOCAL_JOB_TYPES:
            # Local in-process job interrupted by server restart — mark failed
            with get_db_ctx() as db:
                db.execute(
                    "UPDATE jobs SET status = 'failed', error_msg = 'Server restarted (job interrupted)', "
                    "finished_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (job["id"],),
                )
            failed += 1
            continue

        if not (session_info and remote_host and remote_cfg):
            # Remote job with missing config — mark failed
            with get_db_ctx() as db:
                db.execute(
                    "UPDATE jobs SET status = 'failed', error_msg = 'Server restarted (no remote config)', "
                    "finished_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (job["id"],),
                )
            failed += 1
            continue

        from .services.remote import (
            _check_remote_pid_alive, _read_remote_status,
            remote_train_monitor, resume_preprocess_monitor,
        )
        from .services.jobs import registry

        log_path = job["log_path"] or str(s.dlc_path / ".logs" / f"job_{job['id']}.log")
        proc_alive = bool(remote_pid) and _check_remote_pid_alive(remote_cfg, remote_pid)

        # ── Read remote status (works for both alive and dead processes) ──
        if job_type in PREPROCESS_JOB_TYPES:
            status_file = f"{remote_cfg.work_dir}/preprocess_{job['id']}_status.json"
        elif job_type in TRAIN_JOB_TYPES:
            with get_db_ctx() as db:
                subj = db.execute(
                    "SELECT name FROM subjects WHERE id = ?",
                    (job["subject_id"],),
                ).fetchone()
            status_file = f"{remote_cfg.work_dir}/{subj['name']}/status.json" if subj else None
        elif job_type == "deidentify":
            # Single-trial remote deidentify recovery: the worker is
            # detached on the remote, so it survives our restart — just
            # the local poll-and-download loop dies.  Reattach a fresh
            # monitor thread that picks up exactly where the dead one
            # left off (reads remote status.json, downloads outputs when
            # complete, marks job done).
            #
            # Identification: the job has a remote_pid (in tmux_session)
            # and a remote_host other than 'localhost'.  Skip the batch
            # recovery branch below if so.
            if remote_pid and remote_host and remote_host != "localhost":
                from .services.remote import resume_remote_deidentify_monitor
                with get_db_ctx() as db:
                    subj = db.execute(
                        "SELECT name FROM subjects WHERE id = ?",
                        (job["subject_id"],),
                    ).fetchone()
                if subj:
                    import json as _json2
                    _params2 = _json2.loads(job.get("params_json") or "{}") if job.get("params_json") else {}
                    _ti = _params2.get("trial_idx") if isinstance(_params2, dict) else None
                    logger.info(
                        f"Job {job['id']} (deidentify): reattaching monitor for remote PID {remote_pid}"
                    )
                    t = threading.Thread(
                        target=resume_remote_deidentify_monitor,
                        kwargs=dict(
                            job_id=job["id"], cfg=remote_cfg,
                            subject_name=subj["name"], remote_pid=remote_pid,
                            trial_idx=_ti, log_path=log_path, registry=registry,
                        ),
                        daemon=True,
                    )
                    t.start()
                    registry._threads[job["id"]] = t
                    resumed += 1
                    continue
            # (Fallthrough) Batch deidentify recovery: completion signal
            # is the per-trial deidentified video on disk under
            # videos/deidentified/.
            import json as _json
            _params = _json.loads(job.get("params_json") or "{}") if job.get("params_json") else {}
            _trials_batch = _params.get("trials")
            if _trials_batch and isinstance(_trials_batch, list):
                from .services.video import build_trial_map as _btm
                _deident_dir = s.video_path / "deidentified"
                completed_trials = []
                missing_trials = []
                for _t in _trials_batch:
                    _sub = _t.get("subject_name") or ""
                    _tn = _t.get("trial_name") or ""
                    if not (_sub and _tn):
                        continue
                    # Renderer writes <subject>_<trial>.mp4 (full stem).
                    # Check both naming conventions to be robust.
                    _candidates = [
                        _deident_dir / f"{_sub}_{_tn}.mp4",
                        _deident_dir / f"{_tn}.mp4",
                    ]
                    if any(p.exists() and p.stat().st_size > 4096 for p in _candidates):
                        completed_trials.append(f"{_sub} {_tn}")
                        # Stamp outcome so Resume button knows to skip.
                        _t["outcome"] = "ok"
                    else:
                        missing_trials.append(f"{_sub} {_tn}")
                        _t["outcome"] = "failed"
                        _t["outcome_error"] = "interrupted by server restart"
                _n_done = len(completed_trials)
                _n_total = len(_trials_batch)
                logger.info(
                    f"Job {job['id']} (deidentify batch): {_n_done}/{_n_total} trials done before restart"
                )
                # Persist updated outcomes back to params_json so the Jobs
                # page Resume button has a per-trial status to filter on.
                _params["trials"] = _trials_batch
                _new_pjson = _json.dumps(_params)
                if _n_done == _n_total and _n_total > 0:
                    with get_db_ctx() as db:
                        db.execute(
                            "UPDATE jobs SET status='completed', progress_pct=100, "
                            "params_json=?, finished_at=CURRENT_TIMESTAMP WHERE id=?",
                            (_new_pjson, job["id"]),
                        )
                    resumed += 1
                else:
                    _preview = ", ".join(missing_trials[:5])
                    if len(missing_trials) > 5:
                        _preview += f", +{len(missing_trials) - 5} more"
                    _msg = (
                        f"Batch interrupted by restart: {_n_done}/{_n_total} trials "
                        f"completed. Re-submit remaining: {_preview}"
                    )
                    with get_db_ctx() as db:
                        db.execute(
                            "UPDATE jobs SET status='failed', error_msg=?, "
                            "progress_pct=?, params_json=?, "
                            "finished_at=CURRENT_TIMESTAMP WHERE id=?",
                            (_msg[:500], round(100.0 * _n_done / max(1, _n_total), 1),
                             _new_pjson, job["id"]),
                        )
                    failed += 1
                continue
            # Single-trial deidentify falls through to default handling
            # (status_file = None, then proc_alive check or fail).
            status_file = None

        elif job_type == "hrnet":
            # Reconstruct the status file path from job params.  Batch jobs
            # store a ``trials`` list rather than a single trial_name; in
            # that case we handle them below in the batch-aware branch.
            import json as _json
            _params = _json.loads(job.get("params_json") or "{}") if job.get("params_json") else {}
            _trials_batch = _params.get("trials")
            _batch_id = _params.get("_batch_id")

            # ── Long-lived remote batch runner recovery ───────────────
            # Submit Batch jobs store ``_batch_id`` in params_json.  The
            # remote runner is detached and keeps going through local
            # restarts; we just need to reattach a poller thread so the
            # local UI catches up on completed trials.
            if _batch_id:
                from .services.remote import poll_remote_batch
                logger.info(f"Job {job['id']} (batch {_batch_id}): reattaching poller")
                # Marker line in the job's own log file so we can confirm
                # via tail whether this branch fired (vs. recovery
                # silently skipping the job).
                try:
                    with open(log_path, "a") as _diag:
                        _diag.write(
                            f"[recovery] reattaching poller for batch {_batch_id} "
                            f"(job_id={job['id']})\n"
                        )
                except Exception as _e_diag:
                    logger.warning(f"recovery marker write failed: {_e_diag}")
                t = threading.Thread(
                    target=poll_remote_batch,
                    kwargs=dict(
                        job_id=job["id"], cfg=remote_cfg,
                        batch_id=_batch_id, log_path=log_path,
                        registry=registry, parent_extra=_params,
                    ),
                    daemon=True,
                )
                try:
                    t.start()
                except Exception as _e_start:
                    try:
                        with open(log_path, "a") as _diag:
                            _diag.write(f"[recovery] thread start failed: {_e_start!r}\n")
                    except Exception:
                        pass
                    raise
                registry._threads[job["id"]] = t
                resumed += 1
                continue

            if _trials_batch and isinstance(_trials_batch, list):
                # ── Batch HRnet recovery ──────────────────────────────
                # For each trial in the batch, check whether its output
                # files exist locally.  A heatmaps.npz on disk means the
                # full pipeline (inference → SCP-back) finished for that
                # trial.  Tally completed vs missing and set the parent
                # job status accordingly with an accurate error_msg.
                from .services.skeleton_data import _skeleton_dir
                completed_trials = []
                missing_trials = []
                for _t in _trials_batch:
                    _sub = _t.get("subject_name") or ""
                    _tn = _t.get("trial_name") or ""
                    if not (_sub and _tn):
                        continue
                    # Local trial dir is built by remote_hrnet_job from
                    # build_trial_map's full stem (e.g. "Con03_L1") even
                    # though extra_params stores the short name ("L1").
                    # Check both forms to be robust.
                    _candidates = [
                        _skeleton_dir(_sub) / _tn / "hrnet_w18_heatmaps.npz",
                        _skeleton_dir(_sub) / f"{_sub}_{_tn}" / "hrnet_w18_heatmaps.npz",
                    ]
                    if any(p.exists() and p.stat().st_size > 0 for p in _candidates):
                        completed_trials.append(f"{_sub} {_tn}")
                        _t["outcome"] = "ok"
                    else:
                        missing_trials.append(f"{_sub} {_tn}")
                        _t["outcome"] = "failed"
                        _t["outcome_error"] = "interrupted by server restart"
                _n_done = len(completed_trials)
                _n_total = len(_trials_batch)
                logger.info(
                    f"Job {job['id']} (hrnet batch): {_n_done}/{_n_total} trials completed before restart"
                )
                _params["trials"] = _trials_batch
                _new_pjson = _json.dumps(_params)
                if _n_done == _n_total and _n_total > 0:
                    with get_db_ctx() as db:
                        db.execute(
                            "UPDATE jobs SET status='completed', progress_pct=100, "
                            "params_json=?, finished_at=CURRENT_TIMESTAMP WHERE id=?",
                            (_new_pjson, job["id"]),
                        )
                    resumed += 1
                else:
                    # Show first few missing trials for context.
                    _preview = ", ".join(missing_trials[:5])
                    if len(missing_trials) > 5:
                        _preview += f", +{len(missing_trials) - 5} more"
                    _msg = (
                        f"Batch interrupted by restart: {_n_done}/{_n_total} trials "
                        f"completed. Re-submit remaining: {_preview}"
                    )
                    with get_db_ctx() as db:
                        db.execute(
                            "UPDATE jobs SET status='failed', error_msg=?, "
                            "progress_pct=?, params_json=?, finished_at=CURRENT_TIMESTAMP WHERE id=?",
                            (_msg[:500], round(100.0 * _n_done / max(1, _n_total), 1),
                             _new_pjson, job["id"]),
                        )
                    failed += 1
                continue
            _trial_name = _params.get("trial_name", "")
            with get_db_ctx() as db:
                subj = db.execute(
                    "SELECT name FROM subjects WHERE id = ?",
                    (job["subject_id"],),
                ).fetchone()
            if subj and _trial_name:
                status_file = f"{remote_cfg.work_dir}/hrnet_jobs/{subj['name']}_{subj['name']}_{_trial_name}/status.json"
            else:
                status_file = None
        else:
            status_file = None

        remote_status = _read_remote_status(remote_cfg, status_file) if status_file else None
        remote_done = remote_status and remote_status.get("status") == "completed"
        remote_failed = remote_status and remote_status.get("status") == "failed"

        # ── Case 1: Remote job completed ──────────────────────────
        if remote_done:
            # Mark completed SYNCHRONOUSLY so DB is correct even if
            # --reload kills the download thread immediately after.
            logger.info(f"Job {job['id']} ({job_type}): remote completed — marking done synchronously")
            with get_db_ctx() as db:
                db.execute(
                    """UPDATE jobs SET status = 'completed', progress_pct = 100,
                       finished_at = CURRENT_TIMESTAMP WHERE id = ?""",
                    (job["id"],),
                )

            # Spawn best-effort background download (if killed, next startup will retry)
            if job_type in PREPROCESS_JOB_TYPES:
                _spawn_preprocess_download(job, remote_cfg, log_path, registry)
            elif job_type in TRAIN_JOB_TYPES and subj:
                _spawn_train_download(job, subj, s, remote_cfg, log_path, registry)
            elif job_type == "hrnet" and subj:
                # Download HRNet results in background.  (Don't `import
                # threading` here — module-level import at line 4 already
                # provides it; a local import would shadow the global and
                # turn `threading` into an unbound local in the rest of
                # the function.)
                def _dl_hrnet():
                    try:
                        from .services.skeleton_data import _skeleton_dir
                        _params = _json.loads(job.get("params_json") or "{}") if job.get("params_json") else {}
                        _tn = _params.get("trial_name", "")
                        local_dir = _skeleton_dir(subj["name"]) / f"{subj['name']}_{_tn}"
                        local_dir.mkdir(parents=True, exist_ok=True)
                        remote_base = f"{remote_cfg.work_dir}/hrnet_jobs/{subj['name']}_{subj['name']}_{_tn}/output/{subj['name']}_{_tn}"
                        from .services.remote import _scp_base_args
                        import subprocess
                        for fname in ["hrnet_w18_heatmaps.npz", "hand_crop.json"]:
                            subprocess.run(
                                _scp_base_args(remote_cfg) + [f"{remote_cfg.host}:{remote_base}/{fname}", str(local_dir / fname)],
                                capture_output=True, timeout=300,
                            )
                        logger.info(f"Downloaded HRNet results for {subj['name']}")
                    except Exception as e:
                        logger.warning(f"HRNet download failed: {e}")
                threading.Thread(target=_dl_hrnet, daemon=True).start()

            resumed += 1
            continue

        # ── Case 2: Remote job explicitly failed ──────────────────
        if remote_failed:
            error = remote_status.get("error", "Unknown error")
            logger.info(f"Job {job['id']} ({job_type}): remote failed — {error}")
            with get_db_ctx() as db:
                db.execute(
                    "UPDATE jobs SET status = 'failed', error_msg = ?, "
                    "finished_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (f"Remote: {error}", job["id"]),
                )
            failed += 1
            continue

        # ── Case 3: Remote process still alive ────────────────────
        if proc_alive:
            logger.info(f"Job {job['id']} ({job_type}): PID {remote_pid} alive, resuming monitor")
            if job_type in PREPROCESS_JOB_TYPES:
                thread = threading.Thread(
                    target=resume_preprocess_monitor,
                    kwargs=dict(
                        job_id=job["id"], cfg=remote_cfg,
                        log_path=log_path, registry=registry,
                    ),
                    daemon=True,
                )
                thread.start()
                registry._threads[job["id"]] = thread
            elif job_type in TRAIN_JOB_TYPES and subj:
                subject_name = subj["name"]
                local_dlc_dir = s.dlc_path / subject_name

                def _on_resume_complete(jid, returncode, _subj=subject_name):
                    if returncode == 0:
                        from .services.dlc_wrapper import fix_project_path
                        try:
                            fix_project_path(_subj)
                        except Exception:
                            pass
                        with get_db_ctx() as db2:
                            db2.execute(
                                "UPDATE subjects SET stage = 'trained', updated_at = CURRENT_TIMESTAMP "
                                "WHERE name = ?", (_subj,),
                            )

                thread = threading.Thread(
                    target=remote_train_monitor,
                    kwargs=dict(
                        job_id=job["id"], cfg=remote_cfg,
                        local_dlc_dir=local_dlc_dir,
                        subject_name=subject_name,
                        log_path=log_path,
                        progress_parser=None,
                        on_complete=_on_resume_complete,
                        registry=registry,
                        resume=True,
                    ),
                    daemon=True,
                )
                thread.start()
                registry._threads[job["id"]] = thread

            resumed += 1
            continue

        # ── Case 4: Process dead + no completed/failed status ─────
        with get_db_ctx() as db:
            db.execute(
                "UPDATE jobs SET status = 'failed', error_msg = 'Remote process exited without status', "
                "finished_at = CURRENT_TIMESTAMP WHERE id = ?",
                (job["id"],),
            )
        failed += 1

    logger.info(f"Startup recovery: {resumed} resumed, {failed} marked failed")

    # ── Resume any pending downloads for previously-completed jobs ──
    _resume_pending_downloads(s, remote_cfg)


def _spawn_preprocess_download(job, remote_cfg, log_path, registry):
    """Spawn a best-effort background download for a completed preprocess job."""
    from .services.remote import resume_preprocess_monitor
    thread = threading.Thread(
        target=resume_preprocess_monitor,
        kwargs=dict(
            job_id=job["id"], cfg=remote_cfg,
            log_path=log_path, registry=registry,
        ),
        daemon=True,
    )
    thread.start()
    registry._threads[job["id"]] = thread


def _spawn_train_download(job, subj, settings, remote_cfg, log_path, registry):
    """Spawn a best-effort background download for a completed train job."""
    from .services.remote import remote_train_monitor
    subject_name = subj["name"]
    local_dlc_dir = settings.dlc_path / subject_name

    def _on_dl_complete(jid, returncode, _subj=subject_name):
        if returncode == 0:
            from .services.dlc_wrapper import fix_project_path
            try:
                fix_project_path(_subj)
            except Exception:
                pass
            from .db import get_db_ctx
            with get_db_ctx() as db2:
                db2.execute(
                    "UPDATE subjects SET stage = 'trained', updated_at = CURRENT_TIMESTAMP "
                    "WHERE name = ?", (_subj,),
                )

    thread = threading.Thread(
        target=remote_train_monitor,
        kwargs=dict(
            job_id=job["id"], cfg=remote_cfg,
            local_dlc_dir=local_dlc_dir,
            subject_name=subject_name,
            log_path=log_path,
            progress_parser=None,
            on_complete=_on_dl_complete,
            registry=registry,
            resume=True,
        ),
        daemon=True,
    )
    thread.start()
    registry._threads[job["id"]] = thread


def _resume_pending_downloads(s, remote_cfg):
    """Check for completed preprocess jobs whose results weren't fully downloaded.

    Compares each job's subject list against local .deidentified markers.
    If any subjects are missing their blurred videos, spawns a download thread.
    This ensures downloads eventually complete even after repeated --reload kills.
    """
    if not remote_cfg:
        return

    import json as _json
    from .db import get_db_ctx
    from .services.jobs import registry

    with get_db_ctx() as db:
        # Find completed preprocess jobs that have a queue entry with subject list
        rows = db.execute(
            """SELECT j.id, j.job_type, j.log_path, jq.subject_ids
               FROM jobs j
               JOIN job_queue jq ON jq.job_id = j.id
               WHERE j.status = 'completed'
               AND j.job_type IN ('blur', 'mediapipe', 'mediapipe+blur')
               AND j.finished_at > datetime('now', '-7 days')"""
        ).fetchall()

    for row in rows:
        try:
            subjects = _json.loads(row["subject_ids"])
        except (TypeError, _json.JSONDecodeError):
            continue

        job_type = row["job_type"]
        has_blur = job_type in ("blur", "mediapipe+blur")

        if has_blur:
            # Check which subjects are missing the .deidentified marker
            missing = [
                subj for subj in subjects
                if not (s.dlc_path / subj / ".deidentified").exists()
            ]
            if not missing:
                continue

            logger.info(
                f"Job {row['id']} ({job_type}): {len(missing)}/{len(subjects)} "
                f"subjects missing downloads, resuming"
            )
            log_path = row["log_path"] or str(s.dlc_path / ".logs" / f"job_{row['id']}.log")

            from .services.remote import resume_preprocess_monitor
            thread = threading.Thread(
                target=resume_preprocess_monitor,
                kwargs=dict(
                    job_id=row["id"], cfg=remote_cfg,
                    log_path=log_path, registry=registry,
                ),
                daemon=True,
            )
            thread.start()
            registry._threads[row["id"]] = thread
            # Only resume one download at a time to avoid hammering SSH
            break


EXAMPLE_SUBJECT_NAME = "Example"


def _ensure_example_subject(settings):
    """Create the built-in Example subject if sample data exists.

    Uses the downloaded sample video (Con01_R1.mp4) directly as Example_R1.mp4
    via symlink (or copy on Windows). No re-encoding needed.

    Can be hidden via settings.show_example_subject = False.
    """
    import os
    from .config import DATA_DIR
    from .db import get_db_ctx

    sample_video = DATA_DIR / "sample_data" / "Con01_R1.mp4"
    if not sample_video.exists():
        return

    with get_db_ctx() as db:
        existing = db.execute(
            "SELECT * FROM subjects WHERE name = ?", (EXAMPLE_SUBJECT_NAME,)
        ).fetchone()

        if existing:
            # Ensure group_label and camera_name are set (may be missing from older installs)
            needs_update = (
                existing.get("group_label") != "Control"
                or not existing.get("camera_name")
            )
            if needs_update:
                db.execute(
                    "UPDATE subjects SET group_label = 'Control', diagnosis = 'Control', "
                    "camera_name = 'camera1' WHERE name = ?",
                    (EXAMPLE_SUBJECT_NAME,),
                )
            return

        # Create the subject
        db.execute(
            """INSERT INTO subjects (name, stage, dlc_dir, camera_mode, camera_name, diagnosis, group_label)
               VALUES (?, 'created', ?, 'stereo', 'camera1', 'Control', 'Control')""",
            (EXAMPLE_SUBJECT_NAME, EXAMPLE_SUBJECT_NAME),
        )

    # Link sample video as Example_R1.mp4 (no re-encoding)
    video_dir = settings.video_path
    video_dir.mkdir(parents=True, exist_ok=True)
    dest = video_dir / f"{EXAMPLE_SUBJECT_NAME}_R1.mp4"
    if not dest.exists():
        try:
            os.symlink(str(sample_video.resolve()), str(dest))
            logger.info(f"Created Example subject: symlinked {sample_video.name} → {dest}")
        except (OSError, NotImplementedError):
            # Windows without developer mode can't symlink — copy instead
            import shutil
            shutil.copy2(str(sample_video), str(dest))
            logger.info(f"Created Example subject: copied {sample_video.name} → {dest}")

    # Create DLC directory
    dlc_dir = settings.dlc_path / EXAMPLE_SUBJECT_NAME
    dlc_dir.mkdir(parents=True, exist_ok=True)


def _migrate_mano_to_skeleton(s) -> None:
    """One-shot rename of legacy mano* artefacts to skeleton* equivalents.

    On-disk per subject:
      <dlc>/<subj>/mano/                          → skeleton/
      <dlc>/<subj>/skeleton/<trial>/mano_fit.npz             → skeleton_v1.npz
      <dlc>/<subj>/skeleton/<trial>/mano_fit_v2_legacy.npz   → skeleton_v2.npz
      <dlc>/<subj>/skeleton/<trial>/mano_fit_v2.npz          → skeleton_v3.npz
      <dlc>/<subj>/skeleton/<trial>/mano_fit_v2_prev1.npz    → skeleton_v3_prev1.npz
      *params.json sidecars renamed the same way

    DB ``jobs.job_type`` column:
      mano_fit      → skeleton_v1
      mano_fit_v2   → skeleton_v2
      mano_fit_v3   → skeleton_v3

    Idempotent: skips renames where the destination already exists.
    """
    from pathlib import Path
    dlc_root = Path(s.dlc_path) if hasattr(s, "dlc_path") else None
    file_map = {
        "mano_fit.npz":                   "skeleton_v1.npz",
        "mano_fit_v2_legacy.npz":         "skeleton_v2.npz",
        "mano_fit_v2.npz":                "skeleton_v3.npz",
        "mano_fit_v2_prev1.npz":          "skeleton_v3_prev1.npz",
        "mano_fit_params.json":           "skeleton_v1_params.json",
        "mano_fit_v2_legacy_params.json": "skeleton_v2_params.json",
        "mano_fit_v2_params.json":        "skeleton_v3_params.json",
    }
    renamed_dirs = 0
    renamed_files = 0
    if dlc_root and dlc_root.exists():
        for subj_dir in dlc_root.iterdir():
            if not subj_dir.is_dir():
                continue
            legacy_dir = subj_dir / "mano"
            skel_dir   = subj_dir / "skeleton"
            if legacy_dir.exists() and not skel_dir.exists():
                try:
                    legacy_dir.rename(skel_dir)
                    renamed_dirs += 1
                except OSError as e:
                    logger.warning(f"mano→skeleton rename failed for {subj_dir.name}: {e}")
                    continue
            target = skel_dir if skel_dir.exists() else None
            if not target:
                continue
            for trial_dir in target.iterdir():
                if not trial_dir.is_dir():
                    continue
                for old, new in file_map.items():
                    old_p = trial_dir / old
                    new_p = trial_dir / new
                    if old_p.exists() and not new_p.exists():
                        try:
                            old_p.rename(new_p)
                            renamed_files += 1
                        except OSError as e:
                            logger.warning(f"file rename failed {old_p}: {e}")
        if renamed_dirs or renamed_files:
            logger.info(f"Skeleton migration: renamed {renamed_dirs} dir(s) + "
                        f"{renamed_files} file(s) from mano* → skeleton*")

    from .db import get_db_ctx
    try:
        with get_db_ctx() as db:
            job_type_map = {
                "mano_fit":    "skeleton_v1",
                "mano_fit_v2": "skeleton_v2",
                "mano_fit_v3": "skeleton_v3",
            }
            n = 0
            for old, new in job_type_map.items():
                r = db.execute(
                    "UPDATE jobs SET job_type = ? WHERE job_type = ?",
                    (new, old),
                )
                n += r.rowcount or 0
            if n:
                logger.info(f"Skeleton migration: updated job_type on {n} row(s)")
    except Exception as e:
        logger.warning(f"DB job_type migration skipped: {e}")


@app.on_event("startup")
def startup():
    """Initialize database and sync subjects from filesystem."""
    logger.info("Initializing database...")
    init_db()

    s = get_settings()
    if not s.is_configured:
        logger.info("App not configured. Visit /settings to set up paths.")
        return

    # One-shot rename of legacy mano* artefacts on disk + in DB.
    _migrate_mano_to_skeleton(s)

    # Handle stale jobs from prior server sessions
    _recover_stale_jobs(s)

    logger.info("Syncing subjects from filesystem...")
    from .routers.subjects import sync_from_filesystem
    result = sync_from_filesystem()
    removed = result.get('removed', 0)
    logger.info(f"Discovery: {result['created']} new, {result['updated']} updated, {removed} removed, {result['total']} total")

    # Create the built-in Example subject if it doesn't exist
    _ensure_example_subject(s)

    # Start queue manager
    from .services.queue_manager import queue_manager
    queue_manager.recover()
    queue_manager.start()


@app.get("/favicon.ico")
def favicon():
    """Return an empty 204 so browsers stop logging a 404 for /favicon.ico."""
    from fastapi.responses import Response
    return Response(status_code=204)


@app.get("/.well-known/appspecific/com.chrome.devtools.json")
def chrome_devtools_probe():
    """Chrome DevTools probes this path when it's open.  Return 204 to keep
    the log clean."""
    from fastapi.responses import Response
    return Response(status_code=204)


@app.get("/")
def index():
    """Smart landing page: videos browser if no subjects, dashboard otherwise."""
    from .db import get_db_ctx
    from fastapi.responses import RedirectResponse
    try:
        with get_db_ctx() as db:
            count = db.execute(
                "SELECT COUNT(*) as n FROM subjects WHERE name != 'Example'"
            ).fetchone()
            if count and count["n"] > 0:
                return FileResponse(str(STATIC_DIR / "index.html"))
    except Exception:
        pass
    # No subjects (or only Example) — redirect to video browser
    return RedirectResponse(url="/videos", status_code=302)


@app.get("/subjects")
def subjects_page():
    """Serve the dashboard/subjects page directly (no redirect logic)."""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/labeling")
def labeling_page(session: Optional[int] = None):
    """Serve the labeling page. Redirect to events page if session is events type."""
    if session:
        from fastapi.responses import RedirectResponse
        from .db import get_db_ctx
        with get_db_ctx() as db:
            row = db.execute(
                "SELECT session_type, subject_id FROM label_sessions WHERE id = ?", (session,)
            ).fetchone()
            if row and row["session_type"] == "events":
                # Redirect events sessions to the events page
                return RedirectResponse(
                    url=f"/labeling-select?mode=initial&subject={row['subject_id']}", status_code=302
                )
    return FileResponse(str(STATIC_DIR / "labeling.html"))


@app.get("/events")
def events_page(session: Optional[int] = None, subject: Optional[int] = None):
    """Serve the standalone events page."""
    if session:
        return FileResponse(str(STATIC_DIR / "events.html"))
    # No session — find a subject and redirect through labeling-select
    from fastapi.responses import RedirectResponse
    from .db import get_db_ctx
    if subject is None:
        # Use last subject from any active session, or first subject
        with get_db_ctx() as db:
            recent = db.execute(
                "SELECT subject_id FROM label_sessions WHERE status = 'active' ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            if recent:
                subject = recent["subject_id"]
            else:
                first = db.execute("SELECT id FROM subjects ORDER BY id LIMIT 1").fetchone()
                if first:
                    subject = first["id"]
    if subject is not None:
        return RedirectResponse(url=f"/labeling-select?mode=events&dest=events&subject={subject}", status_code=302)
    return RedirectResponse(url="/", status_code=302)


@app.get("/mediapipe")
def mediapipe_page():
    """Serve the MediaPipe page (same labeling UI, different mode)."""
    return FileResponse(str(STATIC_DIR / "labeling.html"))


@app.get("/deidentify")
def deidentify_page():
    """Serve the interactive deidentify/blur page."""
    return FileResponse(str(STATIC_DIR / "deidentify.html"))


@app.get("/analyze")
def analyze_page(subject: Optional[int] = None, trial: Optional[int] = None):
    """Redirect /analyze to /skeleton (Auto page).

    Preserves both ``subject`` and ``trial`` query params so deep-links
    (e.g. from the Jobs-page trial chips) land on the right trial.
    """
    from fastapi.responses import RedirectResponse
    parts = []
    if subject is not None:
        parts.append(f"subject={subject}")
    if trial is not None:
        parts.append(f"trial={trial}")
    qs = ("?" + "&".join(parts)) if parts else ""
    return RedirectResponse(url=f"/labels{qs}", status_code=302)


@app.get("/labels")
def labels_page():
    """Serve the Skeleton viewer page."""
    return FileResponse(str(STATIC_DIR / "labels.html"))


@app.get("/preproc")
def preproc_page():
    """Serve the Pre-proc page (camera trajectory etc.)."""
    return FileResponse(str(STATIC_DIR / "preproc.html"))


@app.get("/oscillations")
def oscillations_page():
    """Serve the oscillation analysis page."""
    return FileResponse(str(STATIC_DIR / "oscillations.html"))


@app.get("/labeling-select")
def labeling_select_page(subject: Optional[int] = None, mode: Optional[str] = None, dest: Optional[str] = None):
    """Smart redirect to labeling page.

    If ``subject`` is given, create a session for that subject with the
    requested ``mode`` (or auto-detect the best mode from the subject's
    pipeline stage).  Otherwise fall back to the most recent session or
    create one for the first subject.
    """
    from fastapi.responses import RedirectResponse
    from .db import get_db_ctx

    # Stages that indicate no DLC predictions yet → default to 'initial'
    _PRE_DLC_STAGES = {
        "created", "videos_linked", "prelabeled", "labeling", "labeled",
        "committed", "training", "training_dataset_created", "trained",
    }
    # Stages that have DLC predictions but no corrections → default to 'corrections'
    _PRE_CORRECTIONS_STAGES = {"analyzed", "refined"}
    # Valid session types
    _VALID_MODES = {"initial", "refine", "corrections", "events", "final"}

    def _get_or_create_session(db, sid: int, stype: str) -> Optional[int]:
        """Return an existing active session of the same type, or create one."""
        # Reuse existing active session (prefer one with labels)
        existing = db.execute(
            """SELECT ls.id, COUNT(fl.id) AS label_count
               FROM label_sessions ls
               LEFT JOIN frame_labels fl ON fl.session_id = ls.id
               WHERE ls.subject_id = ? AND ls.session_type = ? AND ls.status = 'active'
               GROUP BY ls.id
               ORDER BY label_count DESC, ls.id DESC
               LIMIT 1""",
            (sid, stype),
        ).fetchone()
        if existing:
            return existing["id"]
        db.execute(
            "INSERT INTO label_sessions (subject_id, session_type) "
            "VALUES (?, ?)",
            (sid, stype),
        )
        db.commit()
        row = db.execute(
            "SELECT id FROM label_sessions WHERE subject_id = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (sid,),
        ).fetchone()
        return row["id"] if row else None

    def _smart_mode(stage: str, target_dest: str | None = None) -> str:
        """Pick the best labeling mode based on the subject's pipeline stage.

        When dest is 'events', default to events mode for post-analysis subjects.
        Otherwise default to 'initial' (Label tab) for the DLC page.
        """
        if target_dest == "events" and stage not in _PRE_DLC_STAGES:
            return "events"
        if stage in _PRE_DLC_STAGES:
            return "initial"
        if stage in _PRE_CORRECTIONS_STAGES:
            return "corrections"
        return "initial"

    with get_db_ctx() as db:
        # ── Subject explicitly requested ──────────────────────────
        if subject is not None:
            row = db.execute(
                "SELECT * FROM subjects WHERE id = ?", (subject,)
            ).fetchone()
            if not row:
                return RedirectResponse(url="/", status_code=302)

            # Determine session type
            if mode and mode in _VALID_MODES:
                session_type = mode
            else:
                session_type = _smart_mode(row["stage"] or "created", dest)

            try:
                sid = _get_or_create_session(db, subject, session_type)
                if sid:
                    page = dest if dest else "labeling"
                    return RedirectResponse(
                        url=f"/{page}?session={sid}", status_code=302
                    )
                else:
                    logger.warning("labeling-select: _create_session returned None for subject=%s type=%s", subject, session_type)
            except Exception as exc:
                logger.exception("labeling-select: session creation failed for subject=%s type=%s: %s", subject, session_type, exc)
            return RedirectResponse(url="/", status_code=302)

        # ── No subject param — fall back to most recent active session ──
        # Prefer sessions with labels (actual work), then most recent
        # When loading DLC page (not events), exclude events sessions
        _type_filter = "" if dest == "events" else "AND ls.session_type != 'events'"
        recent_session = db.execute(
            f"""SELECT ls.id, COUNT(fl.id) AS label_count
               FROM label_sessions ls
               LEFT JOIN frame_labels fl ON fl.session_id = ls.id
               WHERE ls.status = 'active' {_type_filter}
               GROUP BY ls.id
               ORDER BY label_count DESC, ls.created_at DESC
               LIMIT 1"""
        ).fetchone()

        if recent_session:
            page = dest if dest else "labeling"
            return RedirectResponse(
                url=f"/{page}?session={recent_session['id']}", status_code=302
            )

        # No session found — get first subject
        first_subject = db.execute(
            "SELECT * FROM subjects ORDER BY id LIMIT 1"
        ).fetchone()

        if not first_subject:
            return RedirectResponse(url="/", status_code=302)

        subject_id = first_subject["id"]
        session_type = _smart_mode(first_subject["stage"] or "created", dest)

        try:
            sid = _get_or_create_session(db, subject_id, session_type)
            if sid:
                page = dest if dest else "labeling"
                return RedirectResponse(
                    url=f"/{page}?session={sid}", status_code=302
                )
        except Exception:
            pass

        return RedirectResponse(url="/", status_code=302)


@app.get("/mediapipe-select")
def mediapipe_select_page(subject: Optional[int] = None):
    """Redirect to the new Analyze page."""
    from fastapi.responses import RedirectResponse
    if subject:
        return RedirectResponse(url=f"/labels?subject={subject}", status_code=302)
    return RedirectResponse(url="/labels", status_code=302)


@app.get("/results")
def results_page():
    """Serve the results page."""
    return FileResponse(str(STATIC_DIR / "results.html"))


@app.get("/settings")
def settings_page():
    """Serve the settings page."""
    return FileResponse(str(STATIC_DIR / "settings.html"))


@app.get("/remote")
def remote_page():
    """Serve the remote jobs page."""
    return FileResponse(str(STATIC_DIR / "remote.html"))


@app.get("/onboarding")
def onboarding_page():
    """Serve the subject onboarding page."""
    return FileResponse(str(STATIC_DIR / "onboarding.html"))


@app.get("/videos")
def videos_page():
    """Serve the videos viewer page."""
    return FileResponse(str(STATIC_DIR / "videos.html"))



@app.get("/calibration")
def calibration_page():
    """Serve the camera calibration page."""
    return FileResponse(str(STATIC_DIR / "calibration.html"))


@app.get("/tutorials")
def tutorials_page():
    """Serve the tutorials index page."""
    return FileResponse(str(STATIC_DIR / "tutorials.html"))


@app.get("/tutorial")
def tutorial_page():
    """Serve a single tutorial viewer page."""
    return FileResponse(str(STATIC_DIR / "tutorial.html"))


if __name__ == "__main__":
    import uvicorn
    s = get_settings()
    uvicorn.run("movement_tracker.app:app", host=s.host, port=s.port, reload=True)
