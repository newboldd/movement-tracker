"""Local job execution: MediaPipe on CPU, DeepLabCut on GPU.

Every job runs as a subprocess with its PID recorded in the jobs table,
so a job survives the web app restarting (``--reload`` fires on any edit)
and can be re-attached to on the next boot.
"""
from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path

from ..config import DATA_DIR, get_settings
from ..db import get_db_ctx
from .jobs import registry

logger = logging.getLogger(__name__)

# Progress budget inside dlc_pipeline.py's train/refine modes: the script
# emits PROGRESS lines around training, and DeepLabCut's own "Epoch X/Y"
# output fills in the long stretch between them.
_TRAIN_PCT_START = 6.0
_TRAIN_PCT_END = 75.0

_EPOCH_RE = re.compile(r"[Ee]poch\s*:?\s*(\d+)\s*/\s*(\d+)")


def parse_worker_progress(line: str):
    """Parse a ``PROGRESS:42.5`` line. Returns (pct, epoch, total) or None."""
    line = line.strip()
    if line.startswith("PROGRESS:"):
        try:
            return (float(line.split(":", 1)[1]), None, None)
        except (ValueError, IndexError):
            return None
    return None


def parse_dlc_pipeline_progress(line: str):
    """Parse both our own PROGRESS lines and DeepLabCut's epoch counter.

    Training dominates the wall clock, and DLC reports it as ``Epoch
    X/Y`` on stdout.  Mapping that into the 6-75% band gives a bar that
    moves during the hour-long part instead of sitting at 6%.
    """
    direct = parse_worker_progress(line)
    if direct is not None:
        return direct

    m = _EPOCH_RE.search(line)
    if m:
        current, total = int(m.group(1)), int(m.group(2))
        if total > 0:
            span = _TRAIN_PCT_END - _TRAIN_PCT_START
            pct = _TRAIN_PCT_START + min(current / total, 1.0) * span
            return (pct, current, total)
    return None


class LocalExecutor:
    """Launch the fork's job types as local subprocesses."""

    # ── MediaPipe ───────────────────────────────────────────────────────

    def execute_mediapipe(self, subject_name: str, job_id: int, log_path: str,
                          static_image_mode: bool = False,
                          trial_idx: int | None = None,
                          reverse: bool = False,
                          use_bbox: bool = True):
        """Run a MediaPipe hand pass over a subject's trials.

        The three passes are stored side by side rather than overwriting
        each other, because each wins on different frames:
        ``static_image_mode`` runs the full palm detector per frame (no
        temporal tracker), ``reverse`` feeds frames in descending order so
        the tracker enters a hard frame already locked on, and the plain
        forward pass is the cheapest.  ``trial_idx`` restricts the run to
        one trial, leaving the others' saved landmarks untouched.
        """
        extra = []
        if static_image_mode:
            extra.append("--static-image-mode")
        if trial_idx is not None:
            extra += ["--trial-idx", str(trial_idx)]
        if reverse:
            extra.append("--reverse")
        if not use_bbox:
            extra.append("--no-bbox")
        self._launch_worker("mediapipe", subject_name, job_id, log_path,
                            extra_args=extra)

    # ── Frame export ────────────────────────────────────────────────────

    def execute_export_frames(self, subject_name: str, job_id: int,
                              log_path: str, n_per_trial: int = 20,
                              cameras: list[str] | None = None,
                              dest_dir: str | None = None,
                              package_name: str | None = None,
                              step: int = 1, seed: int | None = None):
        """Select frames for a subject and write them as a labeling package.

        A subprocess like every other job, and for the same reason: it
        decodes every trial of a subject once, which on a long session is
        minutes of work that must not be lost to the app reloading.
        """
        settings = get_settings()
        dest = dest_dir or str(settings.packages_path)

        extra = [
            "--n-per-trial", str(n_per_trial),
            "--dest-dir", dest,
            "--step", str(step),
        ]
        if cameras:
            extra += ["--cameras", *cameras]
        if package_name:
            extra += ["--package-name", package_name]
        if seed is not None:
            extra += ["--seed", str(seed)]

        self._launch_worker("export-frames", subject_name, job_id, log_path,
                            extra_args=extra)

    def _launch_worker(self, job_type: str, subject_name: str, job_id: int,
                       log_path: str, extra_args: list[str] | None = None,
                       progress_parser=None):
        """Launch services/worker.py as a subprocess.

        Invoked by absolute path rather than ``python -m`` because the
        portable Python that run.bat installs on Windows ships a
        ``python3XX._pth`` that overrides ``sys.path`` and ignores
        PYTHONPATH — ``-m`` then can't find the package even with cwd set.
        worker.py puts PROJECT_DIR on sys.path itself at module load.
        """
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        worker_py = str(Path(__file__).resolve().parent / "worker.py")
        cmd = [
            sys.executable, worker_py,
            "--job-type", job_type,
            "--subject", subject_name,
            "--job-id", str(job_id),
            "--data-dir", str(DATA_DIR),
        ]
        if extra_args:
            cmd.extend(extra_args)

        logger.info("Launching worker: %s for %s (job %s)",
                    job_type, subject_name, job_id)

        registry.launch(
            job_id=job_id,
            cmd=cmd,
            log_path=log_path,
            progress_parser=progress_parser or parse_worker_progress,
            on_complete=self._log_tail_on_failure(job_type, subject_name, log_path),
        )

    # ── DeepLabCut ──────────────────────────────────────────────────────

    def execute_dlc(self, mode: str, subject_name: str, job_id: int,
                    log_path: str, gpu_index: int = 0,
                    labels_dir_name: str | None = None):
        """Run the local DLC pipeline in ``train``, ``refine`` or ``analyze`` mode.

        Predictions land in ``dlc/<subject>/labels_v1`` for a first
        training run and ``labels_v2`` after refinement, which is the
        priority order ``dlc_predictions`` reads them back in.
        """
        settings = get_settings()
        dlc_dir = settings.dlc_path / subject_name
        config_path = dlc_dir / "config.yaml"

        if not config_path.exists():
            self._fail(job_id, f"No config.yaml for {subject_name} — commit "
                               f"labels first to create the DLC project")
            return

        if labels_dir_name is None:
            labels_dir_name = "labels_v2" if mode == "refine" else "labels_v1"
        labels_dir = dlc_dir / labels_dir_name
        labels_dir.mkdir(parents=True, exist_ok=True)

        with get_db_ctx() as db:
            subj = db.execute("SELECT camera_mode FROM subjects WHERE name = ?",
                              (subject_name,)).fetchone()
        camera_mode = (subj or {}).get("camera_mode") or settings.default_camera_mode

        script = str(Path(__file__).resolve().parent / "dlc_pipeline.py")
        cmd = [
            settings.python_executable or sys.executable, script,
            "--mode", mode,
            "--config-path", str(config_path),
            "--labels-dir", str(labels_dir),
            "--video-dir", str(settings.video_path),
            "--subject-name", subject_name,
            "--camera-mode", camera_mode,
            "--net-type", settings.dlc_net_type,
            "--cam-names", *(settings.camera_names or ["OS", "OD"]),
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        # Unbuffered so Epoch lines reach the progress parser as they are
        # printed rather than in 8 KB bursts.
        env["PYTHONUNBUFFERED"] = "1"

        logger.info("Starting DLC %s for %s on GPU %s (-> %s)",
                    mode, subject_name, gpu_index, labels_dir_name)

        registry.launch(
            job_id=job_id,
            cmd=cmd,
            log_path=log_path,
            progress_parser=parse_dlc_pipeline_progress,
            on_complete=self._log_tail_on_failure(
                f"dlc-{mode}", subject_name, log_path),
            env=env,
        )

    # ── DeepLabCut installation (two-tier installer, second tier) ───────

    def execute_install_dlc(self, job_id: int, log_path: str):
        """pip install the DLC requirements into the app's own venv.

        The base install deliberately leaves out deeplabcut and torch —
        several GB that someone who only labels frames never needs.  This
        is the Jobs-page button that adds them, equivalent to running
        ``./setup.sh --with-dlc``.
        """
        from ..config import PROJECT_DIR

        req = PROJECT_DIR / "requirements-dlc.txt"
        if not req.exists():
            self._fail(job_id, f"requirements-dlc.txt not found at {req}")
            return

        cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "-r", str(req)]
        logger.info("Installing DeepLabCut into %s", sys.executable)

        def on_complete(jid, returncode):
            if returncode == 0:
                # Re-probe so the Jobs page stops offering the install.
                get_settings().dlc_installed(refresh=True)
            else:
                logger.error("DeepLabCut install failed (rc=%s); see %s",
                             returncode, log_path)

        registry.launch(
            job_id=job_id,
            cmd=cmd,
            log_path=log_path,
            progress_parser=None,
            on_complete=on_complete,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _fail(job_id: int, message: str):
        logger.error("Job %s: %s", job_id, message)
        with get_db_ctx() as db:
            db.execute(
                "UPDATE jobs SET status = 'failed', error_msg = ?, "
                "finished_at = CURRENT_TIMESTAMP WHERE id = ?",
                (message, job_id),
            )

    @staticmethod
    def _log_tail_on_failure(label: str, subject_name: str, log_path: str):
        """Build an on_complete hook that echoes a failed job's log tail.

        Without this a failed job leaves nothing but an exit code in the
        UI, and the actual ImportError or traceback sits in a log file
        under <DATA_DIR>/dlc/.logs/ that nobody thinks to open.
        """
        def on_complete(job_id, returncode):
            logger.info("%s for %s exited with code %s",
                        label, subject_name, returncode)
            if returncode == 0:
                return
            try:
                if log_path and os.path.exists(log_path):
                    with open(log_path, "r", encoding="utf-8",
                              errors="replace") as f:
                        tail = f.readlines()[-30:]
                    if tail:
                        logger.error(
                            "%s for %s rc=%s — last %d lines of %s:\n%s",
                            label, subject_name, returncode, len(tail),
                            log_path, "".join(tail).rstrip())
            except OSError:
                logger.exception("Could not read worker log at %s", log_path)
        return on_complete


# Singleton executor instance
local_executor = LocalExecutor()
