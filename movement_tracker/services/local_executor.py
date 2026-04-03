"""Local job executor: CPU and GPU execution via subprocess or direct Python."""
from __future__ import annotations

import logging
import os
import re
import threading
from pathlib import Path

from ..config import get_settings
from ..db import get_db_ctx
from .jobs import registry
from .dlc_wrapper import (
    fix_project_path,
    ensure_all_rounds_in_config,
    cmd_train_network,
    cmd_create_and_train,
    cmd_analyze_videos,
)
from .mediapipe_prelabel import run_mediapipe, run_pose_prelabels
from .remote_preprocess_script import run_blur_subject

logger = logging.getLogger(__name__)


def parse_dlc_training_progress(line: str) -> float | None:
    """Extract training progress from DLC output.

    Looks for patterns like:
    - "Epoch 45/100"
    - "epoch: 45/100"
    """
    # Look for epoch progress
    match = re.search(r'[Ee]poch\s*:?\s*(\d+)/(\d+)', line)
    if match:
        current = int(match.group(1))
        total = int(match.group(2))
        if total > 0:
            return (current / total) * 100.0
    return None


def parse_mediapipe_progress(line: str) -> float | None:
    """Extract MediaPipe progress from output.

    Looks for patterns like:
    - "Processing video 5/10"
    - "Frames: 150/250"
    """
    # Look for video progress
    match = re.search(r'[Vv]ideo\s*:?\s*(\d+)/(\d+)', line)
    if match:
        current = int(match.group(1))
        total = int(match.group(2))
        if total > 0:
            return (current / total) * 100.0

    # Look for frame progress
    match = re.search(r'[Ff]rames?\s*:?\s*(\d+)/(\d+)', line)
    if match:
        current = int(match.group(1))
        total = int(match.group(2))
        if total > 0:
            return (current / total) * 100.0

    return None


def parse_analysis_progress(line: str) -> float | None:
    """Extract analysis progress from output.

    Looks for patterns like:
    - "Analyzing 5/10 videos"
    - "Processing iteration 45/100"
    """
    # Look for video analysis progress
    match = re.search(r'[Aa]nalyzing\s+(\d+)/(\d+)', line)
    if match:
        current = int(match.group(1))
        total = int(match.group(2))
        if total > 0:
            return (current / total) * 100.0

    # Look for iteration progress
    match = re.search(r'[Ii]teration\s*:?\s*(\d+)/(\d+)', line)
    if match:
        current = int(match.group(1))
        total = int(match.group(2))
        if total > 0:
            return (current / total) * 100.0

    return None


class LocalExecutor:
    """Execute jobs locally on CPU or GPU."""

    def execute_train(self, subject_name: str, gpu_index: int, job_id: int, log_path: str):
        """Execute training locally.

        Args:
            subject_name: Subject to train
            gpu_index: GPU device ID (0 for first GPU, etc.)
            job_id: Database job ID
            log_path: Path to write logs
        """
        settings = get_settings()
        try:
            config_path = fix_project_path(subject_name)
        except FileNotFoundError as e:
            with get_db_ctx() as db:
                db.execute(
                    "UPDATE jobs SET status = 'failed', error_msg = ? WHERE id = ?",
                    (str(e), job_id),
                )
            return

        cmd = cmd_train_network(config_path)

        # Set CUDA device for GPU execution
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

        logger.info(f"Starting training for {subject_name} on GPU {gpu_index}")

        def on_complete(job_id, returncode):
            logger.info(f"Training for {subject_name} completed with code {returncode}")

        # Launch as subprocess
        registry.launch(
            job_id=job_id,
            cmd=cmd,
            log_path=log_path,
            progress_parser=parse_dlc_training_progress,
            on_complete=on_complete,
            env=env,
        )

    def execute_refine(self, subject_name: str, gpu_index: int, job_id: int, log_path: str):
        """Execute refinement training locally.

        Recreates the training dataset (picking up new round2+ labels) then
        resumes training from the latest checkpoint.
        """
        try:
            config_path = fix_project_path(subject_name)
        except FileNotFoundError as e:
            with get_db_ctx() as db:
                db.execute(
                    "UPDATE jobs SET status = 'failed', error_msg = ? WHERE id = ?",
                    (str(e), job_id),
                )
            return

        # Ensure all labeled-data rounds are registered in config video_sets
        added = ensure_all_rounds_in_config(config_path)
        if added:
            logger.info(f"Added video_sets entries for: {', '.join(added)}")

        cmd = cmd_create_and_train(config_path)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

        logger.info(f"Starting refinement training for {subject_name} on GPU {gpu_index}")

        def on_complete(job_id, returncode):
            logger.info(f"Refinement for {subject_name} completed with code {returncode}")

        registry.launch(
            job_id=job_id,
            cmd=cmd,
            log_path=log_path,
            progress_parser=parse_dlc_training_progress,
            on_complete=on_complete,
            env=env,
        )

    def execute_analyze(self, subject_name: str, gpu_index: int, job_id: int, log_path: str):
        """Execute analysis locally.

        Args:
            subject_name: Subject to analyze
            gpu_index: GPU device ID
            job_id: Database job ID
            log_path: Path to write logs
        """
        settings = get_settings()
        try:
            config_path = fix_project_path(subject_name)
        except FileNotFoundError as e:
            with get_db_ctx() as db:
                db.execute(
                    "UPDATE jobs SET status = 'failed', error_msg = ? WHERE id = ?",
                    (str(e), job_id),
                )
            return

        video_dir = str(settings.dlc_path / subject_name)
        cmd = cmd_analyze_videos(config_path, video_dir)

        # Set CUDA device for GPU execution
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

        logger.info(f"Starting analysis for {subject_name} on GPU {gpu_index}")

        def on_complete(job_id, returncode):
            logger.info(f"Analysis for {subject_name} completed with code {returncode}")

        # Launch as subprocess
        registry.launch(
            job_id=job_id,
            cmd=cmd,
            log_path=log_path,
            progress_parser=parse_analysis_progress,
            on_complete=on_complete,
            env=env,
        )

    def _launch_worker(self, job_type: str, subject_name: str, job_id: int,
                       log_path: str, extra_args: list[str] | None = None):
        """Launch a worker subprocess for any CPU job type.

        Uses the universal worker script (services/worker.py) which runs as a
        separate process, survives app restarts (PID tracked in DB), and reports
        progress via PROGRESS:N.N stdout lines.
        """
        import sys
        from .worker import parse_worker_progress
        from ..config import DATA_DIR

        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        python = sys.executable
        cmd = [
            python, "-m", "movement_tracker.services.worker",
            "--job-type", job_type,
            "--subject", subject_name,
            "--job-id", str(job_id),
            "--data-dir", str(DATA_DIR),
        ]
        if extra_args:
            cmd.extend(extra_args)

        logger.info(f"Launching worker subprocess: {job_type} for {subject_name} (job {job_id})")

        def on_complete(jid, returncode):
            logger.info(f"Worker {job_type} for {subject_name} exited with code {returncode}")

        registry.launch(
            job_id=job_id,
            cmd=cmd,
            log_path=log_path,
            progress_parser=parse_worker_progress,
            on_complete=on_complete,
        )

    def execute_mediapipe(self, subject_name: str, job_id: int, log_path: str):
        """Execute MediaPipe preprocessing as a subprocess."""
        self._launch_worker("mediapipe", subject_name, job_id, log_path)

    def execute_pose(self, subject_name: str, job_id: int, log_path: str):
        """Execute Pose detection as a subprocess."""
        self._launch_worker("pose", subject_name, job_id, log_path)

    def execute_deidentify(self, subject_name: str, job_id: int, log_path: str,
                           trial_idx: int | None = None):
        """Execute deidentify render as a subprocess."""
        extra = ["--trial-idx", str(trial_idx)] if trial_idx is not None else []
        self._launch_worker("deidentify", subject_name, job_id, log_path, extra_args=extra)

    def execute_blur(self, subject_names: list[str], job_id: int, log_path: str):
        """Execute face blur preprocessing as a subprocess (first subject only)."""
        # Worker handles one subject at a time; for multi-subject blur,
        # the queue manager should enqueue separate jobs per subject.
        self._launch_worker("blur", subject_names[0], job_id, log_path)

    def execute_vision(self, subject_name: str, job_id: int, log_path: str):
        """Execute Apple Vision hand detection as a subprocess."""
        self._launch_worker("vision", subject_name, job_id, log_path)


# Singleton executor instance
local_executor = LocalExecutor()
