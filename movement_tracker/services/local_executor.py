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

    def execute_mediapipe(self, subject_name: str, job_id: int, log_path: str):
        """Execute MediaPipe preprocessing locally.

        Args:
            subject_name: Subject to process
            job_id: Database job ID
            log_path: Path to write logs
        """
        logger.info(f"Starting MediaPipe for {subject_name}")

        # Create a thread to run MediaPipe in the background
        # This allows progress updates while the function runs
        def run_mediapipe_job():
            try:
                os.makedirs(os.path.dirname(log_path), exist_ok=True)

                with open(log_path, "w") as logfile:
                    def progress_callback(pct: float):
                        logfile.write(f"Processing: {pct:.1f}%\n")
                        logfile.flush()
                        with get_db_ctx() as db:
                            db.execute(
                                "UPDATE jobs SET progress_pct = ? WHERE id = ?",
                                (pct, job_id),
                            )

                    # Run MediaPipe directly
                    result = run_mediapipe(subject_name, progress_callback=progress_callback)
                    logfile.write(f"MediaPipe completed: {result}\n")

                # Mark as complete
                with get_db_ctx() as db:
                    db.execute(
                        """UPDATE jobs SET status = 'completed', progress_pct = 100,
                           finished_at = CURRENT_TIMESTAMP WHERE id = ?""",
                        (job_id,),
                    )
                logger.info(f"MediaPipe for {subject_name} completed")

            except Exception as e:
                logger.exception(f"MediaPipe for {subject_name} failed")
                with open(log_path, "a") as logfile:
                    logfile.write(f"\nError: {e}\n")

                with get_db_ctx() as db:
                    db.execute(
                        """UPDATE jobs SET status = 'failed', error_msg = ?,
                           finished_at = CURRENT_TIMESTAMP WHERE id = ?""",
                        (str(e), job_id),
                    )

        # Update job to running
        with get_db_ctx() as db:
            db.execute(
                """UPDATE jobs SET status = 'running', started_at = CURRENT_TIMESTAMP
                   WHERE id = ?""",
                (job_id,),
            )

        # Start thread
        thread = threading.Thread(target=run_mediapipe_job, daemon=True)
        thread.start()

    def execute_pose(self, subject_name: str, job_id: int, log_path: str):
        """Execute Pose detection locally."""
        logger.info(f"Starting Pose detection for {subject_name}")

        def run_pose_job():
            try:
                os.makedirs(os.path.dirname(log_path), exist_ok=True)

                with open(log_path, "w") as logfile:
                    def progress_callback(pct: float):
                        logfile.write(f"Processing: {pct:.1f}%\n")
                        logfile.flush()
                        with get_db_ctx() as db:
                            db.execute(
                                "UPDATE jobs SET progress_pct = ? WHERE id = ?",
                                (pct, job_id),
                            )

                    result = run_pose_prelabels(subject_name, progress_callback=progress_callback)
                    logfile.write(f"Pose detection completed: {result}\n")

                with get_db_ctx() as db:
                    db.execute(
                        """UPDATE jobs SET status = 'completed', progress_pct = 100,
                           finished_at = CURRENT_TIMESTAMP WHERE id = ?""",
                        (job_id,),
                    )
                logger.info(f"Pose detection for {subject_name} completed")

            except Exception as e:
                logger.exception(f"Pose detection for {subject_name} failed")
                with open(log_path, "a") as logfile:
                    logfile.write(f"\nError: {e}\n")

                with get_db_ctx() as db:
                    db.execute(
                        """UPDATE jobs SET status = 'failed', error_msg = ?,
                           finished_at = CURRENT_TIMESTAMP WHERE id = ?""",
                        (str(e), job_id),
                    )

        with get_db_ctx() as db:
            db.execute(
                """UPDATE jobs SET status = 'running', started_at = CURRENT_TIMESTAMP
                   WHERE id = ?""",
                (job_id,),
            )

        thread = threading.Thread(target=run_pose_job, daemon=True)
        thread.start()

    def execute_blur(self, subject_names: list[str], job_id: int, log_path: str):
        """Execute face blur preprocessing locally.

        Args:
            subject_names: Subjects to blur
            job_id: Database job ID
            log_path: Path to write logs
        """
        logger.info(f"Starting blur for {', '.join(subject_names)}")

        def run_blur_job():
            try:
                settings = get_settings()
                os.makedirs(os.path.dirname(log_path), exist_ok=True)

                with open(log_path, "w") as logfile:
                    for i, subject_name in enumerate(subject_names):
                        logfile.write(f"Processing {subject_name}...\n")
                        logfile.flush()

                        # Get videos for this subject
                        from .mediapipe_prelabel import get_subject_videos
                        videos = get_subject_videos(subject_name)

                        progress_base = (i / len(subject_names)) * 100
                        progress_span = (1 / len(subject_names)) * 100

                        # Run blur - this doesn't have progress callback in current implementation
                        # but we can estimate progress
                        run_blur_subject(
                            subject_name=subject_name,
                            videos=videos,
                            output_dir=str(settings.dlc_path),
                            overrides_file=None,
                        )

                        progress = progress_base + progress_span
                        with get_db_ctx() as db:
                            db.execute(
                                "UPDATE jobs SET progress_pct = ? WHERE id = ?",
                                (progress, job_id),
                            )

                # Mark as complete
                with get_db_ctx() as db:
                    db.execute(
                        """UPDATE jobs SET status = 'completed', progress_pct = 100,
                           finished_at = CURRENT_TIMESTAMP WHERE id = ?""",
                        (job_id,),
                    )
                logger.info(f"Blur for {', '.join(subject_names)} completed")

            except Exception as e:
                logger.exception(f"Blur for {', '.join(subject_names)} failed")
                with open(log_path, "a") as logfile:
                    logfile.write(f"\nError: {e}\n")

                with get_db_ctx() as db:
                    db.execute(
                        """UPDATE jobs SET status = 'failed', error_msg = ?,
                           finished_at = CURRENT_TIMESTAMP WHERE id = ?""",
                        (str(e), job_id),
                    )

        # Update job to running
        with get_db_ctx() as db:
            db.execute(
                """UPDATE jobs SET status = 'running', started_at = CURRENT_TIMESTAMP
                   WHERE id = ?""",
                (job_id,),
            )

        # Start thread
        thread = threading.Thread(target=run_blur_job, daemon=True)
        thread.start()


# Singleton executor instance
local_executor = LocalExecutor()
