#!/usr/bin/env python
"""Remote DLC training script for DLC labeler.

Self-contained (no dlc_app imports). Uploaded to remote host and executed
inside a tmux session. Combines training, stereo video cropping, and
DLC analysis into a single script.

Writes status.json atomically at phase transitions for the local
monitoring loop to poll.

Usage:
    python remote_train_script.py \
        --config-path /path/to/config.yaml \
        --shuffle 1 \
        --labels-dir /path/to/labels_v1 \
        --video-dir /path/to/videos \
        --subject-name MSA01 \
        --cam-names OS OD
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import tempfile
import traceback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ── Status helpers ──────────────────────────────────────────────────────

def _write_status(status_file: str, phase: str, status: str,
                  progress_pct: float = 0.0, error: str | None = None):
    """Write status.json atomically (temp + os.replace)."""
    data = {
        "phase": phase,
        "status": status,
        "progress_pct": round(progress_pct, 1),
        "error": error,
        "pid": os.getpid(),
    }
    # Write to temp file in same directory, then atomic rename
    dir_name = os.path.dirname(os.path.abspath(status_file))
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        os.replace(tmp, status_file)
    except Exception:
        # Clean up temp on failure
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ── Crop logic ──────────────────────────────────────────────────────────

def crop_stereo_videos(video_dir: str, subject_name: str,
                       labels_dir: str, cam_names: list[str]):
    """Crop stereo videos into per-camera halves.

    Finds {subject_name}_*.mp4 in video_dir, splits at midline,
    writes {stem}_{cam_name}.mp4 to labels_dir.
    """
    import cv2

    os.makedirs(labels_dir, exist_ok=True)

    pattern = os.path.join(video_dir, f"{subject_name}_*.mp4")
    videos = sorted(glob.glob(pattern))
    logger.info(f"Found {len(videos)} videos to crop")

    for vpath in videos:
        stem = os.path.splitext(os.path.basename(vpath))[0]
        ext = os.path.splitext(vpath)[1]

        out_left = os.path.join(labels_dir, f"{stem}_{cam_names[0]}{ext}")
        out_right = (os.path.join(labels_dir, f"{stem}_{cam_names[1]}{ext}")
                     if len(cam_names) > 1 else None)

        # Skip if already cropped
        if os.path.exists(out_left) and (out_right is None or os.path.exists(out_right)):
            logger.info(f"Skipping {stem} (already cropped)")
            continue

        cap = cv2.VideoCapture(vpath)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, frame = cap.read()
        if not ret:
            cap.release()
            logger.warning(f"Cannot read {vpath}")
            continue

        h, w = frame.shape[:2]
        mid = w // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Use mp4v on Linux (remote), avc1 on macOS
        codec = "avc1" if sys.platform == "darwin" else "mp4v"
        fourcc = cv2.VideoWriter_fourcc(*codec)

        writer_L = cv2.VideoWriter(out_left, fourcc, fps, (mid, h))
        writer_R = None
        if out_right:
            writer_R = cv2.VideoWriter(out_right, fourcc, fps, (w - mid, h))

        for _ in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                break
            writer_L.write(frame[:, :mid])
            if writer_R:
                writer_R.write(frame[:, mid:])

        cap.release()
        writer_L.release()
        if writer_R:
            writer_R.release()

        logger.info(f"Cropped {stem}: {n_frames} frames")

    logger.info("Crop complete")


# ── Main pipeline ───────────────────────────────────────────────────────

def run_pipeline(config_path: str, shuffle: int, labels_dir: str,
                 video_dir: str, subject_name: str, cam_names: list[str],
                 status_file: str):
    """Run the full train → crop → analyze pipeline."""
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # ── Phase: Train ────────────────────────────────────────────────
    logger.info("=== Phase: Training ===")
    _write_status(status_file, "train", "running", 5.0)

    from deeplabcut.core.engine import Engine
    import deeplabcut

    deeplabcut.train_network(config_path, shuffle=shuffle,
                             engine=Engine.PYTORCH)

    logger.info("=== Training complete ===")
    _write_status(status_file, "train", "running", 75.0)

    # ── Phase: Crop ─────────────────────────────────────────────────
    logger.info("=== Phase: Cropping ===")
    _write_status(status_file, "crop", "running", 76.0)

    crop_stereo_videos(video_dir, subject_name, labels_dir, cam_names)

    logger.info("=== Crop complete ===")
    _write_status(status_file, "crop", "running", 80.0)

    # ── Phase: Analyze ──────────────────────────────────────────────
    logger.info("=== Phase: Analyzing ===")
    _write_status(status_file, "analyze", "running", 81.0)

    deeplabcut.analyze_videos(config_path, labels_dir,
                              shuffle=shuffle, engine=Engine.PYTORCH)
    logger.info("Analysis complete")

    deeplabcut.analyze_videos_converth5_to_csv(labels_dir)
    logger.info("H5 to CSV conversion complete")

    _write_status(status_file, "analyze", "running", 90.0)

    # ── Done ────────────────────────────────────────────────────────
    logger.info("=== All phases complete ===")
    _write_status(status_file, "done", "completed", 100.0)


def main():
    parser = argparse.ArgumentParser(
        description="Remote DLC training pipeline (runs inside tmux)")
    parser.add_argument("--config-path", required=True,
                        help="Path to DLC config.yaml on remote")
    parser.add_argument("--shuffle", type=int, required=True,
                        help="DLC shuffle number")
    parser.add_argument("--labels-dir", required=True,
                        help="Output directory for cropped videos + analysis")
    parser.add_argument("--video-dir", required=True,
                        help="Directory containing source stereo videos")
    parser.add_argument("--subject-name", required=True,
                        help="Subject name (for video filename matching)")
    parser.add_argument("--cam-names", nargs="+", default=["OS", "OD"],
                        help="Camera names for cropped output files")
    parser.add_argument("--status-file", default=None,
                        help="Path for status.json (default: labels-dir/../status.json)")
    args = parser.parse_args()

    status_file = args.status_file or os.path.join(
        os.path.dirname(args.labels_dir), "status.json")

    try:
        _write_status(status_file, "train", "running", 0.0)
        run_pipeline(
            config_path=args.config_path,
            shuffle=args.shuffle,
            labels_dir=args.labels_dir,
            video_dir=args.video_dir,
            subject_name=args.subject_name,
            cam_names=args.cam_names,
            status_file=status_file,
        )
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Pipeline failed: {e}\n{tb}")
        try:
            _write_status(status_file, "failed", "failed", error=str(e))
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
