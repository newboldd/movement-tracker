"""Pipeline step triggers: crop, train, analyze, etc."""

from __future__ import annotations

import subprocess
from pathlib import Path
from fastapi import APIRouter, HTTPException

from ..config import get_settings
from ..db import get_db_ctx
from ..models import RunStepRequest
from ..services.dlc_wrapper import (
    fix_project_path,
    cmd_create_training_dataset,
    cmd_train_network,
    cmd_analyze_videos,
    cmd_create_labeled_video,
    cmd_triangulate,
    cmd_convert_h5_to_csv,
)
from ..services.jobs import registry, parse_dlc_training_progress
from ..services.video import get_subject_videos

router = APIRouter(prefix="/api/subjects", tags=["pipeline"])

# Valid pipeline steps in order
VALID_STEPS = [
    "create_training_dataset",
    "train",
    "crop",
    "analyze",
    "create_labeled_video",
    "triangulate",
]

# Steps that require DeepLabCut (crop and create_training_dataset are native)
DLC_STEPS = {"train", "analyze", "create_labeled_video", "triangulate"}


def _check_dlc_available():
    """Raise HTTPException if DeepLabCut is not installed."""
    settings = get_settings()
    try:
        subprocess.run(
            [settings.python_executable, "-c", "import deeplabcut"],
            capture_output=True, timeout=15,
        ).check_returncode()
    except Exception:
        raise HTTPException(
            400,
            "DeepLabCut is not installed. Install it to use training/analysis features. "
            "Labeling and video cropping work without it.",
        )


def _create_job(db, subject_id: int, job_type: str) -> dict:
    """Insert a new job record and return it."""
    settings = get_settings()
    log_dir = settings.dlc_path / ".logs"
    log_dir.mkdir(exist_ok=True)
    log_path = str(log_dir / f"job_{job_type}_{subject_id}.log")

    db.execute(
        """INSERT INTO jobs (subject_id, job_type, status, log_path)
           VALUES (?, ?, 'pending', ?)""",
        (subject_id, job_type, log_path),
    )
    return db.execute(
        "SELECT * FROM jobs WHERE subject_id = ? ORDER BY id DESC LIMIT 1",
        (subject_id,),
    ).fetchone()


@router.post("/{subject_id}/run-step")
def run_step(subject_id: int, req: RunStepRequest) -> dict:
    """Trigger a pipeline step for a subject."""
    if req.step not in VALID_STEPS:
        raise HTTPException(400, f"Invalid step '{req.step}'. Valid: {VALID_STEPS}")

    if req.step in DLC_STEPS:
        _check_dlc_available()

    settings = get_settings()

    with get_db_ctx() as db:
        subj = db.execute(
            "SELECT * FROM subjects WHERE id = ?", (subject_id,)
        ).fetchone()
        if not subj:
            raise HTTPException(404, "Subject not found")

        job = _create_job(db, subject_id, req.step)

    subject_name = subj["name"]
    dlc_dir = settings.dlc_path
    cam_names = settings.camera_names

    try:
        config_path = fix_project_path(subject_name)
    except FileNotFoundError:
        raise HTTPException(400, f"No DLC config for {subject_name}")

    if req.step == "create_training_dataset":
        _do_create_training_dataset(subject_name, job["id"], config_path)
        return {"job_id": job["id"], "status": "completed"}

    elif req.step == "train":
        cmd = cmd_train_network(config_path)
        progress_parser = parse_dlc_training_progress
        next_stage = "trained"

    elif req.step == "crop":
        # Crop is done inline (not a DLC call) — use our crop script
        _do_crop(subject_name, job["id"])
        return {"job_id": job["id"], "status": "completed"}

    elif req.step == "analyze":
        labels_dir = str(dlc_dir / subject_name / "labels_v1")
        cmd = cmd_analyze_videos(config_path, labels_dir)
        progress_parser = None
        next_stage = "analyzed"

    elif req.step == "create_labeled_video":
        labels_dir = str(dlc_dir / subject_name / "labels_v1")
        cmd = cmd_create_labeled_video(config_path, labels_dir)
        progress_parser = None
        next_stage = "analyzed"

    elif req.step == "triangulate":
        if not settings.calibration_3d_config:
            raise HTTPException(
                400,
                "Configure stereo calibration in Settings to enable triangulation.",
            )
        labels_dir = str(dlc_dir / subject_name / "labels_v1")
        config_3d = settings.calibration_3d_config
        _setup_3d_config(config_path, config_3d)
        cmd = cmd_triangulate(config_3d, labels_dir)
        progress_parser = None
        next_stage = "triangulated"

    def on_complete(jid, returncode):
        if returncode == 0:
            with get_db_ctx() as db:
                db.execute(
                    "UPDATE subjects SET stage = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (next_stage, subject_id),
                )

    registry.launch(
        job["id"], cmd, job["log_path"],
        progress_parser=progress_parser,
        on_complete=on_complete,
    )

    return {"job_id": job["id"], "status": "running"}


def _do_crop(subject_name: str, job_id: int):
    """Crop stereo videos into left/right halves (inline, not subprocess)."""
    import cv2

    settings = get_settings()
    dlc_dir = settings.dlc_path
    cam_names = settings.camera_names

    labels_dir = dlc_dir / subject_name / "labels_v1"
    labels_dir.mkdir(exist_ok=True)

    videos = get_subject_videos(subject_name)
    if not videos:
        with get_db_ctx() as db:
            db.execute(
                "UPDATE jobs SET status = 'failed', error_msg = 'No videos found' WHERE id = ?",
                (job_id,),
            )
        return

    for vid_path in videos:
        vidname = Path(vid_path).stem
        ext = Path(vid_path).suffix

        out_left = labels_dir / f"{vidname}_{cam_names[0]}{ext}"
        out_right = labels_dir / f"{vidname}_{cam_names[1]}{ext}" if len(cam_names) > 1 else None

        if out_left.exists() and (out_right is None or out_right.exists()):
            continue

        cap = cv2.VideoCapture(vid_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, frame = cap.read()
        if not ret:
            cap.release()
            continue

        h, w = frame.shape[:2]
        midline = w // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        writer_left = cv2.VideoWriter(
            str(out_left), cv2.VideoWriter_fourcc(*"avc1"), fps, (midline, h)
        )
        writer_right = None
        if out_right:
            writer_right = cv2.VideoWriter(
                str(out_right), cv2.VideoWriter_fourcc(*"avc1"), fps, (w - midline, h)
            )

        for _ in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                break
            writer_left.write(frame[:, :midline])
            if writer_right:
                writer_right.write(frame[:, midline:])

        cap.release()
        writer_left.release()
        if writer_right:
            writer_right.release()

    with get_db_ctx() as db:
        db.execute(
            """UPDATE jobs SET status = 'completed', progress_pct = 100,
               finished_at = CURRENT_TIMESTAMP WHERE id = ?""",
            (job_id,),
        )
        db.execute(
            "UPDATE subjects SET stage = 'cropped', updated_at = CURRENT_TIMESTAMP WHERE id = (SELECT subject_id FROM jobs WHERE id = ?)",
            (job_id,),
        )


def _do_create_training_dataset(subject_name: str, job_id: int, config_path: str):
    """Create DLC training dataset natively (no DLC dependency).

    Reads labeled-data CSVs, builds a train/test split, and writes
    the training-datasets directory structure that DLC expects.
    """
    import random
    import yaml

    settings = get_settings()
    dlc_path = settings.dlc_path / subject_name

    # Collect all labeled frames from labeled-data subdirs
    labeled_dir = dlc_path / "labeled-data"
    if not labeled_dir.exists():
        with get_db_ctx() as db:
            db.execute(
                "UPDATE jobs SET status = 'failed', error_msg = 'No labeled data found. Commit labels first.' WHERE id = ?",
                (job_id,),
            )
        return

    # Read config.yaml for training fraction
    config = {}
    config_file = Path(config_path)
    if config_file.exists():
        config = yaml.safe_load(config_file.read_text()) or {}

    train_fraction = 0.95
    if "TrainingFraction" in config and config["TrainingFraction"]:
        train_fraction = config["TrainingFraction"][0]

    scorer = config.get("scorer", settings.dlc_scorer)
    net_type = config.get("default_net_type", settings.dlc_net_type)

    # Gather all image paths from labeled-data subdirectories
    all_images = []
    for subdir in sorted(labeled_dir.iterdir()):
        if not subdir.is_dir():
            continue
        csv_file = subdir / "CollectedData_labels.csv"
        if csv_file.exists():
            for img in sorted(subdir.glob("img*.png")):
                all_images.append(str(img.relative_to(dlc_path)))

    if not all_images:
        with get_db_ctx() as db:
            db.execute(
                "UPDATE jobs SET status = 'failed', error_msg = 'No labeled images found' WHERE id = ?",
                (job_id,),
            )
        return

    # Train/test split
    random.seed(42)
    shuffled = all_images[:]
    random.shuffle(shuffled)
    n_train = max(1, int(len(shuffled) * train_fraction))
    train_set = sorted(shuffled[:n_train])
    test_set = sorted(shuffled[n_train:]) if n_train < len(shuffled) else []

    # Write training-datasets directory
    iteration = config.get("iteration", 0)
    frac_str = str(int(train_fraction * 100))
    dataset_name = f"iteration-{iteration}/UnaugmentedDataSet_{config.get('Task', 'project')}{config.get('date', '')}"
    dataset_dir = dlc_path / "training-datasets" / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Write train/test index files
    (dataset_dir / "CollectedData_train.csv").write_text("\n".join(train_set))
    if test_set:
        (dataset_dir / "CollectedData_test.csv").write_text("\n".join(test_set))

    # Update job and subject status
    with get_db_ctx() as db:
        db.execute(
            """UPDATE jobs SET status = 'completed', progress_pct = 100,
               finished_at = CURRENT_TIMESTAMP WHERE id = ?""",
            (job_id,),
        )
        db.execute(
            "UPDATE subjects SET stage = 'training_dataset_created', updated_at = CURRENT_TIMESTAMP WHERE id = (SELECT subject_id FROM jobs WHERE id = ?)",
            (job_id,),
        )


def _setup_3d_config(config_path: str, config_3d_path: str):
    """Write config3d.yaml with this subject's DLC config for both cameras."""
    settings = get_settings()
    config_3d = Path(config_3d_path)
    template_path = config_3d.parent / "config.yaml"
    if not template_path.exists():
        return

    cam_names = settings.camera_names
    lines = template_path.read_text().splitlines(keepends=True)
    new_lines = []
    for line in lines:
        matched = False
        for cam in cam_names:
            if line.strip().startswith(f"config_file_{cam}:"):
                new_lines.append(f"config_file_{cam}: {config_path}\n")
                matched = True
                break
        if not matched:
            new_lines.append(line)

    config_3d.write_text("".join(new_lines))
