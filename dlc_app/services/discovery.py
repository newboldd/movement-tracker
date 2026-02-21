"""Scan dlc/ directories to infer subject state from filesystem artifacts."""

import glob
import re
from pathlib import Path

from ..config import get_settings


# Directories that are label-set bundles, not real subjects
LABEL_SET_PATTERN = re.compile(r".*-labels-\d{4}-\d{2}-\d{2}$")
SKIP_NAMES = {"README.md"}


def _find_videos(subject_name: str) -> list[str]:
    """Find stereo videos for a subject in videos/."""
    settings = get_settings()
    video_dir = settings.video_path
    pattern = str(video_dir / f"{subject_name}_*.mp4")
    videos = sorted(glob.glob(pattern))
    if not videos:
        # Case-insensitive fallback
        all_vids = glob.glob(str(video_dir / "*.mp4"))
        prefix_lower = subject_name.lower() + "_"
        videos = sorted(
            v for v in all_vids
            if Path(v).name.lower().startswith(prefix_lower)
        )
    return [Path(v).name for v in videos]


def _has_snapshots(dlc_path: Path) -> bool:
    """Check if DLC model snapshots exist (pytorch or tensorflow)."""
    pytorch_dir = dlc_path / "dlc-models-pytorch"
    if pytorch_dir.exists():
        if list(pytorch_dir.rglob("snapshot-*.pt")):
            return True
    tf_dir = dlc_path / "dlc-models"
    if tf_dir.exists():
        if list(tf_dir.rglob("snapshot-*.data*")) or list(tf_dir.rglob("snapshot-*.index")):
            return True
    return False


def _has_labeled_data(dlc_path: Path) -> bool:
    """Check if labeled-data directory has CollectedData CSV."""
    labeled_dir = dlc_path / "labeled-data"
    if not labeled_dir.exists():
        return False
    for subdir in labeled_dir.iterdir():
        if subdir.is_dir():
            if (subdir / "CollectedData_labels.csv").exists():
                return True
    return False


def _has_labels_v1(dlc_path: Path) -> bool:
    """Check if labels_v1 (cropped stereo analysis outputs) exist."""
    # Various naming conventions
    for name in ["labels_v1", "labels_v1.0", "labels_v0.1"]:
        d = dlc_path / name
        if d.exists() and d.is_dir():
            return True
    return False


def _has_labeled_videos(dlc_path: Path) -> bool:
    """Check if labeled_videos directory exists (post-analysis)."""
    return (dlc_path / "labeled_videos").exists()


def _has_training_datasets(dlc_path: Path) -> bool:
    """Check if training-datasets directory exists."""
    return (dlc_path / "training-datasets").exists()


def _get_camera_name(dlc_path: Path) -> str | None:
    """Try to extract camera name from config.yaml."""
    config = dlc_path / "config.yaml"
    if not config.exists():
        return None
    try:
        text = config.read_text()
        # Look for video_sets entries to infer camera
        # Just return None for now - camera assignment is in calibration/
        return None
    except Exception:
        return None


def _count_labeled_frames(dlc_path: Path) -> int:
    """Count total labeled frames across all labeled-data subdirs."""
    labeled_dir = dlc_path / "labeled-data"
    if not labeled_dir.exists():
        return 0
    count = 0
    for subdir in labeled_dir.iterdir():
        if subdir.is_dir():
            count += len(list(subdir.glob("img*.png")))
    return count


def infer_stage(dlc_path: Path) -> str:
    """Infer pipeline stage from filesystem artifacts.

    Priority order (highest to lowest):
    - Has labeled_videos/ -> complete (or analyzed)
    - Has labels_v1/ -> triangulated (post-analysis cropped outputs)
    - Has snapshots -> trained
    - Has training-datasets/ -> training_dataset_created
    - Has labeled-data/ with CSV -> committed
    - Has config.yaml -> created
    """
    has_config = (dlc_path / "config.yaml").exists()
    has_labels = _has_labeled_data(dlc_path)
    has_training = _has_training_datasets(dlc_path)
    has_snaps = _has_snapshots(dlc_path)
    has_lv1 = _has_labels_v1(dlc_path)
    has_lv = _has_labeled_videos(dlc_path)

    if has_lv and has_lv1:
        return "complete"
    if has_lv1:
        return "triangulated"
    if has_lv:
        return "analyzed"
    if has_snaps:
        return "trained"
    if has_training:
        return "training_dataset_created"
    if has_labels:
        return "committed"
    if has_config:
        return "created"
    return "created"


def scan_all_subjects() -> list[dict]:
    """Scan dlc/ directory and return info for each subject."""
    settings = get_settings()
    dlc_dir = settings.dlc_path

    if not dlc_dir.exists():
        return []

    subjects = []
    for entry in sorted(dlc_dir.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name in SKIP_NAMES:
            continue
        if LABEL_SET_PATTERN.match(entry.name):
            continue
        if not (entry / "config.yaml").exists():
            continue

        videos = _find_videos(entry.name)
        stage = infer_stage(entry)

        subjects.append({
            "name": entry.name,
            "stage": stage,
            "dlc_dir": entry.name,  # Store relative name only
            "camera_name": _get_camera_name(entry),
            "video_count": len(videos),
            "videos": videos,
            "has_snapshots": _has_snapshots(entry),
            "has_labels": _has_labeled_data(entry),
            "labeled_frame_count": _count_labeled_frames(entry),
        })

    return subjects
