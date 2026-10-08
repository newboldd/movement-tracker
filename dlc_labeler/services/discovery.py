"""Scan dlc/ directories to infer subject state from filesystem artifacts."""
from __future__ import annotations

import glob
import os
import re
from pathlib import Path

from ..config import get_settings


# Directories that are label-set bundles, not real subjects
LABEL_SET_PATTERN = re.compile(r".*-labels-\d{4}-\d{2}-\d{2}$")
SKIP_NAMES = {"README.md"}


def _find_videos(subject_name: str) -> list[str]:
    """Find videos for a subject in videos/.

    In multicam mode, groups per-camera files so each trial is counted once
    (e.g. Subject_L1_cam0.mp4 + Subject_L1_cam1.mp4 → one trial "Subject_L1").
    Returns a list of display names (trial names for multicam, filenames otherwise).
    """
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

    if settings.default_camera_mode == "multicam" and len(videos) > 1:
        # Group by trial stem — strip last _segment as camera name
        trial_stems = set()
        prefix = subject_name + "_"
        prefix_lower = prefix.lower()
        for v in videos:
            stem = Path(v).stem
            if stem.lower().startswith(prefix_lower):
                rest = stem[len(prefix):]
            else:
                rest = stem
            parts = rest.rsplit("_", 1)
            if len(parts) == 2:
                trial_stems.add(f"{subject_name}_{parts[0]}")
            else:
                trial_stems.add(stem)
        # Only use grouped names if grouping actually reduced count
        if len(trial_stems) < len(videos):
            return sorted(trial_stems)

    return [Path(v).name for v in videos]


MIN_VALID_VIDEO_SIZE = 100_000  # 100KB — anything smaller is corrupt/partial


def _find_deidentified_videos(subject_name: str) -> list[str]:
    """Find deidentified videos for a subject in videos/deidentified/.

    Filters out files smaller than 100KB (corrupt/partial renders).
    """
    settings = get_settings()
    deident_dir = settings.video_path / "deidentified"
    if not deident_dir.exists():
        return []
    pattern = str(deident_dir / f"{subject_name}_*.mp4")
    videos = sorted(glob.glob(pattern))
    if not videos:
        all_vids = glob.glob(str(deident_dir / "*.mp4"))
        prefix_lower = subject_name.lower() + "_"
        videos = sorted(
            v for v in all_vids
            if Path(v).name.lower().startswith(prefix_lower)
        )
    # Filter out corrupt/partial files
    return [Path(v).name for v in videos if os.path.getsize(v) >= MIN_VALID_VIDEO_SIZE]


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
    """Check if labels_v1 (DLC model prediction CSVs) exist.

    Requires actual CSV files in the directory — an empty dir or one
    with only mediapipe/training data does not count.
    """
    for name in ["labels_v1", "labels_v1.0", "labels_v0.1"]:
        d = dlc_path / name
        if d.exists() and d.is_dir() and list(d.glob("*.csv")):
            return True
    return False


def _has_mediapipe(dlc_path: Path) -> bool:
    """Check if MediaPipe prelabels exist for this subject."""
    # Per-trial layout: any <subject>/<trial>/mediapipe.npz counts.
    try:
        for trial_dir in dlc_path.iterdir():
            if trial_dir.is_dir() and (trial_dir / "mediapipe.npz").exists():
                return True
    except OSError:
        pass
    return (dlc_path / "mediapipe_prelabels.npz").exists()


def _has_labels_v2(dlc_path: Path) -> bool:
    """Check if labels_v2 (refined DLC outputs) exist."""
    d = dlc_path / "labels_v2"
    return d.exists() and d.is_dir()


def _has_corrections(dlc_path: Path) -> bool:
    """Check if corrections (manually corrected DLC outputs) exist."""
    d = dlc_path / "corrections"
    return d.exists() and d.is_dir()


def _get_camera_name(dlc_path: Path) -> str | None:
    """The subject's calibration camera, if the project records one.

    DLC's config.yaml does not carry one — the camera a subject was shot
    on is assigned in the data directory's camera_assignments.yaml, or
    per subject in the database.  Kept so discovery's row shape matches
    the subjects table.
    """
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
    - Has corrections/ -> corrected
    - Has labels_v2/ -> refined
    - Has labels_v1/ -> analyzed
    - Has snapshots -> trained
    - Has labeled-data/ with CSV -> committed
    - Has config.yaml -> created
    """
    if _has_corrections(dlc_path):
        return "corrected"
    if _has_labels_v2(dlc_path):
        return "refined"
    if _has_labels_v1(dlc_path):
        return "analyzed"
    if _has_snapshots(dlc_path):
        return "trained"
    if _has_labeled_data(dlc_path):
        return "committed"
    if (dlc_path / "config.yaml").exists():
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
