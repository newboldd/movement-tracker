"""Analysis results and plot serving."""

from __future__ import annotations

from pathlib import Path
from fastapi import APIRouter, HTTPException

from ..config import get_settings
from ..db import get_db_ctx

router = APIRouter(prefix="/api/results", tags=["results"])


@router.get("/{subject_id}")
def get_results(subject_id: int) -> dict:
    """Get analysis results summary for a subject."""
    settings = get_settings()

    with get_db_ctx() as db:
        subj = db.execute(
            "SELECT * FROM subjects WHERE id = ?", (subject_id,)
        ).fetchone()
    if not subj:
        raise HTTPException(404, "Subject not found")

    subject_name = subj["name"]
    dlc_path = settings.dlc_path / subject_name

    # Find analysis outputs
    results = {
        "subject": subject_name,
        "stage": subj["stage"],
        "has_labels_v1": (dlc_path / "labels_v1").exists(),
        "has_labeled_videos": (dlc_path / "labeled_videos").exists(),
        "csv_files": [],
        "video_files": [],
    }

    # Find CSV outputs in labels_v1
    labels_dir = dlc_path / "labels_v1"
    if labels_dir.exists():
        results["csv_files"] = [
            f.name for f in sorted(labels_dir.glob("*.csv"))
        ]

    # Find labeled videos
    lv_dir = dlc_path / "labeled_videos"
    if lv_dir.exists():
        results["video_files"] = [
            f.name for f in sorted(lv_dir.glob("*.mp4"))
        ]

    # Check for DLC corrections
    corrections_dir = settings.data_path / "dlc_outputs" / "corrections"
    if corrections_dir.exists():
        corr_files = list(corrections_dir.glob(f"{subject_name}*.csv"))
        results["has_corrections"] = len(corr_files) > 0
        results["correction_files"] = [f.name for f in corr_files]

    return results
