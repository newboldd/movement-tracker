"""MANO 3D hand model viewer API: trial listing, data loading, heatmap serving."""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response

from ..db import get_db_ctx
from ..services.mano_data import (
    list_mano_trials,
    load_mano_trial_data,
    get_heatmap,
    HAND_SKELETON,
    FINGER_GROUPS,
    DISTANCE_OPTIONS,
    JOINT_NAMES,
)
from ..services.video import build_trial_map

router = APIRouter(prefix="/api/mano", tags=["mano"])


def _subject_name(subject_id: int) -> str:
    """Look up subject name by ID."""
    with get_db_ctx() as db:
        row = db.execute(
            "SELECT name FROM subjects WHERE id = ?", (subject_id,)
        ).fetchone()
    if not row:
        raise HTTPException(404, f"Subject {subject_id} not found")
    return row["name"]


@router.get("/{subject_id}/trials")
def get_trials(subject_id: int) -> list[dict]:
    """List trials with available MANO fit data for a subject."""
    name = _subject_name(subject_id)
    return list_mano_trials(name)


@router.get("/{subject_id}/trial/{trial_idx}/data")
def get_trial_data(subject_id: int, trial_idx: int) -> Response:
    """Load bulk MANO viewer data for a trial.

    Returns projected 2D coords, 3D joints, distances, fit quality,
    skeleton/finger constants.  ~5–10 MB JSON for a 1100-frame trial.
    """
    name = _subject_name(subject_id)
    trials = list_mano_trials(name)

    # Find the trial by index
    trial = None
    for t in trials:
        if t["trial_idx"] == trial_idx:
            trial = t
            break

    if trial is None:
        raise HTTPException(404, f"No MANO data for trial index {trial_idx}")

    try:
        data = load_mano_trial_data(name, trial["trial_stem"])
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

    # Serialise — replace NaN/Infinity with null for valid JSON
    import math, numpy as np

    def _default(o):
        """Handle remaining numpy scalars."""
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating, float)):
            v = float(o)
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    body = json.dumps(data, separators=(",", ":"), default=_default)
    # Final safety: replace any remaining NaN/Infinity literals
    body = body.replace("NaN", "null").replace("Infinity", "null")
    return Response(content=body, media_type="application/json")


@router.get("/{subject_id}/trial/{trial_idx}/heatmap")
def get_trial_heatmap(
    subject_id: int,
    trial_idx: int,
    frame: int = Query(..., ge=0),
    joint: int = Query(..., ge=0, le=20),
    side: str = Query("OS"),
):
    """Serve a single 64x64 heatmap for one joint at one frame.

    Returns: {heatmap: [[float]], bbox: [x1,y1,x2,y2], max_val: float}
    """
    name = _subject_name(subject_id)
    trials = list_mano_trials(name)

    trial = None
    for t in trials:
        if t["trial_idx"] == trial_idx:
            trial = t
            break

    if trial is None:
        raise HTTPException(404, f"No MANO data for trial index {trial_idx}")

    result = get_heatmap(name, trial["trial_stem"], frame, joint, side)
    if result is None:
        raise HTTPException(404, "Heatmap not available")
    return result


@router.get("/{subject_id}/trial/{trial_idx}/video")
def get_trial_video(subject_id: int, trial_idx: int):
    """Serve the video file for a trial."""
    name = _subject_name(subject_id)
    try:
        trials = build_trial_map(name)
    except Exception:
        raise HTTPException(404, "No videos found")

    if trial_idx < 0 or trial_idx >= len(trials):
        raise HTTPException(404, f"Trial index {trial_idx} out of range")

    video_path = trials[trial_idx]["video_path"]
    return FileResponse(video_path, media_type="video/mp4")


@router.get("/{subject_id}/video_list")
def get_video_list(subject_id: int) -> list[dict]:
    """List all video trials for a subject (for Videos module — no MANO data required)."""
    name = _subject_name(subject_id)
    try:
        trials = build_trial_map(name)
    except Exception:
        return []
    return [
        {
            "trial_idx": i,
            "trial_name": t["trial_name"],
            "n_frames": t["frame_count"],
            "fps": t["fps"],
            "width": t["width"],
            "height": t["height"],
            # Stereo = side-by-side halves; heuristic: width >= 1.5 × height
            "is_stereo": t["width"] >= int(t["height"] * 1.5),
        }
        for i, t in enumerate(trials)
    ]
