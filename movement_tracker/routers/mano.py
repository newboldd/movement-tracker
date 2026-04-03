"""MANO 3D hand model viewer API: trial listing, data loading, heatmap serving."""
from __future__ import annotations

import json

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response

from ..config import get_settings
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
from ..services.mediapipe_prelabel import load_mediapipe_prelabels

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
def get_trial_video(subject_id: int, trial_idx: int,
                    camera: int | None = Query(None, ge=0)):
    """Serve the video file for a trial, preferring de-identified if configured.

    For multicam trials, pass ``camera=N`` to select a specific camera index.
    Defaults to the primary (first) camera.
    """
    name = _subject_name(subject_id)

    # Use per-subject camera_mode so trial grouping matches video_list
    from ..db import get_db
    db = get_db()
    row = db.execute("SELECT camera_mode FROM subjects WHERE id = ?", (subject_id,)).fetchone()
    settings = get_settings()
    subject_camera_mode = (row["camera_mode"] if row and row["camera_mode"] else
                           settings.default_camera_mode)

    try:
        trials = build_trial_map(name, camera_mode=subject_camera_mode)
    except Exception:
        raise HTTPException(404, "No videos found")

    if trial_idx < 0 or trial_idx >= len(trials):
        raise HTTPException(404, f"Trial index {trial_idx} out of range")

    trial = trials[trial_idx]
    video_path = trial["video_path"]

    # In multicam mode, select the requested camera file
    cameras = trial.get("cameras", [])
    if camera is not None and cameras:
        cam = next((c for c in cameras if c["idx"] == camera), None)
        if cam:
            video_path = cam["path"]
        else:
            raise HTTPException(404, f"Camera index {camera} not found for trial {trial_idx}")

    # Prefer de-identified video if the setting is enabled
    if settings.prefer_deidentified:
        from pathlib import Path
        from ..services.video import _get_no_face_videos
        stem = Path(video_path).stem
        no_face = _get_no_face_videos(name)
        if stem not in no_face:
            deident = Path(video_path).parent / "deidentified" / Path(video_path).name
            if deident.exists():
                video_path = str(deident)

    return FileResponse(video_path, media_type="video/mp4")


@router.get("/{subject_id}/mediapipe_hints")
def get_mediapipe_hints(subject_id: int) -> list[dict]:
    """Return per-trial MediaPipe crop hints: best camera and hand bounding boxes.

    Each item: {trial_idx, best_camera (0|1), bbox_OS, bbox_OD,
                detect_rate_OS, detect_rate_OD}
    bbox_* are [minX, minY, maxX, maxY] in half-frame pixel coordinates,
    padded by 15%, or null if no detections.
    """
    name = _subject_name(subject_id)
    try:
        trials = build_trial_map(name)
    except Exception:
        return []

    hints = []
    for i, t in enumerate(trials):
        try:
            prelabels = load_mediapipe_prelabels(name, t["trial_stem"])
        except Exception:
            continue

        result: dict = {"trial_idx": i, "best_camera": 0,
                        "bbox_OS": None, "bbox_OD": None,
                        "detect_rate_OS": 0.0, "detect_rate_OD": 0.0}

        for side_key, bbox_key, rate_key in [
            ("OS_landmarks", "bbox_OS", "detect_rate_OS"),
            ("OD_landmarks", "bbox_OD", "detect_rate_OD"),
        ]:
            lm = prelabels.get(side_key)  # shape (N, 21, 2) or None
            if lm is None or not hasattr(lm, "shape"):
                continue
            # Detect rate = fraction of frames with at least one non-zero landmark
            valid = np.any(lm != 0, axis=(1, 2))
            rate = float(valid.mean())
            result[rate_key] = rate

            if valid.any():
                pts = lm[valid]  # (M, 21, 2)
                xs, ys = pts[:, :, 0].ravel(), pts[:, :, 1].ravel()
                pad_x = (xs.max() - xs.min()) * 0.15
                pad_y = (ys.max() - ys.min()) * 0.15
                result[bbox_key] = [
                    float(max(0, xs.min() - pad_x)),
                    float(max(0, ys.min() - pad_y)),
                    float(xs.max() + pad_x),
                    float(ys.max() + pad_y),
                ]

        # Best camera = whichever side has higher detect rate
        result["best_camera"] = 0 if result["detect_rate_OS"] >= result["detect_rate_OD"] else 1
        hints.append(result)

    return hints


@router.get("/{subject_id}/video_list")
def get_video_list(subject_id: int) -> list[dict]:
    """List all video trials for a subject (for Videos module — no MANO data required).

    Each trial includes a ``cameras`` list for multicam trials:
    ``[{name, path, idx}]``.  Empty when trial is a single file.
    """
    settings = get_settings()
    name = _subject_name(subject_id)

    # Use per-subject camera_mode if set, otherwise fall back to global default
    from ..db import get_db
    db = get_db()
    row = db.execute("SELECT camera_mode FROM subjects WHERE id = ?", (subject_id,)).fetchone()
    subject_camera_mode = (row["camera_mode"] if row and row["camera_mode"] else
                           settings.default_camera_mode)

    try:
        trials = build_trial_map(name, camera_mode=subject_camera_mode)
    except Exception:
        return []

    result = []
    for i, t in enumerate(trials):
        cameras = t.get("cameras", [])
        entry = {
            "trial_idx": i,
            "trial_name": t["trial_name"],
            "n_frames": t["frame_count"],
            "fps": t["fps"],
            "width": t["width"],
            "height": t["height"],
            # Stereo determined by per-subject camera_mode
            "is_stereo": subject_camera_mode == "stereo",
        }
        # Include camera info for multicam trials (>1 camera file)
        if len(cameras) > 1:
            entry["cameras"] = [
                {"name": c["name"], "idx": c["idx"]}
                for c in cameras
            ]
        result.append(entry)
    return result
