"""Analyze page API: keypoint viewer for MediaPipe, Apple Vision, and DLC data."""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import FileResponse, Response

from ..config import get_settings
from ..db import get_db_ctx
from ..services.video import build_trial_map

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/analyze", tags=["analyze"])

# ── Hand skeleton constants ────────────────────────────────────────────

HAND_SKELETON = [
    [0, 1, 2, 3, 4],       # thumb
    [0, 5, 6, 7, 8],       # index
    [0, 9, 10, 11, 12],    # middle
    [0, 13, 14, 15, 16],   # ring
    [0, 17, 18, 19, 20],   # pinky
]

FINGER_GROUPS = {
    "Wrist": [0],
    "Thumb": [1, 2, 3, 4],
    "Index": [5, 6, 7, 8],
    "Middle": [9, 10, 11, 12],
    "Ring": [13, 14, 15, 16],
    "Pinky": [17, 18, 19, 20],
}

DISTANCE_OPTIONS: dict[str, tuple[int, int]] = {
    "Thumb-Index Aperture": (4, 8),
    "Thumb-Middle Aperture": (4, 12),
    "Index-Middle Aperture": (8, 12),
    "Thumb: CMC-MCP (1-2)": (1, 2),
    "Thumb: MCP-IP (2-3)": (2, 3),
    "Thumb: IP-Tip (3-4)": (3, 4),
    "Index: MCP-PIP (5-6)": (5, 6),
    "Index: PIP-DIP (6-7)": (6, 7),
    "Index: DIP-Tip (7-8)": (7, 8),
    "Middle: MCP-PIP (9-10)": (9, 10),
    "Middle: PIP-DIP (10-11)": (10, 11),
    "Middle: DIP-Tip (11-12)": (11, 12),
    "Wrist-Thumb base (0-1)": (0, 1),
    "Wrist-Index base (0-5)": (0, 5),
    "Wrist-Middle base (0-9)": (0, 9),
}


# ── Helpers ────────────────────────────────────────────────────────────


def _subject_name(subject_id: int) -> str:
    """Look up subject name by ID."""
    with get_db_ctx() as db:
        row = db.execute(
            "SELECT name FROM subjects WHERE id = ?", (subject_id,)
        ).fetchone()
    if not row:
        raise HTTPException(404, f"Subject {subject_id} not found")
    return row["name"]


def _nan_safe(val):
    """Convert a float that may be NaN/Inf to None."""
    if val is None:
        return None
    v = float(val)
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _landmarks_to_list(arr: np.ndarray) -> list:
    """Convert (N, 21, 2) landmark array to JSON-safe nested list.

    Each frame becomes a list of 21 [x, y] pairs, or null if the point
    is all-zero (undetected).
    """
    n_frames = arr.shape[0]
    result = []
    for f in range(n_frames):
        frame_pts = []
        for j in range(arr.shape[1]):
            x, y = float(arr[f, j, 0]), float(arr[f, j, 1])
            if x == 0.0 and y == 0.0:
                frame_pts.append(None)
            elif math.isnan(x) or math.isnan(y):
                frame_pts.append(None)
            else:
                frame_pts.append([x, y])
        result.append(frame_pts)
    return result


def _load_mediapipe(subject_name: str) -> dict | None:
    """Load mediapipe_prelabels.npz for a subject.

    Returns dict with OS and OD landmark arrays (N, 21, 2) or None.
    """
    settings = get_settings()
    npz_path = settings.dlc_path / subject_name / "mediapipe_prelabels.npz"
    if not npz_path.exists():
        return None
    data = np.load(str(npz_path))
    result = {}
    if "OS_landmarks" in data:
        result["OS"] = data["OS_landmarks"]
    if "OD_landmarks" in data:
        result["OD"] = data["OD_landmarks"]
    return result if result else None


def _load_vision(subject_name: str) -> dict | None:
    """Load vision_prelabels.npz for a subject.

    Returns dict with OS and OD landmark arrays (N, 21, 2) or None.
    """
    settings = get_settings()
    npz_path = settings.dlc_path / subject_name / "vision_prelabels.npz"
    if not npz_path.exists():
        return None
    data = np.load(str(npz_path))
    result = {}
    if "OS_landmarks" in data:
        result["OS"] = data["OS_landmarks"]
    if "OD_landmarks" in data:
        result["OD"] = data["OD_landmarks"]
    return result if result else None


def _load_dlc(subject_name: str) -> dict | None:
    """Load DLC predictions for a subject.

    Returns dict: {OS: {thumb: [...], index: [...]}, OD: {...}} or None.
    """
    try:
        from ..services.dlc_predictions import get_dlc_predictions_for_session
        preds = get_dlc_predictions_for_session(subject_name)
    except Exception:
        return None
    return preds


def _compute_2d_distances(
    pts: np.ndarray, joint_a: int, joint_b: int
) -> list:
    """Compute per-frame 2D pixel distance between two joints.

    Args:
        pts: (N, 21, 2) landmark array.
        joint_a, joint_b: Joint indices.

    Returns:
        List of float|None per frame.
    """
    n_frames = pts.shape[0]
    result = []
    for f in range(n_frames):
        a = pts[f, joint_a]
        b = pts[f, joint_b]
        # Skip if either point is all-zero (undetected) or NaN
        if (a[0] == 0.0 and a[1] == 0.0) or (b[0] == 0.0 and b[1] == 0.0):
            result.append(None)
        elif np.isnan(a).any() or np.isnan(b).any():
            result.append(None)
        else:
            result.append(round(float(np.linalg.norm(a - b)), 2))
    return result


def _extract_trial_landmarks(full_arr: np.ndarray, start: int, end: int) -> np.ndarray:
    """Slice a full-session landmark array to a trial's frame range."""
    n = full_arr.shape[0]
    actual_end = min(end + 1, n)
    actual_start = min(start, n)
    if actual_start >= actual_end:
        return np.zeros((0, full_arr.shape[1], full_arr.shape[2]), dtype=full_arr.dtype)
    return full_arr[actual_start:actual_end]


def _extract_trial_dlc(dlc_data: dict, side: str, start: int, end: int) -> dict:
    """Extract trial-range DLC predictions for one camera side.

    DLC data is stored as {camera: {bodypart: [[x,y]|null, ...]}}.
    Returns {bodypart: [[x,y]|null, ...]} sliced to the trial range.
    """
    side_data = dlc_data.get(side, {})
    result = {}
    for bp, coords in side_data.items():
        if bp == "distances":
            continue
        if isinstance(coords, list):
            result[bp] = coords[start:end + 1]
    return result


# ── Endpoints ──────────────────────────────────────────────────────────


@router.get("/{subject_id}/trials")
def get_trials(subject_id: int) -> list[dict]:
    """List trials for a subject with availability flags for each data source."""
    name = _subject_name(subject_id)

    # Use per-subject camera_mode
    from ..db import get_db
    db = get_db()
    row = db.execute("SELECT camera_mode FROM subjects WHERE id = ?", (subject_id,)).fetchone()
    settings = get_settings()
    subject_camera_mode = (row["camera_mode"] if row and row["camera_mode"] else
                           settings.default_camera_mode)

    try:
        trials = build_trial_map(name, camera_mode=subject_camera_mode)
    except Exception:
        return []

    # Check data source availability
    mp_data = _load_mediapipe(name)
    vis_data = _load_vision(name)
    dlc_data = _load_dlc(name)

    result = []
    for i, t in enumerate(trials):
        has_mp = False
        has_vis = False
        has_dlc = False

        start = t["start_frame"]
        end = t["end_frame"]

        # Check mediapipe: any non-zero data in the trial range
        if mp_data:
            for side in ("OS", "OD"):
                arr = mp_data.get(side)
                if arr is not None and arr.shape[0] > start:
                    trial_slice = _extract_trial_landmarks(arr, start, end)
                    if trial_slice.size > 0 and np.any(trial_slice != 0):
                        has_mp = True
                        break

        # Check vision
        if vis_data:
            for side in ("OS", "OD"):
                arr = vis_data.get(side)
                if arr is not None and arr.shape[0] > start:
                    trial_slice = _extract_trial_landmarks(arr, start, end)
                    if trial_slice.size > 0 and np.any(trial_slice != 0):
                        has_vis = True
                        break

        # Check DLC
        if dlc_data:
            for side in ("OS", "OD"):
                side_preds = dlc_data.get(side, {})
                for bp, coords in side_preds.items():
                    if isinstance(coords, list):
                        trial_coords = coords[start:end + 1]
                        if any(c is not None for c in trial_coords):
                            has_dlc = True
                            break
                if has_dlc:
                    break

        result.append({
            "trial_idx": i,
            "trial_name": t["trial_name"],
            "fps": t["fps"],
            "n_frames": t["frame_count"],
            "start_frame": start,
            "end_frame": end,
            "has_mediapipe": has_mp,
            "has_vision": has_vis,
            "has_dlc": has_dlc,
        })

    return result


@router.get("/{subject_id}/trial/{trial_idx}/data")
def get_trial_data(subject_id: int, trial_idx: int) -> Response:
    """Load all keypoint data for one trial.

    Returns MediaPipe, Vision, and DLC landmarks plus computed 2D distances.
    """
    name = _subject_name(subject_id)

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
    start = trial["start_frame"]
    end = trial["end_frame"]
    n_frames = trial["frame_count"]
    fps = trial["fps"]

    # Load raw data sources
    mp_data = _load_mediapipe(name)
    vis_data = _load_vision(name)
    dlc_data = _load_dlc(name)

    # Build mediapipe response
    mediapipe_resp = {}
    mp_trial = {}  # {side: np.ndarray} for distance computation
    if mp_data:
        for side in ("OS", "OD"):
            arr = mp_data.get(side)
            if arr is not None:
                trial_arr = _extract_trial_landmarks(arr, start, end)
                # Pad to n_frames if needed
                if trial_arr.shape[0] < n_frames:
                    pad = np.zeros((n_frames - trial_arr.shape[0], 21, 2))
                    trial_arr = np.concatenate([trial_arr, pad], axis=0)
                mp_trial[side] = trial_arr
                mediapipe_resp[side] = {"landmarks": _landmarks_to_list(trial_arr)}

    # Build vision response
    vision_resp = {}
    vis_trial = {}
    if vis_data:
        for side in ("OS", "OD"):
            arr = vis_data.get(side)
            if arr is not None:
                trial_arr = _extract_trial_landmarks(arr, start, end)
                if trial_arr.shape[0] < n_frames:
                    pad = np.zeros((n_frames - trial_arr.shape[0], 21, 2))
                    trial_arr = np.concatenate([trial_arr, pad], axis=0)
                vis_trial[side] = trial_arr
                vision_resp[side] = {"landmarks": _landmarks_to_list(trial_arr)}

    # Build DLC response
    dlc_resp = {}
    if dlc_data:
        for side in ("OS", "OD"):
            trial_preds = _extract_trial_dlc(dlc_data, side, start, end)
            if trial_preds:
                dlc_resp[side] = trial_preds

    # Compute 2D pixel distances for each source and distance option
    distance_options = [
        {"name": name_key, "joint_a": ja, "joint_b": jb}
        for name_key, (ja, jb) in DISTANCE_OPTIONS.items()
    ]

    distances = {}
    for name_key, (ja, jb) in DISTANCE_OPTIONS.items():
        entry = {}

        # MediaPipe distances (average OS + OD if both available)
        for side in ("OS", "OD"):
            if side in mp_trial:
                key = f"mediapipe_{side}"
                entry[key] = _compute_2d_distances(mp_trial[side], ja, jb)

        for side in ("OS", "OD"):
            if side in vis_trial:
                key = f"vision_{side}"
                entry[key] = _compute_2d_distances(vis_trial[side], ja, jb)

        # DLC distances — DLC only has thumb (joint 4) and index (joint 8)
        # Only compute if the distance involves joints that DLC tracks
        if dlc_data:
            dlc_joint_map = {"thumb": 4, "index": 8}
            bp_a = None
            bp_b = None
            for bp_name, joint_idx in dlc_joint_map.items():
                if joint_idx == ja:
                    bp_a = bp_name
                if joint_idx == jb:
                    bp_b = bp_name

            if bp_a is not None and bp_b is not None:
                for side in ("OS", "OD"):
                    trial_preds = _extract_trial_dlc(dlc_data, side, start, end)
                    coords_a = trial_preds.get(bp_a, [])
                    coords_b = trial_preds.get(bp_b, [])
                    if coords_a and coords_b:
                        dlc_dists = []
                        for f in range(min(len(coords_a), len(coords_b))):
                            if coords_a[f] is not None and coords_b[f] is not None:
                                a = np.array(coords_a[f])
                                b = np.array(coords_b[f])
                                dlc_dists.append(round(float(np.linalg.norm(a - b)), 2))
                            else:
                                dlc_dists.append(None)
                        entry[f"dlc_{side}"] = dlc_dists

        if entry:
            distances[name_key] = entry

    data = {
        "n_frames": n_frames,
        "fps": fps,
        "trial_name": trial["trial_name"],
        "mediapipe": mediapipe_resp if mediapipe_resp else None,
        "vision": vision_resp if vision_resp else None,
        "dlc": dlc_resp if dlc_resp else None,
        "distances": distances,
        "distance_options": distance_options,
        "skeleton": HAND_SKELETON,
        "finger_groups": FINGER_GROUPS,
    }

    # Serialise with NaN-safe handling
    def _default(o):
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
    body = body.replace("NaN", "null").replace("Infinity", "null")
    return Response(content=body, media_type="application/json")


@router.get("/{subject_id}/trial/{trial_idx}/video")
def get_trial_video(subject_id: int, trial_idx: int,
                    camera: int | None = Query(None, ge=0)):
    """Serve the video file for a trial, preferring de-identified if configured."""
    name = _subject_name(subject_id)

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
        from ..services.video import _get_no_face_videos
        stem = Path(video_path).stem
        no_face = _get_no_face_videos(name)
        if stem not in no_face:
            deident = Path(video_path).parent / "deidentified" / Path(video_path).name
            if deident.exists():
                video_path = str(deident)

    return FileResponse(video_path, media_type="video/mp4")


# ── Run detection models ─────────────────────────────────────────────────

@router.post("/{subject_id}/run-mediapipe")
def run_mediapipe(subject_id: int) -> dict:
    """Run MediaPipe hand detection on all trials. Creates a background job."""
    import threading
    from ..services.jobs import registry
    from ..services.mediapipe_prelabel import run_mediapipe as run_mediapipe_all

    subj_name = _subject_name(subject_id)
    settings = get_settings()

    with get_db_ctx() as db:
        log_dir = settings.dlc_path / ".logs"
        log_dir.mkdir(exist_ok=True)
        log_path = str(log_dir / f"job_mediapipe_{subject_id}.log")
        db.execute(
            "INSERT INTO jobs (subject_id, job_type, status, log_path) VALUES (?, 'mediapipe', 'pending', ?)",
            (subject_id, log_path),
        )
        job = db.execute(
            "SELECT id FROM jobs WHERE subject_id = ? ORDER BY id DESC LIMIT 1", (subject_id,)
        ).fetchone()

    job_id = job["id"]
    cancel = registry.register_cancel_event(job_id)

    def _run():
        try:
            with get_db_ctx() as db:
                db.execute("UPDATE jobs SET status = 'running', started_at = CURRENT_TIMESTAMP WHERE id = ?", (job_id,))
            def progress(pct):
                if cancel.is_set(): raise InterruptedError
                with get_db_ctx() as db:
                    db.execute("UPDATE jobs SET progress_pct = ? WHERE id = ?", (pct, job_id))
            run_mediapipe_all(subj_name, progress_callback=progress)
            with get_db_ctx() as db:
                db.execute("UPDATE jobs SET status = 'completed', progress_pct = 100, finished_at = CURRENT_TIMESTAMP WHERE id = ?", (job_id,))
        except InterruptedError:
            with get_db_ctx() as db:
                db.execute("UPDATE jobs SET status = 'cancelled', finished_at = CURRENT_TIMESTAMP WHERE id = ?", (job_id,))
        except Exception as e:
            with get_db_ctx() as db:
                db.execute("UPDATE jobs SET status = 'failed', error_msg = ?, finished_at = CURRENT_TIMESTAMP WHERE id = ?", (str(e), job_id))
        finally:
            registry.unregister_cancel_event(job_id)

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id, "status": "running"}


@router.post("/{subject_id}/run-vision")
def run_vision(subject_id: int) -> dict:
    """Run Apple Vision hand detection on all trials. Creates a background job."""
    import threading
    from ..services.jobs import registry

    subj_name = _subject_name(subject_id)
    settings = get_settings()

    with get_db_ctx() as db:
        log_dir = settings.dlc_path / ".logs"
        log_dir.mkdir(exist_ok=True)
        log_path = str(log_dir / f"job_vision_{subject_id}.log")
        db.execute(
            "INSERT INTO jobs (subject_id, job_type, status, log_path) VALUES (?, 'vision', 'pending', ?)",
            (subject_id, log_path),
        )
        job = db.execute(
            "SELECT id FROM jobs WHERE subject_id = ? ORDER BY id DESC LIMIT 1", (subject_id,)
        ).fetchone()

    job_id = job["id"]
    cancel = registry.register_cancel_event(job_id)

    def _run():
        try:
            from ..services.vision_prelabel import run_vision_hands
            with get_db_ctx() as db:
                db.execute("UPDATE jobs SET status = 'running', started_at = CURRENT_TIMESTAMP WHERE id = ?", (job_id,))
            def progress(pct):
                if cancel.is_set(): raise InterruptedError
                with get_db_ctx() as db:
                    db.execute("UPDATE jobs SET progress_pct = ? WHERE id = ?", (pct, job_id))
            run_vision_hands(subj_name, progress_callback=progress)
            with get_db_ctx() as db:
                db.execute("UPDATE jobs SET status = 'completed', progress_pct = 100, finished_at = CURRENT_TIMESTAMP WHERE id = ?", (job_id,))
        except InterruptedError:
            with get_db_ctx() as db:
                db.execute("UPDATE jobs SET status = 'cancelled', finished_at = CURRENT_TIMESTAMP WHERE id = ?", (job_id,))
        except Exception as e:
            with get_db_ctx() as db:
                db.execute("UPDATE jobs SET status = 'failed', error_msg = ?, finished_at = CURRENT_TIMESTAMP WHERE id = ?", (str(e), job_id))
        finally:
            registry.unregister_cancel_event(job_id)

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id, "status": "running"}


@router.post("/{subject_id}/run-pose")
def run_pose(subject_id: int) -> dict:
    """Run MediaPipe pose detection on all trials. Creates a background job."""
    import threading
    from ..services.jobs import registry
    from ..services.mediapipe_prelabel import run_pose_prelabels

    subj_name = _subject_name(subject_id)
    settings = get_settings()

    with get_db_ctx() as db:
        log_dir = settings.dlc_path / ".logs"
        log_dir.mkdir(exist_ok=True)
        log_path = str(log_dir / f"job_pose_{subject_id}.log")
        db.execute(
            "INSERT INTO jobs (subject_id, job_type, status, log_path) VALUES (?, 'pose', 'pending', ?)",
            (subject_id, log_path),
        )
        job = db.execute(
            "SELECT id FROM jobs WHERE subject_id = ? ORDER BY id DESC LIMIT 1", (subject_id,)
        ).fetchone()

    job_id = job["id"]
    cancel = registry.register_cancel_event(job_id)

    def _run():
        try:
            with get_db_ctx() as db:
                db.execute("UPDATE jobs SET status = 'running', started_at = CURRENT_TIMESTAMP WHERE id = ?", (job_id,))
            def progress(pct):
                if cancel.is_set(): raise InterruptedError
                with get_db_ctx() as db:
                    db.execute("UPDATE jobs SET progress_pct = ? WHERE id = ?", (pct, job_id))
            run_pose_prelabels(subj_name, progress_callback=progress)
            with get_db_ctx() as db:
                db.execute("UPDATE jobs SET status = 'completed', progress_pct = 100, finished_at = CURRENT_TIMESTAMP WHERE id = ?", (job_id,))
        except InterruptedError:
            with get_db_ctx() as db:
                db.execute("UPDATE jobs SET status = 'cancelled', finished_at = CURRENT_TIMESTAMP WHERE id = ?", (job_id,))
        except Exception as e:
            with get_db_ctx() as db:
                db.execute("UPDATE jobs SET status = 'failed', error_msg = ?, finished_at = CURRENT_TIMESTAMP WHERE id = ?", (str(e), job_id))
        finally:
            registry.unregister_cancel_event(job_id)

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id, "status": "running"}


# ── HRNet ─────────────────────────────────────────────────────────────────

@router.get("/{subject_id}/hrnet/status")
def hrnet_status(subject_id: int) -> dict:
    """Check if HRNet dependencies are available."""
    from ..services.hrnet import check_hrnet_available
    return check_hrnet_available()


@router.get("/{subject_id}/hrnet/bbox")
def hrnet_default_bbox(subject_id: int, trial_idx: int = Query(..., ge=0)) -> dict:
    """Compute default bounding boxes from MediaPipe landmarks."""
    from ..services.hrnet import compute_default_bbox

    subj_name = _subject_name(subject_id)
    mp_data = _load_mediapipe(subj_name)
    if mp_data is None:
        raise HTTPException(400, "Run MediaPipe first — no landmarks available")

    trials = _trials(subj_name)
    if trial_idx >= len(trials):
        raise HTTPException(404, f"Trial index {trial_idx} out of range")
    trial = trials[trial_idx]
    start = trial.get("start_frame", 0)
    end = start + trial["frame_count"]

    os_lm = mp_data.get("OS")
    od_lm = mp_data.get("OD")

    bbox_os = compute_default_bbox(os_lm[start:min(end, os_lm.shape[0])]) if os_lm is not None else None
    bbox_od = compute_default_bbox(od_lm[start:min(end, od_lm.shape[0])]) if od_lm is not None else None

    return {"bbox_os": bbox_os, "bbox_od": bbox_od}


@router.post("/{subject_id}/run-hrnet")
def run_hrnet_endpoint(subject_id: int, body: dict = Body(...)) -> dict:
    """Run HRNet heatmap inference on a trial. Creates a background job.

    Body: {trial_idx: int, bbox_os: [x1,y1,x2,y2], bbox_od: [x1,y1,x2,y2]}
    """
    import threading
    from ..services.jobs import registry
    from ..services.hrnet import check_hrnet_available, run_hrnet_trial

    status = check_hrnet_available()
    if not status["available"]:
        raise HTTPException(400, status["message"])

    subj_name = _subject_name(subject_id)
    trial_idx = body.get("trial_idx", 0)
    bbox_os = body.get("bbox_os")
    bbox_od = body.get("bbox_od")

    settings = get_settings()

    with get_db_ctx() as db:
        log_dir = settings.dlc_path / ".logs"
        log_dir.mkdir(exist_ok=True)
        log_path = str(log_dir / f"job_hrnet_{subject_id}.log")
        db.execute(
            "INSERT INTO jobs (subject_id, job_type, status, log_path) VALUES (?, 'hrnet', 'pending', ?)",
            (subject_id, log_path),
        )
        job = db.execute(
            "SELECT id FROM jobs WHERE subject_id = ? ORDER BY id DESC LIMIT 1", (subject_id,)
        ).fetchone()

    job_id = job["id"]
    cancel = registry.register_cancel_event(job_id)

    def _run():
        try:
            with get_db_ctx() as db:
                db.execute("UPDATE jobs SET status = 'running', started_at = CURRENT_TIMESTAMP WHERE id = ?", (job_id,))

            def progress(pct):
                if cancel.is_set():
                    raise InterruptedError
                with get_db_ctx() as db:
                    db.execute("UPDATE jobs SET progress_pct = ? WHERE id = ?", (pct, job_id))

            run_hrnet_trial(
                subj_name, trial_idx,
                bbox_os=bbox_os, bbox_od=bbox_od,
                cancel_event=cancel, progress_callback=progress,
            )
            with get_db_ctx() as db:
                db.execute(
                    "UPDATE jobs SET status = 'completed', progress_pct = 100, "
                    "finished_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (job_id,),
                )
        except InterruptedError:
            with get_db_ctx() as db:
                db.execute("UPDATE jobs SET status = 'cancelled', finished_at = CURRENT_TIMESTAMP WHERE id = ?", (job_id,))
        except Exception as e:
            import logging as _log
            _log.getLogger(__name__).exception(f"HRNet job {job_id} failed")
            with get_db_ctx() as db:
                db.execute(
                    "UPDATE jobs SET status = 'failed', error_msg = ?, "
                    "finished_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (str(e), job_id),
                )
        finally:
            registry.unregister_cancel_event(job_id)

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id, "status": "running"}
