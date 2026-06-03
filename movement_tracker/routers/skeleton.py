"""Skeleton 3D hand model viewer API: trial listing, data loading, heatmap serving, fitting."""
from __future__ import annotations

import json
import threading

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from ..config import get_settings
from ..db import get_db_ctx
from ..services.skeleton_data import (
    list_skeleton_trials,
    load_skeleton_trial_data,
    load_angle_priors,
    get_heatmap,
    HAND_SKELETON,
    FINGER_GROUPS,
    DISTANCE_OPTIONS,
    JOINT_NAMES,
)
from ..services.video import build_trial_map
from ..services.mediapipe_prelabel import load_mediapipe_prelabels

from ..services.mp_error_detection import _CorrectionsCancelled

router = APIRouter(prefix="/api/skeleton", tags=["skeleton"])


class _JobCancelled(Exception):
    """Sentinel raised by progress callbacks when a thread-based job's
    cancel event is set, so the worker's outer ``except`` block can mark
    the job ``cancelled`` rather than ``failed``."""


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
    """List trials with MediaPipe data and/or Skeleton fits for a subject."""
    name = _subject_name(subject_id)
    return list_skeleton_trials(name)


@router.get("/{subject_id}/events")
def get_subject_events(subject_id: int) -> dict:
    """Return saved tapping events for the subject's events.csv.

    Same shape as the session-scoped /api/labeling/sessions/{id}/events
    endpoint — ``{open: [frames], peak: [...], close: [...], pause: [...]}``
    — but keyed by subject so the Labels page can plot them without
    needing an events session.  Returns empty arrays when the file is
    missing.
    """
    from .labeling import _read_events_csv
    name = _subject_name(subject_id)
    return _read_events_csv(name)


@router.get("/{subject_id}/trial/{trial_idx}/data")
def get_trial_data(subject_id: int, trial_idx: int) -> Response:
    """Load bulk Skeleton viewer data for a trial.

    Returns projected 2D coords, 3D joints, distances, fit quality,
    skeleton/finger constants.  ~5–10 MB JSON for a 1100-frame trial.
    """
    name = _subject_name(subject_id)
    trials = list_skeleton_trials(name)

    # Find the trial by index
    trial = None
    for t in trials:
        if t["trial_idx"] == trial_idx:
            trial = t
            break

    if trial is None:
        raise HTTPException(404, f"No Skeleton data for trial index {trial_idx}")

    try:
        data = load_skeleton_trial_data(name, trial["trial_stem"])
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
    joint: int = Query(..., ge=-1, le=20),
    side: str = Query("OS"),
):
    """Serve a single 64x64 heatmap for one joint at one frame.

    Pass ``joint=-1`` to fetch the pre-computed MIP (max over 21 joints),
    which saves the viewer from fetching all 21 per-joint heatmaps.

    Returns: {heatmap: [[float]], bbox: [x1,y1,x2,y2], max_val: float}
    """
    name = _subject_name(subject_id)
    trials = list_skeleton_trials(name)

    trial = None
    for t in trials:
        if t["trial_idx"] == trial_idx:
            trial = t
            break

    if trial is None:
        raise HTTPException(404, f"No Skeleton data for trial index {trial_idx}")

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
            if deident.exists() and deident.stat().st_size > 4096:
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
            prelabels = load_mediapipe_prelabels(name)
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


# ── Historical skeleton fit loading ───────────────────────────────────

@router.get("/{subject_id}/trial/{trial_idx}/fit_history/{slot}")
def get_fit_history_slot(subject_id: int, trial_idx: int, slot: int) -> Response:
    """Load a previous skeleton v2 fit (slot 1-3) with projections and angles."""
    from ..services.skeleton_data import load_v2_fit_history_slot, _load_trial_calibration
    name = _subject_name(subject_id)
    trials = list_skeleton_trials(name)
    trial = None
    for t in trials:
        if t["trial_idx"] == trial_idx:
            trial = t
            break
    if trial is None:
        raise HTTPException(404, "Trial not found")

    calib = _load_trial_calibration(name, trial["trial_stem"])
    data = load_v2_fit_history_slot(name, trial["trial_stem"], slot, calib)
    if data is None:
        raise HTTPException(404, f"No previous fit at slot {slot}")

    import math, numpy as np
    def _default(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating, float)):
            v = float(o)
            return None if math.isnan(v) or math.isinf(v) else v
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")
    body = json.dumps(data, separators=(",", ":"), default=_default)
    body = body.replace("NaN", "null").replace("Infinity", "null")
    return Response(content=body, media_type="application/json")


# ── Joint angle constraint editing ─────────────────────────────────────

@router.get("/joint-constraints")
def get_joint_constraints() -> dict:
    """Return current joint angle constraints (custom if present, else defaults)."""
    from ..config import DATA_DIR
    custom_path = DATA_DIR / "custom_joint_angle_priors.json"
    is_custom = custom_path.exists()
    return {"constraints": load_angle_priors(), "is_custom": is_custom}


@router.put("/joint-constraints")
def put_joint_constraints(body: dict) -> dict:
    """Save custom joint angle constraints."""
    from ..config import DATA_DIR
    custom_path = DATA_DIR / "custom_joint_angle_priors.json"
    custom_path.write_text(json.dumps(body, indent=2))
    return {"ok": True, "is_custom": True}


@router.delete("/joint-constraints")
def delete_joint_constraints() -> dict:
    """Reset to default constraints by removing the custom file."""
    from ..config import DATA_DIR
    custom_path = DATA_DIR / "custom_joint_angle_priors.json"
    if custom_path.exists():
        custom_path.unlink()
    return {"ok": True, "is_custom": False}


class FitRequest(BaseModel):
    trial_idx: int
    stage: int = 1
    w_reproj: float = 1.0
    w_bone: float = 5.0
    w_smooth: float = 1.0
    snap_bones: bool = False
    # Joint-angle regularization is no longer exposed in the UI;
    # default to 0 so the optimizer skips the term unless an old
    # caller explicitly passes a value.
    w_angle: float = 0.0


@router.get("/{subject_id}/fit/status")
def get_fit_status(subject_id: int) -> dict:
    """Check if Skeleton fitting dependencies are available."""
    from ..services.skeleton_v1 import check_skeleton_available
    return check_skeleton_available()


@router.get("/{subject_id}/trial/{trial_idx}/stereo_disparity")
def trial_stereo_disparity(subject_id: int, trial_idx: int) -> dict:
    """Diagnostic: per-joint L vs R epipolar residual at MP-combined
    positions for the requested trial.

    For each frame and joint we triangulate the MP-combined (uL, vL)
    / (uR, vR) pair to a 3D point using the trial's stereo
    calibration, then reproject through cv2.projectPoints (full
    distortion model) back to both cameras.  The residual
    ``proj_L − tgt_L`` and ``proj_R − tgt_R`` is the per-camera
    component of the stereo-input inconsistency — the same thing the
    Skel Fit v1 L2 optimizer has to split between cameras.

    The output groups summary stats per joint (and overall) so the
    user can tell whether the residual is a couple of pixels
    (genuine MP noise, live with it) or much larger (calibration
    worth re-doing).
    """
    import numpy as np
    import cv2
    from ..services.video import build_trial_map
    from ..services.calibration import (
        get_calibration_for_subject, triangulate_points,
    )
    from ..services.skeleton_data import _load_trial_calibration
    from ..services.mediapipe_prelabel import (
        load_mediapipe_combined_prelabels, load_mediapipe_prelabels,
    )

    name = _subject_name(subject_id)
    tmap = build_trial_map(name)
    if trial_idx < 0 or trial_idx >= len(tmap):
        raise HTTPException(404, f"Trial index {trial_idx} out of range")
    trial = tmap[trial_idx]
    trial_stem = trial["trial_name"]
    n_frames = trial["frame_count"]
    start = trial.get("start_frame", 0)

    prelabels = (load_mediapipe_combined_prelabels(name)
                 or load_mediapipe_prelabels(name))
    if prelabels is None:
        raise HTTPException(404, "No MediaPipe prelabels for subject")
    os_lm = prelabels["OS_landmarks"]
    od_lm = prelabels["OD_landmarks"]
    end = min(start + n_frames, os_lm.shape[0])
    mp_L = os_lm[start:end]
    mp_R = od_lm[start:end]

    calib = _load_trial_calibration(name, trial_stem)
    if calib is None:
        calib = get_calibration_for_subject(name)
    if calib is None:
        raise HTTPException(404, "No stereo calibration for subject")

    K1 = calib["K1"]; K2 = calib["K2"]
    d1 = calib["dist1"]; d2 = calib["dist2"]
    R = calib["R"]; T = calib["T"].reshape(3, 1)
    rvec_R, _ = cv2.Rodrigues(R)
    tvec_R = T

    # Triangulate per joint across all frames, then reproject.
    N = mp_L.shape[0]
    joints_3d = np.full((N, 21, 3), np.nan)
    for j in range(21):
        joints_3d[:, j, :] = triangulate_points(mp_L[:, j, :], mp_R[:, j, :], calib)

    # Reproject (distortion-aware) per camera, per (frame, joint).
    proj_L = np.full((N, 21, 2), np.nan)
    proj_R = np.full((N, 21, 2), np.nan)
    for t in range(N):
        valid = ~np.isnan(joints_3d[t, :, 0])
        if not valid.any():
            continue
        pts = joints_3d[t, valid].reshape(-1, 1, 3).astype(np.float64)
        pL_2d, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), K1, d1)
        pR_2d, _ = cv2.projectPoints(pts, rvec_R, tvec_R, K2, d2)
        proj_L[t, valid] = pL_2d.reshape(-1, 2)
        proj_R[t, valid] = pR_2d.reshape(-1, 2)

    res_L = proj_L - mp_L   # (N, 21, 2): projected − target on left
    res_R = proj_R - mp_R   # same for right

    def _stats(a):
        # Median + |median| + max-|.|; ignores NaN.
        flat = a[~np.isnan(a)]
        if flat.size == 0:
            return {"median": None, "abs_median": None, "max_abs": None}
        return {
            "median": float(np.nanmedian(a)),
            "abs_median": float(np.nanmedian(np.abs(flat))),
            "max_abs": float(np.nanmax(np.abs(flat))),
        }

    per_joint: list[dict] = []
    for j in range(21):
        per_joint.append({
            "joint": j,
            "n_frames": int((~np.isnan(joints_3d[:, j, 0])).sum()),
            "L_dx": _stats(res_L[:, j, 0]),
            "L_dy": _stats(res_L[:, j, 1]),
            "R_dx": _stats(res_R[:, j, 0]),
            "R_dy": _stats(res_R[:, j, 1]),
        })
    overall = {
        "L_dx": _stats(res_L[..., 0]),
        "L_dy": _stats(res_L[..., 1]),
        "R_dx": _stats(res_R[..., 0]),
        "R_dy": _stats(res_R[..., 1]),
    }
    return {
        "subject": name,
        "trial_idx": trial_idx,
        "trial_name": trial_stem,
        "n_frames": int(N),
        "overall": overall,
        "per_joint": per_joint,
        "interpretation": (
            "L_dy / R_dy are the per-camera vertical residuals between "
            "the triangulated 3D point reprojected back vs the MP-combined "
            "input.  Signed median ≈ 0 means the residual is symmetric noise. "
            "|median| / max-|.| of a few px = OK (genuine MP per-camera noise). "
            "Anything larger (≥ ~5 px median) points at calibration error."
        ),
    }


@router.get("/{subject_id}/trial_skeleton_status")
def trial_skeleton_status(subject_id: int) -> dict:
    """Per-trial skeleton_v1.npz existence for the given subject.

    Powers the per-trial color coding on the Jobs page Skel Fit v1
    selector: green when the npz already exists, neutral when not.
    """
    from ..services.skeleton_data import _skeleton_dir
    name = _subject_name(subject_id)
    try:
        tmap = build_trial_map(name)
    except Exception:
        return {"trials": []}
    root = _skeleton_dir(name)
    out: list[dict] = []
    for i, t in enumerate(tmap):
        tn = t.get("trial_name", "")
        npz = root / tn / "skeleton_v1.npz" if root else None
        has = bool(npz and npz.exists())
        out.append({
            "trial_idx": i,
            "trial_name": tn,
            "has_skeleton_v1": has,
        })
    return {"trials": out}


@router.post("/{subject_id}/fit")
def run_fit(subject_id: int, req: FitRequest) -> dict:
    """Submit a Skeleton fitting job as a background task."""
    from ..services.skeleton_v1 import check_skeleton_available, run_skeleton_v1_fit
    from ..services.jobs import registry

    status = check_skeleton_available()
    if not status["available"]:
        raise HTTPException(400, status["message"])

    name = _subject_name(subject_id)

    # Find trial stem
    try:
        video_trials = build_trial_map(name)
    except Exception:
        raise HTTPException(404, "No videos found")

    if req.trial_idx < 0 or req.trial_idx >= len(video_trials):
        raise HTTPException(404, f"Trial index {req.trial_idx} out of range")

    trial_stem = video_trials[req.trial_idx]["trial_name"]

    # Create job record
    trial_short = trial_stem.split('_', 1)[1] if '_' in trial_stem else trial_stem
    with get_db_ctx() as db:
        db.execute(
            "INSERT INTO jobs (subject_id, job_type, status, params_json) VALUES (?, 'skeleton_v1', 'pending', ?)",
            (subject_id, json.dumps({"trial_name": trial_short})),
        )
        job = db.execute(
            "SELECT * FROM jobs WHERE subject_id = ? ORDER BY id DESC LIMIT 1",
            (subject_id,),
        ).fetchone()

    job_id = job["id"]
    cancel_event = threading.Event()
    registry._cancel_events[job_id] = cancel_event

    def _run():
        try:
            with get_db_ctx() as db:
                db.execute(
                    "UPDATE jobs SET status = 'running', started_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (job_id,),
                )

            def on_progress(pct):
                with get_db_ctx() as db:
                    db.execute(
                        "UPDATE jobs SET progress_pct = ? WHERE id = ?",
                        (pct, job_id),
                    )

            result = run_skeleton_v1_fit(
                name, trial_stem,
                cancel_event=cancel_event,
                progress_callback=on_progress,
                w_reproj=req.w_reproj,
                w_bone=req.w_bone,
                w_smooth=req.w_smooth,
                snap_bones=req.snap_bones,
                w_angle=req.w_angle,
            )

            if result.get("cancelled"):
                with get_db_ctx() as db:
                    db.execute(
                        "UPDATE jobs SET status = 'cancelled', finished_at = CURRENT_TIMESTAMP WHERE id = ?",
                        (job_id,),
                    )
            else:
                with get_db_ctx() as db:
                    db.execute(
                        "UPDATE jobs SET status = 'completed', progress_pct = 100, "
                        "finished_at = CURRENT_TIMESTAMP WHERE id = ?",
                        (job_id,),
                    )
        except Exception as e:
            import logging
            logging.getLogger(__name__).exception(f"Skeleton v1 fit job {job_id} failed")
            with get_db_ctx() as db:
                db.execute(
                    "UPDATE jobs SET status = 'failed', error_msg = ?, "
                    "finished_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (str(e), job_id),
                )
        finally:
            registry._cancel_events.pop(job_id, None)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return {"job_id": job_id, "trial_stem": trial_stem}


class StereoAlignRequest(BaseModel):
    trial_idx: int
    # Legacy flag (still accepted): True == mode="outline".
    use_outline: bool = False
    # "image" | "outline" | "hybrid".  If unset, falls back to
    # use_outline.
    mode: str | None = None
    # Hybrid only: dilate the outline mask by N pixels before applying
    # to the per-joint image crops.  Ignored for other modes.
    mask_dilate_px: int = 10
    # Image + hybrid: strength of a 2D Gaussian centred on each MP
    # label that weights the Pass-2 phase correlation toward pixels
    # near the joint.  0 = uniform, 1 = sharp.  Ignored in outline.
    gauss_center_weight: float = 0.0


@router.post("/{subject_id}/run_stereo")
def run_stereo(subject_id: int, req: StereoAlignRequest) -> dict:
    """Submit a cross-camera image-alignment job for one trial.

    For each frame and joint, crops a small window around the MediaPipe
    label in each camera and runs phase correlation (no label info used)
    to find the translation that best image-aligns OS onto OD.  Saves
    the per-frame per-joint shifts + correlation responses to
    ``<skeleton>/<stem>/stereo_align.npz``.  The Stereo model on the Auto
    page then draws the opposite-camera label, translated by the
    discovered shift, alongside the MP label of the current camera.
    """
    from ..services.stereo_align import run_stereo_align
    from ..services.jobs import registry

    name = _subject_name(subject_id)
    trials = build_trial_map(name)
    if req.trial_idx < 0 or req.trial_idx >= len(trials):
        raise HTTPException(400, "trial_idx out of range")
    trial_stem = trials[req.trial_idx]["trial_name"]

    with get_db_ctx() as db:
        db.execute(
            "INSERT INTO jobs (subject_id, job_type, status, params_json) "
            "VALUES (?, 'stereo_align', 'pending', ?)",
            (subject_id, json.dumps({"trial_idx": req.trial_idx,
                                      "trial_name": trial_stem})),
        )
        job = db.execute(
            "SELECT * FROM jobs WHERE subject_id = ? ORDER BY id DESC LIMIT 1",
            (subject_id,),
        ).fetchone()
    job_id = job["id"]
    cancel_event = threading.Event()
    registry._cancel_events[job_id] = cancel_event

    def _run():
        try:
            with get_db_ctx() as db:
                db.execute(
                    "UPDATE jobs SET status='running', started_at=CURRENT_TIMESTAMP "
                    "WHERE id=?", (job_id,),
                )
            def on_progress(pct):
                if cancel_event.is_set():
                    raise _JobCancelled()
                with get_db_ctx() as db:
                    db.execute("UPDATE jobs SET progress_pct=? WHERE id=?",
                               (int(pct), job_id))
            _mode = (req.mode or ("outline" if req.use_outline else "image"))
            run_stereo_align(name, req.trial_idx,
                             progress_callback=on_progress,
                             cancel_event=cancel_event,
                             mode=_mode,
                             mask_dilate_px=int(req.mask_dilate_px),
                             gauss_center_weight=float(req.gauss_center_weight))
            with get_db_ctx() as db:
                db.execute(
                    "UPDATE jobs SET status='completed', progress_pct=100, "
                    "finished_at=CURRENT_TIMESTAMP WHERE id=?", (job_id,),
                )
        except _JobCancelled:
            with get_db_ctx() as db:
                db.execute(
                    "UPDATE jobs SET status='cancelled', "
                    "finished_at=CURRENT_TIMESTAMP WHERE id=?", (job_id,),
                )
        except Exception as e:
            import logging
            logging.getLogger(__name__).exception(f"Stereo align job {job_id} failed")
            with get_db_ctx() as db:
                db.execute(
                    "UPDATE jobs SET status='failed', error_msg=?, "
                    "finished_at=CURRENT_TIMESTAMP WHERE id=?",
                    (str(e), job_id),
                )
        finally:
            registry._cancel_events.pop(job_id, None)

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id, "trial_stem": trial_stem}


class HRnetFitRequest(BaseModel):
    trial_idx: int
    cluster_size: int = 1
    w_hm: float = 1.0
    w_disp: float = 1.0
    w_temp: float = 1.0
    overlap_px: float = 8.0
    spike_support: float = 0.0
    edge_margin: int = 0
    w_anchor: float = 1.0
    yzc_y_disp: float = 0.0
    yzc_z_outlier: float = 0.0
    yzc_attr_jump: float = 0.0
    yzc_attr_auc: float = 0.0
    yzc_z_median_mm: float = 0.0
    zsm_z_jump: float = 0.0
    zsm_smooth_window: int = 15
    bone_thresh_mm: float = 0.0
    bone_K: int = 8
    w_bone: float = 1.0
    wrist_smooth_window: int = 5


class HRnetCorrectPreviewRequest(BaseModel):
    """Live-overlay preview of Y/Z-correct error detection + camera
    attribution on the current cluster-centroid peaks."""
    trial_idx: int
    cluster_size: int = 1
    spike_support: float = 0.0
    edge_margin: int = 0
    yzc_y_disp: float = 0.0
    yzc_z_outlier: float = 0.0
    yzc_attr_jump: float = 0.0
    yzc_attr_auc: float = 0.0
    yzc_z_median_mm: float = 0.0


@router.post("/{subject_id}/hrnet_correct_preview")
def hrnet_correct_preview(subject_id: int, req: HRnetCorrectPreviewRequest) -> dict:
    """Return per-(frame, joint, camera) Y-disparity + Z-outlier flags
    and per-camera attribution at the current slider settings, scored on
    cluster-centroid peaks.  Cached per (cluster, spike, edge) tuple so
    slider changes are sub-second after the first call."""
    from ..services.hrnet_fit import compute_yzc_preview
    from ..services.video import build_trial_map

    name = _subject_name(subject_id)
    trials = build_trial_map(name)
    if req.trial_idx >= len(trials):
        raise HTTPException(400, "trial_idx out of range")
    trial_stem = trials[req.trial_idx]["trial_name"]
    return compute_yzc_preview(
        name, trial_stem,
        cluster_size=int(req.cluster_size),
        spike_support=float(req.spike_support),
        edge_margin=int(req.edge_margin),
        yzc_y_disp=float(req.yzc_y_disp),
        yzc_z_outlier=float(req.yzc_z_outlier),
        yzc_attr_jump=float(req.yzc_attr_jump),
        yzc_attr_auc=float(req.yzc_attr_auc),
        yzc_z_median_mm=float(req.yzc_z_median_mm),
    )


@router.post("/{subject_id}/hrnet_fit")
def run_hrnet_fit(subject_id: int, req: HRnetFitRequest) -> dict:
    """Submit an HRnet Fit job (cluster centroid + joint stereo Hungarian).

    Background-threaded; the registry tracks the cancel event so the
    job can be terminated mid-run from the UI."""
    from ..services.hrnet_fit import run_hrnet_fit_for_trial
    from ..services.video import build_trial_map
    from ..services.jobs import registry

    name = _subject_name(subject_id)
    trials = build_trial_map(name)
    if req.trial_idx >= len(trials):
        raise HTTPException(400, f"trial_idx out of range")
    trial_stem = trials[req.trial_idx]["trial_name"]

    with get_db_ctx() as db:
        db.execute(
            "INSERT INTO jobs (subject_id, job_type, status, params_json) "
            "VALUES (?, 'hrnet_fit', 'pending', ?)",
            (subject_id, json.dumps({"trial_name": trial_stem})),
        )
        job = db.execute(
            "SELECT * FROM jobs WHERE subject_id = ? ORDER BY id DESC LIMIT 1",
            (subject_id,),
        ).fetchone()
    job_id = job["id"]
    cancel_event = threading.Event()
    registry._cancel_events[job_id] = cancel_event

    def _run():
        try:
            with get_db_ctx() as db:
                db.execute(
                    "UPDATE jobs SET status='running', started_at=CURRENT_TIMESTAMP "
                    "WHERE id=?", (job_id,),
                )
            def on_progress(pct):
                if cancel_event.is_set():
                    raise _JobCancelled()
                with get_db_ctx() as db:
                    db.execute("UPDATE jobs SET progress_pct=? WHERE id=?",
                               (int(pct), job_id))
            run_hrnet_fit_for_trial(
                name, trial_stem,
                cluster_size=req.cluster_size,
                w_hm=req.w_hm, w_disp=req.w_disp, w_temp=req.w_temp,
                overlap_px=req.overlap_px,
                spike_support=req.spike_support,
                edge_margin=req.edge_margin,
                w_anchor=req.w_anchor,
                yzc_y_disp=req.yzc_y_disp,
                yzc_z_outlier=req.yzc_z_outlier,
                yzc_attr_jump=req.yzc_attr_jump,
                yzc_attr_auc=req.yzc_attr_auc,
                yzc_z_median_mm=req.yzc_z_median_mm,
                zsm_z_jump=req.zsm_z_jump,
                zsm_smooth_window=req.zsm_smooth_window,
                bone_thresh_mm=req.bone_thresh_mm,
                bone_K=req.bone_K,
                w_bone=req.w_bone,
                wrist_smooth_window=req.wrist_smooth_window,
                progress_callback=on_progress,
                cancel_event=cancel_event,
            )
            with get_db_ctx() as db:
                db.execute(
                    "UPDATE jobs SET status='completed', progress_pct=100, "
                    "finished_at=CURRENT_TIMESTAMP WHERE id=?", (job_id,),
                )
        except _JobCancelled:
            with get_db_ctx() as db:
                db.execute(
                    "UPDATE jobs SET status='cancelled', "
                    "finished_at=CURRENT_TIMESTAMP WHERE id=?", (job_id,),
                )
        except Exception as e:
            import logging
            logging.getLogger(__name__).exception(f"HRnet Fit job {job_id} failed")
            with get_db_ctx() as db:
                db.execute(
                    "UPDATE jobs SET status='failed', error_msg=?, "
                    "finished_at=CURRENT_TIMESTAMP WHERE id=?",
                    (str(e), job_id),
                )
        finally:
            registry._cancel_events.pop(job_id, None)

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id, "trial_stem": trial_stem}


# NOTE: the legacy `/fit_v2` endpoint that ran an FK-based v2 fit
# was removed.  The labels UI calls `/fit_v2_legacy` (writes
# skeleton_v2.npz) and `/run_corrections` (writes skeleton_v3.npz)
# directly; `/fit_v2` overlapped v3 output and confused the
# v1/v2/v3 naming.  The FK code lives in services/skeleton_v3.py as
# the private `_run_skeleton_v3_fk_legacy`.



class FitV2LegacyRequest(BaseModel):
    """Request for the frozen 'Skeleton v2' legacy fit (MP + DLC only)."""
    trial_idx: int
    w_mediapipe: float = 10.0
    w_dlc: float = 1.0
    w_bone: float = 0.0
    w_smooth_wrist: float = 1.0
    w_smooth_xy: float = 10.0
    w_smooth_z: float = 10.0
    w_smooth_angles: float = 10.0
    use_angle_constraints: bool = True
    w_constraints: float = 10.0


@router.post("/{subject_id}/fit_v2_legacy")
def run_fit_v2_legacy(subject_id: int, req: FitV2LegacyRequest) -> dict:
    """Submit a frozen Skeleton v2 (legacy absolute-position smoothing) job."""
    from ..services.skeleton_v1 import check_skeleton_available
    from ..services.skeleton_v2 import run_skeleton_v2_fit
    from ..services.jobs import registry

    status = check_skeleton_available()
    if not status["available"]:
        raise HTTPException(400, status["message"])

    name = _subject_name(subject_id)

    try:
        video_trials = build_trial_map(name)
    except Exception:
        raise HTTPException(404, "No videos found")

    if req.trial_idx < 0 or req.trial_idx >= len(video_trials):
        raise HTTPException(404, f"Trial index {req.trial_idx} out of range")

    trial_stem = video_trials[req.trial_idx]["trial_name"]
    trial_short = trial_stem.split('_', 1)[1] if '_' in trial_stem else trial_stem
    with get_db_ctx() as db:
        db.execute(
            "INSERT INTO jobs (subject_id, job_type, status, params_json) VALUES (?, 'skeleton_v2', 'pending', ?)",
            (subject_id, json.dumps({"trial_name": trial_short, "legacy": True})),
        )
        job = db.execute(
            "SELECT * FROM jobs WHERE subject_id = ? ORDER BY id DESC LIMIT 1",
            (subject_id,),
        ).fetchone()

    job_id = job["id"]
    cancel_event = threading.Event()
    registry._cancel_events[job_id] = cancel_event

    def _run():
        try:
            with get_db_ctx() as db:
                db.execute(
                    "UPDATE jobs SET status = 'running', started_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (job_id,),
                )

            def on_progress(pct):
                with get_db_ctx() as db:
                    db.execute("UPDATE jobs SET progress_pct = ? WHERE id = ?", (pct, job_id))

            result = run_skeleton_v2_fit(
                name, trial_stem,
                cancel_event=cancel_event,
                progress_callback=on_progress,
                w_mediapipe=req.w_mediapipe,
                w_dlc=req.w_dlc,
                w_bone=req.w_bone,
                w_smooth_wrist=req.w_smooth_wrist,
                w_smooth_xy=req.w_smooth_xy,
                w_smooth_z=req.w_smooth_z,
                w_smooth_angles=req.w_smooth_angles,
                use_angle_constraints=req.use_angle_constraints,
                w_constraints=req.w_constraints,
            )

            if result.get("cancelled"):
                with get_db_ctx() as db:
                    db.execute(
                        "UPDATE jobs SET status = 'cancelled', finished_at = CURRENT_TIMESTAMP WHERE id = ?",
                        (job_id,),
                    )
            else:
                with get_db_ctx() as db:
                    db.execute(
                        "UPDATE jobs SET status = 'completed', progress_pct = 100, "
                        "finished_at = CURRENT_TIMESTAMP WHERE id = ?",
                        (job_id,),
                    )
        except Exception as e:
            import logging
            logging.getLogger(__name__).exception(f"Skeleton v2 fit job {job_id} failed")
            with get_db_ctx() as db:
                db.execute(
                    "UPDATE jobs SET status = 'failed', error_msg = ?, "
                    "finished_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (str(e), job_id),
                )
        finally:
            registry._cancel_events.pop(job_id, None)

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id, "trial_stem": trial_stem}


class MPErrorRequest(BaseModel):
    """Error-detection slider weights.

    Detection weights (per factor ∈ [0, 1]): z_jump, z_outlier, y_disp,
    bone_length, bone_agreement, angle, reproj, confidence.  Attribution
    weights: jump_2d, confidence.  Correction weights: y_disp (fraction
    of Y-disparity frames to correct per joint, 0..1).
    """
    trial_idx: int
    detection: dict[str, float] = {}
    attribution: dict[str, float] = {}
    corrections: dict[str, float] = {}
    # HRnet peak source for the snap step in run_corrections.
    # "auto" (default), "hungarian", "centroid", "refined", "raw".
    hrnet_source: str = "auto"
    stage: str | None = None
    # Step 0: Stereo-correction config (v3 fit only).  Defaults are a
    # no-op when stereo_dist_px == 0.
    stereo_mode: str = "image"        # image / outline / hybrid
    mask_dilate_px: int = 10           # hybrid only
    gauss_center_weight: float = 0.0   # image + hybrid
    stereo_conf: float = 0.0           # min confidence to consider a stereo label
    stereo_dist_px: float = 0.0        # 0 disables stereo-correction step
    # 2D radius (px) for the occlusion-revert post-pass.  When > 0,
    # any stereo-corrected joint whose Z drops toward an "overlying"
    # joint (within this radius in either camera AND closer to the
    # camera) is rolled back to its MP label.
    stereo_occlusion_px: float = 0.0


def _encode_errors(errors) -> list[list[list[int]]]:
    """Compact encoding: (N, 21, 2) bool → list of list of [L, R] ints."""
    return [[[int(bool(errors[f, j, 0])), int(bool(errors[f, j, 1]))]
             for j in range(errors.shape[1])]
            for f in range(errors.shape[0])]


def _encode_points(arr) -> list[list[list[float]]]:
    """(N, 21, 2) float → nested list, with NaN preserved (as Python None
    since JSON doesn't have NaN)."""
    import math
    out = []
    for f in range(arr.shape[0]):
        row = []
        for j in range(arr.shape[1]):
            x, y = float(arr[f, j, 0]), float(arr[f, j, 1])
            if math.isnan(x) or math.isnan(y):
                row.append(None)
            else:
                row.append([round(x, 2), round(y, 2)])
        out.append(row)
    return out


@router.post("/{subject_id}/mp_errors")
def mp_errors(subject_id: int, req: MPErrorRequest) -> dict:
    """Compute the MediaPipe error matrix for live slider updates.

    When any correction weight > 0, also returns the corrected MP positions
    so the frontend can display them in place of the originals.
    """
    from ..services.mp_error_detection import compute_errors_for_trial

    name = _subject_name(subject_id)
    try:
        video_trials = build_trial_map(name)
    except Exception:
        raise HTTPException(404, "No videos found")
    if req.trial_idx < 0 or req.trial_idx >= len(video_trials):
        raise HTTPException(404, f"Trial index {req.trial_idx} out of range")
    trial_stem = video_trials[req.trial_idx]["trial_name"]

    result = compute_errors_for_trial(name, trial_stem,
                                      req.detection, req.attribution,
                                      req.corrections, stage=req.stage)
    errors = result["errors"]
    corrected = result["n_corrected"] > 0
    return {
        "errors": _encode_errors(errors),
        "n_frames": int(errors.shape[0]),
        "total_flagged": int(errors.sum()),
        "n_corrected": int(result["n_corrected"]),
        "corrected_mp_L": _encode_points(result["mp_L"]) if corrected else None,
        "corrected_mp_R": _encode_points(result["mp_R"]) if corrected else None,
    }


@router.post("/{subject_id}/run_corrections")
def run_corrections(subject_id: int, req: MPErrorRequest) -> dict:
    """Submit the MP correction pipeline as a background job so the UI
    stays responsive while the BL optimisation runs.  Saved output:
    ``skeleton_v3.npz`` + ``mp_errors.npz``."""
    from ..services.mp_error_detection import run_correction_pipeline, save_errors
    from ..services.jobs import registry

    name = _subject_name(subject_id)
    try:
        video_trials = build_trial_map(name)
    except Exception:
        raise HTTPException(404, "No videos found")
    if req.trial_idx < 0 or req.trial_idx >= len(video_trials):
        raise HTTPException(404, f"Trial index {req.trial_idx} out of range")
    trial_stem = video_trials[req.trial_idx]["trial_name"]

    trial_short = trial_stem.split('_', 1)[1] if '_' in trial_stem else trial_stem
    with get_db_ctx() as db:
        db.execute(
            "INSERT INTO jobs (subject_id, job_type, status, params_json) VALUES (?, 'skeleton_v3', 'pending', ?)",
            (subject_id, json.dumps({"trial_name": trial_short})),
        )
        job = db.execute(
            "SELECT * FROM jobs WHERE subject_id = ? ORDER BY id DESC LIMIT 1",
            (subject_id,),
        ).fetchone()
    job_id = job["id"]
    cancel_event = threading.Event()
    registry._cancel_events[job_id] = cancel_event

    det = dict(req.detection); attr = dict(req.attribution)

    def _run():
        try:
            with get_db_ctx() as db:
                db.execute(
                    "UPDATE jobs SET status = 'running', started_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (job_id,),
                )
            def on_progress(pct):
                # Honour the cancel event by raising — the worker's stages
                # call on_progress between phases, so the exception breaks
                # us out of the next stage's expensive loop.
                if cancel_event.is_set():
                    raise _JobCancelled()
                with get_db_ctx() as db:
                    db.execute("UPDATE jobs SET progress_pct = ? WHERE id = ?",
                               (pct, job_id))
            run_correction_pipeline(name, trial_stem, det, attr,
                                    progress_callback=on_progress,
                                    cancel_event=cancel_event,
                                    hrnet_source=req.hrnet_source,
                                    stereo_mode=req.stereo_mode,
                                    stereo_mask_dilate_px=int(req.mask_dilate_px),
                                    stereo_gauss_center_weight=float(req.gauss_center_weight),
                                    stereo_conf=float(req.stereo_conf),
                                    stereo_dist_px=float(req.stereo_dist_px),
                                    stereo_occlusion_px=float(req.stereo_occlusion_px))
            if cancel_event.is_set():
                raise _JobCancelled()
            save_errors(name, trial_stem, det, attr)
            with get_db_ctx() as db:
                db.execute(
                    "UPDATE jobs SET status = 'completed', progress_pct = 100, "
                    "finished_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (job_id,),
                )
        except (_JobCancelled, _CorrectionsCancelled):
            with get_db_ctx() as db:
                db.execute(
                    "UPDATE jobs SET status = 'cancelled', "
                    "finished_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (job_id,),
                )
        except Exception as e:
            import logging
            logging.getLogger(__name__).exception(f"MP corrections job {job_id} failed")
            with get_db_ctx() as db:
                db.execute(
                    "UPDATE jobs SET status = 'failed', error_msg = ?, "
                    "finished_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (str(e), job_id),
                )
        finally:
            registry._cancel_events.pop(job_id, None)

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id, "trial_stem": trial_stem}


@router.post("/prefill_error_caches")
def prefill_error_caches() -> dict:
    """Walk every subject + trial + stage and pre-compute per-stage error
    caches.  Runs in a background thread; returns the new job's ID."""
    from ..services.mp_error_detection import prefill_error_caches_for_all_subjects

    with get_db_ctx() as db:
        # jobs.subject_id is NOT NULL — use the first subject's id as a
        # placeholder for whole-database jobs.  The job_type is enough to
        # disambiguate.
        first = db.execute("SELECT id FROM subjects ORDER BY id LIMIT 1").fetchone()
        if first is None:
            raise HTTPException(400, "No subjects in database")
        placeholder_subj = first["id"]
        db.execute(
            "INSERT INTO jobs (subject_id, job_type, status, params_json) VALUES (?, 'prefill_error_caches', 'pending', '{}')",
            (placeholder_subj,),
        )
        job = db.execute(
            "SELECT * FROM jobs WHERE job_type = 'prefill_error_caches' ORDER BY id DESC LIMIT 1"
        ).fetchone()
    job_id = job["id"]
    cancel_event = threading.Event()
    registry._cancel_events[job_id] = cancel_event

    def _run():
        try:
            with get_db_ctx() as db:
                db.execute("UPDATE jobs SET status = 'running', started_at = CURRENT_TIMESTAMP WHERE id = ?", (job_id,))
            def on_progress(pct):
                if cancel_event.is_set():
                    raise _JobCancelled()
                with get_db_ctx() as db:
                    db.execute("UPDATE jobs SET progress_pct = ? WHERE id = ?", (pct, job_id))
            summary = prefill_error_caches_for_all_subjects(
                progress_callback=on_progress, cancel_event=cancel_event)
            with get_db_ctx() as db:
                db.execute(
                    "UPDATE jobs SET status = 'completed', progress_pct = 100, "
                    "finished_at = CURRENT_TIMESTAMP, error_msg = ? WHERE id = ?",
                    (json.dumps(summary)[:1000], job_id),
                )
        except (_JobCancelled, _CorrectionsCancelled):
            with get_db_ctx() as db:
                db.execute(
                    "UPDATE jobs SET status = 'cancelled', "
                    "finished_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (job_id,),
                )
        except Exception as e:
            import logging
            logging.getLogger(__name__).exception(f"Prefill error-caches job {job_id} failed")
            with get_db_ctx() as db:
                db.execute(
                    "UPDATE jobs SET status = 'failed', error_msg = ?, finished_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (str(e), job_id),
                )
        finally:
            registry._cancel_events.pop(job_id, None)

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id}


@router.post("/{subject_id}/save_mp_errors")
def save_mp_errors(subject_id: int, req: MPErrorRequest) -> dict:
    """Compute and persist the error matrix to ``mp_errors.npz``."""
    from ..services.mp_error_detection import save_errors

    name = _subject_name(subject_id)
    try:
        video_trials = build_trial_map(name)
    except Exception:
        raise HTTPException(404, "No videos found")
    if req.trial_idx < 0 or req.trial_idx >= len(video_trials):
        raise HTTPException(404, f"Trial index {req.trial_idx} out of range")
    trial_stem = video_trials[req.trial_idx]["trial_name"]

    out_path = save_errors(name, trial_stem, req.detection, req.attribution,
                           req.corrections)
    return {"path": str(out_path)}


@router.get("/{subject_id}/video_list")
def get_video_list(subject_id: int) -> list[dict]:
    """List all video trials for a subject (for Videos module — no Skeleton data required).

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
