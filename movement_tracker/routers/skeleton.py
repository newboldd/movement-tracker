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


@router.get("/{subject_id}/trial/{trial_idx}/stereo_fill_data")
def get_trial_stereo_fill_data(subject_id: int, trial_idx: int) -> Response:
    """Lazy-fetched Stereo Fill payload for one trial.

    Skel Fit v1 and the Labels page Stereo Fill model both want the
    filtered+stereo-donated MP combined arrays for a trial.  The
    computation is heavy (calls detect_mask twice through
    build_and_validate_stereo_fill, ~150 ms at trial scale) so the
    main /trial/{idx}/data response defers it — the client fetches
    this endpoint only when the user toggles the Stereo Fill row on.
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
    try:
        data = load_skeleton_trial_data(
            name, trial["trial_stem"], compute_stereo_fill=True)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    # Return only the Stereo Fill keys to keep the payload small.
    out = {
        "stereo_fill_tracked_L": data.get("stereo_fill_tracked_L"),
        "stereo_fill_tracked_R": data.get("stereo_fill_tracked_R"),
        "stereo_fill_joints_3d": data.get("stereo_fill_joints_3d"),
        "stereo_fill_source":    data.get("stereo_fill_source"),
        "distances_stereo_fill": data.get("distances_stereo_fill"),
        "has_stereo_fill":       data.get("has_stereo_fill", False),
    }
    import math, numpy as np
    def _default(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating, float)):
            v = float(o)
            if math.isnan(v) or math.isinf(v): return None
            return v
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")
    body = json.dumps(out, separators=(",", ":"), default=_default)
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
    # Outlier-detection thresholds (see _detect_outlier_per_joint).
    # accel_k / bone_k: lower values mask more aggressively.
    # k_max: longest run length the spike-pair detector accepts.
    accel_k: float = 6.0
    bone_k: float = 6.0
    k_max: int = 30


@router.get("/{subject_id}/fit/status")
def get_fit_status(subject_id: int) -> dict:
    """Check if Skeleton fitting dependencies are available."""
    from ..services.skeleton_v1 import check_skeleton_available
    return check_skeleton_available()


@router.get("/stereo_disparity")
def stereo_disparity_by_name(
    subject: str = Query(..., description="Subject name (e.g. MSA01)"),
    trial: str = Query(..., description="Trial name (e.g. L1) or numeric trial_idx"),
) -> dict:
    """Name-friendly wrapper around the numeric stereo_disparity endpoint.

    Resolves ``subject`` (name) → subject_id and ``trial`` (name or
    numeric idx) → trial_idx, then delegates.  Use this form when
    typing into a browser URL by hand:

        /api/skeleton/stereo_disparity?subject=MSA01&trial=L1
    """
    from ..db import get_db_ctx
    from ..services.video import build_trial_map

    with get_db_ctx() as db:
        row = db.execute(
            "SELECT id FROM subjects WHERE name = ? COLLATE NOCASE", (subject,),
        ).fetchone()
    if row is None:
        raise HTTPException(404, f"Subject {subject!r} not found")
    sid = int(row["id"])

    tidx: int | None = None
    if trial.isdigit() or (trial.startswith("-") and trial[1:].isdigit()):
        tidx = int(trial)
    else:
        # Match by trial_name suffix (e.g. "L1" matches "MSA01_L1")
        # case-insensitive; fall back to exact match.
        name = _subject_name(sid)
        tmap = build_trial_map(name)
        wanted = trial.lower()
        for i, t in enumerate(tmap):
            tn = (t.get("trial_name") or "").lower()
            if tn == wanted or tn.endswith("_" + wanted) or tn.endswith(wanted):
                tidx = i; break
        if tidx is None:
            raise HTTPException(
                404,
                f"Trial {trial!r} not found for {subject!r}. "
                f"Available: {[t.get('trial_name') for t in tmap]}",
            )
    return trial_stereo_disparity(sid, tidx)


class OutlierPreviewRequest(BaseModel):
    trial_idx: int
    # Each signal has an `enable_<name>` flag and one or more
    # threshold parameters.  Disabled signals are skipped.
    enable_vel: bool = False
    vel_k: float = 6.0
    enable_accel: bool = True
    accel_k: float = 6.0
    enable_bone: bool = True
    bone_k: float = 6.0
    enable_ydisp: bool = False
    ydisp_px: float = 5.0
    enable_z: bool = False
    z_k: float = 6.0
    enable_mpconf: bool = False
    mpconf_min: float = 0.5
    enable_stereo: bool = False
    stereo_px: float = 5.0
    enable_stereo_outline: bool = False
    stereo_outline_px: float = 10.0
    stereo_outline_conf_min: float = 0.2
    enable_hrnet: bool = False
    hrnet_min: float = 0.2


def _mp_filter_params_path(subject_name: str, trial_stem: str):
    from ..services.skeleton_data import _skeleton_dir
    return _skeleton_dir(subject_name) / trial_stem / "mp_filter_params.json"


@router.post("/{subject_id}/trial/{trial_idx}/mp_filter_save")
def mp_filter_save(subject_id: int, trial_idx: int,
                   req: OutlierPreviewRequest) -> dict:
    """Persist the user's MP Filter slider state as
    ``mp_filter_params.json`` alongside the trial's skeleton data.

    The same sidecar is read by run_skeleton_v1_fit so the v1 fit
    uses the saved (and live-tuned) filter instead of running its
    own internal detector.
    """
    name = _subject_name(subject_id)
    tmap = build_trial_map(name)
    if trial_idx < 0 or trial_idx >= len(tmap):
        raise HTTPException(404, f"Trial index {trial_idx} out of range")
    trial_stem = tmap[trial_idx]["trial_name"]
    payload = req.model_dump() if hasattr(req, "model_dump") else req.dict()
    payload.pop("trial_idx", None)
    payload["_saved_at"] = json_now_iso()
    path = _mp_filter_params_path(name, trial_stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return {"ok": True, "path": str(path), "params": payload}


@router.delete("/{subject_id}/trial/{trial_idx}/mp_filter_save")
def mp_filter_save_delete(subject_id: int, trial_idx: int) -> dict:
    """Delete the saved MP Filter sidecar for the trial."""
    name = _subject_name(subject_id)
    tmap = build_trial_map(name)
    if trial_idx < 0 or trial_idx >= len(tmap):
        raise HTTPException(404, f"Trial index {trial_idx} out of range")
    trial_stem = tmap[trial_idx]["trial_name"]
    path = _mp_filter_params_path(name, trial_stem)
    existed = path.exists()
    if existed:
        path.unlink()
    return {"ok": True, "existed": existed}


def json_now_iso() -> str:
    from datetime import datetime
    return datetime.now().isoformat(timespec="seconds")


@router.get("/{subject_id}/trial/{trial_idx}/mp_filter_data")
def mp_filter_data(subject_id: int, trial_idx: int) -> dict:
    """Precompute every MP-Filter signal's raw per-(frame, joint[, camera])
    array for one trial and return them base64-encoded as Float32.

    Used by the Labels-page MP Filter panel: fetched once when the
    panel opens, cached in JS, then thresholded + OR'd entirely
    client-side as the user moves sliders.  No round-trip per slider
    tweak.

    Payload sizes scale linearly with N: at N=1100, J=21 we ship
    ~600 KB across all arrays.  Float32 NaN preserves through
    base64 round-trip and reaches JS as JavaScript NaN, which the
    client treats as "no signal" (never fires).
    """
    import base64
    import numpy as np
    from ..services.calibration import get_calibration_for_subject, triangulate_points
    from ..services.skeleton_data import _load_trial_calibration, _skeleton_dir, _load_hrnet_peaks_json
    from ..services.skeleton_v1 import BONES
    from ..services.mediapipe_prelabel import (
        load_mediapipe_combined_prelabels, load_mediapipe_prelabels,
    )
    from ..services.mp_filter import build_signal_data

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
    mp_L = os_lm[start:end].copy()
    mp_R = od_lm[start:end].copy()
    N_total = mp_L.shape[0]

    conf_L = prelabels.get("confidence_OS")
    conf_R = prelabels.get("confidence_OD")
    if conf_L is not None: conf_L = conf_L[start:end]
    if conf_R is not None: conf_R = conf_R[start:end]

    calib = _load_trial_calibration(name, trial_stem)
    if calib is None:
        calib = get_calibration_for_subject(name)

    # HRnet per-(frame, joint, camera) peak score, derived as
    # max-over-spatial-dims of the saved heatmaps.  Loads
    # hrnet_w18_heatmaps.npz lazily; absent → no HRnet signal.
    hrnet_L = None
    hrnet_R = None
    try:
        sk_root = _skeleton_dir(name)
        sk_trial = sk_root / trial_stem
        hm_path = sk_trial / "hrnet_w18_heatmaps.npz"
        if hm_path.exists():
            with np.load(str(hm_path), allow_pickle=False) as _hm:
                if "heatmaps_L" in _hm.files:
                    arrL = _hm["heatmaps_L"]
                    if arrL.ndim == 4:
                        # max over (H, W) → (n_frames_hm, 21)
                        scores = arrL.max(axis=(2, 3)).astype(np.float32)
                        sf = int(_hm["start_frame"]) if "start_frame" in _hm.files else start
                        nf = scores.shape[0]
                        hrnet_L = np.full((N_total, 21), np.nan, dtype=np.float64)
                        rel = sf - start
                        a0 = max(0, -rel); b0 = a0 + nf
                        a1 = max(0, rel);  b1 = a1 + (nf - a0)
                        if b1 > N_total: b1 = N_total
                        hrnet_L[a1:b1, :] = scores[a0:a0 + (b1 - a1), :]
                if "heatmaps_R" in _hm.files:
                    arrR = _hm["heatmaps_R"]
                    if arrR.ndim == 4:
                        scores = arrR.max(axis=(2, 3)).astype(np.float32)
                        sf = int(_hm["start_frame"]) if "start_frame" in _hm.files else start
                        nf = scores.shape[0]
                        hrnet_R = np.full((N_total, 21), np.nan, dtype=np.float64)
                        rel = sf - start
                        a0 = max(0, -rel); b0 = a0 + nf
                        a1 = max(0, rel);  b1 = a1 + (nf - a0)
                        if b1 > N_total: b1 = N_total
                        hrnet_R[a1:b1, :] = scores[a0:a0 + (b1 - a1), :]
    except Exception as _e:
        hrnet_L = None
        hrnet_R = None

    # Triangulate per joint (same path as outlier_preview).
    mp_3d = np.full((N_total, 21, 3), np.nan)
    if calib is not None:
        for j in range(21):
            mp_3d[:, j, :] = triangulate_points(mp_L[:, j, :], mp_R[:, j, :], calib)
    valid = ~np.isnan(mp_L[:, 0, 0]) & ~np.isnan(mp_R[:, 0, 0])
    if calib is not None:
        valid = valid & ~np.isnan(mp_3d[:, 0, 0])
    init_3d = mp_3d.copy() if calib is not None else np.full((N_total, 21, 3), np.nan)
    # Wrist fallback on stray NaNs so the velocity/accel arrays
    # don't carry NaN spikes at the boundaries of valid runs.
    for i in range(N_total):
        if not valid[i]:
            init_3d[i, :, :] = np.nan
            continue
        for j in range(21):
            if np.isnan(init_3d[i, j, 0]):
                if not np.isnan(init_3d[i, 0, 0]):
                    init_3d[i, j] = init_3d[i, 0]
                else:
                    init_3d[i, j] = [0.0, 0.0, 500.0]

    # Outline stereo: per-joint shift magnitude (px) AND per-joint
    # phase-correlation response (~[0,1] confidence).  Both drive
    # the "Stereo error" pair of MP-Filter sliders.  NaN-padded
    # when the trial doesn't have an outline stereo_align npz.
    stereo_shift_mag = None
    stereo_response = None
    try:
        from ..services.stereo_align import load_stereo_align
        _sa_out = load_stereo_align(name, trial_idx, mode='outline')
        if _sa_out is not None and "shifts" in _sa_out:
            _sh = np.asarray(_sa_out["shifts"], dtype=np.float64)
            if _sh.ndim == 3 and _sh.shape[1] == 21 and _sh.shape[2] == 2:
                stereo_shift_mag = np.linalg.norm(_sh, axis=-1)
        if _sa_out is not None and "response" in _sa_out:
            _rsp = np.asarray(_sa_out["response"], dtype=np.float64)
            if _rsp.ndim == 2 and _rsp.shape[1] == 21:
                stereo_response = _rsp
    except Exception:
        stereo_shift_mag = None
        stereo_response = None

    arrays = build_signal_data(
        init_3d, mp_L, mp_R, BONES,
        calib=calib,
        confidence_L=conf_L, confidence_R=conf_R,
        hrnet_L=hrnet_L,    hrnet_R=hrnet_R,
        stereo_shift_mag=stereo_shift_mag,
        stereo_response=stereo_response,
    )

    # Encode each numpy array as a base64 Float32 blob.  Python's
    # default byteorder on x86 / Apple Silicon matches JS
    # Float32Array (little-endian), so no swap is needed.
    encoded: dict = {}
    shapes: dict = {}
    for k, v in arrays.items():
        if isinstance(v, np.ndarray):
            encoded[k] = base64.b64encode(v.tobytes()).decode("ascii")
            shapes[k] = list(v.shape)
        else:
            encoded[k] = v
    # Read any previously-saved MP Filter sidecar so the UI can
    # restore the user's tuned thresholds on panel open.
    saved_params = None
    try:
        sidecar = _mp_filter_params_path(name, trial_stem)
        if sidecar.exists():
            saved_params = json.loads(sidecar.read_text())
    except Exception:
        saved_params = None
    return {
        "N": arrays["N"],
        "J": arrays["J"],
        "valid_mask": valid.tolist(),
        "bones": arrays["bones"],
        "arrays_b64": encoded,
        "shapes": shapes,
        "has_calib":   calib is not None,
        "has_mp_conf": (conf_L is not None or conf_R is not None),
        "has_hrnet":   (hrnet_L is not None or hrnet_R is not None),
        "has_stereo_outline": stereo_shift_mag is not None,
        "saved_params": saved_params,
    }


@router.post("/{subject_id}/outlier_preview")
def outlier_preview(subject_id: int, req: OutlierPreviewRequest) -> dict:
    """Run the multi-signal MP-Filter detector on the requested trial.

    Returns the OR'd (frame, joint) and (frame, joint, camera) masks
    plus a per-signal count so the Labels-page MP Filter panel can
    show which signals fired and live-preview the result.
    """
    import numpy as np
    from ..services.calibration import get_calibration_for_subject, triangulate_points
    from ..services.skeleton_data import _load_trial_calibration, _skeleton_dir, _load_hrnet_peaks_json
    from ..services.skeleton_v1 import BONES
    from ..services.mediapipe_prelabel import (
        load_mediapipe_combined_prelabels, load_mediapipe_prelabels,
    )
    from ..services.mp_filter import detect_mask

    name = _subject_name(subject_id)
    tmap = build_trial_map(name)
    if req.trial_idx < 0 or req.trial_idx >= len(tmap):
        raise HTTPException(404, f"Trial index {req.trial_idx} out of range")
    trial = tmap[req.trial_idx]
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
    mp_L = os_lm[start:end].copy()
    mp_R = od_lm[start:end].copy()
    N_total = mp_L.shape[0]

    # MP confidence — hand-level scalar per (frame, camera).  Saved
    # by the combined+forward pipelines under confidence_OS / _OD.
    conf_L = prelabels.get("confidence_OS")
    conf_R = prelabels.get("confidence_OD")
    if conf_L is not None: conf_L = conf_L[start:end]
    if conf_R is not None: conf_R = conf_R[start:end]

    calib = _load_trial_calibration(name, trial_stem)
    if calib is None:
        calib = get_calibration_for_subject(name)
    # Calib is optional now — only the ydisp / stereo signals need it.

    # HRnet peak scores per (frame, joint, camera).  Best-effort load
    # from the per-trial peaks JSON; absent for trials without HRnet.
    hrnet_L = None
    hrnet_R = None
    try:
        if calib is not None:
            sk_root = _skeleton_dir(name)
            sk_trial = sk_root / trial_stem
            peaks = _load_hrnet_peaks_json(sk_trial)
            if peaks and "refined" in peaks:
                ref = peaks["refined"]
                # Expected layout: ref[cam][frame_str] = {joint_idx_str: [x, y, score]}
                def _scores_array(cam_key):
                    arr = np.full((N_total, 21), np.nan, dtype=np.float64)
                    cam = ref.get(cam_key) or {}
                    for fstr, jdict in cam.items():
                        try: f = int(fstr)
                        except Exception: continue
                        if not (0 <= f < N_total): continue
                        if not isinstance(jdict, dict): continue
                        for jstr, val in jdict.items():
                            try: j = int(jstr)
                            except Exception: continue
                            if not (0 <= j < 21): continue
                            if isinstance(val, (list, tuple)) and len(val) >= 3:
                                try: arr[f, j] = float(val[2])
                                except Exception: pass
                    return arr
                if all(k in ref for k in ("L", "OS")):
                    pass
                hrnet_L = _scores_array("L" if "L" in ref else "OS")
                hrnet_R = _scores_array("R" if "R" in ref else "OD")
    except Exception:
        hrnet_L = None
        hrnet_R = None

    # Triangulate per joint to build init_3d.
    mp_3d = np.full((N_total, 21, 3), np.nan)
    if calib is not None:
        for j in range(21):
            mp_3d[:, j, :] = triangulate_points(mp_L[:, j, :], mp_R[:, j, :], calib)
    valid_mask = (
        ~np.isnan(mp_L[:, 0, 0])
        & ~np.isnan(mp_R[:, 0, 0])
    )
    if calib is not None:
        valid_mask = valid_mask & ~np.isnan(mp_3d[:, 0, 0])
    valid_idx = np.where(valid_mask)[0]
    n_valid = len(valid_idx)
    if n_valid < 3:
        return {
            "n_frames": int(N_total), "n_valid": int(n_valid),
            "joint_mask": [], "frame_mask": [], "camera_mask": [],
            "n_cells_masked": 0, "n_frames_masked": 0,
            "n_cam_L_masked": 0, "n_cam_R_masked": 0,
            "per_signal": {},
        }
    init_3d = mp_3d[valid_idx].copy() if calib is not None else np.full((n_valid, 21, 3), np.nan)
    for i in range(n_valid):
        for j in range(21):
            if np.isnan(init_3d[i, j, 0]):
                if not np.isnan(init_3d[i, 0, 0]):
                    init_3d[i, j] = init_3d[i, 0]
                else:
                    init_3d[i, j] = [0.0, 0.0, 500.0]

    def _slice(a):
        if a is None: return None
        return np.asarray(a)[valid_idx]

    # Outline stereo: per-joint shift magnitude (px) + per-joint
    # phase-corr response (~[0,1]).  Confidence acts as a gate on
    # the distance check, so both arrays are loaded together iff
    # the Stereo error signal is enabled.
    _stereo_shift_mag_v = None
    _stereo_response_v = None
    if req.enable_stereo_outline:
        try:
            from ..services.stereo_align import load_stereo_align
            _sa_out = load_stereo_align(name, req.trial_idx, mode='outline')
            if _sa_out is not None:
                def _stitch(arr):
                    full = np.full((N_total, 21), np.nan, dtype=np.float64)
                    m = min(N_total, arr.shape[0])
                    full[:m] = arr[:m]
                    return full[valid_idx]
                if "shifts" in _sa_out:
                    _sh = np.asarray(_sa_out["shifts"], dtype=np.float64)
                    if _sh.ndim == 3 and _sh.shape[1] == 21 and _sh.shape[2] == 2:
                        _stereo_shift_mag_v = _stitch(np.linalg.norm(_sh, axis=-1))
                if "response" in _sa_out:
                    _rsp = np.asarray(_sa_out["response"], dtype=np.float64)
                    if _rsp.ndim == 2 and _rsp.shape[1] == 21:
                        _stereo_response_v = _stitch(_rsp)
        except Exception:
            _stereo_shift_mag_v = None
            _stereo_response_v = None

    joint_v, cam_v, per_signal = detect_mask(
        init_3d,
        mp_L[valid_idx], mp_R[valid_idx],
        BONES,
        calib=calib,
        confidence_L=_slice(conf_L), confidence_R=_slice(conf_R),
        hrnet_L=_slice(hrnet_L),    hrnet_R=_slice(hrnet_R),
        enable_vel=req.enable_vel,        vel_k=req.vel_k,
        enable_accel=req.enable_accel,    accel_k=req.accel_k,
        enable_bone=req.enable_bone,      bone_k=req.bone_k,
        enable_ydisp=req.enable_ydisp,    ydisp_px=req.ydisp_px,
        enable_z=req.enable_z,            z_k=req.z_k,
        enable_mpconf=req.enable_mpconf,  mpconf_min=req.mpconf_min,
        enable_stereo=req.enable_stereo,  stereo_px=req.stereo_px,
        enable_stereo_outline=req.enable_stereo_outline,
        stereo_outline_px=req.stereo_outline_px,
        stereo_outline_conf_min=req.stereo_outline_conf_min,
        stereo_shift_mag=_stereo_shift_mag_v,
        stereo_response=_stereo_response_v,
        enable_hrnet=req.enable_hrnet,    hrnet_min=req.hrnet_min,
    )
    joint_full = np.zeros((N_total, 21), dtype=bool)
    cam_full = np.zeros((N_total, 21, 2), dtype=bool)
    joint_full[valid_idx] = joint_v
    cam_full[valid_idx] = cam_v
    frame_full = joint_full.any(axis=1)
    return {
        "n_frames": int(N_total),
        "n_valid": int(n_valid),
        "joint_mask": joint_full.astype(bool).tolist(),
        "camera_mask": cam_full.astype(bool).tolist(),
        "frame_mask": frame_full.astype(bool).tolist(),
        "n_cells_masked":   int(joint_full.sum()),
        "n_frames_masked":  int(frame_full.sum()),
        "n_cam_L_masked":   int(cam_full[:, :, 0].sum()),
        "n_cam_R_masked":   int(cam_full[:, :, 1].sum()),
        "per_signal": per_signal,
        "has_calib":      calib is not None,
        "has_mp_conf":    (conf_L is not None or conf_R is not None),
        "has_hrnet":      (hrnet_L is not None or hrnet_R is not None),
    }


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
                accel_k=req.accel_k,
                bone_k=req.bone_k,
                k_max=req.k_max,
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
    # "image" | "outline".  Default "image" when unset.
    mode: str | None = None
    # Outline only: dilate the outline mask by N pixels before
    # applying to the per-joint image crops.  Ignored for image mode.
    mask_dilate_px: int = 10
    # Strength of a 2D Gaussian centred on each MP label that weights
    # the Pass-2 phase correlation toward pixels near the joint.
    # 0 = uniform, 1 = sharp.
    gauss_center_weight: float = 0.0


@router.get("/{subject_id}/trial/{trial_idx}/stereo_params")
def trial_stereo_params(subject_id: int, trial_idx: int) -> dict:
    """Return the saved Stereo-panel knobs for each baked variant
    of this trial.

    The Labels page opens the Stereo panel pre-filled with whatever
    settings produced the existing npz for the currently-selected
    radio (image / outline).  Modes with no baked npz are omitted
    from the response — the frontend falls back to defaults.
    """
    from ..services.stereo_align import load_stereo_align
    name = _subject_name(subject_id)
    trials = build_trial_map(name)
    if trial_idx < 0 or trial_idx >= len(trials):
        raise HTTPException(400, "trial_idx out of range")
    out: dict[str, dict] = {}
    for mode in ("image", "outline"):
        sa = load_stereo_align(name, trial_idx, mode=mode)
        if sa is None:
            continue
        entry: dict = {}
        if "mask_dilate_px" in sa:
            # -1 sentinel = "not applicable for this mode" — only
            # outline actually stores a real value.  Map it to None
            # so the frontend doesn't display "-1 px".
            v = int(sa["mask_dilate_px"])
            entry["mask_dilate_px"] = None if v < 0 else v
        if "gauss_center_weight" in sa:
            g = float(sa["gauss_center_weight"])
            entry["gauss_center_weight"] = None if g < 0 else g
        out[mode] = entry
    return {"per_mode": out}


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
            _mode = req.mode or "image"
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
    """Request for the frozen 'Skeleton v2' legacy fit.

    MP combined is the sole 2D input — the old MP / DLC weights and
    angle-constraint controls were removed when v2 was simplified.
    """
    trial_idx: int
    w_bone: float = 0.0
    w_smooth_wrist: float = 5.0
    w_smooth_xy: float = 10.0
    w_smooth_z: float = 10.0
    w_smooth_angles: float = 10.0


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
                w_bone=req.w_bone,
                w_smooth_wrist=req.w_smooth_wrist,
                w_smooth_xy=req.w_smooth_xy,
                w_smooth_z=req.w_smooth_z,
                w_smooth_angles=req.w_smooth_angles,
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
    stereo_mode: str = "image"        # image / outline
    mask_dilate_px: int = 10           # outline only
    gauss_center_weight: float = 0.0   # image + outline
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
