"""HTTP endpoints for the Pre-proc page.

Phase A surfaces the camera-motion trajectory; later phases add
background extraction, frame quality, and calibration check on the
same prefix.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import FileResponse

from ..db import get_db_ctx
from ..services.jobs import registry
from ..services.video import build_trial_map


router = APIRouter(prefix="/api/preproc", tags=["preproc"])
logger = logging.getLogger(__name__)


class _JobCancelled(Exception):
    pass


def _subject_name(subject_id: int) -> str:
    with get_db_ctx() as db:
        row = db.execute(
            "SELECT name FROM subjects WHERE id=?", (subject_id,)
        ).fetchone()
    if not row:
        raise HTTPException(404, f"Subject {subject_id} not found")
    return row["name"]


# ── Camera trajectory ────────────────────────────────────────────────────

@router.post("/{subject_id}/compute_trajectory")
def compute_trajectory(subject_id: int, body: dict = Body(...)) -> dict:
    """Spawn a camera-motion trajectory job for one trial.

    Body: ``{trial_idx: int, nfeatures?, ransac_thresh?, min_inliers?,
              max_err_px?}``.
    """
    from ..services.camera_motion import compute_camera_trajectory

    name = _subject_name(subject_id)
    trial_idx = int(body.get("trial_idx", 0))
    trials = build_trial_map(name)
    if trial_idx < 0 or trial_idx >= len(trials):
        raise HTTPException(400, f"trial_idx out of range")
    trial_stem = trials[trial_idx]["trial_name"]

    with get_db_ctx() as db:
        db.execute(
            "INSERT INTO jobs (subject_id, job_type, status, params_json) "
            "VALUES (?, 'camera_motion', 'pending', ?)",
            (subject_id, json.dumps({"trial_idx": trial_idx,
                                      "trial_name": trial_stem})),
        )
        job = db.execute(
            "SELECT * FROM jobs WHERE subject_id = ? ORDER BY id DESC LIMIT 1",
            (subject_id,),
        ).fetchone()
    job_id = job["id"]
    cancel_event = threading.Event()
    registry._cancel_events[job_id] = cancel_event

    # Optional knobs from request body — all have sensible defaults.
    kwargs = {}
    for k in ("nfeatures", "ransac_thresh", "min_inliers", "max_err_px"):
        if body.get(k) is not None:
            kwargs[k] = body[k]

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
                               (round(float(pct), 1), job_id))
            compute_camera_trajectory(
                name, trial_idx,
                progress_callback=on_progress,
                cancel_event=cancel_event,
                **kwargs,
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
            logger.exception(f"Camera trajectory job {job_id} failed")
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


@router.get("/{subject_id}/trial/{trial_idx}/trajectory")
def get_trajectory(subject_id: int, trial_idx: int) -> dict:
    """Return a compact summary of the saved trajectory for a trial.

    Returns ``{available: false, ...}`` when no file exists yet.  Per-
    frame translation/rotation arrays are decomposed from the chained
    homographies; the full 3×3 stack stays on disk (too big to send).
    """
    from ..services.camera_motion import load_camera_trajectory, summarise_trajectory

    name = _subject_name(subject_id)
    trials = build_trial_map(name)
    if trial_idx < 0 or trial_idx >= len(trials):
        raise HTTPException(400, f"trial_idx out of range")
    trial_stem = trials[trial_idx]["trial_name"]
    traj = load_camera_trajectory(name, trial_stem)
    if traj is None:
        return {"available": False, "trial_stem": trial_stem}
    summary = summarise_trajectory(traj)
    summary["available"] = True
    summary["trial_stem"] = trial_stem
    return summary


# ── Background extraction ────────────────────────────────────────────────

def _spawn_preproc_job(
    subject_id: int,
    name: str,
    trial_stem: str,
    trial_idx: int,
    job_kind: str,                 # 'stable' | 'background'
    body: dict,
    job_type: str = "background",
) -> dict:
    """Spawn a Stabilize or Background worker thread.

    Two endpoints share this helper: ``/compute_stable`` (just the
    warp pass) and ``/compute_background`` (sample + median + color
    refinement + skin fit, reads stable.mp4).  Hand boundary is
    computed on demand per frame and doesn't route through here.
    """
    from ..services.background import compute_stable, compute_background

    with get_db_ctx() as db:
        db.execute(
            "INSERT INTO jobs (subject_id, job_type, status, params_json) "
            "VALUES (?, ?, 'pending', ?)",
            (subject_id, job_type, json.dumps({
                "trial_idx": trial_idx,
                "trial_name": trial_stem,
                "kind": job_kind,
            })),
        )
        job = db.execute(
            "SELECT * FROM jobs WHERE subject_id = ? ORDER BY id DESC LIMIT 1",
            (subject_id,),
        ).fetchone()
    job_id = job["id"]
    cancel_event = threading.Event()
    registry._cancel_events[job_id] = cancel_event

    if job_kind == "stable":
        worker = compute_stable
        allowed = ()           # no tunable params
        float_params = ()
    elif job_kind == "background":
        worker = compute_background
        allowed = ("max_samples", "downscale", "dilation_px",
                    "palm_grow_px", "color_dilate_px", "skin_leniency")
        float_params = ("skin_leniency",)
    else:
        raise HTTPException(500, f"bad job_kind: {job_kind}")

    kwargs = {}
    for k in allowed:
        v = body.get(k)
        if v is None:
            continue
        kwargs[k] = float(v) if k in float_params else int(v)

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
                               (round(float(pct), 1), job_id))
            worker(
                name, trial_idx,
                progress_callback=on_progress,
                cancel_event=cancel_event,
                **kwargs,
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
            logger.exception(f"{job_kind} job {job_id} failed")
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


@router.post("/{subject_id}/compute_stable")
def compute_stable_endpoint(subject_id: int, body: dict = Body(...)) -> dict:
    """Stage 1: warp every source frame to the reference and bake
    stable.mp4.  No background extraction in this step; run
    ``/compute_background`` afterwards."""
    name = _subject_name(subject_id)
    trial_idx = int(body.get("trial_idx", 0))
    trials = build_trial_map(name)
    if trial_idx < 0 or trial_idx >= len(trials):
        raise HTTPException(400, "trial_idx out of range")
    trial_stem = trials[trial_idx]["trial_name"]
    return _spawn_preproc_job(subject_id, name, trial_stem, trial_idx,
                                "stable", body)


@router.get("/{subject_id}/trial/{trial_idx}/outline_frame")
def get_outline_frame(subject_id: int, trial_idx: int,
                       frame: int, dilation_px: int = 14,
                       open_radius_px: int = 0,
                       include_fg: int = 0) -> dict:
    """On-demand hand boundary for a single frame.

    Replaces the old ``compute_foreground`` bake: the UI calls this as
    the user scrubs frames or moves a slider, and gets back a closed
    contour polygon (per camera in stereo).  Cheap enough to re-fetch
    interactively -- skips the warp pass entirely by reading from
    stable.mp4.

    ``open_radius_px`` clips thin strands off the boundary via a
    morphological open (0 = off).  ``include_fg=1`` also returns a
    JET-colored foreground heatmap PNG cropped to the gate bbox
    (``fg_OS``, ``fg_OD``); off by default since the encoding adds
    ~100 ms per side.
    """
    from ..services.background import compute_outline_frame
    name = _subject_name(subject_id)
    try:
        return compute_outline_frame(name, trial_idx, int(frame),
                                       dilation_px=int(dilation_px),
                                       open_radius_px=int(open_radius_px),
                                       include_fg=bool(int(include_fg)))
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except RuntimeError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/{subject_id}/compute_background")
def compute_background_endpoint(subject_id: int, body: dict = Body(...)) -> dict:
    """Stage 2: read stable.mp4 + camera trajectory + MP keypoints,
    compute the masked-median background, run the color-based
    forearm refinement, fit the skin model, and save
    background.npz.  Requires Stabilize to have produced
    stable.mp4 first."""
    name = _subject_name(subject_id)
    trial_idx = int(body.get("trial_idx", 0))
    trials = build_trial_map(name)
    if trial_idx < 0 or trial_idx >= len(trials):
        raise HTTPException(400, "trial_idx out of range")
    trial_stem = trials[trial_idx]["trial_name"]
    return _spawn_preproc_job(subject_id, name, trial_stem, trial_idx,
                                "background", body)


@router.get("/{subject_id}/trial/{trial_idx}/background")
def get_background(subject_id: int, trial_idx: int) -> dict:
    """Summary of the preproc artifacts for a trial.

    ``available`` is True once background.npz exists (Stage 2 done).
    ``stable_mp4_exists`` is reported independently so the UI can
    tell "Stabilize done, Background not yet" apart from "nothing
    done".
    """
    from ..services.background import (
        load_background, summarise_background, stable_mp4_path)
    from ..services.camera_motion import load_camera_trajectory

    name = _subject_name(subject_id)
    trials = build_trial_map(name)
    if trial_idx < 0 or trial_idx >= len(trials):
        raise HTTPException(400, f"trial_idx out of range")
    trial_stem = trials[trial_idx]["trial_name"]
    stable_exists = stable_mp4_path(name, trial_stem) is not None
    bg = load_background(name, trial_stem)
    if bg is None:
        # background.npz not built yet -- still report is_stereo (from
        # the trajectory) so the UI can crop a stabilized-only overlay
        # to the correct OS/OD half.
        traj = load_camera_trajectory(name, trial_stem)
        is_stereo = bool(traj["is_stereo"]) if traj is not None else None
        return {
            "available": False,
            "trial_stem": trial_stem,
            "stable_mp4_exists": stable_exists,
            "is_stereo": is_stereo,
        }
    summary = summarise_background(name, trial_stem, bg)
    summary["available"] = True
    summary["trial_stem"] = trial_stem
    summary["stable_mp4_exists"] = stable_exists
    return summary


@router.get("/{subject_id}/trial/{trial_idx}/background_image")
def get_background_image(subject_id: int, trial_idx: int,
                          side: str = Query("OS"),
                          kind: str = Query("bg"),
                          ):
    """Serve the background or MAD PNG for one trial.

    ``side`` ∈ {OS, OD}, ``kind`` ∈ {bg, mad}.  The files are saved next
    to ``background.npz`` by ``compute_background``.
    """
    from ..services.background import _preproc_dir, background_exists

    name = _subject_name(subject_id)
    trials = build_trial_map(name)
    if trial_idx < 0 or trial_idx >= len(trials):
        raise HTTPException(400, f"trial_idx out of range")
    trial_stem = trials[trial_idx]["trial_name"]
    side = side.upper()
    if side not in ("OS", "OD"):
        raise HTTPException(400, "side must be OS or OD")
    if kind not in ("bg", "mad"):
        raise HTTPException(400, "kind must be bg or mad")
    if not background_exists(name, trial_stem):
        raise HTTPException(404, "no background — run Compute Background")
    fname = ("background_" if kind == "bg" else "mad_") + side + ".png"
    fpath = _preproc_dir(name, trial_stem) / fname
    if not fpath.exists():
        raise HTTPException(404, f"image not found: {fname}")
    return FileResponse(str(fpath), media_type="image/png")


@router.get("/{subject_id}/trial/{trial_idx}/stable_video")
def get_stable_video(subject_id: int, trial_idx: int):
    """Serve the stabilized mp4 — every frame warped into reference
    coords using the camera trajectory.  Same resolution + fps as the
    source so MP/HRnet/DLC can consume it transparently."""
    from ..services.background import stable_mp4_path

    name = _subject_name(subject_id)
    trials = build_trial_map(name)
    if trial_idx < 0 or trial_idx >= len(trials):
        raise HTTPException(400, f"trial_idx out of range")
    trial_stem = trials[trial_idx]["trial_name"]
    p = stable_mp4_path(name, trial_stem)
    if p is None:
        raise HTTPException(404, "no stable.mp4 — run Compute Background")
    return FileResponse(str(p), media_type="video/mp4")


@router.get("/{subject_id}/trial/{trial_idx}/mp_keypoints")
def get_mp_keypoints(subject_id: int, trial_idx: int) -> dict:
    """Per-frame MP hand keypoints for this trial -- both raw (frame
    pixel coords) and warped into the reference camera's coordinate
    system using the camera trajectory.  Used by the preproc page to
    draw a dilated-skeleton preview on the canvas.

    Returns ``{n_frames, raw_OS, raw_OD, ref_OS, ref_OD}``.  Each array
    has shape ``(N, 21, 2)`` serialised as nested lists, with ``null``
    in place of NaN.  Missing if MediaPipe prelabels aren't available.
    """
    import numpy as np
    from ..services.mediapipe_prelabel import load_mediapipe_prelabels
    from ..services.camera_motion import load_camera_trajectory

    name = _subject_name(subject_id)
    trials = build_trial_map(name)
    if trial_idx < 0 or trial_idx >= len(trials):
        raise HTTPException(400, f"trial_idx out of range")
    trial = trials[trial_idx]
    trial_stem = trial["trial_name"]
    start_frame = int(trial.get("start_frame", 0))
    n_frames = int(trial["frame_count"])

    mp = load_mediapipe_prelabels(name)
    if mp is None or mp.get("OS_landmarks") is None:
        return {"available": False, "reason": "no MP prelabels"}

    os_lm = mp["OS_landmarks"]
    od_lm = mp.get("OD_landmarks")
    end = min(start_frame + n_frames, os_lm.shape[0])
    raw_os = os_lm[start_frame:end].astype(float)
    raw_od = (od_lm[start_frame:end].astype(float)
              if od_lm is not None else None)

    # Apply the saved trajectory (if any) to land the keypoints in ref
    # coords -- so the client can overlay the dilated skeleton on the
    # stabilized view without re-doing the warp math.
    traj = load_camera_trajectory(name, trial_stem)
    ref_os = None; ref_od = None
    if traj is not None:
        H_L = traj["H_to_ref_L"]
        H_R = traj.get("H_to_ref_R")

        def _warp(lm, H):
            if lm is None or H is None:
                return None
            out = np.full_like(lm, np.nan)
            N = min(lm.shape[0], H.shape[0])
            for f in range(N):
                pts = lm[f]                                # (21, 2)
                valid = ~np.isnan(pts).any(axis=-1)
                if not valid.any():
                    continue
                pts_h = np.column_stack([pts[valid], np.ones(int(valid.sum()))]).T   # (3, V)
                w = H[f].astype(np.float64) @ pts_h
                zs = w[2]
                ok = np.abs(zs) > 1e-9
                xy = np.full((int(valid.sum()), 2), np.nan)
                xy[ok] = (w[:2, ok] / zs[ok]).T
                out_frame = out[f]
                out_frame[valid] = xy
                out[f] = out_frame
            return out

        ref_os = _warp(raw_os, H_L)
        ref_od = _warp(raw_od, H_R)

    def _to_json(arr):
        if arr is None:
            return None
        a = np.where(np.isnan(arr), None, np.round(arr, 1))
        # nested-list with None in place of NaN
        return a.tolist()

    return {
        "available": True,
        "n_frames":  int(end - start_frame),
        "raw_OS":    _to_json(raw_os),
        "raw_OD":    _to_json(raw_od),
        "ref_OS":    _to_json(ref_os),
        "ref_OD":    _to_json(ref_od),
    }


@router.get("/{subject_id}/trial/{trial_idx}/warp_at_frame")
def get_warp_at_frame(subject_id: int, trial_idx: int,
                       frame: int = Query(...),
                       side: str = Query("OS"),
                       ) -> dict:
    """Return the 3×3 homography that warps the given frame's pixels
    into the reference frame.

    Used by the canvas overlay to render a "stabilized view" — multiply
    the canvas transform by this matrix and the frame snaps into the
    reference's coordinate system.
    """
    from ..services.camera_motion import load_camera_trajectory
    name = _subject_name(subject_id)
    trials = build_trial_map(name)
    if trial_idx < 0 or trial_idx >= len(trials):
        raise HTTPException(400, f"trial_idx out of range")
    trial_stem = trials[trial_idx]["trial_name"]
    traj = load_camera_trajectory(name, trial_stem)
    if traj is None:
        raise HTTPException(404, "no trajectory yet — run Compute Trajectory")
    H_stack = traj["H_to_ref_L"] if side.upper() == "OS" else traj["H_to_ref_R"]
    if frame < 0 or frame >= H_stack.shape[0]:
        raise HTTPException(400, f"frame {frame} out of range (0..{H_stack.shape[0]-1})")
    H = H_stack[frame]
    return {
        "frame": int(frame),
        "side": side.upper(),
        "H": [[float(x) for x in row] for row in H.tolist()],
        "reference_frame": int(traj["reference_frame"]),
    }
