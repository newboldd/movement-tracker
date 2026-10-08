"""Labeling session endpoints: frames, layers, label CRUD, commit.

One page talks to this router.  It serves video frames and whole trial
videos, every label layer the page can display (MediaPipe passes, DLC
predictions, corrections, committed manual labels), and the write paths:
save labels, save corrections, commit to the DLC training set.
"""
from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import List

import numpy as np
from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import FileResponse, Response

from ..config import get_settings
from ..db import get_db_ctx
from ..models import (
    CommitRequest, LabelBatchSave, SessionCreate, STAGE_INDEX,
)
from ..services.calibration import get_calibration_for_subject, triangulate_points
from ..services.discovery import _count_labeled_frames, _has_mediapipe
from ..services.dlc_predictions import (
    _compute_dlc_distances,
    get_corrections_with_dlc_fallback,
    get_dlc_predictions_for_session,
    get_dlc_predictions_for_stage,
    get_stage_csv_files,
)
from ..services.labels import commit_labels_to_dlc, save_corrections_to_csv
from ..services.mediapipe_prelabel import (
    compute_default_bbox,
    get_mediapipe_for_session,
    recompute_distance_for_frame,
    run_mediapipe_cropped,
)
from ..services.video import (
    _deidentified_path, _get_no_face_videos, build_trial_map, extract_frame,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/labeling", tags=["labeling"])

# The crop box every consumer agrees on.  One box per (subject, trial,
# camera): what you drag on the page is exactly what the next MediaPipe
# run crops to.  The table keeps a model_name column from the upstream
# app's multi-detector days; pinning it to one value here is what makes
# "the box I drew" and "the box MediaPipe used" the same box.
BBOX_MODEL = "run-mediapipe"

# MediaPipe passes the page can display, in the order they are offered.
# Each is a separate npz under <dlc>/<subject>/<trial>/ because each
# wins on different frames — see services/mediapipe_prelabel.
MP_PASSES = [
    {"key": "forward", "file": "mediapipe.npz", "label": "Forward",
     "color": "#00cccc"},
    {"key": "reverse", "file": "mediapipe_reverse.npz", "label": "Reverse",
     "color": "#e040fb"},
    {"key": "static", "file": "mediapipe_static.npz", "label": "Frame-by-frame",
     "color": "#26c6da"},
    {"key": "cropped", "file": "mediapipe_cropped.npz", "label": "Cropped",
     "color": "#7cb342"},
    {"key": "best", "file": "mediapipe_combined.npz", "label": "Best per frame",
     "color": "#ffa726"},
]

# Which pass contributed each frame of the "best per frame" fusion.
# Mirrors COMBINED_SRC_* in services/mediapipe_prelabel.
MP_SOURCE_NAMES = {0: None, 1: "forward", 2: "cropped", 3: "reverse", 4: "static"}


# ── Encoding helpers ───────────────────────────────────────────────────

def _f32(arr: np.ndarray | None) -> dict | None:
    """Pack a float array as base64 little-endian float32 plus its shape.

    A trial's 21 joints across ~1200 frames in 2D and 3D is ~400k
    numbers.  As JSON that is several megabytes of text to generate,
    transfer and parse on every trial switch; as base64 float32 it is
    under a megabyte and decodes in one pass in the browser.  NaN
    survives the round trip, which matters — a missing detection has to
    stay missing rather than become a zero at the origin.
    """
    if arr is None:
        return None
    a = np.ascontiguousarray(np.asarray(arr, dtype="<f4"))
    return {"shape": list(a.shape),
            "b64": base64.b64encode(a.tobytes()).decode("ascii")}


def _nan_list(arr: np.ndarray | None, ndigits: int = 2) -> list | None:
    """Convert a 1-D array to a JSON list with NaN as null."""
    if arr is None:
        return None
    return [None if not np.isfinite(v) else round(float(v), ndigits)
            for v in np.asarray(arr, dtype=float)]


# ── Session lookup helpers ─────────────────────────────────────────────

def _session_and_subject(session_id: int) -> tuple[dict, dict]:
    with get_db_ctx() as db:
        session = db.execute("SELECT * FROM label_sessions WHERE id = ?",
                             (session_id,)).fetchone()
        if not session:
            raise HTTPException(404, "Session not found")
        subj = db.execute("SELECT * FROM subjects WHERE id = ?",
                          (session["subject_id"],)).fetchone()
        if not subj:
            raise HTTPException(404, "Subject not found")
    return session, subj


def _camera_mode(subj: dict) -> str:
    return subj.get("camera_mode") or get_settings().default_camera_mode


# ── Sessions ───────────────────────────────────────────────────────────

@router.post("/{subject_id}/sessions", status_code=201)
def create_session(subject_id: int, req: SessionCreate) -> dict:
    """Create a labeling session, or resume the active one of that type.

    Resuming rather than creating is the point: a student who closes the
    tab and comes back gets their work, and we don't accumulate empty
    sessions that split one person's labels across several rows.

    - ``initial``     places labels on raw video; inherits the previous
                      committed session's labels so a second round starts
                      from the first rather than from nothing.
    - ``corrections`` edits model predictions; starts empty, with the
                      predictions shown underneath as ghosts.
    - ``refine``      promotes corrections into a new training round, so
                      it bumps the subject's iteration.
    """
    with get_db_ctx() as db:
        subj = db.execute("SELECT * FROM subjects WHERE id = ?",
                          (subject_id,)).fetchone()
        if not subj:
            raise HTTPException(404, "Subject not found")

        # Prefer the session with the most labels: if a blank session was
        # created by accident alongside real work, return the real work.
        existing = db.execute(
            """SELECT ls.*, COUNT(fl.id) AS label_count
               FROM label_sessions ls
               LEFT JOIN frame_labels fl ON fl.session_id = ls.id
               WHERE ls.subject_id = ? AND ls.session_type = ?
                 AND ls.status = 'active'
               GROUP BY ls.id
               ORDER BY label_count DESC, ls.id DESC
               LIMIT 1""",
            (subject_id, req.session_type),
        ).fetchone()
        if existing:
            result = dict(existing)
            result.pop("label_count", None)
            return result

        is_refine = req.session_type == "refine"
        is_corrections = req.session_type == "corrections"
        iteration = subj["iteration"] + 1 if is_refine else subj["iteration"]

        cur = db.execute(
            """INSERT INTO label_sessions (subject_id, iteration, session_type)
               VALUES (?, ?, ?)""",
            (subject_id, iteration, req.session_type),
        )
        session_id = cur.lastrowid
        session = db.execute("SELECT * FROM label_sessions WHERE id = ?",
                             (session_id,)).fetchone()

        if not is_refine and not is_corrections:
            prev = db.execute(
                """SELECT id FROM label_sessions
                   WHERE subject_id = ? AND status = 'committed'
                   ORDER BY committed_at DESC LIMIT 1""",
                (subject_id,),
            ).fetchone()
            if prev:
                db.execute(
                    """INSERT INTO frame_labels
                       (session_id, frame_num, trial_idx, side, keypoints, updated_at)
                       SELECT ?, frame_num, trial_idx, side, keypoints,
                              CURRENT_TIMESTAMP
                       FROM frame_labels WHERE session_id = ?""",
                    (session_id, prev["id"]),
                )

            if STAGE_INDEX.get(subj.get("stage", "created"), 0) \
                    < STAGE_INDEX["labeling"]:
                db.execute(
                    "UPDATE subjects SET stage = 'labeling', "
                    "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (subject_id,))

    return session


@router.get("/sessions/{session_id}/info")
def get_session_info(session_id: int) -> dict:
    """Everything the page needs once, at load: trials, bodyparts, cameras."""
    settings = get_settings()
    session, subj = _session_and_subject(session_id)

    camera_mode = _camera_mode(subj)
    trials = build_trial_map(subj["name"], camera_mode=camera_mode)
    total_frames = trials[-1]["end_frame"] + 1 if trials else 0

    # A package carries the bodyparts it was made for.  Whoever labels it
    # may have different ones configured, and labeling the wrong points
    # into someone else's project is not a mistake you find out about
    # until you try to import the result — so the package wins.
    from ..services.labelpack import read_manifest
    package = None
    if trials and trials[0].get("kind") == "frames":
        package = read_manifest(settings.packages_path / subj["name"])
    bodyparts = (package or {}).get("bodyparts") or settings.bodyparts

    trial_info = []
    for t in trials:
        entry = {
            "trial_name": t["trial_name"],
            "start_frame": t["start_frame"],
            "end_frame": t["end_frame"],
            "frame_count": t["frame_count"],
            "fps": t["fps"],
            "frame_offset": t.get("frame_offset", 0),
        }
        if t.get("kind") == "frames":
            # Tells the page not to reach for a video it will not get:
            # this trial is a folder of images.
            entry["kind"] = "frames"
        cameras = t.get("cameras", [])
        if len(cameras) > 1:
            entry["cameras"] = [{"name": c["name"], "idx": c["idx"]}
                                for c in cameras]
        trial_info.append(entry)

    committed_frame_count = 0
    dlc_path = settings.dlc_path / (subj.get("dlc_dir") or subj["name"])
    if dlc_path.exists():
        committed_frame_count = _count_labeled_frames(dlc_path)

    with get_db_ctx() as db:
        rows = db.execute(
            "SELECT trial_idx, camera_name, x1, y1, x2, y2 FROM mp_crop_boxes "
            "WHERE subject_id = ? AND model_name = ?",
            (subj["id"], BBOX_MODEL),
        ).fetchall()
    crop_boxes: dict[int, dict] = {}
    for r in rows:
        crop_boxes.setdefault(r["trial_idx"], {})[r["camera_name"]] = {
            "x1": r["x1"], "y1": r["y1"], "x2": r["x2"], "y2": r["y2"],
        }

    return {
        "session": session,
        "subject": subj,
        "trials": trial_info,
        "total_frames": total_frames,
        "bodyparts": bodyparts,
        "camera_names": settings.camera_names,
        "camera_mode": camera_mode,
        "committed_frame_count": committed_frame_count,
        "crop_boxes": crop_boxes,
        "has_calibration": get_calibration_for_subject(subj["name"]) is not None,
        "mp_passes": MP_PASSES,
        "package": package,
    }


# ── Frames and video ───────────────────────────────────────────────────

@router.get("/sessions/{session_id}/frame")
def get_frame(
    session_id: int,
    n: int = Query(..., description="Global frame number"),
    side: str = Query(..., description="Camera name"),
) -> Response:
    """Serve one video frame as JPEG.

    Used for precise single-frame work.  Playback goes through
    /video instead, which lets the browser decode the stream.
    """
    settings = get_settings()
    _session, subj = _session_and_subject(session_id)

    camera_mode = _camera_mode(subj)
    # multicam subjects name their cameras per trial, not globally.
    if camera_mode != "multicam" and side not in settings.camera_names:
        raise HTTPException(400, f"side must be one of {settings.camera_names}")

    try:
        jpeg_bytes = extract_frame(subj["name"], n, side, camera_mode=camera_mode)
    except Exception as e:
        raise HTTPException(400, str(e))

    return Response(content=jpeg_bytes, media_type="image/jpeg")


@router.get("/sessions/{session_id}/video")
def get_video(
    session_id: int,
    trial: int = Query(0, description="Trial index"),
    side: str = Query("", description="Camera name for multicam subjects"),
) -> FileResponse:
    """Stream a trial's video file so the browser can play it smoothly.

    Playing the real file is what makes scrubbing and playback feel
    immediate; re-encoding frames server-side cannot keep up at 60 fps.
    """
    _session, subj = _session_and_subject(session_id)

    camera_mode = _camera_mode(subj)
    trials = build_trial_map(subj["name"], camera_mode=camera_mode)
    if trial < 0 or trial >= len(trials):
        raise HTTPException(400,
                            f"Trial index {trial} out of range (0-{len(trials) - 1})")

    if trials[trial].get("kind") == "frames":
        raise HTTPException(
            404, "This trial is a folder of frames, not a video — the page "
                 "draws it from /frame instead.")

    video_path = trials[trial]["video_path"]
    if camera_mode == "multicam" and side:
        for cam in trials[trial].get("cameras", []):
            if cam["name"] == side:
                video_path = cam["path"]
                break

    # Prefer the face-blurred render when one exists, so a student
    # labeling someone else's patient video never sees their face.
    settings = get_settings()
    if settings.prefer_deidentified:
        stem = Path(video_path).stem
        if stem not in _get_no_face_videos(subj["name"]):
            deident = _deidentified_path(video_path)
            if deident:
                video_path = deident

    return FileResponse(video_path, media_type="video/mp4",
                        headers={"Cache-Control": "no-store"})


# ── Label layers ───────────────────────────────────────────────────────

@router.get("/sessions/{session_id}/layers")
def get_layers(session_id: int) -> dict:
    """Report which label layers actually have data for this subject.

    The page builds its layer list from this, so a student is never
    offered a checkbox that turns on nothing.
    """
    settings = get_settings()
    _session, subj = _session_and_subject(session_id)
    subject_name = subj["name"]
    dlc_path = settings.dlc_path / subject_name

    trials = build_trial_map(subject_name, camera_mode=_camera_mode(subj))

    # MediaPipe passes are per trial, so report per trial: a subject can
    # have the reverse pass on one trial and not another.
    mp_by_trial: dict[int, list[str]] = {}
    for ti, t in enumerate(trials):
        trial_dir = dlc_path / t["trial_name"]
        present = [p["key"] for p in MP_PASSES
                   if (trial_dir / p["file"]).exists()]
        if present:
            mp_by_trial[ti] = present
    # Fall back to the subject-wide npz written by older runs.
    if not mp_by_trial and _has_mediapipe(dlc_path):
        mp_by_trial = {ti: ["forward"] for ti in range(len(trials))}

    with get_db_ctx() as db:
        committed = db.execute(
            """SELECT COUNT(*) AS cnt FROM frame_labels fl
               JOIN label_sessions ls ON ls.id = fl.session_id
               WHERE ls.subject_id = ? AND ls.status = 'committed'
                 AND ls.session_type != 'corrections'""",
            (subj["id"],),
        ).fetchone()

    stage_files = {}
    stages = []
    for stage in ("dlc", "refine", "corrections"):
        csv_files = get_stage_csv_files(subject_name, stage)
        if csv_files:
            stages.append(stage)
            stage_files[stage] = csv_files

    return {
        "mp_passes": MP_PASSES,
        "mp_by_trial": mp_by_trial,
        "stages": stages,
        "stage_files": stage_files,
        "has_committed_labels": bool(committed and committed["cnt"] > 0),
    }


@router.get("/sessions/{session_id}/trial/{trial_idx}/mp")
def get_trial_mp(session_id: int, trial_idx: int,
                 passes: str = Query("", description="Comma-separated pass keys")
                 ) -> dict:
    """Full 21-joint MediaPipe landmarks for one trial, in 2D and 3D.

    One payload per trial rather than per frame: the page needs the
    whole trial to draw motion context and to keep frame stepping
    instant, and triangulating the trial once is far cheaper than
    triangulating the current frame on every arrow-key press.

    Landmarks are the hand's full 21 joints, not just the labeled
    bodyparts — that is what makes the 3D view readable as a hand and
    what the default zoom is computed from.
    """
    settings = get_settings()
    _session, subj = _session_and_subject(session_id)
    subject_name = subj["name"]

    trials = build_trial_map(subject_name, camera_mode=_camera_mode(subj))
    if trial_idx < 0 or trial_idx >= len(trials):
        raise HTTPException(404, f"Trial index {trial_idx} out of range")
    trial = trials[trial_idx]
    trial_dir = settings.dlc_path / subject_name / trial["trial_name"]

    wanted = {p.strip() for p in passes.split(",") if p.strip()} or None
    calib = get_calibration_for_subject(subject_name)

    out: dict = {
        "trial_idx": trial_idx,
        "trial_name": trial["trial_name"],
        "start_frame": trial["start_frame"],
        "frame_count": trial["frame_count"],
        "has_calibration": calib is not None,
        "passes": {},
    }

    for spec in MP_PASSES:
        key = spec["key"]
        if wanted is not None and key not in wanted:
            continue
        npz_path = trial_dir / spec["file"]
        if not npz_path.exists():
            continue
        try:
            data = np.load(str(npz_path))
        except (OSError, ValueError) as e:
            logger.warning("Could not read %s: %s", npz_path, e)
            continue

        os_lm = data["OS_landmarks"] if "OS_landmarks" in data.files else None
        od_lm = data["OD_landmarks"] if "OD_landmarks" in data.files else None
        if os_lm is None and od_lm is None:
            continue

        entry: dict = {
            "label": spec["label"],
            "color": spec["color"],
            "OS": _f32(os_lm),
            "OD": _f32(od_lm),
        }

        # 3D for the whole trial, in one batched triangulation.
        if calib is not None and os_lm is not None and od_lm is not None:
            n, j = os_lm.shape[0], os_lm.shape[1]
            try:
                pts3d = triangulate_points(os_lm.reshape(-1, 2),
                                           od_lm.reshape(-1, 2), calib)
                entry["joints_3d"] = _f32(pts3d.reshape(n, j, 3))
            except Exception as e:
                logger.warning("Triangulation failed for %s/%s: %s",
                               subject_name, spec["file"], e)

        if "distances" in data.files:
            entry["distances"] = _nan_list(data["distances"])
        if "distances_clean" in data.files:
            entry["distances_clean"] = _nan_list(data["distances_clean"])
        # Which pass won each frame of the fusion — shown in the UI so a
        # surprising "best" frame can be traced back to its source.
        for side_key, npz_key in (("source_OS", "source_OS"),
                                  ("source_OD", "source_OD")):
            if npz_key in data.files:
                entry[side_key] = [MP_SOURCE_NAMES.get(int(v))
                                   for v in data[npz_key]]

        out["passes"][key] = entry

    return out


@router.get("/sessions/{session_id}/stage")
def get_stage_data(
    session_id: int,
    stage: str = Query(..., description="mp | labels | dlc | refine | corrections"),
) -> dict:
    """Labeled-bodypart coordinates for one layer, over the whole subject.

    Shape matches across layers — ``{camera: {bodypart: [[x,y]|null,
    ...]}, distances: [...], joints_3d: {...}}`` — so the page draws any
    of them with the same code.
    """
    valid = ("mp", "labels", "dlc", "refine", "corrections")
    if stage not in valid:
        raise HTTPException(400, f"stage must be one of {valid}")

    _session, subj = _session_and_subject(session_id)
    subject_name = subj["name"]

    if stage == "mp":
        # The fused "best per frame" pass when it exists, else forward.
        data = get_mediapipe_for_session(subject_name, prefer_combined=True)
    elif stage == "labels":
        data = _committed_labels_to_arrays(subj)
    elif stage == "corrections":
        # Corrections where they exist, predictions for untouched trials,
        # so the layer always covers the whole subject.
        data = get_corrections_with_dlc_fallback(subject_name)
    else:
        data = get_dlc_predictions_for_stage(subject_name, stage)

    if not data:
        return {}

    data["joints_3d"] = _bodypart_3d(subject_name, data)
    return data


@router.get("/sessions/{session_id}/dlc_predictions")
def get_dlc_predictions(session_id: int) -> dict:
    """Best available predictions: corrections > labels_v2 > labels_v1."""
    _session, subj = _session_and_subject(session_id)
    data = get_dlc_predictions_for_session(subj["name"])
    if not data:
        return {}
    data["joints_3d"] = _bodypart_3d(subj["name"], data)
    return data


@router.get("/sessions/{session_id}/committed_labels")
def get_committed_labels(session_id: int) -> List[dict]:
    """Manual labels from this subject's committed sessions.

    Shown underneath the current session so hand-placed labels from
    earlier rounds stay visible and are never silently overwritten by a
    model prediction.
    """
    session, _subj = _session_and_subject(session_id)

    with get_db_ctx() as db:
        labels = db.execute(
            """SELECT fl.frame_num, fl.trial_idx, fl.side, fl.keypoints
               FROM frame_labels fl
               JOIN label_sessions ls ON ls.id = fl.session_id
               WHERE ls.subject_id = ? AND ls.status = 'committed'
               ORDER BY ls.committed_at DESC, fl.frame_num""",
            (session["subject_id"],),
        ).fetchall()

    seen = set()
    result = []
    for lbl in labels:
        key = (lbl["frame_num"], lbl["side"])
        if key in seen:
            continue
        seen.add(key)
        if isinstance(lbl["keypoints"], str):
            lbl["keypoints"] = json.loads(lbl["keypoints"])
        result.append(lbl)
    return result


def _bodypart_3d(subject_name: str, data: dict) -> dict | None:
    """Triangulate a layer's labeled bodyparts to 3D.

    Returns ``{bodypart: [[x,y,z]|null, ...]}`` in mm, or None when the
    subject has no stereo calibration or only one camera's worth of
    data — in which case the page shows 2D only rather than inventing
    depth.
    """
    settings = get_settings()
    cam_names = settings.camera_names
    if len(cam_names) < 2:
        return None

    calib = get_calibration_for_subject(subject_name)
    if calib is None:
        return None

    left = data.get(cam_names[0])
    right = data.get(cam_names[1])
    if not left or not right:
        return None

    out: dict[str, list] = {}
    for bp in settings.bodyparts:
        lseq, rseq = left.get(bp), right.get(bp)
        if not lseq or not rseq:
            continue
        n = min(len(lseq), len(rseq))
        pts_l = np.full((n, 2), np.nan)
        pts_r = np.full((n, 2), np.nan)
        for i in range(n):
            if lseq[i] is not None:
                pts_l[i] = lseq[i][:2]
            if rseq[i] is not None:
                pts_r[i] = rseq[i][:2]
        try:
            pts3d = triangulate_points(pts_l, pts_r, calib)
        except Exception as e:
            logger.warning("Triangulation failed for %s/%s: %s",
                           subject_name, bp, e)
            continue
        out[bp] = [None if not np.isfinite(p).all()
                   else [round(float(p[0]), 2), round(float(p[1]), 2),
                         round(float(p[2]), 2)]
                   for p in pts3d]

    return out or None


def _committed_labels_to_arrays(subj: dict) -> dict | None:
    """Committed manual labels in the same array shape as the other layers."""
    settings = get_settings()
    subject_name = subj["name"]
    cam_names = settings.camera_names

    trials = build_trial_map(subject_name)
    if not trials:
        return None
    total_frames = trials[-1]["end_frame"] + 1

    with get_db_ctx() as db:
        rows = db.execute(
            """SELECT fl.frame_num, fl.side, fl.keypoints
               FROM frame_labels fl
               JOIN label_sessions ls ON ls.id = fl.session_id
               WHERE ls.subject_id = ? AND ls.status = 'committed'
                 AND ls.session_type != 'corrections'
               ORDER BY ls.committed_at DESC""",
            (subj["id"],),
        ).fetchall()
    if not rows:
        return None

    result: dict = {cam: {bp: [None] * total_frames
                          for bp in settings.bodyparts}
                    for cam in cam_names}

    seen = set()
    for row in rows:
        key = (row["frame_num"], row["side"])
        if key in seen:
            continue
        seen.add(key)
        if row["side"] not in cam_names or row["frame_num"] >= total_frames:
            continue
        kp = row["keypoints"]
        if isinstance(kp, str):
            kp = json.loads(kp)
        for bp in settings.bodyparts:
            coords = kp.get(bp)
            if coords and coords[0] is not None:
                result[row["side"]][bp][row["frame_num"]] = coords

    has_data = any(c is not None
                   for cam in cam_names
                   for bp in settings.bodyparts
                   for c in result[cam][bp])
    if not has_data:
        return None

    distances = _compute_dlc_distances(result, cam_names, subject_name,
                                       total_frames)
    if distances is not None:
        result["distances"] = distances
    return result


# ── Label CRUD ─────────────────────────────────────────────────────────

@router.get("/sessions/{session_id}/labels")
def get_labels(session_id: int) -> List[dict]:
    """All labels placed in this session."""
    with get_db_ctx() as db:
        labels = db.execute(
            """SELECT frame_num, trial_idx, side, keypoints
               FROM frame_labels WHERE session_id = ? ORDER BY frame_num""",
            (session_id,),
        ).fetchall()
    for lbl in labels:
        if isinstance(lbl["keypoints"], str):
            lbl["keypoints"] = json.loads(lbl["keypoints"])
    return labels


def _mirror_labels_into_package(subject_name: str) -> None:
    """Write a package subject's labels into its own folder, best-effort.

    A failure here must never cost someone their labels — those are
    already committed to the database by the time this runs — so it is
    logged and swallowed.
    """
    from ..services.labelpack import write_labels_into_package
    try:
        write_labels_into_package(subject_name)
    except Exception:
        logger.warning("Could not write labels into the package for %s",
                       subject_name, exc_info=True)


@router.put("/sessions/{session_id}/labels")
def save_labels(session_id: int, req: LabelBatchSave) -> dict:
    """Batch-upsert labels and return the frames' recomputed 3D distances.

    Returning the distances keeps the trace under the video honest the
    moment a point moves, without the page having to refetch the layer.
    """
    session, subj = _session_and_subject(session_id)

    with get_db_ctx() as db:
        for label in req.labels:
            db.execute(
                """INSERT INTO frame_labels
                   (session_id, frame_num, trial_idx, side, keypoints, updated_at)
                   VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                   ON CONFLICT(session_id, frame_num, trial_idx, side)
                   DO UPDATE SET keypoints = excluded.keypoints,
                                 updated_at = CURRENT_TIMESTAMP""",
                (session_id, label.frame_num, label.trial_idx, label.side,
                 json.dumps(label.keypoints)),
            )

    # A package has to stay complete as a folder: the labels belong in it,
    # not only in a database that is not going to be posted back.
    _mirror_labels_into_package(subj["name"])

    affected = {label.frame_num for label in req.labels}
    if not affected:
        return {"saved": 0, "updated_distances": {}}

    # Pull every saved side for the affected frames: triangulation needs
    # both cameras, and the edit may have touched only one of them.
    frame_labels: dict[int, dict] = {}
    with get_db_ctx() as db:
        for frame_num in affected:
            saved = db.execute(
                "SELECT side, keypoints FROM frame_labels "
                "WHERE session_id = ? AND frame_num = ?",
                (session_id, frame_num),
            ).fetchall()
            sides = {}
            for row in saved:
                kp = row["keypoints"]
                sides[row["side"]] = json.loads(kp) if isinstance(kp, str) else kp
            frame_labels[frame_num] = sides

    # For the camera the user did not touch, use the same coordinates the
    # displayed layer uses, so the number under the video matches what is
    # drawn on it.
    stage_data = None
    if session["session_type"] in ("corrections", "refine", "initial"):
        stage_data = get_corrections_with_dlc_fallback(subj["name"])

    updated_distances = {}
    for frame_num, sides in frame_labels.items():
        dist = recompute_distance_for_frame(subj["name"], frame_num, sides,
                                            stage_data=stage_data)
        if dist is not None:
            updated_distances[str(frame_num)] = dist

    return {"saved": len(req.labels), "updated_distances": updated_distances}


@router.delete("/sessions/{session_id}/labels/{frame_num}")
def delete_label(session_id: int, frame_num: int,
                 side: str = Query(...)) -> dict:
    """Delete this session's labels for one frame and camera."""
    _session, subj = _session_and_subject(session_id)
    with get_db_ctx() as db:
        db.execute(
            "DELETE FROM frame_labels "
            "WHERE session_id = ? AND frame_num = ? AND side = ?",
            (session_id, frame_num, side),
        )
    _mirror_labels_into_package(subj["name"])
    return {"deleted": True}


@router.post("/sessions/{session_id}/triangulate")
def triangulate_frame(session_id: int, body: dict = Body(...)) -> dict:
    """Triangulate one frame's 2D points to 3D.

    Body: ``{"points": {bodypart: {camera: [x, y], ...}, ...}}``

    The 3D view needs the points you are dragging right now, not the ones
    last written to disk.  Doing it here rather than in the browser keeps
    one implementation of undistort + triangulate — the same one that
    produced every stored 3D coordinate — so the live marker and the
    saved layer cannot disagree about where a point is.
    """
    _session, subj = _session_and_subject(session_id)
    settings = get_settings()
    cam_names = settings.camera_names
    if len(cam_names) < 2:
        return {"points": {}, "has_calibration": False}

    calib = get_calibration_for_subject(subj["name"])
    if calib is None:
        return {"points": {}, "has_calibration": False}

    points = body.get("points") or {}
    names, pts_l, pts_r = [], [], []
    for bp, cams in points.items():
        left, right = cams.get(cam_names[0]), cams.get(cam_names[1])
        if not left or not right or left[0] is None or right[0] is None:
            continue
        names.append(bp)
        pts_l.append([float(left[0]), float(left[1])])
        pts_r.append([float(right[0]), float(right[1])])

    if not names:
        return {"points": {}, "has_calibration": True}

    pts3d = triangulate_points(np.array(pts_l), np.array(pts_r), calib)
    out = {}
    for bp, p in zip(names, pts3d):
        out[bp] = (None if not np.isfinite(p).all()
                   else [round(float(p[0]), 2), round(float(p[1]), 2),
                         round(float(p[2]), 2)])

    result = {"points": out, "has_calibration": True}
    # The aperture is the measurement the whole pipeline exists to produce,
    # so hand it back alongside the points rather than making the page
    # recompute it from them.
    finite = [v for v in out.values() if v is not None]
    if len(finite) == 2:
        a, b = np.array(finite[0]), np.array(finite[1])
        result["distance"] = round(float(np.linalg.norm(a - b)), 2)
    return result


# ── Commit and corrections ─────────────────────────────────────────────

@router.post("/sessions/{session_id}/save_corrections")
def save_corrections(session_id: int) -> dict:
    """Write this session's edits to the corrections CSVs.

    Separate from commit so edits can be saved mid-session without
    touching the DLC training set — the common case while reviewing a
    long trial.
    """
    session, subj = _session_and_subject(session_id)

    with get_db_ctx() as db:
        labels = db.execute(
            """SELECT frame_num, trial_idx, side, keypoints
               FROM frame_labels WHERE session_id = ?""",
            (session_id,),
        ).fetchall()

    if not labels:
        raise HTTPException(400, "No labels to save")

    return save_corrections_to_csv(subject_name=subj["name"],
                                   session_labels=labels)


@router.post("/sessions/{session_id}/commit")
def commit_session(session_id: int,
                   req: CommitRequest = Body(default=None)) -> dict:
    """Commit the session into the subject's DLC project.

    What that means depends on the session type:

    - ``initial``: every labeled frame becomes a training image —
      extracted as a PNG into ``labeled-data/round{iteration}/`` with a
      CollectedData CSV and H5 beside it.  That is the directory DLC's
      ``create_training_dataset`` reads.
    - ``corrections``: edits are written to ``corrections/`` as DLC-format
      CSVs.  No training data changes; this is the reviewed output.
    - ``refine``: the corrections are saved, and the frames named in
      ``train_frames`` are added to the training set as a new round.
      Only the chosen frames, because adding frames the model already
      gets right costs training time and teaches it nothing.
    """
    session, subj = _session_and_subject(session_id)

    with get_db_ctx() as db:
        labels = db.execute(
            """SELECT frame_num, trial_idx, side, keypoints
               FROM frame_labels WHERE session_id = ?""",
            (session_id,),
        ).fetchall()

    session_type = session["session_type"]
    train_frames = list(req.train_frames) if req is not None else []
    is_refine = session_type == "refine" or (
        session_type == "corrections" and bool(train_frames))

    if not labels and not is_refine:
        raise HTTPException(400, "No labels to commit")

    # Pure corrections: write the reviewed CSVs and stop.
    if session_type == "corrections" and not is_refine:
        result = save_corrections_to_csv(subject_name=subj["name"],
                                         session_labels=labels)
        with get_db_ctx() as db:
            db.execute(
                "UPDATE label_sessions SET status = 'committed', "
                "committed_at = CURRENT_TIMESTAMP WHERE id = ?", (session_id,))
            db.execute(
                "UPDATE subjects SET stage = 'corrected', "
                "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (session["subject_id"],))
        return result

    if is_refine:
        if not train_frames:
            raise HTTPException(400, "No training frames selected")

        # Persist the edits first, so the corrections CSVs are the
        # source of truth for what goes into training.
        if labels:
            save_corrections_to_csv(subject_name=subj["name"],
                                    session_labels=labels)

        training_labels = _training_labels_from_corrections(
            subj, train_frames)
        if not training_labels:
            raise HTTPException(
                400, "No usable labels found in the corrections for the "
                     "selected frames")

        # A refinement always goes into a NEW labeled-data round.
        # commit_labels_to_dlc clears the round directory it writes to, so
        # reusing the current round would delete the frames the first model
        # was trained on.  A "refine" session already carries the bumped
        # iteration; a "corrections" session being used to refine does not.
        round_iteration = (session["iteration"] if session_type == "refine"
                           else subj["iteration"] + 1)

        result = commit_labels_to_dlc(
            subject_name=subj["name"],
            session_labels=training_labels,
            iteration=round_iteration,
            camera_mode=subj.get("camera_mode"),
        )
    else:
        result = commit_labels_to_dlc(
            subject_name=subj["name"],
            session_labels=labels,
            iteration=session["iteration"],
            camera_mode=subj.get("camera_mode"),
        )

    with get_db_ctx() as db:
        db.execute(
            "UPDATE label_sessions SET status = 'committed', "
            "committed_at = CURRENT_TIMESTAMP WHERE id = ?", (session_id,))
        if is_refine:
            db.execute(
                "UPDATE subjects SET iteration = ?, "
                "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (round_iteration, session["subject_id"]))
        else:
            db.execute(
                "UPDATE subjects SET stage = 'committed', "
                "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (session["subject_id"],))

    return result


def _training_labels_from_corrections(subj: dict, train_frames: list) -> list[dict]:
    """Build training rows for the chosen frames from the corrections CSVs.

    The corrections CSVs — not the session's sparse edits — are the
    source: a frame may be worth training on because the prediction was
    already right there, with no manual edit of its own.
    """
    settings = get_settings()
    subject_name = subj["name"]

    corr_data = get_dlc_predictions_for_stage(subject_name, "corrections")
    if not corr_data:
        raise HTTPException(400, "No corrections found for this subject — "
                                 "save corrections before refining")

    trials = build_trial_map(subject_name,
                             camera_mode=subj.get("camera_mode"))
    frame_to_trial = {}
    for ti, t in enumerate(trials):
        for f in range(t["start_frame"], t["end_frame"] + 1):
            frame_to_trial[f] = ti

    out = []
    for vf in train_frames:
        if vf.side not in settings.camera_names:
            continue
        cam_data = corr_data.get(vf.side, {})
        kp = {}
        for bp in settings.bodyparts:
            arr = cam_data.get(bp)
            if arr and vf.frame_num < len(arr) and arr[vf.frame_num] is not None:
                kp[bp] = arr[vf.frame_num]
        if not kp:
            continue
        out.append({
            "frame_num": vf.frame_num,
            "trial_idx": frame_to_trial.get(vf.frame_num, 0),
            "side": vf.side,
            "keypoints": kp,
        })
    return out


# ── Crop boxes (the default zoom while labeling) ───────────────────────

@router.get("/sessions/{session_id}/bbox")
def get_bbox(session_id: int, trial_idx: int = Query(..., ge=0)) -> dict:
    """The trial's crop box per camera, computed from MediaPipe if unsaved.

    This box does two jobs: it frames the default zoom when a frame
    loads, so the hand fills the canvas without anyone panning, and it
    is the region the next MediaPipe run crops to.
    """
    settings = get_settings()
    _session, subj = _session_and_subject(session_id)
    cam_names = settings.camera_names

    with get_db_ctx() as db:
        rows = db.execute(
            "SELECT camera_name, x1, y1, x2, y2 FROM mp_crop_boxes "
            "WHERE subject_id = ? AND trial_idx = ? AND model_name = ?",
            (subj["id"], trial_idx, BBOX_MODEL),
        ).fetchall()
    if rows:
        boxes = {r["camera_name"]: [r["x1"], r["y1"], r["x2"], r["y2"]]
                 for r in rows}
        return {"boxes": boxes, "saved": True}

    # Nothing saved — derive one from this trial's landmarks.
    from ..services.mediapipe_prelabel import load_mediapipe_prelabels
    mp_data = load_mediapipe_prelabels(subj["name"])
    if mp_data is None:
        return {"boxes": {}, "saved": False}

    trials = build_trial_map(subj["name"], camera_mode=_camera_mode(subj))
    if trial_idx >= len(trials):
        raise HTTPException(404, f"Trial index {trial_idx} out of range")
    trial = trials[trial_idx]
    start = trial["start_frame"]
    end = start + trial["frame_count"]

    boxes = {}
    for cam, npz_key in zip(cam_names, ("OS_landmarks", "OD_landmarks")):
        lm = mp_data.get(npz_key)
        if lm is None:
            continue
        window = lm[start:min(end, lm.shape[0])]
        if window.size:
            boxes[cam] = compute_default_bbox(window)

    return {"boxes": boxes, "saved": False}


@router.post("/sessions/{session_id}/bbox")
def save_bbox(session_id: int, body: dict = Body(...)) -> dict:
    """Save crop boxes for a trial, optionally seeding the untouched trials.

    Body: ``{trial_idx, boxes: {cam: {x1,y1,x2,y2}}, apply_to_all}``

    ``apply_to_all`` only fills trials with no box of their own, so it
    never overwrites a box someone already adjusted.
    """
    _session, subj = _session_and_subject(session_id)

    trial_idx = int(body.get("trial_idx", 0))
    boxes = body.get("boxes") or {}
    apply_to_all = bool(body.get("apply_to_all"))
    if not boxes:
        raise HTTPException(400, "boxes must be a non-empty object")

    def _coords(c):
        return (float(c["x1"]), float(c["y1"]), float(c["x2"]), float(c["y2"]))

    applied = [trial_idx]
    with get_db_ctx() as db:
        for cam, coords in boxes.items():
            try:
                x1, y1, x2, y2 = _coords(coords)
            except (KeyError, TypeError, ValueError):
                raise HTTPException(400, f"Bad box for camera {cam}")
            db.execute(
                """INSERT INTO mp_crop_boxes
                       (subject_id, trial_idx, camera_name, model_name,
                        x1, y1, x2, y2)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(subject_id, trial_idx, camera_name, model_name)
                   DO UPDATE SET x1 = excluded.x1, y1 = excluded.y1,
                                 x2 = excluded.x2, y2 = excluded.y2,
                                 updated_at = CURRENT_TIMESTAMP""",
                (subj["id"], trial_idx, cam, BBOX_MODEL, x1, y1, x2, y2),
            )

        if apply_to_all:
            trials = build_trial_map(subj["name"],
                                     camera_mode=_camera_mode(subj))
            for ti in range(len(trials)):
                if ti == trial_idx:
                    continue
                existing = db.execute(
                    "SELECT COUNT(*) AS cnt FROM mp_crop_boxes "
                    "WHERE subject_id = ? AND trial_idx = ? AND model_name = ?",
                    (subj["id"], ti, BBOX_MODEL),
                ).fetchone()
                if existing["cnt"]:
                    continue
                for cam, coords in boxes.items():
                    x1, y1, x2, y2 = _coords(coords)
                    db.execute(
                        """INSERT OR IGNORE INTO mp_crop_boxes
                               (subject_id, trial_idx, camera_name, model_name,
                                x1, y1, x2, y2)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (subj["id"], ti, cam, BBOX_MODEL, x1, y1, x2, y2),
                    )
                applied.append(ti)

    return {"status": "ok", "applied_trials": applied}


@router.post("/sessions/{session_id}/rerun-mediapipe")
def rerun_mediapipe(session_id: int, body: dict = Body(...)) -> dict:
    """Re-run MediaPipe on one trial inside the given crop boxes.

    Body: ``{trial_idx, crops: {cam: {x1,y1,x2,y2}}}``

    This is the tight loop while labeling: adjust the box, re-detect
    this trial, see whether the hand is picked up on the frames that
    were failing.  Whole-subject passes go through the Jobs page.
    """
    import threading
    from ..services.jobs import registry

    _session, subj = _session_and_subject(session_id)

    trial_idx = int(body.get("trial_idx", 0))
    crops = body.get("crops") or {}
    if not crops:
        raise HTTPException(400, "crops must be a non-empty object")

    subject_name = subj["name"]
    camera_mode = _camera_mode(subj)
    trials = build_trial_map(subject_name, camera_mode=camera_mode)
    if trial_idx < 0 or trial_idx >= len(trials):
        raise HTTPException(400, f"Invalid trial_idx {trial_idx}")

    trial = trials[trial_idx]
    settings = get_settings()
    global_cams = settings.camera_names
    is_stereo_video = camera_mode == "stereo"

    cam_jobs = []
    for side, crop in crops.items():
        video_path = trial["video_path"]
        camera_key = side if side in global_cams else (
            global_cams[0] if global_cams else "OS")

        if camera_mode == "multicam":
            for cam in trial.get("cameras", []):
                if cam["name"] == side:
                    video_path = cam["path"]
                    break
            subject_cams = [c["name"] for c in trials[0].get("cameras", [])]
            if side in subject_cams:
                cam_idx = subject_cams.index(side)
                camera_key = (global_cams[cam_idx]
                              if cam_idx < len(global_cams) else side)

        cam_jobs.append({"side": side, "crop": crop,
                         "video_path": video_path, "camera_key": camera_key})

    log_dir = settings.dlc_path / ".logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = str(log_dir / f"mediapipe_crop_{subj['id']}_{trial_idx}.log")

    with get_db_ctx() as db:
        cur = db.execute(
            """INSERT INTO jobs (subject_id, job_type, status, log_path, params_json)
               VALUES (?, 'mediapipe', 'pending', ?, ?)""",
            (subj["id"], log_path,
             json.dumps({"trial_idx": trial_idx,
                         "trial_name": trial["trial_name"],
                         "cropped": True})),
        )
        job_id = cur.lastrowid

    cancel_event = registry.register_cancel_event(job_id)
    n_cams = len(cam_jobs)

    def _run():
        try:
            with get_db_ctx() as db:
                db.execute(
                    "UPDATE jobs SET status = 'running', "
                    "started_at = CURRENT_TIMESTAMP WHERE id = ?", (job_id,))

            for ci, cj in enumerate(cam_jobs):
                base_pct = ci * (100.0 / n_cams)
                span = 100.0 / n_cams

                def progress_cb(pct, _base=base_pct, _span=span):
                    if cancel_event.is_set():
                        raise InterruptedError("Job cancelled")
                    with get_db_ctx() as db:
                        db.execute(
                            "UPDATE jobs SET progress_pct = ? WHERE id = ?",
                            (_base + (pct / 100.0) * _span, job_id))

                run_mediapipe_cropped(
                    subject_name=subject_name,
                    video_path=cj["video_path"],
                    start_frame=trial["start_frame"],
                    frame_count=trial["frame_count"],
                    crop=cj["crop"],
                    camera_key=cj["camera_key"],
                    is_stereo_video=is_stereo_video,
                    stereo_side=cj["side"],
                    progress_callback=progress_cb,
                )

            with get_db_ctx() as db:
                db.execute(
                    "UPDATE jobs SET status = 'completed', progress_pct = 100, "
                    "finished_at = CURRENT_TIMESTAMP WHERE id = ?", (job_id,))
        except InterruptedError:
            with get_db_ctx() as db:
                db.execute(
                    "UPDATE jobs SET status = 'cancelled', "
                    "finished_at = CURRENT_TIMESTAMP WHERE id = ?", (job_id,))
        except Exception as e:
            logger.exception("Cropped MediaPipe failed for %s trial %s",
                             subject_name, trial_idx)
            with get_db_ctx() as db:
                db.execute(
                    "UPDATE jobs SET status = 'failed', error_msg = ?, "
                    "finished_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (str(e), job_id))
        finally:
            registry.unregister_cancel_event(job_id)
            # This job runs in-thread rather than as a subprocess, so the
            # registry monitor never sees it — record it here instead.
            try:
                from ..services.job_history import finalize_job_record
                finalize_job_record(job_id)
            except Exception:
                logger.exception("job_history flush failed for job %s", job_id)

    threading.Thread(target=_run, daemon=True).start()

    return {"status": "ok", "job_id": job_id, "trial_idx": trial_idx}
