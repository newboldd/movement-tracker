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
import threading
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
from ..services.calibration import get_calibration_for_subject
from ..services.discovery import _count_labeled_frames, _has_mediapipe
from ..services.labels import commit_labels_to_dlc
from ..services.mediapipe_data import recompute_distance_for_frame
from ..services.video import (
    _deidentified_path, _get_no_face_videos, build_trial_map, extract_frame,
)

logger = logging.getLogger(__name__)

# Frame selection runs on a thread and reports here; the page polls it.
# In memory on purpose — it is live state of this process, meaningless
# after a restart, and not worth a table.
_selection_lock = threading.Lock()
_selection_progress: dict[str, dict] = {}

router = APIRouter(prefix="/api/labeling", tags=["labeling"])

# The one MediaPipe file this app reads.  It does not run MediaPipe —
# Movement Tracker produces these, and this app labels against them.
MEDIAPIPE_NPZ = "mediapipe.npz"


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


def _bodypart_joints(bodyparts: list[str]) -> dict:
    """Which MediaPipe landmark stands for each bodypart, where one does.

    Guided labeling zooms to the point it is asking for, and MediaPipe's
    guess at that point is the only thing available before anybody has
    labeled it.  Matched by name rather than position so renaming or
    reordering bodyparts cannot quietly point "thumb" at the index
    finger; a bodypart with no obvious landmark simply gets no entry and
    falls back to the whole-hand crop.
    """
    from ..services.mediapipe_data import INDEX_TIP, THUMB_TIP

    known = {"thumb": THUMB_TIP, "index": INDEX_TIP}
    out = {}
    for bp in bodyparts:
        joint = known.get(bp.strip().lower())
        if joint is not None:
            out[bp] = joint
    return out


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

    return {
        "session": session,
        "subject": subj,
        "trials": trial_info,
        "total_frames": total_frames,
        "bodyparts": bodyparts,
        "camera_names": settings.camera_names,
        "camera_mode": camera_mode,
        "committed_frame_count": committed_frame_count,
        "has_calibration": get_calibration_for_subject(subj["name"]) is not None,
        "bodypart_joints": _bodypart_joints(bodyparts),
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

@router.get("/sessions/{session_id}/trial/{trial_idx}/mp")
def get_trial_mp(session_id: int, trial_idx: int) -> dict:
    """The MediaPipe forward pass for one trial: 21 joints, both cameras.

    One payload per trial rather than per frame, so stepping frames stays
    instant.  All 21 joints, not just the labeled bodyparts, because the
    guided zoom aims at a named landmark and the overlay is only readable
    as a hand when the whole hand is there.

    This app does not run MediaPipe; it reads what Movement Tracker left
    behind.  A trial with no npz simply has no landmarks, which the page
    copes with — labeling still works, it is just not pre-aimed.
    """
    settings = get_settings()
    _session, subj = _session_and_subject(session_id)
    subject_name = subj["name"]

    trials = build_trial_map(subject_name, camera_mode=_camera_mode(subj))
    if trial_idx < 0 or trial_idx >= len(trials):
        raise HTTPException(404, f"Trial index {trial_idx} out of range")
    trial = trials[trial_idx]

    out: dict = {
        "trial_idx": trial_idx,
        "trial_name": trial["trial_name"],
        "start_frame": trial["start_frame"],
        "frame_count": trial["frame_count"],
        "OS": None,
        "OD": None,
        "distances": None,
    }

    npz_path = (settings.dlc_path / subject_name / trial["trial_name"]
                / MEDIAPIPE_NPZ)
    if not npz_path.exists():
        return out
    try:
        data = np.load(str(npz_path))
    except (OSError, ValueError) as e:
        logger.warning("Could not read %s: %s", npz_path, e)
        return out

    if "OS_landmarks" in data.files:
        out["OS"] = _f32(data["OS_landmarks"])
    if "OD_landmarks" in data.files:
        out["OD"] = _f32(data["OD_landmarks"])
    for key in ("distances_clean", "distances"):
        if key in data.files:
            out["distances"] = _nan_list(data[key])
            break
    return out


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

    # The camera you have not labeled yet falls back to MediaPipe inside
    # recompute_distance_for_frame, so the number under the video means
    # something before a frame is finished.
    updated_distances = {}
    for frame_num, sides in frame_labels.items():
        dist = recompute_distance_for_frame(subj["name"], frame_num, sides)
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


@router.post("/sessions/{session_id}/commit")
def commit_session(session_id: int,
                   req: CommitRequest = Body(default=None)) -> dict:
    """Write this session's labels into the subject's DeepLabCut project.

    Every labeled frame becomes a training image — a PNG under
    ``labeled-data/round{iteration}/`` with a CollectedData CSV and H5
    beside it, which is the directory DeepLabCut's
    ``create_training_dataset`` reads.  That project is this app's
    output; the app does not train on it.
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
        db.execute(
            "UPDATE subjects SET stage = 'committed', "
            "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (session["subject_id"],))

    return result


# ── Choosing the frames to label ───────────────────────────────────────


@router.get("/{subject_id}/selection")
def get_selection(subject_id: int) -> dict:
    """The subject's chosen labeling frames, if the selection has run.

    ``frames`` is in the order the page shows them, and each carries the
    MediaPipe crop per camera so the grid can be drawn and each frame
    framed on the hand before anybody has labeled it.
    """
    from ..services.frameselect import DEFAULT_N_PICK, load_selection

    with get_db_ctx() as db:
        subj = db.execute("SELECT name FROM subjects WHERE id = ?",
                          (subject_id,)).fetchone()
    if not subj:
        raise HTTPException(404, "Subject not found")

    payload = load_selection(subj["name"])
    if payload is None:
        return {"exists": False, "frames": [], "default_n": DEFAULT_N_PICK}
    return {"exists": True, **payload}


@router.post("/{subject_id}/selection")
def run_selection(subject_id: int, body: dict = Body(default=None)) -> dict:
    """Start the k-means frame selection for a subject, in the background.

    Returns immediately: the clustering decodes every trial, which is a
    minute or two on a real subject and must not be a request the browser
    sits waiting on.  The page polls ``/selection/progress``.

    A thread, not a subprocess: it needs nothing this process does not
    already have — OpenCV and scikit-learn, no DeepLabCut, no GPU.
    """
    import threading

    from ..services.frameselect import (
        DEFAULT_N_PICK, save_selection, select_training_frames,
    )

    body = body or {}
    with get_db_ctx() as db:
        subj = db.execute("SELECT id, name, camera_mode FROM subjects WHERE id = ?",
                          (subject_id,)).fetchone()
    if not subj:
        raise HTTPException(404, "Subject not found")

    n_pick = int(body.get("n_pick") or DEFAULT_N_PICK)
    if not 1 <= n_pick <= 500:
        raise HTTPException(400, "n_pick must be between 1 and 500")
    seed = body.get("seed")
    step = body.get("step")

    subject_name = subj["name"]
    camera_mode = _camera_mode(subj)

    with _selection_lock:
        running = _selection_progress.get(subject_name)
        if running and running.get("status") == "running":
            return {"status": "already-running", **running}
        _selection_progress[subject_name] = {
            "status": "running", "pct": 0.0, "phase": "Starting", "error": None,
        }

    def _set(**kw):
        with _selection_lock:
            _selection_progress.setdefault(subject_name, {}).update(kw)

    def _run():
        try:
            def progress_cb(pct, phase=""):
                _set(pct=min(float(pct), 99.0), phase=phase)

            selections, notes, meta = select_training_frames(
                subject_name, n_pick, camera_mode=camera_mode,
                step=int(step) if step else None,
                seed=int(seed) if seed is not None else None,
                progress=progress_cb)

            if not selections:
                raise RuntimeError(notes[0] if notes else
                                   "No frames could be selected")
            save_selection(subject_name, selections, notes, meta)
            _set(status="done", pct=100.0, phase="Done", error=None)
        except Exception as e:
            logger.exception("Frame selection failed for %s", subject_name)
            _set(status="failed", phase="Failed", error=str(e))

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "running", "n_pick": n_pick}


@router.get("/{subject_id}/selection/progress")
def selection_progress(subject_id: int) -> dict:
    """How far the running selection has got, for the page to poll."""
    with get_db_ctx() as db:
        subj = db.execute("SELECT name FROM subjects WHERE id = ?",
                          (subject_id,)).fetchone()
    if not subj:
        raise HTTPException(404, "Subject not found")
    with _selection_lock:
        state = _selection_progress.get(subj["name"])
    return state or {"status": "idle", "pct": 0.0, "phase": "", "error": None}
