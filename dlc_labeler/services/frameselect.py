"""Choosing which frames are worth labeling.

The selection is DeepLabCut's own k-means method, unchanged in its
mechanics — ``MiniBatchKMeans`` over mean-centred, downsampled greyscale
frames, one frame drawn at random from each cluster.  Keeping it
identical matters: a training set built here should be defensible as
"frames selected the standard DeepLabCut way", and reproducible by
anyone who checks.

What differs is *which pixels* it clusters.  Stock DeepLabCut
downsamples the whole frame to 30 pixels wide.  On a 1920-pixel camera
half that leaves the hand a few pixels across, so the clustering is
driven by arm position, lighting and background — the frames come out
"looking different" in ways that have nothing to do with the finger
configuration a labeler is there to mark.  Cropping to the hand first
puts those 30 pixels where the information is.  DeepLabCut itself reads
frames with ``crop=True``, so clustering a cropped region is what stock
DLC does whenever a crop is configured; this just derives the crop from
MediaPipe instead of from a hand-typed box in config.yaml.

The crop follows the hand frame by frame.  A fixed box would have to be
big enough for every position the hand visits, which on these recordings
means mostly empty pixels, and it makes one unusable frame — a face in
shot, say — contaminate the whole trial rather than a single image.  The
box *size* is held constant across a trial so every exported image has
the same scale and the dataset carries no accidental zoom.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..config import get_settings
from .mediapipe_prelabel import N_JOINTS
from .video import build_trial_map, get_subject_videos

logger = logging.getLogger(__name__)

# DeepLabCut's defaults, from utils/frameselectiontools.py.  Changing any
# of these changes which frames come out, so they are named rather than
# inlined and are surfaced as parameters.
DLC_RESIZE_WIDTH = 30
DLC_BATCH_SIZE = 100
DLC_MAX_ITER = 50
DLC_TOL = 1e-3

# How much bigger than the hand's own extent the crop box is drawn.  1.6
# leaves room for the fingers to splay and for MediaPipe to be a little
# wrong about where the fingertips are, without spending half the image
# on background.
CROP_PADDING = 1.6

# The crop side comes from this percentile of the per-frame hand extent
# rather than the maximum, so one bad MediaPipe frame — a spurious
# landmark out at the edge of the image — cannot inflate the box for the
# whole trial.
CROP_SIDE_PERCENTILE = 95.0

# Never crop smaller than this, whatever MediaPipe reports.  A hand
# detected as a few pixels across is a detection failure, not a small hand.
MIN_CROP_SIDE = 96


@dataclass
class FrameChoice:
    """One selected frame, and the crop that should be written for it."""

    global_frame: int
    local_frame: int
    trial_name: str
    trial_idx: int
    camera: str
    # Crop in the camera half's pixel space: x0/y0 is the top-left corner.
    x0: int
    y0: int
    side: int
    # False when this frame had no MediaPipe landmarks and the crop centre
    # was borrowed from the nearest frame that did.
    mp_available: bool = True


@dataclass
class TrialCrops:
    """Per-frame crop geometry for one trial and camera."""

    trial_idx: int
    trial_name: str
    camera: str
    side: int
    # Per local frame: (x0, y0, mp_available). None where no crop is possible.
    boxes: list[tuple[int, int, bool] | None] = field(default_factory=list)

    @property
    def usable(self) -> list[int]:
        return [i for i, b in enumerate(self.boxes) if b is not None]


def _landmark_source(subject_name: str, trial_name: str) -> Path | None:
    """The best MediaPipe npz for a trial: the fused pass, else forward.

    The fused "best per frame" output already picks, per frame and
    camera, whichever pass triangulated most plausibly — so when it
    exists it is strictly the better set of landmarks to crop around.
    """
    trial_dir = get_settings().dlc_path / subject_name / trial_name
    for name in ("mediapipe_combined.npz", "mediapipe.npz",
                 "mediapipe_reverse.npz", "mediapipe_static.npz",
                 "mediapipe_cropped.npz"):
        path = trial_dir / name
        if path.exists():
            return path
    return None


def load_trial_landmarks(subject_name: str, trial_name: str,
                         camera_index: int) -> np.ndarray | None:
    """(n, 21, 2) landmarks for one trial and camera, or None."""
    path = _landmark_source(subject_name, trial_name)
    if path is None:
        return None
    try:
        data = np.load(str(path))
    except (OSError, ValueError) as e:
        logger.warning("Could not read %s: %s", path, e)
        return None

    key = "OS_landmarks" if camera_index == 0 else "OD_landmarks"
    if key not in data.files:
        return None
    arr = np.asarray(data[key], dtype=float)
    if arr.ndim != 3 or arr.shape[1] != N_JOINTS:
        logger.warning("Unexpected landmark shape %s in %s", arr.shape, path)
        return None
    return arr


def compute_trial_crops(landmarks: np.ndarray, frame_width: int,
                        frame_height: int, *,
                        padding: float = CROP_PADDING,
                        side_percentile: float = CROP_SIDE_PERCENTILE,
                        min_side: int = MIN_CROP_SIDE) -> tuple[int, list]:
    """Work out one square crop per frame from a trial's landmarks.

    Returns ``(side, boxes)`` where boxes[i] is ``(x0, y0, mp_available)``
    or None when no crop could be placed at all.

    The side is constant across the trial and the centre follows the
    hand.  Frames MediaPipe missed borrow the nearest detected frame's
    centre — the hand has not usually moved far, and those frames are
    often the most worth labeling, so dropping them outright would bias
    the training set towards the easy ones.
    """
    n = landmarks.shape[0]
    valid = np.isfinite(landmarks[:, :, 0]) & np.isfinite(landmarks[:, :, 1])
    any_valid = valid.any(axis=1)

    centres: list[tuple[float, float] | None] = [None] * n
    extents = []
    for i in range(n):
        if not any_valid[i]:
            continue
        pts = landmarks[i][valid[i]]
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        centres[i] = ((lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0)
        extents.append(max(hi[0] - lo[0], hi[1] - lo[1]))

    if not extents:
        return 0, [None] * n

    side = int(round(float(np.percentile(extents, side_percentile)) * padding))
    side = max(side, min_side)
    # A crop larger than the frame cannot be placed; clamp to the frame.
    side = min(side, frame_width, frame_height)

    # Nearest detected frame, in either direction, for the gaps.
    detected = [i for i, c in enumerate(centres) if c is not None]
    boxes: list[tuple[int, int, bool] | None] = [None] * n
    half = side / 2.0
    for i in range(n):
        centre = centres[i]
        mp_available = centre is not None
        if centre is None:
            if not detected:
                continue
            # Index of the closest detected frame.
            pos = int(np.argmin([abs(d - i) for d in detected]))
            centre = centres[detected[pos]]

        # Shift, never shrink: every image keeps the same size and scale.
        x0 = int(round(centre[0] - half))
        y0 = int(round(centre[1] - half))
        x0 = max(0, min(x0, frame_width - side))
        y0 = max(0, min(y0, frame_height - side))
        boxes[i] = (x0, y0, mp_available)

    return side, boxes


def _read_crops(video_path: str, crops: TrialCrops, candidates: list[int],
                camera_index: int, camera_mode: str,
                resize_width: int,
                progress: callable | None = None) -> np.ndarray | None:
    """Decode the trial once and return the downsampled crop of each candidate.

    Returns (len(candidates), resize_width, resize_width) greyscale, the
    input DeepLabCut's clustering expects — except square, because the
    crops are square, so every frame contributes the same number of
    pixels regardless of where the hand was.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("Could not open %s", video_path)
        return None

    wanted = set(candidates)
    order = {f: i for i, f in enumerate(candidates)}
    out = np.zeros((len(candidates), resize_width, resize_width), dtype=float)
    filled = np.zeros(len(candidates), dtype=bool)

    # Read straight through rather than seeking per frame: seeking is what
    # makes DeepLabCut's own extraction slow on long videos, and we want
    # most frames anyway.
    local = 0
    last = max(wanted) if wanted else -1
    while local <= last:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if local in wanted:
            box = crops.boxes[local]
            if box is not None:
                x0, y0, _ = box
                half = _camera_half(frame, camera_index, camera_mode)
                patch = half[y0:y0 + crops.side, x0:x0 + crops.side]
                if patch.size:
                    small = cv2.resize(patch, (resize_width, resize_width),
                                       interpolation=cv2.INTER_NEAREST)
                    # DLC greys by averaging the channels rather than using
                    # a luma transform; matched here so the clustering sees
                    # the same numbers it would there.
                    out[order[local]] = small.mean(axis=2)
                    filled[order[local]] = True
        local += 1
        if progress and local % 100 == 0:
            progress(local / max(last + 1, 1))

    cap.release()

    if not filled.any():
        return None
    if not filled.all():
        # Frames the decoder could not produce (a truncated file) would
        # otherwise cluster as identical black images and win a cluster
        # each.  Drop them from the matrix and from the candidate list.
        keep = np.where(filled)[0]
        candidates[:] = [candidates[i] for i in keep]
        out = out[keep]

    return out


def _camera_half(frame: np.ndarray, camera_index: int,
                 camera_mode: str) -> np.ndarray:
    """The pixels belonging to one camera, for the subject's layout."""
    if camera_mode != "stereo":
        return frame
    midline = frame.shape[1] // 2
    return frame[:, midline:] if camera_index == 1 else frame[:, :midline]


def dlc_kmeans_pick(data: np.ndarray, n_pick: int, *,
                    batch_size: int = DLC_BATCH_SIZE,
                    max_iter: int = DLC_MAX_ITER,
                    tol: float = DLC_TOL,
                    seed: int | None = None) -> list[int]:
    """DeepLabCut's k-means frame selection, applied to a prepared matrix.

    A direct transcription of ``KmeansbasedFrameselectioncv2``: mean-centre
    each pixel across frames, flatten, fit MiniBatchKMeans with one cluster
    per frame wanted, then take one frame at random from each cluster.

    Taking a random cluster member rather than the one nearest the centroid
    is DeepLabCut's choice, and it is kept: the centroid frame of a tight
    cluster is the most typical frame in it, and a training set of typical
    frames generalises worse than one that samples the spread.  ``seed``
    makes a run reproducible, which stock DLC does not offer.

    Returns indices into ``data``, in cluster order, as DLC returns them.
    """
    from sklearn.cluster import MiniBatchKMeans

    n_frames = data.shape[0]
    if n_frames <= n_pick:
        return list(range(n_frames))

    # Mean-centre per pixel, then flatten each frame to a vector.
    centred = data - data.mean(axis=0)
    flat = centred.reshape(n_frames, -1)

    effective_batch = batch_size if batch_size <= n_frames else n_frames // 2
    kmeans = MiniBatchKMeans(
        n_clusters=n_pick,
        tol=tol,
        batch_size=max(effective_batch, 1),
        max_iter=max_iter,
        random_state=seed,
        n_init="auto",
    )
    kmeans.fit(flat)

    rng = np.random.default_rng(seed)
    picks: list[int] = []
    for cluster_id in range(n_pick):
        members = np.where(kmeans.labels_ == cluster_id)[0]
        if len(members):
            picks.append(int(members[rng.integers(len(members))]))
    return picks


def select_for_trial(subject_name: str, trial: dict, trial_idx: int,
                     camera: str, camera_index: int, camera_mode: str,
                     n_pick: int, *,
                     step: int = 1,
                     resize_width: int = DLC_RESIZE_WIDTH,
                     seed: int | None = None,
                     progress: callable | None = None
                     ) -> tuple[list[FrameChoice], str | None]:
    """Choose ``n_pick`` frames from one trial and camera.

    Returns ``(choices, reason_skipped)`` — the reason is a short sentence
    for the log when nothing could be selected.
    """
    landmarks = load_trial_landmarks(subject_name, trial["trial_name"],
                                     camera_index)
    if landmarks is None:
        return [], (f"{trial['trial_name']} {camera}: no MediaPipe landmarks "
                    f"— run MediaPipe on this subject first")

    frame_count = int(trial["frame_count"])
    if landmarks.shape[0] < frame_count:
        # A pass run before the video was re-trimmed; use what overlaps.
        frame_count = landmarks.shape[0]

    width = int(trial.get("width") or 0)
    height = int(trial.get("height") or 0)
    if camera_mode == "stereo":
        width //= 2
    if width <= 0 or height <= 0:
        return [], f"{trial['trial_name']} {camera}: unknown frame size"

    side, boxes = compute_trial_crops(landmarks[:frame_count], width, height)
    if side <= 0:
        return [], (f"{trial['trial_name']} {camera}: MediaPipe found no hand "
                    f"in any frame")

    crops = TrialCrops(trial_idx=trial_idx, trial_name=trial["trial_name"],
                       camera=camera, side=side, boxes=boxes)

    candidates = [i for i in crops.usable if i % step == 0]
    if not candidates:
        return [], f"{trial['trial_name']} {camera}: no usable frames"

    data = _read_crops(trial["video_path"], crops, candidates, camera_index,
                       camera_mode, resize_width, progress=progress)
    if data is None:
        return [], f"{trial['trial_name']} {camera}: could not decode frames"

    picked = dlc_kmeans_pick(data, n_pick, seed=seed)

    choices = []
    for idx in sorted(picked):
        local = candidates[idx]
        x0, y0, mp_available = crops.boxes[local]
        choices.append(FrameChoice(
            global_frame=int(trial["start_frame"]) + local,
            local_frame=local,
            trial_name=trial["trial_name"],
            trial_idx=trial_idx,
            camera=camera,
            x0=x0, y0=y0, side=side,
            mp_available=mp_available,
        ))
    return choices, None


def select_for_subject(subject_name: str, n_per_trial: int, *,
                       cameras: list[str] | None = None,
                       camera_mode: str | None = None,
                       step: int = 1,
                       seed: int | None = None,
                       progress: callable | None = None
                       ) -> tuple[list[FrameChoice], list[str]]:
    """Choose frames across every trial and camera of a subject.

    The budget is per trial and camera, which is how DeepLabCut thinks
    about it (it extracts per video) and what keeps one long trial from
    crowding out the rest.

    Returns ``(choices, notes)``; notes are the human-readable reasons any
    trial produced nothing.
    """
    settings = get_settings()
    cam_names = cameras or settings.camera_names
    mode = camera_mode or settings.default_camera_mode

    if not get_subject_videos(subject_name):
        return [], [f"{subject_name}: no videos found"]

    trials = build_trial_map(subject_name, camera_mode=mode)
    if not trials:
        return [], [f"{subject_name}: no trials found"]

    total_units = max(len(trials) * len(cam_names), 1)
    all_choices: list[FrameChoice] = []
    notes: list[str] = []

    unit = 0
    for trial_idx, trial in enumerate(trials):
        for camera_index, camera in enumerate(cam_names):
            def sub_progress(frac, _unit=unit):
                if progress:
                    progress(100.0 * (_unit + frac) / total_units)

            choices, reason = select_for_trial(
                subject_name, trial, trial_idx, camera, camera_index, mode,
                n_per_trial, step=step, seed=seed, progress=sub_progress)
            if reason:
                notes.append(reason)
                logger.info("%s", reason)
            all_choices.extend(choices)
            unit += 1
            if progress:
                progress(100.0 * unit / total_units)

    return all_choices, notes


# ── Picking frames to label, for the whole subject ─────────────────────
#
# Different question from the exporter above.  There the budget is per
# trial and camera, because each image is labeled on its own.  Here one
# pick is one *frame*, labeled on every camera — so a frame has to be
# chosen on how it looks in both cameras at once, and the budget is for
# the subject rather than the trial.

SELECTION_SCHEMA = "dlc-labeler-selection/1"
SELECTION_FILE = "selection.json"

# Default number of frames to label.  DeepLabCut's own default is 20;
# 30 is what this project labels.
DEFAULT_N_PICK = 30

# Cap on how many frames go into the clustering matrix.  DeepLabCut
# considers every frame (``cluster_step=1``), which on a ten-minute trial
# is 18,000 rows for no gain — neighbouring frames are near-duplicates
# and land in the same cluster anyway.  Sampling to this many spreads the
# same coverage over far less work; the step used is recorded.
MAX_CLUSTER_FRAMES = 4000


def _auto_step(total_usable: int, limit: int = MAX_CLUSTER_FRAMES) -> int:
    """Frame stride that keeps the clustering matrix under ``limit`` rows."""
    if total_usable <= limit:
        return 1
    return int(np.ceil(total_usable / limit))


def _trial_camera_crops(subject_name: str, trial: dict, trial_idx: int,
                        cam_names: list[str], camera_mode: str
                        ) -> tuple[list, int, list[str]]:
    """Crop geometry for every camera of one trial.

    Returns ``(per_camera, frame_count, notes)`` where each per-camera
    entry is ``TrialCrops`` or None.  None means MediaPipe has nothing
    for that camera and the whole camera half stands in — which is what
    DeepLabCut would have clustered on anyway.
    """
    frame_count = int(trial["frame_count"])
    width = int(trial.get("width") or 0)
    height = int(trial.get("height") or 0)
    if camera_mode == "stereo":
        width //= 2

    per_camera: list[TrialCrops | None] = []
    notes: list[str] = []
    for camera_index, camera in enumerate(cam_names):
        landmarks = load_trial_landmarks(subject_name, trial["trial_name"],
                                         camera_index)
        if landmarks is None or width <= 0 or height <= 0:
            per_camera.append(None)
            notes.append(f"{trial['trial_name']} {camera}: no MediaPipe "
                         f"landmarks — clustering the whole frame instead")
            continue
        n = min(frame_count, landmarks.shape[0])
        side, boxes = compute_trial_crops(landmarks[:n], width, height)
        if side <= 0:
            per_camera.append(None)
            notes.append(f"{trial['trial_name']} {camera}: MediaPipe found no "
                         f"hand — clustering the whole frame instead")
            continue
        boxes = boxes + [None] * (frame_count - len(boxes))
        per_camera.append(TrialCrops(trial_idx=trial_idx,
                                     trial_name=trial["trial_name"],
                                     camera=camera, side=side, boxes=boxes))
    return per_camera, frame_count, notes


def _trial_candidates(per_camera: list, frame_count: int, step: int) -> list[int]:
    """Local frames usable on every camera, at the given stride."""
    out = []
    for local in range(0, frame_count, step):
        ok = True
        for crops in per_camera:
            if crops is None:
                continue        # whole-frame fallback: always usable
            if local >= len(crops.boxes) or crops.boxes[local] is None:
                ok = False
                break
        if ok:
            out.append(local)
    return out


def _read_frame_features(video_path: str, candidates: list[int],
                         per_camera: list, camera_mode: str,
                         resize_width: int,
                         progress=None) -> np.ndarray | None:
    """Decode a trial once and build one feature row per candidate frame.

    A row is every camera's crop, downsampled and laid side by side, so
    the clustering sees the frame the way the labeler will: both views of
    the hand at the same instant.

    Only candidate frames are retrieved.  ``grab`` advances the decoder
    without converting and copying the image, which is most of the cost
    per frame, so a stride makes this proportionally faster — where
    DeepLabCut's own stride makes it *slower*, because it seeks.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("Could not open %s", video_path)
        return None

    wanted = set(candidates)
    order = {f: i for i, f in enumerate(candidates)}
    n_cam = len(per_camera)
    out = np.zeros((len(candidates), n_cam, resize_width, resize_width),
                   dtype=float)
    filled = np.zeros(len(candidates), dtype=bool)

    last = max(wanted) if wanted else -1
    local = 0
    while local <= last:
        if not cap.grab():
            break
        if local in wanted:
            ok, frame = cap.retrieve()
            if ok and frame is not None:
                row = out[order[local]]
                for camera_index, crops in enumerate(per_camera):
                    half = _camera_half(frame, camera_index, camera_mode)
                    if crops is not None and crops.boxes[local] is not None:
                        x0, y0, _ = crops.boxes[local]
                        patch = half[y0:y0 + crops.side, x0:x0 + crops.side]
                    else:
                        patch = half
                    if patch.size:
                        small = cv2.resize(patch, (resize_width, resize_width),
                                           interpolation=cv2.INTER_NEAREST)
                        # DLC greys by averaging channels, not by a luma
                        # transform; matched so the numbers are the same.
                        row[camera_index] = small.mean(axis=2)
                filled[order[local]] = True
        local += 1
        if progress and local % 200 == 0:
            progress(local / max(last + 1, 1))

    cap.release()

    if not filled.any():
        return None
    if not filled.all():
        keep = np.where(filled)[0]
        candidates[:] = [candidates[i] for i in keep]
        out = out[keep]
    return out


def select_training_frames(subject_name: str, n_pick: int = DEFAULT_N_PICK, *,
                           camera_mode: str | None = None,
                           cameras: list[str] | None = None,
                           step: int | None = None,
                           seed: int | None = None,
                           resize_width: int = DLC_RESIZE_WIDTH,
                           progress=None) -> tuple[list[dict], list[str], dict]:
    """Pick the frames to label for a subject, as DeepLabCut would.

    Same algorithm as ``deeplabcut.extract_frames(algo='kmeans')`` —
    MiniBatchKMeans over mean-centred, 30px-wide greyscale frames, one
    random frame per cluster — with two differences, both deliberate:

    - the clustering runs over the whole subject rather than one video,
      because the 30 frames are a budget for the subject; and
    - what it clusters is the MediaPipe hand crop from each camera rather
      than the whole frame.  At 30px wide, a 1920px camera half leaves
      the hand about two pixels, so whole-frame clusters sort by arm
      position, lighting and background.  Cropping first makes them sort
      by hand posture, which is what a labeling set should span.  Where
      MediaPipe has nothing, the whole frame stands in.

    Returns ``(selections, notes, meta)``.
    """
    settings = get_settings()
    cam_names = cameras or settings.camera_names
    mode = camera_mode or settings.default_camera_mode

    trials = build_trial_map(subject_name, camera_mode=mode)
    if not trials:
        return [], [f"{subject_name}: no trials found"], {}

    notes: list[str] = []
    prepared = []
    total_usable = 0
    for trial_idx, trial in enumerate(trials):
        per_camera, frame_count, trial_notes = _trial_camera_crops(
            subject_name, trial, trial_idx, cam_names, mode)
        notes.extend(trial_notes)
        usable = len(_trial_candidates(per_camera, frame_count, 1))
        total_usable += usable
        prepared.append((trial, trial_idx, per_camera, frame_count, usable))

    if not total_usable:
        return [], notes + [f"{subject_name}: no usable frames"], {}

    chosen_step = step if step and step > 0 else _auto_step(total_usable)

    rows: list[np.ndarray] = []
    index: list[tuple] = []        # (trial, trial_idx, per_camera, local)
    done = 0
    for trial, trial_idx, per_camera, frame_count, usable in prepared:
        candidates = _trial_candidates(per_camera, frame_count, chosen_step)
        if not candidates:
            continue

        def sub_progress(frac, _done=done, _usable=usable):
            if progress:
                progress(100.0 * (_done + frac * _usable) / max(total_usable, 1))

        data = _read_frame_features(trial["video_path"], candidates,
                                    per_camera, mode, resize_width,
                                    progress=sub_progress)
        done += usable
        if data is None:
            notes.append(f"{trial['trial_name']}: could not decode frames")
            continue
        rows.append(data)
        for local in candidates:
            index.append((trial, trial_idx, per_camera, local))
        if progress:
            progress(100.0 * done / max(total_usable, 1))

    if not rows:
        return [], notes + [f"{subject_name}: nothing could be decoded"], {}

    matrix = np.concatenate(rows, axis=0)
    picked = dlc_kmeans_pick(matrix, n_pick, seed=seed)

    selections = []
    for idx in sorted(picked):
        trial, trial_idx, per_camera, local = index[idx]
        crops = {}
        for camera_index, camera in enumerate(cam_names):
            tc = per_camera[camera_index] if camera_index < len(per_camera) else None
            box = tc.boxes[local] if tc is not None else None
            crops[camera] = (None if box is None else
                             {"x0": int(box[0]), "y0": int(box[1]),
                              "side": int(tc.side), "mp": bool(box[2])})
        selections.append({
            "global_frame": int(trial["start_frame"]) + local,
            "local_frame": int(local),
            "trial_idx": int(trial_idx),
            "trial_name": trial["trial_name"],
            "crops": crops,
        })

    meta = {
        "n_pick": int(n_pick),
        "step": int(chosen_step),
        "seed": seed,
        "frames_considered": int(matrix.shape[0]),
        "frames_available": int(total_usable),
        "resize_width": int(resize_width),
        "cameras": list(cam_names),
        "camera_mode": mode,
        "method": "deeplabcut-kmeans-on-mediapipe-crop",
    }
    return selections, notes, meta


def selection_path(subject_name: str) -> Path:
    return get_settings().dlc_path / subject_name / SELECTION_FILE


def save_selection(subject_name: str, selections: list[dict],
                   notes: list[str], meta: dict) -> dict:
    """Write the selection beside the subject's DLC project.

    On disk rather than in the database because it is a property of the
    videos, survives the database being rebuilt, and is the kind of thing
    somebody will want to read, diff or hand to a collaborator.
    """
    from datetime import datetime, timezone

    payload = {
        "schema": SELECTION_SCHEMA,
        "subject": subject_name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **meta,
        "notes": notes,
        "frames": selections,
    }
    path = selection_path(subject_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1))
    logger.info("Wrote %d selected frames to %s", len(selections), path)
    return payload


def load_selection(subject_name: str) -> dict | None:
    """The saved selection for a subject, or None when it has never run."""
    path = selection_path(subject_name)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        logger.warning("Could not read %s: %s", path, e)
        return None
    if not str(payload.get("schema", "")).startswith("dlc-labeler-selection/"):
        return None
    return payload
