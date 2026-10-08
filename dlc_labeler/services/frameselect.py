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
