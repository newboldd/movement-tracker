"""HRNet hand pose heatmap inference service.

Runs HRNet-W18 on cropped hand regions to produce per-joint heatmaps.
Uses timm for the backbone with a simple pose estimation head.
Weights are auto-downloaded on first run.

Requires optional dependencies: torch, timm
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install timm
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..config import get_settings, DATA_DIR
from .mediapipe_prelabel import load_mediapipe_prelabels
from .video import build_trial_map

logger = logging.getLogger(__name__)

HEATMAP_SIZE = 64
N_JOINTS = 21

# Default bbox padding factor (1.25 = 25% padding around the hand)
BBOX_PADDING = 1.25

# HRNet-W18 pretrained weights URL (COCO wholebody hand)
WEIGHTS_URL = "https://download.openmmlab.com/mmpose/hand/hrnetv2/hrnetv2_w18_coco_wholebody_hand_256x256-1c028db7_20210908.pth"
WEIGHTS_FILENAME = "hrnet_w18_hand_256x256.pth"


def check_hrnet_available() -> dict:
    """Check if HRNet dependencies are available."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return {
            "available": False,
            "message": "PyTorch not installed. Run: pip install torch --index-url https://download.pytorch.org/whl/cpu",
        }
    try:
        import timm  # noqa: F401
    except ImportError:
        return {
            "available": False,
            "message": "timm not installed. Run: pip install timm",
        }
    return {"available": True, "message": "Ready"}


def _weights_path() -> Path:
    """Path to cached HRNet weights file."""
    cache_dir = DATA_DIR / "models"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / WEIGHTS_FILENAME


def _download_weights(progress_callback: Callable[[float], None] | None = None):
    """Download HRNet weights if not cached."""
    path = _weights_path()
    if path.exists():
        return

    logger.info(f"Downloading HRNet-W18 weights to {path}...")
    import urllib.request

    def _progress(block_num, block_size, total_size):
        if progress_callback and total_size > 0:
            pct = min(block_num * block_size / total_size * 100, 100)
            progress_callback(pct * 0.15)  # 0-15% of total progress

    urllib.request.urlretrieve(WEIGHTS_URL, str(path), reporthook=_progress)
    logger.info(f"Downloaded HRNet weights ({path.stat().st_size / 1e6:.1f} MB)")


def _build_model():
    """Build HRNet-W18 model with pose estimation head."""
    import torch
    import torch.nn as nn
    import timm

    # Use timm HRNet-W18 backbone
    backbone = timm.create_model("hrnet_w18", pretrained=False, features_only=True)

    class HRNetPose(nn.Module):
        def __init__(self, backbone, n_joints=21):
            super().__init__()
            self.backbone = backbone
            # HRNet outputs multi-scale features; use highest resolution
            # The first feature map is typically 1/4 resolution
            # Add a simple 1x1 conv to predict heatmaps
            self.head = nn.Conv2d(18, n_joints, kernel_size=1)

        def forward(self, x):
            features = self.backbone(x)
            # Use highest resolution feature map (first output)
            hm = self.head(features[0])
            return hm

    model = HRNetPose(backbone)

    # Try loading pretrained weights (best-effort partial load)
    weights_path = _weights_path()
    if weights_path.exists():
        state = torch.load(str(weights_path), map_location="cpu", weights_only=False)
        # mmpose checkpoints wrap in 'state_dict'
        if "state_dict" in state:
            state = state["state_dict"]
        # Filter and remap keys (best-effort)
        model_state = model.state_dict()
        loaded = 0
        for k, v in state.items():
            # Try direct match
            if k in model_state and v.shape == model_state[k].shape:
                model_state[k] = v
                loaded += 1
            # Try stripping 'backbone.' prefix
            elif k.startswith("backbone.") and k[9:] in model_state:
                nk = k[9:]
                if v.shape == model_state[nk].shape:
                    model_state[nk] = v
                    loaded += 1
        model.load_state_dict(model_state, strict=False)
        logger.info(f"Loaded {loaded} weight tensors from {weights_path.name}")

    model.eval()
    return model


def compute_default_bbox(mp_landmarks: np.ndarray, padding: float = BBOX_PADDING) -> list:
    """Compute a default bounding box from MediaPipe landmarks.

    Args:
        mp_landmarks: (N, 21, 2) landmark array for one camera
        padding: expansion factor (1.25 = 25% padding)

    Returns:
        [x1, y1, x2, y2] bounding box in pixel coordinates
    """
    # Collect all valid landmark positions
    valid = ~np.isnan(mp_landmarks[:, :, 0])
    if not valid.any():
        return [0, 0, 256, 256]

    all_pts = mp_landmarks[valid]  # (M, 2)
    x_min, y_min = all_pts[:, 0].min(), all_pts[:, 1].min()
    x_max, y_max = all_pts[:, 0].max(), all_pts[:, 1].max()

    # Expand with padding
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w = (x_max - x_min) * padding
    h = (y_max - y_min) * padding
    # Make square (use max dimension)
    side = max(w, h)

    return [
        round(float(cx - side / 2)),
        round(float(cy - side / 2)),
        round(float(cx + side / 2)),
        round(float(cy + side / 2)),
    ]


def run_hrnet_trial(
    subject_name: str,
    trial_idx: int,
    bbox_os: list | None = None,
    bbox_od: list | None = None,
    cancel_event: threading.Event | None = None,
    progress_callback: Callable[[float], None] | None = None,
) -> dict:
    """Run HRNet inference on a single trial.

    Args:
        subject_name: Subject identifier
        trial_idx: Trial index from the video trial map
        bbox_os: [x1,y1,x2,y2] crop box for OS camera (auto if None)
        bbox_od: [x1,y1,x2,y2] crop box for OD camera (auto if None)
        cancel_event: Threading event to check for cancellation
        progress_callback: Called with progress 0-100

    Returns:
        Dict with output path and stats.
    """
    import torch

    def report(pct):
        if progress_callback:
            progress_callback(pct)

    def cancelled():
        return cancel_event is not None and cancel_event.is_set()

    report(0)
    logger.info(f"[HRNet] {subject_name} trial {trial_idx}")

    # ── Download weights if needed ─────────────────────────────
    _download_weights(progress_callback)
    report(15)

    if cancelled():
        return {"cancelled": True}

    # ── Load trial info ────────────────────────────────────────
    trials = build_trial_map(subject_name)
    if trial_idx < 0 or trial_idx >= len(trials):
        raise ValueError(f"Trial index {trial_idx} out of range")

    trial = trials[trial_idx]
    n_frames = trial["frame_count"]
    start_frame = trial.get("start_frame", 0)
    video_path = trial["video_path"]

    from ..db import get_db_ctx
    with get_db_ctx() as db:
        row = db.execute("SELECT camera_mode FROM subjects WHERE name = ?", (subject_name,)).fetchone()
    settings = get_settings()
    camera_mode = (row["camera_mode"] if row and row["camera_mode"] else settings.default_camera_mode)
    is_stereo = camera_mode == "stereo"

    # ── Load MediaPipe for bbox defaults ───────────────────────
    mp_data = load_mediapipe_prelabels(subject_name)
    if mp_data is not None:
        os_lm = mp_data["OS_landmarks"]
        od_lm = mp_data.get("OD_landmarks")
        end = min(start_frame + n_frames, os_lm.shape[0])
        mp_trial_os = os_lm[start_frame:end]
        mp_trial_od = od_lm[start_frame:end] if od_lm is not None else None
    else:
        mp_trial_os = None
        mp_trial_od = None

    # Compute or use provided bboxes
    if bbox_os is None and mp_trial_os is not None:
        bbox_os = compute_default_bbox(mp_trial_os)
    if bbox_od is None and mp_trial_od is not None:
        bbox_od = compute_default_bbox(mp_trial_od)

    if bbox_os is None:
        raise ValueError("No bounding box for OS camera (run MediaPipe first)")

    logger.info(f"  Bboxes: OS={bbox_os}, OD={bbox_od}")
    report(20)

    # ── Build model ────────────────────────────────────────────
    model = _build_model()
    logger.info(f"  HRNet model ready")
    report(25)

    if cancelled():
        return {"cancelled": True}

    # ── Process video frames ───────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    midline = total_w // 2 if is_stereo else total_w

    heatmaps_os = np.zeros((n_frames, N_JOINTS, HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float16)
    heatmaps_od = np.zeros((n_frames, N_JOINTS, HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float16) if is_stereo and bbox_od else None

    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for f_idx in range(n_frames):
        if cancelled():
            cap.release()
            return {"cancelled": True}

        ret, frame = cap.read()
        if not ret:
            break

        # Process OS (left camera)
        if is_stereo:
            frame_os = frame[:, :midline]
        else:
            frame_os = frame

        hm_os = _infer_crop(model, frame_os, bbox_os)
        heatmaps_os[f_idx] = hm_os

        # Process OD (right camera)
        if heatmaps_od is not None and bbox_od is not None:
            frame_od = frame[:, midline:]
            hm_od = _infer_crop(model, frame_od, bbox_od)
            heatmaps_od[f_idx] = hm_od

        if f_idx % 20 == 0:
            report(25 + (f_idx / n_frames) * 70)

    cap.release()
    report(95)

    # ── Save results ───────────────────────────────────────────
    out_dir = settings.dlc_path / subject_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hrnet_heatmaps.npz"

    save_dict = {
        "heatmaps_OS": heatmaps_os,
        "bbox_OS": np.array(bbox_os, dtype=np.float32),
        "trial_idx": trial_idx,
        "start_frame": start_frame,
        "n_frames": n_frames,
    }
    if heatmaps_od is not None:
        save_dict["heatmaps_OD"] = heatmaps_od
        save_dict["bbox_OD"] = np.array(bbox_od, dtype=np.float32)

    np.savez_compressed(str(out_path), **save_dict)
    report(100)

    logger.info(f"  Saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    return {
        "output_path": str(out_path),
        "n_frames": n_frames,
        "bbox_os": bbox_os,
        "bbox_od": bbox_od,
    }


def _infer_crop(model, frame: np.ndarray, bbox: list) -> np.ndarray:
    """Run HRNet on a cropped region of a frame.

    Args:
        model: HRNet model
        frame: BGR image (H, W, 3)
        bbox: [x1, y1, x2, y2] crop box

    Returns:
        (21, 64, 64) heatmap array as float16
    """
    import torch

    h, w = frame.shape[:2]
    x1 = max(0, int(bbox[0]))
    y1 = max(0, int(bbox[1]))
    x2 = min(w, int(bbox[2]))
    y2 = min(h, int(bbox[3]))

    if x2 <= x1 or y2 <= y1:
        return np.zeros((N_JOINTS, HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float16)

    crop = frame[y1:y2, x1:x2]
    # Resize to 256x256 (HRNet input)
    inp = cv2.resize(crop, (256, 256))
    inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
    inp = inp.astype(np.float32) / 255.0
    # Normalize with ImageNet mean/std
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    inp = (inp - mean) / std
    inp = inp.transpose(2, 0, 1)  # HWC → CHW

    with torch.no_grad():
        tensor = torch.from_numpy(inp).unsqueeze(0)
        output = model(tensor)
        # output shape: (1, 21, H', W') — resize to 64x64
        hm = output[0].numpy()  # (21, H', W')

    # Resize each joint's heatmap to 64x64
    result = np.zeros((N_JOINTS, HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float32)
    n_out = min(hm.shape[0], N_JOINTS)
    for j in range(n_out):
        result[j] = cv2.resize(hm[j], (HEATMAP_SIZE, HEATMAP_SIZE))

    # Normalize to [0, 1] per joint
    for j in range(N_JOINTS):
        mx = result[j].max()
        if mx > 0:
            result[j] /= mx

    return result.astype(np.float16)
