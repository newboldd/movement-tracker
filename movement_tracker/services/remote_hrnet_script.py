#!/usr/bin/env python
"""Remote HRNet inference script.

Self-contained (no dlc_app imports). Uploaded to remote host and executed via SSH.
Runs HRNet-W18 on cropped hand regions to produce per-joint heatmaps.

Usage:
    python remote_hrnet_script.py <video_path> <output_dir> <mediapipe_npz>
        --trial-name <name> --start-frame <n> --n-frames <n>
        [--camera-mode stereo|single]
        [--weights-url <url>] [--weights-path <path>]
        [--bbox-os x1,y1,x2,y2] [--bbox-od x1,y1,x2,y2]
        [--status-file <path>]

Outputs:
    <output_dir>/<trial_name>/hrnet_w18_heatmaps.npz
    <output_dir>/<trial_name>/hand_crop.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _ensure_dependencies():
    """Check and install missing packages before running."""
    missing = []
    for pkg_import, pkg_pip in [("torch", "torch"), ("timm", "timm"),
                                 ("scipy", "scipy"), ("cv2", "opencv-python")]:
        try:
            __import__(pkg_import)
        except ImportError:
            missing.append(pkg_pip)
    if not missing:
        return
    logger.info(f"Installing missing packages: {missing}")
    import subprocess
    # Detect CUDA for torch
    torch_pkg = "torch"
    if "torch" in missing:
        # Try to detect CUDA
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
            if result.returncode == 0:
                torch_pkg = "torch --index-url https://download.pytorch.org/whl/cu121"
                logger.info("CUDA detected — installing torch with GPU support")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.info("No CUDA detected — installing CPU torch")
    for pkg in missing:
        cmd = [sys.executable, "-m", "pip", "install", "--progress-bar=on"]
        if pkg == "torch":
            cmd += torch_pkg.split()
        else:
            cmd.append(pkg)
        logger.info(f"  Running: {' '.join(cmd)}")
        sys.stdout.flush()
        # Stream pip output line by line so the monitor sees activity
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                bufsize=1, universal_newlines=True)
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to install {pkg} (exit code {proc.returncode})")
    logger.info("Dependencies installed successfully")


HEATMAP_SIZE = 64
N_JOINTS = 21
BBOX_PADDING = 1.25

# Per-frame bbox knobs — must mirror movement_tracker/services/hrnet_bbox.py
# so local + remote runs produce equivalent crops.
PERFRAME_PAD_FRAC = 0.25            # ¼ of the longer side on each axis →
                                     # final bbox = 1.5× landmark span
                                     # (≈ HRnet hand-pose training crop scale).
PERFRAME_SMOOTH_HALFWIN = 3
PERFRAME_MIN_PX = 64


def _per_frame_bboxes_from_landmarks(landmarks, pad_frac=PERFRAME_PAD_FRAC,
                                       smooth_halfwin=PERFRAME_SMOOTH_HALFWIN,
                                       min_px=PERFRAME_MIN_PX):
    """Self-contained per-frame bbox computation for the remote runner.
    Mirrors :func:`hrnet_bbox.compute_per_frame_bboxes`.

    Bbox **size is fixed across the trial** (side = p95(landmark_span) ×
    (1 + 2·pad_frac)); only the *center* tracks the hand frame-to-frame.
    For finger-tapping the hand stays at roughly the same camera distance,
    so the actual hand pixel-size is stable; letting bbox size oscillate
    with finger-open/close would inject tap-rhythm-locked noise into the
    heatmap-pixel ↔ image-pixel mapping.

    Returns float32 ``(N, 4)`` array.  NaN frames where MP failed are
    hold-filled from neighbours so every frame gets a valid bbox.
    """
    import numpy as _np
    N = landmarks.shape[0]
    centers = _np.full((N, 2), _np.nan, dtype=_np.float32)
    spans = _np.full(N, _np.nan, dtype=_np.float32)
    for f in range(N):
        pts = landmarks[f]
        if pts is None or _np.isnan(pts).any():
            continue
        x1 = pts[:, 0].min(); y1 = pts[:, 1].min()
        x2 = pts[:, 0].max(); y2 = pts[:, 1].max()
        centers[f] = [(x1 + x2) * 0.5, (y1 + y2) * 0.5]
        spans[f] = max(x2 - x1, y2 - y1)

    # Trial-wide fixed side from the 95th-percentile span (robust to
    # single-frame MP outliers).
    valid_spans = spans[~_np.isnan(spans)]
    if valid_spans.size == 0:
        return _np.full((N, 4), _np.nan, dtype=_np.float32)
    p95_span = float(_np.percentile(valid_spans, 95))
    fixed_side = p95_span * (1.0 + 2.0 * pad_frac)
    fixed_half = fixed_side * 0.5

    # Hold-fill missing centers from neighbours.
    last = None
    for f in range(N):
        if _np.isnan(centers[f, 0]):
            if last is not None:
                centers[f] = last
        else:
            last = centers[f].copy()
    # Forward-fill leading NaN block.
    first = None
    for f in range(N):
        if not _np.isnan(centers[f, 0]):
            first = centers[f].copy(); break
    if first is not None:
        for f in range(N):
            if _np.isnan(centers[f, 0]): centers[f] = first
            else: break

    # Temporal median on centers (suppresses MP wobble).
    if smooth_halfwin > 0:
        sm = _np.empty_like(centers)
        for f in range(N):
            lo = max(0, f - smooth_halfwin); hi = min(N, f + smooth_halfwin + 1)
            win = centers[lo:hi]
            sm[f, 0] = _np.nanmedian(win[:, 0])
            sm[f, 1] = _np.nanmedian(win[:, 1])
        centers = sm

    bb = _np.empty((N, 4), dtype=_np.float32)
    bb[:, 0] = centers[:, 0] - fixed_half
    bb[:, 1] = centers[:, 1] - fixed_half
    bb[:, 2] = centers[:, 0] + fixed_half
    bb[:, 3] = centers[:, 1] + fixed_half

    # Enforce min size (defensive — fixed_side already exceeds min_px in
    # any realistic trial, but a degenerate p95=0 would slip through).
    for f in range(N):
        if _np.isnan(bb[f, 0]): continue
        w = bb[f, 2] - bb[f, 0]; h = bb[f, 3] - bb[f, 1]
        if w < min_px:
            cx = (bb[f, 0] + bb[f, 2]) * 0.5
            bb[f, 0] = cx - min_px * 0.5; bb[f, 2] = cx + min_px * 0.5
        if h < min_px:
            cy = (bb[f, 1] + bb[f, 3]) * 0.5
            bb[f, 1] = cy - min_px * 0.5; bb[f, 3] = cy + min_px * 0.5
    return bb

WEIGHTS_URL = "https://download.openmmlab.com/mmpose/hand/hrnetv2/hrnetv2_w18_coco_wholebody_hand_256x256-1c028db7_20210908.pth"


def _write_status(status_file: str, status: str, progress_pct: float = 0.0,
                  error: str | None = None):
    """Write status.json atomically.

    On Windows, ``os.replace`` raises PermissionError [WinError 5] if the
    destination has an open handle — which happens routinely when the
    local poller fetches ``current_trial.json`` via SCP/SFTP at the same
    moment this writer replaces it.  Retry a few times with a small
    backoff; a stale status one tick old is far better than crashing the
    whole trial.

    Also normalise the destination path so mixed forward/back slashes
    don't confuse Windows handle tracking.
    """
    import time as _time
    status_file = os.path.normpath(status_file)
    data = {
        "status": status,
        "progress_pct": round(progress_pct, 1),
        "error": error,
        "pid": os.getpid(),
    }
    dir_name = os.path.dirname(os.path.abspath(status_file))
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

    # Retry the replace on Windows file-locking errors.  ~2s total.
    last_exc = None
    for attempt in range(10):
        try:
            os.replace(tmp, status_file)
            return
        except (PermissionError, OSError) as e:
            last_exc = e
            _time.sleep(min(0.05 * (1 + attempt), 0.5))
    # Final swallow — drop this update rather than tear down the trial.
    try:
        os.unlink(tmp)
    except OSError:
        pass
    print(f"[remote_hrnet_script] _write_status: giving up after retries: {last_exc}",
          flush=True)


def _download_weights(url: str, path: str, status_cb=None):
    """Download weights if not cached.  Reports progress via ``status_cb``
    (called with a percentage 0–100) so the local poller can see the
    download progressing instead of stalling silently for minutes.

    Size sanity-check: the HRNet-W18 hand weights file is ~78 MB.  A
    previous interrupted download can leave a truncated file at ``path``
    that passes ``os.path.exists`` but fails to load with
    ``PytorchStreamReader failed reading zip archive: failed finding
    central directory``.  We treat anything smaller than 30 MB as
    corrupt and re-download.
    """
    MIN_BYTES = 30 * 1024 * 1024  # 30 MB floor — real file is ~78 MB
    if os.path.exists(path):
        size = os.path.getsize(path)
        if size >= MIN_BYTES:
            logger.info(f"Weights already cached: {path} ({size / 1e6:.1f} MB)")
            return
        logger.warning(
            f"Cached weights at {path} look truncated ({size / 1e6:.1f} MB "
            f"< {MIN_BYTES / 1e6:.0f} MB floor) — redownloading"
        )
        try:
            os.unlink(path)
        except OSError:
            pass
    logger.info(f"Downloading weights to {path}...")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    last_pct = [-1]

    def _hook(blocks_done: int, block_size: int, total_size: int):
        if total_size <= 0:
            return
        pct = min(100, int(100 * blocks_done * block_size / total_size))
        # Throttle to one report per integer percent so we don't spam.
        if pct != last_pct[0]:
            last_pct[0] = pct
            sys.stdout.write(f"  download: {pct}%\n")
            sys.stdout.flush()
            if status_cb is not None:
                try: status_cb(pct)
                except Exception: pass

    # 60-second socket timeout so a stalled connection fails loudly.
    import socket
    prev_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(60)
    try:
        urllib.request.urlretrieve(url, path, reporthook=_hook)
    finally:
        socket.setdefaulttimeout(prev_timeout)
    size = os.path.getsize(path)
    logger.info(f"Downloaded ({size / 1e6:.1f} MB)")
    if size < MIN_BYTES:
        # Remove the bad file so we don't trip the cache-hit branch next time.
        try:
            os.unlink(path)
        except OSError:
            pass
        raise RuntimeError(
            f"Downloaded weights too small ({size} bytes < {MIN_BYTES}); "
            f"download may have been truncated"
        )


def _build_model(weights_path: str):
    """Build HRNet-W18 matching mmpose checkpoint."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import timm

    backbone = timm.create_model("hrnet_w18", pretrained=False)

    class HRNetPose(nn.Module):
        def __init__(self, backbone, n_joints=21):
            super().__init__()
            self.bb = backbone
            self.head = nn.Sequential(
                nn.Conv2d(270, 270, 1),
                nn.BatchNorm2d(270),
                nn.ReLU(inplace=True),
                nn.Conv2d(270, n_joints, 1),
            )

        def forward(self, x):
            x = self.bb.conv1(x)
            x = self.bb.bn1(x)
            x = self.bb.act1(x)
            x = self.bb.conv2(x)
            x = self.bb.bn2(x)
            x = self.bb.act2(x)
            x = self.bb.layer1(x)
            xl = [t(x) if t is not None else x for t in self.bb.transition1]
            xl = self.bb.stage2(xl)
            xl = [self.bb.transition2[i](xl[i] if i < len(xl) else xl[-1])
                  if self.bb.transition2[i] is not None else xl[i]
                  for i in range(len(self.bb.transition2))]
            xl = self.bb.stage3(xl)
            xl = [self.bb.transition3[i](xl[i] if i < len(xl) else xl[-1])
                  if self.bb.transition3[i] is not None else xl[i]
                  for i in range(len(self.bb.transition3))]
            xl = self.bb.stage4(xl)
            target = xl[0].shape[2:]
            parts = [xl[0]]
            for feat in xl[1:]:
                parts.append(F.interpolate(feat, size=target, mode="bilinear", align_corners=True))
            cat = torch.cat(parts, dim=1)
            return self.head(cat)

    model = HRNetPose(backbone)

    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
        if "state_dict" in state:
            state = state["state_dict"]
        model_state = model.state_dict()
        loaded = 0
        for k, v in state.items():
            new_key = k
            if k.startswith("keypoint_head.final_layer."):
                new_key = "head." + k[len("keypoint_head.final_layer."):]
            elif k.startswith("backbone."):
                new_key = "bb." + k[len("backbone."):]
            if new_key in model_state and v.shape == model_state[new_key].shape:
                model_state[new_key] = v
                loaded += 1
        model.load_state_dict(model_state, strict=False)
        logger.info(f"Loaded {loaded}/{len(state)} weights")

    model.eval()
    return model


def compute_default_bbox(mp_landmarks, padding=BBOX_PADDING):
    """Compute bounding box from MediaPipe landmarks (N, 21, 2)."""
    import numpy as np
    valid = ~np.isnan(mp_landmarks[:, :, 0])
    if not valid.any():
        return [0, 0, 256, 256]
    all_pts = mp_landmarks[valid]
    x_min, y_min = all_pts[:, 0].min(), all_pts[:, 1].min()
    x_max, y_max = all_pts[:, 0].max(), all_pts[:, 1].max()
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    side = max((x_max - x_min), (y_max - y_min)) * padding
    return [round(float(cx - side/2)), round(float(cy - side/2)),
            round(float(cx + side/2)), round(float(cy + side/2))]


def _infer_crop(model, frame, bbox, device):
    """Run HRNet on a cropped region. Returns (21, 64, 64) float16."""
    import torch
    import cv2
    import numpy as np
    from scipy.special import expit

    h, w = frame.shape[:2]
    # Requested bbox (may extend outside the frame — happens often with the
    # padded MP-derived bbox when the hand is near a frame edge).
    bx1, by1 = int(bbox[0]), int(bbox[1])
    bx2, by2 = int(bbox[2]), int(bbox[3])
    bw, bh = bx2 - bx1, by2 - by1
    if bw <= 0 or bh <= 0:
        return np.zeros((N_JOINTS, HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float16)

    # Clip to frame for slicing, then zero-pad back up to the requested size.
    # This preserves the heatmap ↔ saved-bbox coordinate mapping: heatmap
    # pixel (0, 0) always corresponds to image pixel (bx1, by1), even when
    # by1 is negative or bx2 is past the frame edge.  Without padding, the
    # cv2.resize would distort aspect ratio and the viewer would mis-place
    # heatmaps for any frame whose bbox extends beyond the image.
    sx1, sy1 = max(0, bx1), max(0, by1)
    sx2, sy2 = min(w, bx2), min(h, by2)
    if sx2 <= sx1 or sy2 <= sy1:
        return np.zeros((N_JOINTS, HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float16)
    crop = np.zeros((bh, bw, frame.shape[2]), dtype=frame.dtype)
    crop[sy1 - by1:sy1 - by1 + (sy2 - sy1),
         sx1 - bx1:sx1 - bx1 + (sx2 - sx1)] = frame[sy1:sy2, sx1:sx2]
    inp = cv2.resize(crop, (256, 256))
    inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    inp = ((inp - mean) / std).transpose(2, 0, 1)

    with torch.no_grad():
        tensor = torch.from_numpy(inp).unsqueeze(0).to(device)
        output = model(tensor)
        hm = output[0].cpu().numpy()

    result = np.zeros((N_JOINTS, HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float32)
    for j in range(min(hm.shape[0], N_JOINTS)):
        result[j] = cv2.resize(hm[j], (HEATMAP_SIZE, HEATMAP_SIZE))

    result = expit(result).astype(np.float32)
    mx = result.max()
    if mx > 0:
        result /= mx
    return result.astype(np.float16)


def run(args):
    status_file = args.status_file

    def report(pct):
        print(f"PROGRESS:hrnet:{pct:.1f}", flush=True)
        if status_file:
            _write_status(status_file, "running", pct)

    # Write status.json immediately so the local poller sees the job is
    # alive even before deps install / heavy imports / weights download.
    report(0)
    sys.stdout.write("Phase: ensure dependencies\n"); sys.stdout.flush()
    _ensure_dependencies()
    report(2)

    sys.stdout.write("Phase: import torch / cv2 / numpy\n"); sys.stdout.flush()
    import torch
    import cv2
    import numpy as np
    report(5)

    # Download weights — reports its own download progress within 5–10%.
    weights_dir = os.path.join(args.output_dir, ".models")
    weights_path = args.weights_path or os.path.join(weights_dir, "hrnet_w18_hand_256x256.pth")
    sys.stdout.write("Phase: download weights\n"); sys.stdout.flush()
    _download_weights(
        args.weights_url or WEIGHTS_URL, weights_path,
        status_cb=lambda dl_pct: report(5 + dl_pct * 0.05),  # maps 0–100 to 5–10
    )
    report(10)

    # Build model + move to GPU.  First cudnn warmup happens implicitly
    # on the first inference call below — can take 10–30 s with no
    # output; report(13) is the marker that "model build started".
    sys.stdout.write("Phase: build model\n"); sys.stdout.flush()
    report(13)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(weights_path)
    model = model.to(device)
    logger.info(f"Model on {device}")
    report(15)

    # Per-frame bboxes (default).  When MP labels are available we
    # derive a tight smoothed bbox per frame so the network sees the
    # hand at full effective resolution every frame.  Static bboxes
    # passed via --bbox-os/--bbox-od are broadcast to every frame for
    # back-compat.
    bbox_os = args.bbox_os
    bbox_od = args.bbox_od
    bboxes_os_pf = None    # (n_frames, 4) when available
    bboxes_od_pf = None
    start, n = args.start_frame, args.n_frames
    if not bbox_os or not bbox_od:
        mp_data = np.load(args.mediapipe_npz)
        os_lm = mp_data.get("OS_landmarks")
        od_lm = mp_data.get("OD_landmarks")
        end = min(start + n, os_lm.shape[0]) if os_lm is not None else start
        if not bbox_os and os_lm is not None:
            bboxes_os_pf = _per_frame_bboxes_from_landmarks(os_lm[start:end])
            valid = ~np.isnan(bboxes_os_pf[:, 0])
            if valid.any():
                bb = bboxes_os_pf[valid]
                bbox_os = [float(bb[:, 0].min()), float(bb[:, 1].min()),
                           float(bb[:, 2].max()), float(bb[:, 3].max())]
        if not bbox_od and od_lm is not None:
            bboxes_od_pf = _per_frame_bboxes_from_landmarks(od_lm[start:end])
            valid = ~np.isnan(bboxes_od_pf[:, 0])
            if valid.any():
                bb = bboxes_od_pf[valid]
                bbox_od = [float(bb[:, 0].min()), float(bb[:, 1].min()),
                           float(bb[:, 2].max()), float(bb[:, 3].max())]
    # If a static bbox was explicitly passed, broadcast it.
    if bboxes_os_pf is None and bbox_os:
        bboxes_os_pf = np.tile(np.asarray(bbox_os, dtype=np.float32), (n, 1))
    if bboxes_od_pf is None and bbox_od:
        bboxes_od_pf = np.tile(np.asarray(bbox_od, dtype=np.float32), (n, 1))

    if not bbox_os:
        raise ValueError("No bounding box for OS camera")

    logger.info(f"Bboxes (union): OS={bbox_os}, OD={bbox_od}")
    if bboxes_os_pf is not None:
        logger.info(f"  Per-frame OS shape={bboxes_os_pf.shape}, "
                    f"OD shape={bboxes_od_pf.shape if bboxes_od_pf is not None else 'None'}")
    report(20)

    # Open video
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video_path}")

    total_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    is_stereo = args.camera_mode == "stereo"
    midline = total_w // 2 if is_stereo else total_w

    n_frames = args.n_frames
    heatmaps_os = np.zeros((n_frames, N_JOINTS, HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float16)
    heatmaps_od = np.zeros((n_frames, N_JOINTS, HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float16) if is_stereo and bbox_od else None

    # ``start_frame`` is on the global concatenated-trial timeline.  Per-trial
    # video files are already trimmed, so seek must be 0; only multi-trial
    # single-file videos use the global offset.  Detect by comparing against
    # the actual video frame count.
    _video_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    seek_frame = args.start_frame if args.start_frame + n_frames <= _video_total else 0
    if seek_frame != args.start_frame:
        logger.info(f"Per-trial video detected: seeking to 0 instead of global frame {args.start_frame}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, seek_frame)
    report(25)

    for f_idx in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        bb_os_f = list(bboxes_os_pf[f_idx]) if bboxes_os_pf is not None else bbox_os
        bb_od_f = list(bboxes_od_pf[f_idx]) if bboxes_od_pf is not None else bbox_od

        frame_os = frame[:, :midline] if is_stereo else frame
        heatmaps_os[f_idx] = _infer_crop(model, frame_os, bb_os_f, device)

        if heatmaps_od is not None and bb_od_f:
            frame_od = frame[:, midline:]
            heatmaps_od[f_idx] = _infer_crop(model, frame_od, bb_od_f, device)

        if f_idx % 20 == 0:
            report(25 + (f_idx / n_frames) * 70)

    cap.release()
    report(95)

    # Save results
    out_dir = os.path.join(args.output_dir, args.trial_name)
    os.makedirs(out_dir, exist_ok=True)

    save_dict = {
        "heatmaps_L": heatmaps_os,
        "trial_idx": args.trial_idx,
        "start_frame": args.start_frame,
        "n_frames": n_frames,
    }
    if heatmaps_od is not None:
        save_dict["heatmaps_R"] = heatmaps_od

    out_path = os.path.join(out_dir, "hrnet_w18_heatmaps.npz")
    np.savez_compressed(out_path, **save_dict)

    # Pre-computed MIP (max over 21 joints) — saves the viewer from
    # fetching all 21 joint heatmaps per frame in MIP mode.
    try:
        mip_dict = {"heatmaps_L_mip": heatmaps_os.max(axis=1).astype(np.float16)}
        if heatmaps_od is not None:
            mip_dict["heatmaps_R_mip"] = heatmaps_od.max(axis=1).astype(np.float16)
        np.savez_compressed(os.path.join(out_dir, "hrnet_w18_mip.npz"), **mip_dict)
    except Exception as e:
        logger.warning(f"Failed to save MIP: {e}")

    # Both legacy union bbox and per-frame array — old readers fall
    # back to ``crop_L``/``crop_R``, new readers prefer
    # ``crop_L_perframe`` / ``crop_R_perframe`` for per-frame
    # heatmap-pixel ↔ image-pixel conversions.
    crop_info = {"crop_L": list(bbox_os)}
    if bbox_od:
        crop_info["crop_R"] = list(bbox_od)
    if bboxes_os_pf is not None:
        crop_info["crop_L_perframe"] = [
            [round(float(v), 1) for v in row] for row in bboxes_os_pf
        ]
    if bboxes_od_pf is not None:
        crop_info["crop_R_perframe"] = [
            [round(float(v), 1) for v in row] for row in bboxes_od_pf
        ]
    with open(os.path.join(out_dir, "hand_crop.json"), "w") as f:
        json.dump(crop_info, f)

    logger.info(f"Saved {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")

    # Print final progress BEFORE writing the completed status — report() also
    # writes status.json with status="running", so calling it after the
    # completed-write would clobber the terminal flag and leave the local
    # poller spinning forever.
    print("PROGRESS:hrnet:100.0", flush=True)
    if status_file:
        _write_status(status_file, "completed", 100)

    return {"output_path": out_path, "n_frames": n_frames}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remote HRNet inference")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("output_dir", help="Output directory for results")
    parser.add_argument("mediapipe_npz", help="Path to mediapipe_prelabels.npz")
    parser.add_argument("--trial-name", required=True)
    parser.add_argument("--trial-idx", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--n-frames", type=int, required=True)
    parser.add_argument("--camera-mode", default="stereo", choices=["stereo", "single"])
    parser.add_argument("--weights-url", default=WEIGHTS_URL)
    parser.add_argument("--weights-path", default=None)
    parser.add_argument("--bbox-os", type=lambda s: [int(x) for x in s.split(",")], default=None)
    parser.add_argument("--bbox-od", type=lambda s: [int(x) for x in s.split(",")], default=None)
    parser.add_argument("--status-file", default=None)
    parser.add_argument("--log-file", default=None)

    args = parser.parse_args()

    if args.log_file:
        fh = open(args.log_file, "w")
        sys.stdout = fh
        sys.stderr = fh

    try:
        run(args)
    except Exception as e:
        logger.error(f"HRNet failed: {e}")
        traceback.print_exc()
        if args.status_file:
            _write_status(args.status_file, "failed", error=str(e))
        sys.exit(1)

import traceback
