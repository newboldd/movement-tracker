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
    missing = []
    try:
        import torch  # noqa: F401
    except ImportError:
        missing.append("torch")
    try:
        import timm  # noqa: F401
    except ImportError:
        missing.append("timm")
    if missing:
        return {
            "available": False,
            "installable": True,
            "missing": missing,
            "message": f"Missing: {', '.join(missing)}. Click Install to download (~550 MB).",
        }
    return {"available": True, "installable": False, "missing": [], "message": "Ready"}


def install_hrnet_deps() -> dict:
    """Install torch (CPU) and timm via pip. Returns status dict."""
    import sys
    import subprocess

    python = sys.executable
    installed = []
    errors = []

    # Check what's missing
    status = check_hrnet_available()
    if status["available"]:
        return {"ok": True, "message": "Already installed"}

    for pkg in status["missing"]:
        try:
            if pkg == "torch":
                logger.info("Installing PyTorch (CPU-only, ~540 MB)...")
                result = subprocess.run(
                    [python, "-m", "pip", "install", "-q", "--disable-pip-version-check",
                     "torch", "--index-url", "https://download.pytorch.org/whl/cpu"],
                    capture_output=True, text=True, timeout=600,
                )
            else:
                logger.info(f"Installing {pkg}...")
                result = subprocess.run(
                    [python, "-m", "pip", "install", "-q", "--disable-pip-version-check", pkg],
                    capture_output=True, text=True, timeout=120,
                )
            if result.returncode == 0:
                installed.append(pkg)
                logger.info(f"Installed {pkg}")
            else:
                errors.append(f"{pkg}: {result.stderr[-200:]}")
                logger.error(f"Failed to install {pkg}: {result.stderr[-200:]}")
        except Exception as e:
            errors.append(f"{pkg}: {e}")

    if errors:
        return {"ok": False, "message": f"Failed: {'; '.join(errors)}", "installed": installed}
    return {"ok": True, "message": f"Installed: {', '.join(installed)}", "installed": installed}


def _refine_peak_centroid(hm: np.ndarray, py: int, px: int,
                          thresh_norm: float = 0.9) -> tuple:
    """Refine a heatmap peak at (py, px) via the value-weighted centroid of
    the contiguous region of pixels whose normalized value ≥ ``thresh_norm``.

    Normalization matches the viewer's display: (val − min) / (max − min),
    so ``thresh_norm`` corresponds 1-to-1 with the threshold slider value
    (e.g. slider at 0.9 → only pixels above this threshold contribute).

    Returns (cx, cy) in sub-pixel heatmap coords (centres in [0, W), [0, H)),
    or the original (px + 0.5, py + 0.5) if the peak itself is below the
    threshold (degenerate heatmap)."""
    H, W = hm.shape
    mx = float(hm.max()); mn = float(hm.min()); rng = mx - mn
    if rng <= 1e-9:
        return (px + 0.5, py + 0.5)
    thr = mn + thresh_norm * rng
    if hm[py, px] < thr:
        return (px + 0.5, py + 0.5)
    # 4-connected BFS flood-fill from the peak over pixels ≥ thr.
    visited = np.zeros((H, W), dtype=bool)
    stack = [(py, px)]
    visited[py, px] = True
    wsum = 0.0; xsum = 0.0; ysum = 0.0
    while stack:
        y, x = stack.pop()
        v = float(hm[y, x])
        wsum += v
        xsum += v * (x + 0.5)
        ysum += v * (y + 0.5)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and not visited[ny, nx] and hm[ny, nx] >= thr:
                visited[ny, nx] = True
                stack.append((ny, nx))
    if wsum <= 0.0:
        return (px + 0.5, py + 0.5)
    return (xsum / wsum, ysum / wsum)


def _argmax_peaks(heatmaps: np.ndarray, bbox) -> np.ndarray:
    """Per-(frame, joint) raw argmax peak in image coords.

    No Hungarian disambiguation, no centroid refinement — just the pixel
    with the largest value in each joint's heatmap.  Returns (N, 21, 2)
    in (x_image_px, y_image_px), NaN where the heatmap is empty.
    """
    N, J, H, W = heatmaps.shape
    out = np.full((N, J, 2), np.nan, dtype=np.float32)
    for f in range(N):
        x1, y1, x2, y2 = _bbox_at(bbox, f)
        bw, bh = float(x2 - x1), float(y2 - y1)
        for j in range(J):
            hm = heatmaps[f, j]
            if hm.size == 0: continue
            mx = float(hm.max())
            if mx <= 0: continue
            idx = int(hm.reshape(-1).argmax())
            py, px = idx // W, idx % W
            out[f, j, 0] = x1 + (px + 0.5) / W * bw
            out[f, j, 1] = y1 + (py + 0.5) / H * bh
    return out


_NEIGH_8 = ((-1, -1), (-1, 0), (-1, 1),
            ( 0, -1),          ( 0, 1),
            ( 1, -1), ( 1, 0), ( 1, 1))


def _cluster_centroid_peak(hm: np.ndarray,
                           cluster_size: int,
                           spike_support: float = 0.0,
                           edge_margin: int = 0) -> tuple:
    """Greedy 8-connected cluster grow from the heatmap argmax, then
    value-weighted centroid of the cluster.

    Spike rejection (when ``spike_support > 0`` or ``edge_margin > 0``):
      Before growing, the chosen argmax seed is validated:
        * Compute ``support = mean(8 in-bounds neighbours) / seed``.
        * Pixels within ``edge_margin`` of any heatmap edge require
          ``support >= 1.0`` (effectively "must be a smooth interior
          peak, not a corner blob").
        * Other pixels require ``support >= spike_support``.
      If validation fails, the seed and a 3×3 neighbourhood are zeroed
      and the next-strongest pixel is tested.  Up to 8 tries.

    ``cluster_size = 1`` falls back to the validated seed pixel centre.

    Returns (cx, cy) in sub-pixel heatmap coords, or (NaN, NaN) when no
    valid peak is found.
    """
    H, W = hm.shape
    if hm.size == 0:
        return (float("nan"), float("nan"))
    mx = float(hm.max())
    if mx <= 0:
        return (float("nan"), float("nan"))

    work = hm
    if spike_support > 0 or edge_margin > 0:
        work = hm.astype(np.float32, copy=True)

    sy: int = -1; sx: int = -1
    for _ in range(8):
        if not (work.max() > 0):
            return (float("nan"), float("nan"))
        idx = int(work.reshape(-1).argmax())
        cand_y, cand_x = idx // W, idx % W
        seed_v = float(work[cand_y, cand_x])
        if spike_support <= 0 and edge_margin <= 0:
            sy, sx = cand_y, cand_x
            break
        # Compute 8-neighbour mean for the candidate.
        nb_sum = 0.0; nb_count = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0: continue
                ny, nx = cand_y + dy, cand_x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    nb_sum += float(hm[ny, nx])
                    nb_count += 1
        nb_mean = (nb_sum / nb_count) if nb_count else 0.0
        support = (nb_mean / seed_v) if seed_v > 1e-9 else 0.0
        edge_dist = min(cand_y, H - 1 - cand_y, cand_x, W - 1 - cand_x)
        # Edge-margin pixels need full support (≈ smooth peak).
        thr = 1.0 if edge_dist < edge_margin else float(spike_support)
        if support >= thr:
            sy, sx = cand_y, cand_x
            break
        # Reject this candidate: zero a 3×3 region around it and retry.
        y0, y1 = max(0, cand_y - 1), min(H, cand_y + 2)
        x0, x1 = max(0, cand_x - 1), min(W, cand_x + 2)
        work[y0:y1, x0:x1] = 0.0
    if sy < 0:
        return (float("nan"), float("nan"))

    cluster_size = max(1, int(cluster_size))
    if cluster_size == 1:
        return (sx + 0.5, sy + 0.5)

    in_cluster = np.zeros((H, W), dtype=bool)
    in_cluster[sy, sx] = True
    cluster_pixels = [(sy, sx)]

    # Frontier = candidate neighbours with their value and squared
    # distance to seed.  Recomputed each step from the current cluster
    # boundary; sets are small (<25 entries for slider max) so a simple
    # linear search is more than fast enough.
    while len(cluster_pixels) < cluster_size:
        best_val = -1.0
        best_d2 = 0
        best_pos = None
        for cy, cx in cluster_pixels:
            for dy, dx in _NEIGH_8:
                ny, nx = cy + dy, cx + dx
                if not (0 <= ny < H and 0 <= nx < W):
                    continue
                if in_cluster[ny, nx]:
                    continue
                v = float(hm[ny, nx])
                if v > best_val + 1e-9:
                    best_val = v
                    best_d2 = (ny - sy) ** 2 + (nx - sx) ** 2
                    best_pos = (ny, nx)
                elif abs(v - best_val) <= 1e-9:
                    d2 = (ny - sy) ** 2 + (nx - sx) ** 2
                    if d2 < best_d2:
                        best_d2 = d2
                        best_pos = (ny, nx)
        if best_pos is None:
            break  # ran into image edges before reaching cluster_size
        in_cluster[best_pos] = True
        cluster_pixels.append(best_pos)

    wsum = 0.0; xsum = 0.0; ysum = 0.0
    for (cy, cx) in cluster_pixels:
        v = float(hm[cy, cx])
        wsum += v
        xsum += v * (cx + 0.5)
        ysum += v * (cy + 0.5)
    if wsum <= 0:
        return (sx + 0.5, sy + 0.5)
    return (xsum / wsum, ysum / wsum)


def _cluster_centroid_peak_auc(hm: np.ndarray,
                               cluster_size: int,
                               spike_support: float = 0.0,
                               edge_margin: int = 0) -> tuple:
    """Same as :func:`_cluster_centroid_peak` but also returns the cluster
    AUC (sum of pixel values in the grown 8-connected cluster).  Used as
    a per-frame, per-joint, per-camera "confidence" signal in the HRnet
    correction pipeline (replaces MP confidence in the detector).

    Returns ``(cx, cy, auc)``; ``auc`` is the sum of the cluster pixel
    values (≥ 0), or 0.0 when no valid peak is found.
    """
    H, W = hm.shape
    if hm.size == 0:
        return (float("nan"), float("nan"), 0.0)
    mx = float(hm.max())
    if mx <= 0:
        return (float("nan"), float("nan"), 0.0)

    work = hm
    if spike_support > 0 or edge_margin > 0:
        work = hm.astype(np.float32, copy=True)

    sy: int = -1; sx: int = -1
    for _ in range(8):
        if not (work.max() > 0):
            return (float("nan"), float("nan"), 0.0)
        idx = int(work.reshape(-1).argmax())
        cand_y, cand_x = idx // W, idx % W
        seed_v = float(work[cand_y, cand_x])
        if spike_support <= 0 and edge_margin <= 0:
            sy, sx = cand_y, cand_x
            break
        nb_sum = 0.0; nb_count = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0: continue
                ny, nx = cand_y + dy, cand_x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    nb_sum += float(hm[ny, nx])
                    nb_count += 1
        nb_mean = (nb_sum / nb_count) if nb_count else 0.0
        support = (nb_mean / seed_v) if seed_v > 1e-9 else 0.0
        edge_dist = min(cand_y, H - 1 - cand_y, cand_x, W - 1 - cand_x)
        thr = 1.0 if edge_dist < edge_margin else float(spike_support)
        if support >= thr:
            sy, sx = cand_y, cand_x
            break
        y0, y1 = max(0, cand_y - 1), min(H, cand_y + 2)
        x0, x1 = max(0, cand_x - 1), min(W, cand_x + 2)
        work[y0:y1, x0:x1] = 0.0
    if sy < 0:
        return (float("nan"), float("nan"), 0.0)

    cluster_size = max(1, int(cluster_size))
    in_cluster = np.zeros((H, W), dtype=bool)
    in_cluster[sy, sx] = True
    cluster_pixels = [(sy, sx)]
    while len(cluster_pixels) < cluster_size:
        best_val = -1.0
        best_d2 = 0
        best_pos = None
        for cy, cx in cluster_pixels:
            for dy, dx in _NEIGH_8:
                ny, nx = cy + dy, cx + dx
                if not (0 <= ny < H and 0 <= nx < W):
                    continue
                if in_cluster[ny, nx]:
                    continue
                v = float(hm[ny, nx])
                if v > best_val + 1e-9:
                    best_val = v
                    best_d2 = (ny - sy) ** 2 + (nx - sx) ** 2
                    best_pos = (ny, nx)
                elif abs(v - best_val) <= 1e-9:
                    d2 = (ny - sy) ** 2 + (nx - sx) ** 2
                    if d2 < best_d2:
                        best_d2 = d2
                        best_pos = (ny, nx)
        if best_pos is None:
            break
        in_cluster[best_pos] = True
        cluster_pixels.append(best_pos)

    wsum = 0.0; xsum = 0.0; ysum = 0.0
    for (cy, cx) in cluster_pixels:
        v = float(hm[cy, cx])
        wsum += v
        xsum += v * (cx + 0.5)
        ysum += v * (cy + 0.5)
    if wsum <= 0:
        return (sx + 0.5, sy + 0.5, 0.0)
    return (xsum / wsum, ysum / wsum, wsum)


def _bbox_at(bbox_in, f: int) -> tuple:
    """Return ``(x1, y1, x2, y2)`` for frame ``f``.  Accepts either a
    static 4-tuple/list (broadcast to every frame) or a per-frame
    ``(N, 4)`` numpy array.  Use inside per-frame loops so heatmap-pixel
    ↔ image-pixel conversions track the per-frame crop."""
    arr = np.asarray(bbox_in)
    if arr.ndim == 1:
        return (float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))
    return (float(arr[f, 0]), float(arr[f, 1]),
            float(arr[f, 2]), float(arr[f, 3]))


def _cluster_centroid_peaks(heatmaps: np.ndarray,
                            bbox,
                            cluster_size: int,
                            spike_support: float = 0.0,
                            edge_margin: int = 0) -> np.ndarray:
    """Per-(frame, joint) cluster-centroid peak in image pixel coords.

    ``bbox`` is either a static 4-tuple (legacy) or a per-frame
    ``(N, 4)`` array; per-frame is preferred when available since it
    keeps the heatmap→image mapping correct on trials where the
    per-frame crop varies.

    Returns (N, 21, 2) float32, NaN where the heatmap was empty.
    """
    N, J, H, W = heatmaps.shape
    out = np.full((N, J, 2), np.nan, dtype=np.float32)
    for f in range(N):
        x1, y1, x2, y2 = _bbox_at(bbox, f)
        bw, bh = float(x2 - x1), float(y2 - y1)
        for j in range(J):
            cx, cy = _cluster_centroid_peak(
                heatmaps[f, j], cluster_size,
                spike_support=spike_support, edge_margin=edge_margin)
            if not (np.isnan(cx) or np.isnan(cy)):
                out[f, j, 0] = x1 + (cx / W) * bw
                out[f, j, 1] = y1 + (cy / H) * bh
    return out


def _cluster_centroid_peaks_with_auc(heatmaps: np.ndarray,
                                      bbox,
                                      cluster_size: int,
                                      spike_support: float = 0.0,
                                      edge_margin: int = 0) -> tuple:
    """Same as :func:`_cluster_centroid_peaks` but also returns per-(frame,
    joint) cluster AUC.  Returns ``(peaks (N,J,2), auc (N,J))``."""
    N, J, H, W = heatmaps.shape
    out = np.full((N, J, 2), np.nan, dtype=np.float32)
    auc = np.zeros((N, J), dtype=np.float32)
    for f in range(N):
        x1, y1, x2, y2 = _bbox_at(bbox, f)
        bw, bh = float(x2 - x1), float(y2 - y1)
        for j in range(J):
            cx, cy, a = _cluster_centroid_peak_auc(
                heatmaps[f, j], cluster_size,
                spike_support=spike_support, edge_margin=edge_margin)
            if not (np.isnan(cx) or np.isnan(cy)):
                out[f, j, 0] = x1 + (cx / W) * bw
                out[f, j, 1] = y1 + (cy / H) * bh
            auc[f, j] = a
    return out, auc


def _refine_peaks_tensor(tgt: np.ndarray, heatmaps: np.ndarray, bbox) -> np.ndarray:
    """Given per-frame per-joint peaks in IMAGE coords (``tgt`` shape
    (N, 21, 2), NaN where unassigned), refine each by flood-filling from
    its heatmap pixel over the ≥90th-percentile-by-normalized-value
    connected region, then mapping the refined sub-pixel centroid back to
    image coords via ``bbox``.  Accepts a static or per-frame bbox."""
    N, J, H, W = heatmaps.shape
    out = tgt.copy()
    for f in range(N):
        x1, y1, x2, y2 = _bbox_at(bbox, f)
        bw = float(x2 - x1); bh = float(y2 - y1)
        for j in range(J):
            ix, iy = tgt[f, j, 0], tgt[f, j, 1]
            if np.isnan(ix) or np.isnan(iy):
                continue
            # Map image coords → heatmap pixel
            u = (float(ix) - x1) / bw
            v = (float(iy) - y1) / bh
            px = int(round(u * W - 0.5))
            py = int(round(v * H - 0.5))
            if not (0 <= px < W and 0 <= py < H):
                continue
            hm = heatmaps[f, j].astype(np.float32)
            cx, cy = _refine_peak_centroid(hm, py, px, thresh_norm=0.9)
            out[f, j, 0] = x1 + (cx / W) * bw
            out[f, j, 1] = y1 + (cy / H) * bh
    return out


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
    """Build HRNet-W18 model matching the mmpose COCO wholebody hand checkpoint.

    The pretrained checkpoint has:
      - backbone.* — HRNet-W18 backbone (conv1/bn1/conv2/bn2/layer1/
        transition1-3/stage2-4)
      - keypoint_head.final_layer.{0,1,3} — Conv2d(270,270,1)+BN+ReLU+Conv2d(270,21,1)

    mmpose aggregates the 4 final HRNet branches [18,36,72,144 ch] by
    upsampling to the highest resolution and concatenating → 270 ch.
    timm's hrnet_w18 backbone (without features_only) has identical key
    names and shapes, so all 1525 backbone weights load directly.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import timm

    backbone = timm.create_model("hrnet_w18", pretrained=False)

    class HRNetPose(nn.Module):
        def __init__(self, backbone, n_joints=21):
            super().__init__()
            self.bb = backbone
            # mmpose head: 270ch (18+36+72+144) → 21 joints
            self.head = nn.Sequential(
                nn.Conv2d(270, 270, 1),
                nn.BatchNorm2d(270),
                nn.ReLU(inplace=True),
                nn.Conv2d(270, n_joints, 1),
            )

        def forward(self, x):
            # ── Stem ──
            x = self.bb.conv1(x)
            x = self.bb.bn1(x)
            x = self.bb.act1(x)
            x = self.bb.conv2(x)
            x = self.bb.bn2(x)
            x = self.bb.act2(x)

            # ── Stage 1 (residual blocks) ──
            x = self.bb.layer1(x)

            # ── Stages 2-4 (multi-resolution branches) ──
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

            # ── Aggregate: upsample all branches to highest resolution, concat ──
            # xl = [18ch@64×64, 36ch@32×32, 72ch@16×16, 144ch@8×8]
            target = xl[0].shape[2:]
            parts = [xl[0]]
            for feat in xl[1:]:
                parts.append(F.interpolate(feat, size=target, mode="bilinear", align_corners=True))
            cat = torch.cat(parts, dim=1)  # (B, 270, H, W)

            return self.head(cat)

    model = HRNetPose(backbone)

    # Load pretrained mmpose weights with key remapping
    weights_path = _weights_path()
    if weights_path.exists():
        state = torch.load(str(weights_path), map_location="cpu", weights_only=False)
        if "state_dict" in state:
            state = state["state_dict"]

        model_state = model.state_dict()
        loaded = 0
        for k, v in state.items():
            # Remap mmpose keys to our model structure:
            # backbone.X → bb.X (timm backbone stored as self.bb)
            # keypoint_head.final_layer.N → head.N
            new_key = k
            if k.startswith("keypoint_head.final_layer."):
                new_key = "head." + k[len("keypoint_head.final_layer."):]
            elif k.startswith("backbone."):
                new_key = "bb." + k[len("backbone."):]

            if new_key in model_state and v.shape == model_state[new_key].shape:
                model_state[new_key] = v
                loaded += 1

        model.load_state_dict(model_state, strict=False)
        total_checkpoint = len(state)
        logger.info(f"Loaded {loaded}/{total_checkpoint} weight tensors from {weights_path.name}")
        if loaded < total_checkpoint * 0.9:
            logger.warning(f"Only {loaded}/{total_checkpoint} weights loaded — heatmap quality may be poor")

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
    device_callback: Callable[[str], None] | None = None,
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

    # ── Per-frame bboxes (default) — derived from MP per frame, smoothed ─
    # Each frame is cropped by its own tight bbox so the network sees the
    # hand at full resolution even when it moves a lot across the trial.
    # Falls back to a static union bbox when MP is unavailable.
    from .hrnet_bbox import compute_per_frame_bboxes, union_bbox
    bboxes_os_pf = None
    bboxes_od_pf = None
    if bbox_os is None and mp_trial_os is not None:
        bboxes_os_pf = compute_per_frame_bboxes(mp_trial_os)
        bbox_os = union_bbox(bboxes_os_pf)
    if bbox_od is None and mp_trial_od is not None:
        bboxes_od_pf = compute_per_frame_bboxes(mp_trial_od)
        bbox_od = union_bbox(bboxes_od_pf)
    # If a static bbox was passed in (legacy path) but no per-frame
    # detail, broadcast it to all frames so the inference loop still
    # has a (n_frames, 4) array to index.
    if bboxes_os_pf is None and bbox_os is not None:
        bboxes_os_pf = np.tile(np.asarray(bbox_os, dtype=np.float32), (n_frames, 1))
    if bboxes_od_pf is None and bbox_od is not None:
        bboxes_od_pf = np.tile(np.asarray(bbox_od, dtype=np.float32), (n_frames, 1))

    if bbox_os is None:
        raise ValueError("No bounding box for OS camera (run MediaPipe first)")

    logger.info(f"  Bboxes (union): OS={bbox_os}, OD={bbox_od}")
    if bboxes_os_pf is not None:
        logger.info(f"  Per-frame bboxes: OS shape={bboxes_os_pf.shape}, "
                    f"OD shape={bboxes_od_pf.shape if bboxes_od_pf is not None else 'None'}")
    report(20)

    # ── Build model ────────────────────────────────────────────
    # Device priority: CUDA → Apple MPS → CPU.
    # MPS gives a ~5-10× speedup vs CPU on Apple-silicon Macs and works for
    # HRNet-W18 inference (no fp64 ops, supported convs / batchnorms).
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = _build_model()
    model = model.to(device)
    logger.info(f"  HRNet model ready on {device}")
    if device_callback:
        try:
            device_callback(str(device))
        except Exception:
            pass
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

    # Seek to start frame.
    # ``start_frame`` from build_trial_map is on the *global* (concatenated)
    # timeline.  When each trial has its own video file, the file is already
    # trimmed to this trial — so the local seek must be 0, not the global
    # offset (otherwise we seek past EOF and read no frames, leaving the
    # heatmap buffer all-zero).  Detect that case by comparing against the
    # actual video frame count.
    _video_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    seek_frame = start_frame if start_frame + n_frames <= _video_total else 0
    if seek_frame != start_frame:
        logger.info(f"  Per-trial video detected: seeking to 0 instead of global frame {start_frame}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, seek_frame)

    for f_idx in range(n_frames):
        if cancelled():
            cap.release()
            return {"cancelled": True}

        ret, frame = cap.read()
        if not ret:
            break

        # Per-frame bbox: each frame is cropped by its own tight bbox so
        # the network sees the hand at full effective resolution even when
        # the trial has lots of motion across the frame.
        bb_os_f = bboxes_os_pf[f_idx] if bboxes_os_pf is not None else bbox_os
        bb_od_f = bboxes_od_pf[f_idx] if bboxes_od_pf is not None else bbox_od

        # Process OS (left camera)
        if is_stereo:
            frame_os = frame[:, :midline]
        else:
            frame_os = frame

        hm_os = _infer_crop(model, frame_os, list(bb_os_f))
        heatmaps_os[f_idx] = hm_os

        # Process OD (right camera)
        if heatmaps_od is not None and bb_od_f is not None:
            frame_od = frame[:, midline:]
            hm_od = _infer_crop(model, frame_od, list(bb_od_f))
            heatmaps_od[f_idx] = hm_od

        if f_idx % 20 == 0:
            report(25 + (f_idx / n_frames) * 70)

    cap.release()
    report(95)

    # ── Save results ───────────────────────────────────────────
    # Save to skeleton trial dir so the viewer can find them
    from .skeleton_data import _skeleton_dir
    trial_stem = trial["trial_name"]
    out_dir = _skeleton_dir(subject_name) / trial_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hrnet_w18_heatmaps.npz"

    save_dict = {
        "heatmaps_L": heatmaps_os,    # viewer expects heatmaps_L / heatmaps_R
        "trial_idx": trial_idx,
        "start_frame": start_frame,
        "n_frames": n_frames,
    }
    if heatmaps_od is not None:
        save_dict["heatmaps_R"] = heatmaps_od

    np.savez_compressed(str(out_path), **save_dict)

    # Pre-computed MIP (max across the 21 joints) per camera — saves the
    # viewer from fetching all 21 per-joint maps every frame when MIP mode
    # is on.  Separate file keeps schema changes to the heatmap npz
    # unnecessary and allows back-fill.
    try:
        mip_dict = {"heatmaps_L_mip": heatmaps_os.max(axis=1).astype(np.float16)}
        if heatmaps_od is not None:
            mip_dict["heatmaps_R_mip"] = heatmaps_od.max(axis=1).astype(np.float16)
        np.savez_compressed(str(out_dir / "hrnet_w18_mip.npz"), **mip_dict)
    except Exception as e:
        logger.warning(f"Failed to save MIP: {e}")

    # Save crop info as JSON.  Both the legacy union bbox (``crop_L``/
    # ``crop_R``) and the per-frame array (``crop_L_perframe`` /
    # ``crop_R_perframe``) are written — old readers fall back to the
    # union, new readers prefer per-frame for correct heatmap-pixel ↔
    # image-pixel conversions on every frame.
    import json as _json
    crop_info: dict = {"crop_L": list(bbox_os)}
    if bbox_od is not None:
        crop_info["crop_R"] = list(bbox_od)
    if bboxes_os_pf is not None:
        crop_info["crop_L_perframe"] = [
            [round(float(v), 1) for v in row] for row in bboxes_os_pf
        ]
    if bboxes_od_pf is not None:
        crop_info["crop_R_perframe"] = [
            [round(float(v), 1) for v in row] for row in bboxes_od_pf
        ]
    (out_dir / "hand_crop.json").write_text(_json.dumps(crop_info))

    # Peak assignments: MP-guided Hungarian (original algorithm), followed
    # by a centroid refinement within each peak's ≥ 90th-percentile
    # connected region to snap to sub-pixel accuracy.
    try:
        logger.info("  Computing peak assignments...")
        import torch as _torch
        from .skeleton_v3 import _load_hrnet_heatmaps, assign_hm_targets_stereo
        from .skeleton_data import JOINT_NAMES as _JN
        hm_L, _bb_pf_L, hm_R, _bb_pf_R = _load_hrnet_heatmaps(
            subject_name, trial["trial_name"], start_frame, n_frames)
        # Legacy peak-assignment path uses a single union bbox; collapse
        # the per-frame array to its union for compatibility.
        def _union_pf(pf):
            if pf is None: return None
            v = ~np.isnan(pf[:, 0])
            if not v.any(): return None
            return [float(pf[v, 0].min()), float(pf[v, 1].min()),
                    float(pf[v, 2].max()), float(pf[v, 3].max())]
        bbox_hm_L = _union_pf(_bb_pf_L)
        bbox_hm_R = _union_pf(_bb_pf_R)
        if hm_L is not None:
            mp_L_arr = mp_trial_os if mp_trial_os is not None else np.zeros((n_frames, 21, 2))
            mp_R_arr = mp_trial_od if mp_trial_od is not None else np.zeros((n_frames, 21, 2))
            _TIP_OFF = {4:(3,0.43),8:(7,0.32),12:(11,0.40),16:(15,0.40),20:(19,0.55),
                        3:(2,0.28),7:(6,0.19),11:(10,0.12),15:(14,0.17),19:(18,0.02),
                        2:(1,0.12),6:(5,0.10),10:(9,0.13),14:(13,0.04),18:(17,0.05)}
            for arr in [mp_L_arr, mp_R_arr]:
                for j, (p, e) in _TIP_OFF.items():
                    if e > 0.01:
                        arr[:, j] = arr[:, j] + e * (arr[:, j] - arr[:, p])
            hm_L_t = _torch.tensor(hm_L, dtype=_torch.float32)
            hm_R_t = _torch.tensor(hm_R, dtype=_torch.float32) if hm_R is not None else None
            tgt_L, tgt_R = assign_hm_targets_stereo(
                _torch.tensor(mp_L_arr, dtype=_torch.float32),
                _torch.tensor(mp_R_arr, dtype=_torch.float32),
                hm_L_t, hm_R_t, bbox_hm_L, bbox_hm_R)
            tgt_L = tgt_L.cpu().numpy().astype(np.float32)
            tgt_R = tgt_R.cpu().numpy().astype(np.float32) if tgt_R is not None else None
            # Raw argmax peaks (pre-Hungarian, pre-refinement) — used by the
            # HRnet sub-stage for "raw" peak display.
            raw_L = _argmax_peaks(heatmaps_os, bbox_os) if heatmaps_os is not None else None
            raw_R = _argmax_peaks(heatmaps_od, bbox_od) if heatmaps_od is not None else None
            # Refinement: centroid of connected ≥90%-norm-value region.
            tgt_L = _refine_peaks_tensor(tgt_L, heatmaps_os, bbox_os)
            if tgt_R is not None and heatmaps_od is not None:
                tgt_R = _refine_peaks_tensor(tgt_R, heatmaps_od, bbox_od)
            peak_result = {
                "subject": subject_name, "trial": trial["trial_name"],
                "n_frames": n_frames, "bbox_L": list(bbox_os),
                "bbox_R": list(bbox_od) if bbox_od is not None else None,
                "joint_names": _JN,
                "peaks_L": {}, "peaks_R": {},          # centroid-refined
                "peaks_L_raw": {}, "peaks_R_raw": {},  # pure argmax
                "method": "mp_hungarian_+_centroid_refine_90pct",
            }
            def _pack(arr):
                out = {}
                for j in range(21):
                    out[_JN[j]] = [
                        None if np.isnan(arr[f, j, 0])
                        else [round(float(arr[f, j, 0]), 1), round(float(arr[f, j, 1]), 1)]
                        for f in range(arr.shape[0])
                    ]
                return out
            peak_result["peaks_L"] = _pack(tgt_L)
            if tgt_R is not None: peak_result["peaks_R"] = _pack(tgt_R)
            if raw_L is not None: peak_result["peaks_L_raw"] = _pack(raw_L)
            if raw_R is not None: peak_result["peaks_R_raw"] = _pack(raw_R)
            import json as _json2
            peak_path = out_dir / "hrnet_peak_assignments.json"
            peak_path.write_text(_json2.dumps(peak_result, separators=(",",":")))
            logger.info(f"  Saved peaks ({peak_path.stat().st_size / 1e3:.0f} KB)")
    except Exception as e:
        logger.warning(f"  Peak assignment failed (non-fatal): {e}")

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
    # Requested bbox dimensions — kept intact even when the bbox extends past
    # frame edges so the heatmap ↔ saved-bbox coordinate mapping holds.  We
    # clip to the frame for the actual slice, then paste into a zero-padded
    # canvas of the requested size before resizing.  Without padding, an
    # asymmetric clip would distort aspect ratio and shift the heatmap on
    # display for any bbox with negative y/x or extents past the frame.
    bx1, by1 = int(bbox[0]), int(bbox[1])
    bx2, by2 = int(bbox[2]), int(bbox[3])
    bw, bh = bx2 - bx1, by2 - by1
    if bw <= 0 or bh <= 0:
        return np.zeros((N_JOINTS, HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float16)
    sx1, sy1 = max(0, bx1), max(0, by1)
    sx2, sy2 = min(w, bx2), min(h, by2)
    if sx2 <= sx1 or sy2 <= sy1:
        return np.zeros((N_JOINTS, HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float16)
    crop = np.zeros((bh, bw, frame.shape[2]), dtype=frame.dtype)
    crop[sy1 - by1:sy1 - by1 + (sy2 - sy1),
         sx1 - bx1:sx1 - bx1 + (sx2 - sx1)] = frame[sy1:sy2, sx1:sx2]
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
        # Auto-detect device from model parameters
        dev = next(model.parameters()).device
        tensor = tensor.to(dev)
        output = model(tensor)
        # output shape: (1, 21, H', W') — resize to 64x64
        hm = output[0].cpu().numpy()  # (21, H', W')

    # Resize each joint's heatmap to 64x64
    result = np.zeros((N_JOINTS, HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float32)
    n_out = min(hm.shape[0], N_JOINTS)
    for j in range(n_out):
        result[j] = cv2.resize(hm[j], (HEATMAP_SIZE, HEATMAP_SIZE))

    # Apply sigmoid to convert logits to probabilities, then normalize
    # globally (not per-joint) to preserve relative confidence
    from scipy.special import expit
    result = expit(result).astype(np.float32)
    mx = result.max()
    if mx > 0:
        result /= mx

    return result.astype(np.float16)
