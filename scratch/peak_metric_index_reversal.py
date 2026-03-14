"""Peak detection using index finger reversal pattern.

At peaks, the index finger reverses direction, so preceding and following frames
have similar finger positions. This metric finds frames where frame[i-1] and frame[i+1]
are similar by computing SSD of a small crop around the index finger tip.

peak_metric[i] = msd(crop[i-1], crop[i+1])
Low values indicate reversal points (good candidates for peaks).
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Config ──────────────────────────────────────────────────────────────────

SUBJECT = "MSA02"
TRIAL_NAME = "MSA02_L1"  # Focus on this trial
CROP_HALF = 24  # 48×48 pixel box around index finger tip
SMOOTHING_SIGMA = 2.0
OUTPUT_DIR = Path("scratch/peak_metric_output")
OUTPUT_DIR.mkdir(exist_ok=True)

from dlc_app.config import get_settings
from dlc_app.services.video import build_trial_map, extract_frame
from dlc_app.services.dlc_predictions import get_dlc_predictions_for_session

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_events(subject_name: str) -> dict[str, list[int]]:
    """Load events.csv for a subject."""
    settings = get_settings()
    events_path = Path(settings.dlc_path) / subject_name / "events.csv"
    events: dict[str, list[int]] = {"open": [], "peak": [], "close": []}
    if not events_path.exists():
        print(f"  [warn] no events.csv at {events_path}")
        return events
    with open(events_path, newline="") as f:
        for row in csv.DictReader(f):
            etype = row["event_type"].strip()
            if etype in events:
                events[etype].append(int(row["frame_num"]))
    for v in events.values():
        v.sort()
    return events


def msd(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared difference per pixel."""
    if a.shape != b.shape:
        return float('inf')
    diff = a - b
    return float(np.sum(diff * diff) / diff.size)


def gaussian_smooth(arr: np.ndarray, sigma: float) -> np.ndarray:
    """1-D Gaussian smoothing."""
    radius = int(3 * sigma)
    if radius < 1:
        return arr.copy()
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(arr, kernel, mode="same")


# ── Main computation ──────────────────────────────────────────────────────────

def compute_distance_trace(subject_name: str, trial_start: int, trial_end: int) -> np.ndarray:
    """Compute 3D distance trace for the trial using DLC data."""
    settings = get_settings()
    cam_names = settings.camera_names

    # Load DLC predictions (includes distance trace)
    dlc_data = get_dlc_predictions_for_session(subject_name)
    if dlc_data is None or "distances" not in dlc_data:
        print("  [warn] No DLC distance data found")
        return np.zeros(trial_end - trial_start + 1)

    distances = dlc_data["distances"]
    trial_distances = distances[trial_start:trial_end + 1]
    return np.array(trial_distances)


def compute_index_reversal_metric(subject_name: str, trial: dict) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """
    Compute index finger reversal metrics for a trial at multiple offsets.

    For each frame and offset k, measure how similar frame[i-k] and frame[i+k] are
    using a small crop around the index finger tip.

    Returns:
        (distance_trace, {offset: reversal_metric, ...})
    """
    settings = get_settings()
    cam_names = settings.camera_names

    trial_name = trial["trial_name"]
    vpath = trial["video_path"]
    sf = trial["start_frame"]
    ef = trial["end_frame"]
    n_frames = trial["frame_count"]

    print(f"Loading DLC coordinates for {trial_name}…")
    dlc_data = get_dlc_predictions_for_session(subject_name)
    if dlc_data is None:
        raise RuntimeError("No DLC data found")

    # Extract index finger coordinates for each camera
    index_coords = {}
    for cam in cam_names:
        if cam in dlc_data and "index" in dlc_data[cam]:
            index_coords[cam] = dlc_data[cam]["index"]
        else:
            index_coords[cam] = None

    # Load distance trace for reference
    distance_trace = compute_distance_trace(subject_name, sf, ef)

    print(f"Processing trial {trial_name}: {n_frames} frames…")
    cap = cv2.VideoCapture(vpath)

    reversal_metrics = {1: np.zeros(n_frames, dtype=np.float64),
                        2: np.zeros(n_frames, dtype=np.float64),
                        3: np.zeros(n_frames, dtype=np.float64),
                        'asymmetric': np.zeros(n_frames, dtype=np.float64)}
    crops_by_frame = {}

    # First pass: extract crops around index finger tip for each frame
    for local_idx in range(n_frames):
        global_idx = sf + local_idx
        ret, frame_bgr = cap.read()
        if not ret:
            break

        h, w = frame_bgr.shape[:2]
        midline = w // 2

        crops_by_frame[local_idx] = {}

        # Process each camera
        for cam_idx, cam in enumerate(cam_names):
            # Split stereo frame
            if cam_idx == 0:
                cam_frame = frame_bgr[:, :midline]
            else:
                cam_frame = frame_bgr[:, midline:]

            # Get index finger position for this frame
            if index_coords[cam] is None or global_idx >= len(index_coords[cam]):
                crops_by_frame[local_idx][cam] = None
                continue

            index_pos = index_coords[cam][global_idx]
            if index_pos is None:
                crops_by_frame[local_idx][cam] = None
                continue

            # Extract crop around index finger tip
            x, y = int(round(index_pos[0])), int(round(index_pos[1]))
            x_min = max(0, x - CROP_HALF)
            y_min = max(0, y - CROP_HALF)
            x_max = min(cam_frame.shape[1], x + CROP_HALF)
            y_max = min(cam_frame.shape[0], y + CROP_HALF)

            if x_max <= x_min or y_max <= y_min:
                crops_by_frame[local_idx][cam] = None
            else:
                crop_region = cam_frame[y_min:y_max, x_min:x_max]
                crop = cv2.cvtColor(crop_region, cv2.COLOR_BGR2GRAY).astype(np.float32)
                crops_by_frame[local_idx][cam] = crop

    cap.release()

    # Second pass: compute reversal metrics at different offsets
    # For offset k, compare frame[i-k] and frame[i+k]
    for offset in [1, 2, 3]:
        for local_idx in range(offset, n_frames - offset):
            prev_crops = crops_by_frame.get(local_idx - offset, {})
            next_crops = crops_by_frame.get(local_idx + offset, {})

            metric_sum = 0.0
            n_valid = 0

            for cam in cam_names:
                prev = prev_crops.get(cam)
                nxt = next_crops.get(cam)

                # Compare frame[i-offset] and frame[i+offset]
                if prev is not None and nxt is not None and prev.shape == nxt.shape:
                    metric_sum += msd(prev, nxt)
                    n_valid += 1

            if n_valid > 0:
                reversal_metrics[offset][local_idx] = metric_sum / n_valid

    # Third pass: asymmetric metric
    # For each frame i, find min distance between frame[i+1] and any of frames [i, i-1, i-2, i-3]
    for local_idx in range(3, n_frames - 1):
        next_crops = crops_by_frame.get(local_idx + 1, {})

        # Find minimum similarity across preceding 3 frames
        min_metric = float('inf')

        for prev_offset in range(0, 4):  # Check frames i, i-1, i-2, i-3
            prev_crops = crops_by_frame.get(local_idx - prev_offset, {})

            metric_sum = 0.0
            n_valid = 0

            for cam in cam_names:
                prev = prev_crops.get(cam)
                nxt = next_crops.get(cam)

                if prev is not None and nxt is not None and prev.shape == nxt.shape:
                    metric_sum += msd(prev, nxt)
                    n_valid += 1

            if n_valid > 0:
                avg_metric = metric_sum / n_valid
                min_metric = min(min_metric, avg_metric)

        if min_metric != float('inf'):
            reversal_metrics['asymmetric'][local_idx] = min_metric

    print(f"  done.")
    return distance_trace, reversal_metrics


def plot_metrics_with_events(distance: np.ndarray, reversal_metrics: dict[int, np.ndarray],
                             trial: dict, events: dict[str, list[int]],
                             out_dir: Path) -> None:
    """Plot distance trace and reversal metrics at different offsets."""
    sf = trial["start_frame"]
    ef = trial["end_frame"]
    trial_name = trial["trial_name"]

    frames = np.arange(sf, ef + 1)

    sm_distance = gaussian_smooth(distance, SMOOTHING_SIGMA)

    # Y-axis limits
    nz_dist = distance[distance > 0]
    ymax_dist = float(np.percentile(nz_dist, 98)) if len(nz_dist) else 1.0

    EVENT_COLORS = {"open": "#2196F3", "peak": "#FF9800", "close": "#E91E63"}
    OFFSET_COLORS = {1: "#4CAF50", 2: "#FF6F00", 3: "#9C27B0", 'asymmetric': "#E91E63"}

    fig, axes = plt.subplots(5, 1, figsize=(20, 16), sharex=True)

    # Distance trace (existing metric)
    ax1 = axes[0]
    ax1.plot(frames, distance, color="silver", alpha=0.4, linewidth=0.5)
    ax1.plot(frames, sm_distance, color="#1565C0", linewidth=1.5, label="Distance trace (DLC)")
    ax1.set_ylim(0, ymax_dist * 1.1)
    ax1.set_ylabel("3D Distance (mm)", fontsize=10)
    ax1.set_title(f"{SUBJECT} – {trial_name} – Distance Trace vs Index Reversal Metrics", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9, loc='upper right')
    for etype, color in EVENT_COLORS.items():
        rug_frames = [f for f in events.get(etype, []) if sf <= f <= ef]
        if rug_frames:
            ax1.vlines(rug_frames, 0, ymax_dist, color=color, linewidth=0.6, alpha=0.4)

    # Reversal metrics at different offsets
    for plot_idx, offset in enumerate([1, 2, 3, 'asymmetric'], 1):
        ax = axes[plot_idx]
        reversal = reversal_metrics[offset]
        sm_reversal = gaussian_smooth(reversal, SMOOTHING_SIGMA)

        nz_rev = reversal[reversal > 0]
        ymax_rev = float(np.percentile(nz_rev, 98)) if len(nz_rev) else 1.0

        ax.plot(frames, reversal, color="silver", alpha=0.4, linewidth=0.5)

        if offset == 'asymmetric':
            label = "Index reversal (min of frame[i+1] vs preceding 3)"
        else:
            label = f"Index reversal (frame[i-{offset}] vs frame[i+{offset}])"

        ax.plot(frames, sm_reversal, color=OFFSET_COLORS[offset], linewidth=1.5, label=label)
        ax.set_ylim(0, ymax_rev * 1.1)
        ax.set_ylabel("MSD (frame similarity)", fontsize=10)
        ax.legend(fontsize=9, loc='upper right')
        for etype, color in EVENT_COLORS.items():
            rug_frames = [f for f in events.get(etype, []) if sf <= f <= ef]
            if rug_frames:
                ax.vlines(rug_frames, 0, ymax_rev, color=color, linewidth=0.6, alpha=0.4)

    axes[-1].set_xlabel("Global frame index", fontsize=10)

    plt.tight_layout()
    out_path = out_dir / f"{SUBJECT}_{trial_name}_reversal_metric.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  → saved {out_path}")


def plot_zoom(distance: np.ndarray, reversal_metrics: dict[int, np.ndarray],
              trial: dict, events: dict[str, list[int]],
              zoom_frames: int = 300, out_dir: Path = None) -> None:
    """Plot zoomed view with all reversal metrics."""
    sf = trial["start_frame"]
    ef = trial["end_frame"]
    trial_name = trial["trial_name"]

    # Zoom window centered on trial
    mid = (sf + ef) // 2
    z_sf = max(sf, mid - zoom_frames // 2)
    z_ef = min(ef, z_sf + zoom_frames - 1)

    z_frames = np.arange(z_sf, z_ef + 1)
    z_distance = distance[z_sf - sf:z_ef - sf + 1]

    sm_distance = gaussian_smooth(z_distance, SMOOTHING_SIGMA)

    # Y-axis limits from full trial
    nz_dist = distance[distance > 0]
    ymax_dist = float(np.percentile(nz_dist, 98)) if len(nz_dist) else 1.0

    EVENT_COLORS = {"open": "#2196F3", "peak": "#FF9800", "close": "#E91E63"}
    OFFSET_COLORS = {1: "#4CAF50", 2: "#FF6F00", 3: "#9C27B0", 'asymmetric': "#E91E63"}

    fig, axes = plt.subplots(5, 1, figsize=(18, 14), sharex=True)

    # Distance trace
    ax1 = axes[0]
    ax1.plot(z_frames, z_distance, color="silver", alpha=0.4, linewidth=0.5, label="Distance (raw)")
    ax1.plot(z_frames, sm_distance, color="#1565C0", linewidth=1.5, label="Distance (smoothed)")
    ax1.set_ylim(0, ymax_dist * 1.1)
    ax1.set_ylabel("3D Distance (mm)", fontsize=10)
    ax1.set_title(f"{SUBJECT} – {trial_name} – Zoom (~{zoom_frames} frames)", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9, loc='upper right')
    for etype, color in EVENT_COLORS.items():
        rug_frames = [f for f in events.get(etype, []) if z_sf <= f <= z_ef]
        if rug_frames:
            ax1.vlines(rug_frames, 0, ymax_dist, color=color, linewidth=0.8, alpha=0.5)

    # Reversal metrics at different offsets
    for plot_idx, offset in enumerate([1, 2, 3, 'asymmetric'], 1):
        ax = axes[plot_idx]
        reversal = reversal_metrics[offset]
        z_reversal = reversal[z_sf - sf:z_ef - sf + 1]
        sm_reversal = gaussian_smooth(z_reversal, SMOOTHING_SIGMA)

        nz_rev = reversal[reversal > 0]
        ymax_rev = float(np.percentile(nz_rev, 98)) if len(nz_rev) else 1.0

        ax.plot(z_frames, z_reversal, color="silver", alpha=0.4, linewidth=0.5, label="(raw)")

        if offset == 'asymmetric':
            label = "min of frame[i+1] vs preceding 3 (smoothed)"
        else:
            label = f"frame[i-{offset}] vs frame[i+{offset}] (smoothed)"

        ax.plot(z_frames, sm_reversal, color=OFFSET_COLORS[offset], linewidth=1.5, label=label)
        ax.set_ylim(0, ymax_rev * 1.1)
        ax.set_ylabel("MSD", fontsize=9)
        ax.legend(fontsize=8, loc='upper right')
        for etype, color in EVENT_COLORS.items():
            rug_frames = [f for f in events.get(etype, []) if z_sf <= f <= z_ef]
            if rug_frames:
                ax.vlines(rug_frames, 0, ymax_rev, color=color, linewidth=0.8, alpha=0.5)

    axes[-1].set_xlabel("Global frame index", fontsize=10)

    plt.tight_layout()
    out_path = out_dir / f"{SUBJECT}_{trial_name}_reversal_zoom.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  → saved {out_path}")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"=== Index Reversal Peak Metric Analysis: {SUBJECT} ===\n")

    events = load_events(SUBJECT)
    print(f"Events loaded: {', '.join(f'{k}:{len(v)}' for k,v in events.items())}\n")

    trials = build_trial_map(SUBJECT)
    trial = next((t for t in trials if t["trial_name"] == TRIAL_NAME), None)

    if not trial:
        print(f"Trial {TRIAL_NAME} not found!")
        sys.exit(1)

    print(f"Computing metrics for {TRIAL_NAME}…")
    distance_trace, reversal_metrics = compute_index_reversal_metric(SUBJECT, trial)

    print(f"\nGenerating plots…")
    plot_metrics_with_events(distance_trace, reversal_metrics, trial, events, OUTPUT_DIR)
    plot_zoom(distance_trace, reversal_metrics, trial, events, zoom_frames=300, out_dir=OUTPUT_DIR)

    print(f"\nDone. Output in {OUTPUT_DIR.resolve()}")
