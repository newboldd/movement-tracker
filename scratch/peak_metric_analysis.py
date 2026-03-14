"""Peak detection using dissimilarity metric.

For each frame, compute:
  peak_metric[i] = distance(frame[i], frame[i-1]) + distance(frame[i], frame[i+1])
                   - distance(frame[i-1], frame[i+1])

High values indicate frames surrounded by more-similar neighbors (characteristic of peaks).
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
CROP_SIZES = [72, 144]  # crop_half values: 144×144 and 288×288 pixel boxes
SMOOTHING_SIGMA = 3.0
ZOOM_FRAMES = 300
OUTPUT_DIR = Path("scratch/peak_metric_output")
OUTPUT_DIR.mkdir(exist_ok=True)

from dlc_app.config import get_settings
from dlc_app.services.video import build_trial_map
from dlc_app.services.dlc_predictions import get_corrections_with_dlc_fallback

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
    """Mean squared difference per pixel (normalized by box size)."""
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

def compute_motion_and_peak_metric(subject_name: str, crop_half: int = 72) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute motion (MSD between consecutive frames) and peak metric.

    Peak metric[i] = MSD(frame[i], frame[i-1]) + MSD(frame[i], frame[i+1])
                     - MSD(frame[i-1], frame[i+1])

    High values indicate frames maximally dissimilar from neighbors (peaks).
    """
    settings = get_settings()
    cam_names = settings.camera_names

    print(f"Loading trial map for {subject_name}…")
    trials = build_trial_map(subject_name)
    total_frames = trials[-1]["end_frame"] + 1
    print(f"  {len(trials)} trials, {total_frames} total frames")

    print(f"Loading coordinates…")
    stage_data = get_corrections_with_dlc_fallback(subject_name)
    if stage_data is None:
        raise RuntimeError("No coordinate data found for subject")

    motion_signal = np.zeros(total_frames, dtype=np.float64)
    peak_metric = np.zeros(total_frames, dtype=np.float64)

    # First pass: extract all crops
    crops_by_frame = {}

    for trial in trials:
        vpath = trial["video_path"]
        sf = trial["start_frame"]
        ef = trial["end_frame"]
        n_local = trial["frame_count"]
        trial_name = trial["trial_name"]
        print(f"  Trial {trial_name}: global {sf}–{ef} ({n_local} frames)")

        cap = cv2.VideoCapture(vpath)

        for local_idx in range(n_local):
            global_idx = sf + local_idx
            ret, frame_bgr = cap.read()
            if not ret:
                break

            h, w = frame_bgr.shape[:2]
            midline = w // 2

            crops_by_frame[global_idx] = {}

            # Process each camera
            for cam_idx, cam in enumerate(cam_names):
                # Crop the stereo frame half for this camera
                if cam_idx == 0:
                    cam_frame = frame_bgr[:, :midline]
                else:
                    cam_frame = frame_bgr[:, midline:]

                cam_data = stage_data.get(cam, {})

                # Get thumb and index coords
                thumb_list = cam_data.get("thumb")
                index_list = cam_data.get("index")

                if thumb_list is None or index_list is None or global_idx >= len(thumb_list) or global_idx >= len(index_list):
                    crops_by_frame[global_idx][cam] = None
                    continue

                thumb = thumb_list[global_idx]
                index = index_list[global_idx]

                # Extract fixed-size crop around both fingers
                if thumb is None or index is None:
                    crops_by_frame[global_idx][cam] = None
                else:
                    x_min = int(round(min(thumb[0], index[0]))) - 30
                    y_min = int(round(min(thumb[1], index[1]))) - 30
                    x_max = int(round(max(thumb[0], index[0]))) + 30
                    y_max = int(round(max(thumb[1], index[1]))) + 30

                    # Clip to frame bounds
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(cam_frame.shape[1], x_max)
                    y_max = min(cam_frame.shape[0], y_max)

                    # Check validity
                    if x_max <= x_min or y_max <= y_min:
                        crops_by_frame[global_idx][cam] = None
                    else:
                        crop_region = cam_frame[y_min:y_max, x_min:x_max]
                        crop = cv2.cvtColor(crop_region, cv2.COLOR_BGR2GRAY).astype(np.float32)
                        crops_by_frame[global_idx][cam] = crop

        cap.release()

    # Second pass: compute motion and peak metrics
    prev_crop = None
    for global_idx in range(total_frames):
        if global_idx not in crops_by_frame:
            prev_crop = None
            continue

        frame_motion = 0.0
        frame_peak = 0.0
        n_valid = 0

        for cam in cam_names:
            crop = crops_by_frame[global_idx].get(cam)

            # Motion metric
            if prev_crop is not None and crop is not None and crop.shape == prev_crop.shape:
                frame_motion += msd(crop, prev_crop)
                n_valid += 1

            prev_crop = crop

        # Peak metric (requires lookahead)
        if global_idx > 0 and global_idx < total_frames - 1:
            prev_crops = crops_by_frame.get(global_idx - 1, {})
            curr_crops = crops_by_frame.get(global_idx, {})
            next_crops = crops_by_frame.get(global_idx + 1, {})

            for cam in cam_names:
                prev = prev_crops.get(cam)
                curr = curr_crops.get(cam)
                nxt = next_crops.get(cam)

                if prev is not None and curr is not None and nxt is not None and \
                   prev.shape == curr.shape == nxt.shape:
                    d_curr_prev = msd(curr, prev)
                    d_curr_next = msd(curr, nxt)
                    d_prev_next = msd(prev, nxt)
                    # Peak metric: curr frame dissimilarity relative to neighbor dissimilarity
                    peak_value = d_curr_prev + d_curr_next - d_prev_next
                    frame_peak += peak_value
                    n_valid += 1

        if n_valid > 0:
            motion_signal[global_idx] = frame_motion / n_valid if global_idx > 0 else 0
            peak_metric[global_idx] = frame_peak / n_valid if global_idx > 0 and global_idx < total_frames - 1 else 0

    print(f"    done.")
    return motion_signal, peak_metric


# ── Plotting ──────────────────────────────────────────────────────────────────

EVENT_COLORS = {"open": "#2196F3", "peak": "#FF9800", "close": "#E91E63"}


def plot_trial_comparison(trial: dict, motion_144: np.ndarray, peak_144: np.ndarray,
                         motion_288: np.ndarray, peak_288: np.ndarray,
                         sm_motion_144: np.ndarray, sm_peak_144: np.ndarray,
                         sm_motion_288: np.ndarray, sm_peak_288: np.ndarray,
                         events: dict[str, list[int]], out_dir: Path) -> None:
    """Create per-trial comparison plots for 144px vs 288px boxes."""
    sf = trial["start_frame"]
    ef = trial["end_frame"]
    trial_name = trial["trial_name"]

    frames = np.arange(sf, ef + 1)

    # Get trial data
    motion_144_trial = motion_144[sf:ef + 1]
    peak_144_trial = peak_144[sf:ef + 1]
    motion_288_trial = motion_288[sf:ef + 1]
    peak_288_trial = peak_288[sf:ef + 1]

    sm_motion_144_trial = sm_motion_144[sf:ef + 1]
    sm_peak_144_trial = sm_peak_144[sf:ef + 1]
    sm_motion_288_trial = sm_motion_288[sf:ef + 1]
    sm_peak_288_trial = sm_peak_288[sf:ef + 1]

    # Choose zoom window (300 frames centered on trial middle)
    mid = (sf + ef) // 2
    z_sf = max(sf, mid - ZOOM_FRAMES // 2)
    z_ef = min(ef, z_sf + ZOOM_FRAMES - 1)

    z_frames = np.arange(z_sf, z_ef + 1)

    # Get zoom data
    z_motion_144 = motion_144[z_sf:z_ef + 1]
    z_peak_144 = peak_144[z_sf:z_ef + 1]
    z_motion_288 = motion_288[z_sf:z_ef + 1]
    z_peak_288 = peak_288[z_sf:z_ef + 1]

    z_sm_motion_144 = sm_motion_144[z_sf:z_ef + 1]
    z_sm_peak_144 = sm_peak_144[z_sf:z_ef + 1]
    z_sm_motion_288 = sm_motion_288[z_sf:z_ef + 1]
    z_sm_peak_288 = sm_peak_288[z_sf:z_ef + 1]

    # Y-axis limits for full trial
    nz_motion_144 = motion_144_trial[motion_144_trial > 0]
    nz_motion_288 = motion_288_trial[motion_288_trial > 0]
    ymax_motion = max(
        float(np.percentile(nz_motion_144, 98)) if len(nz_motion_144) else 1.0,
        float(np.percentile(nz_motion_288, 98)) if len(nz_motion_288) else 1.0
    )

    nz_peak_144 = np.abs(peak_144_trial[peak_144_trial != 0])
    nz_peak_288 = np.abs(peak_288_trial[peak_288_trial != 0])
    ymax_peak = max(
        float(np.percentile(nz_peak_144, 98)) if len(nz_peak_144) else 1.0,
        float(np.percentile(nz_peak_288, 98)) if len(nz_peak_288) else 1.0
    )

    # Create figure with 4 subplots (2 rows, 2 columns)
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

    ax_motion_144 = fig.add_subplot(gs[0, 0])
    ax_peak_144 = fig.add_subplot(gs[1, 0], sharex=ax_motion_144)
    ax_motion_288 = fig.add_subplot(gs[0, 1], sharey=ax_motion_144)
    ax_peak_288 = fig.add_subplot(gs[1, 1], sharex=ax_motion_288, sharey=ax_peak_144)

    # ── 144px Motion ──
    ax_motion_144.plot(frames, motion_144_trial, color="silver", alpha=0.5, linewidth=0.6)
    ax_motion_144.plot(frames, sm_motion_144_trial, color="#1565C0", linewidth=1.0)
    ax_motion_144.set_ylim(0, ymax_motion * 1.05)
    ax_motion_144.set_ylabel("Motion MSD", fontsize=9)
    ax_motion_144.set_title(f"144×144 px – Motion", fontsize=10)
    ax_motion_144.axvspan(z_sf, z_ef, color="gold", alpha=0.12, zorder=0)
    for etype, color in EVENT_COLORS.items():
        rug_frames = [f for f in events.get(etype, []) if sf <= f <= ef]
        if rug_frames:
            ax_motion_144.vlines(rug_frames, 0, ymax_motion, color=color, linewidth=0.4, alpha=0.3)

    # ── 144px Peak ──
    ax_peak_144.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax_peak_144.plot(frames, peak_144_trial, color="silver", alpha=0.5, linewidth=0.6)
    ax_peak_144.plot(frames, sm_peak_144_trial, color="#4CAF50", linewidth=1.0)
    ax_peak_144.set_ylim(-ymax_peak * 1.05, ymax_peak * 1.05)
    ax_peak_144.set_ylabel("Peak Metric", fontsize=9)
    ax_peak_144.set_xlabel("Global frame index", fontsize=9)
    for etype, color in EVENT_COLORS.items():
        rug_frames = [f for f in events.get(etype, []) if sf <= f <= ef]
        if rug_frames:
            ax_peak_144.vlines(rug_frames, -ymax_peak, ymax_peak, color=color, linewidth=0.4, alpha=0.3)

    # ── 288px Motion ──
    ax_motion_288.plot(frames, motion_288_trial, color="silver", alpha=0.5, linewidth=0.6)
    ax_motion_288.plot(frames, sm_motion_288_trial, color="#1565C0", linewidth=1.0)
    ax_motion_288.set_ylabel("Motion MSD", fontsize=9)
    ax_motion_288.set_title(f"288×288 px – Motion", fontsize=10)
    ax_motion_288.axvspan(z_sf, z_ef, color="gold", alpha=0.12, zorder=0)
    for etype, color in EVENT_COLORS.items():
        rug_frames = [f for f in events.get(etype, []) if sf <= f <= ef]
        if rug_frames:
            ax_motion_288.vlines(rug_frames, 0, ymax_motion, color=color, linewidth=0.4, alpha=0.3)

    # ── 288px Peak ──
    ax_peak_288.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax_peak_288.plot(frames, peak_288_trial, color="silver", alpha=0.5, linewidth=0.6)
    ax_peak_288.plot(frames, sm_peak_288_trial, color="#4CAF50", linewidth=1.0)
    ax_peak_288.set_ylabel("Peak Metric", fontsize=9)
    ax_peak_288.set_xlabel("Global frame index", fontsize=9)
    for etype, color in EVENT_COLORS.items():
        rug_frames = [f for f in events.get(etype, []) if sf <= f <= ef]
        if rug_frames:
            ax_peak_288.vlines(rug_frames, -ymax_peak, ymax_peak, color=color, linewidth=0.4, alpha=0.3)

    fig.suptitle(f"{SUBJECT}  –  {trial_name}  –  Peak Metric Comparison (144 vs 288 px)",
                 fontsize=11, fontweight="bold")

    plt.tight_layout()
    out_path = out_dir / f"{SUBJECT}_{trial_name}_peak_comparison.png"
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_metrics(motion: np.ndarray, peak: np.ndarray, events: dict[str, list[int]],
                 crop_size: int, out_dir: Path) -> None:
    """Plot motion and peak metrics side by side."""
    frames = np.arange(len(motion))

    sm_motion = gaussian_smooth(motion, SMOOTHING_SIGMA)
    sm_peak = gaussian_smooth(peak, SMOOTHING_SIGMA)

    # Use percentile-based y-axis limits
    nz_motion = motion[motion > 0]
    nz_peak = peak[peak != 0]

    ymax_motion = float(np.percentile(nz_motion, 98)) if len(nz_motion) else 1.0
    ymax_peak = float(np.percentile(np.abs(nz_peak), 98)) if len(nz_peak) else 1.0
    ymin_peak = -ymax_peak

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)

    # Motion metric
    ax1.plot(frames, motion, color="silver", alpha=0.4, linewidth=0.5)
    ax1.plot(frames, sm_motion, color="#1565C0", linewidth=1.0, label="Motion (MSD)")
    ax1.set_ylim(0, ymax_motion * 1.1)
    ax1.set_ylabel("Mean MSD (px²)")
    ax1.set_title(f"{SUBJECT} – {crop_size}×{crop_size} px box – Motion & Peak Metrics")
    ax1.legend(fontsize=9)
    for etype, color in EVENT_COLORS.items():
        rug_frames = events.get(etype, [])
        if rug_frames:
            ax1.vlines(rug_frames, 0, ymax_motion, color=color, linewidth=0.4, alpha=0.3)

    # Peak metric
    ax2.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax2.plot(frames, peak, color="silver", alpha=0.4, linewidth=0.5)
    ax2.plot(frames, sm_peak, color="#4CAF50", linewidth=1.0, label="Peak metric (dissimilarity)")
    ax2.set_ylim(ymin_peak * 1.1, ymax_peak * 1.1)
    ax2.set_ylabel("Peak Metric")
    ax2.set_xlabel("Global frame index")
    ax2.legend(fontsize=9)
    for etype, color in EVENT_COLORS.items():
        rug_frames = events.get(etype, [])
        if rug_frames:
            ax2.vlines(rug_frames, ymin_peak, ymax_peak, color=color, linewidth=0.4, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / f"{SUBJECT}_{crop_size}px_metrics.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  → saved {out_path}")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"=== Peak Metric Analysis: {SUBJECT} ===\n")

    events = load_events(SUBJECT)
    print(f"Events loaded: {', '.join(f'{k}:{len(v)}' for k,v in events.items())}\n")

    # Compute metrics for both crop sizes
    metrics = {}
    for crop_half in CROP_SIZES:
        crop_size = crop_half * 2
        print(f"Computing metrics for {crop_size}×{crop_size} px box…")
        motion, peak = compute_motion_and_peak_metric(SUBJECT, crop_half)
        metrics[crop_size] = {
            'motion': motion,
            'peak': peak,
            'motion_smooth': gaussian_smooth(motion, SMOOTHING_SIGMA),
            'peak_smooth': gaussian_smooth(peak, SMOOTHING_SIGMA),
        }

    # Plot individual metric plots
    for crop_size in sorted(metrics.keys()):
        print(f"Plotting individual metrics for {crop_size}×{crop_size}…")
        plot_metrics(metrics[crop_size]['motion'], metrics[crop_size]['peak'],
                    events, crop_size, OUTPUT_DIR)

    # Plot per-trial comparisons
    print(f"\nGenerating per-trial comparisons…")
    trials = build_trial_map(SUBJECT)
    for trial in trials:
        print(f"  Plotting {trial['trial_name']}…")
        try:
            plot_trial_comparison(
                trial,
                metrics[144]['motion'], metrics[144]['peak'],
                metrics[288]['motion'], metrics[288]['peak'],
                metrics[144]['motion_smooth'], metrics[144]['peak_smooth'],
                metrics[288]['motion_smooth'], metrics[288]['peak_smooth'],
                events, OUTPUT_DIR
            )
            print(f"    → saved {SUBJECT}_{trial['trial_name']}_peak_comparison.png")
        except Exception as e:
            print(f"    Error: {e}")

    # Save peak metric data for zoom plots
    np.save(OUTPUT_DIR / "peak_144.npy", metrics[144]['peak'])
    np.save(OUTPUT_DIR / "peak_288.npy", metrics[288]['peak'])

    print(f"\nDone. Output in {OUTPUT_DIR.resolve()}")
