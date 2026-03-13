"""Motion metric using dynamic bounding box encompassing both thumb and index.

Instead of fixed-size crops around each fingertip independently, compute
a single bounding box that contains both thumb and index fingers. This:
  - Scales naturally with hand open/close state
  - Captures relative motion between fingers
  - Works better for peak detection (full separation visible)
  - Reduces background noise

The SSD measures how much the entire hand region changes between frames.
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
PADDING = 30        # pixels of padding around thumb/index bounding box
SMOOTHING_SIGMA = 3.0
ZOOM_FRAMES = 300
OUTPUT_DIR = Path("scratch/motion_metric_dynamic_box")
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


def extract_dynamic_box_gray(frame_bgr: np.ndarray, thumb: tuple[float, float],
                            index: tuple[float, float], padding: int) -> Optional[np.ndarray]:
    """Extract grayscale crop from bounding box encompassing both thumb and index."""
    if thumb is None or index is None:
        return None

    h, w = frame_bgr.shape[:2]

    # Bounding box
    x_min = int(round(min(thumb[0], index[0]))) - padding
    y_min = int(round(min(thumb[1], index[1]))) - padding
    x_max = int(round(max(thumb[0], index[0]))) + padding
    y_max = int(round(max(thumb[1], index[1]))) + padding

    # Clip to frame bounds
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)

    # Check validity
    if x_max <= x_min or y_max <= y_min:
        return None

    crop = frame_bgr[y_min:y_max, x_min:x_max]
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)


def ssd(a: np.ndarray, b: np.ndarray) -> float:
    """Sum of squared differences."""
    diff = a - b
    return float(np.sum(diff * diff))


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

def compute_motion_metric_fixed(subject_name: str, crop_half: int = 72) -> np.ndarray:
    """
    Compute SSD using fixed-size bounding boxes (for comparison).
    Returns just the SSD signal.
    """
    settings = get_settings()
    cam_names = settings.camera_names

    trials = build_trial_map(subject_name)
    total_frames = trials[-1]["end_frame"] + 1
    stage_data = get_corrections_with_dlc_fallback(subject_name)

    ssd_signal = np.zeros(total_frames, dtype=np.float64)

    for trial in trials:
        vpath = trial["video_path"]
        sf = trial["start_frame"]
        ef = trial["end_frame"]
        n_local = trial["frame_count"]

        cap = cv2.VideoCapture(vpath)
        prev_crops: dict = {}

        for local_idx in range(n_local):
            global_idx = sf + local_idx
            ret, frame_bgr = cap.read()
            if not ret:
                break

            h, w = frame_bgr.shape[:2]
            midline = w // 2
            frame_ssd = 0.0
            n_valid = 0

            for cam_idx, cam in enumerate(cam_names):
                if cam_idx == 0:
                    cam_frame = frame_bgr[:, :midline]
                else:
                    cam_frame = frame_bgr[:, midline:]

                cam_data = stage_data.get(cam, {})

                for bp in ["thumb", "index"]:
                    coords_list = cam_data.get(bp)
                    if coords_list is None or global_idx >= len(coords_list):
                        continue
                    coords = coords_list[global_idx]
                    if coords is None:
                        prev_crops[(cam, bp)] = None
                        continue

                    cx, cy = coords[0], coords[1]
                    # Fixed-size crop
                    x0 = int(round(cx)) - crop_half
                    y0 = int(round(cy)) - crop_half
                    x1 = x0 + crop_half * 2
                    y1 = y0 + crop_half * 2
                    if x0 < 0 or y0 < 0 or x1 > cam_frame.shape[1] or y1 > cam_frame.shape[0]:
                        prev_crops[(cam, bp)] = None
                        continue

                    crop = cam_frame[y0:y1, x0:x1].copy()
                    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)

                    key = (cam, bp)
                    prev = prev_crops.get(key)
                    if prev is not None and crop.shape == prev.shape:
                        frame_ssd += ssd(crop, prev)
                        n_valid += 1

                    prev_crops[key] = crop

            if n_valid > 0:
                ssd_signal[global_idx] = frame_ssd / n_valid

        cap.release()

    return ssd_signal


def compute_motion_metric_dynamic(subject_name: str) -> tuple[np.ndarray, list[dict], dict]:
    """
    Compute SSD using dynamic bounding boxes.

    Returns:
        ssd_signal: float array of SSD values per frame
        trials: trial metadata
        box_sizes: dict mapping frame index to box dimensions for diagnostics
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

    ssd_signal = np.zeros(total_frames, dtype=np.float64)
    box_sizes = {}  # frame_idx -> (width, height) for diagnostics

    for trial in trials:
        vpath = trial["video_path"]
        sf = trial["start_frame"]
        ef = trial["end_frame"]
        n_local = trial["frame_count"]
        trial_name = trial["trial_name"]
        print(f"  Trial {trial_name}: global {sf}–{ef} ({n_local} frames)")

        cap = cv2.VideoCapture(vpath)
        prev_crop: Optional[np.ndarray] = None

        for local_idx in range(n_local):
            global_idx = sf + local_idx
            ret, frame_bgr = cap.read()
            if not ret:
                break

            h, w = frame_bgr.shape[:2]
            midline = w // 2

            frame_ssd = 0.0
            n_valid = 0

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

                if thumb_list is None or index_list is None:
                    continue
                if global_idx >= len(thumb_list) or global_idx >= len(index_list):
                    continue

                thumb = thumb_list[global_idx]
                index = index_list[global_idx]

                # Extract dynamic box
                crop = extract_dynamic_box_gray(cam_frame, thumb, index, PADDING)
                if crop is None:
                    prev_crop = None
                    continue

                # Compute SSD against previous
                if prev_crop is not None and crop.shape == prev_crop.shape:
                    frame_ssd += ssd(crop, prev_crop)
                    n_valid += 1
                    if cam_idx == 0:  # Track box size for first camera
                        box_sizes[global_idx] = (crop.shape[1], crop.shape[0])

                prev_crop = crop

            if n_valid > 0:
                ssd_signal[global_idx] = frame_ssd / n_valid

        cap.release()
        print(f"    done.")

    return ssd_signal, trials, box_sizes


# ── Plotting ──────────────────────────────────────────────────────────────────

EVENT_COLORS = {"open": "#2196F3", "peak": "#FF9800", "close": "#E91E63"}

def plot_comparison(ssd_fixed: np.ndarray, ssd_dynamic: np.ndarray,
                   trials: list[dict], events: dict[str, list[int]],
                   out_dir: Path) -> None:
    """Compare fixed-box vs dynamic-box SSD signals."""
    total = len(ssd_fixed)
    frames = np.arange(total)

    sm_fixed = gaussian_smooth(ssd_fixed, SMOOTHING_SIGMA)
    sm_dynamic = gaussian_smooth(ssd_dynamic, SMOOTHING_SIGMA)

    nz_fixed = ssd_fixed[ssd_fixed > 0]
    nz_dynamic = ssd_dynamic[ssd_dynamic > 0]
    ymax = max(
        float(np.percentile(nz_fixed, 98)) if len(nz_fixed) else 1,
        float(np.percentile(nz_dynamic, 98)) if len(nz_dynamic) else 1
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)

    # Fixed box
    ax1.plot(frames, ssd_fixed, color="silver", alpha=0.4, linewidth=0.5)
    ax1.plot(frames, sm_fixed, color="#1565C0", linewidth=1.0, label="Fixed 144×144 px box")
    ax1.set_ylim(0, ymax * 1.1)
    ax1.set_ylabel("Mean SSD (px²)")
    ax1.set_title(f"{SUBJECT} – Fixed vs Dynamic Bounding Box Comparison")
    ax1.legend(fontsize=9)

    # Dynamic box
    ax2.plot(frames, ssd_dynamic, color="silver", alpha=0.4, linewidth=0.5)
    ax2.plot(frames, sm_dynamic, color="#4CAF50", linewidth=1.0, label="Dynamic box (thumb+index)")
    ax2.set_ylim(0, ymax * 1.1)
    ax2.set_ylabel("Mean SSD (px²)")
    ax2.set_xlabel("Global frame index")
    ax2.legend(fontsize=9)

    # Event markers on both
    for ax in [ax1, ax2]:
        for etype, color in EVENT_COLORS.items():
            rug_frames = events.get(etype, [])
            if rug_frames:
                ax.vlines(rug_frames, 0, ymax, color=color, linewidth=0.4, alpha=0.3)

    # Trial boundaries
    for t in trials[1:]:
        ax2.axvline(t["start_frame"], color="black", linewidth=0.8, linestyle="--", alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / f"{SUBJECT}_fixed_vs_dynamic_comparison.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  → saved {out_path}")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"=== Dynamic Box Motion Metric Analysis: {SUBJECT} ===\n")

    # Compute fixed-box signal (144×144, the optimal size from comparison)
    print("Computing fixed-box SSD signal (144×144 px)…")
    ssd_fixed = compute_motion_metric_fixed(SUBJECT, crop_half=72)

    # Compute dynamic-box signal
    print("\nComputing dynamic-box SSD signal…")
    ssd_dynamic, trials, box_sizes = compute_motion_metric_dynamic(SUBJECT)

    # Load events
    events = load_events(SUBJECT)

    # Smooth both
    sm_fixed = gaussian_smooth(ssd_fixed, SMOOTHING_SIGMA)
    sm_dynamic = gaussian_smooth(ssd_dynamic, SMOOTHING_SIGMA)

    print(f"\nEvents loaded: {', '.join(f'{k}:{len(v)}' for k,v in events.items())}")

    # Compare statistics
    print("\n== Comparison ==")
    print(f"  Fixed-box median SSD:   {np.median(ssd_fixed[ssd_fixed > 0]):.0f}")
    print(f"  Dynamic-box median SSD: {np.median(ssd_dynamic[ssd_dynamic > 0]):.0f}")

    if box_sizes:
        box_widths = [w for w, h in box_sizes.values()]
        box_heights = [h for w, h in box_sizes.values()]
        print(f"\n  Dynamic box sizes:")
        print(f"    Width:  min={min(box_widths)}, max={max(box_widths)}, mean={np.mean(box_widths):.0f}")
        print(f"    Height: min={min(box_heights)}, max={max(box_heights)}, mean={np.mean(box_heights):.0f}")
        print(f"    (shows box scales with finger separation)")

    # Generate comparison plot
    print("\nGenerating comparison plot…")
    plot_comparison(ssd_fixed, ssd_dynamic, trials, events, OUTPUT_DIR)

    # Save both signals for further analysis
    np.save(OUTPUT_DIR / "ssd_signal_fixed.npy", ssd_fixed)
    np.save(OUTPUT_DIR / "ssd_signal_dynamic.npy", ssd_dynamic)
    print(f"\nDone. Output in {OUTPUT_DIR.resolve()}")
