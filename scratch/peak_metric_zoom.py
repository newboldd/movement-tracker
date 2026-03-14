"""Zoomed peak metric visualization on ~200 frame segments."""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Load the peak metric data
data_dir = Path("scratch/peak_metric_output")
peak_144 = np.load(data_dir / "peak_144.npy") if (data_dir / "peak_144.npy").exists() else None
peak_288 = np.load(data_dir / "peak_288.npy") if (data_dir / "peak_288.npy").exists() else None

if peak_144 is None or peak_288 is None:
    print("Peak metric data not found. Running analysis first...")
    import subprocess
    subprocess.run([sys.executable, "scratch/peak_metric_analysis.py"])
    peak_144 = np.load(data_dir / "peak_144.npy")
    peak_288 = np.load(data_dir / "peak_288.npy")

# Load events
from dlc_app.services.video import build_trial_map
from scratch.peak_metric_analysis import load_events, gaussian_smooth

SUBJECT = "MSA02"
SMOOTHING_SIGMA = 3.0
OUTPUT_DIR = Path("scratch/peak_metric_output")
EVENT_COLORS = {"open": "#2196F3", "peak": "#FF9800", "close": "#E91E63"}

events = load_events(SUBJECT)
trials = build_trial_map(SUBJECT)

sm_peak_144 = gaussian_smooth(peak_144, SMOOTHING_SIGMA)
sm_peak_288 = gaussian_smooth(peak_288, SMOOTHING_SIGMA)

# Create zoomed plots for each trial
for trial in trials:
    sf = trial["start_frame"]
    ef = trial["end_frame"]
    trial_name = trial["trial_name"]

    # Choose a representative 200-frame segment from the middle
    mid = (sf + ef) // 2
    z_sf = max(sf, mid - 100)
    z_ef = min(ef, z_sf + 199)

    z_frames = np.arange(z_sf, z_ef + 1)
    z_peak_144 = peak_144[z_sf:z_ef + 1]
    z_peak_288 = peak_288[z_sf:z_ef + 1]
    z_sm_peak_144 = sm_peak_144[z_sf:z_ef + 1]
    z_sm_peak_288 = sm_peak_288[z_sf:z_ef + 1]

    # Y-axis limits
    nz_peak_144 = np.abs(z_peak_144[z_peak_144 != 0])
    nz_peak_288 = np.abs(z_peak_288[z_peak_288 != 0])
    ymax_peak = max(
        float(np.percentile(nz_peak_144, 98)) if len(nz_peak_144) else 1.0,
        float(np.percentile(nz_peak_288, 98)) if len(nz_peak_288) else 1.0
    )

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    # 144px
    ax1.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax1.plot(z_frames, z_peak_144, color="silver", alpha=0.6, linewidth=1.0, label="Peak metric (raw)")
    ax1.plot(z_frames, z_sm_peak_144, color="#1565C0", linewidth=2.0, label="Peak metric (smooth)")
    ax1.fill_between(z_frames, 0, z_peak_144, where=(z_peak_144 > 0), alpha=0.2, color="#1565C0")
    ax1.set_ylim(-ymax_peak * 1.15, ymax_peak * 1.15)
    ax1.set_ylabel("Peak Metric", fontsize=11)
    ax1.set_title(f"144×144 px box (frames {z_sf}–{z_ef})", fontsize=12)
    ax1.legend(fontsize=10, loc="upper right")
    ax1.grid(True, alpha=0.2)
    for etype, color in EVENT_COLORS.items():
        rug_frames = [f for f in events.get(etype, []) if z_sf <= f <= z_ef]
        if rug_frames:
            ax1.vlines(rug_frames, -ymax_peak, ymax_peak, color=color, linewidth=2, alpha=0.6, label=etype)

    # 288px
    ax2.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax2.plot(z_frames, z_peak_288, color="silver", alpha=0.6, linewidth=1.0, label="Peak metric (raw)")
    ax2.plot(z_frames, z_sm_peak_288, color="#4CAF50", linewidth=2.0, label="Peak metric (smooth)")
    ax2.fill_between(z_frames, 0, z_peak_288, where=(z_peak_288 > 0), alpha=0.2, color="#4CAF50")
    ax2.set_ylim(-ymax_peak * 1.15, ymax_peak * 1.15)
    ax2.set_ylabel("Peak Metric", fontsize=11)
    ax2.set_xlabel("Global frame index", fontsize=11)
    ax2.set_title(f"288×288 px box (frames {z_sf}–{z_ef})", fontsize=12)
    ax2.legend(fontsize=10, loc="upper right")
    ax2.grid(True, alpha=0.2)
    for etype, color in EVENT_COLORS.items():
        rug_frames = [f for f in events.get(etype, []) if z_sf <= f <= z_ef]
        if rug_frames:
            ax2.vlines(rug_frames, -ymax_peak, ymax_peak, color=color, linewidth=2, alpha=0.6, label=etype)

    fig.suptitle(f"{SUBJECT}  –  {trial_name}  –  Peak Metric Zoom (~200 frames)",
                 fontsize=13, fontweight="bold")

    plt.tight_layout()
    out_path = OUTPUT_DIR / f"{SUBJECT}_{trial_name}_peak_zoom.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  → saved {out_path.name}")

print(f"\nDone. Output in {OUTPUT_DIR.resolve()}")
