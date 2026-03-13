"""Motion metric analysis using local SSD around fingertips.

For each frame, crops a small patch around each tracked fingertip,
subtracts the previous frame's patch, and computes the sum of squared
differences as a proxy for local optical flow / motion energy.

Theory:
  - Subjects pause for a few frames after each CLOSE event before the next OPEN.
  - During a pause, the fingertips barely move → low SSD.
  - During a tap, the fingertips move rapidly → high SSD.
  - Expected pattern:
        open → (rising SSD) → peak → (falling SSD) → close → (low SSD plateau) → open → ...

Usage:
    cd /Users/newboldd/code/dlc-labeler
    python scratch/motion_metric_analysis.py
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Config ──────────────────────────────────────────────────────────────────

SUBJECT = "MSA02"
CROP_HALF = 24          # half-size of square patch in pixels (patch = 2*CROP_HALF × 2*CROP_HALF)
SMOOTHING_FRAMES = 3    # median smoothing window for SSD signal
ZOOM_FRAMES = 300       # number of frames in each zoomed panel

# Allow override via environment variable
if "CROP_HALF" in os.environ:
    try:
        CROP_HALF = int(os.environ["CROP_HALF"])
    except ValueError:
        pass

OUTPUT_DIR = Path(f"scratch/motion_metric_output_crop{CROP_HALF*2}px")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Project imports ──────────────────────────────────────────────────────────

from dlc_app.config import get_settings
from dlc_app.services.video import build_trial_map
from dlc_app.services.dlc_predictions import get_corrections_with_dlc_fallback

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_events(subject_name: str) -> dict[str, list[int]]:
    """Load events.csv for a subject → {event_type: [frame_num, ...]}."""
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


def extract_crop_gray(frame_bgr: np.ndarray, cx: float, cy: float,
                      half: int) -> Optional[np.ndarray]:
    """Extract a grayscale crop centred on (cx, cy), or None if OOB."""
    h, w = frame_bgr.shape[:2]
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    x1 = x0 + 2 * half
    y1 = y0 + 2 * half
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None
    crop = frame_bgr[y0:y1, x0:x1]
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)


def ssd(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.sum(diff * diff))


def median_smooth(arr: list[float], half: int) -> list[float]:
    """Apply median filter with symmetric edge padding."""
    n = len(arr)
    out = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out.append(float(np.median(arr[lo:hi])))
    return out


def gaussian_smooth(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Simple 1-D Gaussian smoothing via convolution."""
    radius = int(3 * sigma)
    if radius < 1:
        return arr.copy()
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(arr, kernel, mode="same")


# ── Main computation ──────────────────────────────────────────────────────────

def compute_motion_metric(subject_name: str) -> tuple[np.ndarray, list[dict]]:
    """
    Returns:
        ssd_signal: float array, length = total_frames. Index 0 is always 0.0.
        trials:     trial metadata list from build_trial_map()
    """
    settings = get_settings()
    cam_names = settings.camera_names   # e.g. ['OS', 'OD']

    print(f"Loading trial map for {subject_name}…")
    trials = build_trial_map(subject_name)
    total_frames = trials[-1]["end_frame"] + 1
    print(f"  {len(trials)} trials, {total_frames} total frames")

    print(f"Loading coordinates…")
    stage_data = get_corrections_with_dlc_fallback(subject_name)
    if stage_data is None:
        raise RuntimeError("No coordinate data found for subject")

    ssd_signal = np.zeros(total_frames, dtype=np.float64)

    for trial in trials:
        vpath = trial["video_path"]
        sf = trial["start_frame"]
        ef = trial["end_frame"]
        n_local = trial["frame_count"]
        trial_name = trial["trial_name"]
        print(f"  Trial {trial_name}: global {sf}–{ef} ({n_local} frames) from {Path(vpath).name}")

        cap = cv2.VideoCapture(vpath)
        prev_crops: dict[tuple, Optional[np.ndarray]] = {}  # (cam, bp) → last crop

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
                # Crop the half of the stereo frame for this camera
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
                    crop = extract_crop_gray(cam_frame, cx, cy, CROP_HALF)

                    key = (cam, bp)
                    prev = prev_crops.get(key)
                    if prev is not None and crop is not None:
                        frame_ssd += ssd(crop, prev)
                        n_valid += 1

                    prev_crops[key] = crop

            # Normalise by number of valid pairs so missing data doesn't deflate signal
            if n_valid > 0:
                ssd_signal[global_idx] = frame_ssd / n_valid

        cap.release()
        print(f"    done.")

    return ssd_signal, trials


# ── Plotting ──────────────────────────────────────────────────────────────────

EVENT_COLORS = {"open": "#2196F3", "peak": "#FF9800", "close": "#E91E63"}
EVENT_YPOS   = {"open": 0.92, "peak": 0.84, "close": 0.76}

def _draw_events_on_ax(ax, events, sf, ef, ymax, color_vlines=True):
    legend_patches = []
    for etype, color in EVENT_COLORS.items():
        rug_frames = [f for f in events.get(etype, []) if sf <= f <= ef]
        if not rug_frames:
            continue
        if color_vlines:
            ax.vlines(rug_frames, 0, ymax, color=color, linewidth=0.5, alpha=0.3)
        legend_patches.append(mpatches.Patch(color=color, label=etype))
    return legend_patches


def plot_trial(trial: dict, ssd_signal: np.ndarray, smoothed_g: np.ndarray,
               events: dict[str, list[int]], out_dir: Path,
               detected_opens: list[int] | None = None,
               tolerance_frames: int = 8) -> None:
    sf   = trial["start_frame"]
    ef   = trial["end_frame"]
    fps  = trial["fps"] or 60.0
    frames   = np.arange(sf, ef + 1)
    raw_vals = ssd_signal[sf: ef + 1]
    sm_vals  = smoothed_g[sf: ef + 1]
    ymax     = float(np.percentile(raw_vals[raw_vals > 0], 98)) if raw_vals.max() > 0 else 1.0

    # Choose a representative 300-frame zoom window centred on the middle of the trial
    mid   = (sf + ef) // 2
    z_sf  = max(sf, mid - ZOOM_FRAMES // 2)
    z_ef  = min(ef, z_sf + ZOOM_FRAMES - 1)

    fig = plt.figure(figsize=(18, 9))
    gs  = fig.add_gridspec(3, 2, height_ratios=[3, 1, 3],
                           hspace=0.45, wspace=0.08)

    ax_full  = fig.add_subplot(gs[0, :])   # full trial SSD
    ax_ev    = fig.add_subplot(gs[1, :], sharex=ax_full)  # event rug
    ax_zoom  = fig.add_subplot(gs[2, :])   # zoomed section

    # ── Full trial ─────────────────────────────────────────────────────────
    ax_full.plot(frames, raw_vals, color="silver", alpha=0.5, linewidth=0.6, label="SSD (raw)")
    ax_full.plot(frames, sm_vals,  color="#1565C0", linewidth=1.0,
                 label=f"SSD (Gaussian σ=3)")
    ax_full.set_ylim(0, ymax * 1.05)
    ax_full.set_ylabel("Mean SSD (px²)", fontsize=9)
    ax_full.set_title(
        f"{SUBJECT}  –  {trial['trial_name']}  (patch {2*CROP_HALF}×{2*CROP_HALF} px, {fps:.0f} fps)",
        fontsize=10)
    lp = _draw_events_on_ax(ax_full, events, sf, ef, ymax)
    ax_full.legend(handles=[
        mpatches.Patch(color="silver", label="SSD (raw)"),
        mpatches.Patch(color="#1565C0", label="SSD (smooth)"),
        *lp], loc="upper right", fontsize=7)
    # Shade zoom region
    ax_full.axvspan(z_sf, z_ef, color="gold", alpha=0.12, zorder=0)

    # ── Event rug ──────────────────────────────────────────────────────────
    ax_ev.set_ylim(0, 1)
    ax_ev.set_yticks([])
    ax_ev.set_xlabel("Global frame index", fontsize=9)
    legend_patches = []
    for etype, color in EVENT_COLORS.items():
        rug_frames = [f for f in events.get(etype, []) if sf <= f <= ef]
        if rug_frames:
            ax_ev.vlines(rug_frames, 0, 1, color=color, linewidth=0.8, alpha=0.9)
            legend_patches.append(mpatches.Patch(color=color, label=etype))
    if legend_patches:
        ax_ev.legend(handles=legend_patches, loc="upper right", fontsize=7)

    # ── Zoom ───────────────────────────────────────────────────────────────
    z_frames   = np.arange(z_sf, z_ef + 1)
    z_raw      = ssd_signal[z_sf: z_ef + 1]
    z_sm       = smoothed_g[z_sf: z_ef + 1]
    z_ymax     = float(np.percentile(z_raw[z_raw > 0], 98)) if z_raw.max() > 0 else 1.0

    ax_zoom.plot(z_frames, z_raw, color="silver", alpha=0.5, linewidth=0.8)
    ax_zoom.plot(z_frames, z_sm,  color="#1565C0", linewidth=1.4)
    ax_zoom.set_ylim(0, z_ymax * 1.1)
    ax_zoom.set_ylabel("Mean SSD (px²)", fontsize=9)
    ax_zoom.set_xlabel(f"Frame (zoom: {z_sf}–{z_ef})", fontsize=9)
    ax_zoom.set_title(f"Zoom ({ZOOM_FRAMES} frames) — yellow region above", fontsize=9)

    lp2 = _draw_events_on_ax(ax_zoom, events, z_sf, z_ef, z_ymax)

    # Overlay detected opens with TP/FP/FN classification
    if detected_opens is not None:
        opens_in_win = [f for f in events.get("open", []) if z_sf <= f <= z_ef]
        det_in_win   = [d for d in detected_opens if z_sf <= d <= z_ef]

        matched_det  = set()
        matched_open = set()
        for op in opens_in_win:
            for j, det in enumerate(det_in_win):
                if j in matched_det:
                    continue
                if abs(det - op) <= tolerance_frames:
                    matched_det.add(j)
                    matched_open.add(op)
                    break

        for j, det in enumerate(det_in_win):
            yi = det - z_sf
            y  = float(z_sm[yi]) if 0 <= yi < len(z_sm) else 0.0
            if j in matched_det:   # TP: green dot at the detection frame
                ax_zoom.plot(det, y, "o", color="#4CAF50", markersize=5, zorder=5)
            else:                  # FP: red X
                ax_zoom.plot(det, y, "x", color="#F44336", markersize=6, markeredgewidth=1.5, zorder=5)

        for op in opens_in_win:
            if op not in matched_open:   # FN: orange triangle at the labeled frame
                yi = op - z_sf
                y  = float(z_sm[yi]) if 0 <= yi < len(z_sm) else 0.0
                ax_zoom.plot(op, y, "^", color="#FF9800", markersize=6, zorder=5)

        ax_zoom.legend(handles=[
            *lp2,
            mpatches.Patch(color="#4CAF50", label="TP (detected)"),
            mpatches.Patch(color="#F44336", label="FP (false det)"),
            mpatches.Patch(color="#FF9800", label="FN (missed)"),
        ], loc="upper right", fontsize=7)

    plt.tight_layout()
    out_path = out_dir / f"{SUBJECT}_{trial['trial_name']}_motion_metric.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  → saved {out_path}")


def plot_overview(ssd_signal: np.ndarray, smoothed_g: np.ndarray,
                  trials: list[dict], events: dict[str, list[int]],
                  out_dir: Path) -> None:
    """Single-panel overview across all trials."""
    total  = len(ssd_signal)
    frames = np.arange(total)
    nz     = ssd_signal[ssd_signal > 0]
    ymax   = float(np.percentile(nz, 98)) if len(nz) else 1.0

    fig, ax = plt.subplots(figsize=(22, 4))
    ax.plot(frames, ssd_signal, color="silver", alpha=0.5, linewidth=0.5)
    ax.plot(frames, smoothed_g,   color="#1565C0", linewidth=1.0)
    ax.set_ylim(0, ymax * 1.1)

    _draw_events_on_ax(ax, events, 0, total, ymax)

    # Trial boundaries
    for t in trials[1:]:
        ax.axvline(t["start_frame"], color="black", linewidth=0.8, linestyle="--", alpha=0.4)
        ax.text(t["start_frame"] + 5, ymax * 0.95,
                t["trial_name"], fontsize=7, alpha=0.7)

    ax.set_xlabel("Global frame index")
    ax.set_ylabel("Mean SSD (px²)")
    ax.set_title(f"{SUBJECT} – Motion Metric Overview  (patch {2*CROP_HALF}×{2*CROP_HALF} px)")
    legend_patches = [mpatches.Patch(color=c, label=e) for e, c in EVENT_COLORS.items()]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8)

    plt.tight_layout()
    out_path = out_dir / f"{SUBJECT}_overview_motion_metric.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  → saved {out_path}")


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_open_prediction(ssd_signal: np.ndarray, smoothed_g: np.ndarray,
                              events: dict[str, list[int]], fps: float = 60.0,
                              tolerance_frames: int = 8) -> list[int]:
    """
    For each OPEN event, find the rising edge of the SSD signal (local minimum
    just before the open) and measure how far the smoothed SSD rises before
    the labeled open.  Also try simple threshold-crossing detection and report
    precision/recall vs. the labelled opens.
    """
    opens  = sorted(events.get("open",  []))
    peaks  = sorted(events.get("peak",  []))
    closes = sorted(events.get("close", []))

    n = len(ssd_signal)
    sm = smoothed_g

    print(f"\n== Evaluation ==")
    print(f"  Labeled opens: {len(opens)},  peaks: {len(peaks)},  closes: {len(closes)}")

    # ── Between-tap gap analysis ──
    # Use consecutive open events to define "gaps" (close ≈ end of one cycle)
    if len(opens) >= 2:
        gap_min_ssd    = []   # min SSD in inter-open interval
        tap_peak_ssd   = []   # max SSD between consecutive opens
        # Between open[i] and open[i+1] — split at the midpoint as rough close proxy
        for i in range(len(opens) - 1):
            a, b = opens[i], opens[i + 1]
            mid  = (a + b) // 2
            gap_win = sm[mid:b]
            tap_win = sm[a:mid]
            if len(gap_win) > 0:
                gap_min_ssd.append(float(np.min(gap_win)))
            if len(tap_win) > 0:
                tap_peak_ssd.append(float(np.max(tap_win)))

        gap_arr = np.array(gap_min_ssd)
        tap_arr = np.array(tap_peak_ssd)
        print(f"\n  Gap (pre-open) SSD:  median={np.median(gap_arr):.0f}  "
              f"p25={np.percentile(gap_arr,25):.0f}  p75={np.percentile(gap_arr,75):.0f}")
        print(f"  Tap (post-open) SSD: median={np.median(tap_arr):.0f}  "
              f"p25={np.percentile(tap_arr,25):.0f}  p75={np.percentile(tap_arr,75):.0f}")
        ratio = np.median(tap_arr) / (np.median(gap_arr) + 1e-9)
        print(f"  Tap/gap median ratio: {ratio:.1f}x  ← higher = more separable")

    nonzero = sm[sm > 0]
    if len(nonzero) == 0:
        print("  All SSD zero — cannot evaluate.")
        return

    # ── Local-minimum detector for OPEN events ──────────────────────────────
    # Theory (confirmed by mean tap profile): the SSD signal has a clear local
    # minimum AT the open event (subject pauses before opening fingers).
    # Strategy: find all local minima, then filter by:
    #   - minimum prominence (min must dip to some fraction of the local peak)
    #   - minimum inter-event spacing (≈ min inter-tap interval from events)

    print(f"\n  Local-minimum open-event detection (tolerance ±{tolerance_frames} frames):")

    if len(opens) >= 2:
        inter_open = np.diff(sorted(opens))
        min_spacing = max(4, int(np.percentile(inter_open, 10)))
        print(f"    Min spacing from labeled opens (p10 of inter-tap): {min_spacing} frames")
    else:
        min_spacing = 8

    # Find raw local minima with a guard window equal to min_spacing/2
    half_g = max(2, min_spacing // 3)
    local_mins = []
    for i in range(half_g, n - half_g):
        window = sm[i - half_g: i + half_g + 1]
        if sm[i] == window.min():
            local_mins.append(i)

    # Enforce minimum spacing: keep the smallest in each cluster
    merged = []
    for m in local_mins:
        if merged and m - merged[-1] < min_spacing:
            if sm[m] < sm[merged[-1]]:
                merged[-1] = m
        else:
            merged.append(m)

    print(f"    Raw local mins: {len(local_mins)} → after spacing merge: {len(merged)}")

    # Sweep different maximum-SSD thresholds (relative to global percentile) to filter
    # out minima that are not actually "quiet" (e.g. a local min that is still high)
    best_f1 = 0.0
    best_cfg: dict = {}
    for pct in [40, 50, 60, 70, 80, 100]:
        max_ssd = float(np.percentile(nonzero, pct)) if pct < 100 else float("inf")
        detected = [m for m in merged if sm[m] <= max_ssd]

        matched_det  = set()
        matched_open = set()
        for op in opens:
            for j, det in enumerate(detected):
                if j in matched_det:
                    continue
                if abs(det - op) <= tolerance_frames:
                    matched_det.add(j)
                    matched_open.add(op)
                    break

        tp = len(matched_open)
        fp = len(detected) - tp
        fn = len(opens) - tp
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        if f1 > best_f1:
            best_f1 = f1
            best_cfg = dict(pct=pct, thresh=max_ssd, det=len(detected),
                            tp=tp, fp=fp, fn=fn, prec=prec, rec=rec)
        label = f"≤p{pct}" if pct < 100 else "unfiltered"
        print(f"    {label:12s}:  det={len(detected):3d}  "
              f"TP={tp:3d}  FP={fp:3d}  FN={fn:3d}  "
              f"prec={prec:.2f}  rec={rec:.2f}  F1={f1:.3f}")

    c = best_cfg
    print(f"\n  → Best F1={best_f1:.3f}  "
          f"prec={c['prec']:.2f}  rec={c['rec']:.2f}  "
          f"(≤p{c['pct']}, det={c['det']}, TP={c['tp']}, FP={c['fp']}, FN={c['fn']})")

    # Return the best detection list
    best_max_ssd = c['thresh'] if c['pct'] < 100 else float('inf')
    return [m for m in merged if sm[m] <= best_max_ssd]


# ── Per-tap SSD profile plot ──────────────────────────────────────────────────

def plot_mean_tap_profile(ssd_signal: np.ndarray, smoothed_g: np.ndarray,
                          events: dict[str, list[int]], fps: float,
                          out_dir: Path, window_ms: int = 500) -> None:
    """Average SSD profile aligned to open events."""
    opens = sorted(events.get("open", []))
    if len(opens) < 3:
        return

    half = int(window_ms / 1000.0 * fps)
    n = len(ssd_signal)
    profiles = []
    for op in opens:
        lo, hi = op - half, op + half + 1
        if lo < 0 or hi > n:
            continue
        profiles.append(smoothed_g[lo:hi])

    if not profiles:
        return

    # Trim to common length
    min_len = min(len(p) for p in profiles)
    mat = np.vstack([p[:min_len] for p in profiles])
    mean_p  = mat.mean(axis=0)
    std_p   = mat.std(axis=0)
    t_ms    = (np.arange(min_len) - half) / fps * 1000

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.fill_between(t_ms, mean_p - std_p, mean_p + std_p, alpha=0.25, color="#1565C0")
    ax.plot(t_ms, mean_p, color="#1565C0", linewidth=1.5, label=f"mean±std  (n={len(profiles)})")
    ax.axvline(0, color=EVENT_COLORS["open"], linewidth=1.5, linestyle="--", label="OPEN event")
    ax.set_xlabel("Time relative to OPEN (ms)")
    ax.set_ylabel("Mean SSD (px²)")
    ax.set_title(f"{SUBJECT} – Mean tap profile aligned to OPEN events")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out_path = out_dir / f"{SUBJECT}_mean_tap_profile.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  → saved {out_path}")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"=== Motion Metric Analysis: {SUBJECT} ===\n")

    ssd_signal, trials = compute_motion_metric(SUBJECT)
    events = load_events(SUBJECT)

    # Gaussian-smooth for display and analysis (sigma = ~1.5 frames → ≈25 ms at 60 fps)
    smoothed_g = gaussian_smooth(ssd_signal, sigma=3.0)

    fps_global = trials[0]["fps"] if trials else 60.0
    print(f"\nEvents loaded: {', '.join(f'{k}:{len(v)}' for k,v in events.items())}")

    # Evaluate and get best detected opens
    detected_opens = evaluate_open_prediction(ssd_signal, smoothed_g, events, fps=fps_global)

    # Mean tap profile aligned to open events
    print("\nGenerating mean tap profile plot…")
    plot_mean_tap_profile(ssd_signal, smoothed_g, events, fps_global, OUTPUT_DIR)

    # Per-trial plots (with detected/TP/FP/FN overlays)
    print("\nGenerating per-trial plots…")
    for trial in trials:
        plot_trial(trial, ssd_signal, smoothed_g, events, OUTPUT_DIR,
                   detected_opens=detected_opens)

    # Overview
    print("Generating overview plot…")
    plot_overview(ssd_signal, smoothed_g, trials, events, OUTPUT_DIR)

    print(f"\nDone. Output in {OUTPUT_DIR.resolve()}")
