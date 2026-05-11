#!/usr/bin/env python3
"""Analyse a skeleton v2 fit: detect spikes, boundary pinning, Z jitter.

Usage:
    python scripts/analyse_fit.py Con01 Con01_R1
    python scripts/analyse_fit.py Con01 Con01_R1 --auto-widen   # auto-update constraints
    python scripts/analyse_fit.py Con01 Con01_R1 --fit          # re-run fit first
    python scripts/analyse_fit.py Con01 Con01_R1 --fit --auto-widen --rounds 3
"""
import argparse, json, sys, os
import numpy as np
from pathlib import Path
from collections import Counter

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MT_DATA_DIR", str(Path(__file__).resolve().parent.parent))

from movement_tracker.services.mano_data import load_mano_trial_data, load_angle_priors
from movement_tracker.config import DATA_DIR


def analyse(subject, trial_stem, verbose=True):
    """Return dict with spike counts, boundary pinning, Z stats."""
    data = load_mano_trial_data(subject, trial_stem)
    fps = data['fps']; N = data['n_frames']
    angles = data.get('angles_skel_v2', {}) or data.get('angles_mano', {})
    src = 'v2' if data.get('angles_skel_v2') else 'v1'

    # Spike detection (>6σ local deviation)
    spikes = []
    for name, trace in sorted(angles.items()):
        arr = np.array([v if v is not None else np.nan for v in trace], dtype=np.float64)
        valid = ~np.isnan(arr)
        if valid.sum() < 50: continue
        diffs = np.abs(np.diff(arr[valid]))
        sigma = max(np.median(diffs) * 1.4826, 1.0)
        for i in range(3, N-3):
            if not valid[i]: continue
            nbrs = arr[max(0,i-3):min(N,i+4)][valid[max(0,i-3):min(N,i+4)]]
            if len(nbrs) < 4: continue
            dev = arr[i] - np.median(nbrs)
            if abs(dev) > 6 * sigma:
                spikes.append({'name': name, 'frame': i, 'dev': dev, 'sigma_n': abs(dev)/sigma})

    spike_counts = Counter(s['name'] for s in spikes)

    # Boundary pinning
    priors = load_angle_priors()
    pinning = {}
    for jt in priors.get('joints', []):
        for prefix, lo_key, hi_key in [('Flex', 'flex_min', 'flex_max'), ('Abd', 'abd_min', 'abd_max')]:
            aname = f"{prefix}: {jt['name']}"
            trace = angles.get(aname, [])
            arr = np.array([v if v is not None else np.nan for v in trace], dtype=np.float64)
            v = arr[~np.isnan(arr)]
            if len(v) < 10: continue
            lo, hi = jt[lo_key], jt[hi_key]
            at_lo = ((v >= lo - 1) & (v <= lo + 1)).sum()
            at_hi = ((v >= hi - 1) & (v <= hi + 1)).sum()
            pct_lo = 100 * at_lo / len(v)
            pct_hi = 100 * at_hi / len(v)
            if pct_lo > 5 or pct_hi > 5:
                pinning[aname] = {
                    'lo': lo, 'hi': hi,
                    'pct_lo': round(pct_lo, 1), 'pct_hi': round(pct_hi, 1),
                    'at_lo': int(at_lo), 'at_hi': int(at_hi),
                    'p1': round(float(np.percentile(v, 1)), 1),
                    'p99': round(float(np.percentile(v, 99)), 1),
                }

    # Z stats
    skel_3d = data.get('skel_v2_joints_3d') or data.get('mano_joints_3d')
    z_stats = {}
    if skel_3d:
        j3d = np.full((N, 21, 3), np.nan)
        for f in range(N):
            if skel_3d[f]:
                for j in range(21):
                    if skel_3d[f][j]: j3d[f, j] = skel_3d[f][j]
        diffs = np.diff(j3d, axis=0)
        for ax, name in [(0, 'X'), (1, 'Y'), (2, 'Z')]:
            d = np.abs(diffs[:, :, ax]); d = d[~np.isnan(d)]
            z_stats[name] = {
                'median': round(float(np.median(d)), 2),
                'p95': round(float(np.percentile(d, 95)), 1),
                'p99': round(float(np.percentile(d, 99)), 1),
                'max': round(float(d.max()), 1),
            }

    result = {
        'source': src,
        'n_frames': N,
        'fps': fps,
        'total_spikes': len(spikes),
        'spike_counts': dict(spike_counts.most_common()),
        'boundary_pinning': pinning,
        'z_movement': z_stats,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"  {subject}/{trial_stem} ({src}) — {N} frames @ {fps}fps")
        print(f"{'='*60}")
        print(f"\nTotal spikes (>6σ): {len(spikes)}")
        if spike_counts:
            print(f"\n{'Angle':<22} {'Spikes':>6}")
            print("-" * 32)
            for name, count in spike_counts.most_common(15):
                print(f"  {name:<22} {count:>4}")

        if pinning:
            print(f"\nBoundary pinning (>5%):")
            print(f"  {'Angle':<22} {'Boundary':>12} {'Pinned':>8} {'Data p1/p99':>14}")
            print("  " + "-" * 58)
            for aname, p in sorted(pinning.items(), key=lambda x: -max(x[1]['pct_lo'], x[1]['pct_hi'])):
                side = f"min={p['lo']}" if p['pct_lo'] > p['pct_hi'] else f"max={p['hi']}"
                pct = max(p['pct_lo'], p['pct_hi'])
                print(f"  {aname:<22} {side:>12} {pct:>6.0f}%  [{p['p1']:>6.1f}, {p['p99']:>6.1f}]")

        if z_stats:
            print(f"\nFrame-to-frame movement:")
            for ax in ['X', 'Y', 'Z']:
                s = z_stats[ax]
                print(f"  {ax}: median={s['median']}mm p95={s['p95']}mm p99={s['p99']}mm max={s['max']}mm")

    return result


def suggest_wider_constraints(pinning, margin=5):
    """Suggest widened constraints based on boundary pinning data.

    Respects anatomical hard limits to prevent runaway widening.
    """
    # Anatomical hard limits — the auto-widener cannot go past these
    HARD_LIMITS = {
        'flex_min': -140,  # no joint flexes more than 140°
        'flex_max': 40,    # no joint hyperextends more than 40°
        'abd_min': -50,    # no joint abducts more than 50°
        'abd_max': 50,
    }

    priors = load_angle_priors()
    changes = {}
    for jt in priors.get('joints', []):
        changed = False
        for prefix, lo_key, hi_key in [('Flex', 'flex_min', 'flex_max'), ('Abd', 'abd_min', 'abd_max')]:
            aname = f"{prefix}: {jt['name']}"
            if aname not in pinning: continue
            p = pinning[aname]
            if p['pct_lo'] > 5 and p['p1'] < jt[lo_key] + 5:
                new_lo = max(round(p['p1'] - margin), HARD_LIMITS[lo_key])
                jt[lo_key] = min(jt[lo_key], new_lo)
                changed = True
            if p['pct_hi'] > 5 and p['p99'] > jt[hi_key] - 5:
                new_hi = min(round(p['p99'] + margin), HARD_LIMITS[hi_key])
                jt[hi_key] = max(jt[hi_key], new_hi)
                changed = True
        if changed:
            changes[jt['name']] = {k: jt[k] for k in ['flex_min','flex_max','abd_min','abd_max']}
    return priors, changes


def save_constraints(priors):
    """Save updated constraints to both bundled defaults and custom file."""
    bundled = Path(__file__).resolve().parent.parent / 'movement_tracker' / 'services' / 'joint_angle_priors.json'
    custom = DATA_DIR / 'custom_joint_angle_priors.json'
    text = json.dumps(priors, indent=2)
    bundled.write_text(text)
    custom.write_text(text)
    print(f"  Saved to {bundled.name} and {custom.name}")


def run_fit(subject, trial_stem):
    """Run v2 skeleton fit with current settings."""
    from movement_tracker.services.mano_fitting_v2 import run_v2_fitting
    from movement_tracker.services.mano_data import _load_mano_npz

    print(f"\nRunning v2 fit for {subject}/{trial_stem}...")
    result = run_v2_fitting(
        subject, trial_stem,
        progress_callback=lambda pct: print(f"  {pct:.0f}%", end='\r', flush=True),
    )
    _load_mano_npz.cache_clear()
    print(f"  Done: L={result['mean_error_L']:.1f}px R={result['mean_error_R']:.1f}px")
    return result


def main():
    parser = argparse.ArgumentParser(description='Analyse skeleton fit quality')
    parser.add_argument('subject', help='Subject name (e.g., Con01)')
    parser.add_argument('trial', help='Trial stem (e.g., Con01_R1)')
    parser.add_argument('--fit', action='store_true', help='Re-run v2 fit before analysis')
    parser.add_argument('--auto-widen', action='store_true', help='Auto-widen tight constraints')
    parser.add_argument('--rounds', type=int, default=1, help='Number of fit+analyse+widen rounds')
    parser.add_argument('--margin', type=float, default=10, help='Margin (°) beyond data range for widening')
    args = parser.parse_args()

    for round_n in range(1, args.rounds + 1):
        if args.rounds > 1:
            print(f"\n{'#'*60}")
            print(f"  ROUND {round_n}/{args.rounds}")
            print(f"{'#'*60}")

        if args.fit:
            run_fit(args.subject, args.trial)

        result = analyse(args.subject, args.trial)

        if args.auto_widen and result['boundary_pinning']:
            priors, changes = suggest_wider_constraints(result['boundary_pinning'], margin=args.margin)
            if changes:
                print(f"\nAuto-widening {len(changes)} constraints:")
                for name, vals in changes.items():
                    print(f"  {name}: flex=[{vals['flex_min']}, {vals['flex_max']}] abd=[{vals['abd_min']}, {vals['abd_max']}]")
                save_constraints(priors)
            else:
                print("\nNo constraints need widening.")
                if round_n < args.rounds:
                    print("Stopping early — no further improvements possible.")
                    break
        elif not args.auto_widen and result['boundary_pinning']:
            print(f"\nRun with --auto-widen to automatically fix {len(result['boundary_pinning'])} tight constraints")


if __name__ == '__main__':
    main()
