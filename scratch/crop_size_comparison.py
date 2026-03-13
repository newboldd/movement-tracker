"""Compare motion metric detection quality across different crop box sizes."""

from __future__ import annotations

import subprocess, json, os, re
from pathlib import Path

SUBJECT = "MSA02"
CROP_SIZES = [12, 24, 48, 72]  # CROP_HALF values to test

OUTPUT_BASE = Path("scratch/crop_comparison")
OUTPUT_BASE.mkdir(exist_ok=True)

results = {}

for crop_half in CROP_SIZES:
    crop_px = crop_half * 2
    output_dir = OUTPUT_BASE / f"crop_{crop_px}px"
    output_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running analysis with CROP_HALF={crop_half} ({crop_px}×{crop_px} px box)")
    print(f"{'='*60}")

    # Run the motion metric analysis with the specific crop size
    env = os.environ.copy()
    env["CROP_HALF"] = str(crop_half)

    result = subprocess.run(
        ["python", "scratch/motion_metric_analysis.py"],
        env=env,
        capture_output=True,
        text=True,
        cwd="."
    )

    # Parse output for key metrics
    output_text = result.stderr + result.stdout
    lines = output_text.split('\n')

    metrics = {
        'crop_px': crop_px,
        'crop_half': crop_half,
        'output': output_dir,
    }

    # Extract key stats
    for line in lines:
        if "Best F1=" in line:
            # Extract: → Best F1=0.822  prec=0.87  rec=0.78  (≤p80, det=143, TP=125, FP=18, FN=36)
            m = re.search(r'F1=([\d.]+)', line)
            if m:
                metrics['f1'] = float(m.group(1))
            m = re.search(r'prec=([\d.]+)', line)
            if m:
                metrics['prec'] = float(m.group(1))
            m = re.search(r'rec=([\d.]+)', line)
            if m:
                metrics['rec'] = float(m.group(1))
            m = re.search(r'det=(\d+)', line)
            if m:
                metrics['det'] = int(m.group(1))
            m = re.search(r'TP=(\d+)', line)
            if m:
                metrics['tp'] = int(m.group(1))
            m = re.search(r'FP=(\d+)', line)
            if m:
                metrics['fp'] = int(m.group(1))
            m = re.search(r'FN=(\d+)', line)
            if m:
                metrics['fn'] = int(m.group(1))
        elif "Tap/gap median ratio:" in line:
            ratio_str = line.split(':')[1].strip().split('x')[0].strip()
            try:
                metrics['tap_gap_ratio'] = float(ratio_str)
            except:
                pass

    results[f'crop_{crop_px}px'] = metrics

    # Move output files
    import glob, shutil
    for png in glob.glob("scratch/motion_metric_output/*.png"):
        shutil.copy(png, output_dir / Path(png).name)

    print(f"Saved to: {output_dir}")

# Summary table
print(f"\n\n{'='*80}")
print("SUMMARY: Crop Size Comparison")
print(f"{'='*80}\n")

print(f"{'Box Size':>10} {'F1':>8} {'Prec':>8} {'Rec':>8} {'Detected':>10} {'TP':>6} {'FP':>6} {'FN':>6} {'Ratio':>8}")
print("-" * 80)

for key in sorted(results.keys(), key=lambda k: results[k]['crop_px']):
    m = results[key]
    box_size = f"{m['crop_px']}×{m['crop_px']}"
    f1 = m.get('f1', None)
    prec = m.get('prec', None)
    rec = m.get('rec', None)
    det = m.get('det', None)
    tp = m.get('tp', None)
    fp = m.get('fp', None)
    fn = m.get('fn', None)
    ratio = m.get('tap_gap_ratio', None)

    f1_str    = f"{f1:.3f}" if f1 else "—"
    prec_str  = f"{prec:.2f}" if prec else "—"
    rec_str   = f"{rec:.2f}" if rec else "—"
    det_str   = str(det) if det else "—"
    tp_str    = str(tp) if tp else "—"
    fp_str    = str(fp) if fp else "—"
    fn_str    = str(fn) if fn else "—"
    ratio_str = f"{ratio:.2f}x" if ratio else "—"

    print(f"{box_size:>10} {f1_str:>8} {prec_str:>8} {rec_str:>8} {det_str:>10} {tp_str:>6} {fp_str:>6} {fn_str:>6} {ratio_str:>8}")

print(f"\nOutput directories: {OUTPUT_BASE}/crop_*")
print("\nKey insights:")
print("  • F1: overall detection quality (higher is better)")
print("  • Prec/Rec: precision and recall tradeoff")
print("  • Detected: number of local minima found after filtering")
print("  • TP/FP/FN: true positives, false positives, false negatives")
print("  • Ratio: tap SSD / gap SSD (higher = more separable signal)")
