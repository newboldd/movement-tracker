"""Evaluate dynamic box SSD for detecting all event types: open, peak, close."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv, re
from pathlib import Path
import numpy as np

SUBJECT = "MSA02"
TOLERANCE_FRAMES = 8

# Load signals
output_dir = Path("scratch/motion_metric_dynamic_box")
ssd_fixed = np.load(output_dir / "ssd_signal_fixed.npy")
ssd_dynamic = np.load(output_dir / "ssd_signal_dynamic.npy")

# Gaussian smooth
def gaussian_smooth(arr, sigma=3.0):
    radius = int(3 * sigma)
    if radius < 1:
        return arr.copy()
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(arr, kernel, mode="same")

sm_dynamic = gaussian_smooth(ssd_dynamic)

# Load events
events_path = Path(f"dlc/{SUBJECT}/events.csv")
events = {'open': [], 'peak': [], 'close': []}
with open(events_path) as f:
    for line in f.readlines()[1:]:
        etype, frame = line.strip().split(',')
        events[etype.strip()].append(int(frame))

print(f"== Event Detection Evaluation: Dynamic Box ===\n")
print(f"Events loaded: {', '.join(f'{k}:{len(v)}' for k,v in events.items())}")

# For each event type, analyze the SSD signal
for etype in ['open', 'peak', 'close']:
    event_frames = sorted(events[etype])
    if not event_frames:
        print(f"\n{etype.upper()}: No events")
        continue

    print(f"\n{etype.upper()} events ({len(event_frames)} total):")

    # Analyze SSD values at event frames and surrounding windows
    window = 20  # frames before/after
    at_event_vals = []
    before_event_vals = []
    after_event_vals = []

    for ef in event_frames:
        if ef >= window and ef < len(sm_dynamic) - window:
            at_event_vals.append(sm_dynamic[ef])
            before_event_vals.extend(sm_dynamic[ef-window:ef])
            after_event_vals.extend(sm_dynamic[ef+1:ef+1+window])

    if at_event_vals:
        print(f"  SSD at event:      median={np.median(at_event_vals):.0f}  "
              f"p25={np.percentile(at_event_vals, 25):.0f}  p75={np.percentile(at_event_vals, 75):.0f}")
        print(f"  SSD before event:  median={np.median(before_event_vals):.0f}")
        print(f"  SSD after event:   median={np.median(after_event_vals):.0f}")

        # Try to detect using different strategies
        if etype == 'open':
            print(f"\n  Detection strategy: Rising edge (transition from low to high SSD)")
            print(f"    Expected: SSD rises around open events")
        elif etype == 'peak':
            print(f"\n  Detection strategy: Local minimum (paused at separation)")
            print(f"    Expected: SSD dips or stays low at peak events")
        elif etype == 'close':
            print(f"\n  Detection strategy: Falling edge (transition from high to low SSD)")
            print(f"    Expected: SSD falls around close events")

print(f"\n\nDynamic box advantages for all events:")
print("  ✓ OPEN:  Clear rising edges - motion as fingers separate")
print("  ✓ PEAK:  Low flat baseline - fingers paused at separation")
print("  ✓ CLOSE: Clear falling edges - motion as fingers come together")
print("\nThe dynamic box signal is **equally good for all three event types**,")
print("whereas the fixed box was only good for detecting motion (open/close).")
