"""Distance-trace cleanup.

Extracted from the full Movement Tracker's ``services/metrics.py`` — the
labeling fork needs only the spike filter that keeps the thumb-index
distance trace under the video readable.  No event detection, no
video-derived motion metrics.

Two sources of noise feed the raw trace.  The combined-MediaPipe merge
occasionally picks an (OS, OD) source pair that triangulates to an
aperture far from its temporal neighbours — a 1- to 2-frame spike.  And
even on clean stretches the trace carries sub-millimetre noise.  The
MAD z-score + velocity gate below removes the former without flattening
genuine fast motion; the Gaussian smooth is available but defaults off
because any sigma shifts the integer location of argmax / argmin.
"""

from __future__ import annotations

import numpy as np



def _interp_nans_1d(arr: np.ndarray) -> np.ndarray:
    """Replace NaN values with linear interpolation between the nearest
    finite samples.  Edge NaNs extend the nearest finite value."""
    out = np.asarray(arr, dtype=float).copy()
    mask = np.isnan(out)
    if mask.all() or not mask.any():
        return out
    idx = np.arange(out.size)
    out[mask] = np.interp(idx[mask], idx[~mask], out[~mask])
    return out


def _gaussian1d(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Tiny Gaussian smoothing.  ``sigma`` in samples; radius = ceil(3σ)."""
    if sigma <= 0:
        return np.asarray(arr, dtype=float)
    radius = max(1, int(np.ceil(3 * sigma)))
    t = np.arange(-radius, radius + 1)
    k = np.exp(-(t ** 2) / (2 * sigma * sigma))
    k = k / k.sum()
    return np.convolve(np.asarray(arr, dtype=float), k, mode="same")




def clean_distance_trace(
    dist,
    max_valid_dist: float = 200.0,
    spike_window: int = 3,
    spike_z: float = 3.5,
    spike_velocity: float = 25.0,
    smooth_sigma: float = 0.0,
) -> np.ndarray:
    """Return a cleaned, smoothed copy of a 1-D distance trace.

    Pipeline:
      1. Mark <0 / >max_valid_dist / NaN samples as NaN.
      2. MAD-based spike filter: for each sample t compare it to a
         ``±spike_window`` neighbourhood; if |z| > ``spike_z`` AND
         either single-frame jump exceeds ``spike_velocity``, mark NaN.
      3. Linear interpolation across NaN runs.
      4. Light Gaussian smoothing with ``smooth_sigma``.

    The combination removes the 1- to 2-frame outliers that combined-MP
    occasionally produces without flattening genuine fast motion.
    """
    arr = np.asarray(dist, dtype=float).copy()
    n = arr.size
    if n == 0:
        return arr
    # 1. Range filter.
    arr[~np.isfinite(arr)] = np.nan
    arr[(arr < 0) | (arr > max_valid_dist)] = np.nan
    # 2. MAD spike filter.  Compare each finite sample to its small
    # neighbourhood; a |z| > spike_z _and_ a >spike_velocity jump on
    # either side flags a real spike (vs noise on a stable stretch).
    finite0 = np.isfinite(arr)
    base = _interp_nans_1d(arr)   # interp once so window stats use neighbours
    for t in range(n):
        if not finite0[t]:
            continue
        lo, hi = max(0, t - spike_window), min(n, t + spike_window + 1)
        win = base[lo:hi]
        if win.size < 3:
            continue
        med = float(np.median(win))
        mad = 1.4826 * float(np.median(np.abs(win - med)))
        if mad < 1.0:
            mad = 1.0   # floor — flat stretches shouldn't be hyper-sensitive
        z = abs(base[t] - med) / mad
        if z <= spike_z:
            continue
        dl = abs(base[t] - base[t - 1]) if t > 0 else 0.0
        dr = abs(base[t + 1] - base[t]) if t < n - 1 else 0.0
        if max(dl, dr) > spike_velocity:
            arr[t] = np.nan
    # 3. Re-interpolate NaNs.
    out = _interp_nans_1d(arr)
    # 4. Gaussian smooth.
    return _gaussian1d(out, smooth_sigma)
