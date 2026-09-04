# Preregistered predictions: MSA01 120 Hz video

*Written 2026-09-03, before any analysis of the 120 Hz recording.*

## Model under test

Abnormal extra movements during finger tapping are a **single elementary
event type**: a brief monophasic acceleration pulse (one direction,
~2–4 frames at 60 fps ≈ 30–70 ms of added force) superimposed on the
smooth intended movement. Compound appearances (kink / interrupt-and-
translate / out-and-back loop) are runs of 1–3 elementary pulses in
quick succession, distinguished only by the directions of successive
pulses. Detection: magnitude of the acceleration residual (measured
acceleration minus low-pass "template" acceleration, ≤5 Hz), thresholded
per participant at 2× the median within-movement residual, minimum
2-frame separation.

## What the 60 fps data established (MSA01 L1+L2, DLC corrections labels)

| Quantity | MSA01 (49 movements) | Con01 control (22 movements) |
|---|---|---|
| Pulse rate | 2.8–3.0 / movement (L1 3.3, L2 2.3) | 2.0 / movement |
| Inter-pulse interval | median 4 frames = 67 ms (IQR 3–6) | median 3, IQR 3–12 (boundary doublets) |
| Phase position (median) | 0.59 — spread through the movement; 19 % in middle third | 0.84 — boundary transients only; 0 % in middle third |
| Direction sign-flips between close pulses | 65–73 % (alternating tendency) | 84–94 % (accel/decel pairs by construction) |
| Cross-camera coincidence (±1 frame) | 93 % (chance ≈ 51 %) | 100 % |

Key artifact identified: the lateral-residual power spectrum shows an
"8.4 Hz peak" in BOTH subjects with the same processing — that peak is
the band-pass shape of the filter chain, not physiology. At 60 fps the
only honest rhythmicity measure is the inter-pulse interval; the true
spectral question requires 120 Hz.

The control's detections are real but are the *healthy* transport
accelerations at movement boundaries; the discriminating feature is not
pulse count but **where** pulses fall (mid-movement) and their
independence from phase edges.

## Predictions for the 120 Hz video (same participant, same task)

Process with the identical pipeline, fps-aware: Savitzky–Golay window
≈ 80 ms (9 frames at 120), template low-pass 5 Hz, threshold 2× the
recording's own median within-movement residual, min separation 4
frames (= 33 ms).

1. **Rate stability.** Pulse rate per movement within ±30 % of the
   60 fps estimate: **2–4 pulses/movement** above the 2× threshold.
   The events are physical, so halving the sampling interval must not
   change how many there are.
2. **IPI scales in time, not frames.** Median inter-pulse interval
   **60–75 ms → mode at 7–9 frames** at 120 fps (was 4 at 60), with
   *smaller relative spread* than the 60 fps IPI histogram
   (quantization noise halves). If instead the IPI mode lands at
   ~4 frames again (= 33 ms), the 60 fps "4-frame" spacing was a
   sampling artifact and the model is wrong.
3. **A genuine spectral peak appears.** With Nyquist at 60 Hz, the
   signed lateral acceleration residual should show a spectral peak at
   **13–18 Hz** that (a) exceeds the 2–8 Hz floor by >2×, and (b) is
   **absent when the same pipeline runs on the participant's cleanest
   movements** (lowest-tortuosity third) — the within-recording control
   that separates physiology from filter shaping.
4. **Pulse waveform becomes resolvable.** Individual pulses span
   **4–8 samples** (30–65 ms) with a measurable single-lobed shape;
   pairs that merged at 60 fps (e.g. the dense-burst interior) separate
   into distinct opposed pulses. Concretely: the fraction of detected
   events wider than 100 ms should drop by at least half relative to
   60 fps.
5. **Mid-movement occurrence persists.** ≥15 % of pulses in the middle
   third of the movement (controls: ~0 %). This is the phase-position
   signature, and it should be frame-rate independent.
6. **Direction alternation persists.** Sign-flip rate of the lateral
   component between consecutive pulses ≤ 100 ms apart: **60–80 %**.
7. **Amplitude is frame-rate invariant in integral terms.** Per-pulse
   Δv (time-integral of the residual over the pulse window, in px/s
   after scaling) matches the 60 fps distribution within ~30 %; the
   per-frame peak residual will NOT match (it scales with sampling) —
   integrals are the physical quantity.
8. **Cross-camera coincidence stays high.** ≥85 % of pulses in one
   camera have a partner within ±2 frames (~17 ms) in the other.

Falsification summary: the model dies if rate changes drastically with
frame rate (1), IPI scales in frames rather than ms (2), no
within-recording spectral contrast emerges (3), or coincidence
collapses (8). Predictions 4–7 refine parameters rather than test the
model's core.

## Prerequisites on the 120 Hz video

- Trim/onboard, DLC-label thumb+index (corrections-quality), mark
  open/peak/close events for all movements (pauses not needed).
- The analysis scripts must read fps from the trial metadata — all
  filter corners and windows above are specified in Hz/ms, not frames.

## Clinical hypothesis (added same day, after preregistration above)

Working hypothesis (Dillan): these pulses are an **action equivalent of
polyminimyoclonus** — described in ~30 % of MSA (documented clinically
in MSA01), less frequently in PD, not described in PSP. An action
version has not been systematically sought. If reliably detectable and
prevalent in MSA videos, this could aid differential diagnosis of
parkinsonism, and generates testable electrophysiology predictions
(surface EMG bursts time-locked to detected pulses; EEG jerk-locked
back-averaged potentials as in cortical myoclonus).

## Appendix: first-pass 60 fps cohort screen (untuned)

Same detector (2× per-subject median accel residual), all subjects with
DLC corrections + events; averaged across cameras.  "mid3rd/mov" =
pulses landing in the middle third of the movement per movement — the
statistic that separates intrusions from healthy boundary transients.

| group | n | median mid3rd/mov | range |
|---|---|---|---|
| MSA | 11 | **0.56** | 0.12 – 1.04 |
| PD | 10 | 0.42 | 0.26 – 0.72 |
| PSP | 2 | (0.11 / 2.25) | PSP02 is an extreme outlier (9.0 pulses/mov) — check tracking quality before interpreting |
| Control | 4 | 0.23 | 0.00 – 0.29 |

Reading: gradient in the hypothesized direction (Control < PD < MSA)
with real overlap; the control floor of ~0.2 is the current
false-positive rate of the *unverified* proposer.  The verification
layer (segmented-fit Δv/Δx integrals) and per-subject noise-floor
calibration are expected to lower that floor; the 120 Hz test refines
the pulse parameters first.

Caveat on the "Control" group: it is a convenience sample containing
both healthy controls and subjects with mixed neurologic disorders,
not a confirmed-healthy cohort.  Some of its mid-movement detections
are therefore genuine pathology rather than false positives, and the
~0.2 floor should be read with that in mind.  A confirmed-healthy
subset is needed before the control floor can be taken as a true
false-positive rate.

## Physiological refinements + the 60fps separability wall (added after cohort work)

Refinements from Dillan (movements have NO volitional brake):
- The close is a **mechanical collision** — the finger accelerates until it
  crashes into the thumb. Terminal deceleration at a close is the crash,
  not a movement event. Closes may be systematically marked LATE; the true
  close is the collision (aperture minimum / sudden terminal speed drop),
  and frames after it are post-contact.
- The only genuinely *expected* acceleration events are push-off (open)
  and reverse (aperture apex) — NOT three. Model the intended movement with
  free pulses at those two only; truncate the analysis window at the
  collision so the crash decel never registers.
- Frame 62 (≈3f after open) and ~150 are probably real jerks, not launch/
  brake. The user's pause marks are exploratory, NOT ground truth — a
  detector should EXPLAIN the trajectory shape, not reproduce those frames.

The 60fps separability wall (established quantitatively):
- The prominent mid-opening "dwell-and-reverse" corner (e.g. MSA01_L1 f68:
  finger arrests mid-opening at ~0.4px/f, tangent flips ~120°, then
  redirects) IS a real, cross-camera-confirmed shape feature — with a
  sufficiently stiff smooth model it explains 200-400 px² (more than the
  translation jerks 88/137). The shipped detector missed it only because
  edge knots let the spline round the corner.
- BUT: a dwell/corner jerk and a launch/turnaround are both sharp
  accelerations; stiffening the model to expose corners also exposes every
  ballistic boundary acceleration. Absolute explanatory power (px²) scales
  with movement VIGOR, and controls move far more (Con01: 319px path,
  40px/f peak vs MSA01: 133px, 13px/f). Result: at every threshold the
  stiff detector finds MORE jerks in the healthy control than the patient.
  The dwell/corner class is not separable from vigorous healthy motion at
  60fps.
- Two 60fps-detectable regimes remain: (1) TRANSLATION jerks (leave a
  position offset the flexible-spline position-OMP catches with 0 FP —
  this is the shipped detector) and (2) dwell/corner jerks — below the
  60fps floor.
- 120fps prediction sharpened: a jerk (30-70ms, high-freq) vs a ballistic
  acceleration (>100ms, low-freq) separate in temporal frequency, which
  aliases together at 60fps. A high-pass / matched-filter on the 120fps
  acceleration should recover the dwell/corner jerks independent of vigor —
  the test of whether regime (2) becomes detectable.
