# HRnet Precision Notes

Reference notes on bbox tuning, 2D heatmap precision, and how 2D errors propagate to 3D in the movement-tracker pipeline.

## Recent change applied

Per-frame bbox is now **1.5× square** (centred on the landmark span midpoint, side = `max(landmark_w, landmark_h) × 1.5`). Symmetric in both `services/hrnet_bbox.py` and `services/remote_hrnet_script.py`. Re-run any trial whose precision matters; new heatmaps will have the hand at HRnet's training-time scale and aspect ratio.

---

## Does running HRnet on tighter crops give more spatial precision?

Yes — but it's a U-shape with a real lower bound.

### Why tighter helps (the "more pixels per mm" intuition)

The heatmap is a fixed `64×64` regardless of crop size. So:

- A `200×200` crop → each heatmap cell covers `200/64 ≈ 3.1` image pixels.
- A `400×400` crop → each cell covers `6.25` image pixels.
- A `100×100` crop → each cell covers `1.6` image pixels.

Halve the crop, double the image-space resolution. The sub-pixel localizer (cluster centroid with AUC weighting) interpolates within the heatmap, so its absolute image-pixel precision scales with the crop tightness — **roughly linearly until other limits kick in**.

### Why tighter eventually hurts

HRnet was trained on a specific scale distribution. The convolutional filters at each layer have receptive fields tuned to that scale. Two failure modes when you zoom past the training range:

1. **Receptive-field mismatch.** Late-layer filters that learned "this shape = an MCP knuckle" expected a knuckle to occupy ~10–15% of the input. If your crop makes the knuckle 50% of the input, those filters never see the full feature pattern and confidence collapses. Heatmaps go diffuse, not sharper.

2. **Context starvation.** HRnet uses surrounding palm/finger pixels to disambiguate which knuckle is which (especially for the four MCPs that look similar in isolation). Crop too tight and the network loses the context it needs to assign joints to the right finger.

Training crops for hand-pose HRnets are typically `1.25–1.5×` the keypoint span. Going below `~1.2×` starts showing degradation, and below `~1.0×` (cutting into keypoints) is catastrophic.

### Practical sweet spot for this pipeline

`1.5×` square is at the loose end of the safe range. To test tighter:

- Try `1.35×` and compare per-frame-max-heatmap-confidence on a known-good trial. If confidence stays high (>0.8 mean) and bone-length-fit residual drops, the tighter crop is winning. If confidence drops or residual grows, you've passed the inflection.
- Anything tighter than `1.25×` is likely diminishing returns at best.

---

## Other ways to improve 2D precision (beyond crop tightness)

1. **Higher-resolution heatmaps.** Train/run an HRnet variant with `96×96` or `128×128` output (some MMPose configs offer this). Same `1.5×` crop becomes 1.5–2× more precise. Bigger gain than crop tuning.

2. **Higher input resolution.** Use a `384×384` HRnet model. Roughly the same as above but on the input side.

3. **Sub-pixel quadratic refinement.** Around the heatmap peak, fit a 2D quadratic to the 3×3 neighborhood and find its maximum analytically. Standard pose-estimation post-processing — gains ~0.5 px in image space. The AUC-cluster centroid already does something similar but quadratic fit is a known-strong upgrade.

4. **Per-camera Bayesian fusion.** When MAP-fitting a 3D point, weight each camera's contribution by its heatmap peak confidence. Low-confidence camera contributes less, high-confidence dominates. Done well, this can outperform Hungarian-stereo-then-triangulate by 1–3 mm in depth.

5. **Temporal regularization in 3D.** After triangulation, smooth with a constant-acceleration prior (Kalman or MAP). Each frame's 3D point becomes a weighted blend of its 2D observation and predictions from its neighbors. Effective at reducing residual jitter that survives 2D processing.

---

## How 2D errors amplify in 3D

For stereo triangulation, depth error roughly scales as:

```
σ_depth ≈ (depth² / baseline) × σ_disparity_pixels / focal_pixels
```

With a typical setup (baseline ~10 cm, depth ~50 cm, focal ~1500 px):

| Disparity error | → Depth error |
|---|---|
| 1.0 px | ~17 mm |
| 0.5 px | ~8 mm |
| 0.1 px | ~1.7 mm |

Sub-pixel improvements in 2D matter a lot. Going from a 1-px to 0.3-px disparity error (e.g. by switching from argmax to centroid + quadratic refinement) saves ~12 mm of depth noise.

The transverse axes (X, Y) scale linearly with image-pixel error, much less amplified. It's mostly Z that gets hit.

---

## TL;DR

- `1.5×` square is right today.
- For more 2D precision later, **sub-pixel quadratic refinement** and **higher-resolution heatmaps** give bigger gains than tighter crops.
- Tighter crops are a real but bounded knob; don't chase below `~1.25×` without testing.
- Z (depth) error from stereo is the most amplified — small 2D wins translate to big 3D wins on the depth axis.
