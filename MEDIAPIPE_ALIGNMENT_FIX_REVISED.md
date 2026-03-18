# MediaPipe Frame Alignment - Revised Solution

## The Situation

Your existing MediaPipe NPZ files have frame count mismatches between machines. The challenge is that these files contain **concatenated data from multiple trials**, and safely reordering that data without knowing the exact offset applied during processing is risky.

## The Real Issue

When MediaPipe was processed on different machines:
- Machine A: Counted 2804 total frames across all trials
- Machine B: Counts 2586 total frames across the same trials

The NPZ was saved with one count, but your current trial map reports a different count. This suggests:
1. Different codec/video container handling between machines
2. Different video versions or frame trimming
3. Different cv2 frame counting behavior

## The Best Solution: Reprocess MediaPipe

The **safest and most reliable fix** is to reprocess MediaPipe on your current machine. The code has been fixed to properly handle frame offsets, so new processing will be perfectly aligned.

**Why this is better than trying to fix existing files:**
- ✅ Correct: Uses actual machine's video properties
- ✅ Safe: No risk of data corruption from incorrect shifting
- ✅ Complete: Handles all frame boundary edge cases correctly
- ✅ Future-proof: New data will be perfectly aligned

## How to Reprocess MediaPipe

1. **Delete the misaligned files** (you'll be regenerating them):
   ```bash
   rm ~/code/dlc-labeler/dlc/Con02/mediapipe_prelabels.npz
   # Repeat for other subjects
   ```

2. **Run MediaPipe preprocessing** via the API or command-line:
   ```bash
   # Via the labeling app, trigger preprocessing
   # Or programmatically:
   python -c "
   from dlc_app.services.mediapipe_prelabel import run_mediapipe
   run_mediapipe('Con02')
   "
   ```

3. **Verify alignment**:
   - Open the labeler app
   - Go to "Final: [Subject]" viewer
   - Check that MediaPipe ghost markers align with video frames

## What About Existing Files?

The system will **detect the misalignment** and log a warning:
```
WARNING: Frame count mismatch detected! NPZ stored 2804 frames, but trial map expects 2586.
```

However, without knowing exactly how the offset was (or wasn't) applied during processing, **we cannot safely auto-correct** the data. The safest approach is reprocessing.

## Why Not Just Truncate/Pad?

Your NPZ contains data from multiple trials concatenated together:
- Trial 1: Frames 0-500 (actual)
- Trial 2: Frames 501-1200 (actual)
- Trial 3: Frames 1201-1500 (actual)

If one trial's videos were 50 frames shorter on your machine, that offset would affect all subsequent trials. Simply truncating from the end would corrupt data across multiple trials.

## Timeline

1. **Short-term (immediate):** System will warn you about misalignment
2. **Medium-term (when convenient):** Reprocess MediaPipe files
3. **Long-term (future processing):** Automatic - new MP processing is correctly aligned

## Questions?

If reprocessing isn't feasible right now, you can:
1. Note which subjects have the mismatch (system logs will show them)
2. Reprocess them when you have time
3. In the meantime, use DLC/corrections labels (which work correctly)

The main impact is just that MediaPipe ghost markers won't align perfectly with video frames for existing files.
