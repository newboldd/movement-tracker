# MediaPipe Frame Alignment Fix

## Problem

MediaPipe labels may become misaligned with video frames when processing on different machines that count frames differently from `cv2`. This occurs because:

1. Different machines may have different codec versions or video container handling
2. cv2.VideoCapture can report different frame counts on different systems
3. Previous versions of the MediaPipe processor didn't apply frame offset corrections

## Solutions

You have **two complementary approaches**:

### Option A: Fix Existing NPZ Files (Recommended - One-time)

Use the provided utility script to permanently fix existing misaligned MediaPipe NPZ files.

**Step 1: Dry-run to see what would be fixed**
```bash
cd ~/code/dlc-labeler
python fix_mediapipe_alignment.py --all --dry-run
```

**Step 2: Fix all subjects**
```bash
python fix_mediapipe_alignment.py --all
```

**Or fix specific subjects**
```bash
python fix_mediapipe_alignment.py Con01 Con02 Con03
```

**What happens:**
- Automatically detects frame count mismatches
- Creates backup `.npz.bak` files before modifying
- Truncates or pads arrays to match expected frame count
- Updates `total_frames` metadata

**Example output:**
```
Con01: Frame count mismatch! Stored=100, Expected=90 (diff=10)
Con01: Will truncate 10 frames from end
Con01: ✓ Fixed (truncated to 90 frames)

Con02: Frame count mismatch! Stored=85, Expected=90 (diff=-5)
Con02: Will pad with 5 NaN frames
Con02: ✓ Fixed (padded to 90 frames)

Con03: ✓ Aligned (stored=90, expected=90)
```

### Option B: Automatic Load-Time Detection (Always Active)

The code now automatically detects frame misalignment when loading MediaPipe data and applies corrections on-the-fly if needed. This works transparently without code changes.

**When this activates:**
- Loads each existing NPZ file
- Compares stored frame count with current trial map
- Applies offset correction if misalignment detected
- Logs diagnostic information

**No action needed** - this is always running when you use the app.

## How the Fix Works

### Frame Alignment Logic

When frame counts differ between machines:

**If NPZ has MORE frames than expected (e.g., 100 vs 90):**
- Assumes the NPZ was created on a machine that counted extra trailing frames
- **Solution:** Truncate to expected size (trailing frames are usually NaN anyway)

**If NPZ has FEWER frames than expected (e.g., 85 vs 90):**
- Assumes the NPZ was created on a machine that counted fewer frames
- **Solution:** Pad with NaN at the end (preserves existing data, adds empty slots)

**If frame counts match:**
- No action needed, data is aligned

### What Gets Fixed

The script corrects:
- `OS_landmarks` and `OD_landmarks` arrays (21 joints, 2D coordinates)
- `confidence_OS` and `confidence_OD` arrays (detection confidence per frame)
- `distances` array (3D triangulated distances)
- `total_frames` metadata

Backups are automatically created as `.npz.bak` files.

## Testing After Fix

After running the fix, verify that MediaPipe labels display correctly:

1. **Open the labeling app:** Navigate to a subject you fixed
2. **Check final viewer:** Look at the "Final: [Subject]" view
3. **Verify labels:** MediaPipe ghost markers should align with video frames
4. **Check alignment:** Frame numbers in the UI should match where hands appear in video

## Troubleshooting

**Script fails with "Cannot read video"**
- Video files may have moved
- Check that all video files are still in their expected locations

**Frame count still doesn't match after fix**
- Some videos may have actually been modified between processing
- The script makes a best-effort correction

**Want to revert changes**
- Restore from the `.npz.bak` backup file:
  ```bash
  cp ~/code/dlc-labeler/dlc/[Subject]/mediapipe_prelabels.npz.bak \
     ~/code/dlc-labeler/dlc/[Subject]/mediapipe_prelabels.npz
  ```

## Going Forward

The new MediaPipe processing code includes frame offset logic, so **future runs will not have this issue**. It will:

1. Detect actual decodable frame count from video on the current machine
2. Calculate any frame offset needed
3. Apply offset when saving data
4. Ensure perfect alignment across all machines

## Questions?

If MediaPipe labels still don't align after applying this fix, it may indicate:
- Different video files being used on different machines
- Corrupted video metadata
- Codec compatibility issues

Contact support with details about which subjects are affected and when the issue started.
