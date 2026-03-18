# Quick Start: Fix MediaPipe Frame Alignment

## TL;DR

Run this command to fix all existing MediaPipe files:

```bash
cd ~/code/dlc-labeler
python fix_mediapipe_alignment.py --all
```

Then your MediaPipe labels will display correctly!

## Step-by-Step

### 1. Preview what will change (optional but recommended)
```bash
cd ~/code/dlc-labeler
python fix_mediapipe_alignment.py --all --dry-run
```

This shows you what will be fixed without modifying anything.

### 2. Apply the fix
```bash
python fix_mediapipe_alignment.py --all
```

**What happens:**
- ✓ Detects frame count mismatches
- ✓ Creates backup `.npz.bak` files (safe to keep or delete later)
- ✓ Fixes files in seconds per subject
- ✓ Logs results

**Example output:**
```
Con01: Frame count mismatch! Stored=100, Expected=90
Con01: ✓ Fixed (truncated to 90 frames)

Con02: ✓ Aligned (stored=90, expected=90)

Con03: Frame count mismatch! Stored=85, Expected=90
Con03: ✓ Fixed (padded to 90 frames)

Summary: Fixed 2/3 subjects
```

### 3. Test it works
- Open the labeler app
- Go to any subject (especially ones that were fixed)
- Check the "Final: [Subject]" viewer
- Verify MediaPipe ghost markers align with video frames
- Everything should look correct now! ✓

## That's it!

**Going forward:**
- New MediaPipe processing will automatically be perfectly aligned
- No more frame issues

**If something goes wrong:**
- Restore from backup: `cp dlc/[Subject]/mediapipe_prelabels.npz.bak dlc/[Subject]/mediapipe_prelabels.npz`

## Fix specific subjects only

```bash
python fix_mediapipe_alignment.py Con01 Con02 Con03
```

## For more details

See `MEDIAPIPE_ALIGNMENT_FIX.md` for:
- How the fix works
- Detailed explanation of frame alignment
- Troubleshooting
- What happens under the hood
