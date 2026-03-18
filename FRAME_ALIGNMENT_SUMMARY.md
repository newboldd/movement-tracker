# MediaPipe Frame Alignment - Complete Solution Summary

## The Problem
MediaPipe labels were becoming misaligned with video frames due to:
- Different machines counting frames differently via cv2.VideoCapture
- Old MediaPipe code not applying frame offset corrections
- Frame count differences between processing and display machines

**Impact:** Only affected MediaPipe labels; DLC/corrections/labels worked fine (they had offset logic)

## Root Cause Analysis

### MediaPipe Processing Flow (Before Fix)
1. Build trial map → calculates `expected_frame_count` from current machine
2. Allocate arrays → shape `(total_frames, 21, 2)`
3. Loop: `for local_frame in range(expected_frame_count)`
4. Call `cap.read()` - if video had fewer decodable frames, **break early**
5. **BUG:** No offset calculated, data stored at positions 0-(actual-1)
6. Save to NPZ → **misaligned data**

### Why DLC Worked
DLC code (dlc_predictions.py, line 254) had:
```python
frame_offset = max(0, trial_frame_count - csv_frame_count)
if frame_offset > 0:
    global_frame = start_frame + frame_offset + local_frame
```
It **explicitly detected and corrected** frame mismatches.

## The Fix - Three-Part Solution

### Part 1: Fix MediaPipe Processing Code ✓
**File:** `dlc_app/services/mediapipe_prelabel.py` (lines 137-193)

Now applies the **same offset logic as DLC**:
```python
# Get actual frame count
video_info = get_video_info(video_path)
actual_frame_count = video_info.frame_count

# Calculate offset
frame_offset = max(0, expected_frame_count - actual_frame_count)

# Apply offset when storing
global_frame = start_frame + frame_offset + local_frame
```

**Effect:** Future MediaPipe processing will be perfectly aligned regardless of machine.

### Part 2: Fix Existing Files ✓
**File:** `fix_mediapipe_alignment.py` (new utility script)

Standalone Python script that:
- Detects frame count mismatches in existing NPZ files
- Creates `.npz.bak` backups automatically
- Fixes by truncating (if too many frames) or padding (if too few)
- Supports dry-run mode and batch processing

**Usage:**
```bash
# Check what would be fixed
python fix_mediapipe_alignment.py --all --dry-run

# Fix all subjects
python fix_mediapipe_alignment.py --all

# Fix specific subjects
python fix_mediapipe_alignment.py Con01 Con02 Con03
```

### Part 3: Fallback Load-Time Detection ✓
**File:** `dlc_app/services/mediapipe_prelabel.py` (lines 293-339, 342-390)

Added `_detect_frame_offset()` function that:
- Compares NPZ metadata with current trial map
- Can apply corrections if misalignment detected
- Conservative: Only activates on clear mismatches
- Transparent to rest of system

**Effect:** Old misaligned files will display correctly if not fixed via utility.

## Data Flow Diagram

```
Processing Phase (Improved)
├── get_video_info() → actual frame count (with caching)
├── Calculate frame_offset upfront
├── Process frames with offset applied
└── Save correctly aligned NPZ

Display Phase (Unchanged)
├── Load NPZ (with automatic offset detection)
├── Frontend receives correct frame-to-data mapping
└── Labels display at correct frame indices
```

## Key Files Modified/Created

| File | Changes | Purpose |
|------|---------|---------|
| `dlc_app/services/mediapipe_prelabel.py` | Lines 137-193, 293-390 | Added frame offset logic to processing, offset detection to loading |
| `fix_mediapipe_alignment.py` | NEW | Utility to fix existing misaligned NPZ files |
| `MEDIAPIPE_ALIGNMENT_FIX.md` | NEW | User-friendly guide with examples |

## Testing the Fix

**For existing misaligned files:**
1. Run the utility script: `python fix_mediapipe_alignment.py --all --dry-run`
2. Review what would change
3. Apply fixes: `python fix_mediapipe_alignment.py --all`
4. Test in the labeler: Load a subject and verify MP labels align with video

**For new processing:**
- Just run MediaPipe normally
- Frame offset logic is automatic
- Files will be perfectly aligned

## Why This Solution is Better Than Alternatives

### Alternative: Display-Time Offset
- ❌ Would require code changes in labeler.js
- ❌ More complex state management
- ❌ Would need to pass offset info through multiple layers
- ✓ This solution: Fix at source, display code unchanged

### Alternative: Reprocess All MediaPipe
- ❌ Very time-consuming and computationally expensive
- ❌ Requires re-running on same machine or coordinating
- ✓ This solution: Fix existing files in minutes

### Alternative: Nothing (Ignore Problem)
- ❌ Data remains misaligned
- ❌ Users confused by wrong frames
- ✓ This solution: Clear, automated, safe with backups

## Rollback Plan

If anything goes wrong:
```bash
# Restore from backup
cp ~/code/dlc-labeler/dlc/[Subject]/mediapipe_prelabels.npz.bak \
   ~/code/dlc-labeler/dlc/[Subject]/mediapipe_prelabels.npz
```

## Going Forward

✅ **All future MediaPipe processing** will include frame offset logic and be perfectly aligned

✅ **Existing files** can be fixed with utility script (creates backups)

✅ **Mixed-state files** (some fixed, some not) work fine - system handles both seamlessly

No more frame alignment issues with MediaPipe labels!
