#!/usr/bin/env python3
"""
Download the sample video (Con01_R1.mp4) from Zenodo.

Usage:
    python scripts/download_sample.py

The file is placed in sample_data/ at the project root.
Source: https://doi.org/10.5281/zenodo.19099855  (~24 MB, CC-BY-4.0)
"""

import urllib.request
import pathlib
import sys

ZENODO_URL = "https://zenodo.org/records/19099855/files/Con01_R1.mp4"
DEST_DIR   = pathlib.Path(__file__).parent.parent / "sample_data"
DEST_FILE  = DEST_DIR / "Con01_R1.mp4"


def _progress(count, block_size, total_size):
    if total_size <= 0:
        return
    pct = min(count * block_size / total_size * 100, 100)
    bar = "#" * int(pct / 2)
    sys.stdout.write(f"\r  [{bar:<50}] {pct:5.1f}%")
    sys.stdout.flush()


def main():
    if DEST_FILE.exists():
        print(f"Sample data already present at {DEST_FILE}")
        return

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading sample video from Zenodo (~24 MB)...")
    print(f"  Source : {ZENODO_URL}")
    print(f"  Dest   : {DEST_FILE}")

    try:
        urllib.request.urlretrieve(ZENODO_URL, DEST_FILE, reporthook=_progress)
        print()  # newline after progress bar
        size_mb = DEST_FILE.stat().st_size / 1_000_000
        print(f"Done. ({size_mb:.1f} MB saved to {DEST_FILE})")
    except Exception as exc:
        # Clean up partial download
        if DEST_FILE.exists():
            DEST_FILE.unlink()
        print(f"\nDownload failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
