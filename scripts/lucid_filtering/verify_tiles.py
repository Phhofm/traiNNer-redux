#!/usr/bin/env python3
"""
LUCID Tile Verifier
===================
Author: Philip Hofmann

Description:
This script checks every image file in a directory to ensure it is valid
and fully readable. It is designed to catch "truncated" PNGs caused by
disk-full errors (where the file exists but the data is incomplete).

It does NOT just check file size. It performs a full bitstream decode.

Usage:
  python verify_tiles.py --input /path/to/tiles --corrupted /path/to/corrupted
"""

import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
from tqdm import tqdm


def verify_image(img_path: Path) -> tuple[bool, Path]:
    """Attempts to decode the image. Returns (True, path) if valid, (False, path) if corrupted."""
    try:
        # cv2.imread returns None if decoding fails
        img = cv2.imread(str(img_path))
        if img is None:
            return False, img_path
        return True, img_path
    except Exception:
        return False, img_path


def main() -> None:
    # Lower process priority to keep system responsive
    try:
        os.nice(15)
        print("System Responsiveness Mode: Priority lowered to 15.")
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="Verify image integrity in a directory."
    )
    parser.add_argument(
        "--input", required=True, help="Directory containing tiles to verify"
    )
    parser.add_argument(
        "--corrupted", required=True, help="Directory to move corrupted tiles to"
    )
    parser.add_argument("--workers", type=int, default=None, help="Number of processes")
    args = parser.parse_args()

    in_dir = Path(args.input)
    corr_dir = Path(args.corrupted)
    corr_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {in_dir} for images...")
    image_paths = sorted(in_dir.glob("*.png"))
    print(f"Found {len(image_paths)} images.")

    valid_count = 0
    corrupt_count = 0

    try:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(
                tqdm(
                    executor.map(verify_image, image_paths),
                    total=len(image_paths),
                    desc="Verifying",
                )
            )
    except KeyboardInterrupt:
        print("\n\n!! Interrupted by user. Cleaning up executor...")
        return

    for is_valid, path in results:
        if is_valid:
            valid_count += 1
        else:
            corrupt_count += 1
            # Move to corrupted folder
            dst = corr_dir / path.name
            try:
                shutil.move(str(path), str(dst))
            except Exception as e:
                print(f"Error moving corrupted file {path}: {e}")

    print("\n=== VERIFICATION RESULTS ===")
    print(f"Total Scanned: {len(image_paths)}")
    print(f"✅ Valid:      {valid_count}")
    print(f"❌ Corrupted:  {corrupt_count}")

    if corrupt_count > 0:
        print(f"Corrupted files have been moved to: {corr_dir}")
    else:
        print("No corruption detected! Your dataset is healthy.")


if __name__ == "__main__":
    main()
