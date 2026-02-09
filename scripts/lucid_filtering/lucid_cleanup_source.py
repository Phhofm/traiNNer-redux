#!/usr/bin/env python3
"""
LUCID Source Cleanup & Resume
=============================
Author: Philip Hofmann

Description:
This script helps reclaim disk space after a partial Stage 1 run.
1. It analyzes the tiles in your output folder.
2. It maps them back to the original source files in ImageNet/PASS.
3. If an image is fully "covered" (tiles exist), it deletes the source file.
4. It creates a 'missing_images.txt' for use with lucid_stage1.py --file_list.

Usage:
  python lucid_cleanup_source.py --input /path/to/imagenet --output /path/to/tiles
"""

import argparse
import os
from pathlib import Path

from tqdm import tqdm


def main() -> None:
    # Lower process priority to keep system responsive
    try:
        os.nice(15)
        print("System Responsiveness Mode: Priority lowered to 15.")
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="Clean up source images that have already been tiled."
    )
    parser.add_argument(
        "--input", required=True, help="Original source dataset (ImageNet/PASS)"
    )
    parser.add_argument(
        "--output", required=True, help="Folder containing valid Stage 1 tiles"
    )
    parser.add_argument(
        "--delete", action="store_true", help="Actually delete the source files"
    )
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)

    print(f"Indexing tiles in {out_dir}...")
    # Map from name_base to existence
    # Note: Tile name is [name_base]_[scale]_[y]_[x].png
    # name_base is the relative path with / replaced by _

    found_bases = set()
    for p in tqdm(out_dir.glob("*.png"), desc="Indexing Tiles"):
        # We need to extract the name_base.
        # Since name_base itself can contain underscores, we look for the suffix _[scale]_[y]_[x].png
        # The scales are 100, 75, 50, 25.
        parts = p.stem.split("_")
        if len(parts) < 4:
            continue

        # The last 3 parts are scale, y, x
        # Everything before that is name_base
        name_base = "_".join(parts[:-3])
        found_bases.add(name_base)

    print(f"Found {len(found_bases)} unique source images represented in tiles.")

    print(f"Scanning source images in {in_dir}...")
    valid_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
    source_images = [
        p for p in in_dir.rglob("*") if p.suffix.lower() in valid_extensions
    ]
    print(f"Found {len(source_images)} source images total.")

    deleted_count = 0
    missing_paths = []

    try:
        for src_path in tqdm(source_images, desc="Checking Coverage"):
            try:
                rel_path = src_path.relative_to(in_dir)
            except ValueError:
                rel_path = src_path.name

            name_base = (
                str(Path(rel_path).with_suffix("")).replace("/", "_").replace("\\", "_")
            )

            if name_base in found_bases:
                # Source is processed
                if args.delete:
                    try:
                        src_path.unlink()
                        deleted_count += 1
                    except Exception as e:
                        print(f"Error deleting {src_path}: {e}")
                else:
                    deleted_count += 1  # Count as "covered"
            else:
                missing_paths.append(str(src_path))
    except KeyboardInterrupt:
        print(
            "\n\n!! Interrupted by user. Saving resume list for what was found so far..."
        )

    # Save missing list
    missing_file = Path("resume_stage1.txt")
    with open(missing_file, "w") as f:
        for p in missing_paths:
            f.write(p + "\n")

    print("\n=== CLEANUP RESULTS ===")
    if args.delete:
        print(f"Deleted Source Files: {deleted_count}")
    else:
        print(f"Covered Source Files: {deleted_count} (Would delete with --delete)")

    print(f"Missing Source Files: {len(missing_paths)}")
    print(f"Resume list saved to: {missing_file.absolute()}")

    if len(missing_paths) > 0:
        print("\nTo finish the dataset, run:")
        print(
            f"python scripts/lucid_filtering/lucid_stage1.py {in_dir} {out_dir} stats.csv --file_list {missing_file}"
        )


if __name__ == "__main__":
    main()
