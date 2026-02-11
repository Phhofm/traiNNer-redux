#!/usr/bin/env python3
"""
Copy Images by Score Threshold
==============================
Author: Philip Hofmann

Description:
Reads a CSV of image paths + complexity scores (from icnet_score_only.py)
and copies images that meet the threshold criteria to a target folder.

Usage Examples:
  # Copy top 25% (score >= threshold at 25th percentile)
  python copy_by_score.py --csv scores.csv --output elite/ --top_percent 25

  # Copy all images with score >= 0.45
  python copy_by_score.py --csv scores.csv --output elite/ --min_score 0.45

  # Copy images with score between 0.4 and 0.6
  python copy_by_score.py --csv scores.csv --output elite/ --min_score 0.4 --max_score 0.6
"""

import argparse
import csv
import os
import shutil
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
        description="Copy images from CSV based on score threshold."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Input CSV with columns: image_path, complexity_score",
    )
    parser.add_argument(
        "--output", required=True, help="Output folder to copy selected images to"
    )
    parser.add_argument(
        "--top_percent",
        type=float,
        default=None,
        help="Copy the top X%% of images by score",
    )
    parser.add_argument(
        "--min_score",
        type=float,
        default=None,
        help="Minimum score threshold (inclusive)",
    )
    parser.add_argument(
        "--max_score",
        type=float,
        default=None,
        help="Maximum score threshold (inclusive)",
    )
    parser.add_argument(
        "--symlink", action="store_true", help="Create symlinks instead of copying"
    )
    parser.add_argument(
        "--move", action="store_true", help="Move files instead of copying"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be copied without actually copying",
    )
    args = parser.parse_args()

    # Read CSV
    print(f"Reading {args.csv}...")
    entries = []
    with open(args.csv) as f:
        reader = csv.reader(f)
        _header = next(reader, None)
        for row in reader:
            if len(row) >= 2:
                path = row[0]
                score = float(row[1])
                entries.append((path, score))

    if not entries:
        print("No entries found in CSV!")
        return

    print(f"Total entries: {len(entries)}")

    # Sort by score descending
    entries.sort(key=lambda x: x[1], reverse=True)

    # Determine threshold
    if args.top_percent is not None:
        cutoff_idx = int(len(entries) * (args.top_percent / 100.0))
        if cutoff_idx > 0:
            threshold = entries[cutoff_idx - 1][1]
        else:
            threshold = entries[0][1]
        print(f"Top {args.top_percent}% threshold: score >= {threshold:.4f}")
        selected = [(p, s) for p, s in entries if s >= threshold]
    elif args.min_score is not None:
        min_s = args.min_score
        max_s = args.max_score if args.max_score is not None else float("inf")
        print(f"Score range: {min_s:.4f} <= score <= {max_s:.4f}")
        selected = [(p, s) for p, s in entries if min_s <= s <= max_s]
    else:
        print("ERROR: Specify either --top_percent or --min_score")
        return

    print(f"Selected: {len(selected)} images")

    if args.dry_run:
        print("\n[DRY RUN] Would copy:")
        for p, s in selected[:10]:
            print(f"  {Path(p).name} (score: {s:.4f})")
        if len(selected) > 10:
            print(f"  ... and {len(selected) - 10} more")
        return

    # Create output directory
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy/symlink files
    copied = 0
    for path_str, score in tqdm(selected, desc="Copying"):
        src = Path(path_str)
        if not src.exists():
            continue
        dst = out_dir / src.name

        # Handle duplicates by appending score to filename
        if dst.exists():
            stem = dst.stem
            suffix = dst.suffix
            dst = out_dir / f"{stem}_{score:.4f}{suffix}"

        try:
            if args.symlink:
                if dst.exists():
                    dst.unlink()
                dst.symlink_to(src.resolve())
            elif args.move:
                shutil.move(src, dst)
            else:
                shutil.copy2(src, dst)
            copied += 1
        except KeyboardInterrupt:
            print("\n\n!! Interrupted by user. Exiting gracefully...")
            break
        except Exception as e:
            print(f"Error copying {src}: {e}")

    print(f"\n✅ Finished processing. {copied} images to {out_dir}")


if __name__ == "__main__":
    main()
