#!/usr/bin/env python3
"""
LUCID: Dataset Pruning Tool
===========================
Author: Philip Hofmann
Description:
Deletes low-scoring images from disk based on a complexity CSV.
Use this to free up massive amounts of disk space by keeping only the "Elite" tiles.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(description="Prune dataset by complexity score.")
    parser.add_argument("--csv", required=True, help="Path to complexity_scores.csv")
    parser.add_argument(
        "--top_percent",
        type=float,
        required=True,
        help="Percent of top images to KEEP (e.g. 25)",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually perform deletion (Omit for dry run)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: CSV not found at {csv_path}")
        sys.exit(1)

    # 1. Load scores
    print(f"Loading scores from {csv_path}...")
    items = []
    try:
        with open(csv_path) as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for row in reader:
                if len(row) >= 2:
                    items.append((row[0], float(row[1])))
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    if not items:
        print("No scores found in CSV.")
        sys.exit(1)

    # 2. Sort and calculate threshold
    items.sort(key=lambda x: x[1], reverse=True)

    keep_count = int(len(items) * (args.top_percent / 100.0))
    elite_set = {path for path, score in items[:keep_count]}
    to_delete = items[keep_count:]

    if keep_count > 0:
        cutoff = items[keep_count - 1][1]
    else:
        cutoff = 1.0

    print("\n--- Pruning Strategy ---")
    print(f"Total Images:    {len(items)}")
    print(f"Strategy:        Keep Top {args.top_percent}%")
    print(f"Keeping:         {keep_count} images (Score >= {cutoff:.4f})")
    print(f"Deleting:        {len(to_delete)} images (Score < {cutoff:.4f})")

    if not args.delete:
        print("\n[DRY RUN] No files were deleted. Add --delete to perform pruning.")
        return

    # 3. Perform deletion
    print(f"\nPerforming Pruning (Safety: KEEPing {len(elite_set)} files)...")

    deleted_count = 0
    errors = 0
    freed_bytes = 0

    for path_str, _score in tqdm(to_delete, desc="Pruning"):
        path = Path(path_str)
        if path.exists():
            try:
                # Track size for stats
                freed_bytes += path.stat().st_size
                path.unlink()
                deleted_count += 1
            except Exception as e:
                errors += 1
                if errors < 10:
                    print(f"Error deleting {path}: {e}")

    print("\n--- Pruning Results ---")
    print(f"Successfully deleted: {deleted_count} files")
    print(f"Space freed:          {freed_bytes / (1024**3):.2f} GB")
    if errors > 0:
        print(f"Errors encountered:   {errors}")

    print("\nPruning complete. Your disk has been reclaimed.")


if __name__ == "__main__":
    main()
