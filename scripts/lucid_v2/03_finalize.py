#!/usr/bin/env python3
"""
LUCID v2: Step 03 - Finalize (Consolidate & Append)
==================================================
The final integration step. Moves unique, verified tiles into the Master Elite collection.

Features:
- **Append Mode**: Automatically starts numbering from the last tile in the Master folder.
- **Traceability**: Updates a central master_lineage.csv mapping every tile to its source.
- **Organization**: Standardizes naming to {INDEX}_{TAG}.png

Safe: Uses os.nice(15).
"""

import argparse
import csv
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm


def get_last_master_id(lineage_path: Path) -> int:
    """Reads the master lineage CSV to find the highest index."""
    if not lineage_path.exists():
        return 0

    last_id = 0
    try:
        with open(lineage_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    curr_id = int(row["id"])
                    last_id = max(last_id, curr_id)
                except (ValueError, KeyError):
                    continue
    except Exception as e:
        print(f"Warning: Could not parse {lineage_path.name}: {e}")
    return last_id


def main() -> None:
    try:
        os.nice(15)
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="LUCID v2 - Step 03: Finalize")
    parser.add_argument(
        "--input", required=True, help="Input workspace with verified tiles"
    )
    parser.add_argument("--output", required=True, help="Final Master Elite folder")
    parser.add_argument("--move", action="store_true", help="Move instead of copying")

    args = parser.parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    lineage_path = output_dir / "master_lineage.csv"
    last_id = get_last_master_id(lineage_path)

    # Discovery
    # We recursively find all PNGs in the input, but EXCLUDE any 'corrupted_audit' or 'redundant' folders
    all_pngs = list(input_dir.rglob("*.png"))
    image_paths = [
        p
        for p in all_pngs
        if "corrupted_audit" not in p.parts and "redundant" not in p.parts
    ]
    image_paths = sorted(image_paths)  # Sort for deterministic numbering

    if not image_paths:
        print("No valid tiles found to finalize.")
        return

    # Dataset Tag Extraction
    # We assume Step 01 structure: input_dir / dataset_name / tiles / *.png
    # Or just input_dir being the dataset folder.

    print("--- LUCID v2 Finalize ---")
    if last_id > 0:
        print(f"APPEND MODE: Found existing Master collection ending at ID {last_id}.")
    else:
        print("INITIALIZE: Creating new Master collection.")

    print(f"Consolidating {len(image_paths)} tiles...")

    new_index = last_id + 1
    new_records = []

    for p in tqdm(image_paths, desc="Finalizing"):
        # Determine tag (dataset name)
        # 1. Check if it's in a subfolder of input
        if p.parent.name == "tiles":
            tag = p.parent.parent.name
        else:
            tag = p.parent.name if p.parent != input_dir else input_dir.name

        new_name = f"{new_index}_{tag}.png"
        dst = output_dir / new_name

        # Lineage Record
        new_records.append(
            {
                "id": new_index,
                "final_name": new_name,
                "source_dataset": tag,
                "original_name": p.name,
                "original_path": str(p.absolute()),
            }
        )

        # Copy/Move
        try:
            if args.move:
                shutil.move(str(p), str(dst))
            else:
                shutil.copy2(str(p), str(dst))
            new_index += 1
        except Exception as e:
            print(f"Error finalizing {p.name}: {e}")

    # Write/Append CSV
    file_exists = lineage_path.exists()
    fieldnames = [
        "id",
        "final_name",
        "source_dataset",
        "original_name",
        "original_path",
    ]

    with open(lineage_path, "a" if file_exists else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(new_records)

    print("\n=== FINALIZATION COMPLETE ===")
    print(f"  Total Master Tiles: {new_index - 1}")
    print(f"  Lineage Log:       {lineage_path}")
    print(f"  Destination:       {output_dir}")


if __name__ == "__main__":
    main()
