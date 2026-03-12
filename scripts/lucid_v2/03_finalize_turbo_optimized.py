#!/usr/bin/env python3
"""
LUCID v2: Step 03 - Finalize TURBO OPTIMIZED
==============================================
Optimized version for USB SSD performance with better parallelism
and reduced overhead.

Key Optimizations:
1. Eliminated directory scanning for ID lookup (CSV-only)
2. Used pathlib.rglob for faster file discovery
3. Improved chunking strategy for better I/O scheduling
4. Enhanced progress reporting accuracy
5. Maintained all original functionality and safety features
"""

import argparse
import csv
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm


def get_last_master_id(lineage_path: Path) -> int:
    """
    Reads the master lineage CSV to find the highest index.
    Much faster than directory scanning as CSV is significantly smaller.
    """
    if not lineage_path.exists():
        return 0

    last_id = 0
    try:
        with open(lineage_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    curr_id = int(row["id"])
                    if curr_id > last_id:
                        last_id = curr_id
                except (ValueError, KeyError):
                    continue
    except Exception as e:
        print(f"Warning: Could not parse {lineage_path.name}: {e}")
    return last_id


def process_tile(task: dict[str, Any]) -> dict[str, Any]:
    """Process a single tile (move/copy operation)."""
    src = task["src"]
    dst = task["dst"]
    try:
        if task["move"]:
            try:
                os.rename(src, dst)  # Faster if same filesystem
            except OSError:
                shutil.move(src, dst)  # Fallback for different filesystems
        else:
            shutil.copy2(src, dst)
        return {"record": task["record"], "success": True}
    except Exception as e:
        # Log error but don't crash - we'll skip this tile
        print(f"Error processing {src}: {e}")
        return {"success": False}


def main() -> None:
    # System safety: Run at low priority to avoid freezing UI
    try:
        os.nice(15)
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="LUCID v2 - Finalize TURBO OPTIMIZED (USB SSD friendly)"
    )
    parser.add_argument("--input", required=True, help="Input workspace with verified tiles")
    parser.add_argument("--output", required=True, help="Final Master Elite folder")
    parser.add_argument(
        "--move", action="store_true", help="Move instead of copying (saves space)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker threads (default: 4, adjust based on USB controller)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2500,
        help="Tiles to process per chunk (default: 2500, larger chunks reduce overhead)",
    )

    args = parser.parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    lineage_path = output_dir / "master_lineage.csv"
    last_id = get_last_master_id(lineage_path)

    # File discovery using pathlib (often faster than os.walk)
    print("Indexing remaining tiles...")
    all_pngs = list(input_dir.rglob("*.png"))
    # Filter out corrupted/redundant folders
    image_paths = [
        p
        for p in all_pngs
        if "corrupted_audit" not in p.parts and "redundant" not in p.parts
    ]
    image_paths = sorted(image_paths)  # Deterministic ordering

    if not image_paths:
        print("No remaining tiles found.")
        return

    print("--- LUCID v2 Finalize TURBO OPTIMIZED ---")
    print(f"Remaining: {len(image_paths)} tiles | Resume ID: {last_id}")
    print(f"Using {args.workers} workers with chunk size {args.chunk_size}")

    new_index = last_id + 1
    fieldnames = ["id", "final_name", "source_dataset", "original_name", "original_path"]
    file_exists = lineage_path.exists()

    with open(lineage_path, "a" if file_exists else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        pbar = tqdm(total=len(image_paths), desc="Consolidating", unit="tile")

        # Process in chunks to balance memory and I/O efficiency
        for i in range(0, len(image_paths), args.chunk_size):
            chunk = image_paths[i : i + args.chunk_size]
            tasks = []

            # Prepare tasks for this chunk
            for p in chunk:
                # Extract dataset tag from path structure
                if p.parent.name == "tiles":
                    tag = p.parent.parent.name
                else:
                    tag = p.parent.name if p.parent != input_dir else input_dir.name

                if tag == input_dir.name:
                    tag = "root"

                new_name = f"{new_index}_{tag}.png"
                dst = str(output_dir / new_name)
                src_str = str(p)

                tasks.append(
                    {
                        "src": src_str,
                        "dst": dst,
                        "move": args.move,
                        "record": {
                            "id": new_index,
                            "final_name": new_name,
                            "source_dataset": tag,
                            "original_name": p.name,
                            "original_path": src_str,
                        },
                    }
                )
                new_index += 1

            # Execute chunk with thread pool
            if args.workers > 1:
                with ThreadPoolExecutor(max_workers=args.workers) as executor:
                    # Submit all tasks and collect results as they complete
                    future_to_task = {
                        executor.submit(process_tile, task): task for task in tasks
                    }
                    results = []
                    for future in as_completed(future_to_task):
                        result = future.result()
                        results.append(result)
                        pbar.update(1)  # Update progress for each completed tile
            else:
                # Single-threaded fallback
                results = []
                for task in tasks:
                    result = process_tile(task)
                    results.append(result)
                    pbar.update(1)

            # Write successful records to CSV and flush to disk
            successful_records = [
                r["record"] for r in results if r.get("success", False)
            ]
            if successful_records:
                writer.writerows(successful_records)
                f.flush()  # Ensure data is written to disk for crash recovery

        pbar.close()

    print(f"\nFinal ID: {new_index - 1}")
    print(f"Master lineage saved to: {lineage_path}")


if __name__ == "__main__":
    main()