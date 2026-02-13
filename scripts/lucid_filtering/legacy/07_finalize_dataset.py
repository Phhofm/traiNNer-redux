import argparse
import csv
import shutil
from pathlib import Path
from typing import Any

from tqdm import tqdm


def finalize_dataset(args: Any) -> None:
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_log_path = output_dir / "master_elite_lineage.csv"

    # 1. Discovery
    # We look for subdirectories first. If no subdirs, we treat the input as one dataset.
    subdirs = [d for d in input_dir.iterdir() if d.is_dir()]

    if not subdirs:
        print(
            f"No subdirectories found in {input_dir}. Treating current folder as the dataset."
        )
        # We'll use the input_dir name as the tag
        dataset_buckets = {input_dir.name: list(input_dir.glob("*.png"))}
    else:
        dataset_buckets = {}
        for d in subdirs:
            # If the subdir is "tiles", we try to use its parent name
            tag = d.parent.name if d.name == "tiles" else d.name
            tiles = list(d.rglob("*.png"))
            if tiles:
                dataset_buckets[tag] = tiles

    if not dataset_buckets:
        print("No PNG tiles found.")
        return

    print(f"Found {len(dataset_buckets)} source groups.")
    for tag, tiles in dataset_buckets.items():
        print(f"  - {tag}: {len(tiles)} tiles")

    # 2. Sequential Processing
    global_index = 1
    lineage_data = []

    # Sort tags to ensure deterministic numbering across runs
    sorted_tags = sorted(dataset_buckets.keys())

    for tag in sorted_tags:
        tiles = sorted(dataset_buckets[tag])
        print(f"\nProcessing {tag}...")

        for tile_path in tqdm(tiles, desc=f"Finalizing {tag}"):
            ext = tile_path.suffix
            new_name = f"{global_index}_{tag}{ext}"
            dst_path = output_dir / new_name

            # Record lineage
            lineage_data.append(
                {
                    "id": global_index,
                    "final_name": new_name,
                    "source_dataset": tag,
                    "original_name": tile_path.name,
                    "original_path": str(tile_path.absolute()),
                }
            )

            # Copy or Move
            try:
                if args.move:
                    shutil.move(str(tile_path), str(dst_path))
                else:
                    shutil.copy2(str(tile_path), str(dst_path))
                global_index += 1
            except Exception as e:
                print(f"Error processing {tile_path.name}: {e}")

    # 3. Save Lineage CSV
    print(f"\nWriting lineage log to {csv_log_path}...")
    with open(csv_log_path, "w", newline="") as f:
        if lineage_data:
            writer = csv.DictWriter(f, fieldnames=lineage_data[0].keys())
            writer.writeheader()
            writer.writerows(lineage_data)

    print("\n=== FINALIZATION COMPLETE ===")
    print(f"  Total Tiles: {global_index - 1}")
    print(f"  Output Dir:  {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LUCID Master Elite Finalizer & Traceability Tool"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory containing dataset subfolders",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Final output directory for Master Elite",
    )
    parser.add_argument(
        "--move", action="store_true", help="Move files instead of copying"
    )

    args = parser.parse_args()
    finalize_dataset(args)
