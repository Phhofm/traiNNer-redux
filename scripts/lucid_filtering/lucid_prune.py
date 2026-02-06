import argparse
import csv
import os
import shutil
from pathlib import Path

from tqdm import tqdm


def prune() -> None:
    parser = argparse.ArgumentParser(
        description="Prune LUCID dataset based on PSNR percentiles."
    )
    parser.add_argument(
        "--img_dir",
        required=True,
        help="Folder containing the processed tiles (source)",
    )
    parser.add_argument(
        "--out_dir", required=True, help="Where to save the elite tiles (target)"
    )
    parser.add_argument(
        "--s2_csv_paths",
        nargs="+",
        required=True,
        help="Paths to stage2_psnr.csv files",
    )
    parser.add_argument(
        "--s1_csv_paths",
        nargs="*",
        default=[],
        help="Optional paths to lucid_stage1_stats.csv files for master metadata",
    )
    parser.add_argument(
        "--top_percent",
        type=float,
        default=25.0,
        help="Top percentage to keep (default: 25.0)",
    )
    parser.add_argument(
        "--master_csv",
        required=True,
        help="Path where the final master stats CSV should be saved",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Use symlinks instead of copying (DANGEROUS: Tiles won't be portable)",
    )
    parser.add_argument(
        "--rename_index",
        action="store_true",
        help="Rename tiles to 0.png, 1.png, etc. (mapped in CSV)",
    )
    args = parser.parse_args()

    img_dir = Path(args.img_dir).absolute()
    out_dir = Path(args.out_dir).absolute()
    out_dir.mkdir(parents=True, exist_ok=True)
    master_csv_path = Path(args.master_csv).absolute()
    master_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Build a Stem Map of the physical tiles
    print(f"Scanning {img_dir}...")
    stem_map = {}
    for f in img_dir.glob("*.png"):
        name = f.name
        stem = name[:-4]
        if name not in stem_map:
            stem_map[name] = []
        stem_map[name].append(name)

        parts = stem.split("_")
        if len(parts) > 3:
            orig_stem = "_".join(parts[:-3])
            orig_name = orig_stem + ".png"
            if orig_name not in stem_map:
                stem_map[orig_name] = []
            stem_map[orig_name].append(name)

    # 2. Map Stage 1 Metrics (Optional)
    s1_data = {}  # tile_name -> metrics_dict
    if args.s1_csv_paths:
        print("Collecting Stage 1 metrics...")
        for csv_path in args.s1_csv_paths:
            csv_path = Path(csv_path)
            if not csv_path.exists():
                continue
            with open(csv_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    s1_data[row["tile"]] = row

    # 3. Collect Stage 2 PSNR and build the elite pool
    print("Collecting Stage 2 scores...")
    tile_pool = []  # List of dicts
    for csv_path in args.s2_csv_paths:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            continue

        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    if (
                        row.get("kept") == "False"
                        or not row.get("psnr_x4")
                        or row.get("psnr_x4") == "None"
                    ):
                        continue

                    csv_tile = row["tile"]
                    if csv_tile in stem_map:
                        psnr = float(row["psnr_x4"])
                        # One CSV entry might match multiple physical tiles
                        for actual_name in stem_map[csv_tile]:
                            entry = {
                                "original_tile": actual_name,
                                "source_image": csv_tile,
                                "psnr_x4": psnr,
                                "psnr_x2": row.get("psnr_x2", "N/A"),
                            }
                            # Merge stage 1 metrics if we have them
                            # Try matching by actual name first, then by the source_image (stem)
                            m = s1_data.get(actual_name) or s1_data.get(csv_tile)
                            if m:
                                entry.update(
                                    {
                                        "entropy": m.get("entropy"),
                                        "lap_var": m.get("lap_var"),
                                        "grad_energy": m.get("grad_energy"),
                                        "blockiness": m.get("blockiness"),
                                        "aliasing": m.get("aliasing"),
                                        "noise_ratio": m.get("noise_ratio"),
                                    }
                                )
                            tile_pool.append(entry)
                except:
                    continue

    if not tile_pool:
        print("Error: No matching records found.")
        return

    # 4. Sort and Slice
    print("Sorting by PSNR...")
    tile_pool.sort(key=lambda x: x["psnr_x4"], reverse=True)
    keep_count = int(len(tile_pool) * (args.top_percent / 100.0))
    elite_pool = tile_pool[:keep_count]
    print(f"Selected Top {args.top_percent}%: {len(elite_pool)} tiles.")

    # 5. Execute Pruning and Renaming
    with open(master_csv_path, "w", newline="", encoding="utf-8") as f:
        # Determine CSV header
        fieldnames = [
            "new_filename",
            "original_tile",
            "psnr_x4",
            "psnr_x2",
            "entropy",
            "lap_var",
            "grad_energy",
            "blockiness",
            "aliasing",
            "noise_ratio",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        action_str = "Symlinking" if args.symlink else "Copying"
        print(
            f"{action_str} tiles to {out_dir} and writing Master CSV to {master_csv_path}..."
        )

        for i, entry in enumerate(tqdm(elite_pool)):
            new_name = f"{i}.png" if args.rename_index else entry["original_tile"]
            entry["new_filename"] = new_name

            src = img_dir / entry["original_tile"]
            dst = out_dir / new_name

            if args.symlink:
                if dst.exists():
                    dst.unlink()
                os.symlink(src, dst)
            elif not dst.exists():
                shutil.copy2(src, dst)

            # Filter dict to match header
            csv_row = {k: entry.get(k, "") for k in fieldnames}
            writer.writerow(csv_row)

    print(f"\nSuccess! Elite dataset is ready at: {out_dir}")
    print(f"Master metadata saved to: {master_csv_path}")
    print(f"Minimum PSNR in this set: {elite_pool[-1]['psnr_x4']:.2f} dB")


if __name__ == "__main__":
    prune()
