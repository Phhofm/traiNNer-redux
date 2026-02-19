import csv
import os
import shutil
from pathlib import Path

from tqdm import tqdm


def main() -> None:
    try:
        os.nice(15)
    except Exception:
        pass

    # Paths
    lucid_dir = Path("/home/phips/Documents/dataset/lucid")
    # This should be the folder currently undergoing diversity filtering
    source_dir = lucid_dir / "MASTER_ELITE"
    lineage_path = lucid_dir / "master_lineage_scored.csv"

    # Final Output
    final_output_dir = lucid_dir / "MASTER_ELITE_ULTIMATE"
    final_lineage_path = lucid_dir / "master_lineage_ultimate.csv"

    print("--- Master Elite Sequential Finalizer ---")

    if not source_dir.exists():
        print(f"Error: {source_dir} not found!")
        return
    if not lineage_path.exists():
        print(f"Error: Lineage map {lineage_path} not found!")
        return

    # 1. Load Lineage Meta
    print("Loading lineage metadata...")
    lineage_lookup = {}
    with open(lineage_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            # key by current final_name
            lineage_lookup[row["final_name"]] = row

    # 2. Discover remaining files
    print(f"Scanning {source_dir} for remaining tiles...")
    current_files = sorted(source_dir.glob("*.png"))
    print(f"Found {len(current_files)} tiles after filtering.")

    if not current_files:
        print("No tiles found to finalize!")
        return

    # 3. Create Final Output Dir
    final_output_dir.mkdir(parents=True, exist_ok=True)

    # 4. Sequential Renaming & Record Building
    final_records = []

    # Optional: ensure fieldnames include 'complexity_score' if it's there
    if "complexity_score" not in fieldnames:
        fieldnames.append("complexity_score")

    print("Finalizing sequence...")
    # Use 1-indexed for final count
    for idx, file_path in enumerate(tqdm(current_files, desc="Re-indexing"), 1):
        old_name = file_path.name

        if old_name not in lineage_lookup:
            # Fallback if a ghost wasn't reintegrated but is in the folder
            # We'll tag it as recovered
            source_tag = (
                old_name.split("_", 1)[1].replace(".png", "")
                if "_" in old_name
                else "unknown"
            )
            meta = {
                "id": "RECOVERED",
                "final_name": old_name,
                "source_dataset": source_tag,
                "original_name": old_name,
                "original_path": "UNKNOWN_RECOVERED",
                "complexity_score": "-1.0",
            }
        else:
            meta = lineage_lookup[old_name]

        # New sequential naming: 000001_Source.png
        new_id_str = f"{idx:06d}"
        new_final_name = f"{new_id_str}_{meta['source_dataset']}.png"

        # Update meta record
        final_meta = meta.copy()
        final_meta["final_name"] = new_final_name
        # Keep track of old ID in a comment or separate col if needed,
        # but here we overwrite for the clean CSV.
        final_records.append(final_meta)

        # Physical Move (to save disk space)
        dst_path = final_output_dir / new_final_name
        try:
            # Using move instead of copy2 to save disk space
            shutil.move(str(file_path), str(dst_path))
        except Exception as e:
            print(f"Error moving {old_name}: {e}")

    # 5. Save Final Lineage
    print(f"Saving ultimate lineage to {final_lineage_path}...")
    with open(final_lineage_path, mode="w", encoding="utf-8", newline="") as f:
        # Use updated fieldnames (in case complexity_score was added)
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(final_records)

    print("\n--- Finalization Complete ---")
    print(f"Dataset Size: {len(final_records)} tiles")
    print(f"Location: {final_output_dir}")
    print(f"Lineage: {final_lineage_path}")
    print("You can now safely point your YAML to the ULTIMATE directory.")


if __name__ == "__main__":
    main()
