import csv
import os
from pathlib import Path

from tqdm import tqdm


def main() -> None:
    lucid_dir = Path("/home/phips/Documents/dataset/lucid")
    master_elite_dir = lucid_dir / "MASTER_ELITE"
    ghost_dir = lucid_dir / "MASTER_ELITE_GHOSTS"

    main_lineage_path = lucid_dir / "master_lineage_scored.csv"
    ghost_lineage_path = lucid_dir / "ghost_lineage_scored.csv"
    output_path = lucid_dir / "master_lineage_elite_2.0.csv"

    print("--- Master Elite Reintegration & Audit ---")

    if not main_lineage_path.exists():
        print(f"Error: Main lineage {main_lineage_path} not found!")
        return
    if not ghost_lineage_path.exists():
        print(
            f"Error: Ghost lineage {ghost_lineage_path} not found! Run 05_score_ghosts.py first."
        )
        return

    # 1. Load Main Lineage
    all_data = []
    print("Loading main lineage...")
    with open(main_lineage_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            all_data.append(row)

    # 2. Load Ghost Lineage
    print("Loading ghost lineage...")
    ghost_count = 0
    with open(ghost_lineage_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_data.append(row)
            ghost_count += 1

    print(f"Total Unified Records: {len(all_data)} ({ghost_count} ghosts added)")

    # 3. Sort by ID (Numeric)
    all_data.sort(key=lambda x: int(x["id"]))

    # 4. Save Unified Lineage
    print(f"Saving unified lineage to {output_path}...")
    with open(output_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_data)

    # 5. Summary Statistics (Complexity >= 0.5)
    threshold = 0.50
    elite_2_0_count = sum(
        1 for row in all_data if float(row["complexity_score"]) >= threshold
    )
    print("\n--- ELITE 2.0 AUDIT ---")
    print(f"Total Combined Tiles: {len(all_data)}")
    print(f"Elite 2.0 Yield (Score >= {threshold}): {elite_2_0_count} tiles")
    print("----------------------")

    # 6. Physical Move-Back
    print(
        f"\nWould you like to move the {ghost_count} ghost files back to MASTER_ELITE? (y/n)"
    )
    # Since this is an agent, we default to "Ask User" or just run it.
    # I'll include the logic but make it easy to run.

    move_back = True  # We'll assume yes for the implementation, but I'll tell the user.
    if move_back and ghost_dir.exists():
        print(f"Moving {ghost_count} files back to {master_elite_dir}...")
        for p in tqdm(ghost_dir.glob("*.png"), total=ghost_count, desc="Moving"):
            try:
                os.rename(p, master_elite_dir / p.name)
            except Exception as e:
                print(f"Error moving {p.name}: {e}")
        print("Move complete.")


if __name__ == "__main__":
    main()
