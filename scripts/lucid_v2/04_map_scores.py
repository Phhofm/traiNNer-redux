import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def main() -> None:
    try:
        os.nice(15)
    except Exception:
        pass

    lucid_dir = Path("/home/phips/Documents/dataset/lucid")
    master_elite_dir = lucid_dir / "MASTER_ELITE"
    lineage_path = lucid_dir / "master_lineage.csv"
    output_lineage_path = lucid_dir / "master_lineage_scored.csv"
    ghost_dir = lucid_dir / "MASTER_ELITE_GHOSTS"

    print("--- Master Elite Score Mapper & Ghost Auditor v2 ---")

    # 1. Build Slug Map
    # Mapping from CSV filename slug to master_lineage source_dataset
    slug_map = {
        "bhi": "BHI_HR",
        "coco2017_train": "COCO2017_train_512",
        "coco2017_unlabeled": "COCO2017_unlabeled_512",
        "df2k": "DF2K_train_HR",
        "ffhq": "images1024x1024",
        "hq50k": "HQ50K_HR",
        "inaturalist2019": "inaturalist_2019",
        "lsdir": "LSDIR",
        "nomos8kswf": "nomos8k_sfw",
        "nomosuni": "nomos_uni",
        "uhdiqa": "uhdiqatraining",
        "unsplashlite": "unsplashlite",
    }

    # 2. Build Score Lookup Table
    score_lookup = {}
    tile_only_lookup = {}  # Fallback for unique names
    csv_files = list(lucid_dir.glob("*_master_elite.csv"))
    print(f"Loading scores from {len(csv_files)} source files...")

    for csv_file in tqdm(csv_files, desc="Parsing sources"):
        slug = csv_file.name.replace("_master_elite.csv", "")
        dataset_name = slug_map.get(slug, slug)
        try:
            df = pd.read_csv(csv_file)
            for _, row in df.iterrows():
                # Primary Key: (dataset_slug, tile_name)
                t_name = str(row["tile_name"])
                key = (dataset_name, t_name)
                score = row["complexity_score"]
                score_lookup[key] = score
                tile_only_lookup[t_name] = score
        except Exception as e:
            print(f"Error parsing {csv_file.name}: {e}")

    # 3. Update Master Lineage
    if not lineage_path.exists():
        print(f"Error: {lineage_path} not found!")
        return

    print("Mapping scores to lineage...")
    lineage_df = pd.read_csv(lineage_path)

    # Pre-allocate score column
    lineage_df["complexity_score"] = -1.0

    mapped_count = 0
    fuzzy_count = 0
    for idx, row in tqdm(lineage_df.iterrows(), total=len(lineage_df), desc="Mapping"):
        source = str(row["source_dataset"])
        name = str(row["original_name"])
        key = (source, name)

        if key in score_lookup:
            lineage_df.at[idx, "complexity_score"] = score_lookup[key]
            mapped_count += 1
        elif name in tile_only_lookup:
            lineage_df.at[idx, "complexity_score"] = tile_only_lookup[name]
            fuzzy_count += 1

    print(
        f"Successfully mapped {mapped_count} (exact) + {fuzzy_count} (fuzzy) = {mapped_count + fuzzy_count}/{len(lineage_df)} lineage entries."
    )
    lineage_df.to_csv(output_lineage_path, index=False)
    print(f"Saved updated lineage to: {output_lineage_path}")

    # 4. Audit Ghosts (Files on disk NOT in lineage)
    print("Auditing for remaining ghost tiles...")
    lineage_files = set(lineage_df["final_name"].tolist())

    ghosts = []
    if master_elite_dir.exists():
        with os.scandir(master_elite_dir) as entries:
            for entry in tqdm(entries, desc="Scanning MASTER_ELITE"):
                if entry.is_file() and entry.name.endswith(".png"):
                    if entry.name not in lineage_files:
                        ghosts.append(entry.name)

    if ghosts:
        print(f"Found {len(ghosts)} ghost tiles!")
        ghost_dir.mkdir(parents=True, exist_ok=True)
        print(f"Moving ghosts to {ghost_dir}...")
        for ghost in tqdm(ghosts, desc="Moving"):
            src = master_elite_dir / ghost
            dst = ghost_dir / ghost
            try:
                os.rename(src, dst)
            except Exception as e:
                pass
    else:
        print("No ghost tiles found in MASTER_ELITE.")

    print("--- Audit Complete ---")


if __name__ == "__main__":
    main()
