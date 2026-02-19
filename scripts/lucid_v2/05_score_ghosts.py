import csv
import os
import sys
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

# ICNet Setup
REPO_ROOT = Path("/home/phips/Documents/GitHub/traiNNer-redux")
ICNET_DIR = REPO_ROOT / "datasets" / "preparation" / "complexity"
sys.path.append(str(ICNET_DIR))

try:
    from ICNet import ICNet
except ImportError:
    print(f"FATAL: Could not import ICNet from {ICNET_DIR}")
    sys.exit(1)


def main() -> None:
    try:
        os.nice(15)
    except Exception:
        pass

    lucid_dir = Path("/home/phips/Documents/dataset/lucid")
    ghost_dir = lucid_dir / "MASTER_ELITE_GHOSTS"
    model_path = (
        REPO_ROOT / "datasets" / "preparation" / "complexity" / "complexity.pth"
    )
    output_csv = lucid_dir / "ghost_lineage_scored.csv"

    if not ghost_dir.exists():
        print(f"Error: {ghost_dir} not found!")
        return

    # 1. Setup ICNet
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading ICNet on {device}...")
    model = ICNet(is_pretrain=False)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.to(device).eval()
    if device == "cuda":
        model.half()

    # Pre-processing constants
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    if device == "cuda":
        mean = mean.half()
        std = std.half()

    # 2. Discovery
    ghost_files = sorted(ghost_dir.glob("*.png"))
    if not ghost_files:
        print("No ghost tiles found.")
        return
    print(f"Found {len(ghost_files)} ghost tiles to score.")

    batch_size = 64
    results = []

    # 3. Batch Scoring
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "final_name",
                "source_dataset",
                "original_name",
                "original_path",
                "complexity_score",
            ]
        )

    # Progress bar for batches
    for i in tqdm(range(0, len(ghost_files), batch_size), desc="Scoring Ghosts"):
        batch_paths = ghost_files[i : i + batch_size]
        batch_tensors = []
        batch_metadata = []

        for p in batch_paths:
            try:
                # Load image
                img_bgr = cv2.imread(str(p))
                if img_bgr is None:
                    continue
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                # Metadata from filename: ID_SOURCE.png
                parts = p.name.split("_", 1)
                tile_id = parts[0]
                source = parts[1].replace(".png", "") if len(parts) > 1 else "unknown"

                batch_tensors.append(torch.from_numpy(img_rgb))
                batch_metadata.append(
                    {
                        "id": tile_id,
                        "final_name": p.name,
                        "source": source,
                        "orig_name": p.name,  # Lost original name, using final as proxy
                        "orig_path": "RECOVERED_GHOST",
                    }
                )
            except Exception as e:
                print(f"Error loading {p.name}: {e}")

        if not batch_tensors:
            continue

        # GPU Scoring
        tensors = torch.stack(batch_tensors).to(device).permute(0, 3, 1, 2)
        if device == "cuda":
            tensors = tensors.half()
        tensors = (tensors / 255.0 - mean) / std

        with torch.no_grad():
            scores, _ = model(tensors)
            scores = scores.flatten().cpu().float().numpy().tolist()

        # Write to CSV immediately
        with open(output_csv, "a", newline="") as f:
            writer = csv.writer(f)
            for meta, score in zip(batch_metadata, scores, strict=False):
                writer.writerow(
                    [
                        meta["id"],
                        meta["final_name"],
                        meta["source"],
                        meta["orig_name"],
                        meta["orig_path"],
                        f"{score:.6f}",
                    ]
                )

    print(f"\nScoring Complete. Results saved to {output_csv}")
    print(
        "Next step: Run the reintegration script to merge these back into the lineage."
    )


if __name__ == "__main__":
    main()
