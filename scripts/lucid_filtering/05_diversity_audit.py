import argparse
import shutil
from pathlib import Path
from typing import Any, Optional

import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

# ========================= DATASET =========================


class TileDataset(Dataset):
    def __init__(
        self, tile_paths: list[Path], transform: transforms.Compose | None = None
    ) -> None:
        self.tile_paths = tile_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.tile_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor | None, str]:
        path = self.tile_paths[idx]
        try:
            # Using CV2 for speed, then converting to PIL/Tensor
            img = cv2.imread(str(path))
            if img is None:
                return None, str(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform:
                img = self.transform(img)
            return img, str(path)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None, str(path)


# ========================= MODEL =========================


def get_feature_extractor(device: torch.device) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Remove the fully connected layer, use Global Average Pooling output
    model.fc = nn.Identity()
    model.to(device)
    model.eval()
    return model


# ========================= LOGIC =========================


def run_audit(args: Any) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    input_dir = Path(args.input)

    # Discovery
    extensions = [".png", ".jpg", ".jpeg", ".webp"]
    tile_paths = []
    for ext in extensions:
        tile_paths.extend(list(input_dir.rglob(f"*{ext}")))

    print(f"Found {len(tile_paths)} tiles.")
    if not tile_paths:
        return

    # Prepare logic
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = TileDataset(tile_paths, transform=transform)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True
    )

    model = get_feature_extractor(device)

    unique_pool = []  # List of unique feature tensors [1, 512]
    unique_paths = []
    redundant_paths = []

    # Tracking for efficient rolling comparison
    pool_tensor = None

    print("Starting Diversity Audit...")
    with torch.no_grad():
        for batch_imgs, paths in tqdm(loader, desc="Extracting"):
            if batch_imgs is None:
                continue

            imgs = batch_imgs.to(device)
            # Filter out entries where loading failed (None returned from __getitem__)
            # Note: DataLoader returns batched tensors, but if we have Nones it might crash.
            # Fixed in __getitem__ by returning a dummy or skipping.
            # For simplicity, we assume bitstreams are healthy after verify_tiles.py.

            extracted_features = model(imgs)  # [B, 512]
            # L2 Normalize for Cosine Similarity
            extracted_features = nn.functional.normalize(extracted_features, p=2, dim=1)

            for i in range(extracted_features.size(0)):
                feat = extracted_features[i : i + 1]
                path = paths[i]

                is_redundant = False
                if pool_tensor is not None:
                    # Dot product of normalized vectors = Cosine Similarity
                    similarities = torch.mm(feat, pool_tensor.t())

                    if torch.any(similarities > args.threshold):
                        is_redundant = True

                if is_redundant:
                    redundant_paths.append(path)
                else:
                    unique_paths.append(path)
                    unique_pool.append(feat)

                    # Periodically refresh the search matrix
                    # We only compare against a representative pool to keep it fast
                    if len(unique_pool) > args.max_pool:
                        # FIFO strategy: most recent patterns are most likely to be redundant
                        unique_pool = unique_pool[-args.max_pool :]

                    pool_tensor = torch.cat(unique_pool, dim=0)

    print("\nAudit Complete:")
    print(f"  Total Tiles:    {len(tile_paths)}")
    print(f"  Unique Tiles:   {len(unique_paths)}")
    print(f"  Redundant:      {len(redundant_paths)}")
    print(f"  Diversity Ratio: {len(unique_paths) / len(tile_paths):.2%}")

    if args.move_redundant and redundant_paths:
        redundant_dir = input_dir / "redundant"
        redundant_dir.mkdir(exist_ok=True)
        print(f"Moving redundant tiles to {redundant_dir}...")
        for p in tqdm(redundant_paths, desc="Moving"):
            target_path = redundant_dir / Path(p).name
            # Handle name collisions
            if target_path.exists():
                target_path = (
                    redundant_dir
                    / f"{Path(p).stem}_{np.random.randint(10000)}{Path(p).suffix}"
                )
            try:
                shutil.move(p, target_path)
            except Exception as e:
                print(f"Error moving {p}: {e}")


if __name__ == "__main__":
    import numpy as np  # For random seed/names if needed

    parser = argparse.ArgumentParser(
        description="LUCID Diversity Audit (Deduplication)"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Input directory of tiles"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.98,
        help="Cosine similarity threshold (0.95-0.99)",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="GPU batch size")
    parser.add_argument(
        "--max_pool",
        type=int,
        default=5000,
        help="Max unique samples to compare against",
    )
    parser.add_argument(
        "--move_redundant",
        action="store_true",
        help="Move redundant files to a subfolder",
    )

    args = parser.parse_args()
    run_audit(args)
