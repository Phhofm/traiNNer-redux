#!/usr/bin/env python3
"""
LUCID v2: Step 02 - Dedupe (Diversity Audit)
===========================================
Ensures the dataset maximizes information density by removing redundant textures.

Uses: ResNet18 feature fingerprints + Cosine Similarity.
Safe: Uses os.nice(15) and handles large datasets in chunks.
"""

import argparse
import os
import shutil
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
from tqdm import tqdm


def get_feature_extractor(device: str):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Remove the classification head to get the 512-dim pooling layer
    model = nn.Sequential(*list(model.children())[:-1])
    model.to(device).eval()
    return model


def main() -> None:
    try:
        os.nice(15)
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="LUCID v2 - Step 02: Dedupe")
    parser.add_argument("--input", required=True, help="Input folder of tiles")
    parser.add_argument(
        "--threshold", type=float, default=0.96, help="Similarity threshold (0.90-0.98)"
    )
    parser.add_argument("--batch", type=int, default=64, help="GPU batch size")
    parser.add_argument(
        "--max_pool",
        type=int,
        default=10000,
        help="Max unique tiles to compare against (Sliding Window)",
    )

    args = parser.parse_args()
    input_dir = Path(args.input)
    # Recursively find all PNGs (handles the Step 01 subfolder structure)
    image_paths = sorted(input_dir.rglob("*.png"))

    if not image_paths:
        print("No images found.")
        return

    print("--- LUCID v2 Dedupe ---")
    print(f"Scanning {len(image_paths)} tiles...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_feature_extractor(device)

    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    unique_features = []  # List of torch tensors
    redundant_paths = []

    redundant_dir = input_dir / "redundant"

    # Feature Extraction & Comparison
    for i in tqdm(range(0, len(image_paths), args.batch), desc="Auditing Diversity"):
        batch_paths = image_paths[i : i + args.batch]
        batch_tensors = []
        valid_indices = []

        for idx, p in enumerate(batch_paths):
            try:
                img = Image.open(p).convert("RGB")
                batch_tensors.append(preprocess(img))
                valid_indices.append(idx)
            except Exception:
                continue

        if not batch_tensors:
            continue

        input_batch = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            features = model(input_batch).squeeze()  # [Batch, 512]
            # Normalize for cosine similarity calculation via dot product
            features = nn.functional.normalize(features, p=2, dim=1)

        # Compare against existing unique features
        for idx, feat in enumerate(features):
            is_redundant = False
            if unique_features:
                # Optimized: Compare batch feat against the pool
                pool = torch.stack(unique_features).to(device)
                similarities = torch.mm(feat.unsqueeze(0), pool.t())
                if torch.any(similarities > args.threshold):
                    is_redundant = True

            if is_redundant:
                redundant_paths.append(batch_paths[valid_indices[idx]])
            else:
                unique_features.append(feat.cpu())
                # Sliding window to keep O(N) complexity for massive datasets
                if len(unique_features) > args.max_pool:
                    unique_features.pop(0)

    # Action
    if redundant_paths:
        redundant_dir.mkdir(exist_ok=True)
        print(f"\nMoving {len(redundant_paths)} redundant tiles to {redundant_dir}...")
        for p in tqdm(redundant_paths, desc="Pruning"):
            try:
                shutil.move(str(p), str(redundant_dir / p.name))
            except Exception:
                pass
    else:
        print("\nNo redundant textures found. Dataset is highly diverse!")

    print(f"Unique Tiles Remaining: {len(image_paths) - len(redundant_paths)}")


if __name__ == "__main__":
    main()
