#!/usr/bin/env python3
"""
LUCID v2: Step 02 - Dedupe (Diversity Audit)
===========================================
Ensures the dataset maximizes information density by removing redundant textures.

Uses: ResNet18 feature fingerprints + Cosine Similarity.
Optimized: Batch-wise comparison with sliding window pool for speed and stability.
Safe: Uses os.nice(15) and small sleeps to keep the system responsive.
"""

import argparse
import gc
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
from tqdm import tqdm

# Allow massive images but skip them manually if size > 4096 to prevent stalls
Image.MAX_IMAGE_PIXELS = None


def get_feature_extractor(device: str) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Remove the classification head to get the 512-dim pooling layer
    model = nn.Sequential(*list(model.children())[:-1])
    model.to(device).eval()
    return model


def load_and_preprocess(p_preprocess_tuple: tuple) -> tuple:
    """Helper for parallel image loading."""
    p, preprocess = p_preprocess_tuple
    try:
        with Image.open(p) as img:
            # Option A: Pixel Limit (Skip massive images that stall the CPU)
            w, h = img.size
            if w > 4096 or h > 4096:
                return None, None

            img_rgb = img.convert("RGB")
            return preprocess(img_rgb), p
    except Exception:
        return None, None


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
    # Robust batch size for 1M+ images
    parser.add_argument("--batch", type=int, default=128, help="GPU batch size")
    parser.add_argument(
        "--max_pool",
        type=int,
        default=10000,
        help="Max unique tiles to compare against (Sliding Window)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.01,
        help="Sleep between batches for system responsiveness",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete redundant tiles immediately to free disk space",
    )

    args = parser.parse_args()
    input_dir = Path(args.input)

    # Fast Feedback Phase
    print("--- LUCID v2 Dedupe (Performance Mode) ---")
    print(f"Master Pool: {input_dir}")
    print("Scanning for images (this may take 5-20 minutes for millions of files)...")

    # Recursively find all PNGs and convert to strings early for memory efficiency
    image_paths = [str(p) for p in sorted(input_dir.rglob("*.png"))]

    if not image_paths:
        print("No images found.")
        return

    print(f"Done! Found {len(image_paths)} tiles to audit.")

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

    unique_pool = torch.empty((0, 512), device=device)
    redundant_paths = []

    redundant_dir = input_dir / "redundant"

    # Feature Extraction & Comparison
    # Using ThreadPoolExecutor for parallel I/O and preprocessing
    # Cap workers at 8 to prevent file handle/RAM spikes on 1M+ images
    max_workers = min(8, os.cpu_count() or 4)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i in tqdm(
            range(0, len(image_paths), args.batch), desc="Auditing Diversity"
        ):
            batch_paths = image_paths[i : i + args.batch]

            # Parallel Loading & Preprocessing (CPU)
            worker_args = [(p, preprocess) for p in batch_paths]
            results = list(executor.map(load_and_preprocess, worker_args))

            batch_tensors = [r[0] for r in results if r[0] is not None]
            valid_paths = [r[1] for r in results if r[1] is not None]

            if not batch_tensors:
                continue

            input_batch = torch.stack(batch_tensors).to(device)
            with torch.no_grad():
                features = model(input_batch).squeeze()  # [Batch, 512]
                if features.dim() == 1:  # Handle batch size 1
                    features = features.unsqueeze(0)
                features = nn.functional.normalize(features, p=2, dim=1)

                # --- HYPER-OPTIMIZED COMPARISON ---
                if unique_pool.size(0) > 0:
                    sims = torch.mm(features, unique_pool.t())  # [Batch, Pool]
                    max_sims, _ = sims.max(dim=1)
                else:
                    max_sims = torch.zeros(features.size(0), device=device)

                # Process the batch
                new_unique_feats = []
                for idx, sim in enumerate(max_sims):
                    p = valid_paths[idx]
                    feat = features[idx].unsqueeze(0)

                    if sim > args.threshold:
                        redundant_paths.append(p)
                    else:
                        new_unique_feats.append(feat)

                # Batch Update Unique Pool
                if new_unique_feats:
                    unique_pool = torch.cat([unique_pool, *new_unique_feats], dim=0)
                    # Manage sliding window pool size
                    if unique_pool.size(0) > args.max_pool:
                        unique_pool = unique_pool[-args.max_pool :]

            # Aggressive Memory Cleanup
            del input_batch
            del features
            del max_sims
            if "sims" in locals():
                del sims
            del batch_tensors
            del batch_paths
            del results

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Manual GC is ONLY needed for massive list scans if RAM is leaking.
            # On 1.7M items, it is too slow. Python's auto-GC is better.
            if i > 0 and i % (args.batch * 500) == 0:
                gc.collect()

            if args.sleep > 0:
                time.sleep(args.sleep)

    # Action
    if redundant_paths:
        if args.delete:
            print(f"\nDeleting {len(redundant_paths)} redundant tiles to free space...")
            for p in tqdm(redundant_paths, desc="Deleting"):
                try:
                    p.unlink()
                except Exception:
                    pass
        else:
            redundant_dir.mkdir(exist_ok=True)
            print(
                f"\nMoving {len(redundant_paths)} redundant tiles to {redundant_dir}..."
            )
            for p in tqdm(redundant_paths, desc="Moving"):
                try:
                    # Handle name collisions during move
                    dest = redundant_dir / p.name
                    if dest.exists():
                        dest = (
                            redundant_dir
                            / f"{p.stem}_{int(time.time() * 1000)}{p.suffix}"
                        )
                    shutil.move(str(p), str(dest))
                except Exception:
                    pass
    else:
        print("\nNo redundant textures found. Dataset is highly diverse!")

    print(f"Unique Tiles Remaining: {len(image_paths) - len(redundant_paths)}")


if __name__ == "__main__":
    main()
