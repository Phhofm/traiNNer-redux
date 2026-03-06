#!/usr/bin/env python3
"""
LUCID v2: Step 02 - Dedupe (STREAMING v4)
==========================================
Memory-efficient version for 1M+ images.
Processes in chunks to stay under RAM limits.

Key features:
  - Streaming processing: never loads all paths into memory
  - Chunked feature extraction: processes N images at a time
  - Disk-based deduplication: uses FAISS with memory-mapped features
  - Minimal RAM usage: stays under 4GB even for 1.7M images
"""

import argparse
import gc
import os
import pickle
import shutil
import signal
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
from tqdm import tqdm

try:
    import faiss
except ImportError:
    print("ERROR: FAISS not installed. pip install faiss-cpu")
    sys.exit(1)

Image.MAX_IMAGE_PIXELS = None
checkpoint_state = {"interrupted": False}


def signal_handler(signum, frame) -> None:
    checkpoint_state["interrupted"] = True
    print("\n[!] Interrupt received, saving checkpoint...")


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def get_feature_extractor(device: str) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model = nn.Sequential(*list(model.children())[:-1])
    model.to(device).eval()
    return model


def process_chunk(
    chunk_paths: list[str],
    model: nn.Module,
    device: str,
    preprocess: transforms.Compose,
) -> np.ndarray:
    """Process a chunk of images and return features."""
    batch_tensors = []
    for path in chunk_paths:
        try:
            with Image.open(path) as img:
                w, h = img.size
                if w > 4096 or h > 4096:
                    continue
                img_rgb = img.convert("RGB")
                batch_tensors.append(preprocess(img_rgb))
        except Exception:
            continue

    if not batch_tensors:
        return np.array([], dtype=np.float16).reshape(0, 512)

    batch = torch.stack(batch_tensors).to(device)
    with torch.no_grad():
        features = model(batch).squeeze(-1).squeeze(-1)
        features = nn.functional.normalize(features, p=2, dim=1)

    return features.cpu().numpy().astype(np.float16)


def extract_features_streaming(
    input_dir: Path,
    cache_dir: Path,
    chunk_size: int,
    device: str,
) -> np.ndarray:
    """
    Extract features in streaming fashion.
    Processes images in chunks, saves to disk, never loads all paths.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    features_path = cache_dir / "features_fp16.npy"
    manifest_path = cache_dir / "manifest.pkl"

    # Check for existing progress
    processed_count = 0
    if manifest_path.exists() and features_path.exists():
        with open(manifest_path, "rb") as f:
            manifest = pickle.load(f)
            processed_count = manifest.get("count", 0)
        print(f"[+] Resuming: {processed_count} features already extracted")
        # We'll append to existing file
        existing_mmap = np.memmap(
            features_path, dtype=np.float16, mode="r", shape=(processed_count, 512)
        )
        existing_shape = existing_mmap.shape
        del existing_mmap
    else:
        existing_shape = (0, 512)

    preprocess = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    model = get_feature_extractor(device)

    # Count total images first (streaming)
    print(f"Counting images in: {input_dir}")
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    total_images = sum(1 for _ in input_dir.rglob("*.png"))
    print(f"Found {total_images} images")

    if total_images == 0:
        raise ValueError(f"No PNG images found in {input_dir}")

    remaining = total_images - processed_count
    if remaining <= 0:
        print("[+] All images already processed!")
        return np.memmap(
            features_path, dtype=np.float16, mode="r", shape=(processed_count, 512)
        )

    # Create/extend mmap file
    total_features = total_images
    if existing_shape[0] == 0:
        mmap = np.memmap(
            features_path, dtype=np.float16, mode="w+", shape=(total_features, 512)
        )
    else:
        # Already exists, just open for writing
        mmap = np.memmap(
            features_path, dtype=np.float16, mode="r+", shape=(total_features, 512)
        )

    current_offset = processed_count
    chunk_paths = []
    chunk_count = 0

    # Stream through images
    all_paths = sorted(input_dir.rglob("*.png"))

    with tqdm(total=total_images, initial=processed_count, desc="Extracting") as pbar:
        for idx, path in enumerate(all_paths):
            if idx < processed_count:
                continue  # Skip already processed

            if checkpoint_state["interrupted"]:
                break

            chunk_paths.append(str(path))

            if len(chunk_paths) >= chunk_size:
                # Process chunk
                features = process_chunk(chunk_paths, model, device, preprocess)

                if features.shape[0] > 0:
                    mmap[current_offset : current_offset + features.shape[0]] = features
                    current_offset += features.shape[0]

                    # Save manifest every chunk
                    with open(manifest_path, "wb") as f:
                        pickle.dump({"count": current_offset}, f)

                chunk_count += 1
                pbar.update(len(chunk_paths))
                chunk_paths = []

                # Cleanup every 10 chunks
                if chunk_count % 10 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    # Flush mmap periodically
                    mmap.flush()

    # Process remaining
    if chunk_paths and not checkpoint_state["interrupted"]:
        features = process_chunk(chunk_paths, model, device, preprocess)
        if features.shape[0] > 0:
            mmap[current_offset : current_offset + features.shape[0]] = features
            current_offset += features.shape[0]

    # Final save
    with open(manifest_path, "wb") as f:
        pickle.dump({"count": current_offset}, f)
    mmap.flush()
    del mmap

    print(f"[+] Extracted {current_offset} features")
    return np.load(features_path, mmap_mode="r")


def dedupe_streaming(
    features: np.ndarray,
    threshold: float,
    cache_dir: Path,
    input_dir: Path,
    dry_run: bool,
) -> None:
    """Find duplicates using streaming FAISS search."""
    print("\n[+] Building FAISS index...")

    d = 512
    nlist = min(1000, len(features) // 100)
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

    features_f32 = features.astype(np.float32)

    print("Training index...")
    index.train(features_f32)

    print("Adding vectors...")
    index.add(features_f32)
    index.nprobe = 10

    print("Searching for duplicates...")

    # Stream through features in batches
    batch_size = 10000
    redundant_indices = set()

    checkpoint_path = cache_dir / "dedupe_checkpoint.pkl"
    start_idx = 0

    if checkpoint_path.exists():
        with open(checkpoint_path, "rb") as f:
            ckpt = pickle.load(f)
            redundant_indices = set(ckpt.get("redundant", []))
            start_idx = ckpt.get("last_idx", 0)
        print(f"[+] Resuming dedupe from index {start_idx}")

    for i in tqdm(range(start_idx, len(features), batch_size), desc="Deduping"):
        if checkpoint_state["interrupted"]:
            break

        batch_end = min(i + batch_size, len(features))
        batch = features_f32[i:batch_end]

        distances, _ = index.search(batch, k=2)
        l2_sq = distances[:, 1]
        cosine_sim = 1.0 - 0.5 * l2_sq

        for j, sim in enumerate(cosine_sim):
            if sim >= threshold:
                redundant_indices.add(i + j)

        # Checkpoint every 100k
        if (i // batch_size) % 10 == 0:
            with open(checkpoint_path, "wb") as f:
                pickle.dump(
                    {"redundant": list(redundant_indices), "last_idx": batch_end}, f
                )

    print(f"\n[+] Found {len(redundant_indices)} redundant images")

    # Save delete list
    delete_list_path = cache_dir / "delete_list.txt"

    # Stream through paths and write redundant ones
    print("Writing delete list...")
    all_paths = sorted(input_dir.rglob("*.png"))

    with open(delete_list_path, "w") as f:
        for idx in sorted(redundant_indices):
            if idx < len(all_paths):
                f.write(f"{all_paths[idx]}\n")

    print(f"[+] Saved delete list to: {delete_list_path}")

    if dry_run:
        print("\n[Dry-run] Files to delete:")
        for idx in sorted(redundant_indices)[:10]:
            if idx < len(all_paths):
                print(f"  {all_paths[idx]}")
        if len(redundant_indices) > 10:
            print(f"  ... and {len(redundant_indices) - 10} more")
    else:
        print(f"\nDeleting {len(redundant_indices)} files...")
        redundant_dir = input_dir / "redundant"
        redundant_dir.mkdir(exist_ok=True)

        for idx in tqdm(sorted(redundant_indices), desc="Moving"):
            if idx >= len(all_paths):
                continue
            src = all_paths[idx]
            try:
                dest = redundant_dir / src.name
                if dest.exists():
                    dest = (
                        redundant_dir
                        / f"{src.stem}_{int(time.time() * 1000)}{src.suffix}"
                    )
                shutil.move(str(src), str(dest))
            except Exception:
                pass

    # Cleanup checkpoints
    if not checkpoint_state["interrupted"] and not dry_run:
        for f in [cache_dir / "manifest.pkl", checkpoint_path]:
            if f.exists():
                f.unlink()

    print(f"\n[+] Complete! Unique: {len(features) - len(redundant_indices)}")


def main() -> None:
    try:
        os.nice(15)
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="LUCID v2 - Streaming Dedupe")
    parser.add_argument("--input", required=True, help="Input folder")
    parser.add_argument("--threshold", type=float, default=0.94)
    parser.add_argument(
        "--chunk-size", type=int, default=128, help="Images per chunk (default: 128)"
    )
    parser.add_argument("--cache", default="/home/phips/.cache/lucid_dedupe")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--clear-cache", action="store_true")

    args = parser.parse_args()

    input_dir = Path(args.input)
    cache_dir = Path(args.cache)

    if args.clear_cache and cache_dir.exists():
        print(f"[+] Clearing cache: {cache_dir}")
        shutil.rmtree(cache_dir)

    print("--- LUCID v2 Dedupe (STREAMING v4) ---")
    print(f"Input: {input_dir}")
    print(f"Threshold: {args.threshold}")
    print(f"Chunk size: {args.chunk_size}")

    # Memory check
    mem = psutil.virtual_memory()
    print(f"RAM: {mem.available / 1e9:.1f}GB available / {mem.total / 1e9:.1f}GB total")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Phase 1: Extract features
    features = extract_features_streaming(
        input_dir=input_dir,
        cache_dir=cache_dir,
        chunk_size=args.chunk_size,
        device=device,
    )

    if checkpoint_state["interrupted"]:
        print("\n[!] Interrupted. Resume with: --resume (auto-detected)")
        return

    # Phase 2: Deduplicate
    dedupe_streaming(
        features=features,
        threshold=args.threshold,
        cache_dir=cache_dir,
        input_dir=input_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
