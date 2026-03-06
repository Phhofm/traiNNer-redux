#!/usr/bin/env python3
"""
LUCID v2: Step 02 - Dedupe (FAST v3)
=====================================
Optimized for 1M+ images on USB SSD.

Architecture:
  Phase 1: Extract ResNet18 features using DataLoader with parallel workers
           Cache features to local disk (fp16) for resumability
  Phase 2: FAISS-based approximate nearest neighbor deduplication
           O(N log N) instead of O(N²)

Key optimizations:
  - DataLoader prefetching hides USB I/O latency
  - fp16 features: 50% memory reduction, faster GPU transfers
  - FAISS IVF index: sub-linear search time
  - Checkpoint/resume: survive crashes without losing progress
  - Memory-mapped cache: no need to keep all features in RAM
  - System-friendly: nice/ionice, periodic sleeps, memory monitoring

Usage:
  nice -n 19 ionice -c 3 python3 scripts/lucid_v2/02_dedupe_fast.py \
      --input "/media/phips/Crucial X9/lucid_cc0" \
      --threshold 0.94 \
      --batch 256 \
      --dry-run
"""

import argparse
import gc
import json
import os
import pickle
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import psutil
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

# Try to import FAISS, provide helpful error if missing
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print(
        "ERROR: FAISS not installed. Install with: pip install faiss-gpu-cu11torch (or faiss-cpu)"
    )
    sys.exit(1)

Image.MAX_IMAGE_PIXELS = None

# Global state for signal handling
checkpoint_state = {"interrupted": False}


def signal_handler(signum, frame) -> None:
    checkpoint_state["interrupted"] = True
    print("\n[!] Interrupt received, will save checkpoint before exit...")


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


class ImageDataset(Dataset):
    """Lightweight dataset that loads and preprocesses images on-the-fly."""

    def __init__(self, image_paths: list[str], preprocess: transforms.Compose) -> None:
        self.image_paths = image_paths
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor | None:
        path = self.image_paths[idx]
        try:
            with Image.open(path) as img:
                # Skip massive images
                w, h = img.size
                if w > 4096 or h > 4096:
                    return None
                img_rgb = img.convert("RGB")
                return self.preprocess(img_rgb)
        except Exception:
            return None


def collate_fn(batch: list) -> tuple[torch.Tensor, list[int]]:
    """Custom collate to skip None items and track valid indices."""
    valid_items = [(i, tensor) for i, tensor in enumerate(batch) if tensor is not None]
    if not valid_items:
        # Return empty tensor with correct dtype to avoid pin_memory issues
        return torch.empty(0, dtype=torch.float32), []
    indices, tensors = zip(*valid_items, strict=False)
    return torch.stack(tensors), list(indices)


def get_feature_extractor(device: str) -> nn.Module:
    """ResNet18 without classification head, outputs 512-dim features."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model = nn.Sequential(*list(model.children())[:-1])
    model.to(device).eval()
    return model


def extract_features(
    image_paths: list[str],
    cache_dir: Path,
    batch_size: int,
    num_workers: int,
    device: str,
    checkpoint_interval: int,
    resume: bool = True,
    io_sleep: float = 0.01,
) -> np.ndarray:
    """
    Phase 1: Extract features with DataLoader, cache to disk in fp16.
    Returns memory-mapped array of shape (N, 512).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    features_path = cache_dir / "features_fp16.npy"
    index_path = cache_dir / "extraction_index.pkl"

    # Check for existing cache and verify integrity
    processed_count = 0
    processed_paths = set()
    cache_valid = False

    if resume and features_path.exists() and index_path.exists():
        try:
            with open(index_path, "rb") as f:
                index_data = pickle.load(f)
                processed_paths = set(index_data.get("processed_paths", []))
                # Verify cache integrity
                if len(processed_paths) > 0:
                    # Try to load the mmap file
                    try:
                        mmap = np.load(features_path, mmap_mode="r")
                        if (
                            mmap.shape[0] == len(processed_paths)
                            and mmap.shape[1] == 512
                        ):
                            print(
                                f"[+] Resuming from checkpoint: {len(processed_paths)}/{len(image_paths)} images processed"
                            )
                            # Need to process remaining images
                            remaining_paths = [
                                p for p in image_paths if p not in processed_paths
                            ]
                            if remaining_paths:
                                image_paths = remaining_paths
                                processed_count = len(processed_paths)
                                del mmap
                                cache_valid = True
                            else:
                                print("[+] All images already processed!")
                                return mmap
                        else:
                            print("[!] Cache size mismatch, starting fresh...")
                            processed_paths = set()
                    except Exception as e:
                        print(f"[!] Cache file corrupted ({e}), starting fresh...")
                        processed_paths = set()
                        # Remove corrupted files
                        features_path.unlink(missing_ok=True)
                        index_path.unlink(missing_ok=True)
        except Exception as e:
            print(f"[!] Index file corrupted ({e}), starting fresh...")
            processed_paths = set()
            # Remove corrupted files
            features_path.unlink(missing_ok=True)
            index_path.unlink(missing_ok=True)

    if not image_paths:
        print("[!] No remaining images to process.")
        # Create empty array if nothing to do
        return np.empty((0, 512), dtype=np.float16)

    preprocess = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = ImageDataset(image_paths, preprocess)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,  # Disable pin_memory to avoid CUDA context issues
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=collate_fn,
    )

    model = get_feature_extractor(device)

    # Prepare memory-mapped file for appending
    total_features = len(image_paths) + processed_count
    feature_dim = 512

    # Create or extend the mmap file
    if not features_path.exists() or not cache_valid:
        # Create empty mmap
        mmap = np.memmap(
            features_path,
            dtype=np.float16,
            mode="w+",
            shape=(total_features, feature_dim),
        )
        if processed_count > 0 and cache_valid:
            # Load existing data and copy it
            old_mmap = np.load(
                features_path.with_suffix(".npy.old")
                if features_path.with_suffix(".npy.old").exists()
                else features_path,
                mmap_mode="r",
            )
            if old_mmap.shape[0] <= total_features:
                mmap[: old_mmap.shape[0]] = old_mmap[:]
            del old_mmap
    else:
        # Extend existing mmap - we need to create a new file
        # Load existing data
        old_mmap = np.load(features_path, mmap_mode="r")
        old_features = np.array(old_mmap)  # Copy to memory
        del old_mmap

        # Remove old file and create new one
        features_path.unlink()

        # Create new mmap with extended size
        mmap = np.memmap(
            features_path,
            dtype=np.float16,
            mode="w+",
            shape=(total_features, feature_dim),
        )
        # Copy old data
        mmap[:processed_count] = old_features
        del old_features

    current_offset = processed_count
    processed_paths_list = list(processed_paths) if processed_count > 0 else []

    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (batch_tensors, valid_indices) in enumerate(
            tqdm(dataloader, desc="Extracting Features")
        ):
            if checkpoint_state["interrupted"]:
                break

            if batch_tensors.size(0) == 0:
                continue

            # Move to GPU
            batch_tensors = batch_tensors.to(device, non_blocking=True)

            # Extract features
            features = model(batch_tensors).squeeze(-1).squeeze(-1)  # [B, 512]
            features = nn.functional.normalize(features, p=2, dim=1)

            # Convert to fp16 and save to mmap
            features_np = features.cpu().numpy().astype(np.float16)
            for i in range(features_np.shape[0]):
                mmap[current_offset + i] = features_np[i]
                # Track corresponding path
                orig_idx = valid_indices[i]
                processed_paths_list.append(dataset.image_paths[orig_idx])

            current_offset += features_np.shape[0]

            # Periodic checkpoint save (includes mmap flush)
            if (batch_idx + 1) % checkpoint_interval == 0:
                mmap.flush()  # Only flush on checkpoint, not every batch
                checkpoint_data = {
                    "processed_paths": processed_paths_list,
                    "total_processed": current_offset,
                    "timestamp": time.time(),
                }
                with open(index_path, "wb") as f:
                    pickle.dump(checkpoint_data, f)
                print(
                    f"\n[Checkpoint] Saved: {current_offset}/{total_features} features"
                )

            # Gentle sleep to reduce USB SSD load and keep system responsive
            time.sleep(io_sleep)

            # Aggressive cleanup every N batches (less frequent for performance)
            if batch_idx % 1000 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

    # Final checkpoint
    checkpoint_data = {
        "processed_paths": processed_paths_list,
        "total_processed": current_offset,
        "timestamp": time.time(),
    }
    with open(index_path, "wb") as f:
        pickle.dump(checkpoint_data, f)

    mmap.flush()
    del mmap

    elapsed = time.time() - start_time
    print(
        f"[+] Feature extraction complete: {current_offset} features in {elapsed / 60:.1f} min"
    )

    # Reload as read-only mmap
    return np.load(features_path, mmap_mode="r")


def build_faiss_index(
    features: np.ndarray,
    threshold: float,
    use_gpu: bool = True,
    nlist: int = 1000,
    nprobe: int = 10,
) -> faiss.Index:
    """
    Build FAISS IVF index for approximate nearest neighbor search.
    Uses L2 distance (cosine similarity on normalized vectors = 0.5 * L2²).
    """
    d = features.shape[1]  # 512
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

    # Convert to float32 for FAISS
    features_f32 = features.astype(np.float32)

    print("[+] Training FAISS index...")
    index.train(features_f32)

    print("[+] Adding features to index...")
    index.add(features_f32)

    # Move to GPU if available and requested
    if use_gpu and torch.cuda.is_available():
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            print("[+] Using GPU-accelerated FAISS")
        except Exception as e:
            print(f"[!] GPU FAISS failed: {e}, using CPU")

    index.nprobe = nprobe
    print(f"[+] FAISS index ready: {index.ntotal} vectors, nprobe={nprobe}")
    return index


def find_redundant_faiss(
    features: np.ndarray,
    index: faiss.Index,
    threshold: float,
    batch_size: int,
    checkpoint_interval: int,
    checkpoint_path: Path,
) -> list[int]:
    """
    Phase 2: Find redundant images using FAISS search.
    Returns list of indices to mark as redundant.
    """
    redundant_indices = []
    total = len(features)

    # Load checkpoint if exists
    if checkpoint_path.exists():
        with open(checkpoint_path, "rb") as f:
            ckpt = pickle.load(f)
            redundant_indices = ckpt.get("redundant_indices", [])
            last_processed = ckpt.get("last_processed", 0)
            print(f"[+] Resuming deduplication from index {last_processed}")
            start_idx = last_processed
    else:
        start_idx = 0

    features_f32 = features.astype(np.float32)

    # For IVF index, we need to search in batches to avoid memory explosion
    for i in tqdm(range(start_idx, total, batch_size), desc="Deduplication"):
        if checkpoint_state["interrupted"]:
            break

        batch_end = min(i + batch_size, total)
        batch_features = features_f32[i:batch_end]

        # Search against ALL features (including self)
        # k=2: first neighbor is self, second is the nearest other
        distances, _indices = index.search(batch_features, k=2)

        # distances is squared L2. Convert to cosine similarity:
        # cos_sim = 1 - 0.5 * L2² (since vectors are normalized)
        l2_sq = distances[:, 1]  # Skip self (index 0)
        cosine_sim = 1.0 - 0.5 * l2_sq

        # Mark as redundant if similarity >= threshold
        for j, sim in enumerate(cosine_sim):
            if sim >= threshold:
                redundant_indices.append(i + j)

        # Periodic checkpoint
        if (i // batch_size + 1) % checkpoint_interval == 0:
            ckpt_data = {
                "redundant_indices": redundant_indices,
                "last_processed": batch_end,
                "timestamp": time.time(),
            }
            with open(checkpoint_path, "wb") as f:
                pickle.dump(ckpt_data, f)
            print(
                f"\n[Checkpoint] Processed {batch_end}/{total}, found {len(redundant_indices)} redundant"
            )

        # Gentle sleep
        time.sleep(0.001)

    # Final checkpoint
    ckpt_data = {
        "redundant_indices": redundant_indices,
        "last_processed": total,
        "timestamp": time.time(),
    }
    with open(checkpoint_path, "wb") as f:
        pickle.dump(ckpt_data, f)

    return redundant_indices


def main() -> None:
    try:
        os.nice(15)
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="LUCID v2 - Step 02: Dedupe (FAST)")
    parser.add_argument("--input", required=True, help="Input folder of tiles")
    parser.add_argument(
        "--threshold", type=float, default=0.94, help="Similarity threshold (0.90-0.98)"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=64,
        help="Batch size for extraction (default: 64, smaller is better for USB SSD)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="DataLoader workers (default: 2, use 0 for sequential)",
    )
    parser.add_argument(
        "--nlist", type=int, default=1000, help="FAISS IVF clusters (default: 1000)"
    )
    parser.add_argument(
        "--nprobe", type=int, default=10, help="FAISS nprobe (default: 10)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Only report, don't move/delete files"
    )
    parser.add_argument(
        "--delete", action="store_true", help="Delete redundant files instead of moving"
    )
    parser.add_argument(
        "--cache",
        type=str,
        default="/home/phips/.cache/lucid_dedupe",
        help="Cache directory for features (default: /home/phips/.cache/lucid_dedupe)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=500,
        help="Save checkpoint every N batches (default: 500)",
    )
    parser.add_argument(
        "--max-memory-gb",
        type=float,
        default=0.8,
        help="Max system memory fraction to use (default: 0.8)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous checkpoint (auto-enabled if cache exists)",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear cache directory and start fresh (useful if previous run was interrupted)",
    )
    parser.add_argument(
        "--io-sleep",
        type=float,
        default=0.01,
        help="Sleep seconds between batches to reduce USB SSD load (default: 0.01)",
    )
    parser.add_argument(
        "--no-mmap",
        action="store_true",
        help="Use regular numpy arrays instead of memory-mapped files (faster on NVMe)",
    )

    args = parser.parse_args()
    input_dir = Path(args.input)
    cache_dir = Path(args.cache)

    # Clear cache if requested
    if args.clear_cache and cache_dir.exists():
        print(f"[+] Clearing cache directory: {cache_dir}")
        shutil.rmtree(cache_dir)

    print("--- LUCID v2 Dedupe (FAST v3) ---")
    print(f"Master Pool: {input_dir}")
    print(f"Threshold: {args.threshold}")
    print(f"Cache: {cache_dir}")
    print(f"FAISS: nlist={args.nlist}, nprobe={args.nprobe}")
    print(f"Dry-run: {args.dry_run}")

    # Scan for images
    print("Scanning for images...")
    image_paths = [str(p) for p in sorted(input_dir.rglob("*.png"))]
    if not image_paths:
        print("No images found.")
        return
    print(f"Found {len(image_paths)} tiles.")

    # Memory check
    total_ram_gb = psutil.virtual_memory().total / (1024**3)
    if total_ram_gb < 8:
        print(
            f"[!] Warning: Only {total_ram_gb:.1f}GB RAM detected. Consider reducing --workers."
        )

    # USB SSD warning
    if args.workers > 2:
        print(
            f"[!] Warning: Using {args.workers} workers with USB SSD may cause system freezes."
        )
        print(
            "    Consider: --workers 0 (sequential) or --workers 2 (parallel with throttling)"
        )

    # Phase 1: Feature Extraction
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Using device: {device}")

    # Try GPU first, fallback to CPU if CUDA fails
    try:
        features = extract_features(
            image_paths=image_paths,
            cache_dir=cache_dir,
            batch_size=args.batch,
            num_workers=min(args.workers, os.cpu_count() or 4),
            device=device,
            checkpoint_interval=args.checkpoint_every,
            resume=args.resume,
            io_sleep=args.io_sleep,
        )
    except Exception as e:
        if "CUDA" in str(e) or "pin_memory" in str(e):
            print(f"[!] CUDA error detected: {e}")
            print("[!] Falling back to CPU mode...")
            device = "cpu"
            features = extract_features(
                image_paths=image_paths,
                cache_dir=cache_dir,
                batch_size=args.batch,
                num_workers=min(args.workers, os.cpu_count() or 4),
                device=device,
                checkpoint_interval=args.checkpoint_every,
                resume=args.resume,
                io_sleep=args.io_sleep,
            )
        else:
            raise

    # Phase 2: FAISS Deduplication
    print("\n[+] Building FAISS index...")
    index = build_faiss_index(
        features=features,
        threshold=args.threshold,
        use_gpu=(device == "cuda"),
        nlist=min(
            args.nlist, len(features) // 100
        ),  # Ensure enough vectors per cluster
        nprobe=args.nprobe,
    )

    checkpoint_path = cache_dir / "dedupe_checkpoint.pkl"
    print("[+] Starting deduplication search...")
    redundant_indices = find_redundant_faiss(
        features=features,
        index=index,
        threshold=args.threshold,
        batch_size=args.batch,
        checkpoint_interval=args.checkpoint_every,
        checkpoint_path=checkpoint_path,
    )

    redundant_count = len(redundant_indices)
    unique_count = len(image_paths) - redundant_count
    print("\n=== Results ===")
    print(f"Total images: {len(image_paths)}")
    print(f"Redundant: {redundant_count}")
    print(f"Unique remaining: {unique_count}")
    print(f"Redundancy rate: {redundant_count / len(image_paths) * 100:.1f}%")

    # Action
    if redundant_count > 0:
        redundant_paths = [image_paths[i] for i in sorted(redundant_indices)]

        if args.dry_run:
            # Save list to file for later use
            delete_list_path = cache_dir / "delete_list.txt"
            with open(delete_list_path, "w") as f:
                for p in redundant_paths:
                    f.write(f"{p}\n")
            print(f"\n[Dry-run] Found {len(redundant_paths)} redundant files")
            print(f"[+] Saved delete list to: {delete_list_path}")
            print("[+] To actually delete, run without --dry-run or use:")
            print(f"    cat {delete_list_path} | xargs rm")
            print("\nFirst 10 files that would be deleted:")
            for p in redundant_paths[:10]:
                print(f"  {p}")
            if len(redundant_paths) > 10:
                print(f"  ... and {len(redundant_paths) - 10} more")
        else:
            redundant_dir = input_dir / "redundant"
            if args.delete:
                print(f"\nDeleting {redundant_count} redundant tiles...")
                for p in tqdm(redundant_paths, desc="Deleting"):
                    try:
                        Path(p).unlink()
                    except Exception:
                        pass
            else:
                redundant_dir.mkdir(exist_ok=True)
                print(
                    f"\nMoving {redundant_count} redundant tiles to {redundant_dir}..."
                )
                for p in tqdm(redundant_paths, desc="Moving"):
                    try:
                        dest = redundant_dir / Path(p).name
                        if dest.exists():
                            dest = (
                                redundant_dir
                                / f"{Path(p).stem}_{int(time.time() * 1000)}{Path(p).suffix}"
                            )
                        shutil.move(str(p), str(dest))
                    except Exception:
                        pass
            print("[+] Deduplication complete!")
    else:
        print("\nNo redundant textures found. Dataset is highly diverse!")

    # Cleanup checkpoints on success
    if not checkpoint_state["interrupted"] and not args.dry_run:
        for ckpt in [cache_dir / "extraction_index.pkl", checkpoint_path]:
            if ckpt.exists():
                ckpt.unlink()

    print(f"\nFinal unique tiles: {unique_count}")


if __name__ == "__main__":
    main()
