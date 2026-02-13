#!/usr/bin/env python3
"""
LUCID Stage 2: Complexity Scoring (High-Performance)
==================================================
Author: Philip Hofmann

Description:
Optimized version using multi-threaded DataLoader to saturate the GPU.
This can be up to 4x-6x faster than the sequential version.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

# Fix for "Python stopped working" notifications on Linux:
# Limit background threads within workers to prevent resource deadlock
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp  # Use torch.multiprocessing for spawn safety
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Import ICNet
REPO_ROOT = Path(__file__).parent.parent.parent
ICNET_DIR = REPO_ROOT / "datasets" / "preparation" / "complexity"
sys.path.append(str(ICNET_DIR))

try:
    from ICNet import ICNet
except ImportError:
    print(f"FATAL: Could not import ICNet from {ICNET_DIR}")
    sys.exit(1)

# ====================== CONFIG ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128  # Balanced for high throughput and stability
NUM_WORKERS = 4  # Fewer, higher-priority workers to avoid I/O thrashing

# Fixed Sharing Strategy for high-speed IPC
if sys.platform != "win32":
    try:
        import torch.multiprocessing as mp

        # Use file_descriptor for speed (default), increase ulimit manually if needed
        mp.set_sharing_strategy("file_descriptor")
    except Exception:
        pass


class ICNetDataset(Dataset):
    def __init__(self, image_paths: list[Path]) -> None:
        self.image_paths = image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor | None, str]:
        path = self.image_paths[idx]
        try:
            img = cv2.imread(str(path))
            if img is None:
                return None, str(path)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize early to 512x512 for batch stacking stability.
            # This allows us to handle datasets with mixed resolutions (like diverseg-ip).
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)

            # Return as uint8 (HWC) to save bandwidth
            return torch.from_numpy(img), str(path)
        except Exception:
            return None, str(path)


def safe_collate(batch: list) -> tuple[torch.Tensor, list[str]] | None:
    """Efficiently stacks tensors and filters fails in one pass."""
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        return None

    tensors = torch.stack([item[0] for item in batch])
    paths = [item[1] for item in batch]
    return tensors, paths


def load_icnet(model_path: Path) -> torch.nn.Module:
    print(f"Loading ICNet from {model_path}...")
    model = ICNet(is_pretrain=False)
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    if DEVICE == "cuda":
        model.half()
        print("Enabled FP16 for ICNet")
    return model


def score_images(
    model: torch.nn.Module, image_paths: list[Path], csv_path: Path
) -> dict[str, float]:
    """
    Scores all images and writes results to CSV incrementally.
    Returns dict of {filename: score} for summary stats.
    """
    results = {}

    # Load existing scores if resuming
    if csv_path.exists():
        print("Found existing CSV, loading cached scores...")
        try:
            with open(csv_path) as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header
                for row in reader:
                    if len(row) >= 2:
                        results[row[0]] = float(row[1])
            print(f"Loaded {len(results)} cached scores.")
        except Exception as e:
            print(f"Warning: Could not read cache: {e}")

    # Identify remaining files
    needed_paths = [p for p in image_paths if str(p) not in results]
    print(f"Cached: {len(results)} | To Compute: {len(needed_paths)}")

    if not needed_paths:
        return results

    # Setup DataLoader
    dataset = ICNetDataset(needed_paths)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,  # Set to False to prevent potential hangs with 'spawn' and low priority
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2,
        collate_fn=safe_collate,  # Use top-level function for pickling safety
    )

    # Open CSV for appending
    file_exists = csv_path.exists()
    f_handle = open(csv_path, "a", newline="")
    writer = csv.writer(f_handle)
    if not file_exists:
        writer.writerow(["image_path", "complexity_score"])

    # Prepare normalization constants on GPU
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1).half()
    std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1).half()

    try:
        for batch_data in tqdm(dataloader, desc="Scoring (Turbo)"):
            if batch_data is None:
                continue

            tensors, paths = batch_data
            # Move to GPU immediately
            batch_stack = tensors.to(DEVICE, non_blocking=True)

            # 2. Resizing is now handled in the DataLoader for multi-res stability.
            # We keep the permute and half cast for GPU performance.
            batch_stack = batch_stack.permute(0, 3, 1, 2).half()

            # 3. Normalize
            batch_stack = batch_stack / 255.0
            batch_stack = (batch_stack - mean) / std

            with torch.no_grad():
                scores, _ = model(batch_stack)
                scores = scores.float().cpu().numpy()

                if scores.ndim == 0:
                    scores = [float(scores)]
                else:
                    scores = scores.flatten().tolist()

            # Write to CSV and store in results
            for path_str, score in zip(paths, scores, strict=False):
                results[path_str] = score
                writer.writerow([path_str, f"{score:.6f}"])

            if len(results) % 1000 == 0:
                f_handle.flush()

    except (KeyboardInterrupt, SystemExit):
        print("\n\n!! Interrupted by user. Saving progress and exiting...")
        # Hard shutdown of dataloader to prevent messy multi-thread hangs
        try:
            del dataloader
        except Exception:
            pass
    finally:
        f_handle.flush()
        f_handle.close()

    return results


def print_stats(results: dict[str, float]) -> None:
    if not results:
        print("No results to analyze.")
        return

    scores = list(results.values())
    scores.sort(reverse=True)

    print("\n" + "=" * 50)
    print("COMPLEXITY SCORE STATISTICS")
    print("=" * 50)
    print(f"Total Images: {len(scores)}")
    print(f"Mean Score:   {np.mean(scores):.4f}")
    print(f"Median Score: {np.median(scores):.4f}")
    print(f"Max Score:    {np.max(scores):.4f}")
    print(f"Min Score:    {np.min(scores):.4f}")

    print("\n--- Suggested Thresholds ---")
    for pct in [10, 20, 25, 30, 50]:
        idx = int(len(scores) * (pct / 100.0))
        if idx > 0:
            threshold = scores[idx - 1]
            print(f"Top {pct:2d}% ({idx:>7d} images): score >= {threshold:.4f}")

    print("\n--- Score Distribution ---")
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(scores, bins)
    for i in range(len(bins) - 1):
        pct = (hist[i] / len(scores)) * 100
        bar = "█" * int(pct / 2)
        print(f"{bins[i]:.1f}-{bins[i + 1]:.1f}: {pct:5.1f}% {bar}")


def main() -> None:
    # Lower process priority immediately to keep system responsive (affects children too)
    try:
        # Use 'spawn' instead of 'fork' to prevent library conflicts (cv2/OpenMP)
        # that lead to "Python stopped working" notifications.
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method("spawn", force=True)

        if DEVICE == "cuda":
            torch.backends.cudnn.benchmark = True

        os.nice(15)  # Even lower priority (0 is normal, 19 is lowest)
        print(
            "System Responsiveness Mode: Priority lowered to 15. Start method: spawn."
        )
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Score images with ICNet (TURBO).")
    parser.add_argument("--input", required=True)
    parser.add_argument("--icnet", required=True)
    parser.add_argument("--csv", default="complexity_scores.csv")
    args = parser.parse_args()

    in_dir = Path(args.input)
    csv_path = Path(args.csv)

    print(f"Scanning {in_dir} (Fast Scan)...")
    valid_exts = {".png", ".jpg", ".jpeg", ".webp"}
    # Efficient generator-based scan to handle millions of files without RAM bloat
    image_paths = []
    for root, _, files in os.walk(in_dir):
        for f in files:
            if any(f.lower().endswith(ext) for ext in valid_exts):
                image_paths.append(Path(root) / f)

    if not image_paths:
        print("No images found!")
        sys.exit(1)

    model = load_icnet(Path(args.icnet))
    results = score_images(model, image_paths, csv_path)
    print_stats(results)


if __name__ == "__main__":
    main()
