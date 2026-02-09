#!/usr/bin/env python3
"""
ICNet Complexity Scorer (High-Performance)
=========================================
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

import cv2
import numpy as np
import torch
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
BATCH_SIZE = 128  # Increased for higher throughput
NUM_WORKERS = max(
    1, os.cpu_count() // 2
)  # Conservative: leave half the threads for OS/Remote access


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
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)

            img = img.astype(np.float32) / 255.0
            img = (img - np.array([0.485, 0.456, 0.406])) / np.array(
                [0.229, 0.224, 0.225]
            )
            img = np.transpose(img, (2, 0, 1))
            return torch.from_numpy(img), str(path)
        except Exception:
            return None, str(path)


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
        pin_memory=(DEVICE == "cuda"),
        collate_fn=lambda x: [y for y in x if y[0] is not None],  # Filter failed loads
    )

    # Open CSV for appending
    file_exists = csv_path.exists()
    f_handle = open(csv_path, "a", newline="")
    writer = csv.writer(f_handle)
    if not file_exists:
        writer.writerow(["image_path", "complexity_score"])

    try:
        for batch in tqdm(dataloader, desc="Scoring (Turbo)"):
            if not batch:
                continue

            tensors, paths = zip(*batch, strict=False)
            batch_stack = torch.stack(tensors).to(DEVICE)
            if DEVICE == "cuda":
                batch_stack = batch_stack.half()

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

    except KeyboardInterrupt:
        print("\n\n!! Interrupted by user. Saving progress and exiting...")
    finally:
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
        os.nice(15)  # Even lower priority (0 is normal, 19 is lowest)
        print("System Responsiveness Mode: Priority lowered to 15.")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Score images with ICNet (TURBO).")
    parser.add_argument("--input", required=True)
    parser.add_argument("--icnet", required=True)
    parser.add_argument("--csv", default="complexity_scores.csv")
    args = parser.parse_args()

    in_dir = Path(args.input)
    csv_path = Path(args.csv)

    print(f"Scanning {in_dir}...")
    valid_exts = {".png", ".jpg", ".jpeg", ".webp"}
    image_paths = sorted(
        [p for p in in_dir.rglob("*") if p.suffix.lower() in valid_exts and p.is_file()]
    )

    if not image_paths:
        print("No images found!")
        sys.exit(1)

    model = load_icnet(Path(args.icnet))
    results = score_images(model, image_paths, csv_path)
    print_stats(results)


if __name__ == "__main__":
    main()
