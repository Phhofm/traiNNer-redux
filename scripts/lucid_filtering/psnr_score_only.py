#!/usr/bin/env python3
"""
PSNR Consistency Scorer (Scoring Only)
======================================
Author: Philip Hofmann

Description:
This script scores tiles using the "Stage 2" PSNR consistency metric.
It measures how well a tile's SR reconstruction matches the original HR.

Formula: PSNR(SR(Bicubic(HR)), HR)

High PSNR = "Easy" tile (model can perfectly reconstruct it)
Low PSNR  = "Hard" tile (could be complex detail OR corruption)

Use this to manually inspect whether low-PSNR tiles are:
- Legitimately complex (good for training)
- Actually corrupted/noisy (bad for training)

This script does NOT filter. It only scores and outputs a CSV.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F_torch
from tqdm import tqdm

# ====================== CONFIG ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCALE = 4  # Default SR scale


def calculate_psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 1.0) -> float:
    """Calculate PSNR between two images."""
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return 100.0  # Perfect match
    return float(10 * np.log10((max_val**2) / mse))


def bicubic_downsample(img_tensor: torch.Tensor, scale: int) -> torch.Tensor:
    """Downsample using bicubic interpolation."""
    h, w = img_tensor.shape[2:]
    new_h, new_w = h // scale, w // scale
    return F_torch.interpolate(
        img_tensor, size=(new_h, new_w), mode="bicubic", align_corners=False
    )


def bicubic_upsample(img_tensor: torch.Tensor, scale: int) -> torch.Tensor:
    """Upsample using bicubic interpolation."""
    h, w = img_tensor.shape[2:]
    new_h, new_w = h * scale, w * scale
    return F_torch.interpolate(
        img_tensor, size=(new_h, new_w), mode="bicubic", align_corners=False
    )


def score_images(image_paths, csv_path, scale=4):
    """
    Scores all images for PSNR consistency and writes results to CSV.
    Returns dict of {filename: psnr} for summary stats.
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
    needed_paths = [p for p in image_paths if p.name not in results]
    print(f"Cached: {len(results)} | To Compute: {len(needed_paths)}")

    if not needed_paths:
        return results

    # Open CSV for appending
    file_exists = csv_path.exists()
    f_handle = open(csv_path, "a", newline="")
    writer = csv.writer(f_handle)
    if not file_exists:
        writer.writerow(["image_path", "psnr_score"])

    try:
        for path in tqdm(needed_paths, desc="Scoring PSNR"):
            try:
                # Read image
                img = cv2.imread(str(path))
                if img is None:
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, _ = img.shape

                # Ensure dimensions are divisible by scale
                h = (h // scale) * scale
                w = (w // scale) * scale
                img = img[:h, :w]

                # Convert to tensor
                img_tensor = torch.from_numpy(img).float() / 255.0
                img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

                if DEVICE == "cuda":
                    img_tensor = img_tensor.cuda()

                # Simulate SR pipeline: HR -> LR -> SR
                with torch.no_grad():
                    lr = bicubic_downsample(img_tensor, scale)
                    sr = bicubic_upsample(lr, scale)

                # Calculate PSNR
                hr_np = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
                sr_np = sr.squeeze().permute(1, 2, 0).cpu().numpy()
                sr_np = np.clip(sr_np, 0, 1)

                psnr = calculate_psnr(hr_np, sr_np, max_val=1.0)

                # Store and write
                results[path.name] = psnr
                writer.writerow([str(path), f"{psnr:.4f}"])

                # Periodic flush
                if len(results) % 1000 == 0:
                    f_handle.flush()

            except KeyboardInterrupt:
                print("\n\n!! Interrupted by user. Saving progress and exiting...")
                break
            except Exception as e:
                print(f"Error: {path}: {e}")
                continue

    finally:
        f_handle.close()

    return results


def print_stats(results: dict[str, float]) -> None:
    """Print distribution statistics for the scored dataset."""
    if not results:
        print("No results to analyze.")
        return

    scores = list(results.values())
    scores.sort()  # Low to high (low = potentially problematic)

    print("\n" + "=" * 50)
    print("PSNR CONSISTENCY STATISTICS")
    print("=" * 50)
    print(f"Total Images Scored: {len(scores)}")
    print(f"Mean PSNR:   {np.mean(scores):.2f} dB")
    print(f"Median PSNR: {np.median(scores):.2f} dB")
    print(f"Max PSNR:    {np.max(scores):.2f} dB (easiest)")
    print(f"Min PSNR:    {np.min(scores):.2f} dB (hardest)")
    print(f"Std Dev:     {np.std(scores):.2f} dB")

    # Rejection thresholds (bottom percentiles)
    print("\n--- Rejection Thresholds (Bottom X%) ---")
    for pct in [5, 10, 20, 25, 50]:
        idx = int(len(scores) * (pct / 100.0))
        if idx > 0:
            threshold = scores[idx - 1]
            print(f"Bottom {pct:2d}% ({idx:>7d} images): PSNR < {threshold:.2f} dB")

    # Distribution
    print("\n--- PSNR Distribution ---")
    bins = [0, 18, 20, 22, 24, 26, 28, 30, 35, 40, 100]
    hist, _ = np.histogram(scores, bins)
    for i in range(len(bins) - 1):
        pct = (hist[i] / len(scores)) * 100
        bar = "█" * int(pct / 2)
        label = (
            f"{bins[i]:2d}-{bins[i + 1]:2d}" if bins[i + 1] < 100 else f"{bins[i]:2d}+"
        )
        print(f"{label} dB: {pct:5.1f}% {bar}")

    print("\n--- Interpretation Guide ---")
    print("PSNR < 20 dB: Likely garbage, extreme mismatch, or heavy artifacts")
    print("PSNR 20-25 dB: 'Hard' tiles - could be complex detail OR noise")
    print("PSNR 25-30 dB: Normal range for clean, moderately detailed tiles")
    print("PSNR > 30 dB: 'Easy' tiles - smooth gradients, simple textures")


def main() -> None:
    # Lower process priority to keep system responsive
    try:
        os.nice(15)
        print("System Responsiveness Mode: Priority lowered to 15.")
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="Score images with PSNR consistency metric (no filtering)."
    )
    parser.add_argument("--input", required=True, help="Input folder containing images")
    parser.add_argument(
        "--csv",
        default="psnr_scores.csv",
        help="Output CSV file (default: psnr_scores.csv)",
    )
    parser.add_argument(
        "--scale", type=int, default=4, help="SR scale factor (default: 4)"
    )
    args = parser.parse_args()

    in_dir = Path(args.input)
    csv_path = Path(args.csv)

    # Gather image files
    print(f"Scanning {in_dir}...")
    valid_exts = {".png", ".jpg", ".jpeg", ".webp"}
    image_paths = sorted(
        [p for p in in_dir.rglob("*") if p.suffix.lower() in valid_exts and p.is_file()]
    )

    if not image_paths:
        print("No images found!")
        sys.exit(1)

    print(f"Found {len(image_paths)} images.")
    print(f"Using scale: {args.scale}x")

    # Score all images
    results = score_images(image_paths, csv_path, scale=args.scale)

    # Print statistics
    print_stats(results)

    print(f"\n✅ Scoring complete! Results saved to: {csv_path}")
    print("\n--- Next Steps ---")
    print("1. Sort the CSV by PSNR (ascending) to see the 'worst' tiles first")
    print("2. Visually inspect the low-PSNR tiles to see if they're:")
    print("   - Complex detail (KEEP) or Corrupted garbage (REJECT)")
    print(
        "3. If many low-PSNR tiles are garbage, use copy_by_score.py with --min_score"
    )


if __name__ == "__main__":
    main()
