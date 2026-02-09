#!/usr/bin/env python3
"""
LUCID Stage 3: The Density Gate (Complexity Selection)
======================================================
Author: Philip Hofmann
Strategy: "Complexity-Weighted Purity"

Description:
This script performs the final selection of the Elite training set.
Unlike Stage 2 (which selects for consistency/predictability), this stage
selects for Information Density using ICNet (Image Complexity Network).

It takes the "Safe Pool" from Stage 2 (technically clean, physically valid)
and selects the Top X% of tiles with the highest complexity scores.

This ensures the final dataset is:
1. Technically Clean (Stage 1)
2. Physically Valid (Stage 2)
3. Informatically Dense (Stage 3) -> Solves the "Urban100 Gap"
"""

import argparse
import csv
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Import ICNet from the local directory structure
# Assumes this script is in scripts/lucid_filtering/
# and ICNet is in datasets/preparation/complexity/
REPO_ROOT = Path(__file__).parent.parent.parent
ICNET_DIR = REPO_ROOT / "datasets" / "preparation" / "complexity"
sys.path.append(str(ICNET_DIR))

try:
    from ICNet import ICNet
except ImportError:
    print(f"FATAL: Could not import ICNet from {ICNET_DIR}")
    print("Please check your directory structure.")
    sys.exit(1)


# ====================== CONFIG ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64  # Safe for 12GB VRAM with ICNet
DEFAULT_THRESHOLD_PERCENTILE = 25.0  # Top 25% by default


def load_icnet(model_path):
    print(f"Loading ICNet from {model_path}...")
    model = ICNet(
        is_pretrain=False
    )  # No need to download resnet weights if loading full state
    # We need to handle the case where the saved model might be a full model or just state_dict
    try:
        # Try loading as state dict first (safest)
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
    except Exception:
        # Fallback: maybe it's a full pickle (less safe but common)
        try:
            model = torch.load(model_path, map_location=DEVICE)
        except Exception as e:
            print(f"Failed to load model: {e}")
            sys.exit(1)

    model.to(DEVICE)
    model.eval()

    # FP16 Optimization
    if DEVICE == "cuda":
        model.half()
        print("Enabled FP16 for ICNet")

    return model


def compute_complexity(model, image_paths, csv_path=None):
    """
    Computes complexity scores for a list of paths.
    Returns a dict: {filename: score}
    """
    results = {}

    # Load existing scores if available
    if csv_path:
        csv_path = Path(csv_path)
        if csv_path.exists():
            print(f"Loading cached scores from {csv_path}...")
            try:
                with open(csv_path) as f:
                    reader = csv.reader(f)
                    next(reader, None)  # Skip header
                    for row in reader:
                        if len(row) >= 2:
                            results[row[0]] = float(row[1])
            except Exception as e:
                print(f"Warning: Could not read cache: {e}")

    # Identify missing files
    needed_paths = [p for p in image_paths if p.name not in results]
    print(f"Cached: {len(results)} | To Compute: {len(needed_paths)}")

    if not needed_paths:
        return results

    # Prepare CSV for appending
    f_handle = None
    writer = None
    if csv_path:
        try:
            file_exists = csv_path.exists()
            f_handle = open(csv_path, "a", newline="")
            writer = csv.writer(f_handle)
            if not file_exists:
                writer.writerow(["tile_name", "icnet_score"])
        except Exception as e:
            print(f"Warning: Could not open CSV for writing: {e}")

    # Pre-allocate batch tensors
    batch_tensors = []
    batch_names = []

    try:
        for i, path in enumerate(tqdm(needed_paths, desc="Scoring Complexity")):
            try:
                # Read Image
                img = cv2.imread(str(path))
                if img is None:
                    continue

                # BGR -> RGB & Resize to 512x512 (ICNet native resolution)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)

                # Normalize (Standard ImageNet mean/std)
                img = img.astype(np.float32) / 255.0
                img = (img - np.array([0.485, 0.456, 0.406])) / np.array(
                    [0.229, 0.224, 0.225]
                )

                # CHW
                img = np.transpose(img, (2, 0, 1))

                batch_tensors.append(torch.from_numpy(img))
                batch_names.append(path.name)

                # Process Batch
                if len(batch_tensors) >= BATCH_SIZE or i == len(needed_paths) - 1:
                    if not batch_tensors:
                        continue

                    batch_stack = torch.stack(batch_tensors).to(DEVICE)
                    if DEVICE == "cuda":
                        batch_stack = batch_stack.half()

                    with torch.no_grad():
                        # ICNet returns (score, map) -> we only need score
                        scores, _ = model(batch_stack)
                        scores = scores.float().cpu().numpy()

                        # Handle single-item batch (scalar) vs multi-item (array)
                        if scores.ndim == 0:
                            scores = [float(scores)]
                        else:
                            scores = scores.flatten().tolist()

                    # Store results
                    for name, score in zip(batch_names, scores, strict=False):
                        results[name] = score
                        if writer:
                            writer.writerow([name, f"{score:.6f}"])

                    if f_handle:
                        f_handle.flush()

                    # Reset
                    batch_tensors = []
                    batch_names = []

            except KeyboardInterrupt:
                print("\n\n!! Interrupted by user. Saving progress and exiting...")
                break
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue
    finally:
        if f_handle:
            f_handle.close()

    return results


def main() -> None:
    # Lower process priority to keep system responsive
    try:
        os.nice(15)
        print("System Responsiveness Mode: Priority lowered to 15.")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="LUCID Stage 3: The Density Gate")
    parser.add_argument("--input", required=True, help="Input folder (Stage 2 Output)")
    parser.add_argument(
        "--output", required=True, help="Output folder (Final Elite Set)"
    )
    parser.add_argument("--icnet", required=True, help="Path to complexity.pth")
    parser.add_argument(
        "--top_percent", type=float, default=25.0, help="Top % to keep (default: 25)"
    )
    parser.add_argument(
        "--csv", default="lucid_stage3_density.csv", help="Output CSV log"
    )
    parser.add_argument(
        "--symlink", action="store_true", help="Use symlinks instead of copy"
    )
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Gather files
    print(f"Scanning {in_dir}...")
    valid_exts = {".png", ".jpg", ".jpeg", ".webp"}
    image_paths = sorted(
        [p for p in in_dir.glob("*") if p.suffix.lower() in valid_exts]
    )

    if not image_paths:
        print("No images found!")
        sys.exit(1)

    print(f"Found {len(image_paths)} candidates from Stage 2.")

    # 2. Load Model
    model = load_icnet(args.icnet)

    # 3. Compute Scores
    print("Computing ICNet Complexity Scores...")
    # Pass the CSV path to enable partial resumption
    scores_dict = compute_complexity(model, image_paths, csv_path=Path(args.csv))

    if not scores_dict:
        print("Scoring failed (no results returned).")
        sys.exit(1)

    # 4. Sort and Threshold
    sorted_items = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)

    keep_count = int(len(sorted_items) * (args.top_percent / 100.0))
    elite_set = sorted_items[:keep_count]

    print("\nSelection Results:")
    print(f"Total Candidates: {len(sorted_items)}")
    print(f"Target Selection: Top {args.top_percent}% ({keep_count} tiles)")
    print(f"Max Complexity:   {sorted_items[0][1]:.4f}")
    if len(sorted_items) > 0:
        print(f"Min Complexity:   {sorted_items[-1][1]:.4f}")
    if len(elite_set) > 0:
        print(f"Cutoff Score:     {elite_set[-1][1]:.4f}")

    # 5. Export
    print(f"\nExporting to {out_dir}...")

    # We open a NEW csv for the final selection manifest (or reuse the same logic if preferred,
    # but typically args.csv is the raw scoring log, and we might want a manifest of what was *selected*).
    # The user asked for the raw log to allow resumption, which we handled.
    # Below is for the actual copying.

    # Let's write a 'manifest' CSV? Or just rely on the folder content?
    # The user asked for resumption, so args.csv is handled.
    # Let's just do the copying.

    for name, _score in tqdm(elite_set, desc="Exporting Elite"):
        src = in_dir / name
        dst = out_dir / name

        if args.symlink:
            if dst.exists():
                dst.unlink()
            os.symlink(src, dst)
        elif not dst.exists():  # Don't overwrite if exists to save time? Or overwrite?
            shutil.copy2(src, dst)

    print("\nStage 3 Complete!")
    print(f"Elite Dataset Size: {len(elite_set)}")
    print(f"Saved to: {out_dir}")
    print(f"Scoring log saved to: {args.csv}")


if __name__ == "__main__":
    main()
