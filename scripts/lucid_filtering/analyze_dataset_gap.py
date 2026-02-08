#!/usr/bin/env python3
"""
LUCID Analysis: The Dataset Gap Probe
=====================================
Author: Philip Hofmann
Description:
Compares the ICNet Complexity Distribution of two datasets:
1. "Diverseg-ip" (The Target/Baseline) - Simulated via random crops.
2. "LUCID-Elite-Complex" (Our Dataset) - Pre-tiled 256x256.

Goal:
Determine if 'Diverseg' has higher average complexity than our 'Elite' set.
If Diverseg >> Elite, then our Stage 2 filter was too strict and killed the complex tiles.
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_SIZE = 2000  # Number of samples per dataset
CROP_SIZE = 256


def load_icnet(model_path):
    print(f"Loading ICNet from {model_path}...")
    model = ICNet(is_pretrain=False)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    if DEVICE == "cuda":
        model.half()
    return model


def get_random_crop(img_path, crop_size=256):
    """Reads image and returns a random 256x256 crop."""
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        h, w, c = img.shape

        # If smaller than crop, resize up (bicubic) just like a dataloader might (or skip)
        # But diverseg images are usually > 256.
        if h < crop_size or w < crop_size:
            # Skip small images for fairness, or resize min dim to crop_size
            scale = crop_size / min(h, w)
            img = cv2.resize(
                img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )
            h, w, _c = img.shape

        y = random.randint(0, h - crop_size)
        x = random.randint(0, w - crop_size)
        crop = img[y : y + crop_size, x : x + crop_size]
        return crop
    except Exception:
        return None


def score_images(model, image_paths, is_tiled=False):
    scores = []

    # Randomly sample if too many
    if len(image_paths) > SAMPLE_SIZE:
        sampled_paths = random.sample(image_paths, SAMPLE_SIZE)
    else:
        sampled_paths = image_paths

    for path in tqdm(sampled_paths, desc="Scoring"):
        if is_tiled:
            # Just read the tile
            img = cv2.imread(str(path))
        else:
            # Perform random crop simulation
            img = get_random_crop(path, CROP_SIZE)

        if img is None:
            continue

        # Prepare for ICNet (Resize to 512x512 for standard inference)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512, 512))

        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        img = np.transpose(img, (2, 0, 1))

        tensor = torch.from_numpy(img).unsqueeze(0).to(DEVICE)
        if DEVICE == "cuda":
            tensor = tensor.half()

        with torch.no_grad():
            score, _ = model(tensor)
            scores.append(float(score))

    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diverseg", required=True, help="Path to Diverseg-ip root")
    parser.add_argument("--lucid", required=True, help="Path to LUCID-IP-Complex root")
    parser.add_argument("--icnet", required=True, help="Path to complexity.pth")
    args = parser.parse_args()

    print("Initializing...")
    model = load_icnet(args.icnet)

    # Gather paths
    # Diverseg has subfolders (ImageNet structure)
    print("Scanning Diverseg...")
    diverseg_paths = sorted(
        list(Path(args.diverseg).rglob("*.JPEG"))
        + list(Path(args.diverseg).rglob("*.png"))
        + list(Path(args.diverseg).rglob("*.jpg"))
    )

    print("Scanning LUCID...")
    lucid_paths = sorted(Path(args.lucid).glob("*.png"))

    print(f"Diverseg Pool: {len(diverseg_paths)}")
    print(f"LUCID Pool:    {len(lucid_paths)}")

    # Score
    print("\n--- Analzying Diverseg (Random Crops) ---")
    div_scores = score_images(model, diverseg_paths, is_tiled=False)

    print("\n--- Analzying LUCID (Pre-tiled) ---")
    lucid_scores = score_images(model, lucid_paths, is_tiled=True)

    # Stats
    d_mean = np.mean(div_scores)
    l_mean = np.mean(lucid_scores)
    d_med = np.median(div_scores)
    l_med = np.median(lucid_scores)

    print("\n=== RESULTS ===")
    print("Metric       | Diverseg (Target) | LUCID (Ours)")
    print("-------------|-------------------|-------------")
    print(f"Mean Score   | {d_mean:.4f}            | {l_mean:.4f}")
    print(f"Median Score | {d_med:.4f}            | {l_med:.4f}")
    print(
        f"Max Score    | {np.max(div_scores):.4f}            | {np.max(lucid_scores):.4f}"
    )
    print(
        f"Min Score    | {np.min(div_scores):.4f}            | {np.min(lucid_scores):.4f}"
    )

    print("\nDistribution:")
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    d_hist, _ = np.histogram(div_scores, bins)
    l_hist, _ = np.histogram(lucid_scores, bins)

    print("Score Range | Div % | Luc %")
    for i in range(len(bins) - 1):
        d_p = (d_hist[i] / len(div_scores)) * 100
        l_p = (l_hist[i] / len(lucid_scores)) * 100
        print(f"{bins[i]:.1f} - {bins[i + 1]:.1f} | {d_p:5.1f}% | {l_p:5.1f}%")

    # Conclusion
    print("\n=== CONCLUSION ===")
    if d_mean > l_mean * 1.05:
        print("FAIL: Diverseg is significantly more complex.")
        print("Reason: Stage 2 likely filtered out the 'Hard' tiles.")
    elif l_mean > d_mean * 1.05:
        print("PASS: LUCID is actually MORE complex.")
        print(
            "Reason: The gap must be semantic (Object diversity vs Texture complexity)."
        )
    else:
        print("TIE: Complexity is similar.")


if __name__ == "__main__":
    main()
