#!/usr/bin/env python3
import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

# Add ICNet to path (relative to scripts/lucid_v2/)
REPO_ROOT = Path(__file__).parent.parent.parent
ICNET_DIR = REPO_ROOT / "datasets" / "preparation" / "complexity"
sys.path.append(str(ICNET_DIR))

try:
    from ICNet import ICNet
except ImportError:
    print(f"FATAL: Could not import ICNet from {ICNET_DIR}")
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="LUCID v2 - Metadata Recovery Scorer")
    parser.add_argument("--input", required=True, help="Folder of tiles (MASTER_ELITE)")
    parser.add_argument("--icnet", required=True, help="Path to complexity.pth")
    parser.add_argument("--output", required=True, help="Path to scores.csv output")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--size", type=int, default=512)

    args = parser.parse_args()
    input_dir = Path(args.input)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Model
    model = ICNet(is_pretrain=False, size1=args.size, size2=args.size // 2)
    model.load_state_dict(
        torch.load(args.icnet, map_location=device, weights_only=True)
    )
    model.to(device).eval()
    if device == "cuda":
        model.half()

    # Discovery
    files = sorted(input_dir.glob("*.png"))
    if not files:
        print("No PNG files found in input directory.")
        return

    # Normalization constants (matching 01_ingest.py)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    if device == "cuda":
        mean, std = mean.half(), std.half()

    print(f"Scoring {len(files)} tiles to {args.output}...")

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tile_name", "complexity_score"])

        for i in tqdm(range(0, len(files), args.batch)):
            batch_paths = files[i : i + args.batch]
            batch_tensors = []
            valid_paths = []

            for p in batch_paths:
                img = cv2.imread(str(p))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                batch_tensors.append(torch.from_numpy(img))
                valid_paths.append(p.name)

            if not batch_tensors:
                continue

            tensors = torch.stack(batch_tensors).to(device).permute(0, 3, 1, 2)
            if device == "cuda":
                tensors = tensors.half()

            tensors = (tensors / 255.0 - mean) / std

            with torch.no_grad():
                scores, _ = model(tensors)
                scores = scores.flatten().cpu().float().numpy().tolist()

            for name, score in zip(valid_paths, scores, strict=False):
                writer.writerow([name, f"{score:.6f}"])

            if i % (args.batch * 10) == 0:
                f.flush()

    print("Success! Scores recovered.")


if __name__ == "__main__":
    main()
