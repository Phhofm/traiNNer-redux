#!/usr/bin/env python3
"""
LUCID — Learnable Under-sampling Consistency & Integrity Discovery
================================================================
Stage 2: Multi-Scale Forward-Backward Consistency Filtering
"""

import argparse
import csv
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ====================== CONFIG ======================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SCALES = {
    4: {"psnr_min": 25.0},  # hard gate (adjusted for probe ~26dB avg)
    2: {"psnr_min": 29.0},  # softer gate (easier task)
}

# Global ThreadPool for asynchronous file copying
# Fixed to 4 workers to prevent overwhelming the disk controller
_COPY_EXECUTOR = ThreadPoolExecutor(max_workers=4)

# ====================== MODEL (Required for Unpickling) ======================


class SRProbeNet(torch.nn.Module):
    def __init__(self, scale=4) -> None:
        super().__init__()
        self.scale = scale
        self.head = torch.nn.Conv2d(3, 32, 5, padding=2)
        self.body = torch.nn.Sequential(
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 32, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 32, 3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.tail = torch.nn.Conv2d(32, 3 * (scale**2), 3, padding=1)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return F.pixel_shuffle(x, self.scale)


# ====================== DATASET ======================


class TileDataset(Dataset):
    def __init__(self, paths) -> None:
        self.paths = sorted(paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            # Faster loading with cv2
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError("cv2 failed to load image")

            # cv2 BGR -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # To Tensor (CHW 0-1)
            arr = img.astype(np.float32) / 255.0
            t = torch.from_numpy(arr).permute(2, 0, 1)
            return t, str(path)
        except Exception:
            return None, ""


def collate_fn(batch):
    # Filter out failed loads (None from __getitem__)
    batch = [b for b in batch if b[0] is not None]
    if not batch:
        return None, []

    images, paths = zip(*batch, strict=False)
    images = torch.stack(images)
    return images, paths


# ====================== CORE ======================


def process_batch(model, images, paths, out_dir, log_rows=None):
    # Set lower priority for the processing logic
    try:
        os.nice(10)
    except Exception:
        pass

    if images is None or images.numel() == 0:
        return 0

    images = images.to(DEVICE)
    batch_size = images.shape[0]

    keep_mask = torch.ones(batch_size, dtype=torch.bool, device=DEVICE)
    batch_psnrs = {p: dict.fromkeys(SCALES, 0.0) for p in paths}

    for scale in sorted(SCALES.keys(), reverse=True):
        cfg = SCALES[scale]

        if not keep_mask.any() and log_rows is None:
            break

        lr = F.interpolate(
            images, scale_factor=1 / scale, mode="bicubic", align_corners=False
        )
        with autocast(device_type="cuda"):
            with torch.no_grad():
                sr = model(lr).clamp(0, 1)
        sr = sr.float()

        if sr.shape[-2:] != images.shape[-2:]:
            sr = F.interpolate(
                sr, size=images.shape[-2:], mode="bicubic", align_corners=False
            ).clamp(0, 1)

        mse = ((sr - images) ** 2).mean(dim=[1, 2, 3])
        psnr_vals = 10 * torch.log10(1.0 / (mse + 1e-8))

        for i, p in enumerate(paths):
            batch_psnrs[p][scale] = float(psnr_vals[i].item())

        pass_mask = psnr_vals >= cfg["psnr_min"]
        keep_mask = keep_mask & pass_mask

    cpu_mask = keep_mask.cpu().numpy()
    kept_count = 0

    for i, path_str in enumerate(paths):
        src = Path(path_str)
        is_kept = bool(cpu_mask[i])

        if is_kept:
            dst = out_dir / src.name
            _COPY_EXECUTOR.submit(shutil.copy, src, dst)
            kept_count += 1

        if log_rows is not None:
            row = (
                [src.name]
                + [
                    batch_psnrs[path_str][s]
                    for s in sorted(SCALES.keys(), reverse=True)
                ]
                + [is_kept]
            )
            log_rows.append(row)

    return kept_count


# ====================== MAIN ======================


def main() -> None:
    parser = argparse.ArgumentParser(description="LUCID Stage 2: Consistency Filtering")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--csv", help="Optional CSV path for metadata logging")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--workers", type=int, default=None, help="DataLoader workers (default: 4)"
    )
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reserve CPU resources for the desktop
    num_workers = args.workers if args.workers is not None else 4

    print(f"Loading model on {DEVICE}...")
    model = torch.load(args.weights, map_location=DEVICE, weights_only=False)
    model.eval()
    model.to(DEVICE)

    # Use a generator to load paths lazily
    tile_paths = sorted(in_dir.glob("*.png"))
    num_tiles = len(tile_paths)
    print(f"Found {num_tiles} tiles. Batch size: {args.batch_size}")

    dataset = TileDataset(tile_paths)
    # Use num_workers=0 and pin_memory=False for maximum reliability in long loops
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=False,
    )

    kept = 0
    print("GPU Filtering...")

    csv_file = None
    csv_writer = None
    if args.csv:
        file_exists = os.path.isfile(args.csv)
        csv_file = open(args.csv, "a", newline="")
        csv_writer = csv.writer(csv_file)
        if not file_exists:
            cols = (
                ["tile"]
                + [f"psnr_x{s}" for s in sorted(SCALES.keys(), reverse=True)]
                + ["kept"]
            )
            csv_writer.writerow(cols)

    try:
        for item in tqdm(loader):
            if item is None or item[0] is None:
                continue
            images, paths = item
            log_rows = [] if args.csv else None
            kept += process_batch(model, images, paths, out_dir, log_rows=log_rows)

            if csv_writer and log_rows:
                csv_writer.writerows(log_rows)
    finally:
        if csv_file:
            csv_file.close()

    print("\nShutting down async copy threads...")
    _COPY_EXECUTOR.shutdown(wait=True)

    print(f"Kept {kept}/{num_tiles} tiles")


if __name__ == "__main__":
    main()
