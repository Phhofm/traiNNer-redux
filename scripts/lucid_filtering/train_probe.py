#!/usr/bin/env python3
"""
LUCID — Learnable Under-sampling Consistency & Integrity Discovery
================================================================
Training Script: SR-ProbeNet for Consistency Filtering
"""

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from tqdm import tqdm

# ===================== CONFIG =====================

SCALE = 4
PATCH_SIZE = 192  # HR patch size
BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-4
DEVICE = "cuda"

TRAIN_DIR = "DIV2K_train_HR"
VAL_DIR = "DIV2K_valid_HR"
OUT_PATH = "sr_probe_x4.pth"

# ===================== MODEL =====================


class SRProbeNet(nn.Module):
    def __init__(self, scale=4):
        super().__init__()
        self.scale = scale
        self.head = nn.Conv2d(3, 32, 5, padding=2)
        self.body = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.tail = nn.Conv2d(32, 3 * (scale**2), 3, padding=1)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return F.pixel_shuffle(x, self.scale)


# ===================== DATASET =====================


class HRDataset(Dataset):
    def __init__(self, root):
        self.paths = list(Path(root).glob("*.png"))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        while True:
            try:
                img = Image.open(self.paths[idx]).convert("RGB")
                break
            except (OSError, SyntaxError) as e:
                print(f"Warning: Skipping corrupted image {self.paths[idx]}: {e}")
                idx = random.randint(0, len(self.paths) - 1)
        w, h = img.size

        x = random.randint(0, w - PATCH_SIZE)
        y = random.randint(0, h - PATCH_SIZE)
        img = img.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))

        if random.random() < 0.5:
            img = TF.hflip(img)
        if random.random() < 0.5:
            img = TF.vflip(img)
        if random.random() < 0.5:
            img = img.rotate(90)

        hr = TF.to_tensor(img)
        return hr


# ===================== UTILS =====================


def degrade(hr):
    return F.interpolate(
        hr, scale_factor=1 / SCALE, mode="bicubic", align_corners=False
    )


def psnr(sr, hr):
    mse = F.mse_loss(sr, hr)
    return 10 * torch.log10(1.0 / mse)


# ===================== TRAIN =====================


def main():
    parser = argparse.ArgumentParser(description="Train SR Probe")
    parser.add_argument(
        "--train", type=str, default="DIV2K_train_HR", help="Path to training images"
    )
    parser.add_argument(
        "--val", type=str, default="DIV2K_valid_HR", help="Path to validation images"
    )
    parser.add_argument(
        "--output", type=str, default="sr_probe_x4.pth", help="Path to save model"
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    train_ds = HRDataset(args.train)
    val_ds = HRDataset(args.val)

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = SRProbeNet(scale=SCALE).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best_psnr = 0.0

    print(f"Training on {args.train}, Validating on {args.val}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for hr in tqdm(train_dl, desc=f"Epoch {epoch}"):
            hr = hr.to(DEVICE)
            lr = degrade(hr)
            sr = model(lr)

            loss = F.l1_loss(sr, hr)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item()

        # ---- Validation ----
        model.eval()
        val_psnr = 0.0
        with torch.no_grad():
            for hr in val_dl:
                hr = hr.to(DEVICE)
                lr = degrade(hr)
                sr = model(lr)
                val_psnr += psnr(sr, hr).item()

        val_psnr /= len(val_dl)

        print(
            f"Epoch {epoch:02d} | "
            f"Train L1: {running_loss / len(train_dl):.4f} | "
            f"Val PSNR: {val_psnr:.2f} dB"
        )

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model, args.output)
            print("  ✓ Saved new best probe")

    print(f"Best validation PSNR: {best_psnr:.2f} dB")


if __name__ == "__main__":
    main()
