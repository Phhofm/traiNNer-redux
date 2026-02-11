import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def noise_ratio(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    hf = gray.astype(np.float32) - blur.astype(np.float32)
    hf_energy = np.mean(hf**2)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    ge = np.mean(gx**2 + gy**2)
    return float(hf_energy / (ge + 1.0))


def check_our_tiles() -> None:
    # Check PASS tiles
    in_dir = Path("/home/phips/Documents/dataset/pass-lucid-stage1")
    if not in_dir.exists():
        in_dir = Path("/home/phips/Documents/dataset/imagenet-lucid-stage1")

    valid_exts = {".png", ".jpg"}
    images = []
    print(f"Scanning {in_dir}...")
    for root, _, files in os.walk(in_dir):
        for f in files:
            if f.lower().endswith(".png"):  # Our tiles are PNG
                images.append(Path(root) / f)
                if len(images) > 10000:
                    break  # Scan enough
        if len(images) > 10000:
            break

    print("Checking 500 random tiles from our filtered set...")
    import random

    random.seed(42)
    sample = random.sample(images, min(len(images), 500))

    noise_vals = []
    for img_path in tqdm(sample):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        noise = noise_ratio(gray)
        noise_vals.append(noise)

    print("\nOur Filtered Tiles Noise Distribution:")
    print(f"  Mean:   {np.mean(noise_vals):.4f}")
    print(f"  Median: {np.median(noise_vals):.4f}")
    print(f"  Max:    {np.max(noise_vals):.4f}")
    print(f"  Min:    {np.min(noise_vals):.4f}")


if __name__ == "__main__":
    check_our_tiles()
