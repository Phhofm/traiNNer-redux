import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def entropy(gray):
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum()
    logs = np.log2(hist + 1e-9)
    return -1.0 * (hist * logs).sum()


def laplacian_variance(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def gradient_energy(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    mag = cv2.sqrt(gx**2 + gy**2)
    return float(np.mean(mag))


def noise_ratio(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    hf = gray.astype(np.float32) - blur.astype(np.float32)
    hf_energy = np.mean(hf**2)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    ge = np.mean(gx**2 + gy**2)
    return float(hf_energy / (ge + 1.0))


def check_diverseg() -> None:
    in_dir = Path("/home/phips/Documents/dataset/diverseg-ip")
    valid_exts = {".png", ".jpg", ".jpeg"}
    images = []
    for root, _, files in os.walk(in_dir):
        for f in files:
            if any(f.lower().endswith(ext) for ext in valid_exts):
                images.append(Path(root) / f)

    print(f"Checking 500 random images from {in_dir}...")
    import random

    random.seed(42)
    sample = random.sample(images, min(len(images), 500))

    stats = {
        "passed": 0,
        "fail_entropy": 0,
        "fail_laplacian": 0,
        "fail_noise": 0,
        "total": 0,
    }

    noise_vals = []
    for img_path in tqdm(sample):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # We take a 256x256 center crop to match our tiling logic
        h, w = gray.shape
        if h < 256 or w < 256:
            continue

        y0, x0 = (h - 256) // 2, (w - 256) // 2
        tile = gray[y0 : y0 + 256, x0 : x0 + 256]

        e = entropy(tile)
        lap = laplacian_variance(tile)
        noise = noise_ratio(tile)
        noise_vals.append(noise)

        stats["total"] += 1
        fail = False
        if e < 5.0:
            stats["fail_entropy"] += 1
            fail = True
        if lap < 100:
            stats["fail_laplacian"] += 1
            fail = True
        if noise > 0.15:
            stats["fail_noise"] += 1
            fail = True

        if not fail:
            stats["passed"] += 1

    print("\nDiverseg-ip Noise Distribution:")
    print(f"  Mean:   {np.mean(noise_vals):.4f}")
    print(f"  Median: {np.median(noise_vals):.4f}")
    print(f"  Max:    {np.max(noise_vals):.4f}")
    print(f"  Min:    {np.min(noise_vals):.4f}")

    print("\nDiverseg-ip vs Stage 1 Thresholds:")
    for k, v in stats.items():
        if k != "total":
            print(f"  {k}: {v} ({v / stats['total'] * 100:.1f}%)")
    print(f"  Total Sampled: {stats['total']}")


if __name__ == "__main__":
    check_diverseg()
