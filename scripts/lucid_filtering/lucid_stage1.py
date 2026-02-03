#!/usr/bin/env python3
"""
LUCID — Learnable Under-sampling Consistency & Integrity Discovery
================================================================
Stage 1: Signal-Theoretic Dataset Filtering
"""

import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# ========================= CONFIG =========================


SCALES = [
    (1.0, "100"),
    (0.75, "75"),
    (0.5, "50"),
    (0.25, "25"),
]

THRESHOLDS = {
    "entropy_min": 5.5,
    "lap_var_min": 100.0,
    "lap_var_max": 8000.0,  # Increased for sharp DF2K
    "blockiness_max": 40.0,  # Relaxed
    "aliasing_max": 0.60,  # Relaxed
    "grad_energy_min": 5.0,
    "noise_ratio_max": 0.60,
}

# ========================= METRICS =========================


def entropy(gray):
    # Fast histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    p = hist.ravel() / (hist.sum() + 1e-9)
    # Mask zeros to avoid log(0)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def laplacian_variance(gray):
    # Highly optimized in OpenCV
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def blockiness(gray):
    """
    Vectorized calculation of blockiness at 8x8 boundaries.
    """
    h, w = gray.shape
    if h < 16 or w < 16:
        return 0.0

    # Vertically aligned boundaries (differences between columns)
    v_right = gray[:, 8::8]
    v_left = gray[:, 7::8]
    v_len = min(v_right.shape[1], v_left.shape[1])
    v_diffs = np.abs(
        v_right[:, :v_len].astype(np.int16) - v_left[:, :v_len].astype(np.int16)
    )
    v_score = np.sum(v_diffs)

    # Horizontally aligned boundaries
    h_bottom = gray[8::8, :]
    h_top = gray[7::8, :]
    h_len = min(h_bottom.shape[0], h_top.shape[0])
    h_diffs = np.abs(
        h_bottom[:h_len, :].astype(np.int16) - h_top[:h_len, :].astype(np.int16)
    )
    h_score = np.sum(h_diffs)

    return float(v_score + h_score) / (h * w)


# --- GLOBAL CACHE ---
_MASK_CACHE = {}


def get_aliasing_mask(h, w, r):
    key = (h, w, r)
    if key not in _MASK_CACHE:
        cy, cx = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        _MASK_CACHE[key] = dist_from_center >= r
    return _MASK_CACHE[key]


def aliasing_ratio(gray):
    # FFT is relatively expensive
    f = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.abs(f)
    h, w = gray.shape
    r = int(min(h // 2, w // 2) * 0.75)

    mask = get_aliasing_mask(h, w, r)
    return float(mag[mask].mean() / (mag.mean() + 1e-9))


def gradient_energy(gray):
    # OpenCV Sobel is fast
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.sqrt(gx**2 + gy**2)
    return float(np.mean(mag))


def noise_ratio(gray):
    """
    High-frequency energy NOT explained by gradients.
    """
    # High-pass residual via Gaussian subtraction
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    hf = gray.astype(np.float32) - blur.astype(np.float32)
    hf_energy = np.mean(hf**2)

    # Structured energy proxy
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_energy = np.mean(gx**2 + gy**2)

    return float(hf_energy / (grad_energy + 1e-9))


# ========================= PIPELINE =========================


def process_chunk_worker(chunk):
    """
    Helper for multiprocessing chunking.
    Processes a list of tasks and returns their results.
    """
    results = []
    for task in chunk:
        results.append(process_single_image(task))
    return results


def chunked_gen(it, size):
    """
    Yields chunks of items from an iterator.
    """
    chunk = []
    for item in it:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def process_tile(gray, t):
    """
    TIERED FILTERING (Early Exit)
    ----------------------------
    Tier 1 (Instant): Rejects flat/uninformative or blurry tiles
    Tier 2 (Fast): Rejects low-texture tiles
    Tier 3 (Heavy): Rejects artifacts/noise (most expensive)
    """
    # --- Tier 1: Informational Content & Sharpness ---
    e = entropy(gray)
    if e < t["entropy_min"]:
        return False, "fail_entropy"

    lap = laplacian_variance(gray)
    if lap < t["lap_var_min"] or lap > t["lap_var_max"]:
        return False, "fail_lap_var"

    # --- Tier 2: Texture Strength ---
    ge = gradient_energy(gray)
    if ge < t["grad_energy_min"]:
        return False, "fail_grad_energy"

    # --- Tier 3: Technical Flaws (Expensive) ---
    if blockiness(gray) > t["blockiness_max"]:
        return False, "fail_blockiness"

    if aliasing_ratio(gray) > t["aliasing_max"]:
        return False, "fail_aliasing"

    if noise_ratio(gray) > t["noise_ratio_max"]:
        return False, "fail_noise"

    return True, None


def process_single_image(args_pack):
    # Set lower priority for worker processes to keep system responsive
    try:
        os.nice(10)
    except Exception:
        pass

    # Unpack arguments for multiprocessing
    img_path, in_dir, out_dir, tile_size, thresholds, scales = args_pack

    local_stats = {
        "processed_tiles": 0,
        "saved_tiles": 0,
        "fail_entropy": 0,
        "fail_lap_var": 0,
        "fail_blockiness": 0,
        "fail_aliasing": 0,
        "fail_grad_energy": 0,
        "fail_noise": 0,
        "skipped_corrupt": 0,
    }

    csv_rows = []

    try:
        # Load as RGB then convert to Gray once per scale
        pil_img = Image.open(img_path).convert("RGB")
        img_np = np.array(pil_img)
    except Exception:
        local_stats["skipped_corrupt"] += 1
        return local_stats, csv_rows

    h, w, _ = img_np.shape
    if h < tile_size or w < tile_size:
        return local_stats, csv_rows

    # Unique naming
    try:
        rel_path = img_path.relative_to(in_dir)
    except ValueError:
        rel_path = img_path.name

    name_base = str(Path(rel_path).with_suffix("")).replace("/", "_").replace("\\", "_")

    for s, suf in scales:
        if s != 1.0:
            # INTER_AREA is best for downscaling
            scaled_rgb = cv2.resize(
                img_np, None, fx=s, fy=s, interpolation=cv2.INTER_AREA
            )
        else:
            scaled_rgb = img_np

        sh, sw, _ = scaled_rgb.shape
        if sh < tile_size or sw < tile_size:
            continue

        # Convert to Grayscale ONCE for the entire scaled image
        scaled_gray = cv2.cvtColor(scaled_rgb, cv2.COLOR_RGB2GRAY)

        for y in range(sh // tile_size):
            for x in range(sw // tile_size):
                y0, y1 = y * tile_size, (y + 1) * tile_size
                x0, x1 = x * tile_size, (x + 1) * tile_size

                gray_tile = scaled_gray[y0:y1, x0:x1]

                local_stats["processed_tiles"] += 1
                passed, reason = process_tile(gray_tile, thresholds)

                if passed:
                    # Save with cv2 (usually faster than PIL)
                    # Note: cv2 uses BGR, we have RGB
                    rgb_tile = scaled_rgb[y0:y1, x0:x1]
                    bgr_tile = cv2.cvtColor(rgb_tile, cv2.COLOR_RGB2BGR)
                    fname = f"{name_base}_{suf}_{y}_{x}.png"
                    cv2.imwrite(str(out_dir / fname), bgr_tile)

                    csv_rows.append([fname, suf])
                    local_stats["saved_tiles"] += 1
                else:
                    if reason:
                        local_stats[reason] += 1

    return local_stats, csv_rows


# ========================= MAIN =========================


def main():
    parser = argparse.ArgumentParser(description="LUCID High-Performance Signal Filter")
    parser.add_argument("input", help="Source dataset directory")
    parser.add_argument("output", help="Directory to save filtered tiles")
    parser.add_argument("csv", help="Metadata CSV file")
    parser.add_argument(
        "--tile_size", type=int, default=256, help="Tile size (default: 256)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: cores-1)",
    )
    parser.add_argument(
        "--chunksize", type=int, default=40, help="Images per task chunk (default: 40)"
    )
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. FAST IMAGE FINDING
    print("Finding images...")
    valid_extensions = {".png", ".jpg", ".jpeg"}

    # Materialize once to get total count, but keep it efficient
    images = [p for p in in_dir.rglob("*") if p.suffix.lower() in valid_extensions]
    num_imgs = len(images)

    if num_imgs == 0:
        print("!! No images found in input directory.")
        return

    # Use a default worker count of cores-1 to keep system responsive
    max_workers = (
        args.workers if args.workers is not None else max(1, os.cpu_count() - 1)
    )

    print(f"Found {num_imgs} images. Target: {args.tile_size}px tiles.")
    print(f"Dispatching with {max_workers} workers (Chunksize: {args.chunksize})...")

    global_stats = {
        "processed_tiles": 0,
        "saved_tiles": 0,
        "fail_entropy": 0,
        "fail_lap_var": 0,
        "fail_blockiness": 0,
        "fail_aliasing": 0,
        "fail_grad_energy": 0,
        "fail_noise": 0,
        "skipped_corrupt": 0,
    }

    start_time = time.time()
    total_saved = 0
    total_processed = 0

    from concurrent.futures import as_completed

    with open(args.csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tile", "scale"])

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            pbar = tqdm(total=num_imgs, desc="Processing images")

            # Arguments for workers
            tasks_gen = (
                (p, in_dir, out_dir, args.tile_size, THRESHOLDS, SCALES) for p in images
            )

            # Submit initial batch of chunks
            # We limit the number of active futures to prevent RAM explosion
            max_active_chunks = max_workers * 2
            chunk_generator = chunked_gen(tasks_gen, args.chunksize)

            futures = {}  # {future: chunk_size}

            # Fill the initial buffer
            for _ in range(max_active_chunks):
                try:
                    chunk = next(chunk_generator)
                    fut = executor.submit(process_chunk_worker, chunk)
                    futures[fut] = len(chunk)
                except StopIteration:
                    break

            while futures:
                # wait for any chunk to complete
                for fut in as_completed(futures):
                    chunk_size = futures.pop(fut)
                    try:
                        chunk_results = fut.result()

                        # Aggregate results from chunk
                        for stats, rows in chunk_results:
                            for k, v in stats.items():
                                global_stats[k] += v
                            for row in rows:
                                writer.writerow(row)
                                total_saved += 1
                            total_processed += 1

                        pbar.update(chunk_size)
                    except Exception as e:
                        print(f"\n!! Error in worker chunk: {e}")

                    # Submit a new chunk to replace the finished one
                    try:
                        next_chunk = next(chunk_generator)
                        new_fut = executor.submit(process_chunk_worker, next_chunk)
                        futures[new_fut] = len(next_chunk)
                    except StopIteration:
                        pass

                    # break to iterate the loop and update progress faster
                    break

            pbar.close()

    duration = time.time() - start_time
    imgs_per_sec = total_processed / duration if duration > 0 else 0

    print(f"\nSaved {total_saved} tiles")
    print(f"Time: {duration:.2f}s | Speed: {imgs_per_sec:.2f} images/s")

    print("\n=== Filter Statistics ===")
    print(f"Processed Tiles: {global_stats['processed_tiles']}")
    print(f"Saved Tiles:     {global_stats['saved_tiles']}")
    print(f"Skipped Corrupt: {global_stats['skipped_corrupt']}")
    print("Rejected Reasons:")
    print(f"  Entropy:       {global_stats['fail_entropy']:>10}")
    print(f"  Laplacian:     {global_stats['fail_lap_var']:>10}")
    print(f"  Blockiness:    {global_stats['fail_blockiness']:>10}")
    print(f"  Aliasing:      {global_stats['fail_aliasing']:>10}")
    print(f"  Grad Energy:   {global_stats['fail_grad_energy']:>10}")
    print(f"  Noise Ratio:   {global_stats['fail_noise']:>10}")


if __name__ == "__main__":
    main()
