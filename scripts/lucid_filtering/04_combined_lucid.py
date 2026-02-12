#!/usr/bin/env python3
"""
LUCID Master Elite: Unified High-Performance Pipeline
=====================================================
Stage 1 (Signal Filter) + Stage 2 (Complexity Score) combined.

Logic:
1. CPU Producers: Read image, Tile, Filter (Signal).
2. Queue: Passing tiles sent to Scoring Queue.
3. GPU Consumer: Batch Scoring (ICNet), CSV logging, Disk Write (if >= Target).

Optimized for: Slow External HDDs & High-Resolution (4K) datasets.
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from queue import Empty

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image
from tqdm import tqdm

mp.set_start_method("spawn", force=True)

# Fix for "Python stopped working" notifications on Linux:
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# ========================= CONFIG =========================

# ICNet Import
REPO_ROOT = Path(__file__).parent.parent.parent
ICNET_DIR = REPO_ROOT / "datasets" / "preparation" / "complexity"
sys.path.append(str(ICNET_DIR))

try:
    from ICNet import ICNet
except ImportError:
    print(f"FATAL: Could not import ICNet from {ICNET_DIR}")
    sys.exit(1)

# Default Signal Thresholds
THRESHOLDS = {
    "entropy_min": 5.5,
    "lap_var_min": 100.0,
    "lap_var_max": 8000.0,
    "blockiness_max": 40.0,
    "aliasing_max": 0.60,
    "grad_energy_min": 0.75,
    "noise_ratio_max": 0.60,
}

# ========================= SIGNAL METRICS =========================


def entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    p = hist.ravel() / (hist.sum() + 1e-9)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def laplacian_variance(gray: np.ndarray) -> float:
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def blockiness(gray: np.ndarray) -> float:
    h, w = gray.shape
    if h < 16 or w < 16:
        return 0.0
    v_right, v_left = gray[:, 8::8], gray[:, 7::8]
    v_len = min(v_right.shape[1], v_left.shape[1])
    v_score = np.sum(
        np.abs(v_right[:, :v_len].astype(np.int16) - v_left[:, :v_len].astype(np.int16))
    )
    h_bottom, h_top = gray[8::8, :], gray[7::8, :]
    h_len = min(h_bottom.shape[0], h_top.shape[0])
    h_score = np.sum(
        np.abs(h_bottom[:h_len, :].astype(np.int16) - h_top[:h_len, :].astype(np.int16))
    )
    return float(v_score + h_score) / (h * w)


_MASK_CACHE = {}


def get_aliasing_mask(h: int, w: int, r: int) -> np.ndarray:
    key = (h, w, r)
    if key not in _MASK_CACHE:
        cy, cx = h // 2, w // 2
        y_grid, x_grid = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
        _MASK_CACHE[key] = dist_from_center >= r
    return _MASK_CACHE[key]


def aliasing_ratio(gray: np.ndarray) -> float:
    f = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.abs(f)
    h, w = gray.shape
    r = int(min(h // 2, w // 2) * 0.75)
    mask = get_aliasing_mask(h, w, r)
    return float(mag[mask].mean() / (mag.mean() + 1e-9))


def gradient_energy(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return float(np.mean(cv2.sqrt(gx**2 + gy**2)))


def noise_ratio(gray: np.ndarray) -> float:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    hf = gray.astype(np.float32) - blur.astype(np.float32)
    hf_energy = np.mean(hf**2)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    grad_energy = np.mean(gx**2 + gy**2)
    return float(hf_energy / (grad_energy + 1e-9))


# ========================= WORKERS =========================


def signal_process_worker_wrapper(args: tuple) -> bool:
    return signal_process_worker(*args)


def signal_process_worker(
    img_path: Path, tile_size: int, results_queue: mp.Queue, thresholds: dict
) -> bool:
    """
    Tiered filtering on the CPU.
    """
    try:
        os.nice(15)
        pil_img = Image.open(img_path).convert("RGB")
        img_np = np.array(pil_img)
        h, w, _ = img_np.shape
        name_base = img_path.stem

        if h < tile_size or w < tile_size:
            return False

        for y in range(h // tile_size):
            for x in range(w // tile_size):
                y0, y1 = y * tile_size, (y + 1) * tile_size
                x0, x1 = x * tile_size, (x + 1) * tile_size

                rgb_tile = img_np[y0:y1, x0:x1]
                gray_tile = cv2.cvtColor(rgb_tile, cv2.COLOR_RGB2GRAY)

                # Early Exit Filter
                if entropy(gray_tile) < thresholds["entropy_min"]:
                    continue
                lap = laplacian_variance(gray_tile)
                if lap < thresholds["lap_var_min"] or lap > thresholds["lap_var_max"]:
                    continue
                if gradient_energy(gray_tile) < thresholds["grad_energy_min"]:
                    continue
                if blockiness(gray_tile) > thresholds["blockiness_max"]:
                    continue
                if noise_ratio(gray_tile) > thresholds["noise_ratio_max"]:
                    continue

                results_queue.put(
                    {
                        "data": rgb_tile,
                        "name": f"{name_base}_t{y}_{x}.png",
                        "path": str(img_path),
                        "pos": (y, x),
                    }
                )
        return True
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return False


def gpu_scoring_consumer(
    results_queue: mp.Queue,
    model_path: str,
    out_dir: Path,
    csv_path: Path,
    elite_threshold: float,
    batch_size: int,
) -> None:
    """
    Singleton GPU Consumer. Pulls filtered tiles, scores them, and writes.
    """
    try:
        os.nice(15)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load ICNet
        model = ICNet(is_pretrain=False)
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device).eval()
        if device == "cuda":
            model.half()

        # CSV Header
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["tile_name", "complexity_score", "source"])

        mean = (
            torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1).half()
        )
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1).half()

        buffer = []
        finished_sentinels = 0
        total_pushed = 0

        while True:
            try:
                # Short timeout to allow check for finished state
                item = results_queue.get(timeout=2)
                if item == "FINISHED":
                    finished_sentinels += 1
                else:
                    buffer.append(item)
            except Empty:
                if finished_sentinels > 0:
                    break
                continue

            # Batch Score if buffer is full or we are finishing
            if len(buffer) >= batch_size or (finished_sentinels > 0 and buffer):
                batch_imgs = []
                for b in buffer:
                    img = b["data"]
                    if img.shape[0] != 512 or img.shape[1] != 512:
                        img = cv2.resize(
                            img, (512, 512), interpolation=cv2.INTER_LINEAR
                        )
                    batch_imgs.append(torch.from_numpy(img))

                tensors = (
                    torch.stack(batch_imgs).to(device).permute(0, 3, 1, 2).half()
                    / 255.0
                )
                tensors = (tensors - mean) / std

                with torch.no_grad():
                    scores, _ = model(tensors)
                    scores = scores.flatten().cpu().float().numpy().tolist()

                # Process results
                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    for b, score in zip(buffer, scores, strict=False):
                        writer.writerow([b["name"], f"{score:.6f}", b["path"]])
                        if score >= elite_threshold:
                            # SAVE TO DISK
                            save_path = out_dir / b["name"]
                            bgr = cv2.cvtColor(b["data"], cv2.COLOR_RGB2BGR)
                            cv2.imwrite(str(save_path), bgr)

                total_pushed += len(buffer)
                buffer = []

        print(f"\nGPU Consumer finished. Processed {total_pushed} tiles.")
    except Exception as e:
        print(f"FATAL GPU Error: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LUCID Master Elite Expansion: Combined Pipeline"
    )
    parser.add_argument(
        "--input", required=True, help="Input directory of high-res images"
    )
    parser.add_argument(
        "--output", required=True, help="Base output directory on external HDD"
    )
    parser.add_argument("--icnet", required=True, help="Path to complexity.pth")
    parser.add_argument("--csv", help="Optional: Custom path for the metadata CSV")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Complexity target (Default: 0.50)",
    )
    parser.add_argument(
        "--tile_size", type=int, default=512, help="Tile size (Default: 512)"
    )
    parser.add_argument("--batch", type=int, default=32, help="GPU batch size")
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count() - 2, help="CPU CPU workers"
    )
    args = parser.parse_args()

    in_dir = Path(args.input)
    # Automatic subfolder naming
    dataset_name = (
        in_dir.parent.name
        if in_dir.name in ["HR", "images", "train", "training"]
        else in_dir.name
    )
    base_out = Path(args.output)
    target_out = base_out / dataset_name / "tiles"
    target_out.mkdir(parents=True, exist_ok=True)

    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        csv_path = base_out / dataset_name / f"{dataset_name}_lucid.csv"

    print(f"--- LUCID Master Elite: {dataset_name} ---")
    print(f"Settings: Tile={args.tile_size}px, Score >= {args.threshold}")
    print(f"CSV Metadata: {csv_path}")

    # Scan
    valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
    images = []
    for root, _, files in os.walk(in_dir):
        for f in files:
            if any(f.lower().endswith(ext) for ext in valid_exts):
                images.append(Path(root) / f)

    if not images:
        print("No images found.")
        return

    print(f"Found {len(images)} source images.")

    # Manager for shared queue
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    results_queue = manager.Queue(maxsize=2000)

    # Start GPU Consumer
    gpu_proc = ctx.Process(
        target=gpu_scoring_consumer,
        args=(
            results_queue,
            args.icnet,
            target_out,
            csv_path,
            args.threshold,
            args.batch,
        ),
    )
    gpu_proc.daemon = True
    gpu_proc.start()

    # CPU Producer Pool
    try:
        with ctx.Pool(processes=args.workers) as pool:
            # Prepare arguments for imap
            worker_args = [
                (p, args.tile_size, results_queue, THRESHOLDS) for p in images
            ]

            # Use imap_unordered for progress tracking
            for _ in tqdm(
                pool.imap_unordered(signal_process_worker_wrapper, worker_args),
                total=len(images),
                desc="Filtering (Signal)",
            ):
                pass
    except Exception as e:
        print(f"Manager/Pool error: {e}")
    finally:
        # Finish signals
        results_queue.put("FINISHED")
        gpu_proc.join(timeout=30)
        if gpu_proc.is_alive():
            gpu_proc.terminate()
    print("Pipeline Complete.")


if __name__ == "__main__":
    main()
