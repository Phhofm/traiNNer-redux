#!/usr/bin/env python3
"""
LUCID v2: Step 01 - Ingest (Filter + Score + Verify)
===================================================
A unified high-performance script that prepares raw datasets for the Master Elite collection.

Combines:
1. Signal Filtering (CPU)
2. Complexity Scoring (GPU - ICNet)
3. Bitstream Integrity Verification (Deep Decode)
4. Profile Standardization (Convert to 8-bit RGB)

Safe: Uses os.nice(15) and handles Graceful Interrupts.
"""

import argparse
import csv
import os
import sys
import traceback
from pathlib import Path
from queue import Empty

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image, ImageFile
from tqdm import tqdm

# Force PIL to raise errors for truncated files so we can catch corruption
ImageFile.LOAD_TRUNCATED_IMAGES = False

# ICNet Setup
REPO_ROOT = Path(__file__).parent.parent.parent
ICNET_DIR = REPO_ROOT / "datasets" / "preparation" / "complexity"
sys.path.append(str(ICNET_DIR))

try:
    from ICNet import ICNet
except ImportError:
    print(f"FATAL: Could not import ICNet from {ICNET_DIR}")
    sys.exit(1)

mp.set_start_method("spawn", force=True)

# ========================= SIGNAL CONFIG =========================

THRESHOLDS = {
    "entropy_min": 5.2,  # Slightly relaxed for 512px context
    "lap_var_min": 80.0,
    "lap_var_max": 10000.0,
    "blockiness_max": 45.0,
    "aliasing_max": 0.65,
    "grad_energy_min": 0.70,
    "noise_ratio_max": 0.65,
}

# ========================= METRICS =========================


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


def aliasing_ratio(gray: np.ndarray) -> float:
    f = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.abs(f)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    y_grid, x_grid = np.ogrid[:h, :w]
    r = int(min(h // 2, w // 2) * 0.75)
    mask = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2) >= r
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


def producer_worker(args_tuple: tuple) -> bool:
    img_path, tile_size, queue, thresholds, multiscale, processed_tasks = args_tuple
    try:
        os.nice(15)
        # 1. Profile Audit & Standardize on the fly
        with Image.open(img_path) as pil_img:
            # Deep Integrity Check
            pil_img.load()

            # Standardize
            if pil_img.mode != "RGB":
                rgb_pil = pil_img.convert("RGB")
            else:
                rgb_pil = pil_img

            full_img_np = np.array(rgb_pil)
            h_orig, w_orig, _ = full_img_np.shape

            scales = [1.0]
            if multiscale:
                # Add fractional scales
                scales.extend([0.75, 0.5, 0.25])
                # Add "shortest dimension to tile_size" scale
                min_dim = min(h_orig, w_orig)
                if min_dim > tile_size:
                    scales.append(tile_size / min_dim)

            # Remove duplicates and scales that make the image too small
            scales = sorted({s for s in scales if s > 0}, reverse=True)
            scales = [
                s
                for s in scales
                if int(h_orig * s) >= tile_size and int(w_orig * s) >= tile_size
            ]

            name_base = img_path.stem

            for scale in scales:
                s_label = f"s{int(scale * 100):03d}"
                # Task Resumption Check
                if (str(img_path), s_label) in processed_tasks:
                    continue

                if scale == 1.0:
                    img_np = full_img_np
                else:
                    new_w = int(w_orig * scale)
                    new_h = int(h_orig * scale)
                    img_np = cv2.resize(
                        full_img_np, (new_w, new_h), interpolation=cv2.INTER_CUBIC
                    )

                h, w, _ = img_np.shape

                for y in range(h // tile_size):
                    for x in range(w // tile_size):
                        y0, y1 = y * tile_size, (y + 1) * tile_size
                        x0, x1 = x * tile_size, (x + 1) * tile_size
                        rgb_tile = img_np[y0:y1, x0:x1]
                        gray_tile = cv2.cvtColor(rgb_tile, cv2.COLOR_RGB2GRAY)

                        # Tiered Filter (Track results for CSV)
                        e = entropy(gray_tile)
                        lv = laplacian_variance(gray_tile)
                        ge = gradient_energy(gray_tile)
                        bl = blockiness(gray_tile)
                        nr = noise_ratio(gray_tile)
                        ar = aliasing_ratio(gray_tile)

                        if e < thresholds["entropy_min"]:
                            continue
                        if not (
                            thresholds["lap_var_min"] < lv < thresholds["lap_var_max"]
                        ):
                            continue
                        if ge < thresholds["grad_energy_min"]:
                            continue
                        if bl > thresholds["blockiness_max"]:
                            continue
                        if nr > thresholds["noise_ratio_max"]:
                            continue

                        queue.put(
                            {
                                "data": rgb_tile,
                                "name": f"{name_base}_{s_label}_t{y}_{x}.png",
                                "path": str(img_path),
                                "metrics": [e, lv, ge, bl, nr, ar],
                            }
                        )
        return True
    except Exception as e:
        print(f"Error Ingesting {img_path}: {e}")
        return False


def consumer_worker(
    queue: mp.Queue,
    model_path: str,
    out_dir: Path,
    csv_path: Path,
    threshold: float,
    tile_size: int,
    batch_size: int,
) -> None:
    try:
        os.nice(15)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = ICNet(is_pretrain=False, size1=tile_size, size2=tile_size // 2)
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        model.to(device).eval()
        if device == "cuda":
            model.half()

        # Only write headers if the file does not exist
        file_exists = csv_path.exists()

        # Keep handle open to reduce metadata overhead on HDDs
        with open(csv_path, "a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            if not file_exists:
                writer.writerow(
                    [
                        "tile_name",
                        "complexity_score",
                        "entropy",
                        "lap_var",
                        "grad_energy",
                        "blockiness",
                        "noise_ratio",
                        "aliasing",
                        "source",
                    ]
                )

            mean = (
                torch.tensor([0.485, 0.456, 0.406], device=device)
                .view(1, 3, 1, 1)
                .half()
            )
            std = (
                torch.tensor([0.229, 0.224, 0.225], device=device)
                .view(1, 3, 1, 1)
                .half()
            )

            buffer = []
            finished = 0
            total = 0

            while True:
                try:
                    item = queue.get(timeout=2)
                    if item == "DONE":
                        finished += 1
                    else:
                        buffer.append(item)
                except Empty:
                    if finished > 0:
                        break
                    continue

                if len(buffer) >= batch_size or (finished > 0 and buffer):
                    batch_imgs = [torch.from_numpy(b["data"]) for b in buffer]
                    tensors = (
                        torch.stack(batch_imgs).to(device).permute(0, 3, 1, 2).half()
                        / 255.0
                        - mean
                    ) / std

                    with torch.no_grad():
                        scores, _ = model(tensors)
                        scores = scores.flatten().cpu().float().numpy().tolist()

                    for b, score in zip(buffer, scores, strict=False):
                        m = b["metrics"]
                        writer.writerow(
                            [
                                b["name"],
                                f"{score:.6f}",
                                f"{m[0]:.4f}",
                                f"{m[1]:.2f}",
                                f"{m[2]:.4f}",
                                f"{m[3]:.4f}",
                                f"{m[4]:.4f}",
                                f"{m[5]:.4f}",
                                b["path"],
                            ]
                        )
                        if score >= threshold:
                            cv2.imwrite(
                                str(out_dir / b["name"]),
                                cv2.cvtColor(b["data"], cv2.COLOR_RGB2BGR),
                            )

                    # Flush to disk sporadically
                    csv_file.flush()
                    total += len(buffer)
                    buffer = []

        print(f"\nIngest Pass Complete. Processed {total} tiles.")
    except Exception:
        print("\nFATAL: Consumer Worker failed!")
        traceback.print_exc()


# ========================= MAIN =========================


def main() -> None:
    parser = argparse.ArgumentParser(description="LUCID v2 - Step 01: Ingest")
    parser.add_argument("--input", required=True, help="Input raw dataset")
    parser.add_argument("--output", required=True, help="Output workspace (LUCID Root)")
    parser.add_argument("--icnet", required=True, help="Path to complexity.pth")
    parser.add_argument(
        "--threshold", type=float, default=0.45, help="Complexity target"
    )
    parser.add_argument("--tile_size", type=int, default=512, help="Tile size")
    parser.add_argument("--workers", type=int, default=os.cpu_count() - 2)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument(
        "--resume", action="store_true", help="Skip images already in the score CSV"
    )
    parser.add_argument(
        "--multiscale", action="store_true", help="Enable multi-scale tiling"
    )

    args = parser.parse_args()
    in_dir = Path(args.input)
    dataset_name = (
        in_dir.name
        if in_dir.name not in ["HR", "images", "train"]
        else in_dir.parent.name
    )

    out_root = Path(args.output)
    target_out = out_root / dataset_name / "tiles"
    target_out.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / dataset_name / f"{dataset_name}_scores.csv"

    # Discovery
    valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
    images = [p for p in in_dir.rglob("*") if p.suffix.lower() in valid_exts]

    if args.resume and csv_path.exists():
        print(f"Resuming: Checking {csv_path.name} for processed tasks...")
        processed_tasks = set()  # (source_path, scale_label)
        try:
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Extract scale from tile_name: name_sXXX_tY_X.png
                    # or handle legacy names without scale
                    name = row["tile_name"]
                    scale_label = "s100"
                    if "_s" in name and "_t" in name:
                        parts = name.split("_")
                        for p in parts:
                            if p.startswith("s") and p[1:].isdigit():
                                scale_label = p
                                break
                    processed_tasks.add((row["source"], scale_label))

            # We filter images later in the producer, but we can't easily skip images here
            # if they have SOME missing scales. We'll pass the processed_tasks to the producer.
            print(
                f"Loaded {len(processed_tasks)} already completed (image+scale) tasks."
            )
        except Exception as e:
            print(f"Warning: Could not parse existing CSV for resume: {e}")
            processed_tasks = set()
    else:
        processed_tasks = set()

    if not images:
        print("No images found.")
        return

    print(f"--- LUCID v2 Ingest: {dataset_name} ---")
    print(f"Safety: System priority lowered. Workers: {args.workers}")

    ctx = mp.get_context("spawn")
    results_queue = ctx.Manager().Queue(maxsize=1000)

    consumer = ctx.Process(
        target=consumer_worker,
        args=(
            results_queue,
            args.icnet,
            target_out,
            csv_path,
            args.threshold,
            args.tile_size,
            args.batch,
        ),
    )
    consumer.start()

    try:
        with ctx.Pool(processes=args.workers) as pool:
            task_args = [
                (
                    p,
                    args.tile_size,
                    results_queue,
                    THRESHOLDS,
                    args.multiscale,
                    processed_tasks,
                )
                for p in images
            ]
            for _ in tqdm(
                pool.imap_unordered(producer_worker, task_args),
                total=len(images),
                desc="Ingesting",
            ):
                pass
    except KeyboardInterrupt:
        print("\nInterrupted. Cleaning up...")
    finally:
        results_queue.put("DONE")
        consumer.join(timeout=30)
        if consumer.is_alive():
            consumer.terminate()


if __name__ == "__main__":
    main()
