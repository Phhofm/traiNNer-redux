#!/usr/bin/env python3
"""
LUCID v2: Step 01-CC0 - Turbo Stream Ingest
==========================================
Highly optimized script for processing massive Hugging Face datasets (PD12M, etc.).

Architecture:
1. Fetcher (Thread): Pulls items from HF and puts into raw_queue.
2. Ingestors (Pool): Pull from raw_queue, handle CPU decoding/tiling/filtering.
3. Consumer (GPU): Pulls from tile_queue, runs ICNet, saves to SSD.

Restores speed by parallelizing the network/CPU bottleneck.
"""

import argparse
import csv
import os
import sys
import threading
import traceback
from pathlib import Path
from queue import Empty

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image, ImageFile
from tqdm import tqdm

# Force PIL safety
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = False

# Increase Hugging Face Hub timeouts for massive datasets
os.environ["HF_HUB_READ_TIMEOUT"] = "120"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"

try:
    if mp.get_start_method() != "spawn":
        mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

# ICNet Setup
REPO_ROOT = Path(__file__).parent.parent.parent
ICNET_DIR = REPO_ROOT / "datasets" / "preparation" / "complexity"
sys.path.append(str(ICNET_DIR))

try:
    from ICNet import ICNet
except ImportError:
    print(f"FATAL: Could not import ICNet from {ICNET_DIR}")
    sys.exit(1)

# ========================= SIGNAL CONFIG =========================

THRESHOLDS = {
    "entropy_min": 5.2,
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


def ingest_worker(
    raw_queue: mp.Queue, tile_queue: mp.Queue, tile_size: int, multiscale: bool
) -> None:
    """CPU Worker: Decodes, resizes, tiles, and filters images."""
    try:
        os.nice(15)
        while True:
            try:
                item_pkg = raw_queue.get(timeout=5)
                if item_pkg == "DONE":
                    break

                idx, item = item_pkg

                # Fetch image data
                img_data = item.get("image") or item.get("jpg") or item.get("png")
                if img_data is None:
                    continue

                if not isinstance(img_data, Image.Image):
                    if isinstance(img_data, dict) and "bytes" in img_data:
                        import io

                        img_data = Image.open(io.BytesIO(img_data["bytes"]))
                    else:
                        continue

                if img_data.mode != "RGB":
                    img_data = img_data.convert("RGB")

                full_img_np = np.array(img_data)
                h_orig, w_orig, _ = full_img_np.shape

                scales = [1.0]
                if multiscale:
                    scales.extend([0.75, 0.5, 0.25])
                    min_dim = min(h_orig, w_orig)
                    if min_dim > tile_size:
                        scales.append(tile_size / min_dim)

                scales = sorted({s for s in scales if s > 0}, reverse=True)
                scales = [
                    s
                    for s in scales
                    if int(h_orig * s) >= tile_size and int(w_orig * s) >= tile_size
                ]

                for scale in scales:
                    if scale == 1.0:
                        img_np = full_img_np
                    else:
                        new_w, new_h = int(w_orig * scale), int(h_orig * scale)
                        img_np = cv2.resize(
                            full_img_np, (new_w, new_h), interpolation=cv2.INTER_CUBIC
                        )

                    h, w, _ = img_np.shape
                    s_label = f"s{int(scale * 100):03d}"

                    for y in range(h // tile_size):
                        for x in range(w // tile_size):
                            y0, y1 = y * tile_size, (y + 1) * tile_size
                            x0, x1 = x * tile_size, (x + 1) * tile_size
                            rgb_tile = img_np[y0:y1, x0:x1]
                            gray_tile = cv2.cvtColor(rgb_tile, cv2.COLOR_RGB2GRAY)

                            # Signal Filter
                            e = entropy(gray_tile)
                            if e < THRESHOLDS["entropy_min"]:
                                continue
                            lv = laplacian_variance(gray_tile)
                            if not (
                                THRESHOLDS["lap_var_min"]
                                < lv
                                < THRESHOLDS["lap_var_max"]
                            ):
                                continue
                            ge = gradient_energy(gray_tile)
                            if ge < THRESHOLDS["grad_energy_min"]:
                                continue
                            bl = blockiness(gray_tile)
                            if bl > THRESHOLDS["blockiness_max"]:
                                continue
                            nr = noise_ratio(gray_tile)
                            if nr > THRESHOLDS["noise_ratio_max"]:
                                continue
                            ar = aliasing_ratio(gray_tile)

                            # Put into GPU queue
                            tile_queue.put(
                                {
                                    "data": rgb_tile,
                                    "name": f"stream_{idx}_{s_label}_t{y}_{x}.png",
                                    "idx": str(idx),
                                    "metrics": [e, lv, ge, bl, nr, ar],
                                }
                            )

                # Cleanup
                del img_data
                del full_img_np
                # Regular gc is handled automatically by Python
                # but we can trigger it in main if needed.

            except Empty:
                continue
            except Exception as e:
                print(f"CPU Worker Error on index {idx}: {e}")
                continue
    except Exception as e:
        print(f"FATAL CPU Worker exit: {e}")


def consumer_worker(
    tile_queue: mp.Queue,
    model_path: str,
    out_dir: Path,
    csv_path: Path,
    threshold: float,
    tile_size: int,
    batch_size: int,
) -> None:
    """GPU Worker: Runs ICNet on batches of tiles and saves to disk."""
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

        file_exists = csv_path.exists()
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
                        "source_idx",
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
            finished = False
            total_saved = 0

            while True:
                try:
                    item = tile_queue.get(timeout=2)
                    if item == "DONE":
                        finished = True
                    else:
                        buffer.append(item)
                except Empty:
                    if finished:
                        break
                    continue

                if len(buffer) >= batch_size or (finished and buffer):
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
                                b["idx"],
                            ]
                        )
                        if score >= threshold:
                            cv2.imwrite(
                                str(out_dir / b["name"]),
                                cv2.cvtColor(b["data"], cv2.COLOR_RGB2BGR),
                            )
                            total_saved += 1

                    csv_file.flush()
                    buffer = []

        print(f"\nStream Ingest Segment Complete. Saved {total_saved} tiles.")
    except Exception:
        print("\nFATAL: Consumer Worker failed!")
        traceback.print_exc()


# ========================= MAIN =========================


def fetcher_thread(
    ds_shard, raw_queue, shard_skip, num_shards, shard_idx, pbar_ref
) -> None:
    """Fetcher: Streams from HF (specific shard) and fills the raw queue."""
    try:
        print(f"Fetcher Thread {shard_idx}: Starting (skipping {shard_skip} items)...")
        # Apply skip at shard level (much faster than global skip)
        if shard_skip > 0:
            ds_shard = ds_shard.skip(shard_skip)

        for i, item in enumerate(ds_shard):
            # Calculate global index: (items_in_shard * num_shards) + shard_idx
            actual_idx = ((shard_skip + i) * num_shards) + shard_idx
            raw_queue.put((actual_idx, item))
            pbar_ref[0].update(1)

        print(f"Fetcher Thread {shard_idx}: Finished.")
    except Exception as e:
        print(f"Fetcher Thread {shard_idx} Error: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LUCID v2 CC0 - Turbo Stream Ingest")
    parser.add_argument("--dataset", required=True, help="HF Dataset name")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", required=True, help="Output workspace (SSD)")
    parser.add_argument("--icnet", required=True)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--multiscale", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 4))

    args = parser.parse_args()
    dataset_name = args.dataset.split("/")[-1]
    out_root = Path(args.output)
    target_out = out_root / dataset_name / "tiles"
    target_out.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / dataset_name / f"{dataset_name}_scores.csv"

    last_idx = -1
    if args.resume and csv_path.exists():
        try:
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    idx = int(row["source_idx"])
                    last_idx = max(last_idx, idx)
            print(f"Resuming from index {last_idx + 1}")
        except:
            pass

    # Load Stream with Retries
    from datasets import load_dataset

    print(f"--- LUCID v2 CC0 Ingest: {args.dataset} ---")
    print(
        "Connecting to Hugging Face Hub (this may take 1-2 mins for large datasets)..."
    )

    ds = None
    for attempt in range(5):
        try:
            ds = load_dataset(args.dataset, split=args.split, streaming=True)
            break
        except Exception as e:
            if attempt < 4:
                print(
                    f"Connection Attempt {attempt + 1} failed: {e}. Retrying in 5s..."
                )
                import time

                time.sleep(5)
            else:
                print(f"FATAL: Could not connect to {args.dataset} after 5 attempts.")
                raise e

    # Resumption Parameters
    items_processed = last_idx + 1
    start_count = items_processed

    ctx = mp.get_context("spawn")
    raw_queue = ctx.Queue(maxsize=100)  # Buffer 100 raw images
    tile_queue = ctx.Queue(maxsize=1000)  # Buffer 1000 tiles

    # 1. Start GPU Consumer
    consumer = ctx.Process(
        target=consumer_worker,
        args=(
            tile_queue,
            args.icnet,
            target_out,
            csv_path,
            args.threshold,
            args.tile_size,
            args.batch,
        ),
    )
    consumer.start()

    # 2. Start CPU Ingestors
    ingestors = []
    for _ in range(args.workers):
        p = ctx.Process(
            target=ingest_worker,
            args=(raw_queue, tile_queue, args.tile_size, args.multiscale),
        )
        p.start()
        ingestors.append(p)

    # 3. Start Multi-Shard Fetchers (Parallel Network Threads)
    num_fetchers = 4  # 4 threads is usually enough to saturate bandwidth
    pbar = [tqdm(desc="Turbo-Streaming", initial=start_count)]
    fetcher_threads = []

    for f_idx in range(num_fetchers):
        # Calculate shard-specific skip
        shard_skip = items_processed // num_fetchers
        if f_idx < items_processed % num_fetchers:
            shard_skip += 1

        ds_shard = ds.shard(num_shards=num_fetchers, index=f_idx)
        t = threading.Thread(
            target=fetcher_thread,
            args=(ds_shard, raw_queue, shard_skip, num_fetchers, f_idx, pbar),
        )
        t.start()
        fetcher_threads.append(t)

    try:
        for t in fetcher_threads:
            t.join()

        # End of stream signals
        for _ in range(args.workers + 4):  # 4 is num_fetchers
            raw_queue.put("DONE")

        for p in ingestors:
            p.join()
        tile_queue.put("DONE")
        consumer.join(timeout=30)
    except KeyboardInterrupt:
        print("\nTurbo Interrupted. Cleaning up...")
    finally:
        for p in ingestors:
            if p.is_alive():
                p.terminate()
        if consumer.is_alive():
            consumer.terminate()


if __name__ == "__main__":
    main()
