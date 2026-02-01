# LUCID: Dataset Filtering for High-Fidelity SISR
**L**earnable **U**nder-sampling **C**onsistency & **I**ntegrity **D**iscovery

Developed by **Philip Hofmann**, LUCID is the professional successor to the **BHI** and **PSISRD** filtering methodologies.

---

## 🧠 Philosophy: Integrity over Aesthetics
SISR is fundamentally a mathematical reconstruction task. Traditional filtering often relies on human-centric IQA (e.g., BRISQUE, HyperIQA, ARNIQA) which focuses on how "good" or "sharp" an image looks to a human eye.

LUCID rejects this "human bias" for three core reasons:
1. **Oversharpening is Noise:** Human perception favors sharp edges, but these are often the result of "illegal" sharpening filters in source data. Training on these introduces artifacts into the model.
2. **Technically Clean > Eye-Pleasing:** A model trains on gradients and signal truth. By focusing on **Technical Integrity** (aliasing, entropy, consistency), we provide cleaner supervision that translates directly to higher benchmark PSNR.
3. **Speed for the Millions:** Methods like PSISRD used 10+ deep learning models, making it impossible to filter datasets like ImageNet or PASS in reasonable time. LUCID is 25x-50x faster, designed to "discover" the best data in multi-million image sets.


### Stage 1: Signal Integrity (`lucid_stage1.py`)
Filters tiles based on raw signal statistics to remove technically flawed data before it reaches the GPU.
*   **Reject:** Flat/empty tiles (Low Entropy).
*   **Reject:** Blurry or over-sharpened images (Laplacian Variance).
*   **Reject:** JPEG/Compression artifacts (Blockiness).
*   **Reject:** Moiré/Sampling errors (Aliasing).
*   **Reject:** Sensor Noise (Noise Ratio).

### Stage 2: Learnable Consistency (`lucid_stage2.py`)
Uses a lightweight "SR Probe" network to ensure the mapping between LR and HR is mathematically stable and consistent.
*   **Test:** Can the probe reconstruct the original HR from a downscaled version?
*   **Keep:** Only images with high reconstruction fidelity (PSNR). This ensures the dataset contains only predictable, learnable patterns.

---

## ⚡ High-Performance Features
- **GPU Batching:** Stage 2 processes 32+ tiles simultaneously.
- **Multiprocessing:** Stage 1 utilizes all CPU cores with a tiered "early exit" gate for maximum throughput.
- **Async I/O:** Background threads handle file operations to keep the GPU/CPU fully saturated.
- **Vectorized Math:** All metrics are optimized with NumPy/OpenCV.

---

## 🚀 Quick Start
Everything is controlled through the `lucid.py` orchestrator.

### 1. Train the Probe
Train a small network on your "target" distribution (e.g., DF2K).
```bash
python lucid.py train --train "/path/to/DF2K_HR" --output sr_probe.pth
```

### 2. Run the Full Pipeline
Recommened for most users. This runs Stage 1 (Signal) and Stage 2 (Consistency) sequentially.
```bash
python lucid.py run-all --input "/path/to/raw_data" --output "./lucid_final" --weights sr_probe.pth --tile_size 256
```
*Use `--tile_size 256` for small source images (e.g. PASS).*

### 3. Manual Steps (Optional)
If you want granular control:
```bash
# Stage 1 Only
python lucid.py stage1 --input "/in" --output "/out1" --tile_size 256 --workers 8

# Stage 2 Only (with CSV logging for manual review)
python lucid.py stage2 --input "/out1" --output "/final" --weights sr_probe.pth --csv results.csv
```
---
*LUCID is designed to be lean, fast, and free of human perceptual bias, focusing purely on reconstruction integrity.*
