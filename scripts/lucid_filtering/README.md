# 💎 LUCID: Dataset Filtering for High-Fidelity SISR
**L**earnable **U**nder-sampling **C**onsistency & **I**ntegrity **D**iscovery

Developed by **Philip Hofmann**, LUCID is the professional successor to BHI and PSISRD, designed to discover mathematically perfect training pairs within multi-million image datasets (ImageNet, PASS, etc.).

---

## 🧠 The Philosophy: Integrity over Aesthetics
Traditional IQA (HyperIQA, BRISQUE) focuses on "beauty." LUCID focuses on **Signal Truth**.

SISR is a reconstruction task. For a model to learn efficiently, the High-Res details must be logically consistent with the Low-Res version. If a source image has hidden JPEG noise, illegal sharpening, or was previously upscaled, the "link" between LR and HR is broken. Training on this data confuses the model.

### Stage 1: Signal Cleanliness (`lucid_stage1.py`)
A high-speed CPU filter that removes technically flawed tiles before they touch the GPU.
- **Entropy:** Removes flat or empty regions.
- **Laplacian Variance:** Removes blur and illegal over-sharpening.
- **Blockiness:** Detects and rejects JPEG/compression artifacts.
- **Aliasing:** Detects moiré and sampling errors.
- **Noise Ratio:** Filters out high sensor noise.
- **NEW:** Every kept tile now logs these 6 metrics to `lucid_stage1_stats.csv` for deep dataset analysis.

### Stage 2: Consistency Integrity (`lucid_stage2.py`)
This is the "Mathematical Gate." It uses a lightweight **SR Probe** to test the integrity of each tile.
1. **Downsample:** The tile is downsampled (Bicubic).
2. **Reconstruct:** The SR Probe attempts to recover the original HR.
3. **Verify:** If the PSNR is high, the tile is **Consistent**. The mapping is predictable and clean.
4. **Reject:** If PSNR is low, the tile has a "Consistency Gap"—it contains information that doesn't follow the laws of your degradation model. These are "unreliable teachers" and are purged.

### 4. Elite Dataset Pruning (Optional)
Once you have your merged tile folder and Stage 2 logs, you can prune the dataset to keep only the mathematically "Elite" tiles (e.g., Top 10% or 25%). This is the secret for winning PSNR benchmarks like Urban100.

```bash
python lucid_prune.py \
    --img_dir "/path/to/tiles" \
    --out_dir "/path/to/elite_folder" \
    --master_csv "/path/to/lucid_elite_master.csv" \
    --s2_csv_paths "/path/to/stage2_psnr.csv" \
    --s1_csv_paths "/path/to/stage1_stats.csv" \
    --top_percent 25 \
    --rename_index
```

*   **Copying by default:** The script now performs a full physical copy of the images by default, making the output folder "distribution-ready" and portable.
*   **Separate Metadata:** The `--master_csv` argument allows you to store the dataset statistics anywhere, keeping your image folder clean.
*   **Index Renaming:** Use `--rename_index` to simplify filenames to `0.png`, `1.png`, etc., while maintaining the mapping in the Master CSV.

---

#### 🔬 The SR Probe Architecture
The included `SRProbeNet` is a minimalist, ultra-fast architecture spiritually and structurally similar to **ArtCNN**.
- **Lightweight Design:** It uses a 5x5 head, a 3-layer 32-channel body, and a PixelShuffle tail.
- **Why it works:** By using a "weak" but technically sound network, we create a **High Bar** for filtering. The probe will only reconstruct a tile perfectly if the LR-to-HR mapping is mathematically pure. This acts as a guarantee that your final High-Capacity model (like HAT or SwinIR) will receive 100% clean supervision.
- **Speed:** Capable of processing thousands of tiles per second on modern GPUs, enabling ImageNet-scale filtering in hours.

---

## 🛡️ Production Stability (Scaling to Millions)
LUCID is hardened specifically for large-scale production environments:
- **Disk-Safe Streaming:** Uses `--batch_images` (default 10k) to process in chunks. Temporary tiles are **automatically deleted** between batches, bounding your peak disk usage.
- **Batch Resumption:** Use `--start_batch X` to pick up exactly where you left off if a run is interrupted. Logs are automatically **appended**.
- **RAM Safe:** Uses generator-based loading to keep RAM usage < 400MB even for ImageNet-scale inputs.
- **Desktop Friendly:** Workers run with `os.nice(10)`, keeping your system responsive during 100% load.

---

## 🚀 Usage Guide

### 1. Train your "Mathematical Authority" (The Probe)
Train the probe on a small, perfect dataset (e.g., DF2K or LSDIR).
```bash
python lucid.py train --train "/path/to/DF2K_HR" --output sr_probe.pth
```

### 2. Run the Full Production Pipeline
The standard way to process millions of images safely.
```bash
# Processes in batches of 10k, with 256px tiles
python lucid.py run-all --input "/raw/data" --output "/filtered" --weights sr_probe.pth --temp "/fast/ssd/temp"
```

### 3. Monitoring & Analytics
- `lucid_stage1_stats.csv`: Detailed signal metrics for every tile that passed Stage 1.
- `lucid_stage2_psnr.csv`: Reconstruction scores (x2, x4) for the final survivors.

---
*LUCID: Mathematical integrity for the next generation of SISR models.*
