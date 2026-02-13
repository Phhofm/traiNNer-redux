# 💎 LUCID v2: Simplified Master Elite Pipeline
**The High-Performance Toolkit for SISR Dataset Engineering**

LUCID v2 is a consolidated, 3-step pipeline designed for massive-scale filtering, scoring, and deduplication. It is built for system safety (low priority), speed, and eternal expandability.

---

## 📦 The "Big 3" Workflow

### Step 01: Ingest (Filter + Score + Verify)
**Script:** `01_ingest.py`
Process raw images (LSDIR, DF2K, etc.) into verified, 512px Elite tiles. It standardizes everything to 8-bit RGB and checks for bitstream corruption automatically.

```bash
python 01_ingest.py \
    --input /path/to/RAW_SET \
    --output /path/to/lucid_workspace \
    --icnet /path/to/complexity.pth
```

### Step 02: Dedupe (Diversity Audit)
**Script:** `02_dedupe.py`
Removes redundant textures using ResNet18 fingerprints. This ensures your model only learns from unique, high-novelty patterns.

```bash
python 02_dedupe.py --input /path/to/lucid_workspace
```

### Step 03: Finalize (Consolidate & Append)
**Script:** `03_finalize.py`
The final integration step. It moves verified tiles into your global `MASTER_ELITE` folder with sequential numbering and perfect source traceability.

```bash
# First dataset run (Initialization)
python 03_finalize.py --input /path/to/lucid_workspace --output /path/to/MASTER_ELITE --move

# Subsequent datasets (Append Mode)
# It will auto-detect where the last set ended (e.g., 10405) and start at 10406.
python 03_finalize.py --input /path/to/lucid_workspace_2 --output /path/to/MASTER_ELITE --move
```

---

## 🛡️ System Safety
All scripts automatically run with **`os.nice(15)`**. This means you can keep training, browsing, or working while the pipeline runs in the background. It will use all available processing power without freezing your UI.

---

## 📊 Standard Master Elite Thresholds
- **Complexity:** 0.45 (512px)
- **Diversity:** 0.96 (Cosine Similarity)
- **Signal:** Standard mathematical purity (Noise, Blur, Aliasing)

*LUCID: Mathematical integrity and structural diversity for the next generation of SISR models.*
