# 💎 LUCID v2: Simplified Master Elite Pipeline
**The High-Performance Toolkit for SISR Dataset Engineering**

LUCID v2 is a consolidated, 3-step pipeline designed for massive-scale filtering, scoring, and deduplication. It is built for system safety (low priority), speed, and eternal expandability.

---

## 🏁 The "Finish Line" Sequence (For Existing Tiles)
If you have already run the filtering/scoring and have folders of tiles, follow this final cleanup sequence:

### 1. Unified Audit (Integrity + Profiles)
**Script:** `00_audit.py`
Run this on your **Lucid Root** folder (the one containing LSDIR, DF2K, etc.). It will recursively fix "Palette Transparency" warnings and move any corrupted files to a separate folder.
```bash
python 00_audit.py --input "/media/phips/.../lucid" --fix
```

### 2. Global Deduplication
**Script:** `02_dedupe.py`
Run this on the same **Lucid Root**. It will find redundant textures *across all datasets* to ensure your 1M tiles are unique.
```bash
python 02_dedupe.py --input "/media/phips/.../lucid"
```

### 3. Master Finalization
**Script:** `03_finalize.py`
This consolidates everything into your final **MASTER_ELITE** folder with sequential re-indexing and traceability.
```bash
python 03_finalize.py \
    --input "/media/phips/.../lucid" \
    --output "/media/phips/.../MASTER_ELITE" \
    --move
```

---

## 📦 The "Big 3" Workflow (For New Raw Data)

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

## 💡 Training Performance Tip
**Never train directly from a physical HDD.**
SISR dataloaders perform heavy random-access reads. A physical hard drive's seek time will bottleneck your GPU, leading to extremely low utilization and 5-10x slower training times.

**Recommended:** Finalize your dataset on the HDD for storage, then copy the `MASTER_ELITE` folder to an **SSD (NVMe or SATA)** for the actual training phase.

---

*LUCID: Mathematical integrity and structural diversity for the next generation of SISR models.*
