# 💎 LUCID v2: Dataset Filtering for High-Fidelity SISR
**L**earnable **U**nder-sampling **C**onsistency & **I**ntegrity **D**iscovery

Developed by **Philip Hofmann**, LUCID filters multi-million image datasets (ImageNet, PASS, LSDIR, DF2K) to extract the "Elite" 10% of training tiles for state-of-the-art Super-Resolution models.

---

## 🎯 The "Elite" Strategy
Our research shows that professional datasets like `diverseg-ip` have a much higher information density than standard ImageNet subsets. To beat the baselines, we use a two-pronged "Technical + Cognitive" approach:
1. **Technical Purity:** Rejection of sensor noise, compression artifacts, and blur.
2. **Cognitive Complexity:** Selection of the Top 10-15% of tiles based on structural information density (ICNet).
3. **Master Elite Expansion:** Using 512px context and texture-based deduplication to maximize information-per-pixel.

---

## 📦 The "Master Elite" Workflow Sequence (Step-by-Step)

Follow this sequence to build a 1,000,000+ tile dataset across multiple sources. Point all datasets to a single **LUCID Root** folder (e.g., `/media/phips/.../lucid/`).

### Step 1: Combined Filtering & Scoring (Per Dataset)
Run this for each dataset (LSDIR, DF2K, LIU4Kv2, etc.).
- **What it does:** Extracts 512px tiles, filters purity, and scores complexity in one pass.
- **Folder Structure:** It automatically creates subfolders for each dataset (e.g., `lucid/LSDIR/tiles/`).

```bash
# Example for LSDIR
python 04_combined_lucid.py \
    --input "/media/phips/.../LSDIR/HR" \
    --output "/media/phips/.../lucid" \
    --threshold 0.45 --tile_size 512

# Example for DF2K
python 04_combined_lucid.py \
    --input "/media/phips/.../DF2K/HR" \
    --output "/media/phips/.../lucid" \
    --threshold 0.45 --tile_size 512
```

### Step 2: Global Unified Auditor (Integrity + Profiles)
Run once on the **LUCID Root** folder.
- **What it does:** Performs a deep bitstream decode to catch truncated/corrupt images and standardizes color profiles (e.g., fixing "Palette Transparency" warnings) to 8-bit RGB. This is your "Quality Guarantee" step.

```bash
python 06_verify_profiles.py --input "/media/phips/.../lucid" --fix --workers 16
```

### Step 3: Global Diversity Audit (Deduplication)
Run once on the **LUCID Root** folder.
- **What it does:** Performs a "Global Texture Comparison." If the same texture pattern appears in LSDIR and DF2K, it keeps only one, ensuring maximum learning novelty.

```bash
python 05_diversity_audit.py --input "/media/phips/.../lucid" --threshold 0.96 --move_redundant
```

### Step 4: Master Finalization & Traceability
Run once on the **LUCID Root** folder.
- **What it does:** Collects and renames all files into a single sequence (e.g., `1_lsdir.png`, `14500_df2k.png`) and generates a traceability CSV.

```bash
python 07_finalize_dataset.py \
    --input "/media/phips/.../lucid" \
    --output "/media/phips/.../MASTER_ELITE" \
    --move
```

---

## 🚀 All Toolkit Scripts

| Script | Purpose | When to use? |
|--------|---------|--------------|
| `01_signal_filter.py` | Technical Purity (Stage 1) | Legacy/Standard Pipeline |
| `02_complexity_score.py` | Complexity Scoring (Stage 2) | Legacy/Standard Pipeline |
| `03_elite_selection.py` | Final Data Aggregation | Legacy/Standard Pipeline |
| **`04_combined_lucid.py`** | **Unified Master Expansion** | **Recommended for all new filtering** |
| **`05_diversity_audit.py`** | **Texture Deduplication** | After expansion, before finalization |
| **`06_verify_profiles.py`** | **Color Space Fixer** | After filtering, catches PIL warnings |
| **`07_finalize_dataset.py`** | **Master Tracking & Renaming** | The very last step |

---

## 📊 Understanding the Metrics

| Metric | Threshold | Why? |
|--------|-----------|------|
| **ICNet Score (256px)** | **> 0.50** | Matches visual density of professional datasets (Diverseg). |
| **ICNet Score (512px)** | **> 0.45** | Corrected for 4x area averaging in 512px tiles. |
| **Cosine Similarity** | **< 0.96** | Prunes redundant textures for maximum learning novelty. |
| **Noise Ratio** | **< 0.60** | Balance between sharp detail and JPEG artifact removal. |

---

*LUCID: Mathematical integrity and structural diversity for the next generation of SISR models.*
