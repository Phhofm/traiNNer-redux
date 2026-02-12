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

## 📋 Standard 3-Stage Pipeline

### Stage 1: Technical Signal Filtering
**Script:** `01_signal_filter.py`
Removes technically defective images (blur, flat zones, JPEG artifacts).

### Stage 2: Complexity Scoring
**Script:** `02_complexity_score.py`
Scores tiles using the ICNet model to identify "Elite" structural density.

### Stage 3: Elite Selection
**Script:** `03_elite_selection.py`
Aggregates the final dataset from score CSVs into a unified folder.

---

## 🚀 The "Master Elite" Unified Pipeline (Recommended)

### 1. Combined Filtering & Scoring
**Script:** `04_combined_lucid.py`

Designed for massive external datasets (LSDIR, DF2K, LIU4Kv2) on slow external HDDs. It combines signal filtering and complexity scoring into a single memory-optimized stream, outputting only "Elite" 512px tiles to disk.

```bash
python 04_combined_lucid.py \
    --input /path/to/HR_images \
    --output /path/to/lucid_tiles \
    --icnet /path/to/complexity.pth \
    --threshold 0.45 \
    --tile_size 512
```

### 2. Diversity Audit & Deduplication
**Script:** `05_diversity_audit.py`

Uses ResNet18 feature fingerprints to ensure the dataset isn't dominated by repeating textures (e.g., thousands of identical brick wall patches). This maximizes the "Learning Novelty" of your dataset.

```bash
python 05_diversity_audit.py \
    --input /path/to/lucid_tiles \
    --threshold 0.96 \
    --move_redundant
```

---

## 📊 Understanding the Metrics

| Metric | Threshold | Why? |
|--------|-----------|------|
| **ICNet Score (256px)** | **> 0.50** | Matches the visual density of professional datasets. |
| **ICNet Score (512px)** | **> 0.45** | Corrected for "Area Averaging" in higher-resolution context. |
| **Cosine Similarity** | **< 0.96** | Prunes redundant textures while keeping unique structural nuances. |
| **Noise Ratio** | **< 0.60** | Balance between sharp detail and JPEG artifact removal. |
| **Entropy** | **> 5.5** | Ensures the tile isn't just a flat sky or solid color. |

---

## 🛟 Utility Tools

### Verify Dataset Integrity
**Script:** `verify_dataset.py`
Purges truncated or corrupted images (essential after running near disk capacity).

### Prune Disk Usage
**Script:** `prune_disk.py`
Advanced cleanup for multi-folder dataset management.

---

*LUCID: Mathematical integrity and structural diversity for the next generation of SISR models.*
