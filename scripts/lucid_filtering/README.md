# 💎 LUCID v2: Dataset Filtering for High-Fidelity SISR
**L**earnable **U**nder-sampling **C**onsistency & **I**ntegrity **D**iscovery

Developed by **Philip Hofmann**, LUCID filters multi-million image datasets (ImageNet, PASS) to extract the "Elite" 10% of training tiles for state-of-the-art Super-Resolution models.

---

## 🎯 The "Elite" Strategy
Our research shows that professional datasets like `diverseg-ip` have a much higher information density than standard ImageNet subsets. To beat the baselines, we use a two-pronged "Technical + Cognitive" approach:
1. **Technical Purity:** Rejection of sensor noise, compression artifacts, and blur.
2. **Cognitive Complexity:** Selection of the Top 10-15% of tiles based on structural information density (ICNet).

---

## 📋 The 3-Stage Pipeline

### Stage 1: Technical Signal Filtering
**Script:** `01_signal_filter.py` (formerly `lucid_stage1.py`)

Removes technically defective images (blur, flat zones, JPEG artifacts).
```bash
python 01_signal_filter.py \
    --input /path/to/raw/images \
    --output /path/to/tiles \
    --tile_size 256 \
    --delete_input
```
**Recommended Thresholds:**
- `noise_ratio_max: 0.60` — The "Gold Standard" for keeping high-frequency detail without JPEG noise.
- `entropy_min: 5.5` — Guarantees informative regions.
- `lap_var_min: 100` — Ensures sharpness.

---

### Stage 2: Complexity Scoring
**Script:** `02_complexity_score.py` (formerly `icnet_score_only.py`)

Scores tiles using the ICNet model to identify "Elite" structural density. **Outputs CSV only.**
```bash
python 02_complexity_score.py \
    --input /path/to/tiles \
    --icnet /path/to/complexity.pth \
    --csv global_scores.csv
```
**The Target:** Align your threshold with professionally filtered data.
- **Diverseg-IP Median:** ~0.54
- **Recommended Threshold:** **0.50+** (The Top 10-12% of ImageNet/PASS)

---

### Stage 3: Elite Selection
**Script:** `03_elite_selection.py` (formerly `copy_by_score.py`)

Aggregates the final dataset from the CSV scores.
```bash
python 03_elite_selection.py \
    --csv global_scores.csv \
    --output /path/to/elite_dataset \
    --min_score 0.50 \
    --move
```
*Use `--move` to reclaim disk space immediately.*

---

## 🛟 Utility Tools

### Verify Dataset Integrity
**Script:** `verify_dataset.py`
Purges truncated or corrupted images (essential after running near disk capacity).
```bash
python verify_dataset.py --input /path/to/elite_dataset
```

### Prune Disk Usage
**Script:** `prune_disk.py`
Advanced cleanup for multi-folder dataset management.

---

## 📊 Understanding the Metrics

| Metric | Threshold | Why? |
|--------|-----------|------|
| **ICNet Score** | **> 0.50** | Matches visual density of professional high-quality datasets. |
| **Noise Ratio** | **< 0.60** | Balance between capturing sharp detail and avoiding JPEG sensor noise. |
| **Entropy** | **> 5.5** | Ensures the tile isn't just a flat sky or solid color. |
| **Laplacian** | **> 100** | Standard gate for high-frequency sharpness. |

---

*LUCID: Mathematical integrity for the next generation of SISR models.*
