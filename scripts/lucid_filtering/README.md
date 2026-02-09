# 💎 LUCID: Dataset Filtering for High-Fidelity SISR
**L**earnable **U**nder-sampling **C**onsistency & **I**ntegrity **D**iscovery

Developed by **Philip Hofmann**, LUCID filters multi-million image datasets (ImageNet, PASS) to extract the highest-quality training tiles for Super-Resolution.

---

## 🎯 Goal: Beat the Baselines
The objective is to produce a filtered dataset that achieves **higher PSNR/SSIM** on benchmarks like Urban100 than unfiltered datasets (e.g., Diverseg-ip).

---

## 📋 Quick Start: Recommended Pipeline

```
1. Stage 1         →  Remove blur, flat, artifacts (Technical Gate)
2. Score Complexity →  ICNet scoring (outputs CSV)
3. Score PSNR      →  Optional consistency check (outputs CSV)
4. Manual Review   →  Choose thresholds based on statistics
5. Copy Selection  →  Create final training dataset
```

> [!TIP]
> **Zero-Lag Background Workflow:** All scripts automatically run at ultra-low priority (`os.nice(15)`). You can keep them running in the background while you continue your normal workflow without any desktop lag or SSH delays.

---

## 🔧 Scripts Reference

### Stage 1: Technical Filtering (`lucid_stage1.py`)
Removes technically defective images: blur, flat regions, JPEG artifacts.

```bash
# Process images and delete originals automatically to save space
python lucid_stage1.py \
    --input /path/to/raw/images \
    --output /path/to/stage1_output \
    --tile_size 256 \
    --delete_input
```

**Default Thresholds (recommended):**
- `entropy_min: 5.5` — Rejects flat/empty regions
- `lap_var_min: 100` — Rejects blur
- `lap_var_max: 8000` — Allows sharp detail
- `blockiness_max: 40` — Rejects JPEG artifacts

---

### Step 2: ICNet Complexity Scoring (`icnet_score_only.py`)
Scores all tiles for information density. **Does not filter—outputs CSV only.**

```bash
python icnet_score_only.py \
    --input /path/to/stage1_output \
    --icnet ../../datasets/preparation/complexity/complexity.pth \
    --csv complexity_scores.csv
```

**Output:** Complete CSV with all image paths and complexity scores, plus statistics:
- Mean, median, max, min scores
- Suggested thresholds for Top 10%, 20%, 25%, 30%, 50%
- Score distribution histogram

---

### Step 3 (Optional): PSNR Consistency Scoring (`psnr_score_only.py`)
Evaluates the "Stage 2" consistency metric. Use this to decide if rejecting low-PSNR tiles helps.

```bash
python psnr_score_only.py \
    --input /path/to/stage1_output \
    --csv psnr_scores.csv \
    --scale 4
```

**Purpose:** Low-PSNR tiles are "hard"—they could be:
- **Complex detail** (good for training) → KEEP
- **Corrupted garbage** (bad for training) → REJECT

Manually inspect the lowest-scoring tiles to decide your threshold.

---

### Step 4: Manual Threshold Selection
After scoring, review the statistics printed by each script:

```
=== COMPLEXITY SCORE STATISTICS ===
Top 25% (82101 images): score >= 0.4512
Top 20% (65681 images): score >= 0.4789
```

Choose your cutoff based on:
1. How many tiles you want
2. Visual inspection of borderline tiles

---

### Step 5: Copy Selected Images (`copy_by_score.py`)
Copies images that meet your threshold to a new folder.

```bash
# Copy top 25% by complexity
python copy_by_score.py \
    --csv complexity_scores.csv \
    --output /path/to/elite_dataset \
    --top_percent 25

# OR: Copy by absolute score threshold
python copy_by_score.py \
    --csv complexity_scores.csv \
    --output /path/to/elite_dataset \
    --min_score 0.45

# Preview without copying
python copy_by_score.py \
    --csv complexity_scores.csv \
    --output /path/to/elite_dataset \
    --top_percent 25 \
    --dry_run
```

---

## 🔬 Analysis Tools

### Dataset Gap Analysis (`analyze_dataset_gap.py`)
Compares complexity distributions between two datasets (e.g., your Elite set vs Diverseg).

```bash
python analyze_dataset_gap.py \
    --diverseg /path/to/diverseg-ip \
    --lucid /path/to/your_elite_dataset \
    --icnet ../../datasets/preparation/complexity/complexity.pth
```

---

## 🛟 Resilient Recovery (Disk Space Management)
Use these tools if you run out of disk space or need to resume an interrupted Stage 1.

### 1. Verify Integrity (`verify_tiles.py`)
Disk errors at 99% usage often create truncated (broken) images. This script purges them.
```bash
python scripts/lucid_filtering/verify_tiles.py \
    --input /path/to/tiles \
    --corrupted /path/to/corrupted_bin
```

### 2. Stream-Reclaim Space (`lucid_cleanup_source.py`)
This script maps tiles back to ImageNet source files. It deletes the "covered" sources to free up space, then gives you a resume list for the rest.
```bash
# 1. Identify and delete processed ImageNet files
python scripts/lucid_filtering/lucid_cleanup_source.py \
    --input /path/to/imagenet \
    --output /path/to/valid_tiles \
    --delete

# 2. Finish the remaining images
python scripts/lucid_filtering/lucid_stage1.py \
    /path/to/imagenet /path/to/tiles stats.csv \
    --file_list resume_stage1.txt
```

---

## 📊 Understanding the Metrics

### Stage 1 Metrics (Technical Quality)
| Metric | What It Detects | Good Values |
|--------|-----------------|-------------|
| Entropy | Information content | > 5.5 |
| Laplacian Variance | Sharpness | 100 - 8000 |
| Blockiness | JPEG artifacts | < 40 |
| Aliasing Ratio | Moiré/sampling errors | < 0.6 |
| Noise Ratio | Sensor noise | < 0.6 |

### ICNet Complexity Score
| Score Range | Interpretation |
|-------------|----------------|
| 0.0 - 0.2 | Very simple (gradients, sky) |
| 0.2 - 0.4 | Low detail |
| 0.4 - 0.5 | Moderate complexity |
| 0.5 - 0.7 | High detail (textures, patterns) |
| 0.7 - 1.0 | Very complex (Urban100-like) |

### PSNR Consistency Score
| PSNR | Interpretation |
|------|----------------|
| < 20 dB | Likely garbage or extreme artifacts |
| 20-25 dB | "Hard" tiles—inspect manually |
| 25-30 dB | Normal, clean tiles |
| > 30 dB | "Easy" tiles—smooth, simple |

---

## 💡 Pipeline Philosophy

### Why Skip Stage 2?
Our analysis showed that Stage 2 (PSNR consistency) was killing complex tiles:
- **Diverseg Mean Complexity:** 0.49 (Max: 0.91)
- **LUCID Stage2 Elite:** 0.42 (Max: 0.63)

Stage 2 kept "easy" tiles and rejected the "hard" detail we need for Urban100.

### Recommended Strategy
1. **Stage 1:** Keep defaults—removes only true garbage
2. **ICNet Scoring:** Primary selector—prioritizes information density
3. **PSNR Scoring:** Optional sanity check—reject only extreme outliers (< 18 dB)

---

## 🛡️ System Stability & Safety

Every script in the LUCID toolkit is designed to be a "good neighbor" on your system, allowing for long-running background processing without performance impact.

- **Background Friendly (Zero-Lag):** Scripts automatically run at `os.nice(15)` priority. This ensures they only use "leftover" CPU/GPU cycles, preventing system lag and keeping remote access (SSH/VS Code) perfectly responsive even during 100% load.
- **Graceful Interruption:** All scripts support safe interruption via `Ctrl+C`. The process will stop current work, flush remaining results to disk, and shut down cleanly without data loss or cryptic tracebacks.
- **Reliable Recovery:** Interrupted runs save their progress (partial CSVs or resume lists), allowing you to pick up exactly where you left off.

---

## 📁 File Structure

```
scripts/lucid_filtering/
├── lucid_stage1.py          # Technical filtering
├── icnet_score_only.py      # Complexity scoring (CSV output)
├── psnr_score_only.py       # PSNR scoring (CSV output)
├── copy_by_score.py         # Copy images by threshold
├── analyze_dataset_gap.py   # Compare dataset distributions
├── stage3_density_gate.py   # Legacy: combined scoring + copying
├── lucid_stage2.py          # Legacy: consistency filtering
├── lucid_prune.py           # Legacy: multi-metric pruning
└── README.md                # This file
```

---

*LUCID: Mathematical integrity for the next generation of SISR models.*
