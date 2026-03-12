# LUCID: Mathematical Integrity for SISR Datasets

## Core Idea
Move beyond human-biased quality metrics (MOS, NIQE, etc.) to what actually helps super-resolution networks learn. Focus: **structural diversity and novel patterns** over mere aesthetic appeal.

## The 3-Step Pipeline

### 1. INGEST: Filter + Score + Verify
```bash
python 01_ingest.py \
  --input /path/to/raw_images \
  --output /path/to/lucid_workspace \
  --icnet /path/to/complexity.pth
```
- Converts to 512px RGB tiles
- Standardizes format & checks corruption
- Scores complexity via ICNet (human-perception free)

### 2. DEDUPE: Diversity Audit
```bash
python 02_dedupe.py --input /path/to/lucid_workspace
```
- Removes redundant textures using ResNet18 fingerprints
- Keeps only unique, informative patterns
- Critical for preventing overfitting

### 3. FINALIZE: Consolidate & Append
```bash
# First run (creates master set)
python 03_finalize.py \
  --input /path/to/lucid_workspace \
  --output /path/to/MASTER_ELITE \
  --move

# Later runs (append to existing)
python 03_finalize.py \
  --input /path/to/new_workspace \
  --output /path/to/MASTER_ELITE \
  --move
```
- Creates sequentially named tiles: `00001_dataset.png`
- Maintains `master_lineage.csv` for full traceability
- Uses `os.nice(15)` to keep system responsive

## Personal Journey & Evolution

My path to LUCID came from years of experimenting with SISR dataset creation, evolving from IQA-dependent approaches to mathematically principled curation:

- **Early IQA Exploration**: Questioned metric limitations on PyIQA  
  [Issue #247](https://github.com/chaofengc/IQA-PyTorch/issues/247) |  
  [Issue #182](https://github.com/chaofengc/IQA-PyTorch/issues/182)

- **BHI Dataset**: First principled SISR curation attempt  
  [HuggingFace Dataset](https://huggingface.co/datasets/Phips/BHI) |  
  [Methodology Blog](https://huggingface.co/blog/Phips/bhi-filtering)

- **BHI100 Validation**: Standardized evaluation addressing downscaling inconsistencies (Pillow vs MATLAB)  
  [Results Code](https://github.com/Phhofm/bhi100-sisr-iqa-metrics) |  
  [Live Dashboard](https://phhofm.github.io/bhi100-sisr-iqa-metrics/)

- **PSISRD Validation**: Comprehensive IQA metric evaluation (125+ metrics)  
  [HuggingFace Dataset](https://huggingface.co/datasets/Phips/PSISRD_val125)

**Key Insights from Journey**:
1. **Computational infeasibility**: Filtering with thousands of IQA models revealed days-long processing times for large datasets
2. **Conceptual mismatch**: Human-pleasing metrics (MOS, NIQE, etc.) don't necessarily optimize SR network learning - observed through inconsistent "best" checkpoint selections across different metrics
3. **Degradation sensitivity**: Noticed that even standard datasets like Urban100 gave different results based on downscaling method (Pillow vs MATLAB), motivating standardized validation sets
4. **Key realization**: What helps SISR training isn't what looks pleasing to humans, but what provides meaningful structural gradients during optimization

This led to LUCID's core premise: **Focus on degradation-free, structurally novel data that provides consistent learning signals rather than human-perception scores**.

## Key Principles

### Why LUCID?
- **Problem**: Human perception scores prioritize "pretty" textures, not SR-useful ones
- **Solution**: Mathematical measures of information density (ICNet complexity) and structural novelty (ResNet18 deduplication)
- **Goal**: Maximize Urban100 validation performance through better data curation, not just more data

### Ideology
- **Anti-Bias**: Explicitly avoids metrics trained on human opinion (MOS, NIQE, BIQE, etc.)
- **Focus**: What provides meaningful gradients during HAT-M training - verified through empirical testing
- **Output**: Mathematically sound dataset with complete provenance for ablation studies

### Naming
**LUCID** = Lucent Understanding via Correlation and Independence in Data  
- *Lucent*: Clear, transparent, radiating light (clarity of purpose) - from original concept as "L**earnable** **U**nder-sampling **C**onsistency & **I**ntegrity **D**iscovery"  
- *Understanding*: Structural comprehension over surface appeal  
- *Correlation*: Identifying meaningful relationships in data (complexity scoring via ICNet)  
- *Independence*: Eliminating redundancy through deduplication (ResNet18 fingerprints)  

## Performance Tips
- **Train from SSD/NVMe**: Never train directly from HDD (seek time kills GPU utilization)
- **USB Optimization**: If using external drive:
  - Increase workers (`--workers 4`) and chunk size (`--chunk-size 5000`)
  - Consider processing on internal SSD first, then copying results
- **Resume Safety**: All scripts safely continue after interruption

## Expected Output
- `MASTER_ELITE/` folder with uniformly named tiles
- `master_lineage.csv` mapping each tile to source, original name, and path
- Dataset ready for HAT-M or similar SISR training pipelines

## Validation Approach
1. Create dataset with LUCID pipeline
2. Train HAT-M standard configuration
3. Compare Urban100 validation curves against baseline (e.g., DIV2K, DiverSEG-IP)
4. Expect: Faster convergence and/or higher peak PSNR/SSIM

## Key Learnings from BHI Work
From the BHI filtering experiments ([Blog Post](https://huggingface.co/blog/Phips/bhi-filtering)):
- **Blockiness < 30**: Critical for handling JPG compression artifacts detrimental to SR training
- **HyperIQA >= 0.2**: Removing only the worst tiles (bottom 20%) improved metrics more than keeping only the best
- **IC9600 >= 0.4**: Higher complexity filtering showed benefits, though optimal threshold varied by dataset
- **Surprising finding**: Models trained on 69% reduced datasets (BHI-filtered DF2K) often outperformed full datasets in PSNR/SSIM/DISTS metrics
- **Dataset quantity matters**: Extreme reduction (like ImageNet-BHI at 2.3%) hurt performance, suggesting ideal retention rate exists

## Philosophical Foundation
LUCID represents the culmination of realizing that:
1. SISR networks optimize mathematical loss functions, not human opinion scores
2. Degradation-free, structurally novel data provides more consistent learning signals
3. Diversity prevents overfitting to common textures
4. Traceability enables scientific ablation studies
5. System safety (`os.nice(15)`) allows background processing without disrupting workflow

This approach shifts focus from "What looks good?" to "What helps the network learn best?"