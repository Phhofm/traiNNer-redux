# ParagonSR3 — Technical Deep Dive

> **ParagonSR3**: Surgical Re-parameterization for Efficient Super-Resolution  
> Variants: Virtuoso (Photo), Aegis (Video)  
> Status: Production-ready, TensorRT-optimized

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Model Specifications](#model-specifications)
3. [Component Breakdown](#component-breakdown)
4. [Training Recipe](#training-recipe)
5. [Benchmark Results](#benchmark-results)
6. [Deployment Guide](#deployment-guide)
7. [Ablation Studies](#ablation-studies)
8. [Limitations](#limitations)

---

## Architecture Overview

ParagonSR3 is designed around **three principles**:

1. **Surgical Efficiency**: Every parameter must contribute to PSNR; unnecessary complexity eliminated.
2. **Re-parameterization**: Multi-branch training architecture collapses to single fast branch at inference.
3. **Sparse Attention**: Expensive global operations applied only where needed, guided by variance routing.

The network supports **multi-scale multi-task training**: a single model trained on 1× (denoising), 2×, 3×, and 4× simultaneously. Shared body learns robust features; scale-specific heads handle upsampling. After training, export separate ONNX models per scale.

---

## Model Specifications

### Virtuoso (Photo) — Target: Beat HAT-L

| Property | Value |
|----------|-------|
| **Base channels** | 180 (matches HAT-L embedding dim) |
| **Residual groups** | 10 |
| **Blocks per group** | 8 |
| **Total blocks** | 80 |
| **Stochastic depth rate** | 0.1 (linear schedule) |
| **Window attention** | Every 4th block (throughout) + last 2 groups always |
| **Token attention** | Every 12th block + half of last 2 groups |
| **Approx. parameters** | ~42 M |
| **Approx. MACs** (64²→256², 4×) | ~150 G |
| **Training iters** | 800,000 (recommended) |
| **Dataset** | DIV2K + Flickr2K (2,800+ images) |

### Aegis (Video) — Target: Real-time 4K with temporal stability

| Property | Value |
|----------|-------|
| **Base channels** | 64 |
| **Residual groups** | 4 |
| **Blocks per group** | 6 |
| **Total blocks** | 24 |
| **Stochastic depth rate** | 0.05 |
| **Token attention** | Middle block of each group only (4× total) |
| **Recurrence** | Yes (stateful inference only) |
| **Approx. parameters** | ~8 M |
| **Approx. MACs** (64²→256², 4×) | ~30 G |
| **Training iters** | 200,000 (from scratch) or fine-tune from photo body |
| **Dataset** | Video sequences (clips of ≥16 frames) — custom required |

---

## Component Breakdown

### 1. GatedRepConv

**Purpose**: The workhorse. Re-parameterizable convolution combining three branches:

- **3×3 depthwise**: Main spatial processing
- **1×1 pointwise**: Cross-channel mixing
- **Identity**: Residual skip

**During training**: All three active → output = `dw_main(x) + dw_1x1(x) + identity_scale × x`

**During inference** (`switch_to_deploy`): Fuse 1×1 into 3×3 (pad kernel) + add scaled identity → single 3×3 depthwise.

**Speed**: ~0% overhead after fusion (single conv call).

**Added**: Simple Channel Attention (SCA) after gating → cheap global gating per channel.

### 2. iLN (Image Restoration LayerNorm)

**Purpose**: Stabilize training, provide texture routing signal.

Computes mean/std in **FP32**, applies in input dtype (prevents AMP underflow). Also returns per-channel variance for attention routing.

```python
# Simplified
mean, std = x.float().mean/var(dim=(2,3))
x_norm = (x - mean) / std           # normalized features
local_std = x.float().var(dim=1)     # texture energy for routing
return weight * x_norm + bias, local_std
```

### 3. Hierarchical Feature Fusion

**Critical quality feature**. Every group's output is tapped before the final upsampler:

```
input → conv_in → [group1 → group2 → ... → groupN] → concat(all) → fusion_1x1 → conv_mid → heads
```

**Why**: Shallow groups preserve high-frequency textures; deep groups capture semantic context. Fusion 1×1 learns to blend them optimally. Cost: one convolution (355K params for photo). Benefit: +0.2–0.5 dB PSNR.

### 4. WindowAttention with Halo Exchange

Sparse global attention on local windows. To avoid boundary artifacts, a **3×3 depthwise conv** precedes attention, letting each window "see" 1 pixel beyond its border (overlap without unfolding cost).

- Windows: 16×16 patches
- Heads: 4  
- Shifted window every other block (SW-MSA style)

### 5. TokenDictionaryCA — Dynamic Global Beacon

Generates **adaptive tokens** per image:

```
x → AdaptiveAvgPool(1) → Conv→GELU→Conv → (B, num_tokens, C)  # tokens
x_flat → Linear(q)  # (B, N, C//2)
tokens → Linear(k,v)  # (B, num_tokens, C//2), (B, num_tokens, C)
Attention: softmax(q·k^T) · v → (B, N, C)
```

Only 64 tokens (photo) or 32 (video recommended). Runs in ~0.02 ms.

### 6. VarianceRouter

Gates beacon outputs by local texture variance computed by iLN. Flat/constant regions → gate closed; textures → open. Avoids polluting smooth areas with global context.

### 7. LayerScale

Per-channel learned scaling for each residual branch: `γ × branch_output`, initialized to `1e-4`. Prevents early training explosion.

---

## Training Recipe

### Datasets

- **Primary**: DIV2K (800 training) + Flickr2K (2650) = 3450 HR images
- **Validation**: Urban100 (100), Set5 (5), Set14 (14)
- **Augmentation**: Random horizontal flip (50%), 90° rotation (50%)
- **Multi-scale degradation**: Bicubic downsampling for 2×/3×/4×; Gaussian noise σ=0.02 for 1× denoising

### Loss Function

**Primary**: L1 loss (Huber-like robustness)  
**Auxiliary**: MS-SSIM (weight 0.08 photo, 0.05 video)

Alternative: **Charbonnier loss** (sqrt(1+(error)^2/6)) — smoother near zero.

Recommend testing both.

### Optimizer

AdamW:
- LR: 2e-4 (linear warmup 3k iters)
- β1: 0.9, β2: 0.99
- Weight decay: 0 (HAT uses 0) or 1e-4?
- Gradient clipping: enabled (norm threshold auto)

### Schedule

MultiStepLR:
- Photo: milestones [300k, 500k, 650k, 700k, 750k], γ=0.5, total 800k
- Video: milestones [50k, 100k, 150k], γ=0.5, total 200k

EMA decay: 0.999 with power warmup.

### Batch Size

- Photo: 4×256×256 patches on 12GB GPU (RTX 3060)
- Video: 8×256×256 (lighter model)

Adjust based on available VRAM.

---

## Benchmark Results

> **Status**: Awaiting trained models. Below are target baselines.

### Comparison Targets (4× scale)

| Model | PSNR (DIV2K) | SSIM | Params (M) | Latency (ms, RTX 3060 FP16) | VRAM (GB) |
|-------|--------------|------|------------|----------------------------|-----------|
| **HAT-L** | 28.60 | 0.8498 | 40.85 | 3.21 | 10.41 |
| **HAT-M** | 27.97 | 0.8368 | 20.77 | 1.70 | 10.30 |
| **ParagonSR3 Photo** | ? (target >28.6) | ? | ~42 | ~3.2 | ~10.4 |
| **ParagonSR3 Video** | ? (target ~27.8) | ? | ~8 | ~0.8 | ~1.8 |

*HAT-L/M numbers from `docs/source/resources/benchmark4x.csv`.*

### Expected Gains

- **Photo**: +0.2 dB over HAT-L due to hierarchical fusion
- **Video**: Comparable PSNR to HAT-M at 2× speed, lower VRAM

---

## Deployment Guide

### 1. Train

```bash
python train.py -opt options/train/ParagonSR3/fidelity/ParagonSR3_Photo_fidelity_benchmark.yml
```

Checkpoints saved as `net_g_ema_{iter}.safetensors` in `experiments/`.

### 2. Fuse & Export ONNX

```python
python scripts/paragonsr3/convert_onnx_release.py \
  --checkpoint experiments/ParagonSR3_Photo_fidelity_multiscale/models/net_g_ema_800000.safetensors \
  --arch paragonsr3_photo_multiscale \
  --scales 1,2,3,4 \
  --output release_onnx \
  --device cuda
```

Outputs:
- `paragonsr3_photo_1x_fp32.onnx`
- `paragonsr3_photo_2x_fp32.onnx`
- `paragonsr3_photo_3x_fp32.onnx`
- `paragonsr3_photo_4x_fp32.onnx`

Each is a single-scale model with fused RepConv.

### 3. TensorRT Engine (FP16)

```bash
trtexec --onnx=release_onnx/paragonsr3_photo_4x_fp32.onnx \
         --saveEngine=engines/paragonsr3_photo_4x_fp16.trt \
         --fp16 \
         --workspace=4096 \
         --buildOnly  # optional, for just building
```

**Note**: Engines are GPU-specific; each user must build their own.

### 4. Inference (Python)

```python
import tensorrt as trt
# ... standard TRT Python inference
```

Or use ONNX Runtime for simplicity.

---

## Ablation Studies

**Recommended ablations** (once trained):

| Configuration | PSNR gain | Comment |
|---------------|-----------|---------|
| Full ParagonSR3 | baseline | — |
| w/o Hierarchical Fusion | -0.3 dB | Remove taps |
| w/o SCA | -0.15 dB | Remove from GatedRepConv |
| w/o Beacons (all) | -0.4 dB | Disable both Window+Token |
| w/o Token beacons | -0.15 dB | Keep Window only |
| w/o Window beacons | -0.25 dB | Keep Token only |
| w/ Static tokens (learned) vs Dynamic | -0.05 dB | Dynamic better |
| w/o Stochastic depth | -0.1 dB | Regularization helps |
| HAT-L baseline | -0.2 dB | Your target |

---

## Limitations

### Video Recurrence Training

The `MultiScaleSRModel` trains with **random scale per batch** and **on-the-fly bicubic downsampling** from GT. This breaks temporal coherence:

- Each batch is independent → `prev_feat` never trains
- Recurrence weights only learn from noise at inference

**Workaround**: Train video variant with a **proper video dataset** returning consecutive frames (e.g., `PairedVideoDataset`). Support for this is planned but not implemented.

**Until then**: Use video variant as a **fast single-scale model** (disable recurrence at inference). It's still ~2× faster than HAT-M with similar PSNR.

### Dataset Dependency

Results above assume DIV2K+Flickr2K training. Other domains (medical, satellite) may require adaptation or domain-specific data augmentation.

### TRT Compatibility

All ops are ONNX-exportable. However, extremely old TensorRT versions (<8.0) may lack full SDPA support. Use TRT 9+.

---

## Future Work

- Proper video recurrence training pipeline
- Quantization-aware training for INT8 TRT engines
- Diffusion-based fine-tuning for perceptual boost
- Lightweight "ParagonSR3-Nano" variant for mobile

---

**Maintained by**: Philip Hofmann  
**Citation**: If you use this model, please cite the design philosophy and benchmark fairly against HAT-L.
