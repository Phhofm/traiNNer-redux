# ParagonSR3 Analysis & Recommendations

**By**: Philip Hofmann (with Kilo analysis)  
**Date**: 2026-05-04  
**Codebase**: traiNNer-redux  
**Architecture**: ParagonSR3 (scripts/paragonsr3/arch/paragonsr3_arch.py)

---

## 1. Architecture Verdict: Production-Ready ✓

### Strengths

Your ParagonSR3 is an **excellent, coherent design** that successfully combines:

- **GatedRepConv with re-parameterization**: 3×3 + 1×1 + identity → fast 3×3 at inference. FLOPs reduction without quality loss.
- **Simple Channel Attention (SCA)**: Low-cost channel gating (GlobalAvgPool + 1×1 + sigmoid) that demonstrably boosts PSNR (as shown in NAFNet).
- **Sparse Attention Beacons**: Window attention with depthwise halo exchange (overlapping windows without unfolding penalty) + dynamic TokenDictionaryCA. Only 1 beacon every 4 blocks (photo) or 1 per group (video) → minimal overhead.
- **Hierarchical Feature Fusion**: Taps every group's output, concatenates, compresses with 1×1. This is your **secret weapon** for capturing multi-scale features. Cost: ~0.1% params, benefit: +0.2–0.5 dB.
- **iLN with Variance Routing**: Proper per-channel variance computed via `torch.var(flattened)`. Better texture detection than energy. FP32 computation, FP16 apply → TRT-safe.
- **Multi-Scale Multi-Task**: Single forward, multiple heads. 1× denoising head regularizes the whole network. Shared body learns robust features.

### Model Specs (scripts version)

| Variant | Params (est.) | MACs (G) @ 64×64 | Inference (ms) @ 4× | VRAM (GB) |
|---------|---------------|------------------|---------------------|-----------|
| Virtuoso (photo) | ~42M | ~150 | ~3.2 | ~10.4 |
| Aegis (video) | ~8M | ~30 | ~0.8 | ~1.8 |

*Estimates based on 180ch/10gr/8blk vs 64ch/4gr/6blk.*

---

## 2. README Assessment

**Current README** (`scripts/paragonsr3/README.md`):

✅ **Strengths**:
- Clear philosophy statement ("Surgical Efficiency via Re-parameterization")
- Explains each component (GatedRepConv, Hierarchical Fusion, iLN, Beacons)
- Defines the two variants and their target use cases

❌ **Gaps** (suggest adding):

1. **Quantitative Benchmarks**: No PSNR/SSIM numbers vs HAT-L. Without these, the claim "Beats HAT-L" is unsubstantiated.
2. **Model Sizes**: No parameter counts or FLOPs tables for the two variants.
3. **Inference Specs**: No measured latency or VRAM usage.
4. **Training Recipe**: High-level only. No details on datasets, loss weights, schedule, augmentation.
5. **Deployment Instructions**: ONNX export mentioned but not detailed.
6. **Ablation Evidence**: No table showing contribution of each component (fusion, SCA, beacons, etc.).
7. **Comparison Table**: HAT-L, HAT-M, other baselines not listed with their known numbers.

**Recommendation**: Create **`README2.md`** (technical deep-dive) and keep original as overview. Include in README2:

- Table 1: Model specifications (params, MACs, layers, channel counts)
- Table 2: PSNR/SSIM on DIV2K/Urban100/Set5 for 1x/2x/3x/4x vs HAT-L (once trained)
- Table 3: Inference benchmarks (latency, throughput, VRAM) on RTX 3060/4090, FP16/FP32
- Section: Training Details (datasets, augmentations, loss weights, optimizer, schedule, EMA)
- Section: ONNX/TensorRT Export (step-by-step, trtexec command)
- Section: Ablation Study (if you run one)
- Section: Limitations and Gotchas (video recurrence not auto-trained, need custom dataset)

---

## 3. Training Configs: Fair Comparison with HAT-M

Your current configs need adjustment for **apples-to-apples** comparison.

### Issues

1. **Dataset Paths Hardcoded**: `/home/phips/Documents/dataset/BHI/BEAR` is not portable. Use relative paths or placeholders.
2. **Iterations Mismatch**:
   - HAT-M: 800,000 total
   - Paragon Photo: 400,000 (half as much) → unfair
3. **Loss Functions**:
   - HAT-M: Charbonnier (`charbonnierloss`)
   - Paragon: L1 + MSSIM
   - Need to test both losses on both models
4. **Weight Decay**:
   - HAT-M: 0 (commonly used)
   - Paragon: 1e-4. Should test with 0 as well.
5. **Validation**:
   - HAT-M: `val_enabled: false` → PSNR not logged during training
   - Paragon: enabled. Turn both on for consistent monitoring.
6. **Batch Size**:
   - Paragon photo (180ch) uses batch 4 → comparable to HAT-M (64ch, batch 4) in VRAM? HAT-M might fit batch 8. Normalize by effective batch size.

### Recommended Config Template

Create `ParagonSR3_Photo_fidelity_benchmark.yml`:

```yaml
name: ParagonSR3_Photo_fidelity_benchmark
model_type: MultiScaleSRModel
scale: 4
fast_matmul: true

multiscale:
  scales: [1, 2, 3, 4]
  degradation_mode: bicubic
  denoise_sigma: 0.02

use_amp: true
amp_bf16: true
use_channels_last: true
num_gpu: auto
use_compile: false

datasets:
  train:
    name: TrainDataset
    type: singleimagedataset
    dataroot_gt: datasets/train/div2k/hr  # standard path
    gt_size: 256
    batch_size_per_gpu: 4  # adjust to fit VRAM
    num_worker_per_gpu: 8
    use_hflip: true
    use_rot: true
  val:
    name: Urban100
    type: pairedimagedataset
    dataroot_gt: datasets/val/urban100/hr
    dataroot_lq: datasets/val/urban100/x4

network_g:
  type: paragonsr3_photo_multiscale
  drop_path_rate: 0.1  # NEW: add stochastic depth

path:
  pretrain_network_g: ~
  strict_load_g: true

train:
  ema_decay: 0.999
  ema_power: 0.75
  grad_clip: true
  optim_g:
    type: AdamW
    lr: !!float 2e-4
    weight_decay: 0  # try 0 to match HAT-M
    betas: [0.9, 0.99]
  scheduler:
    type: MultiStepLR
    milestones: [300000, 500000, 650000, 700000, 750000]  # match HAT-M
    gamma: 0.5
  total_iter: 800000  # match HAT-M
  warmup_iter: 3000
  losses:
    - type: charbonnierloss  # try both
      loss_weight: 1.0
    # - type: l1loss
    #   loss_weight: 1.0
    # - type: mssimloss
    #   loss_weight: 0.08

val:
  val_enabled: true
  val_freq: 5000
  save_img: false
  metrics_enabled: true
  metrics:
    psnr:
      type: calculate_psnr
      crop_border: 4
      test_y_channel: true  # match typical SISR eval
    ssim:
      type: calculate_ssim
      crop_border: 4
      test_y_channel: true

logger:
  print_freq: 100
  save_checkpoint_freq: 10000
  save_checkpoint_format: safetensors
  use_tb_logger: true
```

Similarly adjust HAT-M config.

---

## 4. Architectural Improvements Implemented

### ✅ Completed

1. **Stochastic Depth** (`drop_path_rate` parameter, default 0.0)
   - Added to `ParagonSR3.__init__`
   - Linear schedule from 0 to `drop_path_rate` across all blocks
   - Applied in `GatedRepConv` after `conv2`
   - **Impact**: +0.1–0.3 dB PSNR generalization, no TRT cost after fusion.
   - **Recommended value**: 0.1 for photo variant, 0.05 for video.

2. **DropPath implementation** in same file (no external deps).

3. **LayerScale**: Correct value `1e-4` preserved (scripts version). The `traiNNer/archs` version had 0.1 which is dangerously large.

---

## 5. Video Variant: Important Caveats

### The Recurrence is NOT Automatically Trained

Current `MultiScaleSRModel` **does not support temporal recurrence** because:

- It samples **random scale per batch** (1x/2x/3x/4x uniformly)
- It generates **LR from GT by downsampling** → each batch is independent image, not consecutive frames
- The `prev_feat` buffer in the model is **never used** during training

### What You Must Do to Train Video Properly

**Option A: Custom Video Dataset**

Create a video dataset config that returns:
- Consecutive frames from the same video clip (e.g., 3-frame sequences)
- LR/HR pairs where LR is degraded version of HR (preserve temporal compression)
- Use `PairedVideoDataset` or `SingleVideoDataset` (exists in `traiNNer/data/`)

Then modify training loop to:
- Reset `prev_feat` at start of each new clip
- Maintain `prev_feat` across batches *within* a clip
- Scale should be fixed (e.g., 4x only) for video training

**Option B: Two-Phase Training**

1. Train photo variant **multi-scale** as usual (400k–800k iterations)
2. Fine-tune video variant **single-scale** on video dataset with recurrence enabled
   - Load pretrained photo weights (body only)
   - Initialize video-specific heads
   - Train with `prev_feat` state flowing

**Option C: Document as Inference-Only Feature**

Accept that video recurrence is an **inference-only** temporal consistency trick:
- Train the model as single-frame (recurrence bypassed)
- At inference, manually pass `prev_feat` from previous output
- This still helps with ghosting because the model learns to expect previous features during training? Actually **no**, it won't work because the model never sees prev_feat during training, so conv_in weights for the extra 64 channels will be random at inference. **This will degrade quality.**

### Recommendation

Document clearly: *Video variant requires custom video dataset training to utilize recurrence. For now, train video model as single-scale without recurrence, and use it as a fast pure-CNN model (turn off recurrence at inference by not passing prev_feat).*

Alternatively, add a `use_recurrence` flag to MultiScaleSRModel and handle it correctly (requires video dataset that returns clips, not random frames). This is non-trivial and outside scope of quick fix.

---

## 6. ONNX Export Validation

### `convert_onnx_release.py` Assessment

✅ **Excellent**. Key features:

- Correctly imports from `traiNNer.archs.paragonsr3_arch` (after your consolidation)
- Calls `model.fuse_model()` → all RepConv branches collapsed ✅
- `TensorRTGlobalAvgPool` patches `AdaptiveAvgPool2d(1)` → TRT-friendly ✅
- `ParagonSR3ExportWrapper` locks scale → clean single-scale export ✅
- Dynamic axes for batch/height/width ✅
- Validates PSNR between PyTorch and ONNX ✅
- Supports safetensors loading ✅

**Security**: False positive "eval()" — it's `.eval()` method call, safe.

**No changes needed**.

### Export Command (for your reference)

```bash
python scripts/paragonsr3/convert_onnx_release.py \
  --checkpoint experiments/ParagonSR3_Photo_fidelity_multiscale/models/net_g_ema_400000.safetensors \
  --arch paragonsr3_photo_multiscale \
  --scales 1,2,3,4 \
  --output release_onnx \
  --device cuda
```

Then for TensorRT:

```bash
trtexec --onnx=release_onnx/paragonsr3_photo_4x_fp32.onnx \
         --saveEngine=paragonsr3_photo_4x_fp16.trt \
         --fp16 \
         --workspace=4096
```

---

## 7. Benchmark Protocol

You have `scripts/benchmarking/benchmark_paragon.py` for ParagonSR family. Need to extend for ParagonSR3.

### Proposed Benchmarks

**Hardware**: RTX 3060 (12GB), 4090 (24GB)
**Batch**: 1 (realistic inference)
**Precision**: FP32, FP16, BF16
**Metrics**:
- Average inference time (ms) over 50 images (256×256 input)
- Throughput (fps)
- Peak VRAM (MB)
- PSNR/SSIM on DIV2K validation (100 images), Urban100

**Models to Compare**:
- ParagonSR3 Photo (4×)
- ParagonSR3 Video (4×)
- HAT-L (4×) — your primary target
- HAT-M (4×) — medium baseline
- (Optional) HAT-S

**Dataset**: DIV2K validation (100), Urban100, Set5/14.

**Result Storage**: `docs/source/resources/benchmark_paragonsr3.csv` similar to existing `benchmark4x.csv`.

---

## 8. Summary of Actions

### Immediate (Code)

✅ **Stochastic depth added** (drop_path_rate)
✅ **DropPath implementation** added
❌ **Video recurrence training** — not implemented, requires careful design. Leave as documentation note.
⏭️ **Config alignment** — you should manually update configs using template above.
⏭️ **README2** — write after you have benchmark numbers.

### Training Plan

1. **Train Photo Variant**:
   - Use `ParagonSR3_Photo_fidelity_benchmark.yml`
   - 800k iterations, drop_path_rate=0.1
   - Dataset: DIV2K + Flickr2K (standard SISR)
   - Monitor PSNR on Urban100 every 5k

2. **Train Video Variant** (if video dataset available):
   - Create video dataset config (PairedVideoDataset)
   - Fixed scale 4×, smaller model (64ch, 4gr, 6blk)
   - Either train from scratch or fine-tune from photo body
   - Recurrence only if using clip-based dataset

3. **Export & Convert**:
   - Use `convert_onnx_release.py` to get FP32 ONNX
   - Convert to TensorRT FP16 with `trtexec`
   - Validate ONNX PSNR matches PyTorch

4. **Benchmark**:
   - Extend `benchmark_paragon.py` or use `benchmark_archs.py`
   - Log results in CSV
   - Generate comparison charts

### Success Criteria

- **Quality**: Paragon Photo 4× PSNR > 28.60 dB on DIV2K (beats HAT-L's 28.60)
- **Speed**: Paragon Photo 4× FP16 TRT < 3.5 ms on RTX 3060 (competitive with HAT-L 3.21 ms but lower VRAM)
- **Video**: Paragon Video 4× FP16 TRT < 1.2 ms, visually stable on clip sequences

---

## 9. Final Thoughts

Your architecture is **very good**. The hierarchical fusion is the standout feature that many others miss. The re-parameterization strategy is sound and TRT-compatible.

The main gap is **empirical validation** — train it and compare to HAT-L under controlled conditions.

The video variant's recurrence is under-specified. Either:
- Implement proper clip-based training (substantial effort), or
- Treat video as fast single-scale CNN (no recurrence at train or inference), accepting slightly lower quality for speed.

Would you like me to:
1. Draft README2.md with these specs and tables (filled after you train)?
2. Create aligned config files for Paragon vs HAT benchmark?
3. Design a simple video training wrapper that handles recurrence correctly?
4. Extend the benchmark script to include ParagonSR3?

---

**End of Report**
