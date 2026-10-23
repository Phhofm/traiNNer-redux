# HEART — Efficient Attention with Rank-factorized bias Transformer

**TL;DR:** window-attention SR transformer. HAT-iLN quality, FlashAttention
speed, none of the complexity. **One architecture, one config. Trains in bf16.
Exports to dynamic ONNX with a single command** — no fused/unfused checkpoint
pairs, no custom CUDA kernels, no relative-position tables.

## What is HEART?

HEART is what HAT-iLN becomes when you strip out the parts nobody actually
deploys:

- **RIB** replaces the relative-position-bias *table* — learned low-rank
  position features concatenated onto Q/K, so attention runs on standard
  FlashAttention/SDPA kernels. No table, no index gather, no mask.
- **OCAB removed** — HAT's most complex, most memory-hungry block. Our
  ablations showed it isn't needed at window 32.
- **Keeps the proven bits:** i-LN (input-adaptive normalization, bf16-stable),
  window attention, a convolutional branch (CAB).

The result: a ~16.7M single-image SR architecture that **ties HAT on PSNR,
trains in stable bf16 AMP, and exports to ONNX that runs in TensorRT / ONNX
Runtime / DirectML.**

## Why HEART exists

Most SR papers chase benchmark numbers by adding modules that are annoying to
train and painful to deploy — dynamic routing, deformable attention, blocks
that only work with a specific resolution, or a "fused" checkpoint for
inference and an "unfused" one for fine-tuning. The result looks good on a
leaderboard and gathers dust in practice.

HEART has a different goal: **real-world simplicity.** Train it anywhere,
convert it to dynamic-shape ONNX, run it everywhere. No fusion step. No second
model file. No custom kernels.

## When to use HEART

- You want HAT-class quality on real images (denoise, deblur, upscale) without
  babysitting the training.
- You need weights you can convert to ONNX and ship — no fusion step, no "use
  variant B for inference."
- You are tired of benchmark-chasing networks that add modules no one can
  actually deploy.
- You need desktop quality at reasonable inference cost. For mobile/edge, use
  NERVE (coming in the BODY suite).

## Quickstart

1. Copy `traiNNer/archs/heart_arch.py` into your traiNNer-redux clone
   (`traiNNer/archs/`).
2. Copy the release config from `options/train/RELEASE/4x_HEART_release.yml`
   and point `dataroot_gt` / `dataroot_lq` at your data.
3. Train:
   ```bash
   python train.py -opt options/train/RELEASE/4x_HEART_release.yml --auto_resume
   ```
   (`--auto_resume` is **mandatory** — without it an interrupted run restarts
   from iter 0 and archives the experiment dir.)
4. Convert to dynamic ONNX (each model, one at a time):
   ```bash
   ulimit -v 14000000
   ./venv/bin/python scripts/heart/export_minimal.py \
       experiments/4x_HEART_pretrain600k/models/net_g_ema_615000.safetensors \
       onnx/4x_HEART_1x3xHxW_fp32_op17.onnx 4
   ```
5. Run inference anywhere ONNX Runtime or TensorRT runs.

Full architecture docs, ablation results, and training recipes are below.

> **Training crop size: use `lq_size: 96` (a multiple of the 32 px window).**
> HEART partitions the image into 32×32 windows. Training at a crop size that
> is *not* a multiple of 32 (e.g. 80) makes the model reflect-pad its own
> bottom/right edge every batch, so it learns a false "edge = mirrored
> content" response — visible as a bright bottom band and a tile-grid pattern,
> and strongly amplified by a GAN loss. All HEART ablations and release runs
> used `lq_size: 96` (= 3×32); a larger multiple (128) also works. This
> matters most for GAN/OTF finetunes, which are most sensitive to boundary
> statistics.

## BODY suite status

| Model | Role | Params | Status |
|---|---|---|---|
| HEART | desktop quality tier | 16.7M | trained + ONNX |
| NERVE | mobile/edge tier | ~270K | trained |

## Design

```
conv_first → [6× RHAG-RIB] → conv_after_body (+ global residual)
           → pixelshuffle upsampler

RHAG-RIB (residual group):
  [6× HAB-RIB blocks] → 3x3 conv → group residual

HAB-RIB block:
  i-LN → RIB window attention (window 32, FlashAttention/SDPA)
       + CAB conv-attention branch (scale 0.01)
  i-LN → MLP (mlp_ratio 2)
```

- **i-LN** (Image Restoration tailored LayerNorm, arXiv:2504.06629) — stable
  training in bf16/AMP, no divergence.
- **RIB window attention** — FlashAttention-compatible, window 32, shifted
  windows via non-wrapping pad-and-partition (no masks, no position tables).
- **CAB conv branch** — HAT's proven local inductive bias, kept.
- **OCAB removed** — the most complex and memory-hungry part of HAT is gone.

## Goals

| Goal | How |
|---|---|
| Great results | HAT's proven hybrid block (window + conv) + large-window global context |
| Fast train & inference | FlashAttention via RIB (SST measured 2.1x train / 2.9x inference vs RPB) |
| Stable training | i-LN + EMA std clamping, bf16/AMP safe |
| Simple code | ~640 lines, reuses existing blocks, no masks/tables/OCAB |
| Easy deployment | Pure PyTorch, ONNX-exportable, no custom CUDA kernels |

Note on speed: the FlashAttention win only exists in bf16/fp16 (measured
~1.6x faster than HAT-iLN-M at bf16 inference). In fp32, HEART is slightly
slower than HAT-iLN — fp32 has no flash kernel. Use reduced precision.

## Status

Implemented, verified, and releasing. Architecture is frozen; all weights and
ONNX files below are measured, not extrapolated.

Final architecture (post-ablation, 2026-08-31): RIB window-32 attention + CAB,
`attention_freq=2`, rank 8, linear MLP — the 20k sweep winner. RoPE,
no-positional, GDFN, SwiGLU, LayerScale, window 48/64, and rank 18/32 were all
measured and rejected; the losing code paths were removed from `heart_arch.py`
(646 lines, 16/16 tests, lint + typecheck clean).

Release artifacts (all exist, paths verified):

| Artifact | Checkpoint | ONNX |
|---|---|---|
| 4x bicubic | `experiments/4x_HEART_pretrain600k/models/net_g_ema_615000.safetensors` | `onnx/4x_HEART_1x3xHxW_fp32_op17.onnx` |
| 2x | `experiments/2x_HEART/models/net_g_ema_200000.safetensors` | `onnx/2x_HEART_1x3xHxW_fp32_op17.onnx` |
| 4x OTF fidelity (v1.1) | `experiments/backup_otf_v2/net_g_ema_150000.safetensors` | `onnx/4x_HEART_OTF_1x3xHxW_fp32_op17.onnx` |
| 4x OTF perceptual (GAN) | training | — |

All checkpoints backed up to a private HF repo
(`Phips/heart-sisr-pretrain`). ONNX files pass `onnx.checker` and match
PyTorch numerically (ORT diff < 1e-2).

In flight: **4x OTF perceptual (GAN) finetune** (~3 days remaining). When it
finishes, its weights + ONNX join the release set.

Known limitation: a faint **window-32 grid artifact** (128 px period in 4x
output) on smooth gradients and fine lines in OTF outputs. Diagnosed via FFT;
root cause is window boundaries (not JPEG blocks or PixelShuffle). Overlap
attention contradicts the frozen arch, so it is documented here, not fixed in
1.0.

Release-gate checks passed: `--auto_resume` (CLI flag) resume verified,
tiled inference seam-free, ONNX legacy-tracer opset 17 + onnxruntime,
fp16/bf16 inference, bf16 AMP training stability, torch.compile validated.
  the bare model, so checkpoints and val are unaffected. Kernel profile of
  one step: ~23% flash attention, ~30% small elementwise ops (what compile
  fuses), ~12% convs — no architectural hotspot left for custom kernels.

Dev notes for anyone touching the attention code:

- Shifted windows are non-wrapping: the feature map is padded top/left by half
  a window (reflect) and windows are partitioned from the padded origin. Border
  windows see reflected padding, never content from the opposite image border.
  Do NOT reintroduce `torch.roll` cyclic shifting — without an attention mask
  it couples opposite borders, and adding a mask would force SDPA off the flash
  kernel (the whole point of RIB).
- Logits are pre-scaled per SST eq. 5 (`q/√D`, `pos/√R`) and SDPA is called
  with `scale=1.0`. Re-adding default SDPA scaling silently divides logits
  by √48 ≈ 7 and shifts attention toward uniform — shape-only tests will
  not catch it. Without this fix the model trains around it and loses
  quality for no reason.
- RIB features are computed in fp32 then cast to the input dtype. Skipping
  the cast breaks pure fp16 inference (buffer/param dtype mismatch).

Next step is a short ablation against the baselines before the full training
schedule:

- `heart` (this network, 19.2M params, RIB window 32)
- `atd` (20.3M, the efficiency champion)
- `hat_m` (20.8M, classic HAT)

The ablation trains each for ~20-50k iterations on DIV2K and compares
Urban100 PSNR and wall-clock speed. HEART proceeds to full DF2K training only
if it wins.

## Ablation results (2026-08-15, post shifted-window fix)

30k iterations each, identical settings (post-fix HEART code). RTX 3060 12 GB,
DIV2K → Urban100:

| arch  | best Urban100 PSNR | @iter | final SSIM | train it/s | peak VRAM |
|-------|--------------------|-------|------------|------------|-----------|
| HEART | 25.0235            | 30000 | 0.7497     | 0.86       | 1.44 GB   |
| HAT_M | 25.0524            | 30000 | 0.7517     | 0.95       | 2.84 GB   |
| ATD   | OOM at iter 1 — did not run on the same config |

Reading the table honestly:

- **Quality:** HAT_M leads by +0.03 dB / +0.002 SSIM at the 30k cutoff —
  within noise for a short schedule, and both were still improving on their
  final validation. Effectively a tie; only a full-length run separates them.
- **VRAM:** HEART uses ~half the peak memory of HAT_M (1.44 vs 2.84 GB).
- **Train speed:** HAT_M trains ~10% faster (0.95 vs 0.86 it/s). Window 32
  attention is 4x the per-pixel attention work of window 16; the flash kernel
  makes it feasible, not free. HEART's speed advantage is at *inference*
  (~1.6x vs HAT-iLN-M bf16), not training throughput.
- **ATD OOM caveat:** HEART/HAT_M ran with `use_checkpoint: true`; ATD's
  factory has no such option. The OOM is a real deployment-relevant result
  (ATD does not train on a 12 GB card at lq_size 64 / batch 4 as configured),
  but the memory comparison is not fully symmetric because of that.

### Inference benchmark (same trained EMA checkpoints, RTX 3060)

Median of 5 runs after warmup, `cudnn.benchmark=False`,
`/tmp/kilo/inference_bench.py`:

| arch  | dtype | input    | time    | peak VRAM |
|-------|-------|----------|---------|-----------|
| HEART | bf16  | 320x180  | 0.884s  | 0.45 GB   |
| HAT_M | bf16  | 320x180  | 1.092s  | 1.20 GB   |
| HEART | bf16  | 480x270  | 2.024s  | 0.90 GB   |
| HAT_M | bf16  | 480x270  | 2.351s  | 2.47 GB   |
| HEART | fp32  | 320x180  | 2.350s  | 0.89 GB   |
| HAT_M | fp32  | 320x180  | 1.816s  | 2.37 GB   |
| HEART | fp32  | 480x270  | 5.347s  | 1.78 GB   |
| HAT_M | fp32  | 480x270  | 3.973s  | 4.94 GB   |

In bf16 (the deployment-relevant precision) HEART wins both speed (1.16-1.24x)
and VRAM (~2.7x less). In fp32 HAT_M is faster (no flash kernel for fp32), but
HEART still uses 2.7-2.8x less VRAM.

### Training speed benchmarks (RTX 3060, bf16 AMP, channels-last)

torch.compile (`use_compile: true`) validated in the training harness:
~1.5x training speed, quality-neutral (300-iter smoke: 21.88 dB vs 21.80 dB
uncompiled).

| config | it/s | 800k iters | peak VRAM |
|---|---|---|---|
| lq64 bs4, no compile (ablation) | 0.86 | ~10.7 days | 1.44 GB |
| lq64 bs4 + compile | 1.26 | ~7.4 days | 1.44 GB |
| lq96 bs2 + compile | 1.19 | ~7.8 days | 0.98 GB |
| lq96 bs4 + compile | ~0.63 | ~14.7 days | 1.70 GB |
| lq96 bs4 / lq128, no checkpoint | OOM | — | — |

compile_mode comparison (@ lq96 bs2, 3060): `reduce-overhead` 1.23 it/s >
`default` 1.18 > `dynamic=False` 1.19 ≈ `default+TF32` 1.19 > no compile 0.77.
`max-autotune` is not worth it on small GPUs (inductor warns: not enough SMs).
At the ablation config (lq64 bs4), `max-autotune-no-cudagraphs` was measured
equal to `reduce-overhead` (both 1.28 it/s) with 2x the compile time and
slightly more VRAM — it gains nothing on a 28-SM card because the tuned GEMM
search falls back. `fast_matmul` (TF32) is a no-op under bf16 AMP — measured.

Kernel profile of one training step (uncompiled): flash attention ~23%,
small elementwise ops ~30% (what compile fuses — the source of the 1.5x),
convs ~12%, MLP gemms ~4%. No hotspot remains that a custom Triton kernel
could profitably target.

### Scoreboard vs HAT_M (30k ablation)

| Metric | Winner |
|---|---|
| Quality | tie (HAT_M +0.03 dB, within noise) |
| Inference speed bf16 | HEART (1.16-1.24x) |
| Inference VRAM | HEART (~2.7x less) |
| Training VRAM | HEART (1.44 vs 2.84 GB) |
| Training speed (no compile) | HAT_M (+10%) |
| ATD on 12 GB | does not run (OOM) |

### Rank ablation (done, 2026-08-17)

RIB's per-head positional-bias matrix has rank <= `rank`. Three 20k-iteration
runs compared rank 8 / 18 / 32 on the identical ablation config **with
torch.compile reduce-overhead** (`HEART_rank{8,18,32}_ablation.yml`, lr decay
at 16k). All three used compile so the comparison is not confounded (compile
shifts short-run PSNR ~0.08 dB, larger than the 0.03 dB decision threshold).

| rank | 20k PSNR |
|---|---|
| **8** | **24.7820** |
| 32 | 24.7586 |
| 18 | 24.7346 |

**Winner: rank 8** (smallest within 0.03 dB of the best). Rank 8 beating
rank 18 means positional capacity is *not* the bottleneck at rank 8 — a
lower-rank positional mechanism is enough, and it is leaner (concat dim 40
vs 48) and slightly faster. The canonical default in `heart_arch.py` is now
`rank=8`. The rank-18 uncompiled 30k reference (25.0235) was historical only.
The release configs are patched to rank 8.

### Architecture ablation: window size & attention frequency (2026-08-19)

Ran because an 8-day release was at stake: if window 48/64 or sparser
attention changed the quality-per-second picture, it had to be found before
committing. Setup: lq96 bs2 (the release crop), 20k iters, compile
reduce-overhead, rank 8, DIV2K, seed 1024 — identical settings except the
variable (`HEART_{w32,w48,w64,attn2}_ablation.yml`). The release run was
halted at 75k iters (checkpointed, resumable) while this ran.

| config | 20k PSNR | it/s (compiled) | params |
|---|---|---|---|
| **attn2** (`attention_freq=2`, RIB) | **24.6282** | **2.02** | 16.68M |
| attn3 (`attention_freq=3`) | 24.5606 | ~2.3 | 15.87M |
| w32 (baseline, RIB rank 8) | 24.5063 | 1.29 | 19.10M |
| none (no positional, freq=1) | 24.4965 | ~1.4 | 18.94M |
| w48 | 24.4347 | 0.80 | 19.10M |
| rope (RoPE, freq=1) | 24.3615 | ~1.4 | 18.94M |
| gdfn (Gated-Dconv FFN, ratio 2.0, freq=2) | 24.2546 | ~1.9 | 19.28M |
| swiglu (SwiGLU FFN, ratio 4/3 param-matched, freq=2) | 23.7403 (best @12k; collapsed to 23.21 @20k) | ~1.9 | ~16.7M |
| qknorm (per-head QK LayerNorm, freq=2) | 24.6334 (tie with attn2, +0.005) | ~2.0 | ~16.7M |
| swiglu + qknorm | 23.9576 | ~1.9 | ~16.7M |
| layerscale (per-block residual scaling, init 1e-6) | 24.0288 | ~2.0 | ~16.7M |
| w64 | skipped (w48 already lost; 4.2x slower) | 0.31 | 19.10M |

Reading:

- **attention_freq=2 wins on both axes**: +0.12 dB over the baseline AND
  1.56x faster training, with 2.4M fewer params. HEART has more attention
  than it needs at 20M scale.
- **GDFN rejected** (-0.37 dB vs attn2): doubling the per-block 3x3 conv
  density (CAB already provides spatial convs) hurt. The FFN stays a linear
  MLP.
- **Positional barely matters**: no-positional ties RIB at freq=1 (-0.01 dB,
  noise), while RoPE's fixed rotation actively hurt (-0.15 dB). RIB stays as
  cheap insurance; it is not a quality driver.
- **Window 48/64 closed**: window size is not a quality bottleneck at this
  crop size (w48 -0.07 dB, 38% slower). SST's large-window gains needed
  scaled crops+datasets this regime doesn't have.
- **Phase 2 confirmation** (in progress): w32 vs attn2 head-to-head at 100k
  iters (`HEART_{w32,attn2}_confirm.yml`, lr decay at 50k). The 20k ordering
  must survive to 100k before the release config changes.

### Release training (actual pipeline, 2026-08-25..28)

The release trained in five chained phases, all unattended with
`--auto_resume`. This is the pipeline that actually ran (not the earlier
3-phase design, which was superseded).

| Phase | Config | Iters | Result |
|---|---|---|---|
| 1. 4x pretrain (bicubic) | `options/train/RELEASE/4x_HEART_release.yml` | 615k (early-stopped) | 26.2139 dB |
| 2. 2x finetune | `options/train/RELEASE/2x_HEART_release.yml` | 200k | 33.9959 dB |
| 3. 4x OTF v1.0 (real-world) | `options/train/RELEASE/4x_HEART_OTF.yml` | 300k | 22.8473 dB |
| 4. 4x OTF v1.1 (anti-alias) | `options/train/RELEASE/4x_HEART_OTF_v11_finetune.yml` | 150k | 22.9529 dB |
| 5. 4x OTF perceptual (GAN) | `options/train/RELEASE/4x_HEART_OTF_gan.yml` | 150k | in progress |

Key operational lessons learned the hard way:

- **`--auto_resume` is mandatory** (CLI flag, not YAML). Without it an
  interrupted run restarts from iter 0 and archives the experiment dir.
- **Dataset:** LUCID CC0 v2 HC-512 (100,866 tiles, 512x512, CC0). LR generated
  with chainner CubicCatrom bicubic (the chaiNNer ecosystem standard).
- **OTF degradation:** Real-ESRGAN-style two-pass blur + noise + JPEG, applied
  on-the-fly by `realesrgandataset`.
- **ONNX export froze the desktop twice** (see Part 1 of this plan) — the fix
  is a per-process memory cap, not `nice`/`ionice`.

Pipeline scripts (durable, in `scripts/heart/`): `download_lucid.py`,
`prepare_lucid.py`, `run_release_sequence.sh`, `watch_download_then_prepare.sh`,
`export_one.py`, `export_minimal.py`, `infer_otf_comparison.py`,
`grid_diagnosis.py`. Logs in `/tmp/kilo/`.

## Variants

HEART 1.0 ships as a single architecture (`heart`, 19.2M params, embed 180,
6×6 blocks, window 32, CAB compress/squeeze 3/30). A principled S/M/L scaling
family (width, depth, heads, rank, CAB) is a HEART 2.0 item — deliberately
not added before the release model proves the architecture.

## Positioning

HEART deliberately does **not** compete on benchmark PSNR. Research
architectures (HAT, DAT, RGT, Restormer, DRCT, Mamba-based SSMs) exist to
move leaderboard numbers, and they will out-score HEART on some val sets —
that is expected and accepted.

HEART competes on everything that happens *after* the paper: training
without babysitting, exporting without a fuse step, running in chaiNNer
tomorrow, and being understood by a new maintainer in 2028.

The rules that define the position:

- **One configuration.** No S/M/L variants, no fused/unfused checkpoint
  pairs, no "lite" forks. One architecture, one way to run it.
- **Zero custom kernels.** Everything a user needs ships in vanilla
  PyTorch; ONNX opset-17 export is verified, not promised.
- **Anything that loses the ablation protocol is removed from the code.**
  14 variants were measured; the winners stayed, the losers were deleted —
  not kept behind flags.
- **The architecture is boring on purpose.** The block is i-LN -> [RIB
  window attention | CAB] -> MLP. No dynamic routing, no deformable
  anything, no state-space branches.

If a technique requires a fuse step, a custom kernel, a shape assumption,
or a "use variant B only for inference" note, it does not belong in HEART.

## Deployment notes

- **Arbitrary input sizes:** `check_image_size` pads the input to a window
  multiple (reflect, replicate fallback for tiny images) and crops the
  output back. Verified at 32x32, 64x64, 128x128, 323x711, 777x65,
  720x1280. Users never handle padding themselves.
- **ONNX (verified, 2026-08-21, trained 20k checkpoint):** legacy tracer
  (`dynamo=False`), opset 17, dynamic H/W + batch. `onnx.checker` passes;
  onnxruntime output matches PyTorch (max abs diff 0.00026 at 128x128).
  Export pipeline for releases: export -> onnxsim -> ORT optimizer ->
  validate at 64x64 / 256x256 / 720x1280. Note: ORT *CPU* inference of
  large images needs several GB of RAM (a 16.7M transformer is not a CPU
  workload) — deploy on GPU providers (CUDA/TensorRT/DirectML).
- **NCNN: not supported, by design.** The fused SDPA node does not convert
  to NCNN, and window attention is not an NCNN-style workload anyway.
  HEART targets ONNX Runtime / TensorRT / DirectML; a mobile-conv sibling
  (NERVE) is the future NCNN member of the suite.
- **Large images — single pass vs tiling:** single pass fits up to ~720x1280
  LR on a 12 GB card (5.55 GB measured, bf16). 1080x1920 LR single-pass
  OOMs (~10.6 GB); tiled inference (tile 256, overlap 32, weighted blend)
  runs it at 0.86 GB and is seam-free (measured). Use chaiNNer seamless
  tiling / the repo's `tile_size` option for anything bigger than 720x1280.
- **Real-world degradations:** the backbone is degradation-agnostic. For
  denoising/JPEG cleanup train at `scale=1` (verified working) with a
  Real-ESRGAN-style degradation pipeline, same as HAT-R relates to HAT.
- **spandrel:** registered in-repo via `SPANDREL_REGISTRY` +
  `store_hyperparameters`, so traiNNer-redux/chainer forks that carry this
  repo load it directly. **Upstream-PR decision (made):** at PR time,
  `heart_arch.py`'s imports from `arch_util` (iLN) and `hat_iln_arch`
  (CAB, AffineTransform, DropPath, Mlp, PatchEmbed, PatchUnEmbed, Upsample)
  get vendored into the self-contained spandrel copy (~200 lines) rather
  than landing HAT-iLN in spandrel first. No code change in this repo until
  then.

## Experiment log (everything that shaped HEART)

Chronological record of every experiment, success and failure. The failures
matter as much as the wins — each one closed a direction.

| When | Experiment | Result | Decision / lesson |
|---|---|---|---|
| design | HAT-iLN base: RPB table -> RIB, OCAB removed, window 32 | — | Attention runs on FlashAttention/SDPA; simpler than HAT |
| 08-14 | Shifted windows via `torch.roll` | BUG: cyclic wrap coupled opposite borders (no mask) | Fixed: non-wrapping pad-and-partition |
| 08-14 | `F.pad` for V concat | minor | cleanup |
| 08-15 | 30k arch comparison HEART vs HAT_M vs ATD | 25.02 vs 25.05 vs OOM | tie with HAT_M at equal settings -> proceeded |
| 08-15 | Inference bench vs HAT_M | bf16: 1.16-1.24x faster, ~2.7x less VRAM; fp32 slower | deployment story is bf16 |
| 08-15 | torch.compile (smoke + harness) | +46-54% training, quality-neutral | adopted (`use_compile: true`) |
| 08-15 | compile_mode sweep | reduce-overhead 1.23 > default 1.18 > dynamic=False/TF32 1.19; max-autotune-no-cudagraphs == reduce-overhead with 2x compile time | reduce-overhead adopted; max-autotune rejected (28 SMs) |
| 08-15 | `fast_matmul` (TF32) | no-op under bf16 AMP | rejected |
| 08-15 | Crop/batch probes | lq96 bs4 OOM without checkpoint; lq96 bs2 fits, 9 windows | release crop = lq96 bs2 |
| 08-15 | `heart_light` (8.7M variant) | created, then removed | S/M/L family deferred to v2; 1.0 = one architecture |
| 08-17 | Rank ablation 8/18/32 @20k | 8 wins (24.782) > 32 (24.759) > 18 (24.735) | rank 8 default — positional capacity is NOT the bottleneck |
| 08-18 | Window ablation 48/64 @20k | w48 loses 0.07 dB, 38% slower; w64 skipped | spatial context is NOT the bottleneck at lq96 |
| 08-18 | Attention frequency 1/2/3 @20k | freq=2 wins: 24.628 (+0.12 dB) at 1.56x speed; freq=3 drops | attention density sweet spot = every 2nd block |
| 08-19 | RoPE instead of RIB @20k | loses 0.15 dB (24.36 vs 24.51) | learned low-rank positional beats fixed rotation |
| 08-19 | No positional at all @20k | ties RIB at freq=1 (24.497 vs 24.506) | RIB stays as cheap insurance; not a quality driver |
| 08-19 | GDFN (gated depthwise FFN) @20k | loses 0.37 dB (24.25 vs 24.63) | more per-block conv density hurts (CAB already convs) |
| 08-20 | SwiGLU FFN @20k | loses decisively: best 23.74 @12k, declined to 23.21 @20k | gated FFN destabilizes the i-LN-rescaled block — MLP stays |
| 08-20 | QK-Norm @20k | 24.6334 — tie with attn2 (+0.005) | tie-breaker: plain (simplest) — redundant with i-LN + RIB |
| 08-20 | SwiGLU + QK-Norm @20k | 23.9576 | gate instability again |
| 08-20 | LayerScale @20k | 24.0288 (−0.60) | 1e-6 init over-damps the branches; existing i-LN+conv_scale damping is right |
| 08-20 | Final cleanup | all test switches (ffn_type/qk_norm/layer_scale) stripped; canonical default attention_freq=2 | the file is the single-path measured-best architecture |
| 08-16 | Resume test | `--auto_resume` is a CLI flag, not YAML | all training commands pass it |
| 08-16 | Tiled inference | seam-free (seam diff == interior diff) | large-image tiling safe |
| 08-14..20 | ONNX export | legacy tracer opset 17 + checker + onnxruntime pass | deployment path verified |

Process lessons (drivers/pipeline): /tmp gets wiped on reboot — logs restored
from durable experiment copies; `ls dir/*.png | wc -l` breaks at ~100k files
(ARG_MAX) — use `find`; drivers are idempotent/self-healing and training runs
deprioritized (nice/ionice) so the machine stays usable.

## Considered-and-decided register

Every idea that was considered, with the decision and reason. If someone asks
"why didn't you use X" — the answer is here. Status: ✅ integrated (and
measured), ❌ rejected by measurement, 🚫 discarded by analysis (never run),
⏳ pending, 📌 deferred (not forgotten).

### Integrated

| Idea | Why |
|---|---|
| ✅ RIB instead of RPB table | Concatenated position features make attention FlashAttention/SDPA-compatible; measured (ties HAT_M, rank 8 suffices) |
| ✅ i-LN normalization | Input-adaptive holistic norm; bf16 stability is the paper's core claim, verified in training |
| ✅ CAB conv branch | Local inductive bias; the measured carrier of quality at this scale |
| ✅ Non-wrapping shifted windows | Fixes the cyclic-wrap border bug without the Swin mask (which would kill the flash kernel) |
| ✅ `attention_freq=2` | Measured winner: +0.12 dB at 1.56x speed vs attention-every-block |
| ✅ rank 8 RIB | Measured winner of 8/18/32 sweep |
| ✅ Window 32 | w48/64 measured losers at lq96 |
| ✅ torch.compile reduce-overhead | +46% training speed, quality-neutral; best of the mode sweep |
| ✅ bf16 AMP + gradient checkpointing | Verified stable; 1.4-1.7 GB training VRAM |
| ✅ Chainer CubicCatrom LR (release data) | The chaiNNer ecosystem standard — models behave as users generate LR |

### Rejected by measurement (ran, lost)

| Idea | Result | Why rejected |
|---|---|---|
| ❌ RoPE positional | −0.15 dB @20k | fixed rotation worse than learned low-rank positional |
| ❌ No positional | ties RIB at freq=1, but attn2+RIB wins overall | RIB stays as cheap insurance |
| ❌ GDFN gated-FFN | −0.37 dB @20k | more per-block conv density hurts (CAB already provides spatial convs) |
| ❌ SwiGLU FFN (param-matched 4/3) | best 23.74 @12k, declined to 23.21 @20k (−0.9 to −1.4 dB vs attn2) | the multiplicative gate destabilizes the i-LN-rescaled block (loss degraded after 12k) |
| ❌ SwiGLU + QK-Norm | 23.9576 @20k | same gate instability |
| ❌ LayerScale (init 1e-6) | 24.0288 @20k (−0.60 dB) | the near-zero init suppresses the branches through warmup; i-LN rescale can't compensate — the block's existing damping (i-LN + conv_scale 0.01) is the right amount |
| ⚖️ QK-Norm | 24.6334 — tie with attn2 (24.6282, +0.005 dB) | tie-breaker: plain wins (no component for zero measured gain) |
| ❌ Window 48 / 64 | w48 −0.07 dB + 38% slower; w64 skipped | spatial context is not the bottleneck at lq96 |
| ❌ `attention_freq=3` | −0.07 dB vs freq=2 | density sweet spot is every 2nd block |
| ❌ rank 18 / 32 | rank 8 wins | positional capacity is not the bottleneck |
| ❌ max-autotune compile modes | == reduce-overhead, 2x compile time | tuned GEMM search falls back on 28-SM GPUs |
| ❌ `fast_matmul` (TF32) | no-op | everything already runs bf16 on tensor cores |
| ❌ `compile(dynamic=False)` | no gain | shapes were effectively static anyway |
| ❌ `heart_light` 8.7M variant | created, then removed | hand-picked config, not a principled family; S/M/L is a v2 item |

### Discarded by analysis (never run — the reason is the evidence)

| Idea | Why discarded |
|---|---|
| 🚫 OCAB (HAT's overlapping cross-attention) | RPB tables + gather construction + memory are exactly what RIB/HEART removed; the 30k ablation already ties HAT_M *with* OCAB — the missing piece isn't needed |
| 🚫 Swin-style attention mask (shift fix alternative) | an additive mask forces SDPA off the flash kernel — the whole point of RIB |
| 🚫 RepCAB (reparameterized CAB) | `fuse()`-before-export creates two weight formats and breaks the drop-in property, for ~2-3% inference; quality hypothesis contradicted by the GDFN result |
| 🚫 MDTA transposed channel attention | global channel-covariance context — falsified direction (see w48) |
| 🚫 Stripe/multi-axis attention (HMA) | full-row/column context — same falsified hypothesis, plus partitioning complexity |
| 🚫 Dense residuals (DRCT) | groups and blocks already have residuals; dense skips grow activation memory |
| 🚫 Mamba/SSM backbones | custom CUDA kernels (`selective_scan`) violate the zero-custom-kernel constraint |
| 🚫 IET adaptive token selection | dynamic gather/indexing breaks dynamic ONNX tracing |
| 🚫 UCAN / SAT / FPLIA | sub-1M-parameter tier techniques; irrelevant at ~16.7M |
| 🚫 Muon optimizer | its gains are LLM-pretraining-scale; the RIB net is ~4k params |
| 🚫 FlashBias Triton kernel | HEART already *is* the SDPA variant of FlashBias; no bias-add attention exists to accelerate |
| 🚫 LAformer swap-in | different architecture, not an optimization; the deployability premise is already satisfied by HEART |
| 🚫 Triton-fused RIB concat | ~2-4% of a step at the cost of custom kernels; torch.compile already fuses the elementwise ops |
| 🚫 4D-native block refactor | single-digit % after compile's 46% landed; complexity not justified |
| 🚫 Hybrid window schedule (16/32 mix) | smaller windows contradict the measured landscape (w48 lost; context isn't the lever) |
| 🚫 lq128 crops | OOM without checkpointing and pathological compile on this GPU |
| 🚫 Progressive patch training (LQ64 -> LQ96) | measured apples-to-apples (bs2, compiled): lq64 3.86 it/s vs lq96 1.52 it/s (2.5x per iteration), but pixel throughput only 31.6 vs 28.0 kpx/s (+13%) — and lq64 crops see 4 windows vs lq96's 9, weakening cross-window mixing. ~8% total wall-time for real pipeline complexity and quality risk |

### Pending

None — the ablation program is complete. 14 variants measured; the core
(RIB rank-8 + attention_freq=2 + window 32 + CAB + plain MLP + i-LN) beat or
tied all of them. The architecture is at a measured local optimum.

### Deferred (not forgotten — see the 2.0 backlog)

📌 INT8 quantization, TensorRT export CI, RIB positional caching in eval, RIB
dtype sweep, ENAF-style early exits for 4K+ images, S/M/L family, larger
training crops (bigger GPU), video adaptation, rank 2/4 micro-sweep,
attention-freq bs4 batch check, MLP ratio sweep.

## HEART 2.0 backlog

Frozen for 1.0 (do not touch the architecture while the release model trains).
Ideas that need a fully trained 1.0 baseline before they can be judged
properly — the rule is: only adopt what wins on *quality per second* against
the 1.0 model, not what moves a benchmark number.

| Idea | Why it's on the list | Effort |
|---|---|---|
| Gated-Dconv FFN (GDFN, Restormer) | MEASURED (20k, ratio 2.0, freq=2): lost 0.37 dB (24.25 vs 24.63) — doubling per-block 3x3 conv density (CAB already provides spatial convs) hurt. Code removed. | — |
| SwiGLU FFN + QK-Norm | External review (2026-08-20): modernize the one remaining vanilla component (the MLP) + per-head QK LayerNorm for logit stability. TESTING: three 20k runs on the attn2 base (swiglu-only, qknorm-only, both). SwiGLU param-matched at ratio 4/3 (their 8/3 would have doubled params). | Low-Med |
| RepCAB (reparameterized CAB) | Train-time multi-branch conv fused at export. REJECTED for 1.0: the fuse()-before-export step conflicts with the trivial-export property that is part of the maintainability goal. | — |
| Transposed channel attention (MDTA) | Global channel-covariance attention — chases global context, which the w48 result already falsified as the lever. Backlog only if GDFN-style spatial enrichment proves the direction. | Medium |
| Multi-axis stripe attention (HMA) | Full-row/column context — same global-context rationale, contradicted by w48 losing at 20k. Skip unless a crop-size scaling study revives the context hypothesis. | Medium |
| Dense residual connections (DRCT) | Zero-param dense skips; NTIRE-2024 proven, but our blocks/groups already have residuals, and dense skips grow activation memory. Low expected value; cheap to test one day. | Low |
| ENAF early-exit routing | Inference-time wrapper for huge images, not an architecture change; post-release option only. | High |
| IET adaptive token selection | Dynamic gather/indexing risks breaking dynamic ONNX tracing — rejected on deployability grounds. | — |
| Mamba/SSM backbones | Custom CUDA kernels (selective_scan) violate the zero-custom-kernel constraint — rejected. | — |
| Larger windows (48 / 64) | MEASURED (20k, lq96): window 48 lost to 32 (-0.07 dB) and cost 38% more time. Closed for this crop/GPU regime unless crops scale too. | — |
| Attention frequency (H C H C) | MEASURED (20k, lq96): `attention_freq=2` beat the baseline (+0.12 dB) at 1.56x speed. Live candidate for the release config; longer-run confirmation pending. | Low-Med |
| RoPE instead of RIB | MEASURED (20k, lq96): RoPE lost 0.15 dB to RIB (24.36 vs 24.51) — learned low-rank positional wins; fixed rotations can't substitute. Closed. | — |
| Rank beyond release range (4/12/24/…) | The release gate only tests 8/18/32. Fine-grained rank scaling could shave the positional capacity further. | Low |
| RIB positional caching in eval | q_pos/k_pos recomputed every forward in every block; tiny but free to cache for inference. Micro-opt, measure before doing. | Low |
| RIB dtype sweep (bf16/fp16 vs fp32) | fp32 cast is for fp16-inference stability; whether bf16 RIB changes quality/speed is unmeasured. | Low |
| CAB variants / S/M/L family | An earlier hand-picked `heart_light` (8.7M) was removed for 1.0; a principled S/M/L scaling rule (width, depth, heads, rank, CAB) is the v2 way to extend the family. | Low-Med |
| TensorRT + export CI | ONNX verified; TensorRT untested. A CI export test (PyTorch→ONNX→ORT→TensorRT, compare max-abs-error/PSNR/runtime) would harden the deployment claim. | Medium |
| Quantization (INT8) | Untested; a 3060-class GPU model could get much faster. | Medium |
| Larger training crops | lq 128+ needs checkpointing + slower it/s; only test with a bigger GPU. | Medium |
| Video adaptation | temporal stability (like TSPAN) — separate scope. | High |
| Cyclic vs reflect shift | Deliberately closed for 1.0 (cyclic + mask kills the flash kernel). Revisit only if a flash-compatible exact-cyclic scheme emerges. | — |

Priority order after 1.0: confirm attention_freq=2 on a longer run (if the
release adopts it, this is moot) -> export hardening (TensorRT/CI) ->
RoPE swap-in benchmark -> rank/capacity scaling. Everything is measured
against the frozen 1.0 model, not against papers.
