# NERVE — Ablation log & how we landed on the architecture

Research log, first pass: **2026-09-05**. Companion to `NERVE.md` (living spec)
and `nerve_review.md` (pre-training review). Documents the setup, the exact
protocol, results, and every decision — so future maintainers know why NERVE
looks the way it does.

## TL;DR (verdicts)

| Question | Result | Decision |
|---|---|---|
| Keep the 7 LayerNorms? | **No — removing them is +1.15 dB** and ~54% faster training | Norm-free is the canonical config |
| Dilated convs on blocks 3 & 5? | No gain on bicubic data (23.33 vs 23.39) | Drop for now; re-test under a real-degradation (OTF) recipe |
| Conv `bias=True`? | +0.08 dB **with** LN; **0.00 dB** without LN | Keep `bias=False` |
| Tanh gate (arm C)? | Not triggered | A already beats SPAN in PSNR |
| SiLU / StarReLU (NERVE.md F/G)? | Not run this pass | De-prioritized: marginal-dB chasers vs the simplicity thesis |

Headline: **norm-free NERVE (arm D) hit 24.54 dB / 0.724 SSIM on Urban100 x4
under the matched protocol vs 23.39 / 0.566 for the LayerNorm baseline** —
and it exports at **30 ONNX nodes instead of 100**.

## 1. Why an ablation at all

`NERVE.md` "Planned validation" assumed LayerNorm helped bf16 stability (an
educated guess during design, mirroring the fp32-statistics LN fix). The
`nerve_review.md` review flagged this as the single highest-value question:
LayerNorm is ~37-70% of the exported graph and pure memory-bandwidth cost. The
ablations exist to keep only what measurement justifies — the same discipline
that produced HEART.

## 2. Environment (this first pass ran on a Windows PC with an Intel Arc)

- **Machine:** Windows, Intel Core i3-12100F, 16 GB RAM, **Intel Arc A750 8 GB**
  (driver 32.0.101.8864). No NVIDIA/CUDA available.
- **PyTorch:** `2.14.0+xpu` from `https://download.pytorch.org/whl/xpu`
  (official Intel-GPU wheels), Python 3.13.1, bf16 autocast, **no
  torch.compile** (the triton-xpu backend needs a C compiler on PATH on this
  box).
- **Repo:** traiNNer-redux (`train.py` is CUDA-only; untouched). Ablations ran
  through a self-contained lab in `scripts/nerve/` (`ablation_train.py`,
  `nerve_variants.py`, `export_dynamic.py`, data + driver scripts). The
  canonical `traiNNer/archs/nerve_arch.py` stayed frozen during runs.
- All arms ran on the **same machine, same backend, same dtype, same seed**,
  so A-vs-* decisions are internally valid. These numbers are NOT directly
  comparable to RTX-3060/DIV2K numbers from other runs — record
  machine/backend/dtype with every run.

## 3. Data & protocol (matched across every arm)

- **Train:** LUCID CC0 v2 HC-512 (`Phips/lucid-cc0-v2-hc-512`, public,
  100,866 512×512 CC0 tiles). HR kept in `datasets/train/lucid_hc512/hr`;
  bicubic `lr_x4` generated with **chainner CubicCatrom** (chaiNNer standard,
  via `chainner_ext`) to match the HEART release recipe.
- **Val:** Urban100 x4. LR regenerated locally with the same CubicCatrom
  kernel as training (downloaded LR packs use a different kernel and shift
  absolute numbers). Metric = **Y-from-YCbCr, crop_border 4, data range
  [0,1]**, using traiNNer-redux's `calculate_psnr_pt`/`calculate_ssim_pt`
  (Matlab-YCbCr convention). Pure-bicubic baseline measured: **23.26 dB /
  0.660 SSIM** (published 23.14/0.658 — sanity check passed).
- **Per arm:** 20,000 iters, LQ64 crops (GT 256), batch 16, AdamW lr 2e-4
  (betas 0.9/0.99, wd 0), linear warmup 1k, MultiStepLR milestone 16k ×0.5,
  Charbonnier loss (eps 1e-12), EMA 0.999/0.75 (val on EMA net), hflip/rot90,
  seed 1024, input/GT range **[0,1]** (this matches what traiNNer-redux
  actually feeds models — SPAN archs assume it).
- ~100,866 tiles = ~3.2 epochs at this budget; 16 crops/iter.

## 4. Results (best Urban100 val PSNR @ iter, SSIM at that iter)

| Arm | Change vs A | Params | it/s | PSNR | @iter | SSIM | Note |
|---|---|---|---|---|---|---|---|
| A | baseline (LayerNorm) | 273,840 | 4.1 | 23.387 | 20000 | 0.566 | reference |
| D | **norm-free** | 273,168 | 6.4 | **24.541** | 20000 | **0.724** | **winner** |
| E | dilation=2 conv2 blocks 3&5 | 273,840 | 7.5 | 23.334 | 20000 | 0.565 | ~tie with A |
| F | bias=True (still has LN) | 274,512 | 7.5 | 23.465 | 20000 | 0.566 | +0.08 dB |
| B | SPAN (repo `span`, norm=False) | 2,621,964 | 5.5 | 22.526 | 20000 | 0.681 | see caveat |
| H | norm-free + bias (partial, 6k iters) | 273,840 | - | 23.76 @6k | - | 0.681 | **= D @6k** → bias adds nothing on norm-free |

Reading:
- A/E/F all plateaued near the bicubic PSNR (23.3-23.5) with SSIM **below**
  the bicubic baseline (0.565-0.596 vs 0.660): the LN nets learned an
  over-smoothed residual under this bf16/EMA protocol. D does not have this
  problem and improves far faster.
- D @ 6k (23.77) already exceeds A's best at 20k (23.39).
- H ≈ D exactly at the same iters ⇒ the +0.08 dB bias gain seen on the LN
  model (F) does not transfer to the norm-free model. Keep `bias=False`.

## 5. Why LayerNorm hurt (hypotheses, for future researchers)

Not fully root-caused — the win is empirical and decisive, and the protocol is
bf16-autocast + EMA. Candidate mechanisms:
1. LN computes channel statistics per spatial pixel; under bf16 AMP with
   fp32-cast stats this adds noise/numerical churn per forward.
2. LN re-centers each channel per pixel, which can suppress the small
   high-frequency residual signals these tiny nets need to emit, biasing the
   solution toward the bicubic baseline (matching the A/E/F PSNR≈bicubic but
   SSIM<bicubic pattern).
3. The norm adds ~24 non-conv ops × 7 = ~70/100 exported nodes and ~54%
   training slowdown (it/s 4.1 vs 6.4) — pure overhead in this regime.
This matches the classic EDSR-era result: small conv nets do not need
normalization when trained with modern hygiene.

## 6. Export / deployability findings (verified this pass)

- `pytest tests/test_archs -k nerve`: **16/16** (CPU; scales 1-4, both
  registries). ruff + pyright clean on the lab scripts.
- Dynamic-shape fp32 ONNX (opset 20) + `onnx.checker` pass; ORT-CPU vs
  PyTorch max abs diff **3-6e-6** at 48×48 / 323×711 / 720×1280, scales 2 & 4.
  The bicubic `Resize` maps cleanly (unlike some fears) — parity is exact.
- **A vs D node count (the mobile cost of LayerNorm):** A = **100 nodes**
  (ReduceMean×14, Mul×14, Sub×7, Pow×7, Sqrt/Div×7, ...), D = **30 nodes**
  (Conv×14, Add×8, Relu×6, DepthToSpace, Resize). The 7 LayerNorms are ~70 of
  100 exported nodes — far more than the ~37% estimated from NERVE.md, because
  each norm is ~10 ops, not 1.
- Toolchain caveats recorded for future work:
  - torch 2.14 dynamo exporter has an **opset floor of 18** (an "opset 17"
    request emits 18; legacy tracer is broken in this build). Canonical export
    = opset 20.
  - torch.compile on XPU needs a C compiler on PATH (triton-xpu driver build).
  - XPU does not support fp64 → metric code must run on CPU copies.

## 7. Pitfalls found while building the lab (worth knowing)

1. **Data range [0,1]**: traiNNer-redux's dataset returns float **[0,1]**.
   Feeding [0,255] silently changes the task. SPAN-family archs assume [0,1].
2. **Metric convention**: naive `0.299R+0.587G+0.114B` full-range Y reads
   ~1.3 dB lower than the standard Matlab-YCbCr numbers everyone quotes.
   Use the repo's `calculate_psnr_pt(..., test_y_channel=True)`.
3. **Val LR kernel**: downloaded SR-benchmark LR packs (eugenesiow, etc.) may
   use a different downscale kernel than your training LR. Regenerate val LR
   with the training kernel for internal consistency (baseline sanity check
   should reproduce the published bicubic numbers).
4. **SPAN `norm` default**: the repo `span` factory defaults to `norm=False`
   (no internal `(x-mean)*img_range` scaling). Our arm B used that default —
   protocol-fair and NERVE.md-faithful, but the resulting PSNR sits below
   bicubic while SSIM is healthy (scale/offset regime). A norm=True re-run is
   an optional follow-up for curiosity; it does not change the D-vs-A
   structural result.

## 8. What's next (not yet done)

- **Adopt the winner into `traiNNer/archs/nerve_arch.py`** (norm-free, single
  config, `bias=False`, ~273,168 params): remove LayerNorm2D/NerveBlock-norm,
  keep global residual + bicubic residual. Re-run the 16 arch tests, ruff,
  pyright, ONNX export + node-count, then update `NERVE.md`/`nerve_review.md`
  wording and this log.
- Delete the losing variant code once the canonical file is updated.
- Re-test arm E (dilated convs) **only under a Real-ESRGAN-style OTF
  degradation recipe** — dilation targets real-world blur, not bicubic data.
- Optional later: SiLU/StarReLU (NERVE.md F/G), real mobile latency
  (NCNN/CoreML), LPIPS/DISTS, multi-dataset eval, and an eventual SPAN
  norm=True re-run.

## 9. Canonical adoption + size sweep (2026-09-05, second pass)

All of section 8's first two items are **done**:

- `nerve_arch.py` is now the **norm-free canonical** (`dim=48`, `n_blocks=16`,
  `bias=False`, ~688K params, 70 exported ONNX nodes: Conv 34 / Add 18 / ReLU 16
  / DepthToSpace / Resize; ORT-CPU parity 3-6e-6 at 48x48/323x711/720x1280,
  scales 2 & 4; 16/16 arch tests, ruff+pyright clean).
- Losing variants were deleted; the lab (`scripts/nerve/`) now trains the
  canonical arch parameterized by `--dim/--n-blocks` (or `--arch span`).

**Size sweep** (same matched 20k clean-bicubic protocol, dim 48, Urban100 x4):

| depth | params | best PSNR | SSIM | RF (LR px) |
|---|---|---|---|---|
| 6 | 273K | 24.541 | 0.724 | ~29 |
| 12 | 522K | 24.849 | 0.740 | ~53 |
| 16 (chosen) | 688K | 24.936 | 0.743 | ~69 |

Decision rationale for n_blocks 16 as the **single** default: quality still
improves with depth (+0.31 dB for 6->12, +0.09 dB for 12->16 at equal budget),
the ~69px RF gives ~4x margin over the recipe's worst 17px blur kernel, and
688K stays in proven-mobile weight class (Real-ESRGAN `compact` is 621K). If a
future target phone needs a leaner profile it is a one-constant change, not a
variant.

NERVE training configs (for the 3060/Linux box) were added mirroring HEART:
- `options/train/RELEASE/4x_NERVE_release.yml` (clean bicubic fidelity)
- `options/train/RELEASE/4x_NERVE_OTF_fidelity.yml` (OTF fidelity pretrain)
- `options/train/RELEASE/4x_NERVE_OTF_gan.yml` (OTF GAN finetune, same losses
  as 4x_HEART_OTF_gan_lq96)

Remaining for the GAN tier (later, on the GPU box): run release -> OTF
fidelity -> OTF GAN, validate on Urban100_OTF x4 with TopIQ, then benchmark
mobile latency (NCNN/CoreML) and optionally re-test dilation under OTF.

## 10. Multi-scale release chain (2x then 4x-from-2x) — 2026-09-06

EDSR-style cross-scale warm start, validated on this machine (same matched lab
protocol; clean bicubic LUCID data; Urban100 val, YCbCr crop_border=scale):

| Run | setup | params | iters | best PSNR | SSIM |
|---|---|---|---|---|---|
| from-scratch 4x (earlier) | random init | 688K | 20k | 24.94 | 0.743 |
| **release 2x** | random init | 672K | 60k | **31.19** (x2) | 0.918 |
| **release 4x (from 2x)** | 2x body warm start | 688K | 100k | **25.71** | 0.774 |

Findings:
- The 2x->4x transfer loads 33/34 state-dict keys; only the scale-specific
  PixelShuffle head (upsampler.0, different channel counts) resets. Warm start
  is real: 4x-from-2x hit 25.49 dB at only 24k iters vs 24.94 dB at 20k for
  the random-init run.
- Best EMA checkpoints export to dynamic-shape ONNX cleanly
  (ORT-CPU vs PyTorch max diff ~1e-5 at 48x48/323x711/720x1280).

Recipe notes:
- Train order: `2x fidelity` -> `4x fidelity (warm body)` -> `OTF fidelity`
  -> `OTF GAN` (the last two run with traiNNer-redux release configs on the
  GPU box).
- This lab run used lq64/bs16 60-100k iters; the Linux release configs
  (options/train/RELEASE/2x_NERVE_release.yml, 4x_NERVE_release.yml with the
  strict_load_g:false warm start) train longer at lq96 and are the canonical
  release runs.

## 11. Review-2 follow-ups (2026-09-06)

Independent review (nerve_review2.md) confirmed the frozen architecture and
flagged three blind spots; all cheap ones are now closed with data:

### Fair (param-matched) SPAN baseline — NERVE wins by ~2.1 dB
Repo `span` with `feature_channels=26` -> 664,762 params (NERVE is 687,888),
identical 20k matched protocol, Urban100 x4:

| model | params | best PSNR | SSIM |
|---|---|---|---|
| SPAN feat26 | 664,762 | 22.798 | 0.663 |
| NERVE nb16 (seed 1024) | 687,888 | **24.936** | **0.743** |

Even with the parameter budget controlled, repo-SPAN under the identical
[0,1]/norm=False protocol sits below the bicubic baseline (23.26 dB) — NERVE
leads by ~2.1 dB. Caveat that remains: SPAN's native `norm=True` regime and
the GAN/texture tier are still untested; the clean-bicubic claim is now solid,
the GAN-tier claim stays queued.

### Seed robustness — size decision is safe
Canonical nb16 re-run with seed 2048: 24.917 dB / 0.742 vs seed 1024
24.936 / 0.743 -> run-to-run noise is ~±0.02 dB, far below the +0.09 dB
(12->16 blocks) decision margin.

### Multi-dataset eval (release 4x model, standard metric convention)
| set | bicubic | NERVE-4x (from 2x, 100k) | from-scratch (20k) |
|---|---|---|---|
| Set5 | 28.79 | 32.11 | 31.25 |
| Set14 | 25.78 | 27.36 | 27.04 |
| B100 | 25.69 | 26.39 | 26.28 |
| Urban100 | 23.26 | 25.71 | 24.94 |

Still queued (need the GPU box / a device): OTF-GAN smoke with LPIPS + eyes on
images, dilation-under-OTF, NCNN/CoreML latency, and CoreML dynamic-shape
conversion. Toolchain versions should be pinned next to release checkpoints.

## 12. OTF/GAN capability smoke (2026-09-06/07) — the GAN regime now has data

Closes the biggest blind spot from `nerve_review2.md` (§4.4/§7.2): every prior
NERVE measurement was clean-bicubic fidelity; the Real-ESRGAN OTF + GAN regime
had zero measurements. Run on the RTX 3060 box via traiNNer-redux RELEASE
configs, **aggressive smoke scale** (lq64, hard caps) — a network test, NOT a
release run.

**Setup:** stage 1 = 4x OTF fidelity from scratch (no clean checkpoint on this
box), lq64, bs8, 20k iters, const lr 1e-4 (`4x_NERVE_OTF_fidelity_lq64.yml`).
Stage 2 = 4x OTF GAN, warm start from the stage-1 20k EMA, lq64, bs4, 30k iters
(`4x_NERVE_OTF_gan_lq64.yml`) — identical degradation + losses to the proven
HEART OTF GAN (mssim 0.5 / perceptual 0.01 / hsluv 1.0 / gan 0.1, dunet D), so
any failure would isolate to the generator. Val = Urban100 OTF x4.

**Stage 1 (base), PSNR / SSIM / LPIPS:**

| iters | PSNR | SSIM | LPIPS |
|---|---|---|---|
| bicubic ref | 20.55 | 0.533 | 0.737 |
| 5k | 21.21 | 0.587 | 0.564 |
| 10k | 21.38 | 0.602 | 0.526 |
| 15k | 21.43 | 0.608 | 0.505 |
| 20k (cut, plateau) | 21.45 | 0.610 | 0.495 |

**Stage 2 (GAN), TopIQ / LPIPS:**

| iters | TopIQ | LPIPS |
|---|---|---|
| GAN 5k | 0.350 | 0.432 |
| GAN 10k | 0.353 | 0.415 |
| GAN 15k | 0.357 | 0.393 |
| GAN 20k | 0.360 | 0.383 |
| GAN 25k | 0.364 | 0.375 (best LPIPS) |
| GAN 30k | 0.365 (best TopIQ) | 0.376 |

**Verdict: NERVE is GAN-capable at 688K. No parameter increase needed.**
- No collapse/wax signature: TopIQ climbed monotonically to the cap; LPIPS fell
  0.12 (−24%) vs the fidelity base. GAN loss crept up slowly (G pressing), D
  outputs oscillated in a healthy band, structure losses held the base. No NaN.
- Visual check (vision model, 8 Urban100 montages, LR/GT/fidelity/GAN): real
  crisp texture (window grids, planks, perforation holes, rib lines), no grid,
  no halos, no moiré on regular texture, no color speckle, cleaner than fidelity
  and close to GT. Trouble cases (dense regular perforation, glass lattice)
  held up — where small SR nets usually fail.
- Absolute TopIQ (0.365) is far below HEART-GAN (0.513): expected — this is a
  30k smoke from a 20k scratch base, not a 125k+ warm-started release run.
- Implication: the architecture and the shared HEART GAN recipe both hold for
  the tiny pure-conv member. The remaining work is release-scale training
  (lq96, warm-started, long schedule), not architecture.

**Arch cleanup before official training (2026-09-07):** both temporary test
hooks from this pass — the `use_checkpoint` compat param and the `dilate_blocks`
knob — were removed from `nerve_arch.py` before any official release run. The
release arch is exactly: `NerveBlock` (conv-ReLU-conv), `NERVE(dim=48,
n_blocks=16)` with the `nerve()` factory (scale/in_ch/out_ch/dim/n_blocks).
687,888 params at 4x / 672,336 at 2x; 16/16 arch tests; 71-node dynamic ONNX.
The lq64 smoke configs were deleted after the test; this log is the record.

## 13. Dilation-under-OTF A/B (arm E re-test, 2026-09-07) — result: see below

Arm E was a no-op on clean-bicubic data; the queued follow-up was to re-test
`dilation=2` under the OTF recipe, where real blur lives. Protocol is IDENTICAL
to the §12 stage-1 baseline (from scratch, lq64, bs8, 20k, const lr 1e-4);
dilated arm adds `dilate_blocks=[7, 11]` (dilation=2 on conv2 of blocks 8 & 12,
one-indexed) — zero extra params. Decision rule: adopt only on a clear win
(>~0.1 dB OTF-val PSNR, seed noise ≈ ±0.02); otherwise freeze plain.
Expectation was null (RF ~71 px is ~4x the worst 17 px blur kernel).

**Exportability (verified before the run):** the dilated variant exports to
dynamic-shape ONNX identically to plain — same 71 nodes, same op set
(Conv/Relu/Add/DepthToSpace/Resize), ORT-CPU parity ~3.4e-6 at 64x64 and 77x83.
Dilation appears only as the standard Conv `dilations=[2,2]` attribute, which
NCNN supports natively. If dilation ever wins, deployability is unaffected.

**Result (2026-09-07): NULL — freeze the plain architecture.**

| arm | 20k PSNR | SSIM | LPIPS |
|---|---|---|---|
| plain baseline (§12) | 21.454 | 0.6102 | 0.4954 |
| dilated [7,11] | 21.462 | 0.6111 | 0.4958 |

Dilated trails/ties plain at every checkpoint (5k +0.007, 10k +0.006, 20k
+0.008 dB PSNR; LPIPS 0.4958 vs 0.4954) — well inside the ±0.02 seed band.
The RF (~71 px ≈ 4x the worst 17 px blur kernel) already covers OTF blur, so
extra dilated reach has nothing to add. **Architecture decision: NERVE is
frozen as canonical plain norm-free (dim 48, n_blocks 16, bias=False), no
dilation.** Dilation was removed from the arch file before official training
(see the §12 cleanup note); the A/B table above is the record of why.


## 14. Checkerboard / phase-lock fix (2026-09-10) — ICNR + degradation phase jitter

**Problem (visual):** the OTF **fidelity** stage output (and, to a lesser
degree, the OTF **GAN**) shows a fine 2x2 checkerboard + period-4 column
banding on smooth dark regions (anime hair, flat sky). The clean `*_release`
models are free of it (const-gray 2x2 phase spread 0.003/255 vs fidelity
0.169/255 at 135k). It is baked into the head weights, not a sampling glitch:
constant-gray input produces a periodic output signature.

**Root causes identified:**

1. **Phase-locked degradation.** The final resize-back in `RealESRGANModel`
   resamples to the fixed lq grid, so the LR always sits at the same sub-pixel
   phase relative to the GT. The model exploits that lock, and the PixelShuffle
   head learns a phase-dependent DC gain -> checkerboard.
2. **`nearest-exact` in the final resize-back** (`resize_mode_list3`, 25%
   probability in the release OTF configs). Nearest is phase-locked *and*
   aliasing — the single worst case for the grid.
3. **No ICNR init.** The PixelShuffle conv was randomly initialized, so each
   sub-pixel phase starts with an independent bias.

**Fixes (merged, permanent):**

- `traiNNer/utils/icnr.py`: ICNR init + a `icnr_reinit(model)` walker that
  re-initializes every Conv2d directly feeding a PixelShuffle.
- `nerve_arch.py` / `heart_arch.py`: **ICNR is now unconditional** — applied
  automatically at construction in both arches, no flag. It is init-only
  (zero graph/runtime cost, same dynamic-ONNX output) and is the same
  practice `temporal_span_arch.py` (SPAN) already uses.
- `redux_options.py`: `path.icnr_reinit_g` (re-scrub the head after loading a
  warm-start, keeping the body) and `lq_phase_jitter` / `lq_phase_jitter_range`
  (random sub-pixel shift of the final LQ, train-only). **`lq_phase_jitter`
  was briefly defaulted to true, then reverted to false in §15** — jitter
  destabilized the GAN stage and aliased LR is the intended teacher.
- `realesrgan_model.py`: `_lq_phase_jitter` — random (dx,dy) in ±0.5 lq px via
  grid_sample (reflection padding), applied after the final sinc filter.
- `misc.py`: `icnr_reinit` excluded from `make_exp_dirs` path iteration.
- New configs `4x_NERVE_OTF_fidelity_v2.yml` / `4x_NERVE_OTF_gan_v2.yml`:
  `icnr_reinit_g: true` (fidelity only) and anti-aliased
  `resize_mode_list3` (bilinear/bicubic/lanczos, no nearest-exact).
  (Arch `icnr` key is no longer a config option — the arch just does it.)
- `4x_NERVE_L_OTF_fidelity.yml` / `4x_NERVE_L_OTF_gan.yml`: same recipe at
  dim 64 / n_blocks 24 (1.80M params) for the mottling re-test, from scratch.

**Baseline grid metrics (const-gray, /255; ani-hair 2x2 chkr in parens):**

| model | checkerboard 2x2 | x period-4 |
|---|---|---|
| 4x release 485k | 0.003 | 0.014 |
| 4x OTF fidelity v1 135k | 0.169 | 0.127 |
| 4x OTF GAN v1 35k | 0.045 | 0.127 |

Target for v2 fidelity: const-gray 2x2 phase spread << 0.10 (near release
levels). Validation in progress (`scripts/nerve/check_grid.py`).

**Eval note:** v2 fidelity warms up from the clean release weights with an
ICNR-scrubbed head, so its head starts phase-balanced. **Result: the
degradation changes that accompanied this (phase jitter + anti-aliased final
resize) made the GAN visibly worse — see §15.**

## 15. Capacity scaling + the aliasing lesson (2026-09-12/13)

**What went wrong in v2 (result).** The v2 chain (weaker base + sub-pixel
`lq_phase_jitter` + anti-aliased final resize, no nearest-exact) did reduce the
smooth-hair grid mid-training (const-gray 0.169 -> 0.058 @95k) but at a real
cost:

- fidelity PSNR 21.95 (v1) -> 21.23 (v2) on the phase-locked val (~-0.3 dB on a
  fair sub-pixel-shifted eval, where v1 loses 2.4 dB under shift vs v2's 1.0).
- the GAN v2 output is visibly **worse**: thin lines break into beading/dots
  and flat areas stipple. Line high-freq (right-hair LapStd): v1 GAN 35k =
  19.31, v2 GAN 45k = 24.23 (+25%), v2 60k = 25.53. lpips 0.3174 -> 0.4020,
  topiq 0.4065 -> 0.3485.
- the fidelity grid crept back late (0.149 @200k).

**The aliasing lesson (the fix).** *The LR must stay aliased.* `nearest-exact`
downscaling gives jagged LR; training on it teaches the aliased->smooth
(de-alias) mapping. Removing aliasing from the LR (anti-aliased final resize)
plus sub-pixel jitter left the generator phase-uncertain and unable to
de-alias, so adversarial pressure turned thin lines into beading. Actions:

- `resize_mode_list3` restored to include `nearest-exact` in all NERVE OTF
  configs.
- `lq_phase_jitter` default reverted to **false**.
- ICNR (init-only) kept — harmless and correct.

**Capacity scaling (exact v1 GAN recipe; aliased LR; no jitter).** The bigger
model is warm-started from the v1 dim48 fidelity via a width+block expansion
(`scripts/nerve/expand_nerve.py`: old weights top-left, new channels small
random, new blocks near no-op) — verified ~equivalent to v1 fidelity (81.85 dB).

| model | params | topiq @35k | lpips @35k | line HF right/bangs |
|---|---|---|---|---|
| v1 GAN dim48/b16 | 0.69M | 0.4065 | 0.3174 | 19.31 / 15.54 |
| dim64/b24 | 1.80M | **0.4124** | **0.3136** | **18.64 / 14.82** |
| dim64/b24 @40k | 1.80M | 0.4121 | 0.3122 | 19.01 / 15.13 |

Capacity **helps both metrics and line cleanliness** (lower HF = fewer
artifacts); the 1.80M model had not plateaued at 40k (lpips still falling)
while v1 plateaued at 35k.

**Reference parameter counts:** SRVGGNetCompact default 0.62M, x4v3 1.21M,
ultracompact 0.33M; SPAN-S 2.24M, SPAN base 2.62M, SPAN-F64 3.96M. NERVE
1.80M sits in the SPAN class while staying conv-only.

**Protocol (running):** train 1.80M to 150k, then dim64/b32 (2.39M) to 150k
with the same recipe/start/iters -> scaling curve. Next point if needed:
dim96/b24 (4.03M). Deferred: a dedicated aliasing augmentation in the OTF
pipeline; WebP compression (jpeg already generalizes to webp/screenshot DCT).
