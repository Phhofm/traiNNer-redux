# NERVE — Lightweight Single-Image Super-Resolution

The mobile/edge member of the BODY suite. Pure convolution, no attention, no
normalization, no gating, no positional encoding. ~1.8M params. Targets NCNN,
ONNX Runtime Mobile, CoreML.

NERVE does **not** chase benchmark PSNR. It trades the last 0.1 dB for
deployability: one configuration, one file, no fused/unfused checkpoint
pairs, no custom kernels, no clever modules a maintainer has to puzzle over.

**Positioning:** NERVE is not a research novelty. It is a production-grade
lightweight SR model — a modernized, norm-free EDSR re-engineered for 2026
deployment: a 5x5 stem for real-world degradations, plain residual blocks (no
LayerNorm — the norm was ablated out, see `NERVE_ABLATION_LOG.md`), a bicubic
input residual, and modern training hygiene (EMA, bf16 AMP). It is the model
you ship when you need super-resolution in a mobile app *this week*: ~1.8M
parameters, pure convolution, no attention, no fusing, no variants. If you
need desktop quality, use HEART; if you need something that trains in bf16
without clipping, exports to ONNX dynamic shapes on the first try, and converts
to NCNN without unsupported ops, use NERVE. The pitch is not "we beat SPAN by
0.01 dB" — it is "we beat SPAN on deployability, and we tie it on quality."

## What is NERVE?

NERVE is a pure-convolution SR network: **5x5 stem → 16 residual conv blocks →
global residual → PixelShuffle → + bicubic residual.** No attention, no window
partitioning, no normalization, no gating, no positional encoding. The
PixelShuffle head uses **ICNR initialization** (always on) so it cannot train
up a sub-pixel checkerboard, and the OTF degradation pipeline phase-jitters
the LR so the head never locks onto a fixed grid — both are init/data-level,
costing zero at inference and adding nothing to the ONNX graph. At ~1.8M
params it is small enough for mobile and clean enough that every op in its
ONNX graph (`Conv, Add, Relu, DepthToSpace, Resize`) maps to a standard NCNN
layer — no custom op definitions, no fused/unfused checkpoint pairs.

It is the fast, deployable member of the BODY suite: train it in bf16 AMP
without gradient clipping, export to dynamic-shape ONNX on the first try, and
convert to NCNN without touching unsupported ops.

## Why use NERVE?

| If you... | Use NERVE because... |
|---|---|
| need SR in a mobile/edge app | ~1.8M params, pure conv, NCNN/ORT-Mobile/CoreML-clean |
| want dead-simple training | bf16 AMP stable, no gradient clipping, no custom kernels |
| need dynamic input shapes | handles any size — 48×48, 323×711, 720×1280 all verified |
| want ONNX export that just works | 70 nodes, all standard ops, opset-20 verified |
| are tired of attention complexity | no windows, no RPB tables, no gating — one clean block |
| need a license-clean model | trained on CC0 data (LUCID CC0 v2 HC-512) |

## How to use NERVE

### Option A: drop-in (fastest)

Copy `traiNNer/archs/nerve_arch.py` into your traiNNer-redux clone. The file
is fully self-contained — it imports only `torch`, `torch.nn`, and
`spandrel.util.store_hyperparameters`. No dependencies on `arch_util` or
`hat_iln_arch`. Then use a NERVE training config.

### Option B: upstream PR (recommended long-term)

Open a PR to [the-database/traiNNer-redux](https://github.com/the-database/traiNNer-redux)
adding `nerve_arch.py` + training configs. The arch is pure PyTorch with
standard ops, no custom kernels, no fork-specific code — it ports mechanically.

### Option C: spandrel (chaiNNer integration)

For chaiNNer users, add NERVE to [spandrel](https://github.com/chaiNNer-org/spandrel)
so it loads like any other arch. The file is already self-contained for
spandrel (uses `store_hyperparameters` + dual registry) — no helper vendoring
needed, unlike HEART.

## Design philosophy

The spec that initiated NERVE assembled components from recent papers
(StarReLU from MetaFormer, a tanh gate from SPAN, LayerScale from DeiT III).
Each of those chases a metric — FLOPs, spatial selectivity, deep-stack
stability. None of them are "dead simple," and your stated goal is a network
that is robust, maintainable, and usable for years.

So NERVE keeps the spec's *structure* and *scale* but uses simple, proven
components:

| Spec proposed | NERVE uses | Why |
|---|---|---|
| StarReLU | ReLU | StarReLU markets FLOPs savings; ReLU is bf16-safe, universal, obvious |
| `(x + h) * tanh(h)` gate | plain `x + h` | the gate is clever but non-obvious; plain residual is clear |
| LayerScale | none | measured as a **loss** for HEART; unmeasured risk not worth it |

The discipline is the same one that produced HEART: start from a simple
baseline, ablate additions later, keep only what measurement justifies.

## Architecture

```
5x5 stem -> [NerveBlock x 16] -> global residual
         -> PixelShuffle(4) -> + bicubic(x)
```

- **5x5 stem** — larger initial receptive field for real-world degradations
  (blur kernels, JPEG artifacts) without adding attention
- **NerveBlock** — `Conv3x3 -> ReLU -> Conv3x3 -> + x`. Plain residual block;
  norm-free. (LayerNorm was the original design; the first ablation pass
  showed it cost +1.15 dB and ~54% training speed, so it was removed — see
  `NERVE_ABLATION_LOG.md`.)
- **Global residual** — `feat + blocks(feat)`; the network predicts a
  correction to its own shallow features
- **Bicubic residual** — output is `PixelShuffle(feat) + upsample(x)`; the
  network learns only the detail missing from a bicubic baseline

`dim=64`, `n_blocks=24` (~1.8M params, ~101px receptive field on the LR — large
enough to also handle Real-ESRGAN-style OTF degradation training, so one
config serves clean, OTF-fidelity, and OTF-GAN).

The op graph is trivial for inference backends: only `Conv`, `Add`, `ReLU`,
`PixelShuffle`, and bicubic `Resize`. No custom op definitions, no dynamic
indexing, no Flex delegates. NCNN converts this cleanly.

## Constraints

- Pure PyTorch, zero custom CUDA kernels
- bf16 AMP stable (ReLU, no exponential activations)
- Dynamic input shapes, ONNX opset-20 export verified
- Single configuration — no S/M/L variants, no fused/unfused checkpoints
- One file: `traiNNer/archs/nerve_arch.py`

## Verification

- 16/16 arch tests pass (fp32, scales 1–4); lint + typecheck clean
- ONNX export + `onnx.checker` pass with dynamic H/W
- Arbitrary shapes verified: 48x48, 323x711, 720x1280
- Both 2x and 4x

## Relationship to HEART

HEART is the desktop quality tier (window attention + CAB, ~16.7M). NERVE is
the mobile/edge tier (pure conv, ~1.8M). They share engineering standards —
single config, deployable, maintainable — but not architecture. NERVE is not
a shrunken HEART; it is a clean-sheet conv network that happens to share the
same export pipeline and code style.

## What NERVE is not

- Not a HEART variant or a distilled model
- Not an attention network — no RIB, no windows, no SDPA
- Not a multi-scale or UNet architecture — single path
- Not a benchmark chaser — it trades marginal PSNR for deployability
- Not a video SR network — it processes frames independently (see Known
  limitations)

## Known limitations and blind spots

These are honest unknowns, not weaknesses to hide:

- **No quality benchmark yet.** PSNR/SSIM on Set5/Set14/B100/Urban100/Manga109
  are unmeasured. The planned ablations will produce the first numbers.
  *(Updated 2026-09-05: first-pass Urban100 numbers now exist in
  `NERVE_ABLATION_LOG.md`; a full multi-dataset eval is still pending.)*
- **PSNR != perceptual quality.** A lower-PSNR model can look better to
  humans. We do not yet measure LPIPS/DISTS or do human evaluation.
- **Bicubic LR only.** Training uses bicubic downscaling (academic standard).
  Real-world images have unknown degradation (blur, noise, compression).
  Architecture matters less here than the degradation pipeline — a
  Real-ESRGAN-style OTF pipeline matters more for actual use.
- **Inference speed unverified on target hardware.** Training throughput is
  measured; real latency on mobile CPU / NCNN / CoreML is not.
- **Video SR is a different problem.** Running NERVE frame-by-frame on video
  works but produces no temporal consistency (flickering on static regions).
  Video needs a temporal module (optical flow, recurrence) — that is a
  separate project, not a smaller NERVE.
- **Dependency drift.** "Used for years" requires pinned versions (PyTorch,
  onnx, ncnn, coremltools) and documented tested combinations.
- **Reproducibility.** Exact seed, software versions, and hardware should be
  documented so others can reproduce the results.

## Status

Implemented and verified. **Norm-free NERVE (dim 64, n_blocks 24, ~1.8M) is
the canonical config** (the capacity sweep in §15/`NERVE_ABLATION_LOG.md` put
the quality knee at 1.80M: 0.69M -> 1.80M bought a large GAN gain, 1.80M ->
2.39M bought ~4x less per parameter). The historical norm-free/size pass: The LayerNorm baseline lost the first ablation pass by
+1.15 dB (see `NERVE_ABLATION_LOG.md`), and a size sweep (dim 48, n_blocks
6/12/16, matched 20k protocol, Urban100 x4) kept improving with depth:

| depth | params | best PSNR | SSIM |
|---|---|---|---|
| 6 | 273K | 24.54 | 0.724 |
| 12 | 522K | 24.85 | 0.740 |
| 16 (adopted then) | 688K | 24.94 | 0.743 |

The earlier 0.69M pass chose n_blocks 16; the capacity sweep then moved the
default to dim 64 / n_blocks 24 (~101px receptive field) for the OTF GAN
recipe. The 1.80M production chain is `2x_NERVE_release_1p8m.yml` ->
`4x_NERVE_release_1p8m.yml` -> `4x_NERVE_OTF_fidelity_1p8m.yml` ->
`4x_NERVE_OTF_gan_1p8m.yml`.

Release-chain runs (2026-09-06, same matched protocol): 2x fidelity trained to
31.19 dB / 0.918 (Urban100 x2), then the 4x model warm-started from the 2x body
(head resets) trained to **25.71 dB / 0.774 (Urban100 x4)** — the cross-scale
warm start beats random init (24.94 dB at 20k) from early on. Both export to
dynamic ONNX cleanly. Details in `NERVE_ABLATION_LOG.md` section 10.

## Planned validation (post-HEART, when GPU is free)

20k-iteration ablation on DIV2K+Flickr2K, bf16 AMP + torch.compile
reduce-overhead + Charbonnier + EMA, LQ64 patches. Matched protocol so the
numbers are comparable:

| Run | Config | Purpose |
|-----|--------|---------|
| **A: NERVE baseline** | as implemented (48 dim, 6 blocks, ReLU, plain residual) | target |
| **B: faithful SPAN** | re-implemented SPAN in traiNNer-redux with the same protocol | number to beat |
| **C: NERVE + tanh gate** (optional) | add `(x + h) * tanh(h)` to baseline | test if the gate is free quality |
| **D: norm-free NERVE** | remove all 7 LayerNorms, keep global residual | mobile speed win if it ties |
| **E: dilated convs** | `dilation=2` on conv2 of blocks 3 and 5 | cheap RF boost for real-world deblur |
| **F: SiLU activation** | replace ReLU with SiLU (`x * sigmoid(x)`) | test if smooth activation helps |
| **G: StarReLU (fixed)** | replace ReLU with `s * ReLU(x)^2 + b` (learnable s,b per channel) | test MetaFormer-style gating |

Execution note (2026-09-05 first pass): A, B, C(labelled but untriggered),
D, E ran as planned, plus a cheap **bias=True** arm (from the review) and a
norm-free+bias combo (H). F (SiLU) and G (StarReLU) were de-prioritized this
pass — marginal-dB chasers against the simplicity thesis. See
`NERVE_ABLATION_LOG.md` for results and the labeling of each arm.

Success criteria:
- **A ties B within ±0.03 dB:** NERVE wins — simpler, more stable, more
  deployable.
- **A trails B by >0.05 dB:** run C. If C recovers the gap, the gate is
  worth the complexity. If not, accept that SPAN owns the pure-PSNR niche
  and NERVE owns the deployability niche.
- **D ties A:** take the speed win — 7 fewer norms is ~42% fewer exported
  nodes and faster on mobile CPU.
- **E beats A on Urban100/Manga109:** keep the dilated convs for the
  real-world degradation version.
- **F or G beats A:** consider swapping ReLU; measure NCNN/ONNX export
  cleanliness of the winner.

First-pass outcomes (2026-09-05, details in `NERVE_ABLATION_LOG.md`): D beat A
by **+1.15 dB** (not merely a tie), so the norm-free config is adopted; E was
a no-op on bicubic data (re-test under OTF later); bias adds nothing on the
norm-free model; C was not needed (A beat B in PSNR). The architecture is
therefore **norm-free NERVE** — LayerNorm was the wrong guess.

**Pre-training verification:** convert the exported ONNX to CoreML with
dynamic H/W *before* the training run. `F.interpolate(mode="bicubic")` with
dynamic shapes is the single op most likely to fight coremltools; if it
fails, fall back to exporting residual-only and doing the bicubic add
host-side (no architecture change).

## Status update (2026-09-07): OTF/GAN capability confirmed

The last unmeasured claim — "does 688K pure-conv NERVE train productively under
the Real-ESRGAN OTF GAN recipe, or does it wax?" — now has data. An aggressive
smoke (lq64: 20k OTF fidelity from scratch + 30k OTF GAN warm-started) produced
no collapse, LPIPS −24% vs the fidelity base, monotonic TopIQ, and clean
texture on visual inspection (see `NERVE_ABLATION_LOG.md` §12). **Verdict:
architecture and defaults hold at 688K. (Superseded 2026-09-13: the later
capacity sweep in §15 found real gains up to ~1.80M, so a parameter increase
was warranted after all.) A dilation-
under-OTF A/B (arm E re-test, the last queued architecture experiment) is
documented in §13. The remaining work is release-scale training on the canonical
lq96 chain (clean 2x → 4x → OTF fidelity → OTF GAN), not architecture.

Positioning note: scale=1 is supported by the arch (PixelShuffle(1) + identity
bicubic residual are no-ops, covered by the scale 1–4 arch tests), which means
the same clean backbone can serve 1x restoration work (denoise/deblur/exposure)
without any new arch code — see `BODY.md` for the suite roadmap.


## Status update (2026-09-13): capacity scaling + the aliasing lesson

Two threads converged here; full numbers in `NERVE_ABLATION_LOG.md` §15.

**1. Aliasing in the LR is the teacher, not a bug.** A v2 attempt to remove
checkerboard artifacts used an anti-aliased final resize plus a sub-pixel
phase jitter. It backfired: the GAN's thin lines broke into beading/dots and
flat areas stippled (v1 GAN 35k right-hair high-freq 19.31 -> v2 GAN 45k
24.23; lpips 0.3174 -> 0.4020). The lesson: `nearest-exact` (aliased) LR is
what teaches the aliased->smooth mapping. The anti-alias change and the
sub-pixel jitter default were reverted; aliased LR is kept (ICNR init stays,
it is harmless).

**2. Capacity was the real lever.** Scaling the GAN (exact v1 recipe, aliased
LR, no jitter; bigger net warm-started from the v1 dim48 fidelity via
width+block expansion — verified ~equivalent, 81.85 dB):

| model | params | topiq @35k | lpips @35k | line high-freq (right/bangs) |
|---|---|---|---|---|
| v1 GAN dim48/b16 | 0.69M | 0.4065 | 0.3174 | 19.31 / 15.54 |
| dim64/b24 | 1.80M | **0.4124** | **0.3136** | **18.64 / 14.82** |

More capacity improved perceptual metrics **and** reduced line artifacts, and
the 1.80M model had not plateaued at 40k while the 0.69M one plateaued at 35k.

**Reference sizes:** SRVGGNetCompact default 0.62M / x4v3 1.21M / ultracompact
0.33M; SPAN-S 2.24M / SPAN base 2.62M / SPAN-F64 3.96M. NERVE 1.80M is in the
SPAN class while staying conv-only.

**Running:** 1.80M to 150k, then dim64/b32 (2.39M) to 150k on the same
recipe/start/iters, to produce the scaling curve; next point if needed is
dim96/b24 (4.03M). Deferred: dedicated aliasing augmentation in the OTF
pipeline, and WebP compression (jpeg already generalizes to webp/screenshot
DCT). The 1.80M config (dim64/b24) is now the arch default; the production chain is
training at 1.80M and will be exported/benchmarked (NCNN latency) before publish.
