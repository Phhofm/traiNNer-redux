# Ablations

NERVE 4× loss-engine ablations. Everything matched unless noted: patch 64 LR,
batch 4, seed 1024, torch.compile on, EMA on, 30k iters, warm-start from the
official `4x_NERVE_release.safetensors` (1.78M, dim 64 / 24 blocks), Lucid
paired Catmull-Rom LR train data, **Lucid-val** (bundled CC0, 10 tiles, lr_x4)
every 10k. RGB metrics (PSNR/SSIM), crop_border 4.

Trained on an RTX 3060 12 GB, torch 2.11.0+cu128.

> **Caveats:** Lucid-val is same-domain (tiles drawn from the Lucid train pool)
> and 30k is a short horizon — these numbers rank the engines, they are not
> release claims. Visual inspection and an OTF confirmation run for the winner
> are still required.

## Round 1 — is the discriminator needed at all?

| Metric (↑/↓) | S4 Sharpened-DISTS (no D) | S5 TOPIQ-NR (no D) | S0 GAN (dunet+VGG19) |
|---|---|---|---|
| PSNR ↑ | **25.48** | 25.49 | 25.21 |
| SSIM ↑ | 0.7296 | **0.7299** | 0.7305 |
| TOPIQ ↑ | 0.688 | 0.684 | **0.719** |
| LPIPS ↓ | 0.170 | 0.191 | **0.137** |
| DISTS ↓ | 0.122 | 0.128 | **0.120** |

Findings:

- **S4 beats S5 everywhere except a PSNR tie** — TOPIQ-NR-as-loss underperforms
  DISTS-as-loss, and it is the reward-hackable option. S5 is dropped.
- **S4 wins fidelity** (+0.27 dB over GAN at equal SSIM/DISTS) and trains
  ~2× faster with no discriminator memory.
- **S0 (GAN) wins pure perceptual scores** (LPIPS 0.137, TOPIQ 0.719) — the
  "pop" — with the artifact risk that comes with it.
- Round 2 exists to answer: can a better D or a modern feature extractor close
  that pop gap, and can hotter S4 weights close it without a D.

## Round 2 — running

Configs on this branch, same protocol:

| Run | Change vs S0/S4 | Question |
|---|---|---|
| `4x_paired_gan_r3d` | dunet → **R3GANDisc** (0.31M, SN, no BN) | does a modern light D beat dunet? |
| `4x_paired_gan_convnext` | VGG19 → **ConvNeXt-N** perceptual | kills the watercolor look? |
| `4x_paired_gan_dists` | VGG19 → **DISTS** perceptual | DISTS features vs VGG features |
| `4x_paired_sharpened_dists_pop` | S4 with DISTS 0.4→**0.8**, FF 0.15→**0.3** | can S4 gain pop without a D? |

*(results to be filled in)*

## Implementation notes (bug fixes made for these experiments)

- `traiNNer/losses/contextual_loss.py`: the VGG path accumulated into an
  **integer** tensor (`torch.tensor(0, …)`) and crashed with
  `Float can't be cast to Long` — it could never run as written upstream.
  Fixed to float. Gotcha: `layer_weights` keys must be `relu_3_1` (underscore
  before the block number) — `relu3_1` is silently dropped by
  `alt_layers_names` and the loss returns 0.
- `traiNNer/losses/convnext_perceptual_loss.py`: added
  `@torch.amp.custom_fwd(cast_inputs=torch.float32)` like the other
  feature-losses (fp16-autocast safety).
- `traiNNer/losses/topiq_nr_loss.py` (new): frozen CFANet-NR (KonIQ),
  maximizes the MOS; unit-tested for gradient flow.
- `r3gan_disc_arch.py` docstring corrected: **R1 is not wired**; SN only.
