# Fork changes

Every deviation of this fork from [the-database/traiNNer-redux](https://github.com/the-database/traiNNer-redux),
with the reason. **If it's not in this file, we didn't change it.**

## Added

### `options/train/NERVE/`, `options/train/HEART/` — 6 release configs
Training configs for the HEART/NERVE models. The
[HEART / NERVE Trainer](https://github.com/Phhofm/heart-nerve_trainner) GUI
generates its runs from these (paired-fidelity, OTF-fidelity, OTF-GAN per
model/scale).

- Source: ported from `Phhofm/trainner-redux-playground` (research configs,
  reduced from 29 to the 6 the GUI exposes).
- Tested: all 6 parse with this repo's config schema, and the app's self-test
  (train → validate → save → export) runs green against this tree.
- Notes: `manual_seed` is commented out in the OTF configs (a random seed is
  drawn per run); `pretrain_network_g` is `~` — the GUI injects the official
  (or the user's) fidelity pretrain at generation time.

### `scripts/heart/export_minimal.py`, `scripts/nerve/export_dynamic.py`
Checkpoint → ONNX exporters used by the trainer app (fp32/fp16 and ncnn paths).
- Tested: used by the app's export step on real checkpoints.

### `HEART.md`, `NERVE.md`, `NERVE_ABLATION_LOG.md`
Design + ablation documentation for the two architectures.

## Changed

### `schemas/redux-config.schema.json`
Regenerated with `scripts/options/generate_schema.py` so the merged
`nerve` / `heart` architectures appear in the editor schema and pass
config validation.

### `traiNNer/metrics/lpips.py`, `dists.py`, `topiq.py`
Cache the metric network per device instead of rebuilding it for every
validation image — validation calls these once per image, so the rebuild
dominated validation time.
- Tested: app self-test and app test suite.

### `options/train/*/…otf_gan.yml`
Full validation metric set (PSNR/SSIM/LPIPS/DISTS/TOPIQ) instead of the
TOPIQ/LPIPS-only subset, so users see the complete quality picture.
`use_compile: true` — torch.compile now works through the GAN step (G + D
trained 40 iters without inductor errors; verified with torch 2.11 + cu128).

## Not ported (on purpose)

Research experiments from the playground snapshot (BOLT, DocRes, DRFT, NAFNet,
extra losses/schedulers, `single_gt_dataset`, …) — none are used by the trainer
app. Port branch-by-branch if ever needed.
