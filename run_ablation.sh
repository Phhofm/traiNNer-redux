#!/usr/bin/env bash
# NERVE 4x engine ablation: Sharpened-DISTS vs TOPIQ-NR vs GAN (sequential, one GPU)
set -uo pipefail
cd /home/phips/Documents/traiNNer-redux
PY=.venv/bin/python
export PYTHONUNBUFFERED=1

for f in 4x_paired_sharpened_dists 4x_paired_topiq_nr 4x_paired_gan; do
  echo "=== START $f $(date) ==="
  $PY train.py -opt "options/train/NERVE/$f.yml" 2>&1 | tee -a engine_ablation.log
  rc=$?
  echo "=== END $f rc=$rc $(date) ==="
done
echo "=== ABLATION COMPLETE $(date) ==="
