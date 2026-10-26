#!/usr/bin/env bash
# Round 2: 4 engine-ablation runs, sequential (one GPU)
set -uo pipefail
cd /home/phips/Documents/traiNNer-redux
PY=.venv/bin/python
export PYTHONUNBUFFERED=1

for f in 4x_paired_sharpened_dists_pop 4x_paired_gan_dists 4x_paired_gan_convnext 4x_paired_gan_r3d; do
  echo "=== START $f $(date) ==="
  $PY train.py -opt "options/train/NERVE/$f.yml" 2>&1 | tee -a engine_ablation.log
  echo "=== END $f rc=$? $(date) ==="
done
echo "=== ROUND2 COMPLETE $(date) ==="
