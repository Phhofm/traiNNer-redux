#!/usr/bin/env bash
# Round 4: gamma 0.1 + SN-only control
set -uo pipefail
cd /home/phips/Documents/traiNNer-redux
export PYTHONUNBUFFERED=1
PY=.venv/bin/python

for f in 4x_paired_gan_r3d_convnext_r1_g01 4x_paired_gan_r3d_convnext_noreg; do
  echo "=== START $f $(date) ==="
  $PY train.py -opt "options/train/NERVE/$f.yml" 2>&1 | tee -a engine_ablation.log
  echo "=== END $f rc=$? $(date) ==="
done
echo "=== ROUND4 COMPLETE $(date) ==="
