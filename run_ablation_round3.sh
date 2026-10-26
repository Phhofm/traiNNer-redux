#!/usr/bin/env bash
# Round 3: S6 (modern R3GAN + R1) and S7 (S6 from S4b curriculum), sequential
set -uo pipefail
cd /home/phips/Documents/traiNNer-redux
export PYTHONUNBUFFERED=1
PY=.venv/bin/python

for f in 4x_paired_gan_r3d_convnext_r1 4x_paired_gan_r3d_convnext_r1_froms4b; do
  echo "=== START $f $(date) ==="
  $PY train.py -opt "options/train/NERVE/$f.yml" 2>&1 | tee -a engine_ablation.log
  echo "=== END $f rc=$? $(date) ==="
done
echo "=== ROUND3 COMPLETE $(date) ==="
