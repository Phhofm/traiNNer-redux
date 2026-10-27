#!/usr/bin/env bash
# Round 5 (corrected): S10b — R3D no-SN + relativistic + R1/R2 gamma 0.05 + ConvNeXt
set -uo pipefail
cd /home/phips/Documents/traiNNer-redux
export PYTHONUNBUFFERED=1
.venv/bin/python train.py -opt options/train/NERVE/4x_paired_gan_r3gan_correct.yml 2>&1 | tee -a engine_ablation.log
echo "=== ROUND5 (S10b) COMPLETE rc=$? $(date) ==="
