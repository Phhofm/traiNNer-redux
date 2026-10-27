#!/usr/bin/env bash
# Round 5: S10 — the actual R3GAN recipe (no SN, R1)
set -uo pipefail
cd /home/phips/Documents/traiNNer-redux
export PYTHONUNBUFFERED=1
.venv/bin/python train.py -opt options/train/NERVE/4x_paired_gan_r3gan_true.yml 2>&1 | tee -a engine_ablation.log
echo "=== ROUND5 (S10) COMPLETE rc=$? $(date) ==="
