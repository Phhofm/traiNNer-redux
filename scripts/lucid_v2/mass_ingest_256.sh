#!/bin/bash
# LUCID v2: Mass Ingest 256px Migration
# ====================================
# This script tiles and filters 18 datasets to 256px @ >= 0.50 Complexity.

set -e

# --- CONFIGURATION ---
REPO_ROOT="/home/phips/Documents/GitHub/traiNNer-redux"
INGEST_SCRIPT="$REPO_ROOT/scripts/lucid_v2/01_ingest.py"
ICNET_MODEL="$REPO_ROOT/datasets/preparation/complexity/complexity.pth"
OUTPUT_BASE="/media/phips/One Touch/Upscale/TrainDataset/lucid/256x256"
THRESHOLD="0.50"
TILE_SIZE="256"
# Set workers lower for HDD to prevent seek-thrashing
# Setting to 1 for absolute HDD safety and to prevent seek-thrashing
WORKERS="1"

# Ensure Output Base Exists
mkdir -p "$OUTPUT_BASE"

# --- DATASET LIST ---
INPUT_DIRS=(
    "/media/phips/One Touch/datasets/df2k/DF2K_train_HR"
    "/media/phips/One Touch/datasets/cc0/uhdiqatraining"
    "/media/phips/One Touch/datasets/cc0/unsplashlite"
    "/media/phips/One Touch/datasets/cc0/LIU4K‑v2/training"
    "/media/phips/One Touch/Upscale/TrainDataset/HQ50K/HQ50K_HR"
    "/media/phips/One Touch/Upscale/TrainDataset/laion/laion"
    "/media/phips/One Touch/Upscale/TrainDataset/BHI/BHI_HR"
    "/media/phips/One Touch/Upscale/TrainDataset/FFHQ/images1024x1024"
    "/media/phips/One Touch/Upscale/TrainDataset/LSDIR/HR"
    "/media/phips/One Touch/Upscale/TrainDataset/nomos8k_sfw/nomos8k_sfw"
    "/media/phips/One Touch/Upscale/TrainDataset/nomos_uni"
    "/media/phips/One Touch/Upscale/TrainDataset/wip-bhi/unfiltered_but_tiled/COCO2017_train_512"
    "/media/phips/One Touch/Upscale/TrainDataset/wip-bhi/unfiltered_but_tiled/COCO2017_unlabeled_512"
    "/media/phips/One Touch/Upscale/TrainDataset/wip-bhi/unfiltered_but_tiled/inaturalist_2019"
    "/home/phips/Documents/dataset/elite-complex-050"
    "/home/phips/Documents/dataset/PDM/OSISRD/v3/hr"
    "/home/phips/Documents/dataset/cc0/hr"
    "/media/phips/One Touch/Upscale/TrainDataset/exposure_correction/hr"
)

echo "--- Starting Ultimate Elite 256px Migration ---"
echo "Output: $OUTPUT_BASE"
echo "Threshold: $THRESHOLD | Tile: ${TILE_SIZE}px"
echo "------------------------------------------------"

for INPUT in "${INPUT_DIRS[@]}"; do
    if [ -d "$INPUT" ]; then
        echo "Processing: $INPUT"
        # lower priority for safety
        nice -n 15 python3 "$INGEST_SCRIPT" \
            --input "$INPUT" \
            --output "$OUTPUT_BASE" \
            --icnet "$ICNET_MODEL" \
            --threshold "$THRESHOLD" \
            --tile_size "$TILE_SIZE" \
            --workers "$WORKERS" \
            --batch 64 \
            --resume \
            --multiscale
        echo "Finished: $INPUT"
        echo "------------------------------------------------"
    else
        echo "WARNING: Path not found, skipping: $INPUT"
    fi
done

echo "--- Mass Ingest Complete! ---"
echo "Next Step: Run 02_dedupe.py on the unified tiles folder."
