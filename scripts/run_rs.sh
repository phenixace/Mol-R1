#!/bin/bash
# RS: Rejection-Sampling Distillation
# Usage: bash scripts/run_rs.sh [MODEL] [API_BASE]
#
# Examples:
#   Gemma:  bash scripts/run_rs.sh gemma3:12b http://localhost:11434/v1/

set -e

MODEL=${1:-"gemma3:12b"}
API_BASE=${2:-"http://localhost:11434/v1/"}
THRESHOLD=${3:-8}

OUTPUT="data/RS-${MODEL//[:\/]/-}/rs_train.json"

echo "============================================"
echo "RS: Rejection-Sampling Distillation"
echo "  Model:      $MODEL"
echo "  Threshold:  $THRESHOLD"
echo "  Output:     $OUTPUT"
echo "============================================"

python src/distill/rejection_sampling_distill.py \
    --model $MODEL \
    --api_base $API_BASE \
    --train_data data/raw/chebi-20/train.txt \
    --output $OUTPUT \
    --score_threshold $THRESHOLD
