#!/bin/bash
# PRID: Prior Regulation via In-context Distillation
# Usage: bash scripts/run_prid.sh [MODEL] [API_BASE] [API_KEY]
#
# Examples:
#   GPT-4o:  bash scripts/run_prid.sh gpt-4o "" $OPENAI_API_KEY
#   Gemma:   bash scripts/run_prid.sh gemma3:12b http://localhost:11434/v1/ ollama

set -e

MODEL=${1:-"gpt-4o"}
API_BASE=${2:-""}
API_KEY=${3:-""}

OUTPUT="data/PRID-${MODEL//[:\/]/-}/prid_train.json"

echo "============================================"
echo "PRID: Prior Regulation via In-context Distillation"
echo "  Model:   $MODEL"
echo "  Output:  $OUTPUT"
echo "============================================"

CMD="python src/seed/prid_distill.py \
    --model $MODEL \
    --train_data data/raw/chebi-20/train.txt \
    --output $OUTPUT \
    --temperature 0.6 \
    --max_tokens 4096"

[ -n "$API_BASE" ] && CMD="$CMD --api_base $API_BASE"
[ -n "$API_KEY" ] && CMD="$CMD --api_key $API_KEY"

eval $CMD
