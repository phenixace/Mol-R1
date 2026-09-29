#!/bin/bash
# Mol-R1 reasoning self-improvement loop: Imitate (SFT) → Explore (GRPO) → Harvest → repeat
# Usage: bash scripts/run_rsi.sh [BASE_MODEL] [MAX_ITER] [WORK_DIR] [extra src/rsi/loop.py flags...]
#
# Examples:
#   bash scripts/run_rsi.sh meta-llama/Llama-3.1-8B-Instruct 3 outputs/rsi
#   bash scripts/run_rsi.sh meta-llama/Llama-3.1-8B-Instruct 3 outputs/rsi \
#       --eval_data data/raw/chebi-20/validation.txt
#
# Hyper-parameter defaults follow the paper; see `python src/rsi/loop.py --help`.
# Re-running the same command resumes an interrupted run.

set -e

BASE_MODEL=${1:-"meta-llama/Llama-3.1-8B-Instruct"}
MAX_ITER=${2:-3}
WORK_DIR=${3:-"outputs/rsi"}
shift $(( $# < 3 ? $# : 3 ))

python src/rsi/loop.py \
    --base_model "$BASE_MODEL" \
    --seed_data data/MoIA/T0_zh.json \
    --train_data data/raw/chebi-20/train.txt \
    --work_dir "$WORK_DIR" \
    --max_iterations "$MAX_ITER" \
    "$@"
