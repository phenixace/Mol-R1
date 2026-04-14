#!/bin/bash
# MoIA: Molecular Iterative Adaptation
# SFT → RPO → Rejection-Sampling Distill → repeat
# Usage: bash scripts/run_moia.sh [INITIAL_DATA] [MODEL] [MAX_ITER]

set -e

INITIAL_DATA=${1:-"data/MoIA/T0.json"}
MODEL=${2:-"facebook/galactica-125m"}
MAX_ITER=${3:-5}
SFT_GPUS=${4:-2}
RPO_GPUS=${5:-8}
DISTILL_MODEL=${6:-"gemma3:12b"}
WORK_DIR=${7:-"outputs/moia"}

echo "============================================"
echo "MoIA: Molecular Iterative Adaptation"
echo "  Initial data:    $INITIAL_DATA"
echo "  Base model:      $MODEL"
echo "  Max iterations:  $MAX_ITER"
echo "  SFT GPUs:        $SFT_GPUS"
echo "  RPO GPUs:        $RPO_GPUS"
echo "  Distill model:   $DISTILL_MODEL"
echo "  Work dir:        $WORK_DIR"
echo "============================================"

python src/distill/moia.py \
    --base_model $MODEL \
    --initial_data $INITIAL_DATA \
    --raw_train_data data/raw/chebi-20/train.txt \
    --test_data data/raw/chebi-20/test.txt \
    --work_dir $WORK_DIR \
    --max_iterations $MAX_ITER \
    --max_total_steps 100000 \
    --min_new_data_ratio 0.01 \
    --sft_epochs 5 \
    --sft_lr 1e-5 \
    --sft_batch_size 1 \
    --sft_grad_accum 4 \
    --sft_max_seq_len 2048 \
    --rpo_epochs 2 \
    --rpo_lr 1e-6 \
    --rpo_num_generations 5 \
    --rpo_temperature 1.0 \
    --rpo_kl_coef 0.01 \
    --rpo_global_batch_size 128 \
    --rpo_rollout_batch_size 512 \
    --rpo_micro_batch_size 4 \
    --distill_model $DISTILL_MODEL \
    --distill_score_threshold 8 \
    --num_gpus $RPO_GPUS \
    --use_model_predictions
