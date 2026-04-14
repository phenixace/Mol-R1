#!/bin/bash
# Full pipeline: SFT → RPO → Predict → Evaluate
# Usage: bash scripts/run_sft_rpo.sh [DATA_NAME] [MODEL] [NUM_GPUS]

set -e

DATA_NAME=${1:-"RS-G"}
MODEL=${2:-"facebook/galactica-125m"}
NUM_GPUS=${3:-2}
RPO_GPUS=${4:-8}

case $DATA_NAME in
    "PRID-4o")  DATA_PATH="data/PRID-4o/prid_4o_train.json" ;;
    "PRID-G")   DATA_PATH="data/PRID-G/prid_g_train.json" ;;
    "RS-G")     DATA_PATH="data/RS-G/rs_g_train.json" ;;
    "MoIA-T0")  DATA_PATH="data/MoIA/T0.json" ;;
    "MoIA-T1")  DATA_PATH="data/MoIA/T1.json" ;;
    "MoIA-T2")  DATA_PATH="data/MoIA/T2.json" ;;
    *)          DATA_PATH=$DATA_NAME ;;
esac

SFT_DIR="outputs/sft_${DATA_NAME}"
RPO_DIR="outputs/rpo_${DATA_NAME}"

echo "============================================"
echo "SFT + RPO Training Pipeline"
echo "  Data:      $DATA_PATH"
echo "  Model:     $MODEL"
echo "  SFT GPUs:  $NUM_GPUS"
echo "  RPO GPUs:  $RPO_GPUS"
echo "============================================"

# Step 1: SFT
torchrun --nproc_per_node=$NUM_GPUS src/train/sft_train.py \
    --mode cot \
    --model_name $MODEL \
    --data_path $DATA_PATH \
    --output_dir $SFT_DIR \
    --epochs 5 \
    --lr 1e-5 \
    --batch_size 1 \
    --grad_accum 4 \
    --max_seq_len 2048

# Step 2: RPO
torchrun --nproc_per_node=$RPO_GPUS src/train/grpo_train.py \
    --sft_model_dir $SFT_DIR/final \
    --data_path $DATA_PATH \
    --output_dir $RPO_DIR \
    --epochs 2 \
    --lr 1e-6 \
    --num_generations 5 \
    --temperature 1.0 \
    --beta 0.01 \
    --global_batch_size 128 \
    --rollout_batch_size 512 \
    --micro_batch_size 4

# Step 3: Predict
torchrun --nproc_per_node=$RPO_GPUS src/eval/predict.py \
    --mode cot \
    --model_dir $RPO_DIR/final \
    --test_path data/raw/chebi-20/test.txt \
    --output_path $RPO_DIR/predictions.txt \
    --temperature 0.6 \
    --top_p 0.9 \
    --max_tokens 10000

# Step 4: Evaluate
python src/eval/evaluate.py $RPO_DIR/predictions.txt
