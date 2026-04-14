#!/bin/bash
# Single-iteration pipeline: SFT → Predict → Evaluate
# Usage: bash scripts/run_sft_eval.sh [DATA_NAME] [MODEL] [NUM_GPUS]
#   DATA_NAME: PRID-4o | PRID-G | RS-G | MoIA-T0 | MoIA-T1 | MoIA-T2

set -e

DATA_NAME=${1:-"RS-G"}
MODEL=${2:-"facebook/galactica-125m"}
NUM_GPUS=${3:-2}

case $DATA_NAME in
    "PRID-4o")  DATA_PATH="data/PRID-4o/prid_4o_train.json" ;;
    "PRID-G")   DATA_PATH="data/PRID-G/prid_g_train.json" ;;
    "RS-G")     DATA_PATH="data/RS-G/rs_g_train.json" ;;
    "MoIA-T0")  DATA_PATH="data/MoIA/T0.json" ;;
    "MoIA-T1")  DATA_PATH="data/MoIA/T1.json" ;;
    "MoIA-T2")  DATA_PATH="data/MoIA/T2.json" ;;
    *)          DATA_PATH=$DATA_NAME ;;
esac

OUTPUT_DIR="outputs/sft_${DATA_NAME}"

echo "============================================"
echo "SFT Training + Evaluation Pipeline"
echo "  Data:   $DATA_PATH"
echo "  Model:  $MODEL"
echo "  Output: $OUTPUT_DIR"
echo "  GPUs:   $NUM_GPUS"
echo "============================================"

# Step 1: SFT Training
torchrun --nproc_per_node=$NUM_GPUS src/train/sft_train.py \
    --mode cot \
    --model_name $MODEL \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --epochs 5 \
    --lr 1e-5 \
    --batch_size 1 \
    --grad_accum 4 \
    --max_seq_len 2048

# Step 2: Prediction
torchrun --nproc_per_node=$NUM_GPUS src/eval/predict.py \
    --mode cot \
    --model_dir $OUTPUT_DIR/final \
    --test_path data/raw/chebi-20/test.txt \
    --output_path $OUTPUT_DIR/predictions.txt \
    --temperature 0.6 \
    --top_p 0.9 \
    --max_tokens 10000

# Step 3: Evaluation
python src/eval/evaluate.py $OUTPUT_DIR/predictions.txt
