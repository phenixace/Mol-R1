#!/bin/bash
# Harvest with a trained policy on the training set: one shard per GPU, then merge.
# Usage: bash scripts/run_harvest.sh MODEL_DIR OUTPUT_DIR [MAX_ATTEMPTS] [NUM_GPUS] [extra harvest.py flags...]
#
# Example (K=8 samples per instance on 8 GPUs):
#   bash scripts/run_harvest.sh outputs/rsi/iter_2/rl/final outputs/harvest_T3 8 8
#
# Writes OUTPUT_DIR/harvest.json (trace set) and harvest_meta.json, then prints coverage,
# the first-try rate and pass@k of the policy on the training set. Sampling defaults
# (temperature 0.6, top-p 0.9) are those of src/rsi/harvest.py.

set -e

MODEL_DIR=$1
OUTPUT_DIR=$2
MAX_ATTEMPTS=${3:-8}
NUM_GPUS=${4:-8}
shift $(( $# < 4 ? $# : 4 ))
TRAIN_DATA=data/raw/chebi-20/train.txt

if [ -z "$MODEL_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: bash scripts/run_harvest.sh MODEL_DIR OUTPUT_DIR [MAX_ATTEMPTS] [NUM_GPUS]" >&2
    exit 1
fi
mkdir -p "$OUTPUT_DIR"

pids=()
for ((i = 0; i < NUM_GPUS; i++)); do
    CUDA_VISIBLE_DEVICES=$i python src/rsi/harvest.py \
        --model_dir "$MODEL_DIR" \
        --train_data $TRAIN_DATA \
        --output_dir "$OUTPUT_DIR" \
        --shard_id $i \
        --num_shards $NUM_GPUS \
        --max_attempts $MAX_ATTEMPTS \
        --samples_per_round $MAX_ATTEMPTS \
        "$@" > "$OUTPUT_DIR/shard_$i.log" 2>&1 &
    pids+=($!)
done
for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "Shard $i failed; see $OUTPUT_DIR/shard_$i.log" >&2; exit 1; }
done

python src/rsi/harvest.py --merge --train_data $TRAIN_DATA --output_dir "$OUTPUT_DIR" --num_shards $NUM_GPUS
python src/rsi/stats.py --train_data $TRAIN_DATA "$OUTPUT_DIR/harvest.json" --json_out "$OUTPUT_DIR/stats.json"
