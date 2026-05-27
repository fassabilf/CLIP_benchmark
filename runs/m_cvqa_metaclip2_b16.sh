#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 02:00:00
#SBATCH -A lt200394
#SBATCH -J cb_cvqa_metaclip2_b16
#SBATCH -o /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%j.out
#SBATCH -e /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%j.out

set -euo pipefail
source /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/env.sh

module load Mamba/23.11.0-0
source activate mteb_env2

TAG="metaclip2_b16"
MODEL="facebook/metaclip-2-worldwide-b16"
MODEL_TYPE="hf_transformers"
RESULTS="${CB_ROOT}/runs/results/${TAG}"
PREDS="${RESULTS}/preds"
mkdir -p "$PREDS"
OUT="${RESULTS}/cvqa_${TAG}.json"

python "${CB_ROOT}/runs/eval_cvqa.py" \
    --model_type "$MODEL_TYPE" \
    --model "$MODEL" --pretrained "" \
    --cache_dir "$HF_HUB_CACHE" \
    --batch_size 64 \
    --save_predictions "${PREDS}/cvqa_${TAG}.jsonl" \
    --output "$OUT"

echo "DONE cvqa $TAG"
