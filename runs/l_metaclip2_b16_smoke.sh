#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 00:45:00
#SBATCH -A lt200394
#SBATCH -J cb_metaclip2_b16_smoke
#SBATCH -o /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%j.out
#SBATCH -e /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%j.out

set -uo pipefail
source /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/env.sh

module load Mamba/23.11.0-0
source activate mteb_env2

TAG="metaclip2_b16"
MODEL="facebook/metaclip-2-worldwide-b16"
MODEL_TYPE="hf_transformers"
RESULTS="${CB_ROOT}/runs/results/${TAG}"
PREDS="${RESULTS}/preds"
mkdir -p "$PREDS"

# B/16 is ~10x smaller than H/14 — A100-40GB can take a much larger batch.
BS=512

echo "=== smoke: XM3600 en retrieval ==="
python -m clip_benchmark.cli eval \
    --model_type "$MODEL_TYPE" \
    --model "$MODEL" --pretrained "" \
    --model_cache_dir "$HF_HUB_CACHE" \
    --dataset crossmodal3600 \
    --dataset_root "$EVAL_ROOT" \
    --task zeroshot_retrieval \
    --language en \
    --batch_size "$BS" --num_workers 8 \
    --recall_k 1 5 10 \
    --save_predictions "$PREDS" \
    --output "${RESULTS}/xm3600_en_${TAG}.json"

echo "DONE smoke"
ls -la "$RESULTS" "$PREDS"
