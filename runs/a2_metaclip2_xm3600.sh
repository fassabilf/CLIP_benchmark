#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 02:00:00
#SBATCH -A lt200394
#SBATCH -J cb_metaclip2_xm3600
#SBATCH -o /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%j.out
#SBATCH -e /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%j.out

set -euo pipefail
source /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/env.sh

module load Mamba/23.11.0-0
source activate mteb_env2

MODEL="ViT-H-14-worldwide-quickgelu"
PRETRAINED="metaclip2_worldwide"
TAG="metaclip2"
RESULTS="${CB_ROOT}/runs/results/${TAG}"
mkdir -p "$RESULTS"

# 8 langs: SEA focus + en. XM3600 supports id,th,vi (no jv/ms/my/su).
# Use all supported ones from our SEA set + en/zh as reference.
for LANG in en id th vi zh; do
    OUT="${RESULTS}/xm3600_${LANG}_${TAG}.json"
    if [[ -f "$OUT" ]]; then
        echo "skip (exists): $OUT"; continue
    fi
    echo "=== XM3600 lang=$LANG ==="
    python -m clip_benchmark.cli eval \
        --model "$MODEL" --pretrained "$PRETRAINED" \
        --model_cache_dir "$HF_HUB_CACHE" \
        --dataset crossmodal3600 \
        --dataset_root "$EVAL_ROOT" \
        --language "$LANG" \
        --task zeroshot_retrieval \
        --batch_size 64 --num_workers 8 \
        --output "$OUT" \
        --recall_k 1 5 10
done

echo "DONE A2"
