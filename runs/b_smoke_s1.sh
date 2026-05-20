#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 01:00:00
#SBATCH -A lt200394
#SBATCH -J cb_smoke_s1
#SBATCH -o /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%j.out
#SBATCH -e /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%j.out

set -euo pipefail
source /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/env.sh

module load Mamba/23.11.0-0
source activate mteb_env2

MODEL="ViT-T-16-clipbpe"
TAG="s1"
RESULTS="${CB_ROOT}/runs/results/${TAG}"
mkdir -p "$RESULTS"

echo "=== Smoke: ImageNet-1k val (en) ==="
python -m clip_benchmark.cli eval \
    --model "$MODEL" --pretrained "$S1_CKPT" \
    --dataset imagenet1k-unverified \
    --dataset_root "$IMAGENET_ROOT" \
    --task zeroshot_classification \
    --language en \
    --batch_size 512 --num_workers 8 \
    --output "${RESULTS}/imagenet1k_${TAG}.json"

echo "=== Smoke: XM3600 en ==="
python -m clip_benchmark.cli eval \
    --model "$MODEL" --pretrained "$S1_CKPT" \
    --dataset crossmodal3600 \
    --dataset_root "$EVAL_ROOT" \
    --task zeroshot_retrieval \
    --language en \
    --batch_size 512 --num_workers 8 \
    --recall_k 1 5 10 \
    --output "${RESULTS}/xm3600_en_${TAG}.json"

echo "DONE B"
echo "Expected: ImageNet-1k acc1 ~= 0.428 (per MKD_MODELS.md)"
