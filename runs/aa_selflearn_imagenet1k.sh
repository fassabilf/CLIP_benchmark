#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 01:00:00
#SBATCH -A lt200394
#SBATCH -J selflearn_imagenet1k
#SBATCH -o /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%j.out
#SBATCH -e /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%j.out

set -euo pipefail
source /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/env.sh

module load Mamba/23.11.0-0
source activate mteb_env2

TAG="selflearn_mammoth_v1"
MODEL="ViT-T-16"
CKPT="$SELFLEARN_MAMMOTH_V1_E32_CKPT"

RESULTS="${CB_ROOT}/runs/results/${TAG}"
mkdir -p "$RESULTS"

run() {
    local OUT="$1"; shift
    if [[ -f "$OUT" ]]; then echo "skip (exists): $OUT"; return 0; fi
    timed_run "$(basename "$OUT" .json)" \
        python -m clip_benchmark.cli eval \
            --model "$MODEL" --pretrained "$CKPT" \
            --batch_size 512 --num_workers 8 \
            --output "$OUT" \
            "$@"
}

stage "$TAG: ImageNet-1k (en)"
run "${RESULTS}/imagenet1k_${TAG}.json" \
    --dataset imagenet1k-unverified \
    --dataset_root "$IMAGENET_ROOT" \
    --task zeroshot_classification --language en

echo "DONE — results in $RESULTS"
echo "Expected top-1 ~0.30 based on training log (epoch 32 eval: 0.2998)"
