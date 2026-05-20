#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 01:30:00
#SBATCH -A lt200394
#SBATCH -J cb_sweep_e_s${SLURM_ARRAY_TASK_ID}
#SBATCH -a 1-3
#SBATCH -o /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%A_%a.out
#SBATCH -e /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%A_%a.out

set -euo pipefail
source /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/env.sh

module load Mamba/23.11.0-0
source activate mteb_env2

case "$SLURM_ARRAY_TASK_ID" in
    1) TAG="s1"; MODEL="ViT-T-16-clipbpe"; CKPT="$S1_CKPT" ;;
    2) TAG="s2"; MODEL="ViT-T-16";         CKPT="$S2_CKPT" ;;
    3) TAG="s3"; MODEL="ViT-T-16";         CKPT="$S3_CKPT" ;;
esac

RESULTS="${CB_ROOT}/runs/results/${TAG}"
mkdir -p "$RESULTS"

echo "=== $TAG: Babel-ImageNet (8 langs SEA+en) ==="
for LANG in en id jv ms my su th vi; do
    OUT="${RESULTS}/babel_imagenet_${LANG}_${TAG}.json"
    if [[ -f "$OUT" ]]; then echo "skip $OUT"; continue; fi
    python -m clip_benchmark.cli eval \
        --model "$MODEL" --pretrained "$CKPT" \
        --batch_size 512 --num_workers 8 \
        --dataset babel_imagenet \
        --dataset_root "$IMAGENET_ROOT" \
        --task zeroshot_classification --language "$LANG" \
        --output "$OUT" \
        || echo "  (lang $LANG failed; continuing)"
done

echo "DONE $TAG babel"
