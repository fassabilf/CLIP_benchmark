#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 02:00:00
#SBATCH -A lt200394
#SBATCH -J cb_cvqa_${SLURM_ARRAY_TASK_ID}
#SBATCH -a 1-4
#SBATCH -o /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%A_%a.out
#SBATCH -e /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%A_%a.out

set -euo pipefail
source /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/env.sh

module load Mamba/23.11.0-0
source activate mteb_env2

case "$SLURM_ARRAY_TASK_ID" in
    1) TAG="s1";        MODEL="ViT-T-16-clipbpe";          PRETRAINED="$S1_CKPT";          BS=128 ;;
    2) TAG="s2";        MODEL="ViT-T-16";                  PRETRAINED="$S2_CKPT";          BS=128 ;;
    3) TAG="s3";        MODEL="ViT-T-16";                  PRETRAINED="$S3_CKPT";          BS=128 ;;
    4) TAG="metaclip2"; MODEL="ViT-H-14-worldwide-quickgelu"; PRETRAINED="metaclip2_worldwide"; BS=16 ;;
esac

OUT="${CB_ROOT}/runs/results/${TAG}/cvqa_${TAG}.json"
mkdir -p "$(dirname "$OUT")"

python "${CB_ROOT}/runs/eval_cvqa.py" \
    --model "$MODEL" --pretrained "$PRETRAINED" \
    --cache_dir "$HF_HUB_CACHE" \
    --batch_size "$BS" \
    --output "$OUT"

echo "DONE $TAG"
