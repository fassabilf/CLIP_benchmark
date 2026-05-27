#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 02:00:00
#SBATCH -A lt200394
#SBATCH -J cb_cvqa_clipkd_b16_laion
#SBATCH -o /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%j.out
#SBATCH -e /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%j.out

set -euo pipefail
source /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/env.sh

module load Mamba/23.11.0-0
source activate mteb_env2

TAG="clipkd_b16_laion"
MODEL="ViT-B-16"
PRETRAINED="/project/lt200394-thllmV/benchmark/pretrained/clipkd_released/ViT_B_16-laion400m_e32.pt"
RESULTS="${CB_ROOT}/runs/results/${TAG}"
PREDS="${RESULTS}/preds"
mkdir -p "$PREDS"
OUT="${RESULTS}/cvqa_${TAG}.json"

python "${CB_ROOT}/runs/eval_cvqa.py" \
    --model_type open_clip \
    --model "$MODEL" --pretrained "$PRETRAINED" \
    --batch_size 64 \
    --save_predictions "${PREDS}/cvqa_${TAG}.jsonl" \
    --output "$OUT"

echo "DONE cvqa $TAG"
