#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 04:00:00
#SBATCH -A lt200394
#SBATCH -J wc_h14
#SBATCH -o /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%j.out
#SBATCH -e /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%j.out

# WorldCuisines VQA eval for MetaCLIP2-ViT-H-14-worldwide-quickgelu (upper-bound reference).
# Loaded via open_clip pretrained registry (metaclip2_worldwide). Uses mc2_eval_env.
# Images must be pre-downloaded via download_worldcuisines_images.py on login node.

set -euo pipefail
source /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/env.sh

module load Mamba/23.11.0-0
source activate mc2_eval_env

TAG="metaclip2"
MODEL="ViT-H-14-worldwide-quickgelu"
PRETRAINED="metaclip2_worldwide"
WC_IMAGES="/project/lt200394-thllmV/kd_dataset/eval/worldcuisines"
RESULTS="${CB_ROOT}/runs/results/${TAG}"
PREDS="${RESULTS}/preds"
mkdir -p "$RESULTS" "$PREDS"

echo "Job start: $(date)  TAG=$TAG  MODEL=$MODEL  PRETRAINED=$PRETRAINED"

for TASK in task1 task2; do
    OUT="${RESULTS}/worldcuisines_${TASK}_test_large_${TAG}.json"
    PRED="${PREDS}/worldcuisines_${TASK}_test_large_${TAG}.jsonl"
    if [[ -f "$OUT" ]]; then
        echo "  skip (exists): $(basename "$OUT")"
        continue
    fi
    stage "WorldCuisines $TASK (test_large) — $TAG"
    timed_run "worldcuisines_${TASK}_${TAG}" \
        python "${CB_ROOT}/runs/eval_worldcuisines.py" \
            --model "$MODEL" \
            --pretrained "$PRETRAINED" \
            --wc_images_dir "$WC_IMAGES" \
            --task "$TASK" \
            --split "test_large" \
            --batch_size 64 \
            --save_predictions "$PRED" \
            --output "$OUT"
done

echo "Job done: $(date)  TAG=$TAG"
