#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 03:00:00
#SBATCH -A lt200394
#SBATCH -J wc_eval_b16
#SBATCH -o /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%j.out
#SBATCH -e /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%j.out

# WorldCuisines VQA eval for MetaCLIP2-B16-worldwide teacher (reference model).
# Uses hf_transformers model type via mteb_env2 (same env as other B16 teacher evals).
# Images must be pre-downloaded via download_worldcuisines_images.py on login node.

set -euo pipefail
source /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/env.sh

module load Mamba/23.11.0-0
source activate mteb_env2

TAG="metaclip2_b16"
MODEL="facebook/metaclip-2-worldwide-b16"
MODEL_TYPE="hf_transformers"
WC_IMAGES="/project/lt200394-thllmV/kd_dataset/eval/worldcuisines"
RESULTS="${CB_ROOT}/runs/results/${TAG}"
PREDS="${RESULTS}/preds"
mkdir -p "$RESULTS" "$PREDS"

echo "Job start: $(date)  TAG=$TAG"

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
            --pretrained "" \
            --model_type "$MODEL_TYPE" \
            --wc_images_dir "$WC_IMAGES" \
            --task "$TASK" \
            --split "test_large" \
            --batch_size 64 \
            --save_predictions "$PRED" \
            --output "$OUT"
done

echo "Job done: $(date)  TAG=$TAG"
