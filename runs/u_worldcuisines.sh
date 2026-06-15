#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 04:00:00
#SBATCH -A lt200394
#SBATCH -J wc_eval_${SLURM_ARRAY_TASK_ID}
#SBATCH -a 1-5
#SBATCH -o /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%A_%a.out
#SBATCH -e /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%A_%a.out

# WorldCuisines VQA eval for student checkpoints (ViT-T-16 CLIP-BPE).
# Array: 1=e0(init)  2=mc2-v1-e32  3=mc2-cc12m-e32  4=abl-WIT-e32  5=abl-CG-e32
# Images must be pre-downloaded via download_worldcuisines_images.py on login node.

set -euo pipefail
source /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/env.sh

module load Mamba/23.11.0-0
source activate mc2_eval_env

case "$SLURM_ARRAY_TASK_ID" in
    1) TAG="mc2_e0";    CKPT="$MC2_E0_CKPT" ;;
    2) TAG="mc2_e32";   CKPT="$MC2_E32_CKPT" ;;
    3) TAG="mc2cc_e32"; CKPT="$MC2CC_E32_CKPT" ;;
    4) TAG="mc2wit_e32"; CKPT="$MC2WIT_E32_CKPT" ;;
    5) TAG="mc2cg_e32"; CKPT="$MC2CG_E32_CKPT" ;;
    *) echo "unknown array idx $SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac

MODEL="ViT-T-16"
WC_IMAGES="/project/lt200394-thllmV/kd_dataset/eval/worldcuisines"
RESULTS="${CB_ROOT}/runs/results/${TAG}"
PREDS="${RESULTS}/preds"
mkdir -p "$RESULTS" "$PREDS"

echo "Job start: $(date)  TAG=$TAG  CKPT=$CKPT"

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
            --pretrained "$CKPT" \
            --wc_images_dir "$WC_IMAGES" \
            --task "$TASK" \
            --split "test_large" \
            --batch_size 64 \
            --save_predictions "$PRED" \
            --output "$OUT"
done

echo "Job done: $(date)  TAG=$TAG"
