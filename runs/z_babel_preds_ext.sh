#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 01:00:00
#SBATCH -A lt200394
#SBATCH -J babel_preds_ext_%a
#SBATCH -a 1-2
#SBATCH -o /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%A_%a.out
#SBATCH -e /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%A_%a.out

# Re-run Babel-IN classification WITH --save_predictions for external baselines
# (TinyCLIP, MobileCLIP2-S0) that are missing pred files.
# Skip condition is based on the PRED file, not the JSON.
#
# Array: 1=tinyclip  2=mobileclip2_s0
#
# Usage:
#   sbatch runs/z_babel_preds_ext.sh

set -euo pipefail
source /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/env.sh

module load Mamba/23.11.0-0
source activate mteb_env2

case "$SLURM_ARRAY_TASK_ID" in
    1)
        TAG="tinyclip"
        MODEL_ARGS=(--model "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M")
        BATCH=256
        ;;
    2)
        TAG="mobileclip2_s0"
        MODEL_ARGS=(--model_type open_clip --model "MobileCLIP2-S0" --pretrained "dfndr2b")
        BATCH=256
        ;;
    *) echo "unknown array idx $SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac

RESULTS="${CB_ROOT}/runs/results/${TAG}"
PREDS="${RESULTS}/preds"
mkdir -p "$RESULTS" "$PREDS"

# Skip if PRED file already exists (JSON may already exist — overwrite is fine)
run_preds() {
    local PRED_FILE="$1"
    local OUT="$2"
    shift 2
    if [[ -f "$PRED_FILE" ]]; then
        echo "  [skip] $(basename "$PRED_FILE") already exists"
        return 0
    fi
    timed_run "$(basename "$OUT" .json)" \
        python -m clip_benchmark.cli eval \
            "${MODEL_ARGS[@]}" \
            --batch_size "$BATCH" --num_workers 8 \
            --save_predictions "$PREDS" \
            --output "$OUT" \
            "$@"
}

echo "Job start: $(date)  TAG=$TAG"

stage "$TAG: Babel-IN (8 langs) — saving topk predictions"
for LANG in en id jv ms my su th vi; do
    run_preds \
        "${PREDS}/babel_imagenet_${LANG}_${TAG}_pred.jsonl" \
        "${RESULTS}/babel_imagenet_${LANG}_${TAG}.json" \
        --dataset babel_imagenet \
        --dataset_root "$IMAGENET_ROOT" \
        --task zeroshot_classification --language "$LANG" \
        || echo "  (lang $LANG failed; continuing)"
done

stage "$TAG: DONE — pred files in $PREDS"
