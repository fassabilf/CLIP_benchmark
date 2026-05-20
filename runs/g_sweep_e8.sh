#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 03:00:00
#SBATCH -A lt200394
#SBATCH -J cb_sweep_e8_${SLURM_ARRAY_TASK_ID}
#SBATCH -a 1-2
#SBATCH -o /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%A_%a.out
#SBATCH -e /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%A_%a.out

set -euo pipefail
source /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/env.sh

module load Mamba/23.11.0-0
source activate mteb_env2

# "e10"-equivalent early checkpoints. S1 & S3 have epoch_8 (closest to 10);
# S2 only has epoch_32 so it's not part of this sweep.
case "$SLURM_ARRAY_TASK_ID" in
    1) TAG="s1_e8"; MODEL="ViT-T-16-clipbpe"; CKPT="$S1_E8_CKPT" ;;
    2) TAG="s3_e8"; MODEL="ViT-T-16";         CKPT="$S3_E8_CKPT" ;;
    *) echo "unknown array idx"; exit 1 ;;
esac

RESULTS="${CB_ROOT}/runs/results/${TAG}"
mkdir -p "$RESULTS"

run() {
    local OUT="$1"; shift
    if [[ -f "$OUT" ]]; then echo "skip (exists): $OUT"; return 0; fi
    python -m clip_benchmark.cli eval \
        --model "$MODEL" --pretrained "$CKPT" \
        --batch_size 512 --num_workers 8 \
        --output "$OUT" \
        "$@"
}

echo "=== $TAG: ImageNet-1k val (en) ==="
run "${RESULTS}/imagenet1k_${TAG}.json" \
    --dataset imagenet1k-unverified \
    --dataset_root "$IMAGENET_ROOT" \
    --task zeroshot_classification --language en

echo "=== $TAG: Babel-ImageNet (8 SEA+en) ==="
for LANG in en id jv ms my su th vi; do
    run "${RESULTS}/babel_imagenet_${LANG}_${TAG}.json" \
        --dataset babel_imagenet \
        --dataset_root "$IMAGENET_ROOT" \
        --task zeroshot_classification --language "$LANG" \
        || echo "  (lang $LANG not supported or failed; continuing)"
done

echo "=== $TAG: XM3600 retrieval (id/th/vi + en/zh) ==="
for LANG in en id th vi zh; do
    run "${RESULTS}/xm3600_${LANG}_${TAG}.json" \
        --dataset crossmodal3600 \
        --dataset_root "$EVAL_ROOT" \
        --task zeroshot_retrieval --language "$LANG" \
        --recall_k 1 5 10
done

echo "DONE $TAG"
