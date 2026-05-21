#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 06:00:00
#SBATCH -A lt200394
#SBATCH -J cb_sweep_wit_${SLURM_ARRAY_TASK_ID}
#SBATCH -a 1-2
#SBATCH -o /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%A_%a.out
#SBATCH -e /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%A_%a.out

# Full eval suite for Habibi's WIT-multilingual run (clipkd ViT-T-16 ← ViT-B-16-SigLIP2,
# train_wit.csv, 32 epochs). Self-contained: merges c/d/e/f sweeps so the existing
# scripts stay untouched. Array 1 = epoch_8, 2 = epoch_32.

set -euo pipefail
source /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/env.sh

module load Mamba/23.11.0-0
source activate mteb_env2

case "$SLURM_ARRAY_TASK_ID" in
    1) TAG="wit_e8";  MODEL="ViT-T-16"; CKPT="$WIT_E8_CKPT" ;;
    2) TAG="wit_e32"; MODEL="ViT-T-16"; CKPT="$WIT_E32_CKPT" ;;
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

# SEA + en using FLORES codes
LANGS=(eng_Latn ind_Latn jav_Latn zsm_Latn mya_Mymr sun_Latn tha_Thai vie_Latn)

echo "=== $TAG: Flickr30k-200 (8 langs) ==="
for LANG in "${LANGS[@]}"; do
    run "${RESULTS}/flickr30k_200_${LANG}_${TAG}.json" \
        --dataset flickr30k-200 \
        --dataset_root "$EVAL_ROOT" \
        --task zeroshot_retrieval --language "$LANG" \
        --recall_k 1 5 10 \
        || echo "  (lang $LANG failed; continuing)"
done

echo "=== $TAG: XTD-200 (8 langs) ==="
for LANG in "${LANGS[@]}"; do
    run "${RESULTS}/xtd200_${LANG}_${TAG}.json" \
        --dataset xtd200 \
        --dataset_root "$EVAL_ROOT" \
        --task zeroshot_retrieval --language "$LANG" \
        --recall_k 1 5 10 \
        || echo "  (lang $LANG failed; continuing)"
done

echo "=== $TAG: CVQA (4-way MC) ==="
CVQA_OUT="${RESULTS}/cvqa_${TAG}.json"
if [[ -f "$CVQA_OUT" ]]; then
    echo "skip (exists): $CVQA_OUT"
else
    python "${CB_ROOT}/runs/eval_cvqa.py" \
        --model "$MODEL" --pretrained "$CKPT" \
        --cache_dir "$HF_HUB_CACHE" \
        --batch_size 128 \
        --output "$CVQA_OUT"
fi

echo "DONE $TAG"
