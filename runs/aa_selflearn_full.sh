#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 06:00:00
#SBATCH -A lt200394
#SBATCH -J selflearn_e${SLURM_ARRAY_TASK_ID}
#SBATCH -a 1-4
#SBATCH -o /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%A_%a.out
#SBATCH -e /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%A_%a.out

set -euo pipefail
source /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/env.sh

module load Mamba/23.11.0-0
source activate mteb_env2

MODEL="ViT-T-16"
FLORES_LANGS=(eng_Latn ind_Latn jav_Latn zsm_Latn mya_Mymr sun_Latn tha_Thai vie_Latn)

case "$SLURM_ARRAY_TASK_ID" in
    1) EPOCH=8;  CKPT="$SELFLEARN_MAMMOTH_V1_E8_CKPT"  ;;
    2) EPOCH=16; CKPT="$SELFLEARN_MAMMOTH_V1_E16_CKPT" ;;
    3) EPOCH=24; CKPT="$SELFLEARN_MAMMOTH_V1_E24_CKPT" ;;
    4) EPOCH=32; CKPT="$SELFLEARN_MAMMOTH_V1_E32_CKPT" ;;
    *) echo "unknown array idx"; exit 1 ;;
esac

TAG="selflearn_mammoth_v1_e${EPOCH}"
RESULTS="${CB_ROOT}/runs/results/${TAG}"
PREDS="${RESULTS}/preds"
mkdir -p "$RESULTS" "$PREDS"

run() {
    local OUT="$1"; shift
    if [[ -f "$OUT" ]]; then echo "skip (exists): $OUT"; return 0; fi
    timed_run "$(basename "$OUT" .json)" \
        python -m clip_benchmark.cli eval \
            --model "$MODEL" --pretrained "$CKPT" \
            --batch_size 512 --num_workers 8 \
            --save_predictions "$PREDS" \
            --output "$OUT" \
            "$@"
}

stage "$TAG: ImageNet-1k (en)"
run "${RESULTS}/imagenet1k_${TAG}.json" \
    --dataset imagenet1k-unverified \
    --dataset_root "$IMAGENET_ROOT" \
    --task zeroshot_classification --language en

stage "$TAG: Babel-IN (8 langs)"
for LANG in en id jv ms my su th vi; do
    run "${RESULTS}/babel_imagenet_${LANG}_${TAG}.json" \
        --dataset babel_imagenet \
        --dataset_root "$IMAGENET_ROOT" \
        --task zeroshot_classification --language "$LANG" \
        || echo "  (lang $LANG failed; continuing)"
done

stage "$TAG: XM3600 retrieval (en/id/th/vi)"
for LANG in en id th vi; do
    run "${RESULTS}/xm3600_${LANG}_${TAG}.json" \
        --dataset crossmodal3600 \
        --dataset_root "$EVAL_ROOT" \
        --task zeroshot_retrieval --language "$LANG" \
        --recall_k 1 5 10
done

stage "$TAG: Flickr30k-200 (8 langs)"
for LANG in "${FLORES_LANGS[@]}"; do
    run "${RESULTS}/flickr30k_200_${LANG}_${TAG}.json" \
        --dataset flickr30k-200 \
        --dataset_root "$EVAL_ROOT" \
        --task zeroshot_retrieval --language "$LANG" \
        --recall_k 1 5 10 \
        || echo "  (lang $LANG failed; continuing)"
done

stage "$TAG: XTD-200 (8 langs)"
for LANG in "${FLORES_LANGS[@]}"; do
    run "${RESULTS}/xtd200_${LANG}_${TAG}.json" \
        --dataset xtd200 \
        --dataset_root "$EVAL_ROOT" \
        --task zeroshot_retrieval --language "$LANG" \
        --recall_k 1 5 10 \
        || echo "  (lang $LANG failed; continuing)"
done

stage "$TAG: ALL DONE"
echo "Results in: $RESULTS"
echo "Run: python3 runs/compute_per_lang_mean.py $TAG"
