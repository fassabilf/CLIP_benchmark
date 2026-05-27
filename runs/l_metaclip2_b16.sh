#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 06:00:00
#SBATCH -A lt200394
#SBATCH -J cb_metaclip2_b16
#SBATCH -o /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%j.out
#SBATCH -e /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%j.out

set -uo pipefail
source /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/env.sh

module load Mamba/23.11.0-0
source activate mteb_env2

TAG="metaclip2_b16"
MODEL="facebook/metaclip-2-worldwide-b16"
MODEL_TYPE="hf_transformers"
RESULTS="${CB_ROOT}/runs/results/${TAG}"
PREDS="${RESULTS}/preds"
mkdir -p "$PREDS"

# B/16 on A100-40GB
BS=512

run() {
    local OUT="$1"; shift
    if [[ -f "$OUT" ]]; then echo "skip $OUT"; return 0; fi
    python -m clip_benchmark.cli eval \
        --model_type "$MODEL_TYPE" \
        --model "$MODEL" --pretrained "" \
        --model_cache_dir "$HF_HUB_CACHE" \
        --batch_size "$BS" --num_workers 8 \
        --save_predictions "$PREDS" \
        --output "$OUT" \
        "$@" || echo "  (failed: $OUT)"
}

stage "ImageNet-1k val (en)"
run "${RESULTS}/imagenet1k_${TAG}.json" \
    --dataset imagenet1k-unverified \
    --dataset_root "$IMAGENET_ROOT" \
    --task zeroshot_classification --language en

stage "Babel-ImageNet (8 langs SEA+en)"
for LANG in en id jv ms my su th vi; do
    run "${RESULTS}/babel_imagenet_${LANG}_${TAG}.json" \
        --dataset babel_imagenet \
        --dataset_root "$IMAGENET_ROOT" \
        --task zeroshot_classification --language "$LANG"
done

stage "XM3600 retrieval (en + 4 SEA-relevant)"
for LANG in en id th vi zh; do
    run "${RESULTS}/xm3600_${LANG}_${TAG}.json" \
        --dataset crossmodal3600 \
        --dataset_root "$EVAL_ROOT" \
        --task zeroshot_retrieval --language "$LANG" \
        --recall_k 1 5 10
done

LANGS_FLORES=(eng_Latn ind_Latn jav_Latn zsm_Latn mya_Mymr sun_Latn tha_Thai vie_Latn)

stage "Flickr30k-200 (8 langs)"
for LANG in "${LANGS_FLORES[@]}"; do
    run "${RESULTS}/flickr30k_200_${LANG}_${TAG}.json" \
        --dataset flickr30k-200 \
        --dataset_root "$EVAL_ROOT" \
        --task zeroshot_retrieval --language "$LANG" \
        --recall_k 1 5 10
done

stage "XTD-200 (8 langs)"
for LANG in "${LANGS_FLORES[@]}"; do
    run "${RESULTS}/xtd200_${LANG}_${TAG}.json" \
        --dataset xtd200 \
        --dataset_root "$EVAL_ROOT" \
        --task zeroshot_retrieval --language "$LANG" \
        --recall_k 1 5 10
done

echo "DONE metaclip2_b16 full sweep"
