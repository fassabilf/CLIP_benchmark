#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 03:00:00
#SBATCH -A lt200394
#SBATCH -J cb_sweep_mc2single_${SLURM_ARRAY_TASK_ID}
#SBATCH -a 1-12
#SBATCH -o /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%A_%a.out
#SBATCH -e /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%A_%a.out

# Full eval suite for Habibi's 3 single-dataset metaclip2_kd v2 runs (Jun 2026):
# ViT-T-16 (CLIP-BPE) <- MetaCLIP2-B16-worldwide, random student init, each trained on
# one multilingual dataset: WIT (idx 1-4), CulturalGround-OE (idx 5-8), Bloom (idx 9-12).
# Eval in mc2_eval_env (habibi's open_clip, ViT-T-16 = CLIP-BPE config).
# Array: 1-4=mc2wit e8/16/24/32, 5-8=mc2cg e8/16/24/32, 9-12=mc2bloom e8/16/24/32.

set -euo pipefail
source /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/env.sh

module load Mamba/23.11.0-0
source activate mc2_eval_env

case "$SLURM_ARRAY_TASK_ID" in
    1)  TAG="mc2wit_e8";    MODEL="ViT-T-16"; CKPT="$MC2WIT_E8_CKPT" ;;
    2)  TAG="mc2wit_e16";   MODEL="ViT-T-16"; CKPT="$MC2WIT_E16_CKPT" ;;
    3)  TAG="mc2wit_e24";   MODEL="ViT-T-16"; CKPT="$MC2WIT_E24_CKPT" ;;
    4)  TAG="mc2wit_e32";   MODEL="ViT-T-16"; CKPT="$MC2WIT_E32_CKPT" ;;
    5)  TAG="mc2cg_e8";     MODEL="ViT-T-16"; CKPT="$MC2CG_E8_CKPT" ;;
    6)  TAG="mc2cg_e16";    MODEL="ViT-T-16"; CKPT="$MC2CG_E16_CKPT" ;;
    7)  TAG="mc2cg_e24";    MODEL="ViT-T-16"; CKPT="$MC2CG_E24_CKPT" ;;
    8)  TAG="mc2cg_e32";    MODEL="ViT-T-16"; CKPT="$MC2CG_E32_CKPT" ;;
    9)  TAG="mc2bloom_e8";  MODEL="ViT-T-16"; CKPT="$MC2BLOOM_E8_CKPT" ;;
    10) TAG="mc2bloom_e16"; MODEL="ViT-T-16"; CKPT="$MC2BLOOM_E16_CKPT" ;;
    11) TAG="mc2bloom_e24"; MODEL="ViT-T-16"; CKPT="$MC2BLOOM_E24_CKPT" ;;
    12) TAG="mc2bloom_e32"; MODEL="ViT-T-16"; CKPT="$MC2BLOOM_E32_CKPT" ;;
    *) echo "unknown array idx"; exit 1 ;;
esac

RESULTS="${CB_ROOT}/runs/results/${TAG}"
PREDS="${RESULTS}/preds"
mkdir -p "$RESULTS" "$PREDS"

run() {
    local OUT="$1"; shift
    if [[ -f "$OUT" ]]; then echo "  skip (exists): $(basename "$OUT")"; return 0; fi
    timed_run "$(basename "$OUT" .json)" \
        python -m clip_benchmark.cli eval \
            --model "$MODEL" --pretrained "$CKPT" \
            --batch_size 512 --num_workers 8 \
            --save_predictions "$PREDS" \
            --output "$OUT" \
            "$@"
}

echo "Job start: $(date)  TAG=$TAG  CKPT=$CKPT"

stage "$TAG: ImageNet-1k val (en)"
run "${RESULTS}/imagenet1k_${TAG}.json" \
    --dataset imagenet1k-unverified \
    --dataset_root "$IMAGENET_ROOT" \
    --task zeroshot_classification --language en

stage "$TAG: Babel-ImageNet (8 SEA+en)"
for LANG in en id jv ms my su th vi; do
    run "${RESULTS}/babel_imagenet_${LANG}_${TAG}.json" \
        --dataset babel_imagenet \
        --dataset_root "$IMAGENET_ROOT" \
        --task zeroshot_classification --language "$LANG" \
        || echo "  (lang $LANG not supported or failed; continuing)"
done

stage "$TAG: XM3600 retrieval (id/th/vi + en/zh)"
for LANG in en id th vi zh; do
    run "${RESULTS}/xm3600_${LANG}_${TAG}.json" \
        --dataset crossmodal3600 \
        --dataset_root "$EVAL_ROOT" \
        --task zeroshot_retrieval --language "$LANG" \
        --recall_k 1 5 10
done

LANGS=(eng_Latn ind_Latn jav_Latn zsm_Latn mya_Mymr sun_Latn tha_Thai vie_Latn)

stage "$TAG: Flickr30k-200 (8 langs)"
for LANG in "${LANGS[@]}"; do
    run "${RESULTS}/flickr30k_200_${LANG}_${TAG}.json" \
        --dataset flickr30k-200 \
        --dataset_root "$EVAL_ROOT" \
        --task zeroshot_retrieval --language "$LANG" \
        --recall_k 1 5 10 \
        || echo "  (lang $LANG failed; continuing)"
done

stage "$TAG: XTD-200 (8 langs)"
for LANG in "${LANGS[@]}"; do
    run "${RESULTS}/xtd200_${LANG}_${TAG}.json" \
        --dataset xtd200 \
        --dataset_root "$EVAL_ROOT" \
        --task zeroshot_retrieval --language "$LANG" \
        --recall_k 1 5 10 \
        || echo "  (lang $LANG failed; continuing)"
done

stage "$TAG: CVQA (4-way MC)"
CVQA_OUT="${RESULTS}/cvqa_${TAG}.json"
if [[ -f "$CVQA_OUT" ]]; then
    echo "  skip (exists): $(basename "$CVQA_OUT")"
else
    timed_run "cvqa" python "${CB_ROOT}/runs/eval_cvqa.py" \
        --model "$MODEL" --pretrained "$CKPT" \
        --cache_dir "$HF_HUB_CACHE" \
        --batch_size 128 \
        --save_predictions "${PREDS}/cvqa_${TAG}_pred.jsonl" \
        --output "$CVQA_OUT"
fi

stage "$TAG: DONE"
