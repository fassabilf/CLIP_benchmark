#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 03:00:00
#SBATCH -A lt200394
#SBATCH -J clipkd_ablation4_${SLURM_ARRAY_TASK_ID}
#SBATCH -a 0-15
#SBATCH -o /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%A_%a.out
#SBATCH -e /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%A_%a.out

# Full eval suite for Asenthil's 4-way KD-loss ablation (all "Alldata": CC12M + full SEA
# multilingual blend + Bloom + mammoth_vl_sea, 12.72M samples, 32ep, ViT-T-16 <-
# MetaCLIP2-ViT-B-16-worldwide): ckd-only, icl-only, fd-only, no-KD baseline.
# Checkpoints are CLIP-BPE (vocab=49408) -> eval with mc2_eval_env / arch "ViT-T-16", same as
# clipkd_mammoth_only_v6 / clipkd_mammoth_bpe_v1 (NOT mteb_env2's SigLIP2 256000-vocab config).
# Array: 16 tasks = 4 models x 4 epochs (8/16/24/32). See runs/env.sh for the CKPT env vars.

set -euo pipefail
source /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/env.sh

module load Mamba/23.11.0-0
source activate mc2_eval_env

MODELS=(ckd_v1 fdonly_v1 icl_v1 nokd_v1)
EPOCHS=(8 16 24 32)

MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / 4 ))
EPOCH_IDX=$(( SLURM_ARRAY_TASK_ID % 4 ))
NAME="${MODELS[$MODEL_IDX]}"
EPOCH="${EPOCHS[$EPOCH_IDX]}"

TAG="clipkd_${NAME}_e${EPOCH}"
MODEL="ViT-T-16"

case "$NAME" in
    ckd_v1)    CKPT_VAR="CLIPKD_CKD_V1_E${EPOCH}_CKPT" ;;
    fdonly_v1) CKPT_VAR="CLIPKD_FDONLY_V1_E${EPOCH}_CKPT" ;;
    icl_v1)    CKPT_VAR="CLIPKD_ICL_V1_E${EPOCH}_CKPT" ;;
    nokd_v1)   CKPT_VAR="CLIPKD_NOKD_V1_E${EPOCH}_CKPT" ;;
    *) echo "unknown model $NAME"; exit 1 ;;
esac
CKPT="${!CKPT_VAR}"

if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: checkpoint not found at $CKPT (var $CKPT_VAR)"
    exit 1
fi

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
