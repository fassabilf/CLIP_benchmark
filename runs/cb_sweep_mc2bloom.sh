#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 07:00:00
#SBATCH -A lt200394
#SBATCH -J cb_sweep_mc2bloom_${SLURM_ARRAY_TASK_ID}
#SBATCH -a 1-4
#SBATCH -o /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%A_%a.out
#SBATCH -e /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%A_%a.out

# Full eval suite for Habibi's metaclip2_kd ablation-Bloom run: ViT-T-16 student,
# RANDOM init (unlike v1/v2/v3 which continue from the English init), distilled from
# MetaCLIP2-ViT-B-16-worldwide, trained ONLY on Bloom (21,534 pairs, 76 SEA langs,
# manifest_bloom.csv). Same teacher / loss (clipkd) as abl-WIT, abl-CG, and v1/v2/v3.
# Student is CLIP-BPE (vocab 49408 ctx 77), eval with arch "ViT-T-16" from mc2_eval_env
# (habibi's open_clip, NOT open_clip_edit where ViT-T-16 is the SigLIP2 config).
# Array: 1=epoch_8, 2=epoch_16, 3=epoch_24, 4=epoch_32.
#
# WorldCuisines only runs on the epoch_32 task (array idx 4) — same pattern as the
# other models (Table 5 has no per-epoch WC breakdown, e32 only). That's why walltime
# is bumped to 7h: tasks 1-3 finish in ~3h, task 4 needs the extra ~4h for WC on top.
#
# NOTE: checkpoint filename pattern (epoch_N.pt) is assumed from the e0/init registry
# entry. Double-check against the actual contents of checkpoints/ before submitting —
# the fail-fast check below will catch a mismatch but won't tell you the right name.
# Images for WorldCuisines must be pre-downloaded via download_worldcuisines_images.py
# on the login node first (skip if already done for the other models — shared image dir).

set -euo pipefail
source /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/env.sh

module load Mamba/23.11.0-0
source activate mc2_eval_env

BLOOM_CKPT_DIR="/project/lt200394-thllmV/multilingual-clip-kd/open_clip/experiments/metaclip2_kd/clipkd_ViT-T-16_from_ViT-B-16-MetaClip_pretrained_v2_bloom/checkpoints"
WC_IMAGES="/project/lt200394-thllmV/kd_dataset/eval/worldcuisines"

case "$SLURM_ARRAY_TASK_ID" in
    1) EPOCH=8 ;;
    2) EPOCH=16 ;;
    3) EPOCH=24 ;;
    4) EPOCH=32 ;;
    *) echo "unknown array idx"; exit 1 ;;
esac

TAG="mc2bloom_e${EPOCH}"
MODEL="ViT-T-16"
CKPT="${BLOOM_CKPT_DIR}/epoch_${EPOCH}.pt"

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

if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: checkpoint not found at $CKPT"
    echo "Check ${BLOOM_CKPT_DIR} for the actual filename and fix the CKPT pattern above."
    exit 1
fi

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

# WorldCuisines — only at e32, same pattern as the other models (no per-epoch WC).
if [[ "$EPOCH" -eq 32 ]]; then
    for TASK in task1 task2; do
        stage "$TAG: WorldCuisines $TASK (test_large)"
        WC_OUT="${RESULTS}/worldcuisines_${TASK}_test_large_${TAG}.json"
        WC_PRED="${PREDS}/worldcuisines_${TASK}_test_large_${TAG}.jsonl"
        if [[ -f "$WC_OUT" ]]; then
            echo "  skip (exists): $(basename "$WC_OUT")"
        else
            timed_run "worldcuisines_${TASK}_${TAG}" \
                python "${CB_ROOT}/runs/eval_worldcuisines.py" \
                    --model "$MODEL" \
                    --pretrained "$CKPT" \
                    --wc_images_dir "$WC_IMAGES" \
                    --task "$TASK" \
                    --split "test_large" \
                    --batch_size 64 \
                    --save_predictions "$WC_PRED" \
                    --output "$WC_OUT"
        fi
    done
fi

stage "$TAG: DONE"
