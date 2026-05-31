#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 03:00:00
#SBATCH -A lt200394
#SBATCH -J cb_sweep_mc2_${SLURM_ARRAY_TASK_ID}
#SBATCH -a 1-5
#SBATCH -o /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%A_%a.out
#SBATCH -e /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%A_%a.out

# Full eval suite for Habibi's LATEST run (metaclip2_kd): ViT-T-16 student distilled
# from the multilingual MetaCLIP2-ViT-B-16-worldwide teacher on the 3-dataset SEA blend
# (cultural-ground + WIT + bloom). Student is CLIP-BPE (vocab 49408 ctx 77), so we eval
# with arch name "ViT-T-16" RESOLVED FROM HABIBI'S OWN open_clip via env mc2_eval_env
# (NOT open_clip_edit, where ViT-T-16 is the SigLIP2 256000 config).
# Array: 1=epoch_0(pre-KD init), 2=epoch_8, 3=epoch_16, 4=epoch_24, 5=epoch_32.

set -euo pipefail
source /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/env.sh

module load Mamba/23.11.0-0
source activate mc2_eval_env

case "$SLURM_ARRAY_TASK_ID" in
    1) TAG="mc2_e0";  MODEL="ViT-T-16"; CKPT="$MC2_E0_CKPT" ;;
    2) TAG="mc2_e8";  MODEL="ViT-T-16"; CKPT="$MC2_E8_CKPT" ;;
    3) TAG="mc2_e16"; MODEL="ViT-T-16"; CKPT="$MC2_E16_CKPT" ;;
    4) TAG="mc2_e24"; MODEL="ViT-T-16"; CKPT="$MC2_E24_CKPT" ;;
    5) TAG="mc2_e32"; MODEL="ViT-T-16"; CKPT="$MC2_E32_CKPT" ;;
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
