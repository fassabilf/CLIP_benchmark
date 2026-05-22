#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 02:00:00
#SBATCH -A lt200394
#SBATCH -J cb_sweep_e8r_${SLURM_ARRAY_TASK_ID}
#SBATCH -a 1-2
#SBATCH -o /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%A_%a.out
#SBATCH -e /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%A_%a.out

# Fill-in for the e8 early-epoch sweep: g_sweep_e8.sh only ran imagenet1k +
# babel_imagenet + xm3600, so flickr30k-200 / xtd200 / cvqa cells were empty
# for s1_e8 and s3_e8. This adds exactly those. Array 1 = s1_e8, 2 = s3_e8.

set -euo pipefail
source /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/env.sh

module load Mamba/23.11.0-0
source activate mteb_env2

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
