#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 02:00:00
#SBATCH -A lt200394
#SBATCH -J cb_sweep_d_s${SLURM_ARRAY_TASK_ID}
#SBATCH -a 1-3
#SBATCH -o /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%A_%a.out
#SBATCH -e /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%A_%a.out

set -euo pipefail
source /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/env.sh

module load Mamba/23.11.0-0
source activate mteb_env2

case "$SLURM_ARRAY_TASK_ID" in
    1) TAG="s1"; MODEL="ViT-T-16-clipbpe"; CKPT="$S1_CKPT" ;;
    2) TAG="s2"; MODEL="ViT-T-16";         CKPT="$S2_CKPT" ;;
    3) TAG="s3"; MODEL="ViT-T-16";         CKPT="$S3_CKPT" ;;
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

echo "DONE $TAG"
