#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 04:00:00
#SBATCH -A lt200394
#SBATCH -J mc2_enc_probe
#SBATCH -a 0-64%10
#SBATCH -o /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%A_%a.out
#SBATCH -e /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%A_%a.out

# Phase A (parallel): one array task per (checkpoint x dataset[-lang]). Each task = 1 GPU + 16
# cores + 15 decode workers, so many GPUs encode concurrently and the dataloader is well-fed
# (the bottleneck is JPEG decode, not the tiny ViT-T-16). 26 tasks, <=10 concurrent (%10).
# Encoder auto-skips existing .npz -> safe to resubmit.

set -euo pipefail
source /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/env.sh
module load Mamba/23.11.0-0
source activate mc2_eval_env

ENC=/lustrefs/disk/project/lt200394-thllmV/kd_dataset/scripts/encode_train_embeddings.py
EMB_ROOT=/lustrefs/disk/project/lt200394-thllmV/kd_dataset/eval/train_probe/emb

# Full epoch curve: e0 (init) -> e8 -> e16 -> e24 -> e32 (final). Already-encoded shards skip.
TAGS=(mc2_e0 mc2_e8 mc2_e16 mc2_e24 mc2_e32)
# 13 dataset entries per tag: "dataset:lang" (lang empty -> whole dataset, used for bloom).
DSENTRIES=(
  "bloom:" \
  "cgoe:id" "cgoe:jv" "cgoe:ms" "cgoe:su" "cgoe:th" "cgoe:vi" \
  "wit:id"  "wit:jv"  "wit:ms"  "wit:my"  "wit:th"  "wit:vi"
)
JOBS=()
for t in "${TAGS[@]}"; do for de in "${DSENTRIES[@]}"; do JOBS+=("$t:$de"); done; done

IFS=':' read -r TAG DS LANG <<< "${JOBS[$SLURM_ARRAY_TASK_ID]}"
case "$TAG" in
  mc2_e0)  CKPT="$MC2_E0_CKPT" ;;
  mc2_e8)  CKPT="$MC2_E8_CKPT" ;;
  mc2_e16) CKPT="$MC2_E16_CKPT" ;;
  mc2_e24) CKPT="$MC2_E24_CKPT" ;;
  mc2_e32) CKPT="$MC2_E32_CKPT" ;;
  *) echo "bad tag $TAG"; exit 1 ;;
esac

LANG_ARG=()
[[ -n "$LANG" ]] && LANG_ARG=(--langs "$LANG")

echo "task=$SLURM_ARRAY_TASK_ID tag=$TAG ds=$DS lang=${LANG:-ALL}"
echo "open_clip: $(python -c 'import open_clip; print(open_clip.__file__)')"
srun python "$ENC" \
    --ckpt "$CKPT" --tag "$TAG" --model ViT-T-16 \
    --datasets "$DS" "${LANG_ARG[@]}" \
    --batch-size 512 --num-workers 15 \
    --out-dir "$EMB_ROOT"
echo "DONE task $SLURM_ARRAY_TASK_ID"
