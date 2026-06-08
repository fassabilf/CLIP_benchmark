#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 04:00:00
#SBATCH -A lt200394
#SBATCH -J mc2cc_enc_probe
#SBATCH -a 0-64%10
#SBATCH -o /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%A_%a.out
#SBATCH -e /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%A_%a.out

# Phase A (parallel): encode CC12M model (mc2cc) over the same SEA training shards used
# for the mc2 (SEA-blend) probe — lets MrPing compare CC12M vs multilingual on in-domain
# data at gallery=10000. One array task per (checkpoint x dataset[-lang]), <=10 concurrent.
# Encoder auto-skips existing .npz -> safe to resubmit.

set -euo pipefail
source /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/env.sh
module load Mamba/23.11.0-0
source activate mc2_eval_env

ENC=/lustrefs/disk/project/lt200394-thllmV/kd_dataset/scripts/encode_train_embeddings.py
EMB_ROOT=/lustrefs/disk/project/lt200394-thllmV/kd_dataset/eval/train_probe/emb

# mc2cc_e0 reuses the same init checkpoint as mc2_e0 (same random init, pre-KD).
TAGS=(mc2cc_e0 mc2cc_e8 mc2cc_e16 mc2cc_e24 mc2cc_e32)
DSENTRIES=(
  "bloom:" \
  "cgoe:id" "cgoe:jv" "cgoe:ms" "cgoe:su" "cgoe:th" "cgoe:vi" \
  "wit:id"  "wit:jv"  "wit:ms"  "wit:my"  "wit:th"  "wit:vi"
)
JOBS=()
for t in "${TAGS[@]}"; do for de in "${DSENTRIES[@]}"; do JOBS+=("$t:$de"); done; done

IFS=':' read -r TAG DS LANG <<< "${JOBS[$SLURM_ARRAY_TASK_ID]}"
case "$TAG" in
  mc2cc_e0)  CKPT="$MC2_E0_CKPT" ;;
  mc2cc_e8)  CKPT="$MC2CC_E8_CKPT" ;;
  mc2cc_e16) CKPT="$MC2CC_E16_CKPT" ;;
  mc2cc_e24) CKPT="$MC2CC_E24_CKPT" ;;
  mc2cc_e32) CKPT="$MC2CC_E32_CKPT" ;;
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
