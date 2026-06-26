#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 04:00:00
#SBATCH -A lt200394
#SBATCH -J mc2v3_enc_probe
#SBATCH -a 0-51%10
#SBATCH -o /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%A_%a.out
#SBATCH -e /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%A_%a.out

# Phase A: encode mc2-v3 checkpoints (e8/e16/e24/e32) over the same SEA training shards
# used for mc2 and mc2cc probes (bloom + cgoe×6 + wit×6 = 13 entries per epoch).
# 4 epochs × 13 entries = 52 tasks, <=10 concurrent. e0 init == mc2_e0, already encoded.
# Encoder auto-skips existing .npz -> safe to resubmit.
# After all tasks finish, run s_phase_b_retrieval.sh to get cos/R@1 for results_main.tex.

set -euo pipefail
source /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/env.sh
module load Mamba/23.11.0-0
source activate mc2_eval_env

ENC=/lustrefs/disk/project/lt200394-thllmV/kd_dataset/scripts/encode_train_embeddings.py
EMB_ROOT=/lustrefs/disk/project/lt200394-thllmV/kd_dataset/eval/train_probe/emb

TAGS=(mc2v3_e8 mc2v3_e16 mc2v3_e24 mc2v3_e32)
DSENTRIES=(
  "bloom:" \
  "cgoe:id" "cgoe:jv" "cgoe:ms" "cgoe:su" "cgoe:th" "cgoe:vi" \
  "wit:id"  "wit:jv"  "wit:ms"  "wit:my"  "wit:th"  "wit:vi"
)
JOBS=()
for t in "${TAGS[@]}"; do for de in "${DSENTRIES[@]}"; do JOBS+=("$t:$de"); done; done

IFS=':' read -r TAG DS LANG <<< "${JOBS[$SLURM_ARRAY_TASK_ID]}"
case "$TAG" in
  mc2v3_e8)  CKPT="$MC2V3_E8_CKPT" ;;
  mc2v3_e16) CKPT="$MC2V3_E16_CKPT" ;;
  mc2v3_e24) CKPT="$MC2V3_E24_CKPT" ;;
  mc2v3_e32) CKPT="$MC2V3_E32_CKPT" ;;
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
