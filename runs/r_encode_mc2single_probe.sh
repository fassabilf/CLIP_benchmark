#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 04:00:00
#SBATCH -A lt200394
#SBATCH -J mc2single_enc_probe
#SBATCH -a 0-12%10
#SBATCH -o /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%A_%a.out
#SBATCH -e /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%A_%a.out

# Phase A: encode the 3 single-dataset metaclip2_kd v2 models (mc2wit / mc2cg / mc2bloom)
# each on its own training dataset only (per PumeTu: "just use the training dataset it was
# actually trained on"). Epoch_32 only for initial comparison. 13 tasks total.
# Encoder auto-skips existing .npz -> safe to resubmit.

set -euo pipefail
source /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/env.sh
module load Mamba/23.11.0-0
source activate mc2_eval_env

ENC=/lustrefs/disk/project/lt200394-thllmV/kd_dataset/scripts/encode_train_embeddings.py
EMB_ROOT=/lustrefs/disk/project/lt200394-thllmV/kd_dataset/eval/train_probe/emb

# mc2wit -> WIT only (6 lang slices)
# mc2cg  -> CulturalGround-OE only (6 lang slices)
# mc2bloom -> Bloom only (1 entry, all langs interleaved)
JOBS=(
  "mc2wit_e32:wit:id"
  "mc2wit_e32:wit:jv"
  "mc2wit_e32:wit:ms"
  "mc2wit_e32:wit:my"
  "mc2wit_e32:wit:th"
  "mc2wit_e32:wit:vi"
  "mc2cg_e32:cgoe:id"
  "mc2cg_e32:cgoe:jv"
  "mc2cg_e32:cgoe:ms"
  "mc2cg_e32:cgoe:su"
  "mc2cg_e32:cgoe:th"
  "mc2cg_e32:cgoe:vi"
  "mc2bloom_e32:bloom:"
)

IFS=':' read -r TAG DS LANG <<< "${JOBS[$SLURM_ARRAY_TASK_ID]}"
case "$TAG" in
  mc2wit_e32)   CKPT="$MC2WIT_E32_CKPT" ;;
  mc2cg_e32)    CKPT="$MC2CG_E32_CKPT" ;;
  mc2bloom_e32) CKPT="$MC2BLOOM_E32_CKPT" ;;
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
