#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 04:00:00
#SBATCH -A lt200394
#SBATCH -J mc2single_enc_cross
#SBATCH -a 0-13%10
#SBATCH -o /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%A_%a.out
#SBATCH -e /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%A_%a.out

# Cross-dataset train-probe: encode mc2wit_e32 on CG-OE + Bloom, and mc2cg_e32 on WIT +
# Bloom. Outputs go into the existing emb/mc2wit_e32/ and emb/mc2cg_e32/ dirs alongside
# the in-domain .npz files, so phase B re-run will extend their train_probe_summary.csv.
# Encoder auto-skips existing .npz -> safe to resubmit.

set -euo pipefail
source /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/env.sh
module load Mamba/23.11.0-0
source activate mc2_eval_env

ENC=/lustrefs/disk/project/lt200394-thllmV/kd_dataset/scripts/encode_train_embeddings.py
EMB_ROOT=/lustrefs/disk/project/lt200394-thllmV/kd_dataset/eval/train_probe/emb

# mc2wit_e32 probed on CG-OE (6 lang slices) + Bloom (all)
# mc2cg_e32  probed on WIT  (6 lang slices) + Bloom (all)
JOBS=(
  "mc2wit_e32:cgoe:id"
  "mc2wit_e32:cgoe:jv"
  "mc2wit_e32:cgoe:ms"
  "mc2wit_e32:cgoe:su"
  "mc2wit_e32:cgoe:th"
  "mc2wit_e32:cgoe:vi"
  "mc2wit_e32:bloom:"
  "mc2cg_e32:wit:id"
  "mc2cg_e32:wit:jv"
  "mc2cg_e32:wit:ms"
  "mc2cg_e32:wit:my"
  "mc2cg_e32:wit:th"
  "mc2cg_e32:wit:vi"
  "mc2cg_e32:bloom:"
)

IFS=':' read -r TAG DS LANG <<< "${JOBS[$SLURM_ARRAY_TASK_ID]}"
case "$TAG" in
  mc2wit_e32) CKPT="$MC2WIT_E32_CKPT" ;;
  mc2cg_e32)  CKPT="$MC2CG_E32_CKPT" ;;
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
