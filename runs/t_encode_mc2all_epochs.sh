#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 04:00:00
#SBATCH -A lt200394
#SBATCH -J mc2single_enc_all
#SBATCH -a 0-103%10
#SBATCH -o /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%A_%a.out
#SBATCH -e /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%A_%a.out

# Full cross-dataset + multi-epoch train-probe for mc2wit and mc2cg.
# Each model is probed on ALL three datasets (wit / cgoe / bloom) at e8/16/24/32.
# Encoder auto-skips existing .npz -> safe to run alongside or after job 5845115.
# 104 tasks total (0-103).
#
# Layout:
#   0-51  : mc2wit e8/16/24/32 × {wit(6), cgoe(6), bloom(1)}
#   52-103: mc2cg  e8/16/24/32 × {cgoe(6), wit(6), bloom(1)}

set -euo pipefail
source /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/env.sh
module load Mamba/23.11.0-0
source activate mc2_eval_env

ENC=/lustrefs/disk/project/lt200394-thllmV/kd_dataset/scripts/encode_train_embeddings.py
EMB_ROOT=/lustrefs/disk/project/lt200394-thllmV/kd_dataset/eval/train_probe/emb

JOBS=(
  # mc2wit_e8 — wit (0-5), cgoe (6-11), bloom (12)
  "mc2wit_e8:wit:id"
  "mc2wit_e8:wit:jv"
  "mc2wit_e8:wit:ms"
  "mc2wit_e8:wit:my"
  "mc2wit_e8:wit:th"
  "mc2wit_e8:wit:vi"
  "mc2wit_e8:cgoe:id"
  "mc2wit_e8:cgoe:jv"
  "mc2wit_e8:cgoe:ms"
  "mc2wit_e8:cgoe:su"
  "mc2wit_e8:cgoe:th"
  "mc2wit_e8:cgoe:vi"
  "mc2wit_e8:bloom:"
  # mc2wit_e16 — wit (13-18), cgoe (19-24), bloom (25)
  "mc2wit_e16:wit:id"
  "mc2wit_e16:wit:jv"
  "mc2wit_e16:wit:ms"
  "mc2wit_e16:wit:my"
  "mc2wit_e16:wit:th"
  "mc2wit_e16:wit:vi"
  "mc2wit_e16:cgoe:id"
  "mc2wit_e16:cgoe:jv"
  "mc2wit_e16:cgoe:ms"
  "mc2wit_e16:cgoe:su"
  "mc2wit_e16:cgoe:th"
  "mc2wit_e16:cgoe:vi"
  "mc2wit_e16:bloom:"
  # mc2wit_e24 — wit (26-31), cgoe (32-37), bloom (38)
  "mc2wit_e24:wit:id"
  "mc2wit_e24:wit:jv"
  "mc2wit_e24:wit:ms"
  "mc2wit_e24:wit:my"
  "mc2wit_e24:wit:th"
  "mc2wit_e24:wit:vi"
  "mc2wit_e24:cgoe:id"
  "mc2wit_e24:cgoe:jv"
  "mc2wit_e24:cgoe:ms"
  "mc2wit_e24:cgoe:su"
  "mc2wit_e24:cgoe:th"
  "mc2wit_e24:cgoe:vi"
  "mc2wit_e24:bloom:"
  # mc2wit_e32 — wit (39-44), cgoe (45-50), bloom (51) [likely skip: already encoded]
  "mc2wit_e32:wit:id"
  "mc2wit_e32:wit:jv"
  "mc2wit_e32:wit:ms"
  "mc2wit_e32:wit:my"
  "mc2wit_e32:wit:th"
  "mc2wit_e32:wit:vi"
  "mc2wit_e32:cgoe:id"
  "mc2wit_e32:cgoe:jv"
  "mc2wit_e32:cgoe:ms"
  "mc2wit_e32:cgoe:su"
  "mc2wit_e32:cgoe:th"
  "mc2wit_e32:cgoe:vi"
  "mc2wit_e32:bloom:"
  # mc2cg_e8 — cgoe (52-57), wit (58-63), bloom (64)
  "mc2cg_e8:cgoe:id"
  "mc2cg_e8:cgoe:jv"
  "mc2cg_e8:cgoe:ms"
  "mc2cg_e8:cgoe:su"
  "mc2cg_e8:cgoe:th"
  "mc2cg_e8:cgoe:vi"
  "mc2cg_e8:wit:id"
  "mc2cg_e8:wit:jv"
  "mc2cg_e8:wit:ms"
  "mc2cg_e8:wit:my"
  "mc2cg_e8:wit:th"
  "mc2cg_e8:wit:vi"
  "mc2cg_e8:bloom:"
  # mc2cg_e16 — cgoe (65-70), wit (71-76), bloom (77)
  "mc2cg_e16:cgoe:id"
  "mc2cg_e16:cgoe:jv"
  "mc2cg_e16:cgoe:ms"
  "mc2cg_e16:cgoe:su"
  "mc2cg_e16:cgoe:th"
  "mc2cg_e16:cgoe:vi"
  "mc2cg_e16:wit:id"
  "mc2cg_e16:wit:jv"
  "mc2cg_e16:wit:ms"
  "mc2cg_e16:wit:my"
  "mc2cg_e16:wit:th"
  "mc2cg_e16:wit:vi"
  "mc2cg_e16:bloom:"
  # mc2cg_e24 — cgoe (78-83), wit (84-89), bloom (90)
  "mc2cg_e24:cgoe:id"
  "mc2cg_e24:cgoe:jv"
  "mc2cg_e24:cgoe:ms"
  "mc2cg_e24:cgoe:su"
  "mc2cg_e24:cgoe:th"
  "mc2cg_e24:cgoe:vi"
  "mc2cg_e24:wit:id"
  "mc2cg_e24:wit:jv"
  "mc2cg_e24:wit:ms"
  "mc2cg_e24:wit:my"
  "mc2cg_e24:wit:th"
  "mc2cg_e24:wit:vi"
  "mc2cg_e24:bloom:"
  # mc2cg_e32 — cgoe (91-96), wit (97-102), bloom (103) [likely skip: already encoded]
  "mc2cg_e32:cgoe:id"
  "mc2cg_e32:cgoe:jv"
  "mc2cg_e32:cgoe:ms"
  "mc2cg_e32:cgoe:su"
  "mc2cg_e32:cgoe:th"
  "mc2cg_e32:cgoe:vi"
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
  mc2wit_e8)  CKPT="$MC2WIT_E8_CKPT"  ;;
  mc2wit_e16) CKPT="$MC2WIT_E16_CKPT" ;;
  mc2wit_e24) CKPT="$MC2WIT_E24_CKPT" ;;
  mc2wit_e32) CKPT="$MC2WIT_E32_CKPT" ;;
  mc2cg_e8)   CKPT="$MC2CG_E8_CKPT"   ;;
  mc2cg_e16)  CKPT="$MC2CG_E16_CKPT"  ;;
  mc2cg_e24)  CKPT="$MC2CG_E24_CKPT"  ;;
  mc2cg_e32)  CKPT="$MC2CG_E32_CKPT"  ;;
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
