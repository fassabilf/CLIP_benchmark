#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 04:00:00
#SBATCH -A lt200394
#SBATCH -J metaclip2_enc_probe
#SBATCH -a 0-12%10
#SBATCH -o /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%A_%a.out
#SBATCH -e /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%A_%a.out

# Phase A (parallel): train-probe encoding for the MetaCLIP2 teacher (metaclip2,
# ViT-H-14-worldwide). One array task per (dataset[:lang]) shard. 13 entries
# (bloom + cgoe×6 + wit×6), single checkpoint -> array 0-12, <=10 concurrent.
# This is the teacher upper-bound row for the train-probe table (cf. mc2_* / selflearn).
# NOTE: open_clip only ships worldwide weights for H-14/bigG (no ViT-B-16-worldwide),
# so this open_clip-based probe uses the H-14 teacher (tag "metaclip2"). The B-16
# distillation teacher would need encode_train_embeddings.py patched to load via HF.
# Encoder auto-skips existing .npz -> safe to resubmit.
# After all tasks finish, run ad_metaclip2_retrieval_probe.sh (Phase B) for cos/R@1.
#
# Model loading: encode_train_embeddings.py builds the model with
#   open_clip.create_model_and_transforms(args.model, pretrained=args.ckpt)
# so `--ckpt` is an open_clip *pretrained tag* (not necessarily a .pt path). For the
# MetaCLIP2 teacher we therefore pass the open_clip worldwide arch + its pretrained tag.
# Confirm the exact (model, pretrained) pair registered in your open_clip with:
#   python -c "import open_clip, pprint; pprint.pprint([p for p in open_clip.list_pretrained() if 'worldwide' in p[0].lower()])"
# and update OC_MODEL / OC_PRETRAINED below if they differ.

set -euo pipefail
source /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/env.sh
module load Mamba/23.11.0-0
source activate mc2_eval_env

ENC=/lustrefs/disk/project/lt200394-thllmV/kd_dataset/scripts/encode_train_embeddings.py
EMB_ROOT=/lustrefs/disk/project/lt200394-thllmV/kd_dataset/eval/train_probe/emb

TAG="metaclip2"
# open_clip arch + pretrained tag (passed as --model / --ckpt -> create_model_and_transforms).
# Verify with open_clip.list_pretrained() (see header) and adjust if needed.
OC_MODEL="ViT-H-14-worldwide-quickgelu"
OC_PRETRAINED="metaclip2_worldwide"

# 13 dataset entries: "dataset:lang" (empty lang -> whole dataset, used for bloom).
DSENTRIES=(
  "bloom:" \
  "cgoe:id" "cgoe:jv" "cgoe:ms" "cgoe:su" "cgoe:th" "cgoe:vi" \
  "wit:id"  "wit:jv"  "wit:ms"  "wit:my"  "wit:th"  "wit:vi"
)

IFS=':' read -r DS LANG <<< "${DSENTRIES[$SLURM_ARRAY_TASK_ID]}"

LANG_ARG=()
[[ -n "$LANG" ]] && LANG_ARG=(--langs "$LANG")

echo "task=$SLURM_ARRAY_TASK_ID tag=$TAG model=$OC_MODEL pretrained=$OC_PRETRAINED ds=$DS lang=${LANG:-ALL}"
echo "open_clip: $(python -c 'import open_clip; print(open_clip.__file__)')"

# H/14 is a large teacher; keep batch modest vs the ViT-T-16 student probe (512).
srun python "$ENC" \
    --ckpt "$OC_PRETRAINED" --tag "$TAG" --model "$OC_MODEL" \
    --datasets "$DS" "${LANG_ARG[@]}" \
    --batch-size 128 --num-workers 15 \
    --out-dir "$EMB_ROOT"

echo "DONE task $SLURM_ARRAY_TASK_ID"
