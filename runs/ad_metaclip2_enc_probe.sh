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

# Phase A (parallel): train-probe encoding for the MetaCLIP2 TEACHER (metaclip2_b16,
# ViT-B-16-worldwide). One array task per (dataset[:lang]) shard. 13 entries
# (bloom + cgoe×6 + wit×6), single checkpoint -> array 0-12, <=10 concurrent.
# This is the teacher upper-bound row for the train-probe table (cf. mc2_* / selflearn).
# Encoder auto-skips existing .npz -> safe to resubmit.
# After all tasks finish, run ad_metaclip2_retrieval_probe.sh (Phase B) for cos/R@1.
#
# NOTE on model loading: metaclip2_b16 is an HF model (no local ViT-T-16 .pt), so the
# encoder is invoked with the same flags the eval uses (l_metaclip2_b16.sh):
#   --model_type hf_transformers --model facebook/metaclip-2-worldwide-b16
#   --pretrained "" --model_cache_dir "$HF_HUB_CACHE"
# This assumes encode_train_embeddings.py forwards these to the same loader as
# clip_benchmark.cli. If the encoder only supports open_clip checkpoints, use the
# OPEN_CLIP alternative block below instead.

set -euo pipefail
source /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/env.sh
module load Mamba/23.11.0-0
source activate mteb_env2

ENC=/lustrefs/disk/project/lt200394-thllmV/kd_dataset/scripts/encode_train_embeddings.py
EMB_ROOT=/lustrefs/disk/project/lt200394-thllmV/kd_dataset/eval/train_probe/emb

TAG="metaclip2_b16"
MODEL="facebook/metaclip-2-worldwide-b16"
MODEL_TYPE="hf_transformers"

# 13 dataset entries: "dataset:lang" (empty lang -> whole dataset, used for bloom).
DSENTRIES=(
  "bloom:" \
  "cgoe:id" "cgoe:jv" "cgoe:ms" "cgoe:su" "cgoe:th" "cgoe:vi" \
  "wit:id"  "wit:jv"  "wit:ms"  "wit:my"  "wit:th"  "wit:vi"
)

IFS=':' read -r DS LANG <<< "${DSENTRIES[$SLURM_ARRAY_TASK_ID]}"

LANG_ARG=()
[[ -n "$LANG" ]] && LANG_ARG=(--langs "$LANG")

echo "task=$SLURM_ARRAY_TASK_ID tag=$TAG model=$MODEL ds=$DS lang=${LANG:-ALL}"
echo "open_clip: $(python -c 'import open_clip; print(open_clip.__file__)')"

# B/16 is ~15x the params of ViT-T-16 -> smaller batch than the student probe (512).
srun python "$ENC" \
    --model_type "$MODEL_TYPE" \
    --model "$MODEL" --pretrained "" \
    --model_cache_dir "$HF_HUB_CACHE" \
    --tag "$TAG" \
    --datasets "$DS" "${LANG_ARG[@]}" \
    --batch-size 256 --num-workers 15 \
    --out-dir "$EMB_ROOT"

# --- OPEN_CLIP alternative (use if encode_train_embeddings.py does NOT accept
#     --model_type / --model_cache_dir and instead loads via open_clip pretrained tags) ---
# srun python "$ENC" \
#     --model "ViT-B-16-worldwide-quickgelu" --pretrained "metaclip2_worldwide" \
#     --tag "$TAG" \
#     --datasets "$DS" "${LANG_ARG[@]}" \
#     --batch-size 256 --num-workers 15 \
#     --out-dir "$EMB_ROOT"

echo "DONE task $SLURM_ARRAY_TASK_ID"
