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

# Phase A (parallel): train-probe encoding for the MetaCLIP2 TEACHER. One array task
# per (dataset[:lang]) shard. 13 entries (bloom + cgoe×6 + wit×6) -> array 0-12, <=10
# concurrent. This fills the teacher row of the train-probe table (cf. mc2_* / selflearn).
# Encoder auto-skips existing .npz -> safe to resubmit.
# After all tasks finish, run ad_metaclip2_retrieval_probe.sh (Phase B) for cos/R@1.
#
# Uses the PATCHED encoder ${CB_ROOT}/runs/encode_train_embeddings.py, which defers the
# HF path to clip_benchmark.models.load_clip (the same wrapper the eval uses).
#
# DEFAULT = the actual B-16 distillation teacher (facebook/metaclip-2-worldwide-b16) via
# --model-type hf_transformers (open_clip has no ViT-B-16-worldwide weights). Needs an env
# with transformers + clip_benchmark importable -> mteb_env2.
# To probe the H-14 upper bound instead, use the OPEN_CLIP block at the bottom (tag metaclip2).

set -euo pipefail
source /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/env.sh
module load Mamba/23.11.0-0
source activate mteb_env2

ENC="${CB_ROOT}/runs/encode_train_embeddings.py"     # patched copy (HF + open_clip)
EMB_ROOT=/lustrefs/disk/project/lt200394-thllmV/kd_dataset/eval/train_probe/emb

TAG="metaclip2_b16"
HF_MODEL="facebook/metaclip-2-worldwide-b16"

# 13 dataset entries: "dataset:lang" (empty lang -> whole dataset, used for bloom).
DSENTRIES=(
  "bloom:" \
  "cgoe:id" "cgoe:jv" "cgoe:ms" "cgoe:su" "cgoe:th" "cgoe:vi" \
  "wit:id"  "wit:jv"  "wit:ms"  "wit:my"  "wit:th"  "wit:vi"
)

IFS=':' read -r DS LANG <<< "${DSENTRIES[$SLURM_ARRAY_TASK_ID]}"

LANG_ARG=()
[[ -n "$LANG" ]] && LANG_ARG=(--langs "$LANG")

echo "task=$SLURM_ARRAY_TASK_ID tag=$TAG model=$HF_MODEL (hf_transformers) ds=$DS lang=${LANG:-ALL}"

# B/16 teacher loaded from the local HF cache (HF_HUB_OFFLINE=1, no download).
srun python "$ENC" \
    --model-type hf_transformers \
    --model "$HF_MODEL" --model-cache-dir "$HF_HUB_CACHE" \
    --tag "$TAG" \
    --datasets "$DS" "${LANG_ARG[@]}" \
    --batch-size 256 --num-workers 15 \
    --out-dir "$EMB_ROOT"

# --- OPEN_CLIP alternative: H-14 worldwide upper bound (tag metaclip2; run in mc2_eval_env) ---
# srun python "$ENC" \
#     --model-type open_clip \
#     --model "ViT-H-14-worldwide-quickgelu" --ckpt "metaclip2_worldwide" \
#     --tag "metaclip2" \
#     --datasets "$DS" "${LANG_ARG[@]}" \
#     --batch-size 128 --num-workers 15 \
#     --out-dir "$EMB_ROOT"

echo "DONE task $SLURM_ARRAY_TASK_ID"
