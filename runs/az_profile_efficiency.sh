#!/bin/bash
#SBATCH -p gpu-devel
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 00:40:00
#SBATCH -A lt200394
#SBATCH -J cb_profile_efficiency
#SBATCH -o /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%j.out
#SBATCH -e /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%j.out
#
# Efficiency profile (params / FLOPs / latency / throughput / peak memory) for the
# paper's compact-model comparison set plus the teacher, all on ONE GPU.
#
# Answers reviewer XLob's major weakness 2 and GX74's minor point: the efficiency
# claim was carried by parameter count alone.
#
# Submit:  sbatch runs/az_profile_efficiency.sh
#
# Env: mc2_eval_env, whose open_clip is open_clip_phabibi — there `ViT-T-16` is the
# CLIP-BPE config (vocab 49408, ctx 77) that SEA-CLIP-Tiny actually uses. Running
# this in mteb_env2/clipkd_env would silently build the 256000-vocab SigLIP2
# ViT-T-16 and report a ~90M text tower.

set -uo pipefail
cd /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark
source runs/env.sh
mkdir -p runs/logs

PYTHON="/home/ffirdaus/.conda/envs/mc2_eval_env/bin/python"

echo "================================================================"
echo "  Efficiency profile  |  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PYTHON -c "import torch, open_clip; print('torch', torch.__version__, '| open_clip', open_clip.__file__)"
echo "================================================================"

# 1. Guard: the ViT-T-16 config in this env must match the trained checkpoint's
#    token_embedding, otherwise every text-tower number below is for the wrong model.
stage "vocab guard: ViT-T-16 config vs SEA-CLIP-Tiny checkpoint"
$PYTHON runs/profile_models.py \
    --model "ViT-T-16" --pretrained "$CLIPKD_MAMMOTH_BPE_V1_E32_CKPT" \
    --model-type open_clip --key _vocab_guard --label "vocab guard" \
    --check-vocab --no-latency --out-dir "$(pwd)/runs/results/profile/_guard"

# 2. The real sweep: all four models, same GPU, same input settings.
stage "profiling 4 models (bs 1 and 64)"
$PYTHON runs/profile_models.py \
    --rows tinyclip mobileclip2_s0 sea_clip_tiny teacher_b16 \
    --batch-size 1 64 --warmup 20 --runs 100 \
    --exclude-projections --latex

echo "=== done $(date) ==="
