#!/bin/bash
#SBATCH -p compute-devel
#SBATCH -N 1 -c 8
#SBATCH --ntasks-per-node=1
#SBATCH -t 00:30:00
#SBATCH -A lt200394
#SBATCH -J tokenizer_audit
#SBATCH -o /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%j.out
#SBATCH -e /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%j.out

# Tokenizer audit for the rebuttal: student CLIP-BPE vs teacher XLM-V.
# CPU-only, no checkpoints, no internet needed (XLM-V is in the offline HF cache).
# Usually fast enough to just run on the transfer node; this is the fallback.

set -euo pipefail
source /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/env.sh

module load Mamba/23.11.0-0
source activate mc2_eval_env

cd "$CB_ROOT"
python runs/tokenizer_audit.py \
    --eval-root "$EVAL_ROOT" \
    --teacher facebook/xlm-v-base \
    --ctx 77 \
    --verify-openclip

echo "DONE tokenizer_audit"
