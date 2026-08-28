#!/bin/bash
#SBATCH -p gpu-devel
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 01:00:00
#SBATCH -A lt200394
#SBATCH -J cb_profile_repeat
#SBATCH -o /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%j.out
#SBATCH -e /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%j.out
#
# Three independent repetitions of the GPU profile, into rep1/ rep2/ rep3/.
#
# Batch-1 latency is kernel-launch-bound, so it is precise within a run (std
# < 0.3 ms) but reproduces to only ~10% across allocations — host contention and
# GPU clock state move it. Params and FLOPs are deterministic and identical
# across repetitions; only latency needs the median.
#
# Each repetition starts from a fresh process so allocator and thermal state do
# not carry over within a repetition the way they did in the single-pass run.
#
# Submit:  sbatch runs/az_profile_repeat.sh
# Then:    python3 runs/summarize_repeats.py

set -uo pipefail
cd /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark
source runs/env.sh
mkdir -p runs/logs

PYTHON="/home/ffirdaus/.conda/envs/mc2_eval_env/bin/python"

nvidia-smi --query-gpu=name,clocks.max.sm --format=csv,noheader

for rep in 1 2 3; do
    stage "repetition $rep of 3"
    $PYTHON runs/profile_models.py \
        --rows tinyclip mobileclip2_s0 sea_clip_tiny teacher_b16 \
        --batch-size 1 1024 --warmup 20 --runs 100 \
        --out-dir "$(pwd)/runs/results/profile/rep${rep}" \
        2>&1 | grep -vE "Loading weights|WARNING"
done

echo "=== done $(date) ==="
