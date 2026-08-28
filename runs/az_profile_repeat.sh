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

# REPS repetitions into ${OUTBASE}1..N. Three repetitions inside ONE allocation
# agree to ~1%, so they measure repeatability, not reproducibility: to capture the
# ~10% that moves between allocations, submit this script several times with
# REPS=1 and a distinct OUTBASE:
#   for i in 1 2 3; do sbatch --export=ALL,REPS=1,OUTBASE=alloc$i runs/az_profile_repeat.sh; done
REPS="${REPS:-3}"
OUTBASE="${OUTBASE:-rep}"

for rep in $(seq 1 "$REPS"); do
    stage "repetition $rep of $REPS (${OUTBASE})"
    $PYTHON runs/profile_models.py \
        --rows tinyclip mobileclip2_s0 sea_clip_tiny teacher_b16 \
        --batch-size 1 1024 --warmup 20 --runs 100 \
        --out-dir "$(pwd)/runs/results/profile/${OUTBASE}${rep}" \
        2>&1 | grep -vE "Loading weights|WARNING"
done

echo "=== done $(date) ==="
