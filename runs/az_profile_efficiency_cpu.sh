#!/bin/bash
#SBATCH -p compute-devel
#SBATCH -N 1 -c 16
#SBATCH --ntasks-per-node=1
#SBATCH -t 00:40:00
#SBATCH -A lt200394
#SBATCH -J cb_profile_efficiency_cpu
#SBATCH -o /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%j.out
#SBATCH -e /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%j.out
#
# CPU counterpart of runs/az_profile_efficiency.sh: single-sample latency on CPU,
# which is the setting a "deployable in a low-resource region" claim is really
# about. fp32 (no autocast on CPU), fewer runs since each pass is far slower.
#
# Only the LATENCY columns of this run are usable. FLOPs come out ~0.36 GFLOPs
# light per image tower here because the CPU's fused SDPA kernel hides the
# attention matmuls from FlopCounterMode; quote the GPU run's FLOPs instead.
#
# Results go to runs/results/profile/cpu/ so they never overwrite the GPU JSONs —
# latency is device-bound and the two must not be mixed in one table.
#
# Submit:  sbatch runs/az_profile_efficiency_cpu.sh

set -uo pipefail
cd /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark
source runs/env.sh
mkdir -p runs/logs

PYTHON="/home/ffirdaus/.conda/envs/mc2_eval_env/bin/python"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

echo "================================================================"
echo "  Efficiency profile (CPU)  |  $(date)"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS  on $(lscpu | sed -n 's/^Model name: *//p')"
echo "================================================================"

stage "profiling 4 models on CPU (bs 1)"
$PYTHON runs/profile_models.py \
    --rows tinyclip mobileclip2_s0 sea_clip_tiny teacher_b16 \
    --device cpu --fp32 --batch-size 1 --warmup 5 --runs 30 \
    --no-memory --exclude-projections \
    --out-dir "$(pwd)/runs/results/profile/cpu"

echo "=== done $(date) ==="
