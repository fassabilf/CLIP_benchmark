#!/bin/bash
#SBATCH -p compute
#SBATCH -N 1 -c 16
#SBATCH --ntasks-per-node=1
#SBATCH -t 00:30:00
#SBATCH -A lt200394
#SBATCH -J train_probe_phase_b
#SBATCH -o /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%j.out
#SBATCH -e /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%j.out

# Phase B: compute train-probe retrieval metrics (R@1/5/10, cosine summary) from cached
# .npz embeddings for all tags that have been encoded. Gallery size bumped to 10000
# (was 1000) for more reliable estimates. Overwrites train_probe_summary.csv in each
# tag dir — re-run after any new encode job finishes.
# Covers: mc2_e{0..32}, mc2cc_e{0..32}, mc2wit/mc2cg/mc2bloom_e32.

set -euo pipefail
source /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/env.sh
module load Mamba/23.11.0-0
source activate mc2_eval_env

RETRIEVAL=/lustrefs/disk/project/lt200394-thllmV/kd_dataset/scripts/retrieval_from_emb.py
EMB_ROOT=/lustrefs/disk/project/lt200394-thllmV/kd_dataset/eval/train_probe/emb

for TAG_DIR in "$EMB_ROOT"/*/; do
    TAG=$(basename "$TAG_DIR")
    NPZ_COUNT=$(find "$TAG_DIR" -name "*.npz" | wc -l)
    if [[ "$NPZ_COUNT" -eq 0 ]]; then
        echo "  skip $TAG (no .npz found)"
        continue
    fi
    echo "=== Phase B: $TAG ($NPZ_COUNT shards) ==="
    python "$RETRIEVAL" \
        --emb-dir "$TAG_DIR" \
        --tag "$TAG" \
        --gallery-size 10000 \
        --repeats 5
    echo "  -> wrote $TAG_DIR/train_probe_summary.csv"
done

echo "Phase B complete: $(date)"
