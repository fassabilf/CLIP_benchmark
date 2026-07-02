#!/bin/bash
# Commit and push ckdonly_v1 eval results to GitHub.
# (Per-sample prediction dumps are excluded via runs/.gitignore and live on HF instead:
# fassabilf/sea-clip-eval-predictions.)
# Run from the CLIP_benchmark repo root on the cluster.
set -e

cd "$(dirname "$0")/.."

git add \
    runs/results/ckdonly_v1_e8/ \
    runs/results/ckdonly_v1_e16/ \
    runs/results/ckdonly_v1_e24/ \
    runs/results/ckdonly_v1_e32/

git commit -m "add ckdonly_v1 eval results (e8/16/24/32, all benchmarks)"

git push origin main
echo "Done — results pushed to GitHub."
