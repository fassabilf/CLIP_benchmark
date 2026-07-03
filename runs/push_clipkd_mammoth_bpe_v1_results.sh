#!/bin/bash
# Commit and push clipkd_mammoth_bpe_v1 eval results to GitHub.
# (Per-sample prediction dumps are excluded via runs/.gitignore and live on HF instead:
# fassabilf/sea-clip-eval-predictions.)
# Run from the CLIP_benchmark repo root on the cluster's transfer node (login node has no
# outbound internet). Uses the `fassabilf` remote (has a PAT embedded in the URL) — `origin`
# has no stored creds on the cluster and will 401.
set -e

cd "$(dirname "$0")/.."

git add \
    runs/results/clipkd_mammoth_bpe_v1_e8/ \
    runs/results/clipkd_mammoth_bpe_v1_e16/ \
    runs/results/clipkd_mammoth_bpe_v1_e24/ \
    runs/results/clipkd_mammoth_bpe_v1_e32/

git commit -m "add clipkd_mammoth_bpe_v1 eval results (e8/16/24/32, all benchmarks)"

git push fassabilf main:main
echo "Done — results pushed to GitHub."
