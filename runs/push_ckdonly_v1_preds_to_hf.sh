#!/bin/bash
# Push ckdonly_v1 per-sample prediction dumps (preds/) to the HF dataset repo
# fassabilf/sea-clip-eval-predictions. Run from the CLIP_benchmark repo root
# on the cluster (login/transfer node) where the HF token is cached.
set -euo pipefail

cd "$(dirname "$0")/.."

source runs/env.sh
module load Mamba/23.11.0-0
source activate mteb_env2

# runs/env.sh sets HF_HUB_OFFLINE=1 for offline compute nodes; this script is
# meant to run on the transfer/login node, which has internet access.
unset HF_HUB_OFFLINE

python runs/push_ckdonly_v1_preds_to_hf.py
echo "Done — ckdonly_v1 predictions pushed to HuggingFace."
