#!/bin/bash
# Download TinyCLIP and MobileCLIP2-S0 into the cluster HF Hub cache.
# Run on the LOGIN NODE (compute nodes have no internet — HF_HUB_OFFLINE=1).
#
# Usage (login node, after activating any env that has huggingface_hub + open_clip):
#   module load Mamba/23.11.0-0 && source activate mteb_env2
#   bash runs/download_ext_models.sh

set -euo pipefail
cd "$(dirname "$0")/.."

# Source env.sh for cache paths, then flip offline flags off for download
source /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/env.sh
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0

echo "========================================================"
echo "  Download: TinyCLIP + MobileCLIP2-S0 → HF Hub cache"
echo "  HF_HUB_CACHE = $HF_HUB_CACHE"
echo "  $(date)"
echo "========================================================"

# ── TinyCLIP ─────────────────────────────────────────────────────────────────
echo ""
echo "[1/2] TinyCLIP — wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M"
python3 - <<'EOF'
from huggingface_hub import snapshot_download
import os

repo = "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M"
path = snapshot_download(repo, cache_dir=os.environ["HF_HUB_CACHE"])
print(f"  done: {path}")
EOF

# ── MobileCLIP2-S0 (timm/open_clip, pretrained=dfndr2b) ──────────────────────
# open_clip downloads this from timm/MobileCLIP2-S0-OpenCLIP on HF Hub.
# Triggering model creation forces the weight download into HF cache.
echo ""
echo "[2/2] MobileCLIP2-S0 — timm/MobileCLIP2-S0-OpenCLIP (dfndr2b)"
python3 - <<'EOF'
import open_clip, os

print("  loading open_clip model to trigger weight download...")
model, _, _ = open_clip.create_model_and_transforms(
    "MobileCLIP2-S0",
    pretrained="dfndr2b",
    cache_dir=os.environ["HF_HUB_CACHE"],
)
print("  done: MobileCLIP2-S0 weights cached")
EOF

echo ""
echo "All downloads complete. Compute nodes can now eval offline."
