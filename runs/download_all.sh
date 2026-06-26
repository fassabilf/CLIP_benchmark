#!/bin/bash
# Download all models and benchmark datasets in parallel.
# Run once before evaluation:
#   source /venv/main/bin/activate
#   bash runs/download_all.sh
#
# Set HUGGING_FACE_HUB_TOKEN before running, or export it.

set -uo pipefail
cd "$(dirname "$0")/.."
source runs/env_local.sh

echo "================================================================"
echo "  Parallel Download: Models + Benchmark Datasets"
echo "  $(date)"
echo "================================================================"

PIDS=()
LABELS=()

# ── Models ───────────────────────────────────────────────────────────────────
if [[ ! -d "$TINYCLIP_DIR" ]] || [[ -z "$(ls -A "$TINYCLIP_DIR" 2>/dev/null)" ]]; then
    echo "[start] TinyCLIP → $TINYCLIP_DIR"
    python3 -c "
from huggingface_hub import snapshot_download
snap = snapshot_download('wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M',
    token='$HUGGING_FACE_HUB_TOKEN', local_dir='$TINYCLIP_DIR')
print('TinyCLIP done:', snap)
" > /tmp/dl_tinyclip.log 2>&1 &
    PIDS+=($!); LABELS+=("TinyCLIP")
else
    echo "[skip]  TinyCLIP already at $TINYCLIP_DIR"
fi

if [[ ! -f "$MC2_PT" ]]; then
    MC2_DIR="$(dirname "$MC2_PT")"
    echo "[start] MobileCLIP2-S0 → $MC2_DIR"
    python3 -c "
from huggingface_hub import snapshot_download
snap = snapshot_download('apple/MobileCLIP2-S0',
    token='$HUGGING_FACE_HUB_TOKEN', local_dir='$MC2_DIR')
print('MobileCLIP2-S0 done:', snap)
" > /tmp/dl_mc2.log 2>&1 &
    PIDS+=($!); LABELS+=("MobileCLIP2-S0")
else
    echo "[skip]  MobileCLIP2-S0 already at $MC2_PT"
fi

# ── XM3600 captions ──────────────────────────────────────────────────────────
XM_CAP="${EVAL_ROOT}/xm3600/captions.jsonl"
if [[ ! -f "$XM_CAP" ]]; then
    echo "[start] XM3600 captions → ${EVAL_ROOT}/xm3600/"
    mkdir -p "${EVAL_ROOT}/xm3600"
    (
        cd "${EVAL_ROOT}/xm3600"
        wget -q --show-progress \
            https://google.github.io/crossmodal-3600/web-data/captions.zip \
            -O captions.zip && unzip -q captions.zip && rm captions.zip
        echo "XM3600 captions done"
    ) > /tmp/dl_xm3600_caps.log 2>&1 &
    PIDS+=($!); LABELS+=("XM3600-captions")
else
    echo "[skip]  XM3600 captions already exist"
fi

# ── XM3600 images ────────────────────────────────────────────────────────────
XM_IMG="${EVAL_ROOT}/xm3600/images"
if [[ ! -d "$XM_IMG" ]] || [[ -z "$(ls -A "$XM_IMG" 2>/dev/null)" ]]; then
    echo "[start] XM3600 images (~3.5 GB) → ${XM_IMG}/"
    mkdir -p "$XM_IMG"
    (
        wget -q --show-progress \
            https://open-images-dataset.s3.amazonaws.com/crossmodal-3600/images.tgz \
            -O /tmp/xm3600_images.tgz
        tar -xzf /tmp/xm3600_images.tgz -C "$XM_IMG" --strip-components=1
        rm /tmp/xm3600_images.tgz
        echo "XM3600 images done"
    ) > /tmp/dl_xm3600_imgs.log 2>&1 &
    PIDS+=($!); LABELS+=("XM3600-images")
else
    echo "[skip]  XM3600 images already exist"
fi

# ── Flickr30k-200 images ─────────────────────────────────────────────────────
FLICKR_IMG="${EVAL_ROOT}/flickr30k_200/images"
if [[ ! -d "$FLICKR_IMG" ]] || [[ -z "$(ls -A "$FLICKR_IMG" 2>/dev/null)" ]]; then
    echo "[start] Flickr30k-200 images → ${FLICKR_IMG}/"
    mkdir -p "$FLICKR_IMG"
    (
        wget -q --show-progress \
            https://nllb-data.com/test/flickr30k/images.tar.gz \
            -O /tmp/flickr30k_images.tar.gz
        tar -xzf /tmp/flickr30k_images.tar.gz -C "$FLICKR_IMG" --strip-components=1
        rm /tmp/flickr30k_images.tar.gz
        echo "Flickr30k-200 images done"
    ) > /tmp/dl_flickr30k.log 2>&1 &
    PIDS+=($!); LABELS+=("Flickr30k-200-images")
else
    echo "[skip]  Flickr30k-200 images already exist"
fi

# ── XTD-200 images ───────────────────────────────────────────────────────────
XTD_IMG="${EVAL_ROOT}/xtd200/images"
if [[ ! -d "$XTD_IMG" ]] || [[ -z "$(ls -A "$XTD_IMG" 2>/dev/null)" ]]; then
    echo "[start] XTD-200 images → ${XTD_IMG}/"
    mkdir -p "$XTD_IMG"
    (
        wget -q --show-progress \
            https://nllb-data.com/test/xtd10/images.tar.gz \
            -O /tmp/xtd200_images.tar.gz
        tar -xzf /tmp/xtd200_images.tar.gz -C "$XTD_IMG" --strip-components=1
        rm /tmp/xtd200_images.tar.gz
        echo "XTD-200 images done"
    ) > /tmp/dl_xtd200.log 2>&1 &
    PIDS+=($!); LABELS+=("XTD-200-images")
else
    echo "[skip]  XTD-200 images already exist"
fi

# ── Wait for all downloads with progress reporting ───────────────────────────
echo ""
echo "Waiting for ${#PIDS[@]} download(s): ${LABELS[*]}"
FAILED=()
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    label="${LABELS[$i]}"
    if wait "$pid"; then
        printf "  [done] %s\n" "$label"
    else
        printf "  [FAIL] %s (rc=%d) — see /tmp/dl_*.log\n" "$label" "$?"
        FAILED+=("$label")
    fi
done

echo ""
if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "All downloads complete. Ready to run eval scripts."
else
    echo "FAILED: ${FAILED[*]}"
    echo "Check logs in /tmp/dl_*.log"
    exit 1
fi
