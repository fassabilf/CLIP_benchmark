#!/bin/bash
# Evaluate MobileCLIP2-S0 (apple/MobileCLIP2-S0) on the full SEA-CLIP
# benchmark suite.  Run from CLIP_benchmark root:
#   source /venv/main/bin/activate
#   bash runs/eval_mobileclip2.sh
#
# MobileCLIP2-S0 is registered in open_clip (via ml-mobileclip / open_clip_torch).
# We use model_type=open_clip with the local .pt checkpoint.
# Results land in runs/results/mobileclip2_s0/*.json
# Override data paths via env vars (see runs/env_local.sh).

set -euo pipefail
cd "$(dirname "$0")/.."
source runs/env_local.sh

TAG="mobileclip2_s0"
MODEL="MobileCLIP2-S0"
# Use the timm-converted open_clip checkpoint (dfndr2b), NOT the Apple .pt file.
# The Apple checkpoint (apple/MobileCLIP2-S0) uses different key names than open_clip.
# dfndr2b is auto-downloaded from timm/MobileCLIP2-S0-OpenCLIP on HF Hub.
PRETRAINED="dfndr2b"
RESULTS="${CB_ROOT}/runs/results/${TAG}"
PREDS="${RESULTS}/preds"
mkdir -p "$RESULTS" "$PREDS"

# ── eval helper: skip if output exists ──────────────────────────────────────
run() {
    local OUT="$1"; shift
    if [[ -f "$OUT" ]]; then
        echo "  [skip] $(basename "$OUT") already exists"
        return 0
    fi
    timed_run "$(basename "$OUT" .json)" \
        python3 -m clip_benchmark.cli eval \
            --model_type open_clip \
            --model "$MODEL" \
            --pretrained "$PRETRAINED" \
            --batch_size 256 --num_workers 16 \
            --save_predictions "$PREDS" \
            --output "$OUT" \
            "$@"
}

echo "================================================================"
echo "  MobileCLIP2-S0 Evaluation  |  $(date)"
echo "  Pretrained: $PRETRAINED (timm/MobileCLIP2-S0-OpenCLIP)"
echo "  Output    : $RESULTS"
echo "================================================================"

# ── 1. ImageNet-1k ──────────────────────────────────────────────────────────
if [[ -d "$IMAGENET_ROOT" ]]; then
    stage "$TAG: ImageNet-1k (en)"
    run "${RESULTS}/imagenet1k_${TAG}.json" \
        --dataset imagenet1k-unverified \
        --dataset_root "$IMAGENET_ROOT" \
        --task zeroshot_classification --language en
else
    echo "  [skip] IMAGENET_ROOT=$IMAGENET_ROOT not found — skipping ImageNet-1k & Babel-IN"
fi

# ── 2. Babel-ImageNet ────────────────────────────────────────────────────────
if [[ -d "$IMAGENET_ROOT" ]]; then
    stage "$TAG: Babel-IN (8 langs)"
    total=8; done=0
    for LANG in en id jv ms my su th vi; do
        run "${RESULTS}/babel_imagenet_${LANG}_${TAG}.json" \
            --dataset babel_imagenet \
            --dataset_root "$IMAGENET_ROOT" \
            --task zeroshot_classification --language "$LANG" \
            || echo "  (lang $LANG failed; continuing)"
        done=$(( done + 1 ))
        echo "  [progress] Babel-IN: $done/$total langs done"
    done
fi

# ── 3. XM3600 ───────────────────────────────────────────────────────────────
stage "$TAG: XM3600 retrieval (en/id/th/vi)"
total=4; done=0
for LANG in en id th vi; do
    run "${RESULTS}/xm3600_${LANG}_${TAG}.json" \
        --dataset crossmodal3600 \
        --dataset_root "$EVAL_ROOT" \
        --task zeroshot_retrieval --language "$LANG" \
        --recall_k 1 5 10
    done=$(( done + 1 ))
    echo "  [progress] XM3600: $done/$total langs done"
done

# ── 4. Flickr30k-200 ────────────────────────────────────────────────────────
LANGS=(eng_Latn ind_Latn jav_Latn zsm_Latn mya_Mymr sun_Latn tha_Thai vie_Latn)
stage "$TAG: Flickr30k-200 (8 langs)"
total=${#LANGS[@]}; done=0
for LANG in "${LANGS[@]}"; do
    run "${RESULTS}/flickr30k_200_${LANG}_${TAG}.json" \
        --dataset flickr30k-200 \
        --dataset_root "$EVAL_ROOT" \
        --task zeroshot_retrieval --language "$LANG" \
        --recall_k 1 5 10 \
        || echo "  (lang $LANG failed; continuing)"
    done=$(( done + 1 ))
    echo "  [progress] Flickr30k-200: $done/$total langs done"
done

# ── 5. XTD-200 ──────────────────────────────────────────────────────────────
stage "$TAG: XTD-200 (8 langs)"
total=${#LANGS[@]}; done=0
for LANG in "${LANGS[@]}"; do
    run "${RESULTS}/xtd200_${LANG}_${TAG}.json" \
        --dataset xtd200 \
        --dataset_root "$EVAL_ROOT" \
        --task zeroshot_retrieval --language "$LANG" \
        --recall_k 1 5 10 \
        || echo "  (lang $LANG failed; continuing)"
    done=$(( done + 1 ))
    echo "  [progress] XTD-200: $done/$total langs done"
done

stage "$TAG: ALL DONE"
echo ""
echo "Results in: $RESULTS"
echo "Run: python3 runs/compute_per_lang_mean.py $TAG"
