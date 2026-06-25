#!/bin/bash
# Local environment for TinyCLIP / MobileCLIP2 evaluation (no SLURM, no HPC paths).
# Source this before running eval scripts: source runs/env_local.sh

export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:?set HUGGING_FACE_HUB_TOKEN before sourcing}"
export HF_TOKEN="${HF_TOKEN:-$HUGGING_FACE_HUB_TOKEN}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

CB_ROOT="${CB_ROOT:-/workspace/CLIP_benchmark}"

# Dataset roots — override with env vars if data lives elsewhere
EVAL_ROOT="${EVAL_ROOT:-/workspace/data/eval}"
IMAGENET_ROOT="${IMAGENET_ROOT:-/workspace/data/imagenet}"

# Model paths set after download
TINYCLIP_DIR="${TINYCLIP_DIR:-/workspace/data/models/tinyclip}"
MC2_PT="${MC2_PT:-/workspace/data/models/mobileclip2_s0/mobileclip2_s0.pt}"

# ── timing helpers (same as cluster env.sh) ─────────────────────────────────
_SWEEP_T0="${_SWEEP_T0:-$(date +%s)}"
export _SWEEP_T0

stage() {
    local label="$1"
    local now elapsed
    now=$(date +%s)
    elapsed=$(( now - _SWEEP_T0 ))
    printf '\n=== [%s | +%dm%02ds] %s ===\n' \
        "$(date '+%H:%M:%S')" $((elapsed/60)) $((elapsed%60)) "$label"
}

timed_run() {
    local name="$1"; shift
    local t0 t1 dt
    t0=$(date +%s)
    "$@"
    local rc=$?
    t1=$(date +%s); dt=$(( t1 - t0 ))
    printf '  [%s] %s took %dm%02ds (rc=%d)\n' \
        "$(date '+%H:%M:%S')" "$name" $((dt/60)) $((dt%60)) "$rc"
    return $rc
}
