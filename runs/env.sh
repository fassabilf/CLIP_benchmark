#!/bin/bash
# Shared env for all CLIP_benchmark sbatch jobs.
# Compute nodes have NO internet — all downloads must be on login node first.

export HF_HOME="/project/lt200394-thllmV/benchmark/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

CB_ROOT="/lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark"
EVAL_ROOT="/project/lt200394-thllmV/kd_dataset/eval"
IMAGENET_ROOT="/project/lt200394-thllmV/mkd-exp/datasets/imagenet"

# Student checkpoints (skip S4 NaN per MKD_MODELS.md)
S1_CKPT="/project/lt200394-thllmV/mkd-exp/open_clip/experiments/20260322_185708/clipkd_ViT-T-16_from_ViT-B-32_4927040/checkpoints/epoch_32.pt"
S2_CKPT="/project/lt200394-thllmV/multilingual-clip-kd/open_clip/experiments/siglip2_kd/clipkd_ViT-T-16_from_ViT-B-16-SigLIP2_v2/checkpoints/epoch_32.pt"
S3_CKPT="/project/lt200394-thllmV/multilingual-clip-kd/open_clip/experiments/siglip2_kd/clipkd_ViT-T-16_from_ViT-B-16-SigLIP2_v3/checkpoints/epoch_100.pt"

# Early-epoch variants (e8 ~ "edi e10" — S2 has no early ckpt so skip)
S1_E8_CKPT="/project/lt200394-thllmV/mkd-exp/open_clip/experiments/20260322_185708/clipkd_ViT-T-16_from_ViT-B-32_4927040/checkpoints/epoch_8.pt"
S3_E8_CKPT="/project/lt200394-thllmV/multilingual-clip-kd/open_clip/experiments/siglip2_kd/clipkd_ViT-T-16_from_ViT-B-16-SigLIP2_v3/checkpoints/epoch_8.pt"

# Habibi WIT-multilingual run (ViT-T-16 ← ViT-B-16-SigLIP2, train_wit.csv, 32ep lr2e-3)
WIT_E8_CKPT="/project/lt200394-thllmV/multilingual-clip-kd/open_clip/experiments/siglip2_kd/clipkd_ViT-T-16_from_ViT-B-16-SigLIP2_multilingual/checkpoints/epoch_8.pt"
WIT_E32_CKPT="/project/lt200394-thllmV/multilingual-clip-kd/open_clip/experiments/siglip2_kd/clipkd_ViT-T-16_from_ViT-B-16-SigLIP2_multilingual/checkpoints/epoch_32.pt"

# Habibi multilingual_v1 run: 3-dataset blend (cultural-ground + WIT + bloom), 32ep lr2e-3.
MV1_E8_CKPT="/project/lt200394-thllmV/multilingual-clip-kd/open_clip/experiments/siglip2_kd/clipkd_ViT-T-16_from_ViT-B-16-SigLIP2_multilingual_v1/checkpoints/epoch_8.pt"
MV1_E32_CKPT="/project/lt200394-thllmV/multilingual-clip-kd/open_clip/experiments/siglip2_kd/clipkd_ViT-T-16_from_ViT-B-16-SigLIP2_multilingual_v1/checkpoints/epoch_32.pt"

# Habibi metaclip2_kd run (latest, May31): ViT-T-16 <- MetaCLIP2-B16-worldwide,
# 3-dataset SEA blend (cultural-ground + WIT + bloom), 32ep lr2e-3.
# Student is CLIP-BPE (vocab 49408 ctx 77). EVAL in env mc2_eval_env (pinned to habibi's
# open_clip), arch name = ViT-T-16 (HIS config). Do NOT eval with open_clip_edit's
# ViT-T-16 (that's the SigLIP2 256000 config). E0 = pre-KD init = epoch-0 baseline.
MC2_DIR="/project/lt200394-thllmV/multilingual-clip-kd/open_clip/experiments/metaclip2_kd/clipkd_ViT-T-16_from_ViT-B-16-MetaClip_pretrained_v1/checkpoints"
MC2_E0_CKPT="/project/lt200394-thllmV/multilingual-clip-kd/open_clip/pretrained/student/clipkd_vit_t_16_init_clean.pt"
MC2_E8_CKPT="$MC2_DIR/epoch_8.pt"
MC2_E16_CKPT="$MC2_DIR/epoch_16.pt"
MC2_E24_CKPT="$MC2_DIR/epoch_24.pt"
MC2_E32_CKPT="$MC2_DIR/epoch_32.pt"

# --- timing helpers (source me, then call `stage` / `timed_run`) ---
# Usage in sweep scripts:
#   stage "Babel-ImageNet (8 langs)"   # prints banner + start time
#   timed_run "imagenet1k" python ...  # runs cmd, prints elapsed + ETA hint
# Set _SWEEP_T0 once at start to enable wall-clock since job-start.

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
    # timed_run <short-name> <command...>
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

# S1 = asvant ViT-T-16 + CLIP BPE → arch ViT-T-16-clipbpe (new config in open_clip_edit)
# S2/S3 = Habibi ViT-T-16 + SigLIP2 HFTokenizer → arch ViT-T-16 (canonical config)
# MetaCLIP-2 = ViT-H-14-worldwide-quickgelu + metaclip2_worldwide (open_clip stock, HF Hub timm/...)
