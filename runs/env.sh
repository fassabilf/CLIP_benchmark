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

# Habibi metaclip2_kd v2 (May31): ViT-T-16 (CLIP-BPE) <- MetaCLIP2-B16-worldwide,
# trained on CC12M (10.97M English pairs) = "cc12m-baseline". Same init/teacher/loss as
# the v1 SEA-blend run above; e0 init == MC2_E0_CKPT (reuse mc2_e0). Eval in mc2_eval_env.
MC2CC_DIR="/project/lt200394-thllmV/multilingual-clip-kd/open_clip/experiments/metaclip2_kd/clipkd_ViT-T-16_from_ViT-B-16-MetaClip_pretrained_v2/checkpoints"
MC2CC_E8_CKPT="$MC2CC_DIR/epoch_8.pt"
MC2CC_E16_CKPT="$MC2CC_DIR/epoch_16.pt"
MC2CC_E24_CKPT="$MC2CC_DIR/epoch_24.pt"
MC2CC_E32_CKPT="$MC2CC_DIR/epoch_32.pt"

# Habibi single-dataset metaclip2_kd v2 runs (Jun 2026): ViT-T-16 (CLIP-BPE) <- MetaCLIP2-B16-worldwide,
# one run per dataset (WIT / CulturalGround-OE / Bloom). Random student init. Eval in mc2_eval_env.
MC2WIT_DIR="/project/lt200394-thllmV/multilingual-clip-kd/open_clip/experiments/metaclip2_kd/clipkd_ViT-T-16_from_ViT-B-16-MetaClip_pretrained_v2_wit/checkpoints"
MC2WIT_E8_CKPT="$MC2WIT_DIR/epoch_8.pt"
MC2WIT_E16_CKPT="$MC2WIT_DIR/epoch_16.pt"
MC2WIT_E24_CKPT="$MC2WIT_DIR/epoch_24.pt"
MC2WIT_E32_CKPT="$MC2WIT_DIR/epoch_32.pt"

MC2CG_DIR="/project/lt200394-thllmV/multilingual-clip-kd/open_clip/experiments/metaclip2_kd/clipkd_ViT-T-16_from_ViT-B-16-MetaClip_pretrained_v2_cg/checkpoints"
MC2CG_E8_CKPT="$MC2CG_DIR/epoch_8.pt"
MC2CG_E16_CKPT="$MC2CG_DIR/epoch_16.pt"
MC2CG_E24_CKPT="$MC2CG_DIR/epoch_24.pt"
MC2CG_E32_CKPT="$MC2CG_DIR/epoch_32.pt"

MC2BLOOM_DIR="/project/lt200394-thllmV/multilingual-clip-kd/open_clip/experiments/metaclip2_kd/clipkd_ViT-T-16_from_ViT-B-16-MetaClip_pretrained_v2_bloom/checkpoints"
MC2BLOOM_E8_CKPT="$MC2BLOOM_DIR/epoch_8.pt"
MC2BLOOM_E16_CKPT="$MC2BLOOM_DIR/epoch_16.pt"
MC2BLOOM_E24_CKPT="$MC2BLOOM_DIR/epoch_24.pt"
MC2BLOOM_E32_CKPT="$MC2BLOOM_DIR/epoch_32.pt"

# Habibi metaclip2_kd v3 (Jun 2026): ViT-T-16 (CLIP-BPE) <- MetaCLIP2-B16-worldwide,
# trained on CC12M + full SEA blend (CG-OE-filt × 6 langs + WIT-hf-base × 6 langs + Bloom).
# Same init (clipkd_vit_t_16_init_clean.pt) / teacher / loss as v1 & v2. Eval in mc2_eval_env.
MC2V3_DIR="/project/lt200394-thllmV/multilingual-clip-kd/open_clip/experiments/metaclip2_kd/clipkd_ViT-T-16_from_ViT-B-16-MetaClip_pretrained_v3/checkpoints"
MC2V3_E8_CKPT="$MC2V3_DIR/epoch_8.pt"
MC2V3_E16_CKPT="$MC2V3_DIR/epoch_16.pt"
MC2V3_E24_CKPT="$MC2V3_DIR/epoch_24.pt"
MC2V3_E32_CKPT="$MC2V3_DIR/epoch_32.pt"

# Ashpun selflearn_mammoth_v1: ViT-T-16, no KD (contrastive loss only),
# trained on SEA blend (CG-OE-filt×6 + WIT-hf-base×6) + CC12M + Bloom multilingual, 32ep.
SELFLEARN_MAMMOTH_V1_DIR="/project/lt200394-thllmV/mkd-exp/open_clip/experiments/selflearn_multidata_mammoth_v1/selflearn_ViT-T-16_multidata_mammoth_v1/checkpoints"
SELFLEARN_MAMMOTH_V1_E8_CKPT="$SELFLEARN_MAMMOTH_V1_DIR/epoch_8.pt"
SELFLEARN_MAMMOTH_V1_E16_CKPT="$SELFLEARN_MAMMOTH_V1_DIR/epoch_16.pt"
SELFLEARN_MAMMOTH_V1_E24_CKPT="$SELFLEARN_MAMMOTH_V1_DIR/epoch_24.pt"
SELFLEARN_MAMMOTH_V1_E32_CKPT="$SELFLEARN_MAMMOTH_V1_DIR/epoch_32.pt"

# Habibi clipkd_multidata_mammoth_v1: ViT-T-16 <- MetaCLIP2-ViT-B-16-worldwide (clipkd distill),
# same student config as MC2 family (CLIP-BPE, eval in mc2_eval_env, NOT open_clip_edit's
# SigLIP2 ViT-T-16). Trained on CC12M + full SEA blend (CG-OE-filt×6 + WIT-hf-base×6 + Bloom)
# + multilingual ml-train + mammoth_vl_sea, 32ep.
MAMMOTH_KD_DIR="/project/lt200394-thllmV/mkd-exp/open_clip/experiments/clipkd_multidata_mammoth_v1/clipkd_ViT-T-16_from_MetaCLIP2-ViT-B-16-worldwide_multidata_mammoth_v1/checkpoints"
MAMMOTH_KD_E8_CKPT="$MAMMOTH_KD_DIR/epoch_8.pt"
MAMMOTH_KD_E16_CKPT="$MAMMOTH_KD_DIR/epoch_16.pt"
MAMMOTH_KD_E24_CKPT="$MAMMOTH_KD_DIR/epoch_24.pt"
MAMMOTH_KD_E32_CKPT="$MAMMOTH_KD_DIR/epoch_32.pt"

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
