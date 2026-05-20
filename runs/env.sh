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

# S1 = asvant ViT-T-16 + CLIP BPE → arch ViT-T-16-clipbpe (new config in open_clip_edit)
# S2/S3 = Habibi ViT-T-16 + SigLIP2 HFTokenizer → arch ViT-T-16 (canonical config)
# MetaCLIP-2 = ViT-H-14-worldwide-quickgelu + metaclip2_worldwide (open_clip stock, HF Hub timm/...)
