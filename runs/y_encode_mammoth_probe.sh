#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 04:00:00
#SBATCH -A lt200394
#SBATCH -J mammoth_enc_probe
#SBATCH -a 0-7
#SBATCH -o /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%A_%a.out
#SBATCH -e /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%A_%a.out

# Phase A: back-fill the "Mammoth R@1" train-probe column (tab:training) for every model
# that is currently a row in Table 4, EXCEPT selflearn_mammoth_v1 and ckdonly_v1 (both use
# the SigLIP2/HFTokenizer 256000-vocab student -- out of scope for this backfill per
# request; skip rather than debug that fork). One array task per tag (8 total) -- each
# does a single "mammoth" dataset job against mammoth_vl_sea_wds (109 flat shards, no
# per-lang split, same shape as the existing bloom-all job).
#
# Uses the PATCHED encoder ${CB_ROOT}/runs/encode_train_embeddings.py (adds the
# hf_transformers path for the teacher) so we don't have to touch the shared
# /project/lt200394-thllmV/kd_dataset/scripts/ copy.
#
# Conda env differs PER TAG: the teacher is SigLIP2/HF (mteb_env2); everything else in
# this list is the CLIP-BPE mc2 family (mc2_eval_env). Mixing the wrong env in causes a
# token_embedding.weight shape mismatch on checkpoint load (see runs/env.sh comments).
# Each array task is an independent process, so activating a different env per case is
# safe here.
#
# Encoder auto-skips existing .npz -> safe to resubmit.
#
# After all 8 tasks finish, run runs/y_retrieval_mammoth_probe.sh (Phase B).

set -euo pipefail
source /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/env.sh
module load Mamba/23.11.0-0

ENC="${CB_ROOT}/runs/encode_train_embeddings.py"
EMB_ROOT=/lustrefs/disk/project/lt200394-thllmV/kd_dataset/eval/train_probe/emb

case "$SLURM_ARRAY_TASK_ID" in
    0) TAG="mc2_e32";                    ENV="mc2_eval_env"; CKPT="$MC2_E32_CKPT" ;;
    1) TAG="mc2cc_e32";                  ENV="mc2_eval_env"; CKPT="$MC2CC_E32_CKPT" ;;
    2) TAG="mc2v3_e32";                  ENV="mc2_eval_env"; CKPT="$MC2V3_E32_CKPT" ;;
    3) TAG="clipkd_mammoth_bpe_v1_e32";  ENV="mc2_eval_env"; CKPT="$CLIPKD_MAMMOTH_BPE_V1_E32_CKPT" ;;
    4) TAG="mc2wit_e32";                 ENV="mc2_eval_env"; CKPT="$MC2WIT_E32_CKPT" ;;
    5) TAG="mc2bloom_e32";               ENV="mc2_eval_env"; CKPT="$MC2BLOOM_E32_CKPT" ;;
    6) TAG="mc2cg_e32";                  ENV="mc2_eval_env"; CKPT="$MC2CG_E32_CKPT" ;;
    7) TAG="metaclip2_b16";              ENV="mteb_env2";    CKPT="" ;;
    *) echo "unknown array idx"; exit 1 ;;
esac

source activate "$ENV"

echo "task=$SLURM_ARRAY_TASK_ID tag=$TAG env=$ENV"

if [[ "$SLURM_ARRAY_TASK_ID" == "7" ]]; then
    srun python "$ENC" \
        --model-type hf_transformers \
        --model "facebook/metaclip-2-worldwide-b16" --model-cache-dir "$HF_HUB_CACHE" \
        --tag "$TAG" \
        --datasets mammoth \
        --batch-size 256 --num-workers 15 \
        --out-dir "$EMB_ROOT"
else
    echo "open_clip: $(python -c 'import open_clip; print(open_clip.__file__)')"
    srun python "$ENC" \
        --ckpt "$CKPT" --tag "$TAG" --model ViT-T-16 \
        --datasets mammoth \
        --batch-size 256 --num-workers 15 \
        --out-dir "$EMB_ROOT"
fi

echo "DONE task $SLURM_ARRAY_TASK_ID"
