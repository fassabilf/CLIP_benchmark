#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 04:00:00
#SBATCH -A lt200394
#SBATCH -J clipkd_mammoth_bpe_v1_enc_probe
#SBATCH -a 0-51%10
#SBATCH -o /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%A_%a.out
#SBATCH -e /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%A_%a.out

# Phase A: encode clipkd_mammoth_bpe_v1 checkpoints (e8/e16/e24/e32) over the same SEA training
# shards used for the mc2/mammoth probes (bloom + cgoe x6 + wit x6 = 13 entries per epoch).
# 4 epochs x 13 entries = 52 tasks, <=10 concurrent.
# Checkpoint's token_embedding is vocab=49408 (CLIP-BPE tokenizer) -> use mc2_eval_env, NOT
# mteb_env2 (that's the SigLIP2 256000-vocab config used by ckdonly_v1/MAMMOTH_KD_* despite
# similar dir naming) -> size mismatch on token_embedding.weight.
# Encoder auto-skips existing .npz -> safe to resubmit.
# After all tasks finish, run runs/ai_clipkd_mammoth_bpe_v1_retrieval_probe.sh (Phase B) to get
# cos/R@1.

set -euo pipefail
source /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/env.sh
module load Mamba/23.11.0-0
source activate mc2_eval_env

ENC=/lustrefs/disk/project/lt200394-thllmV/kd_dataset/scripts/encode_train_embeddings.py
EMB_ROOT=/lustrefs/disk/project/lt200394-thllmV/kd_dataset/eval/train_probe/emb

TAGS=(clipkd_mammoth_bpe_v1_e8 clipkd_mammoth_bpe_v1_e16 clipkd_mammoth_bpe_v1_e24 clipkd_mammoth_bpe_v1_e32)
DSENTRIES=(
  "bloom:" \
  "cgoe:id" "cgoe:jv" "cgoe:ms" "cgoe:su" "cgoe:th" "cgoe:vi" \
  "wit:id"  "wit:jv"  "wit:ms"  "wit:my"  "wit:th"  "wit:vi"
)
JOBS=()
for t in "${TAGS[@]}"; do for de in "${DSENTRIES[@]}"; do JOBS+=("$t:$de"); done; done

IFS=':' read -r TAG DS LANG <<< "${JOBS[$SLURM_ARRAY_TASK_ID]}"
case "$TAG" in
  clipkd_mammoth_bpe_v1_e8)  CKPT="$CLIPKD_MAMMOTH_BPE_V1_E8_CKPT" ;;
  clipkd_mammoth_bpe_v1_e16) CKPT="$CLIPKD_MAMMOTH_BPE_V1_E16_CKPT" ;;
  clipkd_mammoth_bpe_v1_e24) CKPT="$CLIPKD_MAMMOTH_BPE_V1_E24_CKPT" ;;
  clipkd_mammoth_bpe_v1_e32) CKPT="$CLIPKD_MAMMOTH_BPE_V1_E32_CKPT" ;;
  *) echo "bad tag $TAG"; exit 1 ;;
esac

if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: checkpoint not found at $CKPT"
  echo "Check $CLIPKD_MAMMOTH_BPE_V1_DIR for the actual filenames."
  exit 1
fi

LANG_ARG=()
[[ -n "$LANG" ]] && LANG_ARG=(--langs "$LANG")

echo "task=$SLURM_ARRAY_TASK_ID tag=$TAG ds=$DS lang=${LANG:-ALL}"
echo "open_clip: $(python -c 'import open_clip; print(open_clip.__file__)')"
srun python "$ENC" \
    --ckpt "$CKPT" --tag "$TAG" --model ViT-T-16 \
    --datasets "$DS" "${LANG_ARG[@]}" \
    --batch-size 512 --num-workers 15 \
    --out-dir "$EMB_ROOT"
echo "DONE task $SLURM_ARRAY_TASK_ID"
