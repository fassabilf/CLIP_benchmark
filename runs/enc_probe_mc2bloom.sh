#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 04:00:00
#SBATCH -A lt200394
#SBATCH -J mc2bloom_enc_probe
#SBATCH -a 0-51%10
#SBATCH -o /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%A_%a.out
#SBATCH -e /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%A_%a.out

# Phase A: encode ablation-Bloom checkpoints (e8/e16/e24/e32) over the same SEA training
# shards used for the mc2v3 / mc2wit / mc2cg probes (bloom + cgoe×6 + wit×6 = 13 entries
# per epoch). 4 epochs × 13 entries = 52 tasks, <=10 concurrent.
# Bloom is in-domain here (ablation-Bloom was trained on it); cgoe/wit are cross-domain
# (model never saw those pairs) — same setup used to show source specificity for
# ablation-WIT and ablation-CG (Table 1 / Update Log finding #3). This closes the
# "Bloom ablation still pending" gap in Open Question #5.
# Encoder auto-skips existing .npz -> safe to resubmit.
# After all tasks finish, run s_phase_b_retrieval.sh to get cos/R@1 for results_main.tex.

set -euo pipefail
source /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/env.sh
module load Mamba/23.11.0-0
source activate mc2_eval_env

ENC=/lustrefs/disk/project/lt200394-thllmV/kd_dataset/scripts/encode_train_embeddings.py
EMB_ROOT=/lustrefs/disk/project/lt200394-thllmV/kd_dataset/eval/train_probe/emb

TAGS=(mc2bloom_e8 mc2bloom_e16 mc2bloom_e24 mc2bloom_e32)
DSENTRIES=(
  "bloom:" \
  "cgoe:id" "cgoe:jv" "cgoe:ms" "cgoe:su" "cgoe:th" "cgoe:vi" \
  "wit:id"  "wit:jv"  "wit:ms"  "wit:my"  "wit:th"  "wit:vi"
)
JOBS=()
for t in "${TAGS[@]}"; do for de in "${DSENTRIES[@]}"; do JOBS+=("$t:$de"); done; done

IFS=':' read -r TAG DS LANG <<< "${JOBS[$SLURM_ARRAY_TASK_ID]}"
case "$TAG" in
  mc2bloom_e8)  CKPT="$MC2BLOOM_E8_CKPT" ;;
  mc2bloom_e16) CKPT="$MC2BLOOM_E16_CKPT" ;;
  mc2bloom_e24) CKPT="$MC2BLOOM_E24_CKPT" ;;
  mc2bloom_e32) CKPT="$MC2BLOOM_E32_CKPT" ;;
  *) echo "bad tag $TAG"; exit 1 ;;
esac

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
