#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 00:30:00
#SBATCH -A lt200394
#SBATCH -J cb_smoke_mc2_e8
#SBATCH -o /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%j.out
#SBATCH -e /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/%x_%j.out

# Smoke gate for habibi's metaclip2_kd run: ImageNet-1k val on epoch_8 only.
# Validates mc2_eval_env (habibi's open_clip) + arch ViT-T-16 (CLIP-BPE) + ckpt load.
# Expected top1 ~ 0.038 (training-logged e8 imagenet-zeroshot-val-top1).

set -euo pipefail
source /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/env.sh
module load Mamba/23.11.0-0
source activate mc2_eval_env

echo "open_clip from: $(python -c 'import open_clip; print(open_clip.__file__)')"
python -m clip_benchmark.cli eval \
    --model ViT-T-16 --pretrained "$MC2_E8_CKPT" \
    --dataset imagenet1k-unverified --dataset_root "$IMAGENET_ROOT" \
    --task zeroshot_classification --language en \
    --batch_size 512 --num_workers 8 \
    --output /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/mc2_smoke_imagenet_e8.json

echo "=== RESULT ==="
cat /lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/logs/mc2_smoke_imagenet_e8.json
echo
echo "GATE: expect acc1 ~ 0.038"
