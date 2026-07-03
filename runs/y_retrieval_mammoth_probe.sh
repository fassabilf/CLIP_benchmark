#!/bin/bash
# Phase B: recompute train_probe_summary.csv (adds the mammoth-all row) for every tag
# back-filled by runs/y_encode_mammoth_probe.sh (Phase A). Run on a login/transfer node
# AFTER Phase A finishes.
#
# Usage:
#   bash runs/y_retrieval_mammoth_probe.sh
# Monitor:
#   tail -f nohup_retrieval_mammoth.out
#
# Output: ${EMB_ROOT}/<tag>/train_probe_summary.csv (existing csv backed up first).
# Merge the updated per-tag CSVs into train_probe_curve_long_combined.csv (via
# combine_train_probe_summaries.py) to add the Mammoth column to the train-probe table
# (analyze_train_probe.py / compute_tables.py pick up the "mammoth" dataset automatically).

set -uo pipefail

# retrieval_from_emb_safe.py is pure CPU (numpy/torch/pandas). Call the env's python by
# absolute path rather than `module load` + `source activate` -- those are shell
# functions that don't survive the `nohup bash "$0" &` re-exec below (this script isn't
# run under sbatch/a login shell, unlike the encode scripts where the pattern works).
PYTHON="/home/ffirdaus/.conda/envs/mc2_eval_env/bin/python"

SCRIPT="/project/lt200394-thllmV/kd_dataset/scripts/retrieval_from_emb_safe.py"
EMB_ROOT="/project/lt200394-thllmV/kd_dataset/eval/train_probe/emb"

TAGS=(
  mc2_e32
  mc2cc_e32
  mc2v3_e32
  clipkd_mammoth_bpe_v1_e32
  mc2wit_e32
  mc2bloom_e32
  mc2cg_e32
  metaclip2_b16
)

DRIVER_LOG="$(pwd)/nohup_retrieval_mammoth.out"

run_one() {
  local TAG="$1"
  local EMB_DIR="${EMB_ROOT}/${TAG}"
  local OUT_CSV="${EMB_DIR}/train_probe_summary.csv"
  local LOG_FILE="${EMB_DIR}/${TAG}_retrieval.log"

  echo "=== [$TAG] starting $(date) ==="

  if [[ ! -d "$EMB_DIR" ]]; then
    echo "=== [$TAG] SKIP: emb dir not found at $EMB_DIR ==="
    return
  fi
  if [[ ! -f "${EMB_DIR}/mammoth-all.npz" ]]; then
    echo "=== [$TAG] SKIP: mammoth-all.npz not found (Phase A not done for this tag?) ==="
    return
  fi

  if [[ -f "$OUT_CSV" ]]; then
    cp "$OUT_CSV" "${OUT_CSV}.bak.$(date +%Y%m%d_%H%M%S)"
    echo "=== [$TAG] backed up existing csv ==="
  fi

  "$PYTHON" "$SCRIPT" \
      --emb-dir "$EMB_DIR" \
      --tag "$TAG" \
      --out-csv "$OUT_CSV" \
      --log-file "$LOG_FILE" \
      --gallery-size 10000 --repeats 5

  local RC=$?
  if [[ $RC -ne 0 ]]; then
    echo "=== [$TAG] FAILED (exit code $RC), see $LOG_FILE ==="
  else
    echo "=== [$TAG] done $(date), csv -> $OUT_CSV ==="
  fi
}

main() {
  if [[ ! -f "$SCRIPT" ]]; then
    echo "ERROR: script not found at $SCRIPT" >&2
    exit 1
  fi
  for TAG in "${TAGS[@]}"; do
    run_one "$TAG"
  done
  echo "=== ALL DONE $(date) ==="
}

if [[ "${_RETRIEVAL_BG:-0}" == "1" ]]; then
  main
  exit 0
fi

export _RETRIEVAL_BG=1
nohup bash "$0" > "$DRIVER_LOG" 2>&1 &
PID=$!
echo "Started PID=$PID"
echo "  driver log -> $DRIVER_LOG"
echo ""
echo "Tags queued (sequential, gallery_size=10000): ${TAGS[*]}"
echo ""
echo "Monitor with:"
echo "  tail -f $DRIVER_LOG"
