#!/bin/bash
# Phase B: run retrieval_from_emb_safe.py for ALL epochs of selflearn_mammoth_v1.
# Run on login/transfer node after ac_selflearn_enc_probe.sh finishes.
#
# Usage:
#   bash runs/ac_selflearn_retrieval_probe.sh
# Monitor:
#   tail -f nohup_retrieval_selflearn_mammoth_v1.out

set -uo pipefail

SCRIPT="/project/lt200394-thllmV/kd_dataset/scripts/retrieval_from_emb_safe.py"
EMB_ROOT="/project/lt200394-thllmV/kd_dataset/eval/train_probe/emb"

TAGS=(selflearn_mammoth_v1_e8 selflearn_mammoth_v1_e16 selflearn_mammoth_v1_e24 selflearn_mammoth_v1_e32)

DRIVER_LOG="$(pwd)/nohup_retrieval_selflearn_mammoth_v1.out"

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

  if [[ -f "$OUT_CSV" ]]; then
    cp "$OUT_CSV" "${OUT_CSV}.bak.$(date +%Y%m%d_%H%M%S)"
    echo "=== [$TAG] backed up existing csv ==="
  fi

  python "$SCRIPT" \
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
