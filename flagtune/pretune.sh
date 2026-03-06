#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="./log/flagtune/pretune"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/pretune.log"
: > "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

run_cmd() {
  echo
  echo "[RUN] $*"
  "$@"
}
run_cmd  pytest benchmark/test_blas_perf.py -m "mm" -s --shape_file flagtune/hot_mm_shapes.yaml --level core --mode kernel --dtypes bfloat16 --parallel 8 -v

echo

echo "[DONE] pretune.sh finished successfully."
echo "[LOG] Output saved to $LOG_FILE"
