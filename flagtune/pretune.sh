#!/usr/bin/env bash
set -euo pipefail

MODEL="qwen3.5"
OP="mm"
CACHE_DIR="/root/.flaggems"
BEST="true"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --op)
      OP="$2"
      shift 2
      ;;
    --cache-dir)
      CACHE_DIR="$2"
      shift 2
      ;;
    --best)
      BEST="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--model <model_name>] [--op <op_name>] [--cache-dir <dir>] [--best <true|false>]"
      echo "Default: $0 --model qwen3.5 --op mm --cache-dir /root/.flaggems --best true"
      echo "Example: $0 --model qwen3.5 --op mm --cache-dir /root/.flaggems --best true"
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ -n "$OP" ]]; then
  LOG_DIR="./log/flagtune/${MODEL}/${OP}/pretune"
else
  LOG_DIR="./log/flagtune/${MODEL}/pretune"
fi
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/pretune.log"
: > "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

run_cmd() {
  echo
  echo "[RUN] $*"
  "$@"
}

print_stage_banner() {
  local stage_name="$1"
  echo "[INFO] ============================= pretuning ${OP} operator with ${stage_name} configuration.============================="
}

clear_flaggems_cache() {
  echo "[INFO] Deleting Flaggems cache $CACHE_DIR."
  run_cmd rm -rf "$CACHE_DIR"
}

replace_op_config() {
  echo "[INFO] Replacing ${OP} configuration."
  run_cmd python3 flagtune/processing/config_replace.py --tune-yaml "$MM_TUNE_YAML" --target "$MM_PY"
}

restore_mm_from_backup() {
  local suffix="${1:-}"
  if [[ -n "$MM_PY_BACKUP" && -f "$MM_PY_BACKUP" ]]; then
    cp "$MM_PY_BACKUP" "$MM_PY"
    echo "[RESTORE] Restored $MM_PY${suffix}"
  fi
}

run_mm_benchmark() {
  local stage_name="$1"
  local shape_file="$2"
  local need_clear_cache="${3:-true}"
  local need_replace_config="${4:-false}"

  print_stage_banner "$stage_name"
  if [[ "${need_clear_cache,,}" == "true" ]]; then
    clear_flaggems_cache
  fi
  if [[ "${need_replace_config,,}" == "true" ]]; then
    replace_op_config
  fi

  run_cmd pytest benchmark/test_blas_perf.py -m "$OP" -s --shape_file "$shape_file" --level core --mode kernel --dtypes bfloat16 --parallel 8 -v
}

MM_PY="src/flag_gems/runtime/backend/_nvidia/hopper/ops/mm.py"
MM_TUNE_YAML="flagtune/tune-config/mm_hopper_tma.yaml"
MM_PY_BACKUP=""
REPORT_MD="flagtune/reports/${MODEL}_${OP}.md"
REPORT_XLSX="flagtune/reports/${MODEL}_${OP}.xlsx"

cleanup() {
  if [[ -n "$MM_PY_BACKUP" && -f "$MM_PY_BACKUP" ]]; then
    restore_mm_from_backup
    rm -f "$MM_PY_BACKUP"
  fi
}

trap cleanup EXIT

echo "[INFO] Flagtune started."

run_cmd python flagtune/processing/shape-gen.py --model "$MODEL"
if [[ "$OP" == "mm" ]]; then
  MM_PY_BACKUP="$(mktemp /tmp/mm.py.backup.XXXXXX)"
  cp "$MM_PY" "$MM_PY_BACKUP"

  run_mm_benchmark "default" "flagtune/shape-config/${MODEL}.yaml" true false
  run_mm_benchmark "expand" "flagtune/shape-config/${MODEL}.yaml" true true

  echo "[INFO] Generating pretune report."
  run_cmd python3 flagtune/processing/summary.py --model "$MODEL" --op "$OP"

  if [[ "${BEST,,}" == "true" ]]; then
    restore_mm_from_backup " before best benchmark reruns."
 
    run_mm_benchmark "best default" "flagtune/shape-config/${MODEL}_lose.yaml" true false
    run_mm_benchmark "best expand" "flagtune/shape-config/${MODEL}_gain.yaml" false true
  fi

fi


echo

echo "[DONE] Flagtune finished successfully."
echo "[LOG] Output Log saved to $LOG_FILE."
echo "[REPORT] Markdown report saved to $REPORT_MD."
echo "[REPORT] Excel report saved to $REPORT_XLSX."
