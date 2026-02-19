#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="parallel_tune_test.log"
: > "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

run_cmd() {
  echo
  echo "[RUN] $*"
  "$@"
}

run_cmd pytest -s ./tests/test_norm_ops.py::test_accuracy_rmsnorm --parallel 8
run_cmd pytest -s ./tests/test_convolution_ops.py::test_accuracy_conv1d --parallel 8

run_cmd pytest -s ./benchmark/test_norm_perf.py::test_perf_rms_norm --dtypes bfloat16 --parallel 8
run_cmd pytest -s ./benchmark/test_convolution_perf.py::test_perf_conv2d --dtypes bfloat16 --parallel 8
run_cmd pytest -s ./benchmark/test_blas_perf.py::test_blas_benchmark -m mm --mode kernel --dtypes bfloat16 --parallel 8
run_cmd pytest -s ./benchmark/test_blas_perf.py::test_blas_benchmark -m addmm --mode kernel --dtypes bfloat16 --parallel 8

echo

echo "[DONE] parallel_tune_test.sh finished successfully."
echo "[LOG] Output saved to $LOG_FILE"
