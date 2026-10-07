#!/usr/bin/env bash
# Run the k=1 latency-recall sweep for every dataset-specific runner.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/latency-recall-k1-${RUN_ID}}"
mkdir -p "${LOG_ROOT}"

export K=1
export BUILD=0
export METHODS="${METHODS:-pca}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc)}"
export STATS_THREADS="${STATS_THREADS:-${OMP_NUM_THREADS}}"
export PERF_THREADS="${PERF_THREADS:-${OMP_NUM_THREADS}}"

run_dataset() {
  local script="$1"
  local dataset="$2"
  local nprobes="$3"
  local out_dir="${LOG_ROOT}/${dataset}"

  echo "[$(date -Is)] START dataset=${dataset} k=${K} nprobes=${nprobes}" | tee -a "${LOG_ROOT}/run.log"
  OUT_DIR="${out_dir}" NPROBES="${nprobes}" \
    bash "${ROOT}/scripts/${script}" >>"${LOG_ROOT}/run.log" 2>&1
  echo "[$(date -Is)] DONE dataset=${dataset} summary=${out_dir}/summary.csv" | tee -a "${LOG_ROOT}/run.log"
}

run_dataset run_nuswide.sh nuswide \
  "1 3 5 7 10 30 50 70 100 150 200 250 300 350 400 450 500 510 513 516 0"
run_dataset run_fasion_mnist_784.sh fasion_mnist_784 \
  "1 3 5 7 10 20 30 40 50 60 70 80 90 150 200 0"
run_dataset run_msong.sh msong_holdout \
  "1 3 5 7 10 30 50 70 100 150 200 250 300 350 600 800 0"
run_dataset run_sift1m.sh sift1m \
  "1 3 5 7 10 30 50 70 100 150 200 250 300 400 600 800 0"
run_dataset run_glove25.sh glove25 \
  "1 3 5 7 10 30 50 70 100 150 200 250 300 350 400 500 600 750 900 0"
run_dataset run_starlightcurves.sh StarLightCurves \
  "1 3 5 7 10 12 14 16 18 0"

echo "[$(date -Is)] FINISHED results=${LOG_ROOT}" | tee -a "${LOG_ROOT}/run.log"
