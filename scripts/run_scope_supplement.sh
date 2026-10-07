#!/usr/bin/env bash
# Supplemental pivot-scope experiments used to derive a cross-dataset policy.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/logs/pivot-scope-supplement-${RUN_ID}}"
DRIVER="${ROOT}/scripts/run_pivot_latency.sh"

mkdir -p "${OUT_ROOT}"

run_set() {
  local dataset="$1" pivots="$2" scope="$3" loops="$4"
  ONLY_DATASETS="${dataset}" \
  PIVOT_COUNTS="${pivots}" \
  SCOPE="${scope}" \
  PERF_LOOPS="${loops}" \
  OUT_ROOT="${OUT_ROOT}" \
  SKIP_EXISTING=1 \
  bash "${DRIVER}"
}

# Matching per-list/global runs at the best pivot observed in the earlier sweep.
run_set fasion_mnist_784 "1 48" per_list 100
run_set fasion_mnist_784 "48" global 100

run_set msong_holdout "1 34" per_list 100
run_set msong_holdout "34" global 100

run_set sift1m "1 23" per_list 50
run_set sift1m "23" global 50

run_set glove25 "1 14" per_list 100
run_set glove25 "14" global 100

# Low-P points missing from the small-dataset global sweep.
run_set HandOutlines "1 7 16 64" global 1000

echo "[$(date -Is)] SUPPLEMENT FINISHED results=${OUT_ROOT}" | tee -a "${OUT_ROOT}/supplement.log"
