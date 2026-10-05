#!/usr/bin/env bash
# The downloaded query file overlaps the source base. Build a derived holdout
# dataset first and leave the downloaded files untouched.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_ROOT="${BENCH_ROOT:-/mnt/nvme/wxy/benchmarks}"
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  python3 "${SCRIPT_DIR}/prepare_disjoint_fvecs.py" \
    --source-base "${BENCH_ROOT}/msong/origin/msong_base.fvecs" \
    --source-query "${BENCH_ROOT}/msong/origin/msong_query.fvecs" \
    --output-base "${BENCH_ROOT}/msong_holdout/origin/msong_holdout_base.fvecs" \
    --output-query "${BENCH_ROOT}/msong_holdout/origin/msong_holdout_query.fvecs"
fi

DATASET="msong_holdout"
DEFAULT_NLIST=1000
DEFAULT_NPROBES="100"
DEFAULT_PIVOT_COUNTS="21 42" # ceil(420 * {5%, 10%})
DATA_WARNING="derived from msong after removing every exact query match from the indexed base"
source "${SCRIPT_DIR}/_run_dataset_dual.sh"
