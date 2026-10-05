#!/usr/bin/env bash
# Override example: PIVOT_COUNTS="16 64 128" NPROBES="10 100" bash scripts/run_sift1m.sh
DATASET="sift1m"
DEFAULT_NLIST=1000
DEFAULT_NPROBES="100"
DEFAULT_PIVOT_COUNTS="7 13" # ceil(128 * {5%, 10%})
DATA_WARNING="query vectors 2017 and 2688 are duplicates; neither exactly overlaps the base"
source "$(dirname "$0")/_run_dataset_dual.sh"
