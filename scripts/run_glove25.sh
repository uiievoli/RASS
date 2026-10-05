#!/usr/bin/env bash
# Override example: PIVOT_COUNTS="16 64 128" NPROBES="10 100" bash scripts/run_glove25.sh
DATASET="glove25"
DEFAULT_NLIST=1024
DEFAULT_NPROBES="100"
DEFAULT_PIVOT_COUNTS="2 3" # ceil(25 * {5%, 10%})
source "$(dirname "$0")/_run_dataset_dual.sh"
