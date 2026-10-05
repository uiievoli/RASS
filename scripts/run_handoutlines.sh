#!/usr/bin/env bash
# Override example: PIVOT_COUNTS="8 16 32" NPROBES="4 8" bash scripts/run_handoutlines.sh
DATASET="HandOutlines"
DEFAULT_NLIST=32
DEFAULT_NPROBES="8"
DEFAULT_PIVOT_COUNTS="136 271" # ceil(2710 * {5%, 10%})
DEFAULT_SCOPE="global"
DATA_WARNING="coordinate 0 appears to contain the UCR class label; results include that feature"
source "$(dirname "$0")/_run_dataset_dual.sh"
