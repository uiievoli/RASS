#!/usr/bin/env bash
# Override example: PIVOT_COUNTS="8 16 32" NPROBES="8 16" bash scripts/run_starlightcurves.sh
DATASET="StarLightCurves"
DEFAULT_NLIST=128
DEFAULT_NPROBES="16"
DEFAULT_PIVOT_COUNTS="52 103" # ceil(1025 * {5%, 10%})
DEFAULT_SCOPE="global"
DATA_WARNING="coordinate 0 appears to contain the UCR class label; results include that feature"
source "$(dirname "$0")/_run_dataset_dual.sh"
