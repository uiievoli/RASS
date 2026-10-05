#!/usr/bin/env bash
# Override example: PIVOT_COUNTS="16 64 128" NPROBES="16 64" bash scripts/run_nuswide.sh
DATASET="nuswide"
DEFAULT_NLIST=512
DEFAULT_NPROBES="64"
DEFAULT_PIVOT_COUNTS="25 50" # ceil(500 * {5%, 10%})
DATA_WARNING="base vector 25081 is all-zero; the file is otherwise structurally valid"
source "$(dirname "$0")/_run_dataset_dual.sh"
