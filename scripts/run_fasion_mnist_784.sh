#!/usr/bin/env bash
# Override example: PIVOT_COUNTS="8 16 32" NPROBES="16 32" bash scripts/run_fasion_mnist_784.sh
DATASET="fasion_mnist_784"
DEFAULT_NLIST=256
DEFAULT_NPROBES="32"
DEFAULT_PIVOT_COUNTS="40 79" # ceil(784 * {5%, 10%})
source "$(dirname "$0")/_run_dataset_dual.sh"
