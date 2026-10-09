#!/usr/bin/env bash
# Canonical eight-dataset entry point. The implementation remains in the
# legacy-named script so existing automation keeps working.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ONLY_DATASETS="${ONLY_DATASETS:-nuswide,fasion_mnist_784,msong_holdout,sift1m,glove25,StarLightCurves,dbpedia1536m_holdout,sift1b}"
exec "${ROOT}/scripts/run_all_10_datasets.sh" "$@"
