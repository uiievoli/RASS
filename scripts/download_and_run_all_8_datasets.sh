#!/usr/bin/env bash
# Canonical download/build/run entry point for the active eight datasets.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${ROOT}/scripts/download_and_run_all_10_datasets.sh" "$@"
