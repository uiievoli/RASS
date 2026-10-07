#!/usr/bin/env bash
# Canonical eight-dataset entry point. The implementation remains in the
# legacy-named script so existing automation keeps working.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${ROOT}/scripts/run_all_10_datasets.sh" "$@"
