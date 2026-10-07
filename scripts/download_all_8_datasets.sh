#!/usr/bin/env bash
# Canonical downloader for the active eight-dataset suite.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${ROOT}/scripts/download_all_10_datasets.sh" "$@"
