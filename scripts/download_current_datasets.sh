#!/usr/bin/env bash
# Current experiment suite: ten active datasets; SIFT1B is opt-in.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 -u "${ROOT}/scripts/download_current_datasets.py" "$@"
