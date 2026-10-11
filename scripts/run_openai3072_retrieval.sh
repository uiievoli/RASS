#!/usr/bin/env bash
# Independent entry point for calibration, fine-P screening and retrieval curves.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "$ROOT/scripts/run_openai3072_fine_pivot.py" "$@"
