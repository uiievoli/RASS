#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$ROOT/datasets/cosine_supplement}"
RAW_DIR="${RAW_DIR:-}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/cosine_supplement_$(date +%Y%m%d-%H%M%S)}"
DOWNLOAD_ONLY=0
DRY_RUN=0
FORWARD=()
while (($#)); do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --raw-dir) RAW_DIR="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --download-only) DOWNLOAD_ONLY=1; shift ;;
    --dry-run) DRY_RUN=1; FORWARD+=("$1"); shift ;;
    -h|--help)
      echo 'Usage: download_and_run_cosine_supplement.sh [--data-root DIR] [--raw-dir DIR] [--log-dir DIR] [--download-only] [experiment options]'
      "$ROOT/scripts/run_cosine_supplement.sh" --help; exit ;;
    *) FORWARD+=("$1"); shift ;;
  esac
done
RAW_DIR="${RAW_DIR:-$DATA_ROOT/.downloads}"
if ((!DRY_RUN)); then mkdir -p "$LOG_DIR"; exec > >(tee -a "$LOG_DIR/pipeline.log") 2>&1; fi
download=("$ROOT/scripts/download_cosine_datasets.sh" --data-root "$DATA_ROOT" --raw-dir "$RAW_DIR" \
  --datasets glove-200-angular,landmark-dino-768-cosine)
((DRY_RUN)) && download+=(--dry-run)
"${download[@]}"
if ((!DOWNLOAD_ONLY)); then
  "$ROOT/scripts/run_cosine_supplement.sh" --data-root "$DATA_ROOT" --log-dir "$LOG_DIR" "${FORWARD[@]}"
fi
