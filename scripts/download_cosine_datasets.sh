#!/usr/bin/env bash
# Download canonical ANN-Benchmarks GloVe-angular and VIBE Landmark-DINO-cosine.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$ROOT/datasets/cosine_supplement}"
RAW_DIR="${RAW_DIR:-}"
ONLY_DATASETS="${ONLY_DATASETS:-glove-200-angular,landmark-dino-768-cosine}"
DRY_RUN=0
while (($#)); do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --raw-dir) RAW_DIR="$2"; shift 2 ;;
    --datasets) ONLY_DATASETS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) echo 'Usage: download_cosine_datasets.sh [--data-root DIR] [--raw-dir DIR] [--datasets LIST] [--dry-run]'; exit ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
RAW_DIR="${RAW_DIR:-$DATA_ROOT/.downloads}"
read -r -a datasets <<< "${ONLY_DATASETS//,/ }"
for dataset in "${datasets[@]}"; do
  case "$dataset" in
    glove-200-angular|landmark-dino-768-cosine) ;;
    *) echo "Unknown cosine dataset: $dataset" >&2; exit 2 ;;
  esac
done
if ((!DRY_RUN)); then
  python3 -c 'import h5py,numpy'
  mkdir -p "$RAW_DIR"
  # One writer per raw directory; aborted processes release this lock.
  exec 9>"$RAW_DIR/.download.lock"
  flock -n 9 || { echo "Another cosine download owns $RAW_DIR/.download.lock" >&2; exit 1; }
fi
run() { if ((DRY_RUN)); then printf '%q ' "$@"; printf '\n'; else "$@"; fi; }
for dataset in "${datasets[@]}"; do
  checksum=()
  case "$dataset" in
    glove-200-angular)
      dimension=200; expected=962819488
      url="${GLOVE_200_URL:-https://ann-benchmarks.com/glove-200-angular.hdf5}" ;;
    landmark-dino-768-cosine)
      dimension=768; expected=2341733696
      url="${LANDMARK_DINO_URL:-https://huggingface.co/datasets/vector-index-bench/vibe/resolve/07b387891a221b7b073b83d2f752b76462e5fa03/landmark-dino-768-cosine.hdf5}"
      checksum=(--sha256 b13bf651afbc5963fa2ecca73fa78bac9fe68889528935bdcf26ab333dd792d2) ;;
  esac
  file="$RAW_DIR/$dataset.hdf5"
  if [[ ! -f "$file" ]]; then
    echo "[$(date -Is)] DOWNLOAD $dataset"
    # Resume from the partial file. A completed partial (e.g. interruption before
    # rename) is validated below without sending a failing Range request.
    if ((DRY_RUN)) || [[ ! -f "$file.partial" || "$(stat -c %s "$file.partial")" != "$expected" ]]; then
      run curl --fail --location --retry 8 --retry-all-errors --connect-timeout 30 \
        --continue-at - --output "$file.partial" "$url"
    fi
    if ((!DRY_RUN)); then
      [[ "$(stat -c %s "$file.partial")" == "$expected" ]] || { echo "Incomplete download: $file.partial" >&2; exit 1; }
    fi
    run python3 "$ROOT/scripts/prepare_cosine_hdf5.py" --hdf5 "$file.partial" --dataset "$dataset" --dimension "$dimension" --validate-only
    run mv "$file.partial" "$file"
  fi
  if ((!DRY_RUN)); then
    [[ "$(stat -c %s "$file")" == "$expected" ]] || { echo "Unexpected size: $file" >&2; exit 1; }
  fi
  run python3 "$ROOT/scripts/prepare_cosine_hdf5.py" --hdf5 "$file" --dataset "$dataset" --dimension "$dimension" \
    --data-root "$DATA_ROOT" "${checksum[@]}"
done
