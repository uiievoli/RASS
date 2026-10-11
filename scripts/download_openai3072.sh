#!/usr/bin/env bash
# Precomputed OpenAI vectors; no OpenAI API key or embedding generation required.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT=""
RAW_DIR=""
LOCAL_HDF5=""
REVISION=b1e7761d0873d92cc1c7721f565ba76cb44b9c63
DATASET=simplewiki-openai-3072-normalized
SHA256=5b012e832cc6ffd72c248ce5e8ac12d34441c7692a4546e2a5672ebe2d4d4aa3
URL="https://huggingface.co/datasets/vector-index-bench/vibe/resolve/${REVISION}/${DATASET}.hdf5"
DRY_RUN=0
usage() {
  cat <<'EOF'
Usage: bash scripts/download_openai3072.sh --data-root DIR [options]
  --raw-dir DIR       Download cache, default DATA_ROOT/.downloads/openai3072.
  --hdf5 FILE         Use a manually downloaded local HDF5 without network access.
  --url URL           Mirror override; must still match the expected checksum.
  --sha256 HASH       Explicit source override, useful for local smoke fixtures.
  --dry-run          Print download, checksum and export commands only.
EOF
}
while (($#)); do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --raw-dir) RAW_DIR="$2"; shift 2 ;;
    --hdf5) LOCAL_HDF5="$2"; shift 2 ;;
    --url) URL="$2"; shift 2 ;;
    --sha256) SHA256="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -n "$DATA_ROOT" ]] || { echo '--data-root is required' >&2; exit 2; }
[[ "$SHA256" =~ ^[a-fA-F0-9]{64}$ ]] || { echo 'Invalid SHA256' >&2; exit 2; }
RAW_DIR="${RAW_DIR:-$DATA_ROOT/.downloads/openai3072}"
run() { if ((DRY_RUN)); then printf '%q ' "$@"; printf '\n'; else "$@"; fi; }
if ((DRY_RUN)); then
  file="${LOCAL_HDF5:-$RAW_DIR/$DATASET.hdf5}"
  [[ -n "$LOCAL_HDF5" ]] || run curl --fail --location --retry 8 --retry-all-errors \
    --connect-timeout 30 --continue-at - --output "$file.partial" "$URL"
  run sha256sum "${LOCAL_HDF5:-$file.partial}"
  [[ -n "$LOCAL_HDF5" ]] || run mv "$file.partial" "$file"
  run python3 "$ROOT/scripts/prepare_cosine_hdf5.py" --hdf5 "$file" --dataset "$DATASET" \
    --dimension 3072 --data-root "$DATA_ROOT" --sha256 "$SHA256" --batch-size 1024
  exit 0
fi
python3 -c 'import h5py,numpy'
mkdir -p "$RAW_DIR"
exec 9>"$RAW_DIR/.download.lock"
flock -n 9 || { echo 'Another downloader is active; OS lock releases on process exit.' >&2; exit 1; }
matches() { [[ -f "$1" ]] && [[ "$(sha256sum "$1" | cut -d ' ' -f 1)" == "${SHA256,,}" ]]; }
if [[ -n "$LOCAL_HDF5" ]]; then
  file="$LOCAL_HDF5"
  matches "$file" || { echo "Checksum mismatch: $file (preserved)" >&2; exit 1; }
else
  file="$RAW_DIR/$DATASET.hdf5"
  if [[ ! -f "$file" ]]; then
    # Check first: an interrupted rename can leave a complete partial. This
    # avoids sending a Range request at EOF and receiving HTTP 416.
    if ! matches "$file.partial"; then
      echo "DOWNLOAD $DATASET revision=$REVISION"
      curl --fail --location --retry 8 --retry-all-errors --connect-timeout 30 \
        --continue-at - --output "$file.partial" "$URL"
    fi
    matches "$file.partial" || { echo "Incomplete/corrupt partial: $file.partial (preserved)" >&2; exit 1; }
    python3 "$ROOT/scripts/prepare_cosine_hdf5.py" --hdf5 "$file.partial" --dimension 3072 --validate-only
    mv "$file.partial" "$file"
  fi
  matches "$file" || { echo "Checksum mismatch: $file (preserved)" >&2; exit 1; }
fi
python3 "$ROOT/scripts/prepare_cosine_hdf5.py" --hdf5 "$file" --dataset "$DATASET" \
  --dimension 3072 --data-root "$DATA_ROOT" --sha256 "${SHA256,,}" --batch-size 1024
python3 - "$DATA_ROOT" "$URL" "$REVISION" "$SHA256" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])/'simplewiki-openai-3072-normalized'
metadata=json.loads((root/'origin/cosine_source.json').read_text())
official=sys.argv[4].lower()=='5b012e832cc6ffd72c248ce5e8ac12d34441c7692a4546e2a5672ebe2d4d4aa3'
metadata.update(download_url=sys.argv[2],revision=sys.argv[3],expected_sha256=sys.argv[4],
                official_snapshot=official,
                provider='VIBE' if official else 'custom checksum override',
                embedding_model='text-embedding-3-large' if official else 'not verified',
                dataset_status='upstream archived; snapshot remains available' if official else 'custom source')
(root/'download_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
print('READY',root,'base=',metadata['base_vectors'],'queries=',metadata['queries'],'dimension=',metadata['dimension'])
PY
