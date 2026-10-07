#!/usr/bin/env bash
# Download the eight active datasets used by scripts/run_all_8_datasets.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-}"
RAW_DIR="${RAW_DIR:-}"
ONLY_DATASETS="${ONLY_DATASETS:-}"
DRY_RUN=0
DOWNLOAD_CONNECTIONS="${DOWNLOAD_CONNECTIONS:-16}"

TRIBASE_BUNDLE_ID="${TRIBASE_BUNDLE_ID:-12wFLDNStJU02pEn7VcAs00LyS7uzcAbl}"
DBPEDIA_REPO="${DBPEDIA_REPO:-Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M}"
BIGANN_REPO="${BIGANN_REPO:-jkhe/bigann}"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

usage() {
  cat <<'EOF'
Usage:
  scripts/download_all_8_datasets.sh --data-root DIR [options]

Options:
  --data-root DIR       Final Tribase dataset root (required).
  --raw-dir DIR         Archive/raw download root; default DATA_ROOT/.downloads.
  --datasets LIST       Comma/space-separated subset of the eight datasets.
  --dry-run             Print download and preparation commands only.
  -h, --help            Show this help.

Dataset names:
  nuswide fasion_mnist_784 msong_holdout sift1m glove25
  StarLightCurves dbpedia1536m_holdout sift1b

The first six datasets are extracted as ready-to-use fvecs. DBpedia is
downloaded from Hugging Face Parquet and prepared as a disjoint fvecs holdout.
SIFT1B stays in its original integer format on disk. query.cpp decodes it to
float in RAM and builds the same float Index as the other datasets.

Dependencies:
  curl, unzip, tar, gzip, gdown, and Python numpy+pyarrow.
  aria2c is optional and enables multi-connection HTTP/S3 downloads.
EOF
}

while (($#)); do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --raw-dir) RAW_DIR="$2"; shift 2 ;;
    --datasets) ONLY_DATASETS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "${DATA_ROOT}" ]] || { echo "--data-root is required" >&2; exit 2; }
RAW_DIR="${RAW_DIR:-${DATA_ROOT}/.downloads}"

BUNDLE_DATASETS=(
  nuswide fasion_mnist_784 msong_holdout sift1m glove25 StarLightCurves
)
ALL_DATASETS=("${BUNDLE_DATASETS[@]}" dbpedia1536m_holdout sift1b)

selected() {
  local needle="$1"
  [[ -z "${ONLY_DATASETS}" ]] && return 0
  local normalized=" ${ONLY_DATASETS//,/ } "
  [[ "${normalized}" == *" ${needle} "* ]]
}

if [[ -n "${ONLY_DATASETS}" ]]; then
  read -r -a requested <<<"${ONLY_DATASETS//,/ }"
  for name in "${requested[@]}"; do
    known=0
    for dataset in "${ALL_DATASETS[@]}"; do
      [[ "${name}" == "${dataset}" ]] && { known=1; break; }
    done
    ((known)) || { echo "Unknown dataset: ${name}" >&2; exit 2; }
  done
fi

log() { echo "[$(date -Is)] $*"; }

run() {
  if ((DRY_RUN)); then printf '%q ' "$@"; printf '\n'; else "$@"; fi
}

need() {
  ((DRY_RUN)) && return 0
  command -v "$1" >/dev/null || { echo "Required command not found: $1" >&2; exit 1; }
}

download() {
  local url="$1" output="$2"
  if [[ -s "${output}" ]]; then log "SKIP existing ${output}"; return; fi
  run mkdir -p "$(dirname "${output}")"
  log "DOWNLOAD ${url}"
  if command -v aria2c >/dev/null; then
    run aria2c --continue=true --max-connection-per-server="${DOWNLOAD_CONNECTIONS}" \
      --split="${DOWNLOAD_CONNECTIONS}" --min-split-size=16M --file-allocation=none \
      --max-tries=0 --retry-wait=5 --allow-overwrite=true --auto-file-renaming=false \
      --dir="$(dirname "${output}")" \
      --out="$(basename "${output}").partial" "${url}"
  else
    run curl --fail --location --retry 8 --retry-all-errors \
      --continue-at - --output "${output}.partial" "${url}"
  fi
  if ((!DRY_RUN)) && [[ ! -s "${output}.partial" ]]; then
    echo "Download completed without a usable file: ${output}.partial" >&2
    exit 1
  fi
  run mv "${output}.partial" "${output}"
}

decompress_gzip() {
  local source="$1" destination="$2"
  if [[ -s "${destination}" ]]; then
    log "SKIP decompressed ${destination}"
    return
  fi
  if ((DRY_RUN)); then
    echo "gzip -dc ${source} > ${destination}.partial && mv ${destination}.partial ${destination}"
    return
  fi
  mkdir -p "$(dirname "${destination}")"
  gzip -dc "${source}" >"${destination}.partial"
  mv "${destination}.partial" "${destination}"
}

parquet_valid() {
  local path="$1"
  [[ -s "${path}" ]] || return 1
  python3 - "${path}" <<'PY' >/dev/null 2>&1
import sys
import pyarrow.parquet as pq

path = sys.argv[1]
parquet = pq.ParquetFile(path)
if parquet.metadata.num_rows <= 0 or parquet.metadata.num_row_groups <= 0:
    raise SystemExit(1)
PY
}

bundle_needed=0
for dataset in "${BUNDLE_DATASETS[@]}"; do
  selected "${dataset}" && bundle_needed=1
done

if ((bundle_needed)); then
  need unzip
  if ! command -v gdown >/dev/null && ((!DRY_RUN)); then
    echo "The Tribase bundle requires gdown: python3 -m pip install gdown" >&2
    exit 1
  fi
  bundle_zip="${RAW_DIR}/tribase-benchmarks.zip"
  bundle_extract="${RAW_DIR}/tribase-benchmarks"
  bundle_marker="${bundle_extract}/.extract-complete"
  if [[ ! -s "${bundle_zip}" ]]; then
    run mkdir -p "${RAW_DIR}"
    log "DOWNLOAD Tribase benchmark bundle"
    # Passing the Drive file ID directly works with both older gdown releases
    # (which do not provide --fuzzy) and current releases.
    run gdown --continue -O "${bundle_zip}.partial" "${TRIBASE_BUNDLE_ID}"
    run mv "${bundle_zip}.partial" "${bundle_zip}"
  fi
  if [[ ! -f "${bundle_marker}" ]]; then
    run mkdir -p "${bundle_extract}"
    run unzip -n "${bundle_zip}" -d "${bundle_extract}"
    run touch "${bundle_marker}"
  fi
  for dataset in nuswide fasion_mnist_784 sift1m glove25 StarLightCurves; do
    selected "${dataset}" || continue
    target="${DATA_ROOT}/${dataset}/origin"
    if [[ -s "${target}/${dataset}_base.fvecs" && -s "${target}/${dataset}_query.fvecs" ]]; then
      log "SKIP prepared ${dataset}"
      continue
    fi
    if ((DRY_RUN)); then
      echo "copy ${dataset}/origin from ${bundle_extract} to ${target}"
      continue
    fi
    source_origin="$(find "${bundle_extract}" -type d -path "*/${dataset}/origin" -print -quit)"
    [[ -n "${source_origin}" ]] || { echo "${dataset}/origin not found in bundle" >&2; exit 1; }
    mkdir -p "${target}"
    cp -a "${source_origin}/." "${target}/"
  done
  if selected msong_holdout; then
    msong_target="${DATA_ROOT}/msong/origin"
    if [[ ! -s "${msong_target}/msong_base.fvecs" || ! -s "${msong_target}/msong_query.fvecs" ]]; then
      if ((DRY_RUN)); then
        echo "copy msong/origin from ${bundle_extract} to ${msong_target}"
      else
        source_origin="$(find "${bundle_extract}" -type d -path '*/msong/origin' -print -quit)"
        [[ -n "${source_origin}" ]] || { echo "msong/origin not found in bundle" >&2; exit 1; }
        mkdir -p "${msong_target}"
        cp -a "${source_origin}/." "${msong_target}/"
      fi
    fi
    run python3 "${ROOT}/scripts/prepare_disjoint_fvecs.py" \
      --source-base "${msong_target}/msong_base.fvecs" \
      --source-query "${msong_target}/msong_query.fvecs" \
      --output-base "${DATA_ROOT}/msong_holdout/origin/msong_holdout_base.fvecs" \
      --output-query "${DATA_ROOT}/msong_holdout/origin/msong_holdout_query.fvecs"
  fi
fi

if selected dbpedia1536m_holdout; then
  need curl
  if ((!DRY_RUN)); then
    python3 -c 'import numpy, pyarrow' >/dev/null 2>&1 || {
      echo "DBpedia preparation requires: python3 -m pip install numpy pyarrow" >&2
      exit 1
    }
  fi
  parquet_dir="${RAW_DIR}/dbpedia1536m/parquet"
  for shard in $(seq -w 0 25); do
    filename="train-000${shard#0}-of-00026.parquet"
    # seq -w differs between implementations; printf provides the canonical name.
    filename="$(printf 'train-%05d-of-00026.parquet' "$((10#${shard}))")"
    parquet_path="${parquet_dir}/${filename}"
    if [[ -s "${parquet_path}" ]] && ! parquet_valid "${parquet_path}"; then
      log "INCOMPLETE/CORRUPT ${parquet_path}; resume download"
      if [[ ! -e "${parquet_path}.partial" || "${parquet_path}" -nt "${parquet_path}.partial" ]]; then
        run mv -f "${parquet_path}" "${parquet_path}.partial"
      else
        run rm -f "${parquet_path}"
      fi
    fi
    download "${HF_ENDPOINT}/datasets/${DBPEDIA_REPO}/resolve/main/data/${filename}?download=true" \
      "${parquet_path}"
    if ((!DRY_RUN)) && ! parquet_valid "${parquet_path}"; then
      log "ERROR downloaded Parquet is incomplete: ${parquet_path}"
      mv -f "${parquet_path}" "${parquet_path}.partial"
      echo "Run the same command again to resume this shard." >&2
      exit 1
    fi
  done
  dbpedia_full="${RAW_DIR}/dbpedia1536m/prepared"
  if [[ ! -s "${dbpedia_full}/origin/dbpedia1536m_meta.json" ||
        ! -s "${dbpedia_full}/origin/dbpedia1536m_base.fvecs" ||
        ! -s "${dbpedia_full}/origin/dbpedia1536m_query.fvecs" ]]; then
    log "PREPARE DBpedia Parquet -> fvecs (this writes about 6.1 GB)"
    run python3 -u "${ROOT}/scripts/prepare_dbpedia1536m.py" \
      --parquet_dir "${parquet_dir}" --out_dir "${dbpedia_full}" \
      --dataset dbpedia1536m --nq 1000 --seed 42
  fi
  if [[ ! -s "${DATA_ROOT}/dbpedia1536m_holdout/origin/dbpedia1536m_holdout_meta.json" ||
        ! -s "${DATA_ROOT}/dbpedia1536m_holdout/origin/dbpedia1536m_holdout_base.fvecs" ||
        ! -s "${DATA_ROOT}/dbpedia1536m_holdout/origin/dbpedia1536m_holdout_query.fvecs" ]]; then
    log "PREPARE DBpedia disjoint holdout"
    run python3 -u "${ROOT}/scripts/split_dbpedia_holdout.py" \
      --src_base "${dbpedia_full}/origin/dbpedia1536m_base.fvecs" \
      --meta "${dbpedia_full}/origin/dbpedia1536m_meta.json" \
      --out_dir "${DATA_ROOT}/dbpedia1536m_holdout/origin" \
      --dataset dbpedia1536m_holdout
  fi
fi

if selected sift1b; then
  sift_raw="${DATA_ROOT}/sift1b/raw"
  # Direct HTTP downloads avoid huggingface_hub lock files and retain the same
  # .partial + resume behavior as the other large files.
  download "${HF_ENDPOINT}/datasets/${BIGANN_REPO}/resolve/main/bigann_base.bvecs.gz?download=true" \
    "${sift_raw}/bigann_base.bvecs.gz"
  download "${HF_ENDPOINT}/datasets/${BIGANN_REPO}/resolve/main/bigann_query.bvecs.gz?download=true" \
    "${sift_raw}/bigann_query.bvecs.gz"
  download "${HF_ENDPOINT}/datasets/${BIGANN_REPO}/resolve/main/bigann_gnd.tar.gz?download=true" \
    "${sift_raw}/bigann_gnd.tar.gz"
  need gzip; need tar
  decompress_gzip "${sift_raw}/bigann_base.bvecs.gz" "${sift_raw}/bigann_base.bvecs"
  decompress_gzip "${sift_raw}/bigann_query.bvecs.gz" "${sift_raw}/bigann_query.bvecs"
  if [[ ! -f "${sift_raw}/.gnd-extract-complete" ]]; then
    run tar -xzf "${sift_raw}/bigann_gnd.tar.gz" -C "${sift_raw}"
    run touch "${sift_raw}/.gnd-extract-complete"
  fi
  run mkdir -p "${DATA_ROOT}/sift1b/origin"
  run ln -sfn ../raw/bigann_base.bvecs \
    "${DATA_ROOT}/sift1b/origin/sift1b_base.bvecs"
  run ln -sfn ../raw/bigann_query.bvecs \
    "${DATA_ROOT}/sift1b/origin/sift1b_query.bvecs"
fi

if ((!DRY_RUN)); then
  failed=0
  for dataset in nuswide fasion_mnist_784 msong_holdout sift1m glove25 StarLightCurves dbpedia1536m_holdout; do
    selected "${dataset}" || continue
    for path in \
      "${DATA_ROOT}/${dataset}/origin/${dataset}_base.fvecs" \
      "${DATA_ROOT}/${dataset}/origin/${dataset}_query.fvecs"; do
      if [[ ! -s "${path}" ]]; then
        echo "INCOMPLETE prepared dataset file: ${path}" >&2
        failed=1
      fi
    done
  done
  if selected sift1b; then
    for path in \
      "${DATA_ROOT}/sift1b/raw/bigann_base.bvecs" \
      "${DATA_ROOT}/sift1b/raw/bigann_query.bvecs" \
      "${DATA_ROOT}/sift1b/raw/gnd/idx_1000M.ivecs" \
      "${DATA_ROOT}/sift1b/origin/sift1b_base.bvecs" \
      "${DATA_ROOT}/sift1b/origin/sift1b_query.bvecs"; do
      [[ -s "${path}" ]] || { echo "INCOMPLETE SIFT1B file: ${path}" >&2; failed=1; }
    done
  fi
  ((failed == 0)) || exit 1
  log "VALIDATION complete: every requested dataset is ready"
fi

log "COMPLETE data_root=${DATA_ROOT} raw_dir=${RAW_DIR}"
