#!/usr/bin/env bash
# One entry point: download/prepare datasets, build binaries if needed, then run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-}"
RAW_DIR="${RAW_DIR:-}"
LOG_DIR="${LOG_DIR:-}"
CPU_SET="${CPU_SET:-0-31}"
THREADS="${THREADS:-}"
BUILD_BATCH_VECTORS="${BUILD_BATCH_VECTORS:-262144}"
COARSE_HNSW_M="${COARSE_HNSW_M:-32}"
COARSE_HNSW_EF_CONSTRUCTION="${COARSE_HNSW_EF_CONSTRUCTION:-200}"
COARSE_HNSW_EF_SEARCH="${COARSE_HNSW_EF_SEARCH:-128}"
PHASE="${PHASE:-all}"
ONLY_DATASETS="${ONLY_DATASETS:-}"
SKIP_BUILD=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  scripts/download_and_run_all_8_datasets.sh \
    --data-root DIR --log-dir DIR --cpus LIST [options]

Options:
  --data-root DIR       Dataset and index root (required).
  --raw-dir DIR         Download cache; default DATA_ROOT/.downloads.
  --log-dir DIR         Experiment log/result root (required).
  --cpus LIST           taskset CPU list, e.g. 0-31 or 0-15,32-47.
  --threads N           Defaults to the number of CPUs in the binding.
  --build-batch-vectors N
                        SIFT1B decoded build batch (default 262144).
  --coarse-hnsw-m N     HNSW coarse graph degree (default 32).
  --coarse-hnsw-ef-construction N
                        HNSW construction breadth (default 200).
  --coarse-hnsw-ef-search N
                        HNSW assignment breadth (default 128).
  --phase NAME          all (default), perf, or stats.
  --datasets LIST       Comma/space-separated subset of the eight datasets.
  --skip-build          Require existing build-perf/build-stats binaries.
  --dry-run             Print every download, build and experiment command.
  -h, --help            Show this help.

SIFT1B remains byte-coded on disk, then query.cpp decodes it into RAM and uses
the same float Index/IVF search path as the other datasets.
EOF
}

while (($#)); do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --raw-dir) RAW_DIR="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --cpus) CPU_SET="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --build-batch-vectors) BUILD_BATCH_VECTORS="$2"; shift 2 ;;
    --coarse-hnsw-m) COARSE_HNSW_M="$2"; shift 2 ;;
    --coarse-hnsw-ef-construction) COARSE_HNSW_EF_CONSTRUCTION="$2"; shift 2 ;;
    --coarse-hnsw-ef-search) COARSE_HNSW_EF_SEARCH="$2"; shift 2 ;;
    --phase) PHASE="$2"; shift 2 ;;
    --datasets) ONLY_DATASETS="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "${DATA_ROOT}" ]] || { echo "--data-root is required" >&2; exit 2; }
[[ -n "${LOG_DIR}" ]] || { echo "--log-dir is required" >&2; exit 2; }
[[ "${PHASE}" =~ ^(all|perf|stats)$ ]] || { echo "invalid --phase: ${PHASE}" >&2; exit 2; }
command -v taskset >/dev/null || { echo "taskset is required" >&2; exit 1; }
if [[ -z "${THREADS}" ]]; then
  THREADS="$(taskset -c "${CPU_SET}" nproc)" || {
    echo "Invalid or unavailable CPU set: ${CPU_SET}" >&2; exit 2;
  }
fi
RAW_DIR="${RAW_DIR:-${DATA_ROOT}/.downloads}"

run() {
  if ((DRY_RUN)); then printf '%q ' "$@"; printf '\n'; else "$@"; fi
}

if ((!DRY_RUN)); then
  mkdir -p "${LOG_DIR}"
  exec > >(tee -a "${LOG_DIR}/pipeline.log") 2>&1
fi

download_args=(
  "${ROOT}/scripts/download_all_8_datasets.sh"
  --data-root "${DATA_ROOT}" --raw-dir "${RAW_DIR}"
)
[[ -n "${ONLY_DATASETS}" ]] && download_args+=(--datasets "${ONLY_DATASETS}")
((DRY_RUN)) && download_args+=(--dry-run)

build_target() {
  local directory="$1" stats="$2"
  local binary="${ROOT}/${directory}/bin/query"
  if ((SKIP_BUILD)); then
    [[ -x "${binary}" ]] || { echo "Missing required binary: ${binary}" >&2; exit 1; }
    echo "[$(date -Is)] SKIP build by request: ${binary}"
    return
  fi
  # Always reconfigure and perform an incremental build.  Merely checking that
  # a binary exists can reuse objects from an older source tree or a build with
  # the wrong ENABLE_STATS setting, which previously caused missing counters
  # and undefined references on a copied checkout.
  run taskset -c "${CPU_SET}" cmake -S "${ROOT}" -B "${ROOT}/${directory}" \
    -DCMAKE_BUILD_TYPE=Release -DENABLE_STATS="${stats}" -DENABLE_LTO=OFF
  run taskset -c "${CPU_SET}" cmake --build "${ROOT}/${directory}" \
    --target query -j "${THREADS}"
}

if [[ "${PHASE}" == all || "${PHASE}" == perf ]]; then
  build_target build-perf OFF
fi
if [[ "${PHASE}" == all || "${PHASE}" == stats ]]; then
  build_target build-stats ON
fi

# Compile before starting multi-hundred-GB downloads so missing compiler,
# Eigen, MKL, or linker dependencies fail immediately.
echo "[$(date -Is)] DOWNLOAD/PREPARE"
if ((DRY_RUN)); then "${download_args[@]}"; else run "${download_args[@]}"; fi

echo "[$(date -Is)] EXPERIMENT"
experiment_args=(
  "${ROOT}/scripts/run_all_8_datasets.sh"
  --data-root "${DATA_ROOT}" --log-dir "${LOG_DIR}"
  --cpus "${CPU_SET}" --threads "${THREADS}"
  --build-batch-vectors "${BUILD_BATCH_VECTORS}"
  --coarse-hnsw-m "${COARSE_HNSW_M}"
  --coarse-hnsw-ef-construction "${COARSE_HNSW_EF_CONSTRUCTION}"
  --coarse-hnsw-ef-search "${COARSE_HNSW_EF_SEARCH}"
  --phase "${PHASE}"
)
[[ -n "${ONLY_DATASETS}" ]] && experiment_args+=(--datasets "${ONLY_DATASETS}")
((DRY_RUN)) && experiment_args+=(--dry-run)
if ((DRY_RUN)); then "${experiment_args[@]}"; else run "${experiment_args[@]}"; fi

if ((!DRY_RUN)) && [[ "${PHASE}" == all && -z "${ONLY_DATASETS}" ]]; then
  if python3 -c 'import matplotlib' >/dev/null 2>&1; then
    echo "[$(date -Is)] PLOT recall > 0.9"
    MPLCONFIGDIR="${LOG_DIR}/.matplotlib" python3 \
      "${ROOT}/scripts/plot_all10_recall_gt09.py" --log-root "${LOG_DIR}"
  else
    echo "[$(date -Is)] WARNING matplotlib is unavailable; CSV results are complete but plots were skipped" >&2
  fi
fi
echo "[$(date -Is)] COMPLETE data=${DATA_ROOT} logs=${LOG_DIR}"
