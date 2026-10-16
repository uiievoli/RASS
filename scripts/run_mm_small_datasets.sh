#!/usr/bin/env bash
# Portable small-dataset MM performance experiment for a fresh server.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT=""
DBPEDIA_ROOT=""
LOG_DIR=""
CPU_SET="0-31"
OFFLINE_THREADS=8
BUILD_JOBS=8
SKIP_BUILD=0
BUILD_INDEX=0

usage() {
  cat <<'EOF'
Usage:
  scripts/run_mm_small_datasets.sh --data-root DIR --log-dir DIR [options]

Options:
  --data-root DIR       Parent of DATASET/origin and DATASET/index.
  --dbpedia-root DIR    Optional DBpedia parent or dbpedia1536m_holdout directory.
  --log-dir DIR         New or empty result directory.
  --cpus LIST           taskset CPU list (default 0-31).
  --offline-threads N   PCA/signature preparation threads (default 8).
  --build-jobs N        Parallel compiler jobs (default 8).
  --build-index         Build a Triangle index if a reusable P0 index is absent.
  --skip-build          Reuse build-mm-portable/{perf,stats} binaries.
  -h, --help            Show this help.

Datasets: fasion_mnist_784, msong_holdout, sift1m, glove25,
          StarLightCurves and dbpedia1536m_holdout. NUS-WIDE, HandOutlines,
          SpaceV1B and SIFT1B are deliberately excluded.
EOF
}

while (($#)); do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --dbpedia-root) DBPEDIA_ROOT="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --cpus) CPU_SET="$2"; shift 2 ;;
    --offline-threads) OFFLINE_THREADS="$2"; shift 2 ;;
    --build-jobs) BUILD_JOBS="$2"; shift 2 ;;
    --build-index) BUILD_INDEX=1; shift ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$DATA_ROOT" ]] || { echo "--data-root is required" >&2; exit 2; }
[[ -n "$LOG_DIR" ]] || { echo "--log-dir is required" >&2; exit 2; }
[[ "$OFFLINE_THREADS" =~ ^[1-9][0-9]*$ ]] || { echo "invalid --offline-threads" >&2; exit 2; }
[[ "$BUILD_JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "invalid --build-jobs" >&2; exit 2; }
command -v taskset >/dev/null || { echo "taskset is required" >&2; exit 1; }

args=(
  "$ROOT/scripts/run_mm_portable.py"
  --data-dir "$DATA_ROOT"
  --log-dir "$LOG_DIR"
  --preset active
  --cpus "$CPU_SET"
  --offline-threads "$OFFLINE_THREADS"
  --build-jobs "$BUILD_JOBS"
)
[[ -n "$DBPEDIA_ROOT" ]] && args+=(--dbpedia-root "$DBPEDIA_ROOT")
((BUILD_INDEX)) && args+=(--build-index)
((SKIP_BUILD)) && args+=(--skip-build)

exec python3 -u "${args[@]}"
