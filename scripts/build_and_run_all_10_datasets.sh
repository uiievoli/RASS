#!/usr/bin/env bash
# Compile, prepare/reuse indexes, run all ten datasets, then plot recall > 0.9.
# Dataset inputs must already be downloaded; no duplicate large-vector files.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-}"
LOG_DIR="${LOG_DIR:-}"
CPU_SET="${CPU_SET:-0-31}"
THREADS="${THREADS:-}"
PHASE="${PHASE:-all}"
ONLY_DATASETS="${ONLY_DATASETS:-}"
PERF_BIN="${PERF_BIN:-$ROOT/build-perf/bin/query}"
STATS_BIN="${STATS_BIN:-$ROOT/build-stats/bin/query}"
BUILD_JOBS="${BUILD_JOBS:-16}"
SKIP_BUILD=0
SKIP_PLOTS=0
DRY_RUN=0
ARGS=()
while (($#)); do
  case "$1" in
    --skip-build) SKIP_BUILD=1; shift ;;
    --skip-plots) SKIP_PLOTS=1; shift ;;
    --build-jobs) BUILD_JOBS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; ARGS+=("$1"); shift ;;
    --data-root) DATA_ROOT="$2"; ARGS+=("$1" "$2"); shift 2 ;;
    --log-dir) LOG_DIR="$2"; ARGS+=("$1" "$2"); shift 2 ;;
    --cpus) CPU_SET="$2"; ARGS+=("$1" "$2"); shift 2 ;;
    --threads) THREADS="$2"; ARGS+=("$1" "$2"); shift 2 ;;
    --phase) PHASE="$2"; ARGS+=("$1" "$2"); shift 2 ;;
    --datasets) ONLY_DATASETS="$2"; ARGS+=("$1" "$2"); shift 2 ;;
    --perf-bin) PERF_BIN="$2"; ARGS+=("$1" "$2"); shift 2 ;;
    --stats-bin) STATS_BIN="$2"; ARGS+=("$1" "$2"); shift 2 ;;
    -h|--help)
      cat <<'HELP'
Usage: scripts/build_and_run_all_10_datasets.sh --data-root DIR --log-dir DIR [options]

Runs all ten datasets by default. Existing binaries are incrementally rebuilt;
indexes and verified results are reused. The three recall > 0.9 plots are saved
after an all-phase run. Inputs must already exist; no dataset download occurs.

Additional options:
  --skip-build     Use existing binaries (their timing/stats protocol is checked).
  --skip-plots     Collect complete CSVs without requiring matplotlib.
  --build-jobs N   Parallel compilation jobs, default 16.

All experiment options below are forwarded to the runner:
HELP
      "$ROOT/scripts/run_all_10_datasets.sh" --help
      exit 0 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
[[ "$BUILD_JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "invalid --build-jobs" >&2; exit 2; }
# Validate all runner arguments before compiling or starting a long experiment.
"$ROOT/scripts/run_all_10_datasets.sh" "${ARGS[@]}" --dry-run >/dev/null
[[ -n "$DATA_ROOT" && -n "$LOG_DIR" ]] || { echo "data/log roots are required" >&2; exit 2; }
if [[ -z "$THREADS" ]]; then THREADS="$(taskset -c "$CPU_SET" nproc)"; fi
run() { if ((DRY_RUN)); then printf '%q ' "$@"; printf '\n'; else "$@"; fi; }
if ((!DRY_RUN)); then
  if ((!SKIP_PLOTS)) && [[ "$PHASE" == all ]]; then
    python3 -c 'import matplotlib' || {
      echo "Plotting requires matplotlib; install it or pass --skip-plots." >&2; exit 1;
    }
  fi
  mkdir -p "$LOG_DIR"
  exec > >(tee -a "$LOG_DIR/pipeline.log") 2>&1
fi
build() {
  local binary="$1" stats="$2" directory
  ((SKIP_BUILD)) && return
  [[ "$binary" == */bin/query ]] || {
    echo "To compile, --perf-bin/--stats-bin must end in /bin/query; otherwise use --skip-build." >&2; exit 2;
  }
  directory="$(dirname "$(dirname "$binary")")"
  echo "[$(date -Is)] BUILD stats=$stats directory=$directory"
  run taskset -c "$CPU_SET" cmake -S "$ROOT" -B "$directory" \
    -DCMAKE_BUILD_TYPE=Release -DENABLE_STATS="$stats" -DENABLE_LTO=OFF
  run taskset -c "$CPU_SET" cmake --build "$directory" --target query -j "$BUILD_JOBS"
}
if [[ "$PHASE" == all || "$PHASE" == perf ]]; then build "$PERF_BIN" OFF; fi
if [[ "$PHASE" == all || "$PHASE" == stats ]]; then build "$STATS_BIN" ON; fi
export PERF_BIN STATS_BIN
if ((DRY_RUN)); then
  "$ROOT/scripts/run_all_10_datasets.sh" "${ARGS[@]}"
else
  run "$ROOT/scripts/run_all_10_datasets.sh" "${ARGS[@]}"
fi
if ((!SKIP_PLOTS)) && [[ "$PHASE" == all ]]; then
  plot=(python3 "$ROOT/scripts/plot_all10_recall_gt09.py" --log-root "$LOG_DIR")
  [[ -z "$ONLY_DATASETS" ]] || plot+=(--datasets "${ONLY_DATASETS// /,}")
  run env MPLCONFIGDIR="$LOG_DIR/.matplotlib" XDG_CACHE_HOME="$LOG_DIR/.cache" MPLBACKEND=Agg "${plot[@]}"
fi
echo "[$(date -Is)] COMPLETE logs=$LOG_DIR"
