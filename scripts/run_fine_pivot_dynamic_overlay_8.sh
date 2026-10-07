#!/usr/bin/env bash
# Dynamic-P supplement aligned with logs/fine_pivot_screen_8_20261004.
# Reuses each static sweep's Pmax rich PCA index; it never rebuilds an index.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/mnt/nvme/wxy/benchmarks}"
DBPEDIA_ROOT="${DBPEDIA_ROOT:-/mnt/nvme/wxy}"
STATIC_ROOT="${STATIC_ROOT:-${ROOT}/logs/fine_pivot_screen_8_20261004}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/fine_pivot_dynamic_overlay_8_20261007}"
CPU_SET="${CPU_SET:-0-31}"
THREADS="${THREADS:-32}"
PERF_REPEATS="${PERF_REPEATS:-3}"
PIVOT_SEED="${PIVOT_SEED:-20261005}"
BLOCK_SIZE="${BLOCK_SIZE:-2}"
PERF_BIN="${PERF_BIN:-${ROOT}/build-perf/bin/query}"
STATS_BIN="${STATS_BIN:-${ROOT}/build-stats/bin/query}"
ONLY_DATASETS="${ONLY_DATASETS:-}"
PHASE="${PHASE:-all}"
DRY_RUN=0
VERBOSE=0

usage() {
  cat <<'EOF'
Usage: scripts/run_fine_pivot_dynamic_overlay_8.sh [options]

Options:
  --data-root DIR       Root of the first six datasets.
  --dbpedia-root DIR    Root containing dbpedia1536m_holdout.
  --static-root DIR     Existing fine static-P experiment directory.
  --log-dir DIR         Dynamic supplement output directory.
  --cpus LIST           taskset CPU list (default 0-31).
  --threads N           OpenMP threads (default 32).
  --perf-repeats N      Independent perf runs (default 3).
  --phase NAME          all (default), perf, or stats.
  --datasets LIST       Comma/space-separated subset of the seven datasets.
  --pivot-seed N        Must match the rich indexes (default 20261005).
  --block-size N        Dynamic projection block size (default 2).
  --perf-bin PATH       ENABLE_STATS=OFF query binary.
  --stats-bin PATH      ENABLE_STATS=ON query binary.
  --verbose             Pass --verbose to query.
  --dry-run             Validate configuration and print commands.
  -h, --help            Show this help.

The target nprobe, PCA scope and Pmax exactly match fine_screen_summary.csv:
  nuswide               per_list nprobe=3   Pmax=32
  fasion_mnist_784      per_list nprobe=7   Pmax=96
  msong_holdout         per_list nprobe=16  Pmax=96
  sift1m                per_list nprobe=34  Pmax=40
  glove25               global   nprobe=50  Pmax=26
  StarLightCurves       global   nprobe=4   Pmax=104
  dbpedia1536m_holdout  per_list nprobe=160 Pmax=256

On a complete all-phase run, dynamic_overlay_summary.csv contains the exact
query-list-weighted mean pivot count, quartiles, zero-P share, pruning and perf.
EOF
}

while (($#)); do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --dbpedia-root) DBPEDIA_ROOT="$2"; shift 2 ;;
    --static-root) STATIC_ROOT="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --cpus) CPU_SET="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --perf-repeats) PERF_REPEATS="$2"; shift 2 ;;
    --phase) PHASE="$2"; shift 2 ;;
    --datasets) ONLY_DATASETS="$2"; shift 2 ;;
    --pivot-seed) PIVOT_SEED="$2"; shift 2 ;;
    --block-size) BLOCK_SIZE="$2"; shift 2 ;;
    --perf-bin) PERF_BIN="$2"; shift 2 ;;
    --stats-bin) STATS_BIN="$2"; shift 2 ;;
    --verbose) VERBOSE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$PHASE" =~ ^(all|perf|stats)$ ]] || {
  echo "--phase must be all, perf, or stats" >&2; exit 2;
}
for value in "$THREADS" "$PERF_REPEATS" "$BLOCK_SIZE"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    echo "Expected a positive integer, got: $value" >&2; exit 2;
  }
done
[[ "$PIVOT_SEED" =~ ^[0-9]+$ ]] || {
  echo "--pivot-seed must be a non-negative integer" >&2; exit 2;
}
command -v taskset >/dev/null || { echo "taskset is required" >&2; exit 1; }

DATASETS=(nuswide fasion_mnist_784 msong_holdout sift1m glove25 StarLightCurves dbpedia1536m_holdout)
declare -A NLIST=(
  [nuswide]=512 [fasion_mnist_784]=256 [msong_holdout]=1000 [sift1m]=1000
  [glove25]=1024 [StarLightCurves]=128 [dbpedia1536m_holdout]=1000
)
declare -A NPROBE=(
  [nuswide]=3 [fasion_mnist_784]=7 [msong_holdout]=16 [sift1m]=34
  [glove25]=50 [StarLightCurves]=4 [dbpedia1536m_holdout]=160
)
declare -A PMAX=(
  [nuswide]=32 [fasion_mnist_784]=96 [msong_holdout]=96 [sift1m]=40
  [glove25]=26 [StarLightCurves]=104 [dbpedia1536m_holdout]=256
)
declare -A SCOPE=(
  [nuswide]=per_list [fasion_mnist_784]=per_list [msong_holdout]=per_list [sift1m]=per_list
  [glove25]=global [StarLightCurves]=global [dbpedia1536m_holdout]=per_list
)
declare -A PERF_LOOPS=(
  [nuswide]=50 [fasion_mnist_784]=10 [msong_holdout]=5 [sift1m]=2
  [glove25]=10 [StarLightCurves]=50 [dbpedia1536m_holdout]=1
)

selected() {
  [[ -z "$ONLY_DATASETS" ]] && return 0
  local values=" ${ONLY_DATASETS//,/ } "
  [[ "$values" == *" $1 "* ]]
}

dataset_root() {
  [[ "$1" == dbpedia1536m_holdout ]] && echo "$DBPEDIA_ROOT" || echo "$DATA_ROOT"
}

index_path() {
  local root="$1" dataset="$2"
  printf '%s/%s/index/v10_nlist_%s_metric_l2_opt_1_subk_15_subNprobeRatio_1_mp_%s_pca_P%s_seed%s.index\n' \
    "$root" "$dataset" "${NLIST[$dataset]}" "${SCOPE[$dataset]}" \
    "${PMAX[$dataset]}" "$PIVOT_SEED"
}

log() {
  local message="[$(date -Is)] $*"
  echo "$message"
  ((DRY_RUN)) || echo "$message" >>"$LOG_DIR/run.log"
}

run_bound() {
  if ((DRY_RUN)); then
    printf '%q ' env -u TRIBASE_TRACE -u EDGE_DEVICE_ENABLED \
      OMP_NUM_THREADS="$THREADS" OMP_PROC_BIND=close OMP_PLACES=cores \
      taskset -c "$CPU_SET" "$@"
    printf '\n'
  else
    env -u TRIBASE_TRACE -u EDGE_DEVICE_ENABLED \
      OMP_NUM_THREADS="$THREADS" OMP_PROC_BIND=close OMP_PLACES=cores \
      taskset -c "$CPU_SET" "$@"
  fi
}

csv_complete() {
  local path="$1" dataset="$2"
  [[ -s "$path" ]] || return 1
  python3 - "$path" "$dataset" "${NPROBE[$dataset]}" "${PMAX[$dataset]}" "${SCOPE[$dataset]}" <<'PY'
import csv
import sys

path, dataset, nprobe, pmax, scope = sys.argv[1:]
try:
    with open(path, newline="") as stream:
        rows = list(csv.DictReader(stream))
except (OSError, csv.Error):
    raise SystemExit(1)
matches = [
    row for row in rows
    if row.get("dataset") == dataset
    and int(float(row["nprobe"])) == int(nprobe)
    and int(float(row["pivot_count"])) == int(pmax)
    and row.get("multipivot_scope") == scope
    and row.get("multipivot_mode") == "projection"
]
raise SystemExit(0 if len(matches) == 1 else 1)
PY
}

validate_static_summary() {
  python3 - "$STATIC_ROOT/fine_screen_summary.csv" <<'PY'
import csv
import sys

expected = {
    "nuswide": (3, "per_list"),
    "fasion_mnist_784": (7, "per_list"),
    "msong_holdout": (16, "per_list"),
    "sift1m": (34, "per_list"),
    "glove25": (50, "global"),
    "StarLightCurves": (4, "global"),
    "dbpedia1536m_holdout": (160, "per_list"),
}
with open(sys.argv[1], newline="") as stream:
    rows = {row["dataset"]: row for row in csv.DictReader(stream)}
for dataset, (nprobe, scope) in expected.items():
    row = rows.get(dataset)
    if row is None or int(float(row["nprobe"])) != nprobe or row["scope"] != scope:
        raise SystemExit(
            f"static summary mismatch for {dataset}: expected nprobe={nprobe}, scope={scope}"
        )
PY
}

run_case() {
  local phase="$1" dataset="$2" repeat="$3"
  local binary loops root rich output csv partial logfile manifest
  local -a command
  if [[ "$phase" == perf ]]; then
    binary="$PERF_BIN"; loops="${PERF_LOOPS[$dataset]}"
  else
    binary="$STATS_BIN"; loops=1
  fi
  root="$(dataset_root "$dataset")"
  rich="$(index_path "$root" "$dataset")"
  output="$LOG_DIR/$phase/$dataset"
  csv="$output/dynamic_overlay_rep${repeat}.csv"
  partial="$csv.partial"
  logfile="$output/dynamic_overlay_rep${repeat}.log"
  manifest="$LOG_DIR/manifests/${dataset}_dynamic_overlay_${phase}_rep${repeat}.csv"

  if csv_complete "$csv" "$dataset"; then
    log "SKIP $phase/$dataset/rep$repeat"
    return
  fi
  if ((!DRY_RUN)); then
    [[ -x "$binary" ]] || { echo "Missing binary: $binary" >&2; exit 1; }
    [[ -s "$rich" ]] || { echo "Missing rich PCA index: $rich" >&2; exit 1; }
    mkdir -p "$output" "$LOG_DIR/manifests"
  fi

  command=(
    "$binary" --benchmarks_path "$root" --dataset "$dataset"
    --input_format fvecs --output_format bin --metric l2 --k 1
    --nlist "${NLIST[$dataset]}" --nprobes "${NPROBE[$dataset]}"
    --cache --signature_precision float32 --loop "$loops" --csv "$partial"
    --opt_levels OPT_TRIANGLE --multipivot_modes projection
    --multipivot_scope "${SCOPE[$dataset]}" --multipivot_method pca
    --pivot_counts "${PMAX[$dataset]}" --projection_block_size "$BLOCK_SIZE"
    --projection_dynamic --pivot_seed "$PIVOT_SEED"
    --pivot_manifest "$manifest" --load_index "$rich"
  )
  ((VERBOSE)) && command+=(--verbose)
  log "START $phase/$dataset/rep$repeat nprobe=${NPROBE[$dataset]} scope=${SCOPE[$dataset]} Pmax=${PMAX[$dataset]} loops=$loops"
  if ((DRY_RUN)); then
    run_bound "${command[@]}"
    return
  fi
  rm -f "$partial" "$logfile.partial"
  if run_bound "${command[@]}" >"$logfile.partial" 2>&1; then
    csv_complete "$partial" "$dataset" || {
      log "FAILED incomplete CSV: $partial"; return 1;
    }
    if [[ "$phase" == stats ]] && ! grep -q '^active_pivots:.*=' "$logfile.partial"; then
      log "FAILED stats log has no active-pivot histogram: $logfile.partial"
      return 1
    fi
    mv "$partial" "$csv"
    mv "$logfile.partial" "$logfile"
    log "DONE $phase/$dataset/rep$repeat"
  else
    local rc=$?
    log "FAILED $phase/$dataset/rep$repeat rc=$rc; inspect $logfile.partial"
    return "$rc"
  fi
}

if ((!DRY_RUN)); then
  [[ -f "$STATIC_ROOT/fine_screen_summary.csv" ]] || {
    echo "Missing $STATIC_ROOT/fine_screen_summary.csv" >&2; exit 1;
  }
  validate_static_summary
  mkdir -p "$LOG_DIR"
fi

log "CONFIG data=$DATA_ROOT dbpedia=$DBPEDIA_ROOT static=$STATIC_ROOT output=$LOG_DIR cpus=$CPU_SET threads=$THREADS phase=$PHASE repeats=$PERF_REPEATS seed=$PIVOT_SEED block=$BLOCK_SIZE"
for dataset in "${DATASETS[@]}"; do
  selected "$dataset" || continue
  if [[ "$PHASE" == all || "$PHASE" == perf ]]; then
    for ((repeat=1; repeat<=PERF_REPEATS; ++repeat)); do
      run_case perf "$dataset" "$repeat"
    done
  fi
  if [[ "$PHASE" == all || "$PHASE" == stats ]]; then
    run_case stats "$dataset" 1
  fi
done

if ((!DRY_RUN)); then
  complete=1
  for dataset in "${DATASETS[@]}"; do
    [[ -s "$LOG_DIR/stats/$dataset/dynamic_overlay_rep1.csv" &&
       -s "$LOG_DIR/stats/$dataset/dynamic_overlay_rep1.log" ]] || complete=0
    for ((repeat=1; repeat<=PERF_REPEATS; ++repeat)); do
      [[ -s "$LOG_DIR/perf/$dataset/dynamic_overlay_rep${repeat}.csv" ]] || complete=0
    done
  done
  if ((complete)); then
    python3 "$ROOT/scripts/summarize_fine_pivot_dynamic_overlay_8.py" \
      --input-root "$LOG_DIR" --static-root "$STATIC_ROOT"
  else
    log "SUMMARY deferred until all seven datasets have perf and stats outputs"
  fi
fi
log "COMPLETE results=$LOG_DIR"
