#!/usr/bin/env bash
# Record every dynamic query-list pivot choice and plot per-query distributions.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/mnt/nvme/wxy/benchmarks}"
DBPEDIA_ROOT="${DBPEDIA_ROOT:-/mnt/nvme/wxy}"
STATIC_ROOT="${STATIC_ROOT:-${ROOT}/logs/fine_pivot_screen_8_20261004}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/query_pivot_distribution_8_20261007}"
CPU_SET="${CPU_SET:-0-31}"
THREADS="${THREADS:-32}"
PIVOT_SEED="${PIVOT_SEED:-20261005}"
BLOCK_SIZE="${BLOCK_SIZE:-2}"
STATS_BIN="${STATS_BIN:-${ROOT}/build-stats/bin/query}"
ONLY_DATASETS="${ONLY_DATASETS:-}"
DRY_RUN=0
VERBOSE=0
SKIP_PLOT=0
BUILD_MISSING=0

usage() {
  cat <<'EOF'
Usage: scripts/run_query_pivot_distribution_8.sh [options]

Options:
  --data-root DIR       Root of the first six datasets.
  --dbpedia-root DIR    Root containing dbpedia1536m_holdout.
  --static-root DIR     Fine static-P result root (for oracle markers).
  --log-dir DIR         Raw visits, query summaries and figure output root.
  --cpus LIST           taskset CPU list (default 0-31).
  --threads N           OpenMP threads (default 32).
  --datasets LIST       Comma/space-separated subset; reruns are resumable.
  --pivot-seed N        Must match existing Pmax indexes (default 20261005).
  --block-size N        Dynamic projection block size (default 2).
  --stats-bin PATH      ENABLE_STATS=ON query binary.
  --build-missing       Build a missing rich PCA index. Reuses the matching
                        Triangle/IVF index when present; otherwise builds fully.
  --skip-plot           Keep raw outputs without running the final analyzer.
  --verbose             Pass --verbose to query.
  --dry-run             Validate arguments and print commands.
  -h, --help            Show this help.

This is a stats-only pass. It reuses the existing rich PCA indexes and writes
one list_visits row for every query x probed-list pair. It does not rebuild an
index and does not replace perf-build latency/QPS measurements.
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
    --datasets) ONLY_DATASETS="$2"; shift 2 ;;
    --pivot-seed) PIVOT_SEED="$2"; shift 2 ;;
    --block-size) BLOCK_SIZE="$2"; shift 2 ;;
    --stats-bin) STATS_BIN="$2"; shift 2 ;;
    --build-missing) BUILD_MISSING=1; shift ;;
    --skip-plot) SKIP_PLOT=1; shift ;;
    --verbose) VERBOSE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for value in "$THREADS" "$BLOCK_SIZE"; do
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

triangle_index_path() {
  local root="$1" dataset="$2"
  printf '%s/%s/index/v10_nlist_%s_metric_l2_opt_1_subk_15_subNprobeRatio_1_mp_per_list_pca_P0_seed%s.index\n' \
    "$root" "$dataset" "${NLIST[$dataset]}" "$PIVOT_SEED"
}

baseline_index_path() {
  local root="$1" dataset="$2"
  printf '%s/%s/index/v10_nlist_%s_metric_l2_opt_0_subk_15_subNprobeRatio_1_mp_global_affine_fps_P0_seed0.index\n' \
    "$root" "$dataset" "${NLIST[$dataset]}"
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

outputs_complete() {
  local dataset="$1"
  [[ -s "$LOG_DIR/stats/$dataset.csv" &&
     -s "$LOG_DIR/list_visits/$dataset.csv" &&
     -f "$LOG_DIR/complete/$dataset.done" ]]
}

validate_visits() {
  local path="$1" dataset="$2"
  python3 - "$path" "$dataset" "${NPROBE[$dataset]}" "${PMAX[$dataset]}" "${SCOPE[$dataset]}" <<'PY'
import csv
import sys

path, dataset, nprobe, pmax, scope = sys.argv[1:]
count = 0
queries = set()
with open(path, newline="") as stream:
    for row in csv.DictReader(stream):
        if (
            row["dataset"] != dataset
            or int(float(row["nprobe"])) != int(nprobe)
            or int(float(row["pivot_count"])) != int(pmax)
            or row["multipivot_scope"] != scope
        ):
            raise SystemExit(1)
        count += 1
        queries.add(int(float(row["query_id"])))
if count == 0 or count != len(queries) * int(nprobe):
    raise SystemExit(1)
PY
}

build_rich_index() {
  local dataset="$1" root="$2" rich="$3" triangle baseline source logfile manifest
  local -a command source_args=()
  triangle="$(triangle_index_path "$root" "$dataset")"
  baseline="$(baseline_index_path "$root" "$dataset")"
  if [[ -s "$triangle" ]]; then
    source="$triangle"
    source_args=(--from_index "$source")
  elif [[ -s "$baseline" ]]; then
    source="$baseline"
    source_args=(--from_index "$source")
  else
    source="none (full IVF build)"
  fi
  logfile="$LOG_DIR/index_build/$dataset.log"
  manifest="$LOG_DIR/manifests/${dataset}_P${PMAX[$dataset]}_build.csv"
  command=(
    "$STATS_BIN" --benchmarks_path "$root" --dataset "$dataset"
    --input_format fvecs --output_format bin --metric l2 --k 1
    --nlist "${NLIST[$dataset]}" --nprobes "${NPROBE[$dataset]}"
    --cache --signature_precision float32 --loop 1
    --opt_levels OPT_TRIANGLE --multipivot_modes projection
    --multipivot_scope "${SCOPE[$dataset]}" --multipivot_method pca
    --pivot_counts "${PMAX[$dataset]}" --projection_block_size 0
    --pivot_seed "$PIVOT_SEED" --pivot_manifest "$manifest"
    --train_only "${source_args[@]}"
  )
  ((VERBOSE)) && command+=(--verbose)
  log "BUILD $dataset rich_index=$rich source=$source"
  if ((DRY_RUN)); then
    run_bound "${command[@]}"
    return
  fi
  mkdir -p "$LOG_DIR/index_build" "$LOG_DIR/manifests"
  if run_bound "${command[@]}" >"$logfile.partial" 2>&1; then
    [[ -s "$rich" ]] || {
      log "FAILED build completed without expected index: $rich"; return 1;
    }
    mv "$logfile.partial" "$logfile"
    log "BUILT $dataset rich_index=$rich"
  else
    local rc=$?
    log "FAILED index build for $dataset rc=$rc; inspect $logfile.partial"
    return "$rc"
  fi
}

run_dataset() {
  local dataset="$1" root rich stats partial_stats visits partial_visits logfile manifest
  local -a command
  if outputs_complete "$dataset"; then
    log "SKIP $dataset"
    return
  fi
  root="$(dataset_root "$dataset")"
  rich="$(index_path "$root" "$dataset")"
  stats="$LOG_DIR/stats/$dataset.csv"
  partial_stats="$stats.partial"
  visits="$LOG_DIR/list_visits/$dataset.csv"
  partial_visits="$visits.partial"
  logfile="$LOG_DIR/logs/$dataset.log"
  manifest="$LOG_DIR/manifests/${dataset}_query_pivot_distribution.csv"

  if ((!DRY_RUN)); then
    [[ -x "$STATS_BIN" ]] || { echo "Missing stats binary: $STATS_BIN" >&2; exit 1; }
    mkdir -p "$LOG_DIR"/{stats,list_visits,logs,manifests,complete}
  fi
  if [[ ! -s "$rich" ]]; then
    if ((BUILD_MISSING)); then
      build_rich_index "$dataset" "$root" "$rich"
    elif ((DRY_RUN)); then
      log "MISSING rich PCA index (add --build-missing to build): $rich"
    else
      echo "Missing rich PCA index: $rich" >&2
      echo "Rerun with --build-missing to upgrade a matching Triangle/IVF index." >&2
      exit 1
    fi
  fi
  command=(
    "$STATS_BIN" --benchmarks_path "$root" --dataset "$dataset"
    --input_format fvecs --output_format bin --metric l2 --k 1
    --nlist "${NLIST[$dataset]}" --nprobes "${NPROBE[$dataset]}"
    --cache --signature_precision float32 --loop 1 --csv "$partial_stats"
    --opt_levels OPT_TRIANGLE --multipivot_modes projection
    --multipivot_scope "${SCOPE[$dataset]}" --multipivot_method pca
    --pivot_counts "${PMAX[$dataset]}" --projection_block_size "$BLOCK_SIZE"
    --projection_dynamic --pivot_seed "$PIVOT_SEED"
    --pivot_manifest "$manifest" --load_index "$rich"
    --dump_list_stats "$partial_visits"
  )
  ((VERBOSE)) && command+=(--verbose)
  log "START $dataset nprobe=${NPROBE[$dataset]} scope=${SCOPE[$dataset]} Pmax=${PMAX[$dataset]}"
  if ((DRY_RUN)); then
    run_bound "${command[@]}"
    return
  fi
  rm -f "$partial_stats" "$partial_visits" "$logfile.partial" "$LOG_DIR/complete/$dataset.done"
  if run_bound "${command[@]}" >"$logfile.partial" 2>&1; then
    validate_visits "$partial_visits" "$dataset" || {
      log "FAILED invalid query-list dump: $partial_visits"; return 1;
    }
    [[ -s "$partial_stats" ]] || {
      log "FAILED missing stats CSV: $partial_stats"; return 1;
    }
    mv "$partial_stats" "$stats"
    mv "$partial_visits" "$visits"
    mv "$logfile.partial" "$logfile"
    touch "$LOG_DIR/complete/$dataset.done"
    log "DONE $dataset rows=$(($(wc -l < "$visits") - 1))"
  else
    local rc=$?
    log "FAILED $dataset rc=$rc; inspect $logfile.partial"
    return "$rc"
  fi
}

if ((!DRY_RUN)); then
  [[ -f "$STATIC_ROOT/fine_screen_summary.csv" ]] || {
    echo "Missing static summary: $STATIC_ROOT/fine_screen_summary.csv" >&2; exit 1;
  }
  mkdir -p "$LOG_DIR"
fi

log "CONFIG data=$DATA_ROOT dbpedia=$DBPEDIA_ROOT static=$STATIC_ROOT output=$LOG_DIR cpus=$CPU_SET threads=$THREADS seed=$PIVOT_SEED block=$BLOCK_SIZE"
for dataset in "${DATASETS[@]}"; do
  selected "$dataset" || continue
  run_dataset "$dataset"
done

if ((!DRY_RUN && !SKIP_PLOT)); then
  complete=1
  for dataset in "${DATASETS[@]}"; do
    outputs_complete "$dataset" || complete=0
  done
  if ((complete)); then
    python3 "$ROOT/scripts/plot_query_pivot_distribution_8.py" \
      --input-root "$LOG_DIR" \
      --static-summary "$STATIC_ROOT/fine_screen_summary.csv"
  else
    log "PLOT deferred until all eight query-list dumps are complete"
  fi
fi
log "COMPLETE results=$LOG_DIR"
