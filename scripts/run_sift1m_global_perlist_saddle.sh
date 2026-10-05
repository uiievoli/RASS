#!/usr/bin/env bash
# Fine static-prefix sweep for the SIFT1M global/per-list latency saddle.
#
# Both scopes share exactly the same IVF and Triangle indexes, nprobe, pivot
# prefixes, CPU binding, and PCA seed. Perf and stats are kept in separate
# binaries so timing is not polluted by detailed instrumentation.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-}"
LOG_DIR="${LOG_DIR:-}"
CPU_SET="${CPU_SET:-0-31}"
THREADS="${THREADS:-}"
PHASE="${PHASE:-all}"
PERF_REPEATS="${PERF_REPEATS:-3}"
PERF_LOOPS="${PERF_LOOPS:-5}"
NPROBE="${NPROBE:-30}"
NLIST="${NLIST:-1000}"
PIVOT_SEED="${PIVOT_SEED:-20261006}"
PIVOT_COUNTS="${PIVOT_COUNTS:-4 8 12 16 18 20 22 24 26 28 30 32 36 40}"
PERF_BIN="${PERF_BIN:-${ROOT}/build-perf/bin/query}"
STATS_BIN="${STATS_BIN:-${ROOT}/build-stats/bin/query}"
DRY_RUN=0
VERBOSE=0
SKIP_PLOT=0

usage() {
  cat <<'EOF'
Usage:
  scripts/run_sift1m_global_perlist_saddle.sh \
    --data-root DIR --log-dir DIR [options]

Required:
  --data-root DIR       Root containing sift1m/origin/*.fvecs.
  --log-dir DIR         Dedicated output root for this experiment.

Options:
  --cpus LIST           taskset CPU list (default 0-31).
  --threads N           OpenMP threads; defaults to bound CPU count.
  --phase NAME          all (default), perf, or stats.
  --perf-repeats N      Independent perf processes (default 3).
  --perf-loops N        Timed query loops per perf process (default 5).
  --nprobe N            Fixed IVF probes (default 30).
  --nlist N             IVF cluster count (default 1000).
  --pivot-counts LIST   Quoted/CSV prefix list. Default:
                        "4 8 12 16 18 20 22 24 26 28 30 32 36 40".
  --pivot-seed N        PCA/index namespace (default 20261006).
  --perf-bin PATH       ENABLE_STATS=OFF query binary.
  --stats-bin PATH      ENABLE_STATS=ON query binary.
  --skip-plot           Do not generate the two-panel cost figure.
  --verbose             Pass --verbose to query.
  --dry-run             Validate and print commands without running.
  -h, --help            Show this help.

P includes the centroid, so a P-prefix uses P-1 PCA coordinates. Completed
CSVs and compatible indexes are reused, making reruns resumable.
EOF
}

while (($#)); do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --cpus) CPU_SET="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --phase) PHASE="$2"; shift 2 ;;
    --perf-repeats) PERF_REPEATS="$2"; shift 2 ;;
    --perf-loops) PERF_LOOPS="$2"; shift 2 ;;
    --nprobe) NPROBE="$2"; shift 2 ;;
    --nlist) NLIST="$2"; shift 2 ;;
    --pivot-counts) PIVOT_COUNTS="$2"; shift 2 ;;
    --pivot-seed) PIVOT_SEED="$2"; shift 2 ;;
    --perf-bin) PERF_BIN="$2"; shift 2 ;;
    --stats-bin) STATS_BIN="$2"; shift 2 ;;
    --skip-plot) SKIP_PLOT=1; shift ;;
    --verbose) VERBOSE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$DATA_ROOT" ]] || { echo "--data-root is required" >&2; exit 2; }
[[ -n "$LOG_DIR" ]] || { echo "--log-dir is required" >&2; exit 2; }
[[ "$PHASE" =~ ^(all|perf|stats)$ ]] || {
  echo "--phase must be all, perf, or stats" >&2; exit 2;
}
for value in "$PERF_REPEATS" "$PERF_LOOPS" "$NPROBE" "$NLIST"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    echo "Expected a positive integer, got: $value" >&2; exit 2;
  }
done
[[ "$PIVOT_SEED" =~ ^[0-9]+$ ]] || {
  echo "--pivot-seed must be a non-negative integer" >&2; exit 2;
}
command -v taskset >/dev/null || { echo "taskset is required" >&2; exit 1; }
if [[ -z "$THREADS" ]]; then
  THREADS="$(taskset -c "$CPU_SET" nproc)" || {
    echo "Invalid or unavailable CPU set: $CPU_SET" >&2; exit 2;
  }
fi
[[ "$THREADS" =~ ^[1-9][0-9]*$ ]] || {
  echo "--threads must be a positive integer" >&2; exit 2;
}

DATA_ROOT="${DATA_ROOT%/}"
LOG_DIR="${LOG_DIR%/}"
PIVOT_COUNTS="${PIVOT_COUNTS//,/ }"
read -r -a PIVOTS <<<"$PIVOT_COUNTS"
((${#PIVOTS[@]} > 0)) || { echo "--pivot-counts is empty" >&2; exit 2; }
previous=0
for pivot in "${PIVOTS[@]}"; do
  [[ "$pivot" =~ ^[0-9]+$ ]] && ((pivot >= 2)) || {
    echo "Invalid pivot count: $pivot (expected integer >= 2)" >&2; exit 2;
  }
  ((pivot > previous)) || {
    echo "Pivot counts must be strictly increasing" >&2; exit 2;
  }
  previous="$pivot"
done
PMAX="${PIVOTS[$((${#PIVOTS[@]} - 1))]}"

BASE="${DATA_ROOT}/sift1m/origin/sift1m_base.fvecs"
QUERY="${DATA_ROOT}/sift1m/origin/sift1m_query.fvecs"
if ((!DRY_RUN)); then
  [[ -f "$BASE" ]] || { echo "Missing base vectors: $BASE" >&2; exit 1; }
  [[ -f "$QUERY" ]] || { echo "Missing query vectors: $QUERY" >&2; exit 1; }
  if [[ "$PHASE" == all || "$PHASE" == perf ]]; then
    [[ -x "$PERF_BIN" ]] || { echo "Missing perf binary: $PERF_BIN" >&2; exit 1; }
  fi
  if [[ "$PHASE" == all || "$PHASE" == stats ]]; then
    [[ -x "$STATS_BIN" ]] || { echo "Missing stats binary: $STATS_BIN" >&2; exit 1; }
  fi
  mkdir -p "$LOG_DIR/manifests"
fi

index_path() {
  local opt="$1" scope="$2" method="$3" pivots="$4" seed="$5"
  printf '%s/sift1m/index/v10_nlist_%s_metric_l2_opt_%s_subk_15_subNprobeRatio_1_mp_%s_%s_P%s_seed%s.index\n' \
    "$DATA_ROOT" "$NLIST" "$opt" "$scope" "$method" "$pivots" "$seed"
}

BASELINE_INDEX="$(index_path 0 global affine_fps 0 0)"
TRIANGLE_INDEX="$(index_path 1 per_list pca 0 "$PIVOT_SEED")"
GLOBAL_INDEX="$(index_path 1 global pca "$PMAX" "$PIVOT_SEED")"
PER_LIST_INDEX="$(index_path 1 per_list pca "$PMAX" "$PIVOT_SEED")"

log() {
  local line="[$(date -Is)] $*"
  echo "$line"
  ((DRY_RUN)) || echo "$line" >>"$LOG_DIR/run.log"
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
  local path="$1" expected_text="$2"
  [[ -s "$path" ]] || return 1
  python3 - "$path" "$NPROBE" "$expected_text" <<'PY'
import csv
import sys

path, nprobe, expected_text = sys.argv[1:]
expected = {(int(nprobe), int(p)) for p in expected_text.split()}
try:
    with open(path, newline="") as stream:
        found = {
            (int(float(row["nprobe"])), int(float(row["pivot_count"])))
            for row in csv.DictReader(stream)
        }
except (OSError, ValueError, KeyError, csv.Error):
    raise SystemExit(1)
raise SystemExit(0 if expected.issubset(found) else 1)
PY
}

run_case() {
  local phase="$1" tag="$2" repeat="$3" expected="$4" expected_index="$5"
  shift 5
  local binary loops output csv partial logfile rc
  local -a command
  if [[ "$phase" == perf ]]; then
    binary="$PERF_BIN"
    loops="$PERF_LOOPS"
  else
    binary="$STATS_BIN"
    loops=1
  fi
  output="$LOG_DIR/$phase/sift1m"
  csv="$output/${tag}_rep${repeat}.csv"
  partial="$csv.partial"
  logfile="$output/${tag}_rep${repeat}.log"

  if csv_complete "$csv" "$expected" && { ((DRY_RUN)) || [[ -s "$expected_index" ]]; }; then
    log "SKIP $phase/sift1m/$tag/rep$repeat"
    return
  fi

  command=(
    "$binary" --benchmarks_path "$DATA_ROOT" --dataset sift1m
    --input_format fvecs --output_format bin --metric l2 --k 1
    --nlist "$NLIST" --nprobes "$NPROBE" --cache
    --signature_precision float32 --loop "$loops" --csv "$partial"
    "$@"
  )
  ((VERBOSE)) && command+=(--verbose)
  log "START $phase/sift1m/$tag/rep$repeat loops=$loops"
  if ((DRY_RUN)); then
    run_bound "${command[@]}"
    return
  fi
  mkdir -p "$output"
  rm -f "$partial" "$logfile.partial"
  if run_bound "${command[@]}" >"$logfile.partial" 2>&1; then
    csv_complete "$partial" "$expected" || {
      log "FAILED incomplete CSV: $partial"; return 1;
    }
    [[ -s "$expected_index" ]] || {
      log "FAILED expected index was not created: $expected_index"; return 1;
    }
    mv "$partial" "$csv"
    mv "$logfile.partial" "$logfile"
    log "DONE $phase/sift1m/$tag/rep$repeat"
  else
    rc=$?
    log "FAILED $phase/sift1m/$tag/rep$repeat rc=$rc; inspect $logfile.partial"
    return "$rc"
  fi
}

run_phase() {
  local phase="$1" repeats="$2"
  local repeat scope rich_index
  for ((repeat=1; repeat<=repeats; ++repeat)); do
    run_case "$phase" baseline "$repeat" 0 "$BASELINE_INDEX" \
      --opt_levels OPT_NONE --multipivot_modes none \
      --multipivot_scope global --multipivot_method affine_fps --pivot_counts 0

    run_case "$phase" triangle "$repeat" 1 "$TRIANGLE_INDEX" \
      --opt_levels OPT_TRIANGLE --multipivot_modes none \
      --multipivot_scope per_list --multipivot_method pca --pivot_counts 1 \
      --pivot_seed "$PIVOT_SEED" --from_index "$BASELINE_INDEX"

    for scope in global per_list; do
      [[ "$scope" == global ]] && rich_index="$GLOBAL_INDEX" || rich_index="$PER_LIST_INDEX"
      run_case "$phase" "${scope}_static" "$repeat" "$PIVOT_COUNTS" "$rich_index" \
        --opt_levels OPT_TRIANGLE --multipivot_modes projection \
        --multipivot_scope "$scope" --multipivot_method pca --pivot_counts "$PMAX" \
        --active_pivot_counts "${PIVOTS[@]}" --projection_block_size 0 \
        --pivot_seed "$PIVOT_SEED" --from_index "$TRIANGLE_INDEX" \
        --pivot_manifest "$LOG_DIR/manifests/sift1m_${scope}_${phase}_rep${repeat}.csv"
    done
  done
}

log "CONFIG data=$DATA_ROOT log=$LOG_DIR cpus=$CPU_SET threads=$THREADS phase=$PHASE nlist=$NLIST nprobe=$NPROBE pivots='$PIVOT_COUNTS' seed=$PIVOT_SEED perf_repeats=$PERF_REPEATS perf_loops=$PERF_LOOPS"

if [[ "$PHASE" == all || "$PHASE" == perf ]]; then
  run_phase perf "$PERF_REPEATS"
fi
if [[ "$PHASE" == all || "$PHASE" == stats ]]; then
  run_phase stats 1
fi

if ((!DRY_RUN && !SKIP_PLOT)) && [[ "$PHASE" == all ]]; then
  python3 "$ROOT/scripts/plot_sift1m_global_local_cost.py" \
    --input-root "$LOG_DIR" --output-dir "$LOG_DIR/figures" --nprobe "$NPROBE"
fi
log "COMPLETE results=$LOG_DIR"
