#!/usr/bin/env bash
# Fine static-PCA prefix screen for the eight in-memory fvecs datasets.
# P includes the centroid; a P-pivot plan therefore uses P-1 PCA coordinates.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/mnt/nvme/wxy/benchmarks}"
DBPEDIA_ROOT="${DBPEDIA_ROOT:-/mnt/nvme/wxy}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/fine_pivot_screen_8_20261004}"
CPU_SET="${CPU_SET:-0-31}"
THREADS="${THREADS:-32}"
PERF_REPEATS="${PERF_REPEATS:-3}"
PIVOT_SEED="${PIVOT_SEED:-20261005}"
PERF_BIN="${PERF_BIN:-${ROOT}/build-perf/bin/query}"
STATS_BIN="${STATS_BIN:-${ROOT}/build-stats/bin/query}"
ONLY_DATASETS="${ONLY_DATASETS:-}"
PHASE="${PHASE:-all}"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: scripts/run_fine_pivot_screen_8.sh [options]
  --data-root DIR       Root of the first seven datasets.
  --dbpedia-root DIR    Root containing dbpedia1536m_holdout.
  --log-dir DIR         Output directory.
  --cpus LIST           taskset CPU list (default 0-31).
  --threads N           OpenMP threads (default 32).
  --perf-repeats N      Independent perf processes (default 3).
  --datasets LIST       Comma/space-separated subset.
  --phase NAME          all, perf, or stats.
  --pivot-seed N        Fresh cache namespace (default 20261005).
  --dry-run             Print commands only.
EOF
}

while (($#)); do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --dbpedia-root) DBPEDIA_ROOT="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --cpus) CPU_SET="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --perf-repeats) PERF_REPEATS="$2"; shift 2 ;;
    --datasets) ONLY_DATASETS="$2"; shift 2 ;;
    --phase) PHASE="$2"; shift 2 ;;
    --pivot-seed) PIVOT_SEED="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$PHASE" =~ ^(all|perf|stats)$ ]] || { echo "invalid --phase" >&2; exit 2; }
[[ -x "$PERF_BIN" && -x "$STATS_BIN" ]] || { echo "query binaries are missing" >&2; exit 1; }

DATASETS=(nuswide fasion_mnist_784 msong_holdout sift1m glove25 HandOutlines StarLightCurves dbpedia1536m_holdout)
declare -A NLIST=(
  [nuswide]=512 [fasion_mnist_784]=256 [msong_holdout]=1000 [sift1m]=1000
  [glove25]=1024 [HandOutlines]=32 [StarLightCurves]=128 [dbpedia1536m_holdout]=1000
)
# These brackets contain the expected first Recall >= 0.99 point and nearby points.
declare -A NPROBES=(
  [nuswide]="2 3" [fasion_mnist_784]="5 6 7"
  [msong_holdout]="10 12 14 16 18 20" [sift1m]="30 32 34 36"
  [glove25]="40 45 50" [HandOutlines]="3 4 5"
  [StarLightCurves]="3 4 5" [dbpedia1536m_holdout]="150 160 175 200"
)
declare -A P_VALUES=(
  [nuswide]="$(seq -s ' ' 2 16) 20 24 28 32"
  [fasion_mnist_784]="4 8 12 $(seq -s ' ' 16 2 80) 84 88 92 96"
  [msong_holdout]="4 8 12 16 $(seq -s ' ' 20 2 80) 84 88 92 96"
  [sift1m]="$(seq -s ' ' 2 40)"
  [glove25]="$(seq -s ' ' 2 26)"
  [HandOutlines]="$(seq -s ' ' 2 32) 40 48 56 64"
  [StarLightCurves]="$(seq -s ' ' 4 2 104)"
  [dbpedia1536m_holdout]="4 8 16 32 $(seq -s ' ' 48 4 256)"
)
# One prefix run evaluates many configurations, so use lower per-row loops here.
declare -A PERF_LOOPS=(
  [nuswide]=50 [fasion_mnist_784]=10 [msong_holdout]=5 [sift1m]=2
  [glove25]=10 [HandOutlines]=100 [StarLightCurves]=50 [dbpedia1536m_holdout]=1
)

selected() {
  [[ -z "$ONLY_DATASETS" ]] && return 0
  local values=" ${ONLY_DATASETS//,/ } "
  [[ "$values" == *" $1 "* ]]
}
dataset_root() { [[ "$1" == dbpedia1536m_holdout ]] && echo "$DBPEDIA_ROOT" || echo "$DATA_ROOT"; }
index_path() {
  local root="$1" ds="$2" opt="$3" scope="$4" method="$5" pivots="$6" seed="$7"
  printf '%s/%s/index/v10_nlist_%s_metric_l2_opt_%s_subk_15_subNprobeRatio_1_mp_%s_%s_P%s_seed%s.index\n' \
    "$root" "$ds" "${NLIST[$ds]}" "$opt" "$scope" "$method" "$pivots" "$seed"
}
log() { echo "[$(date -Is)] $*" | tee -a "$LOG_DIR/run.log"; }
run_bound() {
  if ((DRY_RUN)); then printf '%q ' env -u TRIBASE_TRACE -u EDGE_DEVICE_ENABLED OMP_NUM_THREADS="$THREADS" OMP_PROC_BIND=close OMP_PLACES=cores taskset -c "$CPU_SET" "$@"; printf '\n'
  else env -u TRIBASE_TRACE -u EDGE_DEVICE_ENABLED OMP_NUM_THREADS="$THREADS" OMP_PROC_BIND=close OMP_PLACES=cores taskset -c "$CPU_SET" "$@"
  fi
}
csv_complete() {
  local path="$1" probes="$2" pivots="$3"
  [[ -s "$path" ]] || return 1
  python3 - "$path" "$probes" "$pivots" <<'PY'
import csv, sys
path, probes, pivots = sys.argv[1:]
want={(int(a),int(b)) for a in probes.split() for b in pivots.split()}
try:
    with open(path,newline='') as f:
        got={(int(float(r['nprobe'])),int(float(r['pivot_count']))) for r in csv.DictReader(f)}
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if want <= got else 1)
PY
}
run_case() {
  local phase="$1" ds="$2" tag="$3" rep="$4" expected="$5" expected_index="$6"; shift 6
  local bin loops root out csv partial logfile
  [[ "$phase" == perf ]] && { bin="$PERF_BIN"; loops="${PERF_LOOPS[$ds]}"; } || { bin="$STATS_BIN"; loops=1; }
  root="$(dataset_root "$ds")"; out="$LOG_DIR/$phase/$ds"
  csv="$out/${tag}_rep${rep}.csv"; partial="$csv.partial"; logfile="$out/${tag}_rep${rep}.log"
  if csv_complete "$csv" "${NPROBES[$ds]}" "$expected" && [[ -s "$expected_index" ]]; then log "SKIP $phase/$ds/$tag/rep$rep"; return; fi
  mkdir -p "$out"
  local -a probes; read -r -a probes <<<"${NPROBES[$ds]}"
  local -a cmd=("$bin" --benchmarks_path "$root" --dataset "$ds" --input_format fvecs --output_format bin --metric l2 --k 1 --nlist "${NLIST[$ds]}" --nprobes "${probes[@]}" --cache --signature_precision float32 --loop "$loops" --csv "$partial" "$@")
  log "START $phase/$ds/$tag/rep$rep loops=$loops"
  ((DRY_RUN)) && { run_bound "${cmd[@]}"; return; }
  rm -f "$partial" "$logfile.partial"
  run_bound "${cmd[@]}" >"$logfile.partial" 2>&1
  csv_complete "$partial" "${NPROBES[$ds]}" "$expected" || { log "FAILED incomplete $partial"; return 1; }
  [[ -s "$expected_index" ]] || { log "FAILED missing index $expected_index"; return 1; }
  mv "$partial" "$csv"; mv "$logfile.partial" "$logfile"
  log "DONE $phase/$ds/$tag/rep$rep"
}
run_phase_dataset() {
  local ds="$1" phase="$2" repeats="$3" root baseline triangle pmax
  root="$(dataset_root "$ds")"
  baseline="$(index_path "$root" "$ds" 0 global affine_fps 0 0)"
  triangle="$(index_path "$root" "$ds" 1 per_list pca 0 "$PIVOT_SEED")"
  local -a active; read -r -a active <<<"${P_VALUES[$ds]}"; pmax="${active[-1]}"
  local -a scopes=(per_list)
  case "$ds" in glove25|HandOutlines|StarLightCurves) scopes=(global);; nuswide) scopes=(global per_list);; esac
  for ((rep=1; rep<=repeats; ++rep)); do
    run_case "$phase" "$ds" baseline "$rep" 0 "$baseline" --opt_levels OPT_NONE --multipivot_modes none --multipivot_scope global --multipivot_method affine_fps --pivot_counts 0
    run_case "$phase" "$ds" triangle "$rep" 1 "$triangle" --opt_levels OPT_TRIANGLE --multipivot_modes none --multipivot_scope per_list --multipivot_method pca --pivot_counts 1 --pivot_seed "$PIVOT_SEED" --from_index "$baseline"
    for scope in "${scopes[@]}"; do
      local rich; rich="$(index_path "$root" "$ds" 1 "$scope" pca "$pmax" "$PIVOT_SEED")"
      run_case "$phase" "$ds" "${scope}_static" "$rep" "${P_VALUES[$ds]}" "$rich" --opt_levels OPT_TRIANGLE --multipivot_modes projection --multipivot_scope "$scope" --multipivot_method pca --pivot_counts "$pmax" --active_pivot_counts "${active[@]}" --projection_block_size 0 --pivot_seed "$PIVOT_SEED" --from_index "$triangle" --pivot_manifest "$LOG_DIR/manifests/${ds}_${scope}_${phase}_rep${rep}.csv"
    done
  done
}

mkdir -p "$LOG_DIR" "$LOG_DIR/manifests"
log "CONFIG fine-screen data=$DATA_ROOT dbpedia=$DBPEDIA_ROOT cpus=$CPU_SET threads=$THREADS repeats=$PERF_REPEATS seed=$PIVOT_SEED"
for ds in "${DATASETS[@]}"; do
  selected "$ds" || continue
  root="$(dataset_root "$ds")"
  [[ -f "$root/$ds/origin/${ds}_base.fvecs" ]] || { echo "missing $ds base" >&2; exit 1; }
  [[ "$PHASE" == all || "$PHASE" == perf ]] && run_phase_dataset "$ds" perf "$PERF_REPEATS"
  [[ "$PHASE" == all || "$PHASE" == stats ]] && run_phase_dataset "$ds" stats 1
done
log "COMPLETE $LOG_DIR"
