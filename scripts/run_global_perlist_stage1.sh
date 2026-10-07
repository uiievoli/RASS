#!/usr/bin/env bash
# Stage 1 of the global-vs-per-list PCA experiment.
#
# For every selected dataset this script creates one canonical IVF-Flat index,
# upgrades it once with Triangle metadata, and derives both the global and the
# per-list PCA indexes from that exact Triangle index.  Static PCA prefixes are
# then evaluated from one Pmax index per scope, so no configuration retrains IVF.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-}"
DBPEDIA_ROOT="${DBPEDIA_ROOT:-}"
LOG_DIR="${LOG_DIR:-}"
CPU_SET="${CPU_SET:-0-31}"
THREADS="${THREADS:-}"
PHASE="${PHASE:-all}"
ONLY_DATASETS="${ONLY_DATASETS:-}"
PERF_REPEATS="${PERF_REPEATS:-3}"
SCOPE_PIVOT_SEED="${SCOPE_PIVOT_SEED:-20261004}"
PERF_BIN="${PERF_BIN:-${ROOT}/build-perf/bin/query}"
STATS_BIN="${STATS_BIN:-${ROOT}/build-stats/bin/query}"
DRY_RUN=0
VERBOSE=0

usage() {
  cat <<'EOF'
Usage:
  scripts/run_global_perlist_stage1.sh \
    --data-root DIR --log-dir DIR --cpus LIST [options]

Required:
  --data-root DIR       Root containing the eight fvecs dataset directories.
  --log-dir DIR         Output root for CSV files, logs and manifests.

Options:
  --dbpedia-root DIR    Root containing dbpedia1536m_holdout; defaults to DATA_ROOT.
  --cpus LIST           taskset CPU list, e.g. 0-31 or 0-15,32-47.
  --threads N           OpenMP threads; defaults to the bound CPU count.
  --phase NAME          all (default), perf, or stats.
  --datasets LIST       Comma/space-separated subset of the seven datasets.
  --perf-repeats N      Independent performance processes (default 3).
  --pivot-seed N        PCA seed and cache namespace (default 20261004).
  --verbose             Pass --verbose to query.
  --dry-run             Validate inputs and print commands without running them.
  -h, --help            Show this help.

The experiment is static PCA only.  P includes the centroid, so P projection
plans contain P-1 PCA coordinates.  Existing complete CSVs and indexes are
reused, making the script safe to resume.
EOF
}

while (($#)); do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --dbpedia-root) DBPEDIA_ROOT="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --cpus) CPU_SET="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --phase) PHASE="$2"; shift 2 ;;
    --datasets) ONLY_DATASETS="$2"; shift 2 ;;
    --perf-repeats) PERF_REPEATS="$2"; shift 2 ;;
    --pivot-seed) SCOPE_PIVOT_SEED="$2"; shift 2 ;;
    --verbose) VERBOSE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "${DATA_ROOT}" ]] || { echo "--data-root is required" >&2; exit 2; }
[[ -n "${LOG_DIR}" ]] || { echo "--log-dir is required" >&2; exit 2; }
[[ "${PHASE}" =~ ^(all|perf|stats)$ ]] || {
  echo "--phase must be all, perf, or stats" >&2; exit 2;
}
[[ "${PERF_REPEATS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "--perf-repeats must be a positive integer" >&2; exit 2;
}
[[ "${SCOPE_PIVOT_SEED}" =~ ^[0-9]+$ ]] || {
  echo "--pivot-seed must be a non-negative integer" >&2; exit 2;
}
command -v taskset >/dev/null || { echo "taskset is required" >&2; exit 1; }
if [[ -z "${THREADS}" ]]; then
  THREADS="$(taskset -c "${CPU_SET}" nproc)" || {
    echo "Invalid or unavailable CPU set: ${CPU_SET}" >&2; exit 2;
  }
fi
[[ "${THREADS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "--threads must be a positive integer" >&2; exit 2;
}
DBPEDIA_ROOT="${DBPEDIA_ROOT:-${DATA_ROOT}}"

DATASETS=(
  nuswide fasion_mnist_784 msong_holdout sift1m glove25
  StarLightCurves dbpedia1536m_holdout
)

declare -A NLIST=(
  [nuswide]=512 [fasion_mnist_784]=256 [msong_holdout]=1000 [sift1m]=1000
  [glove25]=1024 [StarLightCurves]=128
  [dbpedia1536m_holdout]=1000
)

# Fixed operating points near recall 0.9, an intermediate recall, and recall 0.99.
declare -A NPROBES=(
  [nuswide]="2 3"
  [fasion_mnist_784]="3 5 7"
  [msong_holdout]="6 10 20"
  [sift1m]="9 20 30"
  [glove25]="8 20 50"
  [StarLightCurves]="3 5"
  [dbpedia1536m_holdout]="14 30 75 150"
)

# Each scope builds only Pmax; --active_pivot_counts evaluates these prefixes.
declare -A P_VALUES=(
  [nuswide]="4 8 16 32 50"
  [fasion_mnist_784]="4 8 16 32 64"
  [msong_holdout]="4 8 16 32 42 64"
  [sift1m]="4 8 16 24"
  [glove25]="2 4 8 16 25"
  [StarLightCurves]="4 16 32 64 103"
  [dbpedia1536m_holdout]="4 16 32 64 128"
)

# Make every timed query command long enough to suppress scheduler noise.
declare -A PERF_LOOPS=(
  [nuswide]=500 [fasion_mnist_784]=50 [msong_holdout]=20 [sift1m]=5
  [glove25]=10 [StarLightCurves]=200
  [dbpedia1536m_holdout]=1
)

selected() {
  local needle="$1"
  [[ -z "${ONLY_DATASETS}" ]] && return 0
  local normalized=" ${ONLY_DATASETS//,/ } "
  [[ "${normalized}" == *" ${needle} "* ]]
}

if [[ -n "${ONLY_DATASETS}" ]]; then
  read -r -a requested <<<"${ONLY_DATASETS//,/ }"
  for value in "${requested[@]}"; do
    known=0
    for dataset in "${DATASETS[@]}"; do
      [[ "${value}" == "${dataset}" ]] && { known=1; break; }
    done
    ((known)) || { echo "Unknown dataset: ${value}" >&2; exit 2; }
  done
fi

dataset_root() {
  if [[ "$1" == dbpedia1536m_holdout ]]; then
    printf '%s\n' "${DBPEDIA_ROOT}"
  else
    printf '%s\n' "${DATA_ROOT}"
  fi
}

log() {
  local line="[$(date -Is)] $*"
  echo "${line}"
  ((DRY_RUN)) || echo "${line}" >>"${LOG_DIR}/run.log"
}

run_bound() {
  if ((DRY_RUN)); then
    printf '%q ' env -u TRIBASE_TRACE -u EDGE_DEVICE_ENABLED \
      OMP_NUM_THREADS="${THREADS}" OMP_PROC_BIND=close OMP_PLACES=cores \
      taskset -c "${CPU_SET}" "$@"
    printf '\n'
  else
    env -u TRIBASE_TRACE -u EDGE_DEVICE_ENABLED \
      OMP_NUM_THREADS="${THREADS}" OMP_PROC_BIND=close OMP_PLACES=cores \
      taskset -c "${CPU_SET}" "$@"
  fi
}

csv_complete() {
  local path="$1" probes="$2" pivots="$3"
  [[ -s "${path}" ]] || return 1
  python3 - "${path}" "${probes}" "${pivots}" <<'PY'
import csv
import sys

path, probes_text, pivots_text = sys.argv[1:]
expected = {
    (int(nprobe), int(pivot))
    for nprobe in probes_text.split()
    for pivot in pivots_text.split()
}
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

index_path() {
  local root="$1" dataset="$2" opt="$3" scope="$4" method="$5" pivots="$6" seed="${7:-0}"
  printf '%s/%s/index/v10_nlist_%s_metric_l2_opt_%s_subk_15_subNprobeRatio_1_mp_%s_%s_P%s_seed%s.index\n' \
    "${root}" "${dataset}" "${NLIST[${dataset}]}" "${opt}" \
    "${scope}" "${method}" "${pivots}" "${seed}"
}

run_case() {
  local phase="$1" dataset="$2" tag="$3" repeat="$4"
  local expected_pivots="$5" expected_index="$6"
  shift 6

  local binary loops root output csv partial logfile
  if [[ "${phase}" == perf ]]; then
    binary="${PERF_BIN}"
    loops="${PERF_LOOPS[${dataset}]}"
  else
    binary="${STATS_BIN}"
    loops=1
  fi
  root="$(dataset_root "${dataset}")"
  output="${LOG_DIR}/${phase}/${dataset}"
  csv="${output}/${tag}_rep${repeat}.csv"
  partial="${csv}.partial"
  logfile="${output}/${tag}_rep${repeat}.log"

  if csv_complete "${csv}" "${NPROBES[${dataset}]}" "${expected_pivots}" && \
     { ((DRY_RUN)) || [[ -s "${expected_index}" ]]; }; then
    log "SKIP ${phase}/${dataset}/${tag}/rep${repeat}"
    return
  fi

  local -a probes command
  read -r -a probes <<<"${NPROBES[${dataset}]}"
  command=(
    "${binary}" --benchmarks_path "${root}" --dataset "${dataset}"
    --input_format fvecs --output_format bin --metric l2 --k 1
    --nlist "${NLIST[${dataset}]}" --nprobes "${probes[@]}" --cache
    --signature_precision float32 --loop "${loops}" --csv "${partial}"
    "$@"
  )
  ((VERBOSE)) && command+=(--verbose)

  log "START ${phase}/${dataset}/${tag}/rep${repeat} loops=${loops}"
  if ((DRY_RUN)); then
    run_bound "${command[@]}"
    return
  fi
  mkdir -p "${output}"
  rm -f "${partial}" "${logfile}.partial"
  if run_bound "${command[@]}" >"${logfile}.partial" 2>&1; then
    if ! csv_complete "${partial}" "${NPROBES[${dataset}]}" "${expected_pivots}"; then
      log "FAILED incomplete CSV: ${partial}"
      return 1
    fi
    [[ -s "${expected_index}" ]] || {
      log "FAILED expected index was not created: ${expected_index}"
      return 1
    }
    mv "${partial}" "${csv}"
    mv "${logfile}.partial" "${logfile}"
    log "DONE ${phase}/${dataset}/${tag}/rep${repeat}"
  else
    local rc=$?
    log "FAILED ${phase}/${dataset}/${tag}/rep${repeat} rc=${rc}; see ${logfile}.partial"
    return "${rc}"
  fi
}

run_dataset_phase() {
  local dataset="$1" phase="$2" repeats="$3"
  local root baseline_index triangle_index pmax
  local global_index per_list_index
  root="$(dataset_root "${dataset}")"
  baseline_index="$(index_path "${root}" "${dataset}" 0 global affine_fps 0)"
  triangle_index="$(index_path "${root}" "${dataset}" 1 per_list pca 0 "${SCOPE_PIVOT_SEED}")"
  read -r -a active <<<"${P_VALUES[${dataset}]}"
  pmax="${active[$((${#active[@]} - 1))]}"
  global_index="$(index_path "${root}" "${dataset}" 1 global pca "${pmax}" "${SCOPE_PIVOT_SEED}")"
  per_list_index="$(index_path "${root}" "${dataset}" 1 per_list pca "${pmax}" "${SCOPE_PIVOT_SEED}")"

  for ((repeat=1; repeat<=repeats; ++repeat)); do
    run_case "${phase}" "${dataset}" baseline "${repeat}" 0 "${baseline_index}" \
      --opt_levels OPT_NONE --multipivot_modes none \
      --multipivot_scope global --multipivot_method affine_fps --pivot_counts 0

    run_case "${phase}" "${dataset}" triangle "${repeat}" 1 "${triangle_index}" \
      --opt_levels OPT_TRIANGLE --multipivot_modes none \
      --multipivot_scope per_list --multipivot_method pca --pivot_counts 1 \
      --pivot_seed "${SCOPE_PIVOT_SEED}" \
      --from_index "${baseline_index}"

    run_case "${phase}" "${dataset}" global_static "${repeat}" \
      "${P_VALUES[${dataset}]}" "${global_index}" \
      --opt_levels OPT_TRIANGLE --multipivot_modes projection \
      --multipivot_scope global --multipivot_method pca --pivot_counts "${pmax}" \
      --active_pivot_counts "${active[@]}" --projection_block_size 0 \
      --pivot_seed "${SCOPE_PIVOT_SEED}" \
      --from_index "${triangle_index}" \
      --pivot_manifest "${LOG_DIR}/manifests/${dataset}_global_${phase}_rep${repeat}.csv"

    run_case "${phase}" "${dataset}" per_list_static "${repeat}" \
      "${P_VALUES[${dataset}]}" "${per_list_index}" \
      --opt_levels OPT_TRIANGLE --multipivot_modes projection \
      --multipivot_scope per_list --multipivot_method pca --pivot_counts "${pmax}" \
      --active_pivot_counts "${active[@]}" --projection_block_size 0 \
      --pivot_seed "${SCOPE_PIVOT_SEED}" \
      --from_index "${triangle_index}" \
      --pivot_manifest "${LOG_DIR}/manifests/${dataset}_per_list_${phase}_rep${repeat}.csv"
  done
}

summarize() {
  ((DRY_RUN)) && return
  python3 - "${LOG_DIR}" <<'PY'
import csv
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
pattern = re.compile(r"(baseline|triangle|global_static|per_list_static)_rep([0-9]+)\.csv")
rows = []
for phase in ("perf", "stats"):
    phase_root = root / phase
    if not phase_root.exists():
        continue
    for path in sorted(phase_root.glob("*/*.csv")):
        match = pattern.fullmatch(path.name)
        if not match:
            continue
        dataset = path.parent.name
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                qps = float(row.get("qps", 0) or 0)
                row = dict(row)
                row.update({
                    "phase": phase,
                    "experiment": match.group(1),
                    "repeat": match.group(2),
                    "dataset": dataset,
                    "latency_ms": 1000.0 / qps if qps > 0 else 0.0,
                })
                rows.append(row)
if rows:
    leading = ["phase", "dataset", "experiment", "repeat", "latency_ms"]
    fields = leading + [key for key in rows[0] if key not in leading]
    with (root / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {root / 'summary.csv'} rows={len(rows)}")
PY
}

if ((!DRY_RUN)); then
  mkdir -p "${LOG_DIR}" "${LOG_DIR}/manifests"
fi
for binary in "${PERF_BIN}" "${STATS_BIN}"; do
  if [[ "${PHASE}" == perf && "${binary}" == "${STATS_BIN}" ]] || \
     [[ "${PHASE}" == stats && "${binary}" == "${PERF_BIN}" ]]; then
    continue
  fi
  ((DRY_RUN)) || [[ -x "${binary}" ]] || {
    echo "Missing query binary: ${binary}" >&2; exit 1;
  }
done

log "CONFIG data_root=${DATA_ROOT} dbpedia_root=${DBPEDIA_ROOT} log_dir=${LOG_DIR} cpus=${CPU_SET} threads=${THREADS} phase=${PHASE} perf_repeats=${PERF_REPEATS} scope_pivot_seed=${SCOPE_PIVOT_SEED}"

for dataset in "${DATASETS[@]}"; do
  selected "${dataset}" || continue
  root="$(dataset_root "${dataset}")"
  if ((!DRY_RUN)); then
    for path in \
      "${root}/${dataset}/origin/${dataset}_base.fvecs" \
      "${root}/${dataset}/origin/${dataset}_query.fvecs"; do
      [[ -f "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 1; }
    done
  fi
  if [[ "${PHASE}" == all || "${PHASE}" == perf ]]; then
    run_dataset_phase "${dataset}" perf "${PERF_REPEATS}"
  fi
  if [[ "${PHASE}" == all || "${PHASE}" == stats ]]; then
    run_dataset_phase "${dataset}" stats 1
  fi
done

summarize
log "COMPLETE output=${LOG_DIR}"
