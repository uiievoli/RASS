#!/usr/bin/env bash
# Run the complete eight-dataset Tribase comparison. The legacy filename is
# retained for compatibility. HandOutlines and SpaceV are no longer selected.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-}"
LOG_DIR="${LOG_DIR:-}"
CPU_SET="${CPU_SET:-0-31}"
THREADS="${THREADS:-}"
BUILD_BATCH_VECTORS="${BUILD_BATCH_VECTORS:-262144}"
COARSE_HNSW_M="${COARSE_HNSW_M:-32}"
COARSE_HNSW_EF_CONSTRUCTION="${COARSE_HNSW_EF_CONSTRUCTION:-200}"
COARSE_HNSW_EF_SEARCH="${COARSE_HNSW_EF_SEARCH:-128}"
PHASE="${PHASE:-all}"
ONLY_DATASETS="${ONLY_DATASETS:-}"
PERF_BIN="${PERF_BIN:-${ROOT}/build-perf/bin/query}"
STATS_BIN="${STATS_BIN:-${ROOT}/build-stats/bin/query}"
SIFT1B_DIR="${SIFT1B_DIR:-}"
DRY_RUN=0
VERBOSE=0

usage() {
  cat <<'EOF'
Usage:
  scripts/run_all_8_datasets.sh --data-root DIR --log-dir DIR [options]

Required:
  --data-root DIR       Root containing the eight active dataset directories.
  --log-dir DIR         Output root for CSV files, logs and manifests.

CPU control:
  --cpus LIST           Linux CPU list passed to taskset, e.g. 0-31 or 0-15,32-47.
  --threads N           OpenMP/worker threads; defaults to the number of bound CPUs.
  --build-batch-vectors N
                        Decoded vectors retained per SIFT1B build batch
                        (default 262144, about 128 MiB at D=128).
  --coarse-hnsw-m N     HNSW coarse graph degree for SIFT1B (default 32).
  --coarse-hnsw-ef-construction N
                        HNSW construction breadth (default 200).
  --coarse-hnsw-ef-search N
                        HNSW centroid-assignment breadth (default 128).

Selection:
  --phase NAME          all (default), perf, or stats.
  --datasets LIST       Subset of the eight active dataset names.
  --sift1b-dir DIR      Override DATA_ROOT/sift1b/raw.
  --verbose             Pass --verbose to query and retain detailed build logs.
  --dry-run             Validate arguments and print commands without running them.
  -h, --help            Show this help.

Expected common layout:
  DATA_ROOT/<dataset>/origin/<dataset>_base.fvecs
  DATA_ROOT/<dataset>/origin/<dataset>_query.fvecs
  DATA_ROOT/<dataset>/result/groundtruth_1.bin

Large inputs remain byte-coded on disk and are decoded to float when query.cpp
loads them. Their indexes use the same float Index/IVF format as other data.

The script creates/reuses indexes inside each dataset directory. Completed CSVs
are skipped, so rerunning the same command resumes an interrupted experiment.
EOF
}

while (($#)); do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --cpus) CPU_SET="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --build-batch-vectors) BUILD_BATCH_VECTORS="$2"; shift 2 ;;
    --coarse-hnsw-m) COARSE_HNSW_M="$2"; shift 2 ;;
    --coarse-hnsw-ef-construction) COARSE_HNSW_EF_CONSTRUCTION="$2"; shift 2 ;;
    --coarse-hnsw-ef-search) COARSE_HNSW_EF_SEARCH="$2"; shift 2 ;;
    --phase) PHASE="$2"; shift 2 ;;
    --datasets) ONLY_DATASETS="$2"; shift 2 ;;
    --sift1b-dir) SIFT1B_DIR="$2"; shift 2 ;;
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
command -v taskset >/dev/null || { echo "taskset is required" >&2; exit 1; }
if [[ -z "${THREADS}" ]]; then
  THREADS="$(taskset -c "${CPU_SET}" nproc)" || {
    echo "Invalid or unavailable CPU set: ${CPU_SET}" >&2; exit 2;
  }
fi
[[ "${THREADS}" =~ ^[1-9][0-9]*$ ]] || { echo "invalid --threads: ${THREADS}" >&2; exit 2; }
[[ "${BUILD_BATCH_VECTORS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "invalid --build-batch-vectors: ${BUILD_BATCH_VECTORS}" >&2; exit 2;
}
for hnsw_value in "${COARSE_HNSW_M}" "${COARSE_HNSW_EF_CONSTRUCTION}" \
                  "${COARSE_HNSW_EF_SEARCH}"; do
  [[ "${hnsw_value}" =~ ^[1-9][0-9]*$ ]] || {
    echo "invalid HNSW coarse parameter: ${hnsw_value}" >&2; exit 2;
  }
done

SIFT1B_DIR="${SIFT1B_DIR:-${DATA_ROOT}/sift1b/raw}"
ALL_DATASETS=(
  nuswide fasion_mnist_784 msong_holdout sift1m glove25
  StarLightCurves dbpedia1536m_holdout sift1b
)
FLOAT_DATASETS=("${ALL_DATASETS[@]}")

declare -A INPUT_FORMAT=(
  [nuswide]=fvecs [fasion_mnist_784]=fvecs [msong_holdout]=fvecs [sift1m]=fvecs
  [glove25]=fvecs [StarLightCurves]=fvecs
  [dbpedia1536m_holdout]=fvecs [sift1b]=bvecs
)

declare -A NLIST=(
  [nuswide]=512 [fasion_mnist_784]=256 [msong_holdout]=1000 [sift1m]=1000
  [glove25]=1024 [StarLightCurves]=128
  [dbpedia1536m_holdout]=1000 [sift1b]=32768
)
declare -A SEARCH_K=(
  [nuswide]=1 [fasion_mnist_784]=1 [msong_holdout]=1 [sift1m]=1
  [glove25]=1 [StarLightCurves]=1 [dbpedia1536m_holdout]=1 [sift1b]=10
)
declare -A QUERY_COUNT=(
  [nuswide]=0 [fasion_mnist_784]=0 [msong_holdout]=0 [sift1m]=0
  [glove25]=0 [StarLightCurves]=0 [dbpedia1536m_holdout]=0 [sift1b]=10000
)
declare -A NPROBES=(
  [nuswide]="1 2 3 5 8 16"
  [fasion_mnist_784]="1 3 5 7 10 20"
  [msong_holdout]="1 3 5 10 20 30 50 100"
  [sift1m]="1 5 10 20 30 50 70 100"
  [glove25]="1 5 10 20 30 50 70 100"
  [StarLightCurves]="1 2 3 5 7 10"
  # DBpedia first crosses recall 0.9 near nprobe=20.  Keep several points
  # above that threshold so the high-recall curves are actual curves rather
  # than a single point.
  [dbpedia1536m_holdout]="1 3 5 10 20 30 50 75 100 150 200"
  [sift1b]="32 64 128 256 512 1024"
)
declare -A STATS_NPROBES=(
  [nuswide]="3" [fasion_mnist_784]="7" [msong_holdout]="30" [sift1m]="50"
  [glove25]="50" [StarLightCurves]="5"
  # Detailed counters are needed at every high-recall DBpedia point to plot
  # overall pruning rate against recall.
  [dbpedia1536m_holdout]="20 30 50 75 100 150 200"
  [sift1b]="32 64 128 256 512 1024"
)
declare -A PCA_P=(
  [nuswide]=50 [fasion_mnist_784]=64 [msong_holdout]=42 [sift1m]=16
  [glove25]=8 [StarLightCurves]=103 [dbpedia1536m_holdout]=128 [sift1b]=16
)
declare -A PCA_SCOPE=(
  [nuswide]=global [fasion_mnist_784]=global [msong_holdout]=per_list
  [sift1m]=per_list [glove25]=per_list
  [StarLightCurves]=global [dbpedia1536m_holdout]=global
  [sift1b]=global
)
declare -A BEST_P=(
  [nuswide]=2 [fasion_mnist_784]=48 [msong_holdout]=34 [sift1m]=23
  [glove25]=14 [StarLightCurves]=18 [dbpedia1536m_holdout]=128 [sift1b]=16
)
declare -A BEST_SCOPE=(
  [nuswide]=per_list [fasion_mnist_784]=per_list [msong_holdout]=per_list
  [sift1m]=per_list [glove25]=global
  [StarLightCurves]=global [dbpedia1536m_holdout]=per_list
  [sift1b]=global
)
declare -A LOOPS=(
  [nuswide]=20 [fasion_mnist_784]=1 [msong_holdout]=1 [sift1m]=1
  [glove25]=1 [StarLightCurves]=20 [dbpedia1536m_holdout]=1 [sift1b]=1
)

declare -A EXTERNAL_GROUNDTRUTH=(
  [sift1b]="${SIFT1B_DIR}/gnd/idx_1000M.ivecs"
)

selected() {
  local needle="$1"
  [[ -z "${ONLY_DATASETS}" ]] && return 0
  local normalized=" ${ONLY_DATASETS//,/ } "
  [[ "${normalized}" == *" ${needle} "* ]]
}

if [[ -n "${ONLY_DATASETS}" ]]; then
  read -r -a requested_datasets <<<"${ONLY_DATASETS//,/ }"
  for requested in "${requested_datasets[@]}"; do
    known=0
    for dataset in "${ALL_DATASETS[@]}"; do
      [[ "${requested}" == "${dataset}" ]] && { known=1; break; }
    done
    ((known)) || { echo "Unknown dataset in --datasets: ${requested}" >&2; exit 2; }
  done
fi

quote_command() {
  printf '%q ' "$@"
  printf '\n'
}

run_bound() {
  if ((DRY_RUN)); then
    quote_command env -u TRIBASE_TRACE -u EDGE_DEVICE_ENABLED \
      OMP_NUM_THREADS="${THREADS}" OMP_PROC_BIND=close OMP_PLACES=cores \
      taskset -c "${CPU_SET}" "$@"
  else
    env -u TRIBASE_TRACE -u EDGE_DEVICE_ENABLED \
      OMP_NUM_THREADS="${THREADS}" OMP_PROC_BIND=close OMP_PLACES=cores \
      taskset -c "${CPU_SET}" "$@"
  fi
}

log() {
  local line="[$(date -Is)] $*"
  echo "${line}"
  ((DRY_RUN)) || echo "${line}" >>"${LOG_DIR}/run.log"
}

require_file() {
  ((DRY_RUN)) && return 0
  [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 1; }
}

csv_has_probes() {
  local path="$1" expected="$2"
  [[ -s "${path}" ]] || return 1
  python3 - "${path}" "${expected}" <<'PY'
import csv
import sys

path, expected_text = sys.argv[1:]
expected = {int(value) for value in expected_text.split()}
try:
    with open(path, newline="") as stream:
        found = {
            int(float(row["nprobe"]))
            for row in csv.DictReader(stream)
            if row.get("nprobe") not in (None, "")
        }
except (OSError, ValueError, KeyError, csv.Error):
    raise SystemExit(1)
raise SystemExit(0 if expected.issubset(found) else 1)
PY
}

csv_has_config() {
  local path="$1" expected_pivots="$2" expected_scope="$3"
  [[ -s "${path}" ]] || return 1
  python3 - "${path}" "${expected_pivots}" "${expected_scope}" <<'PY'
import csv
import sys

path, expected_pivots, expected_scope = sys.argv[1:]
try:
    with open(path, newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise SystemExit(1)
    if expected_pivots and any(
        int(float(row.get("pivot_count", -1))) != int(expected_pivots)
        for row in rows
    ):
        raise SystemExit(1)
    if expected_scope and any(
        row.get("multipivot_scope", "") != expected_scope for row in rows
    ):
        raise SystemExit(1)
except (OSError, ValueError, csv.Error):
    raise SystemExit(1)
PY
}

csv_has_high_recall_points() {
  local path="$1" minimum="$2"
  [[ -s "${path}" ]] || return 1
  python3 - "${path}" "${minimum}" <<'PY'
import csv
import sys

path, minimum = sys.argv[1], int(sys.argv[2])
try:
    with open(path, newline="") as stream:
        count = sum(float(row.get("recall", 0) or 0) > 0.9 for row in csv.DictReader(stream))
except (OSError, ValueError, csv.Error):
    raise SystemExit(1)
raise SystemExit(0 if count >= minimum else 1)
PY
}

run_float_mode() {
  local phase="$1" dataset="$2" mode="$3"; shift 3
  local binary output csv tmp logfile
  if [[ "${phase}" == perf ]]; then binary="${PERF_BIN}"; else binary="${STATS_BIN}"; fi
  output="${LOG_DIR}/${phase}/${dataset}"
  csv="${output}/${mode}.csv"
  tmp="${csv}.partial"
  logfile="${output}/${mode}.log"
  local -a probes
  local probes_text
  if [[ "${phase}" == perf ]]; then
    probes_text="${NPROBES[${dataset}]}"
  else
    probes_text="${STATS_NPROBES[${dataset}]}"
  fi
  read -r -a probes <<<"${probes_text}"
  local expected_pivots="" expected_scope="" argument_index
  local -a run_arguments=("$@")
  for ((argument_index=0; argument_index + 1 < ${#run_arguments[@]}; ++argument_index)); do
    case "${run_arguments[argument_index]}" in
      --pivot_counts) expected_pivots="${run_arguments[argument_index + 1]}" ;;
      --multipivot_scope) expected_scope="${run_arguments[argument_index + 1]}" ;;
    esac
  done
  if csv_has_probes "${csv}" "${probes_text}" &&
     csv_has_config "${csv}" "${expected_pivots}" "${expected_scope}"; then
    log "SKIP complete ${phase}/${dataset}/${mode}: ${csv}"
    return
  fi
  if [[ -s "${csv}" ]]; then
    log "RERUN incomplete ${phase}/${dataset}/${mode}; expected nprobes: ${probes_text}"
  fi
  ((DRY_RUN)) || { mkdir -p "${output}" "${LOG_DIR}/manifests"; rm -f "${tmp}"; }
  local loops="${LOOPS[${dataset}]}"
  [[ "${phase}" == stats ]] && loops=1
  local -a command=(
    "${binary}" --benchmarks_path "${DATA_ROOT}" --dataset "${dataset}"
    --input_format "${INPUT_FORMAT[${dataset}]}" --output_format bin --metric l2
    --k "${SEARCH_K[${dataset}]}" --nq "${QUERY_COUNT[${dataset}]}"
    --nlist "${NLIST[${dataset}]}" --nprobes "${probes[@]}" --cache
    --signature_precision float32 --loop "${loops}" --csv "${tmp}" "$@"
  )
  if [[ -n "${EXTERNAL_GROUNDTRUTH[${dataset}]:-}" ]]; then
    command+=(--groundtruth_path "${EXTERNAL_GROUNDTRUTH[${dataset}]}")
  fi
  ((VERBOSE)) && command+=(--verbose)
  if [[ "${INPUT_FORMAT[${dataset}]}" == bvecs ||
        "${INPUT_FORMAT[${dataset}]}" == i8bin ]]; then
    command+=(
      --build_batch_vectors "${BUILD_BATCH_VECTORS}"
      --coarse_builder ivf_hnsw
      --coarse_hnsw_m "${COARSE_HNSW_M}"
      --coarse_hnsw_ef_construction "${COARSE_HNSW_EF_CONSTRUCTION}"
      --coarse_hnsw_ef_search "${COARSE_HNSW_EF_SEARCH}"
      --centroids_path
      "${DATA_ROOT}/${dataset}/index/shared_centroids_ivfhnsw_nlist_${NLIST[${dataset}]}_M${COARSE_HNSW_M}_efc${COARSE_HNSW_EF_CONSTRUCTION}_efs${COARSE_HNSW_EF_SEARCH}_l2.bin"
    )
  fi
  log "START ${phase}/${dataset}/${mode} cpus=${CPU_SET} threads=${THREADS}"
  if ((DRY_RUN)); then
    run_bound "${command[@]}"
  else
    if run_bound "${command[@]}" >"${logfile}.partial" 2>&1; then
      mv "${tmp}" "${csv}"
      mv "${logfile}.partial" "${logfile}"
    else
      log "FAILED ${phase}/${dataset}/${mode}; see ${logfile}.partial"
      return 1
    fi
  fi
  log "DONE ${phase}/${dataset}/${mode}"
}

run_float_dataset() {
  local phase="$1" dataset="$2"
  local extension="${INPUT_FORMAT[${dataset}]}"
  local -a shared_ivf_args=()
  require_file "${DATA_ROOT}/${dataset}/origin/${dataset}_base.${extension}"
  require_file "${DATA_ROOT}/${dataset}/origin/${dataset}_query.${extension}"
  if [[ -n "${EXTERNAL_GROUNDTRUTH[${dataset}]:-}" ]]; then
    require_file "${EXTERNAL_GROUNDTRUTH[${dataset}]}"
  fi
  if [[ "${dataset}" == sift1b ]]; then
    # Keep one physical index for each billion-scale byte dataset. The global
    # The SIFT1B global PCA rich index (P=16) contains the full IVF
    # payload, Triangle radii and PCA signatures. Every search mode below loads
    # it read-only and selects only its runtime pruning path.
    local rich_index="${DATA_ROOT}/${dataset}/index/v10_nlist_${NLIST[${dataset}]}_metric_l2_opt_1_subk_15_subNprobeRatio_1_mp_${PCA_SCOPE[${dataset}]}_pca_P${PCA_P[${dataset}]}_seed0_coarse_ivfhnsw_M${COARSE_HNSW_M}_efc${COARSE_HNSW_EF_CONSTRUCTION}_efs${COARSE_HNSW_EF_SEARCH}.index"

    run_float_mode "${phase}" "${dataset}" pca10 \
      --opt_levels OPT_TRIANGLE --multipivot_modes projection \
      --multipivot_scope "${PCA_SCOPE[${dataset}]}" --multipivot_method pca \
      --pivot_counts "${PCA_P[${dataset}]}" --projection_block_size 0 --pivot_seed 0 \
      --pivot_manifest "${LOG_DIR}/manifests/${dataset}_pca10.csv"

    run_float_mode "${phase}" "${dataset}" baseline \
      --opt_levels OPT_NONE --multipivot_modes none --multipivot_scope global \
      --multipivot_method affine_fps --pivot_counts 0 --load_index "${rich_index}"
    run_float_mode "${phase}" "${dataset}" triangle \
      --opt_levels OPT_TRIANGLE --multipivot_modes none --multipivot_scope per_list \
      --multipivot_method pca --pivot_counts 1 --load_index "${rich_index}"

    local source="${LOG_DIR}/${phase}/${dataset}/pca10.csv"
    local target="${LOG_DIR}/${phase}/${dataset}/static_best.csv"
    local probes_text
    if [[ "${phase}" == perf ]]; then
      probes_text="${NPROBES[${dataset}]}"
    else
      probes_text="${STATS_NPROBES[${dataset}]}"
    fi
    if ((DRY_RUN)); then
      echo "reuse ${source} as ${target}"
    elif ! csv_has_probes "${target}" "${probes_text}"; then
      cp "${source}" "${target}"
    fi

    run_float_mode "${phase}" "${dataset}" dynamic \
      --opt_levels OPT_TRIANGLE --multipivot_modes projection \
      --multipivot_scope "${PCA_SCOPE[${dataset}]}" --multipivot_method pca \
      --pivot_counts "${PCA_P[${dataset}]}" --projection_block_size 2 \
      --projection_dynamic --pivot_seed 0 \
      --pivot_manifest "${LOG_DIR}/manifests/${dataset}_dynamic.csv" \
      --load_index "${rich_index}"
    return
  fi
  run_float_mode "${phase}" "${dataset}" baseline \
    --opt_levels OPT_NONE --multipivot_modes none --multipivot_scope global \
    --multipivot_method affine_fps --pivot_counts 0
  if [[ "${extension}" == bvecs || "${extension}" == i8bin ]]; then
    # The billion-scale coarse assignment is the expensive operation. Upgrade
    # every pruning variant from this exact baseline IVF so all methods share
    # identical centroids, list membership, IDs and vector order.
    local baseline_index="${DATA_ROOT}/${dataset}/index/v10_nlist_${NLIST[${dataset}]}_metric_l2_opt_0_subk_15_subNprobeRatio_1_mp_global_affine_fps_P0_seed0_coarse_ivfhnsw_M${COARSE_HNSW_M}_efc${COARSE_HNSW_EF_CONSTRUCTION}_efs${COARSE_HNSW_EF_SEARCH}.index"
    shared_ivf_args=(--from_index "${baseline_index}")
  fi
  run_float_mode "${phase}" "${dataset}" triangle \
    --opt_levels OPT_TRIANGLE --multipivot_modes none --multipivot_scope per_list \
    --multipivot_method pca --pivot_counts 1 "${shared_ivf_args[@]}"
  run_float_mode "${phase}" "${dataset}" pca10 \
    --opt_levels OPT_TRIANGLE --multipivot_modes projection \
    --multipivot_scope "${PCA_SCOPE[${dataset}]}" --multipivot_method pca \
    --pivot_counts "${PCA_P[${dataset}]}" --projection_block_size 0 --pivot_seed 0 \
    --pivot_manifest "${LOG_DIR}/manifests/${dataset}_pca10.csv" \
    "${shared_ivf_args[@]}"
  if [[ "${BEST_P[${dataset}]}" == "${PCA_P[${dataset}]}" &&
        "${BEST_SCOPE[${dataset}]}" == "${PCA_SCOPE[${dataset}]}" ]]; then
    local source="${LOG_DIR}/${phase}/${dataset}/pca10.csv"
    local target="${LOG_DIR}/${phase}/${dataset}/static_best.csv"
    local probes_text
    if [[ "${phase}" == perf ]]; then
      probes_text="${NPROBES[${dataset}]}"
    else
      probes_text="${STATS_NPROBES[${dataset}]}"
    fi
    if ((DRY_RUN)); then
      echo "reuse ${source} as ${target}"
    elif ! csv_has_probes "${target}" "${probes_text}"; then
      cp "${source}" "${target}"
    fi
  else
    run_float_mode "${phase}" "${dataset}" static_best \
      --opt_levels OPT_TRIANGLE --multipivot_modes projection \
      --multipivot_scope "${BEST_SCOPE[${dataset}]}" --multipivot_method pca \
      --pivot_counts "${BEST_P[${dataset}]}" --projection_block_size 0 --pivot_seed 0 \
      --pivot_manifest "${LOG_DIR}/manifests/${dataset}_static_best.csv" \
      "${shared_ivf_args[@]}"
  fi
  run_float_mode "${phase}" "${dataset}" dynamic \
    --opt_levels OPT_TRIANGLE --multipivot_modes projection \
    --multipivot_scope "${PCA_SCOPE[${dataset}]}" --multipivot_method pca \
    --pivot_counts "${PCA_P[${dataset}]}" --projection_block_size 2 \
    --projection_dynamic --pivot_seed 0 \
    --pivot_manifest "${LOG_DIR}/manifests/${dataset}_dynamic.csv" \
    "${shared_ivf_args[@]}"
}

summarize() {
  ((DRY_RUN)) && return
  python3 - "${LOG_DIR}" <<'PY'
import csv
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
for phase in ("perf", "stats"):
    rows = []
    for path in sorted((root / phase).glob("*/*.csv")):
        if path.name.endswith(".partial"):
            continue
        dataset = path.parent.name
        mode_from_file = path.stem
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                row["dataset"] = dataset
                row["experiment"] = row.get("mode") or mode_from_file
                qps = float(row.get("qps", 0) or 0)
                if not row.get("latency_ms"):
                    row["latency_ms"] = 1000.0 / qps if qps else 0.0
                if row.get("prune_rate") not in (None, ""):
                    row["overall_prune_rate"] = row["prune_rate"]
                else:
                    def value(name):
                        try:
                            return float(row.get(name, 0) or 0)
                        except ValueError:
                            return 0.0
                    exact = value("candidate_distance_computations")
                    pruned = (
                        value("tri") + value("tri_large")
                        + value("multipivot_pruned")
                        + value("microblock_vectors_pruned")
                    )
                    row["overall_prune_rate"] = pruned / (exact + pruned) if exact + pruned else 0.0
                rows.append(row)
    if not rows:
        continue
    preferred = [
        "dataset", "experiment", "nprobe", "recall", "qps", "latency_ms",
        "overall_prune_rate",
    ]
    fields = preferred + sorted({key for row in rows for key in row} - set(preferred))
    with (root / f"{phase}_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {root / f'{phase}_summary.csv'} rows={len(rows)}")
PY
}

validate_outputs() {
  ((DRY_RUN)) && return
  local dataset phase mode probes path failed=0
  for dataset in "${FLOAT_DATASETS[@]}"; do
    selected "${dataset}" || continue
    for phase in perf stats; do
      if [[ "${PHASE}" == perf && "${phase}" == stats ]] ||
         [[ "${PHASE}" == stats && "${phase}" == perf ]]; then
        continue
      fi
      if [[ "${phase}" == perf ]]; then
        probes="${NPROBES[${dataset}]}"
      else
        probes="${STATS_NPROBES[${dataset}]}"
      fi
      for mode in baseline triangle pca10 static_best dynamic; do
        path="${LOG_DIR}/${phase}/${dataset}/${mode}.csv"
        if ! csv_has_probes "${path}" "${probes}"; then
          echo "INCOMPLETE: ${path}; required nprobes: ${probes}" >&2
          failed=1
        fi
        if [[ "${dataset}" == dbpedia1536m_holdout ]] &&
           ! csv_has_high_recall_points "${path}" 5; then
          echo "INCOMPLETE: ${path}; fewer than five recall > 0.9 points" >&2
          failed=1
        fi
      done
    done
  done
  ((failed == 0)) || return 1
  log "VALIDATION complete: every requested mode and nprobe is present"
}

if ((!DRY_RUN)); then
  mkdir -p "${LOG_DIR}"
  if [[ "${PHASE}" == all || "${PHASE}" == perf ]]; then
    for dataset in "${FLOAT_DATASETS[@]}"; do
      if selected "${dataset}"; then require_file "${PERF_BIN}"; break; fi
    done
  fi
  if [[ "${PHASE}" == all || "${PHASE}" == stats ]]; then
    for dataset in "${FLOAT_DATASETS[@]}"; do
      if selected "${dataset}"; then require_file "${STATS_BIN}"; break; fi
    done
  fi
fi

log "CONFIG data_root=${DATA_ROOT} log_dir=${LOG_DIR} cpus=${CPU_SET} threads=${THREADS} phase=${PHASE} large_vectors=query-float-index"
for dataset in "${FLOAT_DATASETS[@]}"; do
  selected "${dataset}" || continue
  if [[ "${PHASE}" == all || "${PHASE}" == perf ]]; then
    run_float_dataset perf "${dataset}"
  fi
  if [[ "${PHASE}" == all || "${PHASE}" == stats ]]; then
    run_float_dataset stats "${dataset}"
  fi
done

summarize
validate_outputs
log "COMPLETE output=${LOG_DIR}"
