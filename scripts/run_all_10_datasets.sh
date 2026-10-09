#!/usr/bin/env bash
# Run all ten datasets through the shared native float Index/IVF pipeline.
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
SPACEV_DIR="${SPACEV_DIR:-}"
SIFT1B_INDEX="${SIFT1B_INDEX:-}"
SPACEV_INDEX="${SPACEV_INDEX:-}"
COARSE_HNSW_QUERY_BATCH="${COARSE_HNSW_QUERY_BATCH:-8192}"
PERF_REPEATS="${PERF_REPEATS:-3}"
WARMUP_LOOPS="${WARMUP_LOOPS:-1}"
PIVOT_SEED="${PIVOT_SEED:-0}"
OMP_WAIT_POLICY="${OMP_WAIT_POLICY:-PASSIVE}"
RESULT_HELPER="${ROOT}/scripts/recall_sweep_results.py"
CONFIG_FILE="${CONFIG_FILE:-}"
DRY_RUN=0
VERBOSE=0

usage() {
  cat <<'EOF'
Usage:
  scripts/run_all_10_datasets.sh --data-root DIR --log-dir DIR [options]

Required:
  --data-root DIR       Root containing the ten dataset directories.
  --log-dir DIR         Output root for CSV files, logs and manifests.

CPU control:
  --cpus LIST           Linux CPU list passed to taskset, e.g. 0-31 or 0-15,32-47.
  --threads N           OpenMP/worker threads; defaults to the number of bound CPUs.
  --build-batch-vectors N
                        Decoded vectors retained per SIFT1B/SpaceV build batch
                        (default 262144, about 128 MiB at D=128).
  --coarse-hnsw-m N     HNSW coarse graph degree for SIFT1B/SpaceV (default 32).
  --coarse-hnsw-ef-construction N
                        HNSW construction breadth (default 200).
  --coarse-hnsw-ef-search N
                        HNSW centroid-assignment breadth (default 128).

Selection:
  --coarse-hnsw-query-batch N
                        Maximum vectors in each HNSW assignment call (default 8192).
  --perf-bin PATH       Stats-disabled query binary.
  --stats-bin PATH      Stats-enabled query binary.
  --phase NAME          all (default), perf, or stats.
  --perf-repeats N      Independent process runs; default 3, median search time.
  --warmup-loops N      Untimed passes for each nprobe/method; default 1.
  --pivot-seed N        PCA seed; default 0 (reuses the SIFT1B PCA16 index).
  --omp-wait-policy P   PASSIVE (default) or ACTIVE, recorded with each run.
  --datasets LIST       Subset of the ten dataset names.
  --config FILE         Validated JSON profiles for a separate supplement.
  --sift1b-dir DIR      Override DATA_ROOT/sift1b/raw; link inputs if origin is absent.
  --spacev-dir DIR      Override DATA_ROOT/spacev1b/raw; link inputs if origin is absent.
  --sift1b-index FILE   Reuse an existing native global PCA index (P >= 16).
  --spacev-index FILE   Reuse an existing native global PCA index (P >= 11).
  --verbose             Pass --verbose to query and retain detailed build logs.
  --dry-run             Validate arguments and print commands without running them.
  -h, --help            Show this help.

Expected common layout:
  DATA_ROOT/<dataset>/origin/<dataset>_base.fvecs
  DATA_ROOT/<dataset>/origin/<dataset>_query.fvecs
  DATA_ROOT/<dataset>/result/groundtruth_1.bin

Large inputs remain byte-coded on disk and are decoded to float when query.cpp
loads them. Their indexes use the same float Index/IVF format as other data.

The script creates/reuses indexes inside each dataset directory.
Each method shares one IVF partitioning; PCA methods use cached rich indexes.
Perf saves raw repetitions and their median; stats covers every sweep nprobe.
Only matching search_only_v2 measurements are resumed; old results are rerun.
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
    --coarse-hnsw-query-batch) COARSE_HNSW_QUERY_BATCH="$2"; shift 2 ;;
    --perf-bin) PERF_BIN="$2"; shift 2 ;;
    --stats-bin) STATS_BIN="$2"; shift 2 ;;
    --phase) PHASE="$2"; shift 2 ;;
    --perf-repeats) PERF_REPEATS="$2"; shift 2 ;;
    --warmup-loops) WARMUP_LOOPS="$2"; shift 2 ;;
    --pivot-seed) PIVOT_SEED="$2"; shift 2 ;;
    --omp-wait-policy) OMP_WAIT_POLICY="$2"; shift 2 ;;
    --datasets) ONLY_DATASETS="$2"; shift 2 ;;
    --config) CONFIG_FILE="$2"; shift 2 ;;
    --sift1b-dir) SIFT1B_DIR="$2"; shift 2 ;;
    --spacev-dir) SPACEV_DIR="$2"; shift 2 ;;
    --sift1b-index) SIFT1B_INDEX="$2"; shift 2 ;;
    --spacev-index) SPACEV_INDEX="$2"; shift 2 ;;
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
for hnsw_value in "${COARSE_HNSW_QUERY_BATCH}" "${COARSE_HNSW_M}" "${COARSE_HNSW_EF_CONSTRUCTION}" \
                  "${COARSE_HNSW_EF_SEARCH}"; do
  [[ "${hnsw_value}" =~ ^[1-9][0-9]*$ ]] || {
    echo "invalid HNSW coarse parameter: ${hnsw_value}" >&2; exit 2;
  }
done

for value in "$PERF_REPEATS" "$WARMUP_LOOPS"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "repeats/warmups must be positive" >&2; exit 2; }
done
[[ "$PIVOT_SEED" =~ ^[0-9]+$ ]] || { echo "invalid --pivot-seed" >&2; exit 2; }
[[ "$OMP_WAIT_POLICY" =~ ^(PASSIVE|ACTIVE)$ ]] || { echo "invalid --omp-wait-policy" >&2; exit 2; }
SIFT1B_DIR="${SIFT1B_DIR:-${DATA_ROOT}/sift1b/raw}"
SPACEV_DIR="${SPACEV_DIR:-${DATA_ROOT}/spacev1b/raw}"
ALL_DATASETS=(
  nuswide fasion_mnist_784 msong_holdout sift1m glove25
  HandOutlines StarLightCurves dbpedia1536m_holdout sift1b spacev1b
)
FLOAT_DATASETS=("${ALL_DATASETS[@]}")

declare -A INPUT_FORMAT=(
  [nuswide]=fvecs [fasion_mnist_784]=fvecs [msong_holdout]=fvecs [sift1m]=fvecs
  [glove25]=fvecs [HandOutlines]=fvecs [StarLightCurves]=fvecs
  [dbpedia1536m_holdout]=fvecs [sift1b]=bvecs [spacev1b]=i8bin
)

declare -A NLIST=(
  [nuswide]=512 [fasion_mnist_784]=256 [msong_holdout]=1000 [sift1m]=1000
  [glove25]=1024 [HandOutlines]=32 [StarLightCurves]=128
  [dbpedia1536m_holdout]=1000 [sift1b]=32768 [spacev1b]=32768
)
declare -A SEARCH_K=(
  [nuswide]=1 [fasion_mnist_784]=1 [msong_holdout]=1 [sift1m]=1
  [glove25]=1 [HandOutlines]=1 [StarLightCurves]=1 [dbpedia1536m_holdout]=1 [sift1b]=10 [spacev1b]=10
)
declare -A QUERY_COUNT=(
  [nuswide]=0 [fasion_mnist_784]=0 [msong_holdout]=0 [sift1m]=0
  [glove25]=0 [HandOutlines]=0 [StarLightCurves]=0 [dbpedia1536m_holdout]=0 [sift1b]=10000 [spacev1b]=29316
)
declare -A NPROBES=(
  [nuswide]="1 2 3 5 8 16"
  [fasion_mnist_784]="1 3 5 7 10 20"
  [msong_holdout]="1 3 5 10 16 20 30 50 100"
  [sift1m]="1 5 10 20 30 34 50 70 100"
  [glove25]="1 5 10 20 30 50 70 100"
  [HandOutlines]="1 2 3 4 5 7 10"
  [StarLightCurves]="1 2 3 4 5 7 10"
  # DBpedia first crosses recall 0.9 near nprobe=20.  Keep several points
  # above that threshold so the high-recall curves are actual curves rather
  # than a single point.
  [dbpedia1536m_holdout]="1 3 5 10 20 30 50 75 100 150 160 200"
  [sift1b]="32 64 128 256 512 1024"
  [spacev1b]="8 16 32 64 128 256 512 1024"
)
declare -A PCA_P=(
  [nuswide]=50 [fasion_mnist_784]=64 [msong_holdout]=42 [sift1m]=16
  [glove25]=8 [HandOutlines]=271 [StarLightCurves]=103 [dbpedia1536m_holdout]=128 [sift1b]=16 [spacev1b]=11
)
declare -A PCA_SCOPE=(
  [nuswide]=global [fasion_mnist_784]=global [msong_holdout]=per_list
  [sift1m]=per_list [glove25]=per_list [HandOutlines]=global
  [StarLightCurves]=global [dbpedia1536m_holdout]=global
  [sift1b]=global [spacev1b]=global
)
declare -A BEST_P=(
  [nuswide]=2 [fasion_mnist_784]=48 [msong_holdout]=34 [sift1m]=23
  [glove25]=14 [HandOutlines]=7 [StarLightCurves]=18 [dbpedia1536m_holdout]=128 [sift1b]=16 [spacev1b]=11
)
declare -A BEST_SCOPE=(
  [nuswide]=per_list [fasion_mnist_784]=per_list [msong_holdout]=per_list
  [sift1m]=per_list [glove25]=global [HandOutlines]=global
  [StarLightCurves]=global [dbpedia1536m_holdout]=per_list
  [sift1b]=global [spacev1b]=global
)
# Dynamic is an independent comparison, not a truncation of PCA10%.
declare -A DYNAMIC_PMAX=(
  [nuswide]=32 [fasion_mnist_784]=96 [msong_holdout]=96 [sift1m]=40
  [glove25]=26 [HandOutlines]=64 [StarLightCurves]=104 [dbpedia1536m_holdout]=256 [sift1b]=16 [spacev1b]=11
)
declare -A DYNAMIC_SCOPE=(
  [nuswide]=per_list [fasion_mnist_784]=per_list [msong_holdout]=per_list
  [sift1m]=per_list [glove25]=global [HandOutlines]=global [StarLightCurves]=global
  [dbpedia1536m_holdout]=per_list [sift1b]=global [spacev1b]=global
)
declare -A LOOPS=(
  [nuswide]=50 [fasion_mnist_784]=10 [msong_holdout]=5 [sift1m]=2
  [glove25]=10 [HandOutlines]=200 [StarLightCurves]=50 [dbpedia1536m_holdout]=1 [sift1b]=1 [spacev1b]=1
)

declare -A EXTERNAL_GROUNDTRUTH=(
  [sift1b]="${SIFT1B_DIR}/gnd/idx_1000M.ivecs"
  [spacev1b]="${SPACEV_DIR}/groundtruth.30K.i32bin"
)
declare -A DIMENSION
declare -A NORMALIZE SOURCE_METRIC
if [[ -n "$CONFIG_FILE" ]]; then
  config_lines="$(python3 "$RESULT_HELPER" config "$CONFIG_FILE" --data-root "$DATA_ROOT")"
  ALL_DATASETS=()
  previous_dataset=""
  while IFS=$'\t' read -r dataset key value; do
    if [[ "$dataset" != "$previous_dataset" ]]; then
      ALL_DATASETS+=("$dataset"); previous_dataset="$dataset"; INPUT_FORMAT["$dataset"]=fvecs
    fi
    declare -n config_array="$key"
    config_array["$dataset"]="$value"
  done <<< "$config_lines"
  unset -n config_array
fi

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

quote_command() { printf '%q ' "$@"; printf '\n'; }
log() {
  echo "[$(date -Is)] $*"
  ((DRY_RUN)) || echo "[$(date -Is)] $*" >> "$LOG_DIR/run.log"
}
require_file() {
  ((DRY_RUN)) && return 0
  [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; return 1; }
}
link_raw_input() {
  local source="$1" target="$2"
  [[ -e "$target" ]] && return
  if ((DRY_RUN)); then quote_command ln -s "$source" "$target"; return; fi
  require_file "$source"
  [[ ! -L "$target" ]] || { echo "Broken input symlink: $target" >&2; return 1; }
  mkdir -p "$(dirname "$target")"
  # Absolute symlink targets also work when data-root/raw-dir are relative.
  ln -s "$(realpath "$source")" "$target"
}
validate_input() {
  local dataset="$1" extension="${INPUT_FORMAT[$1]}" origin="$DATA_ROOT/$1/origin" info nb nq
  if [[ "$dataset" == sift1b ]]; then
    link_raw_input "$SIFT1B_DIR/bigann_base.bvecs" "$origin/${dataset}_base.bvecs"
    link_raw_input "$SIFT1B_DIR/bigann_query.bvecs" "$origin/${dataset}_query.bvecs"
  elif [[ "$dataset" == spacev1b ]]; then
    link_raw_input "$SPACEV_DIR/base.1B.i8bin" "$origin/${dataset}_base.i8bin"
    link_raw_input "$SPACEV_DIR/query.30K.i8bin" "$origin/${dataset}_query.i8bin"
  fi
  require_file "$origin/${dataset}_base.$extension"
  require_file "$origin/${dataset}_query.$extension"
  [[ -z "${EXTERNAL_GROUNDTRUTH[$dataset]:-}" ]] || require_file "${EXTERNAL_GROUNDTRUTH[$dataset]}"
  if ((DRY_RUN)); then return; fi
  info="$(python3 "$RESULT_HELPER" input-info "$origin/${dataset}_base.$extension" \
    "$origin/${dataset}_query.$extension" --nq "${QUERY_COUNT[$dataset]}")"
  local dimension requested="${QUERY_COUNT[$dataset]}"
  read -r nb dimension nq <<< "$info"
  DIMENSION[$dataset]="$dimension"
  ((requested > 0)) || requested="$nq"
  local label_status=not_applicable static_origin=fine_sweep
  [[ -z "$CONFIG_FILE" ]] || static_origin=reference_not_fine_sweep_optimum
  [[ "$dataset" != spacev1b ]] || static_origin=reference_P11_not_fine_sweep_optimum
  if [[ "$dataset" == HandOutlines || "$dataset" == StarLightCurves ]]; then
    label_status=check_source_format
    if [[ "$dataset" == HandOutlines && "${DIMENSION[$dataset]}" == 2710 ]] ||
       [[ "$dataset" == StarLightCurves && "${DIMENSION[$dataset]}" == 1025 ]]; then
      label_status=suspected_label_coordinate_preserved
      log "INPUT $dataset D=${DIMENSION[$dataset]} appears to include a label coordinate; using the existing input/groundtruth as supplied"
    fi
  fi
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' "$dataset" "$nb" "${DIMENSION[$dataset]}" "$nq" \
    "$requested" "${NLIST[$dataset]}" "$label_status" "$static_origin" "${SOURCE_METRIC[$dataset]:-l2}" \
    "${NORMALIZE[$dataset]:-0}" >> "$LOG_DIR/input_manifest.csv"
}
run_bound() {
  local -a command=(env -u TRIBASE_TRACE -u EDGE_DEVICE_ENABLED
    OMP_NUM_THREADS="$THREADS" OMP_PROC_BIND=close OMP_PLACES=cores
    OMP_DYNAMIC=FALSE OMP_WAIT_POLICY="$OMP_WAIT_POLICY"
    MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    taskset -c "$CPU_SET" "$@")
  if ((DRY_RUN)); then quote_command "${command[@]}"; else "${command[@]}"; fi
}
snapshot() {
  date -Is
  cat /proc/loadavg
  free -b
  cat /proc/self/cgroup
  # Support both cgroup v1 and v2; unavailable counters remain explicit.
  local group
  group="$(awk -F: '$1=="0" || $2=="memory" {print $3; exit}' /proc/self/cgroup)"
  local file
  for file in /sys/fs/cgroup"${group}"/memory.events \
              /sys/fs/cgroup"${group}"/memory.current \
              /sys/fs/cgroup"${group}"/memory.max \
              /sys/fs/cgroup/memory"${group}"/memory.limit_in_bytes \
              /sys/fs/cgroup/memory"${group}"/memory.max_usage_in_bytes \
              /sys/fs/cgroup/memory"${group}"/memory.failcnt \
              /sys/fs/cgroup/memory"${group}"/memory.oom_control; do
    if [[ -r "$file" ]]; then echo "$file"; cat "$file"; fi
  done
}
execute_logged() {
  local binary="$1" logpath="$2"; shift 2
  if ((DRY_RUN)); then run_bound "$binary" "$@"; return; fi
  mkdir -p "$(dirname "$logpath")"
  quote_command "$binary" "$@" > "${logpath}.command"
  snapshot > "${logpath}.resources"
  local status=0
  if [[ -x /usr/bin/time ]]; then
    run_bound /usr/bin/time -v -o "${logpath}.time" "$binary" "$@" > "${logpath}.partial" 2>&1 || status=$?
  else
    run_bound "$binary" "$@" > "${logpath}.partial" 2>&1 || status=$?
  fi
  snapshot >> "${logpath}.resources"
  if ((status)); then
    log "FAILED status=$status see ${logpath}.partial and ${logpath}.resources"
    return "$status"
  fi
  mv "${logpath}.partial" "$logpath"
}

# Common input/search parameters, reused by construction and every measurement.
common_arguments() {
  local dataset="$1"
  COMMON=(--benchmarks_path "$DATA_ROOT" --dataset "$dataset"
    --input_format "${INPUT_FORMAT[$dataset]}" --output_format bin --metric l2
    --k "${SEARCH_K[$dataset]}" --nq "${QUERY_COUNT[$dataset]}"
    --nlist "${NLIST[$dataset]}" --signature_precision float32 --pivot_seed "$PIVOT_SEED")
  [[ "${NORMALIZE[$dataset]:-0}" != 1 ]] || COMMON+=(--normalize)
  if [[ -n "${EXTERNAL_GROUNDTRUTH[$dataset]:-}" ]]; then
    COMMON+=(--groundtruth_path "${EXTERNAL_GROUNDTRUTH[$dataset]}")
  fi
  if [[ "$dataset" == sift1b || "$dataset" == spacev1b ]]; then
    COMMON+=(--build_batch_vectors "$BUILD_BATCH_VECTORS" --coarse_builder ivf_hnsw
      --coarse_hnsw_m "$COARSE_HNSW_M" --coarse_hnsw_ef_construction "$COARSE_HNSW_EF_CONSTRUCTION"
      --coarse_hnsw_ef_search "$COARSE_HNSW_EF_SEARCH"
      --coarse_hnsw_query_batch "$COARSE_HNSW_QUERY_BATCH"
      --centroids_path "$DATA_ROOT/$dataset/index/shared_centroids_ivfhnsw_nlist_${NLIST[$dataset]}_M${COARSE_HNSW_M}_efc${COARSE_HNSW_EF_CONSTRUCTION}_efs${COARSE_HNSW_EF_SEARCH}_l2.bin")
  fi
  ((VERBOSE)) && COMMON+=(--verbose)
  return 0
}

# Every regular dataset is derived from one Triangle IVF. Old independent PCA
# caches are deliberately bypassed via a distinct output/cache path.
build_index() {
  local dataset="$1" label="$2" target="$3"; shift 3
  local -a arguments=("${COMMON[@]}" --nprobes 1 --loop 1 --warmup_loops "$WARMUP_LOOPS"
    --train_only --index_path "$target" "$@")
  local recipe source source_state="" i
  local -a geometry=()
  for ((i=0; i+1<${#arguments[@]}; ++i)); do
    if [[ "${arguments[i]}" == --from_index ]]; then
      source="${arguments[i+1]}"
      if ((!DRY_RUN)); then source_state="$(stat -c '%n:%s:%Y' "$source")"; fi
    fi
  done
  # Log directories and measurement settings do not change index geometry.
  for ((i=0; i<${#arguments[@]}; ++i)); do
    case "${arguments[i]}" in
      --pivot_manifest|--warmup_loops|--loop|--nprobes|--groundtruth_path|--k|--nq) i=$((i+1)) ;;
      --verbose) ;;
      *) geometry+=("${arguments[i]}") ;;
    esac
  done
  local input_state=""
  if [[ -n "$CONFIG_FILE" ]] && ((!DRY_RUN)); then
    input_state="$(stat -c '%n:%s:%y' "$DATA_ROOT/$dataset/origin/${dataset}_base.${INPUT_FORMAT[$dataset]}")"
  fi
  recipe="$(quote_command "${geometry[@]}") $source_state"
  [[ -z "$input_state" ]] || recipe+=" $input_state"
  if ((!DRY_RUN)) && [[ -s "$target" && -f "${target}.origin" ]] &&
     [[ "$(cat "${target}.origin")" == "$recipe" ]]; then
    log "REUSE shared index $target"
    return
  fi
  log "PREPARE $dataset/$label index=$target"
  execute_logged "$BUILD_BIN" "$LOG_DIR/build/$dataset/$label.log" "${arguments[@]}"
  if ((!DRY_RUN)); then
    require_file "$target"
    printf '%s\n' "$recipe" > "${target}.origin"
  fi
}

declare -A RICH_INDEX RICH_P
prepare_dataset() {
  local dataset="$1" extension="${INPUT_FORMAT[$1]}" scope p pmax root
  require_file "$DATA_ROOT/$dataset/origin/${dataset}_base.$extension"
  require_file "$DATA_ROOT/$dataset/origin/${dataset}_query.$extension"
  [[ -z "${EXTERNAL_GROUNDTRUTH[$dataset]:-}" ]] || require_file "${EXTERNAL_GROUNDTRUTH[$dataset]}"
  common_arguments "$dataset"
  if [[ "$dataset" == sift1b || "$dataset" == spacev1b ]]; then
    # Keep a single native rich index for each large dataset. No int8-special
    # index is loaded by the float query.cpp workflow.
    local required="${PCA_P[$dataset]}" physical_count override=""
    p="${DYNAMIC_PMAX[$dataset]}"; ((p<=required)) || required="$p"
    p="${BEST_P[$dataset]}"; ((p<=required)) || required="$p"
    SHARED_TRIANGLE="$DATA_ROOT/$dataset/index/v10_nlist_${NLIST[$dataset]}_metric_l2_opt_1_subk_15_subNprobeRatio_1_mp_global_pca_P${required}_seed${PIVOT_SEED}_coarse_ivfhnsw_M${COARSE_HNSW_M}_efc${COARSE_HNSW_EF_CONSTRUCTION}_efs${COARSE_HNSW_EF_SEARCH}.index"
    [[ "$dataset" != sift1b ]] || override="$SIFT1B_INDEX"
    [[ "$dataset" != spacev1b ]] || override="$SPACEV_INDEX"
    [[ -z "$override" ]] || SHARED_TRIANGLE="$override"
    physical_count="$required"
    if ((!DRY_RUN)) && [[ -f "$SHARED_TRIANGLE" || -n "$override" ]]; then
      # Query validates the full file when loading. This cheap header check
      # rejects incompatible byte indexes before a multi-hundred-GB allocation.
      if ! physical_count="$(python3 "$RESULT_HELPER" index-info "$SHARED_TRIANGLE" \
        --dimension "${DIMENSION[$dataset]}" --nlist "${NLIST[$dataset]}" \
        --minimum-pivots "$required" --seed "$PIVOT_SEED")"; then
        [[ -z "$override" ]] || return 1
        log "PRESERVE incompatible cache $SHARED_TRIANGLE; use a separate native float index"
        SHARED_TRIANGLE="${SHARED_TRIANGLE%.index}_float_ram.index"
        physical_count="$required"
        if [[ -f "$SHARED_TRIANGLE" ]]; then
          physical_count="$(python3 "$RESULT_HELPER" index-info "$SHARED_TRIANGLE" \
            --dimension "${DIMENSION[$dataset]}" --nlist "${NLIST[$dataset]}" \
            --minimum-pivots "$required" --seed "$PIVOT_SEED")"
        fi
      fi
    fi
    log "PREPARE/REUSE $dataset/PCA${physical_count} index=$SHARED_TRIANGLE"
    local -a load=(--cache --index_path "$SHARED_TRIANGLE")
    [[ -z "$override" ]] || load=(--load_index "$SHARED_TRIANGLE")
    if [[ -z "$override" && ! -f "$SHARED_TRIANGLE" ]]; then
      local coarse_tag="_coarse_ivfhnsw_M${COARSE_HNSW_M}_efc${COARSE_HNSW_EF_CONSTRUCTION}_efs${COARSE_HNSW_EF_SEARCH}"
      local cached
      for cached in \
        "$DATA_ROOT/$dataset/index/v10_nlist_${NLIST[$dataset]}_metric_l2_opt_0_subk_15_subNprobeRatio_1_mp_global_affine_fps_P0_seed0${coarse_tag}.index" \
        "$DATA_ROOT/$dataset/index/v10_nlist_${NLIST[$dataset]}_metric_l2_opt_1_subk_15_subNprobeRatio_1_mp_per_list_pca_P0_seed0${coarse_tag}.index"; do
        if [[ -f "$cached" ]] && { ((DRY_RUN)) || python3 "$RESULT_HELPER" index-info "$cached" \
            --dimension "${DIMENSION[$dataset]}" --nlist "${NLIST[$dataset]}" >/dev/null; }; then
          load+=(--from_index "$cached")
          log "UPGRADE existing large IVF $cached"
          break
        fi
      done
    fi
    execute_logged "$BUILD_BIN" "$LOG_DIR/build/$dataset/pca${physical_count}.log" "${COMMON[@]}" \
      --nprobes 32 --train_only \
      "${load[@]}" \
      --opt_levels OPT_TRIANGLE --multipivot_modes projection --multipivot_scope global \
      --multipivot_method pca --pivot_counts "$physical_count"
    RICH_INDEX[global]="$SHARED_TRIANGLE"
    RICH_P[global]="$physical_count"
    return
  fi
  local unit_tag=""
  [[ "${NORMALIZE[$dataset]:-0}" != 1 ]] || unit_tag=_unit
  root="$DATA_ROOT/$dataset/index/recall_shared_v2_nlist_${NLIST[$dataset]}_seed${PIVOT_SEED}${unit_tag}"
  SHARED_TRIANGLE="${root}_triangle.index"
  local baseline="$DATA_ROOT/$dataset/index/v10_nlist_${NLIST[$dataset]}_metric_l2_opt_0_subk_15_subNprobeRatio_1_mp_global_affine_fps_P0_seed0${unit_tag}.index"
  local -a source=()
  if [[ ! -f "$baseline" ]]; then
    baseline="$DATA_ROOT/$dataset/index/v10_nlist_${NLIST[$dataset]}_metric_l2_opt_1_subk_15_subNprobeRatio_1_mp_per_list_pca_P0_seed0${unit_tag}.index"
  fi
  if [[ -n "$CONFIG_FILE" && -f "$baseline" && "$baseline" -ot "$DATA_ROOT/$dataset/origin/${dataset}_base.${INPUT_FORMAT[$dataset]}" ]]; then
    log "SKIP base IVF older than the exported corpus: $baseline"
    baseline=""
  fi
  if [[ -f "$baseline" ]] && { ((DRY_RUN)) || python3 "$RESULT_HELPER" index-info "$baseline" \
      --dimension "${DIMENSION[$dataset]}" --nlist "${NLIST[$dataset]}" >/dev/null; }; then
    source=(--from_index "$baseline")
  fi
  if ((!DRY_RUN)) && [[ -f "$SHARED_TRIANGLE" ]] && ! python3 "$RESULT_HELPER" index-info "$SHARED_TRIANGLE" \
      --dimension "${DIMENSION[$dataset]}" --nlist "${NLIST[$dataset]}" >/dev/null 2>&1; then
    # Preserve caches made before e.g. stripping a UCR label coordinate.
    root="${root}_d${DIMENSION[$dataset]}"
    SHARED_TRIANGLE="${root}_triangle.index"
  fi
  build_index "$dataset" triangle "$SHARED_TRIANGLE" \
    --opt_levels OPT_TRIANGLE --multipivot_modes none --multipivot_scope per_list \
    --multipivot_method pca --pivot_counts 1 "${source[@]}"
  for scope in global per_list; do
    pmax=0
    [[ "${PCA_SCOPE[$dataset]}" != "$scope" ]] || pmax="${PCA_P[$dataset]}"
    if [[ "${BEST_SCOPE[$dataset]}" == "$scope" ]]; then
      p="${BEST_P[$dataset]}"; ((p<=pmax)) || pmax="$p"
    fi
    if [[ "${DYNAMIC_SCOPE[$dataset]}" == "$scope" ]]; then
      p="${DYNAMIC_PMAX[$dataset]}"; ((p<=pmax)) || pmax="$p"
    fi
    ((pmax)) || continue
    RICH_INDEX[$scope]="${root}_${scope}_pca_P${pmax}.index"
    RICH_P[$scope]="$pmax"
    build_index "$dataset" "$scope" "${RICH_INDEX[$scope]}" \
      --opt_levels OPT_TRIANGLE --multipivot_modes projection --multipivot_scope "$scope" \
      --multipivot_method pca --pivot_counts "$pmax" --from_index "$SHARED_TRIANGLE" \
      --pivot_manifest "$LOG_DIR/manifests/${dataset}_${scope}_rich.csv"
  done
}

method_arguments() {
  local dataset="$1" method="$2"
  MODE=none; OPT=1; SCOPE=per_list; P=1
  case "$method" in
    baseline) OPT=0; SCOPE=global; P=0 ;;
    triangle) ;;
    pca10) MODE=projection; SCOPE="${PCA_SCOPE[$dataset]}"; P="${PCA_P[$dataset]}" ;;
    static_best) MODE=projection; SCOPE="${BEST_SCOPE[$dataset]}"; P="${BEST_P[$dataset]}" ;;
    dynamic) MODE=projection; SCOPE="${DYNAMIC_SCOPE[$dataset]}"; P="${DYNAMIC_PMAX[$dataset]}" ;;
  esac
  if [[ "$MODE" == none ]]; then
    METHOD=(--load_index "$SHARED_TRIANGLE" --opt_levels "$( ((OPT)) && echo OPT_TRIANGLE || echo OPT_NONE )"
      --multipivot_modes none --multipivot_scope "$SCOPE" --multipivot_method pca --pivot_counts "$P")
  else
    METHOD=(--load_index "${RICH_INDEX[$SCOPE]}" --opt_levels OPT_TRIANGLE
      --multipivot_modes projection --multipivot_scope "$SCOPE" --multipivot_method pca
      --pivot_counts "${RICH_P[$SCOPE]}"
      --pivot_manifest "$LOG_DIR/manifests/${dataset}_${method}.csv")
    if [[ "$method" == dynamic ]]; then
      # Dynamic selects inside exactly its configured Pmax, even if PCA10 uses
      # a larger shared rich index. Do not precompute a static residual prefix.
      METHOD+=(--active_pivot_count "$P" --projection_dynamic --projection_block_size 2)
    elif [[ "$P" == "${RICH_P[$SCOPE]}" ]]; then
      # The full index already stores its final residual. Avoid preparing an
      # identical extra residual array, especially for a billion vectors.
      METHOD+=(--active_pivot_count "$P" --projection_block_size 0)
    else
      METHOD+=(--active_pivot_counts "$P" --projection_block_size 0)
    fi
  fi
}

check_csv() {
  local path="$1" phase="$2" dataset="$3" loops="$4"
  python3 "$RESULT_HELPER" check "$path" --dataset "$dataset" --nlist "${NLIST[$dataset]}" \
    --pivots "$P" --seed "$PIVOT_SEED" --scope "$SCOPE" --opt "$OPT" --mode "$MODE" \
    --phase "$phase" --loops "$loops" --warmups "$WARMUP_LOOPS" --probes "${PROBES[@]}"
}
run_measurement() {
  local phase="$1" dataset="$2" method="$3" rep="$4" binary loops directory csv
  method_arguments "$dataset" "$method"
  read -r -a PROBES <<< "${NPROBES[$dataset]}"
  if [[ "$phase" == perf ]]; then
    binary="$PERF_BIN"; loops="${LOOPS[$dataset]}"
    directory="$LOG_DIR/perf_repeats/$dataset"; csv="$directory/${method}_rep${rep}.csv"
  else
    binary="$STATS_BIN"; loops=1
    directory="$LOG_DIR/stats/$dataset"; csv="$directory/$method.csv"
  fi
  local logfile="${csv%.csv}.log" fingerprint
  local -a arguments=("${COMMON[@]}" --nprobes "${PROBES[@]}" --loop "$loops"
    --warmup_loops "$WARMUP_LOOPS" --csv "${csv}.partial" "${METHOD[@]}")
  if ((!DRY_RUN)); then
    local index index_state="" i
    for ((i=0;i+1<${#METHOD[@]};++i)); do
      if [[ "${METHOD[i]}" == --load_index ]]; then
        index="${METHOD[i+1]}"; index_state="$(stat -c '%n:%s:%Y' "$index")"
      fi
    done
    fingerprint="$(quote_command "$binary" "${arguments[@]}") ${BINARY_HASH[$phase]} $index_state $CPU_SET $THREADS $OMP_WAIT_POLICY"
    if [[ -f "${csv}.recipe" ]] && [[ "$(cat "${csv}.recipe")" == "$fingerprint" ]] &&
       check_csv "$csv" "$phase" "$dataset" "$loops" >/dev/null 2>&1; then
      log "SKIP verified $phase/$dataset/$method rep=$rep"
      return
    fi
    mkdir -p "$directory"
    # Remove only this unfinished output; complete old files remain until success.
    rm -f "${csv}.partial"
  fi
  log "START $phase/$dataset/$method rep=$rep scope=$SCOPE P=$P loops=$loops warmups=$WARMUP_LOOPS"
  execute_logged "$binary" "$logfile" "${arguments[@]}"
  if ((!DRY_RUN)); then
    check_csv "${csv}.partial" "$phase" "$dataset" "$loops"
    mv "${csv}.partial" "$csv"
    printf '%s\n' "$fingerprint" > "${csv}.recipe"
  fi
  log "DONE $phase/$dataset/$method rep=$rep"
}

BUILD_BIN="$PERF_BIN"
[[ "$PHASE" != stats ]] || BUILD_BIN="$STATS_BIN"
declare -A BINARY_HASH
if ((!DRY_RUN)); then
  mkdir -p "$LOG_DIR/manifests" "$LOG_DIR/build"
  for phase in perf stats; do
    [[ "$PHASE" == all || "$PHASE" == "$phase" ]] || continue
    binary="$PERF_BIN"; expected=0
    [[ "$phase" != stats ]] || { binary="$STATS_BIN"; expected=1; }
    require_file "$binary"
    if ! info="$("$binary" --benchmark_info 2>&1)"; then
      echo "Old/incompatible $phase binary: $binary. Rebuild query before running." >&2
      echo "$info" >&2
      exit 1
    fi
    [[ "$info" == "benchmark_protocol=search_only_v2 stats_enabled=$expected" ]] || {
      echo "Wrong/old $phase binary: $binary ($info). Rebuild query before running." >&2; exit 1;
    }
    BINARY_HASH[$phase]="$(sha256sum "$binary" | cut -d ' ' -f 1)"
  done
  {
    date -Is
    git -C "$ROOT" rev-parse HEAD
    git -C "$ROOT" status --short
    sha256sum "$ROOT/src/query.cpp" "$ROOT/src/stats.h" "$0" "$RESULT_HELPER"
    uname -a
    lscpu
    taskset -c "$CPU_SET" sh -c 'taskset -pc $$; nproc'
    printf 'protocol=search_only_v2 threads=%s cpus=%s repeats=%s warmups=%s wait_policy=%s\n' \
      "$THREADS" "$CPU_SET" "$PERF_REPEATS" "$WARMUP_LOOPS" "$OMP_WAIT_POLICY"
    for phase in "${!BINARY_HASH[@]}"; do echo "$phase ${BINARY_HASH[$phase]}"; done
    snapshot
  } >> "$LOG_DIR/environment.log"
fi

log "CONFIG protocol=search_only_v2 data=$DATA_ROOT cpus=$CPU_SET threads=$THREADS phase=$PHASE repeats=$PERF_REPEATS warmups=$WARMUP_LOOPS wait=$OMP_WAIT_POLICY"
MODES=(baseline triangle pca10 static_best dynamic)
same_static_config() {
  [[ "${BEST_P[$1]}" == "${PCA_P[$1]}" && "${BEST_SCOPE[$1]}" == "${PCA_SCOPE[$1]}" ]]
}
if ((!DRY_RUN)); then
  printf 'dataset,base_vectors,dimension,available_queries,requested_queries,nlist,ucr_label_status,static_configuration_origin,source_metric,unit_normalization\n' > "$LOG_DIR/input_manifest.csv"
fi
# Check every requested input before any expensive index construction begins.
for dataset in "${ALL_DATASETS[@]}"; do
  selected "$dataset" || continue
  validate_input "$dataset"
done
for dataset in "${ALL_DATASETS[@]}"; do
  selected "$dataset" || continue
  prepare_dataset "$dataset"
  if [[ "$PHASE" == all || "$PHASE" == perf ]]; then
    for ((rep=1; rep<=PERF_REPEATS; ++rep)); do
      # Rotate method order so one method is not always first/last in the run.
      for ((offset=0; offset<${#MODES[@]}; ++offset)); do
        method="${MODES[$(((rep-1+offset)%${#MODES[@]}))]}"
        if [[ "$method" == static_best ]] && same_static_config "$dataset"; then
          log "ALIAS perf/$dataset/static_best=pca10 rep=$rep (identical configuration)"
          continue
        fi
        run_measurement perf "$dataset" "$method" "$rep"
      done
    done
    for method in "${MODES[@]}"; do
      source_method="$method"
      if [[ "$method" == static_best ]] && same_static_config "$dataset"; then source_method=pca10; fi
      inputs=()
      for ((rep=1;rep<=PERF_REPEATS;++rep)); do inputs+=("$LOG_DIR/perf_repeats/$dataset/${source_method}_rep${rep}.csv"); done
      if ((DRY_RUN)); then
        quote_command python3 "$RESULT_HELPER" aggregate --output "$LOG_DIR/perf/$dataset/$method.csv" "${inputs[@]}"
      else
        python3 "$RESULT_HELPER" aggregate --output "$LOG_DIR/perf/$dataset/$method.csv" "${inputs[@]}"
      fi
    done
  fi
  if [[ "$PHASE" == all || "$PHASE" == stats ]]; then
    for method in "${MODES[@]}"; do
      if [[ "$method" == static_best ]] && same_static_config "$dataset"; then
        log "ALIAS stats/$dataset/static_best=pca10 (identical configuration)"
        if ((!DRY_RUN)); then cp "$LOG_DIR/stats/$dataset/pca10.csv" "$LOG_DIR/stats/$dataset/static_best.csv"; fi
      else
        run_measurement stats "$dataset" "$method" 1
      fi
    done
  fi
done
if ((!DRY_RUN)); then
  python3 "$RESULT_HELPER" summarize "$LOG_DIR"
fi
log "COMPLETE output=$LOG_DIR"
