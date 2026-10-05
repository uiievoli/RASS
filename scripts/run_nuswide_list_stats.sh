#!/usr/bin/env bash
# Detailed NUS-WIDE IVF-list pruning analysis at nprobe=3.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_ROOT="${BENCH_ROOT:-/mnt/nvme/wxy/benchmarks}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/logs/nuswide-list-stats-${RUN_ID}}"
STATS_BIN="${STATS_BIN:-${ROOT}/build-stats/bin/query}"
PERF_BIN="${PERF_BIN:-${ROOT}/build-perf/bin/query}"

NLIST="${NLIST:-512}"
NPROBE="${NPROBE:-3}"
PIVOT_COUNTS="${PIVOT_COUNTS:-4 25 50 64}"
SCOPES="${SCOPES:-per_list global}"
THREADS="${THREADS:-32}"
PERF_LOOPS="${PERF_LOOPS:-100}"
NQ="${NQ:-0}"
SIGNATURE_PRECISION="${SIGNATURE_PRECISION:-float32}"
PIVOT_SEED="${PIVOT_SEED:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"
RUN_STATS="${RUN_STATS:-1}"
RUN_PERF="${RUN_PERF:-1}"
RUN_LIST_STATS="${RUN_LIST_STATS:-1}"

mkdir -p "${OUT_ROOT}"/{stats,perf,list_run_stats,list_visits,logs,manifests}
MASTER_LOG="${OUT_ROOT}/run.log"
IVFFLAT_INDEX="${BENCH_ROOT}/nuswide/index/v9_nlist_${NLIST}_metric_l2_opt_0_subk_15_subNprobeRatio_1_mp_global_affine_fps_P0_seed0.index"
TRIANGLE_INDEX="${BENCH_ROOT}/nuswide/index/v9_nlist_${NLIST}_metric_l2_opt_1_subk_15_subNprobeRatio_1_mp_per_list_pca_P0_seed0.index"

log() { echo "[$(date -Is)] $*" | tee -a "${MASTER_LOG}"; }
complete() { [[ -f "$1" && "$(wc -l < "$1")" -eq 2 ]]; }

run_query() {
  local label="$1" binary="$2" loops="$3" csv_path="$4" log_path="$5"
  shift 5
  if [[ "${SKIP_EXISTING}" == 1 ]] && complete "${csv_path}"; then
    log "SKIP ${label}"
    return
  fi
  log "START ${label} loops=${loops}"
  if [[ "${DRY_RUN}" == 1 ]]; then
    printf '%q ' env -u TRIBASE_TRACE -u EDGE_DEVICE_ENABLED \
      OMP_NUM_THREADS="${THREADS}" "${binary}" "$@" --loop "${loops}" --csv "${csv_path}"
    echo
    return
  fi
  rm -f "${csv_path}"
  env -u TRIBASE_TRACE -u EDGE_DEVICE_ENABLED OMP_NUM_THREADS="${THREADS}" \
    "${binary}" "$@" --loop "${loops}" --csv "${csv_path}" >"${log_path}" 2>&1
  complete "${csv_path}" || { log "FAIL ${label}: invalid CSV ${csv_path}"; return 1; }
  log "DONE ${label}"
}

run_config() {
  local tag="$1"
  shift
  local args=("$@")

  if [[ "${RUN_STATS}" == 1 ]]; then
    run_query "stats/${tag}" "${STATS_BIN}" 1 \
      "${OUT_ROOT}/stats/${tag}.csv" "${OUT_ROOT}/logs/${tag}__stats.log" "${args[@]}"
  fi

  if [[ "${RUN_PERF}" == 1 ]]; then
    run_query "perf/${tag}" "${PERF_BIN}" "${PERF_LOOPS}" \
      "${OUT_ROOT}/perf/${tag}.csv" "${OUT_ROOT}/logs/${tag}__perf.log" "${args[@]}"
  fi

  local visit_path="${OUT_ROOT}/list_visits/${tag}.csv"
  if [[ "${RUN_LIST_STATS}" != 1 ]]; then
    :
  elif [[ "${SKIP_EXISTING}" == 1 && -s "${visit_path}" ]] && \
     complete "${OUT_ROOT}/list_run_stats/${tag}.csv"; then
    log "SKIP list_stats/${tag}"
  else
    rm -f "${visit_path}"
    run_query "list_stats/${tag}" "${STATS_BIN}" 1 \
      "${OUT_ROOT}/list_run_stats/${tag}.csv" \
      "${OUT_ROOT}/logs/${tag}__list_stats.log" \
      "${args[@]}" --dump_list_stats "${visit_path}"
  fi
}

common=(
  --benchmarks_path "${BENCH_ROOT}"
  --dataset nuswide
  --input_format fvecs
  --output_format bin
  --metric l2
  --k 1
  --nq "${NQ}"
  --nlist "${NLIST}"
  --nprobes "${NPROBE}"
  --cache
  --signature_precision "${SIGNATURE_PRECISION}"
  --opt_levels OPT_TRIANGLE
)

run_config triangle_P1 \
  "${common[@]}" \
  --multipivot_modes none \
  --multipivot_scope per_list \
  --multipivot_method pca \
  --pivot_counts 1 \
  --from_index "${IVFFLAT_INDEX}"

for scope in ${SCOPES}; do
  [[ "${scope}" == per_list || "${scope}" == global ]] || {
    echo "invalid scope: ${scope}" >&2
    exit 1
  }
  for pivot_count in ${PIVOT_COUNTS}; do
    tag="pca_${scope}_P${pivot_count}"
    run_config "${tag}" \
      "${common[@]}" \
      --multipivot_modes projection \
      --multipivot_scope "${scope}" \
      --multipivot_method pca \
      --pivot_counts "${pivot_count}" \
      --pivot_seed "${PIVOT_SEED}" \
      --pivot_manifest "${OUT_ROOT}/manifests/${tag}.csv" \
      --from_index "${TRIANGLE_INDEX}"
  done
done

if [[ "${DRY_RUN}" != 1 ]]; then
  python3 "${ROOT}/scripts/analyze_nuswide_list_stats.py" "${OUT_ROOT}" | tee -a "${MASTER_LOG}"
fi
log "FINISHED results=${OUT_ROOT}"
