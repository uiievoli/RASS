#!/usr/bin/env bash
# Fine-grained static-prefix sweeps for the two small datasets whose best P is large.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_ROOT="${BENCH_ROOT:-/mnt/nvme/wxy/benchmarks}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/logs/small-global-fine-pivot-$(date +%Y%m%d-%H%M%S)}"
STATS_BIN="${STATS_BIN:-${ROOT}/build-stats/bin/query}"
PERF_BIN="${PERF_BIN:-${ROOT}/build-perf/bin/query}"
THREADS="${THREADS:-32}"
MAX_PIVOTS="${MAX_PIVOTS:-512}"

DATASETS=(fasion_mnist_784 StarLightCurves)
declare -A NLIST=([fasion_mnist_784]=256 [StarLightCurves]=128)
declare -A NPROBE=([fasion_mnist_784]=3 [StarLightCurves]=3)
declare -A STEP=([fasion_mnist_784]=2 [StarLightCurves]=1)
declare -A PERF_LOOPS=([fasion_mnist_784]=3 [StarLightCurves]=20)

mkdir -p "${OUT_ROOT}"
MASTER_LOG="${OUT_ROOT}/run.log"

run_query() {
  local dataset="$1" phase="$2" binary="$3" loops="$4" csv="$5" log="$6"
  shift 6
  env -u TRIBASE_TRACE -u EDGE_DEVICE_ENABLED OMP_NUM_THREADS="${THREADS}" \
    "${binary}" \
      --benchmarks_path "${BENCH_ROOT}" --dataset "${dataset}" \
      --input_format fvecs --output_format bin --metric l2 --k 1 --nq 0 \
      --nlist "${NLIST[${dataset}]}" --nprobes "${NPROBE[${dataset}]}" --cache \
      --opt_levels OPT_TRIANGLE --multipivot_method pca --pivot_seed 0 \
      --signature_precision float32 --loop "${loops}" --csv "${csv}" \
      "$@" >"${log}" 2>&1
}

for dataset in "${DATASETS[@]}"; do
  out="${OUT_ROOT}/${dataset}"
  mkdir -p "${out}"/{stats,perf,logs,manifests,figures}
  triangle_index="${BENCH_ROOT}/${dataset}/index/v10_nlist_${NLIST[${dataset}]}_metric_l2_opt_1_subk_15_subNprobeRatio_1_mp_per_list_pca_P0_seed0.index"
  [[ -f "${triangle_index}" ]] || { echo "missing ${triangle_index}" >&2; exit 1; }

  active=()
  step="${STEP[${dataset}]}"
  for ((p=2; p<=MAX_PIVOTS; p+=step)); do active+=("${p}"); done
  if (( active[${#active[@]}-1] != MAX_PIVOTS )); then active+=("${MAX_PIVOTS}"); fi

  echo "[$(date -Is)] START ${dataset} Triangle" | tee -a "${MASTER_LOG}"
  for phase in stats perf; do
    binary="${STATS_BIN}"; loops=1
    [[ "${phase}" == perf ]] && { binary="${PERF_BIN}"; loops="${PERF_LOOPS[${dataset}]}"; }
    run_query "${dataset}" "${phase}" "${binary}" "${loops}" \
      "${out}/${phase}/triangle_P1.csv" "${out}/logs/triangle_P1__${phase}.log" \
      --multipivot_scope per_list --multipivot_modes none --pivot_counts 1 \
      --from_index "${triangle_index}"
  done

  echo "[$(date -Is)] START ${dataset} P2..P${MAX_PIVOTS} step=${STEP[${dataset}]}" | tee -a "${MASTER_LOG}"
  for phase in stats perf; do
    binary="${STATS_BIN}"; loops=1
    [[ "${phase}" == perf ]] && { binary="${PERF_BIN}"; loops="${PERF_LOOPS[${dataset}]}"; }
    run_query "${dataset}" "${phase}" "${binary}" "${loops}" \
      "${out}/${phase}/pca_prefix_sweep.csv" "${out}/logs/pca_prefix_sweep__${phase}.log" \
      --multipivot_scope global --multipivot_modes projection \
      --pivot_counts "${MAX_PIVOTS}" --active_pivot_counts "${active[@]}" \
      --pivot_manifest "${out}/manifests/pca_P${MAX_PIVOTS}.csv" \
      --from_index "${triangle_index}"
  done

  python3 "${ROOT}/scripts/summarize_fine_pivot.py" \
    --stats-dir "${out}/stats" --perf-dir "${out}/perf" \
    --output "${out}/summary.csv"
  python3 "${ROOT}/scripts/plot_fine_pivot.py" \
    --summary "${out}/summary.csv" --output-dir "${out}/figures" \
    --title "${dataset}" --nprobe "${NPROBE[${dataset}]}"
  echo "[$(date -Is)] DONE ${dataset}" | tee -a "${MASTER_LOG}"
done

echo "[$(date -Is)] FINISHED ${OUT_ROOT}" | tee -a "${MASTER_LOG}"
