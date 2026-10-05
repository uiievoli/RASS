#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PERF_BIN="${PERF_BIN:-${ROOT}/build-perf/bin/query}"
OLD_BENCH_ROOT="${OLD_BENCH_ROOT:-/mnt/nvme/wxy/benchmarks}"
DBPEDIA_ROOT="${DBPEDIA_ROOT:-/mnt/nvme/wxy}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/logs/progressive-oracle-$(date +%Y%m%d-%H%M%S)}"
THREADS="${THREADS:-32}"
NQ="${NQ:-1000}"

# P includes the fixed center, so the progressive projection depth is P-1.
DATASETS=(nuswide fasion_mnist_784 msong_holdout sift1m glove25 HandOutlines StarLightCurves dbpedia1536m_holdout)
if [[ -n "${ONLY_DATASETS:-}" ]]; then
  read -r -a DATASETS <<<"${ONLY_DATASETS}"
fi
declare -A ROOTS=(
  [nuswide]="${OLD_BENCH_ROOT}" [fasion_mnist_784]="${OLD_BENCH_ROOT}"
  [msong_holdout]="${OLD_BENCH_ROOT}" [sift1m]="${OLD_BENCH_ROOT}"
  [glove25]="${OLD_BENCH_ROOT}" [HandOutlines]="${OLD_BENCH_ROOT}"
  [StarLightCurves]="${OLD_BENCH_ROOT}" [dbpedia1536m_holdout]="${DBPEDIA_ROOT}"
)
declare -A NLIST=(
  [nuswide]=512 [fasion_mnist_784]=256 [msong_holdout]=1000 [sift1m]=1000
  [glove25]=1024 [HandOutlines]=32 [StarLightCurves]=128 [dbpedia1536m_holdout]=1000
)
declare -A NPROBE=(
  [nuswide]=3 [fasion_mnist_784]=7 [msong_holdout]=30 [sift1m]=50
  [glove25]=50 [HandOutlines]=5 [StarLightCurves]=5 [dbpedia1536m_holdout]=10
)
declare -A PIVOTS=(
  [nuswide]=50 [fasion_mnist_784]=64 [msong_holdout]=42 [sift1m]=13
  [glove25]=8 [HandOutlines]=271 [StarLightCurves]=103 [dbpedia1536m_holdout]=128
)
declare -A SCOPE=(
  [nuswide]=global [fasion_mnist_784]=global [msong_holdout]=per_list
  [sift1m]=per_list [glove25]=per_list [HandOutlines]=global
  [StarLightCurves]=global [dbpedia1536m_holdout]=global
)

mkdir -p "${OUT_ROOT}/oracle" "${OUT_ROOT}/manifests"
for dataset in "${DATASETS[@]}"; do
  root="${ROOTS[${dataset}]}"
  output="${OUT_ROOT}/oracle/${dataset}_progressive_oracle.csv"
  log="${OUT_ROOT}/oracle/${dataset}.log"
  echo "[$(date -Is)] START ${dataset} P=${PIVOTS[${dataset}]} scope=${SCOPE[${dataset}]} nprobe=${NPROBE[${dataset}]}" | tee -a "${OUT_ROOT}/run.log"
  OMP_NUM_THREADS="${THREADS}" OMP_PROC_BIND="close" OMP_PLACES="cores" \
    "${PERF_BIN}" \
      --benchmarks_path "${root}" --dataset "${dataset}" \
      --input_format fvecs --output_format bin --metric l2 --k 1 --nq "${NQ}" \
      --nlist "${NLIST[${dataset}]}" --nprobes "${NPROBE[${dataset}]}" --cache \
      --opt_levels OPT_TRIANGLE --multipivot_modes projection \
      --multipivot_scope "${SCOPE[${dataset}]}" --multipivot_method pca \
      --pivot_counts "${PIVOTS[${dataset}]}" --pivot_seed 0 --signature_precision float32 \
      --pivot_manifest "${OUT_ROOT}/manifests/${dataset}.csv" \
      --progressive_oracle "${output}" --progressive_oracle_only --verbose \
      >"${log}" 2>&1
  echo "[$(date -Is)] DONE ${dataset}" | tee -a "${OUT_ROOT}/run.log"
done

python3 "${ROOT}/scripts/analyze_progressive_oracle.py" \
  "${OUT_ROOT}"/oracle/*_progressive_oracle.csv \
  --output "${OUT_ROOT}/oracle_summary.csv"
echo "[$(date -Is)] COMPLETE output=${OUT_ROOT}"
