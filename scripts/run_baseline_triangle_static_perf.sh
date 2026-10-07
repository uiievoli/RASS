#!/usr/bin/env bash
# Performance-only comparison points missing from the dynamic-P experiment.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${PERF_BIN:-${ROOT}/build-perf/bin/query}"
OLD_ROOT="${OLD_BENCH_ROOT:-/mnt/nvme/wxy/benchmarks}"
DB_ROOT="${DBPEDIA_ROOT:-/mnt/nvme/wxy}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/logs/baseline-triangle-static-$(date +%Y%m%d-%H%M%S)}"
THREADS="${THREADS:-32}"

DATASETS=(nuswide fasion_mnist_784 msong_holdout sift1m glove25 HandOutlines StarLightCurves dbpedia1536m_holdout)
declare -A ROOTS=(
  [nuswide]="${OLD_ROOT}" [fasion_mnist_784]="${OLD_ROOT}"
  [msong_holdout]="${OLD_ROOT}" [sift1m]="${OLD_ROOT}"
  [glove25]="${OLD_ROOT}" [HandOutlines]="${OLD_ROOT}"
  [StarLightCurves]="${OLD_ROOT}" [dbpedia1536m_holdout]="${DB_ROOT}"
)
declare -A NLIST=(
  [nuswide]=512 [fasion_mnist_784]=256 [msong_holdout]=1000 [sift1m]=1000
  [glove25]=1024 [HandOutlines]=32 [StarLightCurves]=128 [dbpedia1536m_holdout]=1000
)
declare -A NPROBE=(
  [nuswide]=3 [fasion_mnist_784]=7 [msong_holdout]=30 [sift1m]=50
  [glove25]=50 [HandOutlines]=5 [StarLightCurves]=5 [dbpedia1536m_holdout]=10
)
declare -A SOURCE_P=(
  [nuswide]=50 [fasion_mnist_784]=64 [msong_holdout]=42 [sift1m]=16
  [glove25]=8 [HandOutlines]=271 [StarLightCurves]=103 [dbpedia1536m_holdout]=128
)
declare -A SOURCE_SCOPE=(
  [nuswide]=global [fasion_mnist_784]=global [msong_holdout]=per_list
  [sift1m]=per_list [glove25]=per_list [HandOutlines]=global
  [StarLightCurves]=global [dbpedia1536m_holdout]=global
)
declare -A BEST_P=(
  [nuswide]=2 [fasion_mnist_784]=48 [msong_holdout]=34 [sift1m]=23
  [glove25]=14 [HandOutlines]=7 [StarLightCurves]=18 [dbpedia1536m_holdout]=128
)
declare -A BEST_SCOPE=(
  [nuswide]=per_list [fasion_mnist_784]=per_list [msong_holdout]=per_list
  [sift1m]=per_list [glove25]=global [HandOutlines]=global
  [StarLightCurves]=global [dbpedia1536m_holdout]=per_list
)
declare -A LOOPS=(
  [nuswide]=50 [fasion_mnist_784]=3 [msong_holdout]=3 [sift1m]=2
  [glove25]=3 [HandOutlines]=100 [StarLightCurves]=30 [dbpedia1536m_holdout]=1
)

mkdir -p "${OUT_ROOT}"/{csv,logs,manifests}
echo "[$(date -Is)] output=${OUT_ROOT}" | tee "${OUT_ROOT}/run.log"

run_one() {
  local dataset="$1" mode="$2"
  shift 2
  local csv="${OUT_ROOT}/csv/${dataset}_${mode}.csv"
  local log="${OUT_ROOT}/logs/${dataset}_${mode}.log"
  echo "[$(date -Is)] START ${dataset}_${mode}" | tee -a "${OUT_ROOT}/run.log"
  env -u TRIBASE_TRACE -u EDGE_DEVICE_ENABLED \
    OMP_NUM_THREADS="${THREADS}" OMP_PROC_BIND=close OMP_PLACES=cores \
    "${BIN}" --benchmarks_path "${ROOTS[${dataset}]}" --dataset "${dataset}" \
    --input_format fvecs --output_format bin --metric l2 --k 1 \
    --nlist "${NLIST[${dataset}]}" --nprobes "${NPROBE[${dataset}]}" --cache \
    --signature_precision float32 --loop "${LOOPS[${dataset}]}" --csv "${csv}" \
    "$@" >"${log}" 2>&1
  echo "[$(date -Is)] DONE ${dataset}_${mode}" | tee -a "${OUT_ROOT}/run.log"
}

for dataset in "${DATASETS[@]}"; do
  source_index="${ROOTS[${dataset}]}/${dataset}/index/v10_nlist_${NLIST[${dataset}]}_metric_l2_opt_1_subk_15_subNprobeRatio_1_mp_${SOURCE_SCOPE[${dataset}]}_pca_P${SOURCE_P[${dataset}]}_seed0.index"
  baseline_index="${ROOTS[${dataset}]}/${dataset}/index/v10_nlist_${NLIST[${dataset}]}_metric_l2_opt_0_subk_15_subNprobeRatio_1_mp_global_affine_fps_P0_seed0.index"
  baseline_source="${source_index}"
  triangle_source="${baseline_index}"

  run_one "${dataset}" baseline \
    --opt_levels OPT_NONE --multipivot_modes none --multipivot_scope global \
    --multipivot_method affine_fps --pivot_counts 0 --from_index "${baseline_source}"

  run_one "${dataset}" triangle \
    --opt_levels OPT_TRIANGLE --multipivot_modes none --multipivot_scope per_list \
    --multipivot_method pca --pivot_counts 1 --from_index "${triangle_source}"

  run_one "${dataset}" static_best \
    --opt_levels OPT_TRIANGLE --multipivot_modes projection \
    --multipivot_scope "${BEST_SCOPE[${dataset}]}" --multipivot_method pca \
    --pivot_counts "${BEST_P[${dataset}]}" --pivot_seed 0 \
    --pivot_manifest "${OUT_ROOT}/manifests/${dataset}_static_best.csv" \
    --from_index "${source_index}"
done

python3 - "${OUT_ROOT}" <<'PY'
import csv, re, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = []
for path in sorted((root / "csv").glob("*.csv")):
    match = re.fullmatch(r"(.+)_(baseline|triangle|static_best)\.csv", path.name)
    if not match:
        continue
    with path.open(newline="") as stream:
        row = next(csv.DictReader(stream))
    row["dataset"] = match.group(1)
    row["mode"] = match.group(2)
    qps = float(row.get("qps", 0) or 0)
    row["latency_ms"] = 1000.0 / qps if qps else 0.0
    rows.append(row)
if rows:
    fields = ["dataset", "mode"] + [key for key in rows[0] if key not in ("dataset", "mode")]
    with (root / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
print(f"wrote {root / 'summary.csv'} rows={len(rows)}")
PY

echo "[$(date -Is)] COMPLETE output=${OUT_ROOT}"
