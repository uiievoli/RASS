#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PERF_BIN="${PERF_BIN:-${ROOT}/build-perf/bin/query}"
OLD_BENCH_ROOT="${OLD_BENCH_ROOT:-/mnt/nvme/wxy/benchmarks}"
DBPEDIA_ROOT="${DBPEDIA_ROOT:-/mnt/nvme/wxy}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/logs/progressive-block-$(date +%Y%m%d-%H%M%S)}"
THREADS="${THREADS:-4}"
BLOCKS="${BLOCKS:-0 1 2 4 8}"

DATASETS=(nuswide fasion_mnist_784 msong_holdout sift1m glove25 StarLightCurves dbpedia1536m_holdout)
declare -A ROOTS=(
  [nuswide]="${OLD_BENCH_ROOT}" [fasion_mnist_784]="${OLD_BENCH_ROOT}"
  [msong_holdout]="${OLD_BENCH_ROOT}" [sift1m]="${OLD_BENCH_ROOT}"
  [glove25]="${OLD_BENCH_ROOT}"
  [StarLightCurves]="${OLD_BENCH_ROOT}" [dbpedia1536m_holdout]="${DBPEDIA_ROOT}"
)
declare -A NLIST=(
  [nuswide]=512 [fasion_mnist_784]=256 [msong_holdout]=1000 [sift1m]=1000
  [glove25]=1024 [StarLightCurves]=128 [dbpedia1536m_holdout]=1000
)
declare -A NPROBE=(
  [nuswide]=3 [fasion_mnist_784]=7 [msong_holdout]=30 [sift1m]=50
  [glove25]=50 [StarLightCurves]=5 [dbpedia1536m_holdout]=10
)
declare -A PIVOTS=(
  [nuswide]=50 [fasion_mnist_784]=64 [msong_holdout]=42 [sift1m]=13
  [glove25]=8 [StarLightCurves]=103 [dbpedia1536m_holdout]=128
)
declare -A SCOPE=(
  [nuswide]=global [fasion_mnist_784]=global [msong_holdout]=per_list
  [sift1m]=per_list [glove25]=per_list
  [StarLightCurves]=global [dbpedia1536m_holdout]=global
)
declare -A LOOPS=(
  [nuswide]=50 [fasion_mnist_784]=3 [msong_holdout]=3 [sift1m]=2
  [glove25]=3 [StarLightCurves]=30 [dbpedia1536m_holdout]=1
)

if [[ -n "${ONLY_DATASETS:-}" ]]; then read -r -a DATASETS <<<"${ONLY_DATASETS}"; fi
mkdir -p "${OUT_ROOT}/csv" "${OUT_ROOT}/logs" "${OUT_ROOT}/manifests"
echo "[$(date -Is)] output=${OUT_ROOT}" | tee "${OUT_ROOT}/run.log"

for dataset in "${DATASETS[@]}"; do
  for block in ${BLOCKS}; do
    tag="${dataset}_block${block}"
    csv="${OUT_ROOT}/csv/${tag}.csv"
    log="${OUT_ROOT}/logs/${tag}.log"
    echo "[$(date -Is)] START ${tag}" | tee -a "${OUT_ROOT}/run.log"
    OMP_NUM_THREADS="${THREADS}" OMP_PROC_BIND=close OMP_PLACES=cores \
      "${PERF_BIN}" --benchmarks_path "${ROOTS[${dataset}]}" --dataset "${dataset}" \
      --input_format fvecs --output_format bin --metric l2 --k 1 \
      --nlist "${NLIST[${dataset}]}" --nprobes "${NPROBE[${dataset}]}" --cache \
      --opt_levels OPT_TRIANGLE --multipivot_modes projection \
      --multipivot_scope "${SCOPE[${dataset}]}" --multipivot_method pca \
      --pivot_counts "${PIVOTS[${dataset}]}" --pivot_seed 0 --signature_precision float32 \
      --projection_block_size "${block}" \
      --pivot_manifest "${OUT_ROOT}/manifests/${dataset}.csv" \
      --loop "${LOOPS[${dataset}]}" --csv "${csv}" >"${log}" 2>&1
    echo "[$(date -Is)] DONE ${tag}" | tee -a "${OUT_ROOT}/run.log"
  done
done

python3 - "${OUT_ROOT}" <<'PY'
import csv, re, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = []
for path in sorted((root / "csv").glob("*.csv")):
    match = re.fullmatch(r"(.+)_block(\d+)\.csv", path.name)
    if not match:
        continue
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            row["dataset"] = match.group(1)
            row["block_size"] = int(match.group(2))
            row["qps"] = float(row.get("qps", 0) or 0)
            row["latency_ms"] = 1000.0 / row["qps"] if row["qps"] else 0.0
            rows.append(row)
if rows:
    fields = ["dataset", "block_size"] + [k for k in rows[0] if k not in ("dataset", "block_size")]
    with (root / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {root / 'summary.csv'} rows={len(rows)}")
PY
echo "[$(date -Is)] COMPLETE output=${OUT_ROOT}"
