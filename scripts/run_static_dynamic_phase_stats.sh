#!/usr/bin/env bash
# Fixed-nprobe phase timing for best-static PCA versus dynamic PCA.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${STATS_BIN:-${ROOT}/build-stats/bin/query}"
OLD_ROOT="${OLD_BENCH_ROOT:-/mnt/nvme/wxy/benchmarks}"
DB_ROOT="${DBPEDIA_ROOT:-/mnt/nvme/wxy}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/logs/static-dynamic-phases-$(date +%Y%m%d-%H%M%S)}"
THREADS="${THREADS:-32}"
LOOP="${LOOP:-2}"

DATASETS=(nuswide fasion_mnist_784 msong_holdout sift1m glove25 HandOutlines StarLightCurves dbpedia1536m_holdout)
declare -A ROOTS=([nuswide]="${OLD_ROOT}" [fasion_mnist_784]="${OLD_ROOT}" [msong_holdout]="${OLD_ROOT}" [sift1m]="${OLD_ROOT}" [glove25]="${OLD_ROOT}" [HandOutlines]="${OLD_ROOT}" [StarLightCurves]="${OLD_ROOT}" [dbpedia1536m_holdout]="${DB_ROOT}")
declare -A NLIST=([nuswide]=512 [fasion_mnist_784]=256 [msong_holdout]=1000 [sift1m]=1000 [glove25]=1024 [HandOutlines]=32 [StarLightCurves]=128 [dbpedia1536m_holdout]=1000)
declare -A NPROBE=([nuswide]=3 [fasion_mnist_784]=7 [msong_holdout]=30 [sift1m]=50 [glove25]=50 [HandOutlines]=5 [StarLightCurves]=5 [dbpedia1536m_holdout]=10)
declare -A DYNAMIC_P=([nuswide]=50 [fasion_mnist_784]=64 [msong_holdout]=42 [sift1m]=16 [glove25]=8 [HandOutlines]=271 [StarLightCurves]=103 [dbpedia1536m_holdout]=128)
declare -A DYNAMIC_SCOPE=([nuswide]=global [fasion_mnist_784]=global [msong_holdout]=per_list [sift1m]=per_list [glove25]=per_list [HandOutlines]=global [StarLightCurves]=global [dbpedia1536m_holdout]=global)
declare -A STATIC_P=([nuswide]=4 [fasion_mnist_784]=64 [msong_holdout]=64 [sift1m]=16 [glove25]=8 [HandOutlines]=4 [StarLightCurves]=64 [dbpedia1536m_holdout]=128)
declare -A STATIC_SCOPE=([nuswide]=global [fasion_mnist_784]=global [msong_holdout]=per_list [sift1m]=per_list [glove25]=per_list [HandOutlines]=global [StarLightCurves]=global [dbpedia1536m_holdout]=global)

mkdir -p "${OUT_ROOT}"/{csv,logs,manifests}
run_mode() {
  local dataset="$1" mode="$2" scope="$3" pivots="$4"
  shift 4
  echo "[$(date -Is)] START ${dataset}_${mode}" | tee -a "${OUT_ROOT}/run.log"
  env -u TRIBASE_TRACE -u EDGE_DEVICE_ENABLED OMP_NUM_THREADS="${THREADS}" \
    OMP_PROC_BIND=close OMP_PLACES=cores "${BIN}" \
    --benchmarks_path "${ROOTS[${dataset}]}" --dataset "${dataset}" \
    --input_format fvecs --output_format bin --metric l2 --k 1 --cache \
    --nlist "${NLIST[${dataset}]}" --nprobes "${NPROBE[${dataset}]}" \
    --opt_levels OPT_TRIANGLE --multipivot_modes projection \
    --multipivot_scope "${scope}" --multipivot_method pca --pivot_counts "${pivots}" \
    --signature_precision float32 --pivot_seed 0 --loop "${LOOP}" \
    --pivot_manifest "${OUT_ROOT}/manifests/${dataset}_${mode}.csv" \
    --csv "${OUT_ROOT}/csv/${dataset}_${mode}.csv" "$@" \
    >"${OUT_ROOT}/logs/${dataset}_${mode}.log" 2>&1
  echo "[$(date -Is)] DONE ${dataset}_${mode}" | tee -a "${OUT_ROOT}/run.log"
}

for dataset in "${DATASETS[@]}"; do
  run_mode "${dataset}" static_best "${STATIC_SCOPE[${dataset}]}" "${STATIC_P[${dataset}]}" \
    --projection_block_size 0
  run_mode "${dataset}" dynamic "${DYNAMIC_SCOPE[${dataset}]}" "${DYNAMIC_P[${dataset}]}" \
    --projection_block_size 2 --projection_dynamic
done

python3 - "${OUT_ROOT}" <<'PY'
import csv, re, sys
from pathlib import Path
root=Path(sys.argv[1]); rows=[]
for path in sorted((root/'csv').glob('*.csv')):
    match=re.fullmatch(r'(.+)_(static_best|dynamic)\.csv',path.name)
    if not match: continue
    row=next(csv.DictReader(path.open(newline='')))
    row['dataset']=match.group(1); row['mode']=match.group(2); rows.append(row)
fields=['dataset','mode']+[k for k in rows[0] if k not in ('dataset','mode')]
with (root/'summary.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
print(f'wrote {root/"summary.csv"} rows={len(rows)}')
PY
