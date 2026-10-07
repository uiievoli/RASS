#!/usr/bin/env bash
# One fixed-nprobe stats pass used only for overall pruning-rate comparison.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${STATS_BIN:-${ROOT}/build-stats/bin/query}"
OLD_ROOT="${OLD_BENCH_ROOT:-/mnt/nvme/wxy/benchmarks}"
DB_ROOT="${DBPEDIA_ROOT:-/mnt/nvme/wxy}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/logs/fiveway-fixed-pruning-$(date +%Y%m%d-%H%M%S)}"
THREADS="${THREADS:-32}"

DATASETS=(nuswide fasion_mnist_784 msong_holdout sift1m glove25 HandOutlines StarLightCurves dbpedia1536m_holdout)
declare -A ROOTS=([nuswide]="${OLD_ROOT}" [fasion_mnist_784]="${OLD_ROOT}" [msong_holdout]="${OLD_ROOT}" [sift1m]="${OLD_ROOT}" [glove25]="${OLD_ROOT}" [HandOutlines]="${OLD_ROOT}" [StarLightCurves]="${OLD_ROOT}" [dbpedia1536m_holdout]="${DB_ROOT}")
declare -A NLIST=([nuswide]=512 [fasion_mnist_784]=256 [msong_holdout]=1000 [sift1m]=1000 [glove25]=1024 [HandOutlines]=32 [StarLightCurves]=128 [dbpedia1536m_holdout]=1000)
declare -A NPROBE=([nuswide]=3 [fasion_mnist_784]=7 [msong_holdout]=30 [sift1m]=50 [glove25]=50 [HandOutlines]=5 [StarLightCurves]=5 [dbpedia1536m_holdout]=10)
declare -A PCA_P=([nuswide]=50 [fasion_mnist_784]=64 [msong_holdout]=42 [sift1m]=16 [glove25]=8 [HandOutlines]=271 [StarLightCurves]=103 [dbpedia1536m_holdout]=128)
declare -A PCA_SCOPE=([nuswide]=global [fasion_mnist_784]=global [msong_holdout]=per_list [sift1m]=per_list [glove25]=per_list [HandOutlines]=global [StarLightCurves]=global [dbpedia1536m_holdout]=global)
declare -A BEST_P=([nuswide]=2 [fasion_mnist_784]=48 [msong_holdout]=34 [sift1m]=23 [glove25]=14 [HandOutlines]=7 [StarLightCurves]=18 [dbpedia1536m_holdout]=128)
declare -A BEST_SCOPE=([nuswide]=per_list [fasion_mnist_784]=per_list [msong_holdout]=per_list [sift1m]=per_list [glove25]=global [HandOutlines]=global [StarLightCurves]=global [dbpedia1536m_holdout]=per_list)

mkdir -p "${OUT_ROOT}"/{csv,logs,manifests}
run_mode() {
  local dataset="$1" mode="$2"; shift 2
  echo "[$(date -Is)] START ${dataset}_${mode}" | tee -a "${OUT_ROOT}/run.log"
  env OMP_NUM_THREADS="${THREADS}" OMP_PROC_BIND=close OMP_PLACES=cores "${BIN}" \
    --benchmarks_path "${ROOTS[${dataset}]}" --dataset "${dataset}" \
    --input_format fvecs --output_format bin --metric l2 --k 1 --loop 1 --cache \
    --nlist "${NLIST[${dataset}]}" --nprobes "${NPROBE[${dataset}]}" \
    --signature_precision float32 --csv "${OUT_ROOT}/csv/${dataset}_${mode}.csv" \
    "$@" >"${OUT_ROOT}/logs/${dataset}_${mode}.log" 2>&1
  echo "[$(date -Is)] DONE ${dataset}_${mode}" | tee -a "${OUT_ROOT}/run.log"
}

for dataset in "${DATASETS[@]}"; do
  run_mode "${dataset}" baseline --opt_levels OPT_NONE --multipivot_modes none --multipivot_scope global --multipivot_method affine_fps --pivot_counts 0
  run_mode "${dataset}" triangle --opt_levels OPT_TRIANGLE --multipivot_modes none --multipivot_scope per_list --multipivot_method pca --pivot_counts 1
  run_mode "${dataset}" pca10 --opt_levels OPT_TRIANGLE --multipivot_modes projection --multipivot_scope "${PCA_SCOPE[${dataset}]}" --multipivot_method pca --pivot_counts "${PCA_P[${dataset}]}" --projection_block_size 0 --pivot_seed 0 --pivot_manifest "${OUT_ROOT}/manifests/${dataset}_pca10.csv"
  if [[ "${BEST_P[${dataset}]}" == "${PCA_P[${dataset}]}" && "${BEST_SCOPE[${dataset}]}" == "${PCA_SCOPE[${dataset}]}" ]]; then
    cp "${OUT_ROOT}/csv/${dataset}_pca10.csv" "${OUT_ROOT}/csv/${dataset}_static_best.csv"
  else
    run_mode "${dataset}" static_best --opt_levels OPT_TRIANGLE --multipivot_modes projection --multipivot_scope "${BEST_SCOPE[${dataset}]}" --multipivot_method pca --pivot_counts "${BEST_P[${dataset}]}" --projection_block_size 0 --pivot_seed 0 --pivot_manifest "${OUT_ROOT}/manifests/${dataset}_static_best.csv"
  fi
  run_mode "${dataset}" dynamic --opt_levels OPT_TRIANGLE --multipivot_modes projection --multipivot_scope "${PCA_SCOPE[${dataset}]}" --multipivot_method pca --pivot_counts "${PCA_P[${dataset}]}" --projection_block_size 2 --projection_dynamic --pivot_seed 0 --pivot_manifest "${OUT_ROOT}/manifests/${dataset}_dynamic.csv"
done

python3 - "${OUT_ROOT}" <<'PY'
import csv,re,sys
from pathlib import Path
root=Path(sys.argv[1]); rows=[]
for p in sorted((root/'csv').glob('*.csv')):
 m=re.fullmatch(r'(.+)_(baseline|triangle|pca10|static_best|dynamic)\.csv',p.name)
 if not m: continue
 row=next(csv.DictReader(p.open(newline=''))); row['dataset']=m.group(1); row['mode']=m.group(2)
 speed=float(row.get('pruning_speedup',0) or 0)
 row['overall_prune_rate']=1.0-1.0/speed if speed>0 else 0.0
 rows.append(row)
fields=['dataset','mode','overall_prune_rate']+[k for k in rows[0] if k not in ('dataset','mode','overall_prune_rate')]
with (root/'summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
print(f'wrote {root/"summary.csv"} rows={len(rows)}')
PY
