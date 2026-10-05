#!/usr/bin/env bash
# Extend the DBpedia1536 five-mode sweep through the high-recall region.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${PERF_BIN:-${ROOT}/build-perf/bin/query}"
DB_ROOT="${DBPEDIA_ROOT:-/mnt/nvme/wxy}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/logs/dbpedia-high-recall-$(date +%Y%m%d-%H%M%S)}"
THREADS="${THREADS:-32}"
NPROBES_TEXT="${NPROBES:-30 50 75 100 150 200}"
read -r -a PROBES <<<"${NPROBES_TEXT}"

mkdir -p "${OUT_ROOT}"/{csv,logs,manifests}

run_mode() {
  local mode="$1"
  shift
  echo "[$(date -Is)] START dbpedia1536m_holdout_${mode}" | tee -a "${OUT_ROOT}/run.log"
  env -u TRIBASE_TRACE -u EDGE_DEVICE_ENABLED OMP_NUM_THREADS="${THREADS}" \
    OMP_PROC_BIND=close OMP_PLACES=cores "${BIN}" \
    --benchmarks_path "${DB_ROOT}" --dataset dbpedia1536m_holdout \
    --input_format fvecs --output_format bin --metric l2 --k 1 --loop 1 --cache \
    --nlist 1000 --nprobes "${PROBES[@]}" --signature_precision float32 \
    --csv "${OUT_ROOT}/csv/dbpedia1536m_holdout_${mode}.csv" \
    "$@" >"${OUT_ROOT}/logs/dbpedia1536m_holdout_${mode}.log" 2>&1
  echo "[$(date -Is)] DONE dbpedia1536m_holdout_${mode}" | tee -a "${OUT_ROOT}/run.log"
}

run_mode baseline --opt_levels OPT_NONE --multipivot_modes none \
  --multipivot_scope global --multipivot_method affine_fps --pivot_counts 0
run_mode triangle --opt_levels OPT_TRIANGLE --multipivot_modes none \
  --multipivot_scope per_list --multipivot_method pca --pivot_counts 1
run_mode pca10 --opt_levels OPT_TRIANGLE --multipivot_modes projection \
  --multipivot_scope global --multipivot_method pca --pivot_counts 128 \
  --projection_block_size 0 --pivot_seed 0 \
  --pivot_manifest "${OUT_ROOT}/manifests/dbpedia1536m_holdout_pca10.csv"
run_mode static_best --opt_levels OPT_TRIANGLE --multipivot_modes projection \
  --multipivot_scope global --multipivot_method pca --pivot_counts 64 \
  --projection_block_size 0 --pivot_seed 0 \
  --pivot_manifest "${OUT_ROOT}/manifests/dbpedia1536m_holdout_static_best.csv"
run_mode dynamic --opt_levels OPT_TRIANGLE --multipivot_modes projection \
  --multipivot_scope global --multipivot_method pca --pivot_counts 128 \
  --projection_block_size 2 --projection_dynamic --pivot_seed 0 \
  --pivot_manifest "${OUT_ROOT}/manifests/dbpedia1536m_holdout_dynamic.csv"

python3 - "${OUT_ROOT}" <<'PY'
import csv, re, sys
from pathlib import Path
root = Path(sys.argv[1]); rows = []
for path in sorted((root / "csv").glob("*.csv")):
    match = re.fullmatch(r"(.+)_(baseline|triangle|pca10|static_best|dynamic)\.csv", path.name)
    if not match:
        continue
    for row in csv.DictReader(path.open(newline="")):
        row["dataset"] = match.group(1); row["mode"] = match.group(2)
        qps = float(row.get("qps", 0) or 0)
        row["latency_ms"] = 1000.0 / qps if qps else 0.0
        rows.append(row)
fields = ["dataset", "mode"] + [k for k in rows[0] if k not in ("dataset", "mode")]
with (root / "summary.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader(); writer.writerows(rows)
print(f"wrote {root / 'summary.csv'} rows={len(rows)}")
PY
