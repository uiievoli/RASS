#!/usr/bin/env bash
# Fine-grained static PCA-prefix sweep on one P=24 SIFT1M index.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_ROOT="${BENCH_ROOT:-/mnt/nvme/wxy/benchmarks}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/logs/sift1m-fine-pivot-$(date +%Y%m%d-%H%M%S)}"
STATS_BIN="${STATS_BIN:-${ROOT}/build-stats/bin/query}"
PERF_BIN="${PERF_BIN:-${ROOT}/build-perf/bin/query}"
THREADS="${THREADS:-32}"
PERF_LOOPS="${PERF_LOOPS:-10}"
MAX_PIVOTS="${MAX_PIVOTS:-24}"
NPROBE="${NPROBE:-10}"
NLIST="${NLIST:-1000}"

mkdir -p "${OUT_ROOT}"/{stats,perf,logs,manifests,figures}
MASTER_LOG="${OUT_ROOT}/run.log"
TRIANGLE_INDEX="${BENCH_ROOT}/sift1m/index/v10_nlist_${NLIST}_metric_l2_opt_1_subk_15_subNprobeRatio_1_mp_per_list_pca_P0_seed0.index"

if [[ ! -f "${TRIANGLE_INDEX}" ]]; then
  echo "missing Triangle index: ${TRIANGLE_INDEX}" >&2
  exit 1
fi

run_one() {
  local phase="$1" binary="$2" loops="$3" active_p="$4"
  local tag
  local -a mp_args
  if ((active_p == 1)); then
    tag="triangle_P1"
    mp_args=(--multipivot_modes none --pivot_counts 1)
  else
    tag="pca_P${active_p}_from_P${MAX_PIVOTS}"
    mp_args=(
      --multipivot_modes projection
      --pivot_counts "${MAX_PIVOTS}"
      --active_pivot_count "${active_p}"
      --projection_prefix_length "$((active_p - 1))"
      --pivot_manifest "${OUT_ROOT}/manifests/pca_P${MAX_PIVOTS}.csv"
    )
  fi
  local csv="${OUT_ROOT}/${phase}/${tag}.csv"
  local log="${OUT_ROOT}/logs/${tag}__${phase}.log"
  echo "[$(date -Is)] START ${phase} active_P=${active_p} built_P=${MAX_PIVOTS}" | tee -a "${MASTER_LOG}"
  env -u TRIBASE_TRACE -u EDGE_DEVICE_ENABLED OMP_NUM_THREADS="${THREADS}" \
    "${binary}" \
      --benchmarks_path "${BENCH_ROOT}" --dataset sift1m \
      --input_format fvecs --output_format bin --metric l2 --k 1 --nq 0 \
      --nlist "${NLIST}" --nprobes "${NPROBE}" --cache \
      --opt_levels OPT_TRIANGLE --multipivot_scope per_list \
      --multipivot_method pca --pivot_seed 0 --signature_precision float32 \
      --from_index "${TRIANGLE_INDEX}" --loop "${loops}" --csv "${csv}" \
      "${mp_args[@]}" >"${log}" 2>&1
  [[ $(wc -l <"${csv}") -eq 2 ]]
  echo "[$(date -Is)] DONE  ${phase} active_P=${active_p}" | tee -a "${MASTER_LOG}"
}

# P counts the centroid, so P=24 means 23 PCA coordinates. P=1 is Triangle.
for active_p in $(seq 1 "${MAX_PIVOTS}"); do
  run_one stats "${STATS_BIN}" 1 "${active_p}"
  run_one perf "${PERF_BIN}" "${PERF_LOOPS}" "${active_p}"
done

python3 - "${OUT_ROOT}" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])

def one(path):
    with path.open(newline="") as f:
        return next(csv.DictReader(f))

fields = [
    "pivot_count", "pca_dimensions", "nprobe", "recall", "latency_ms", "qps",
    "triangle_pruned", "multipivot_checks", "multipivot_pruned",
    "multipivot_prune_pct", "overall_prune_pct",
    "candidate_distance_computations", "query_signature_us_per_query",
    "candidate_decision_us_per_query", "other_us_per_query",
]
rows = []
for stats_path in sorted((root / "stats").glob("*.csv")):
    perf_path = root / "perf" / stats_path.name
    s, p = one(stats_path), one(perf_path)
    active = int(s["pivot_count"])
    exact = float(s["candidate_distance_computations"])
    speedup = float(s["pruning_speedup"])
    total = exact * speedup
    tri = float(s["tri"]) + float(s["tri_large"])
    checks = float(s["multipivot_checks"])
    mp = float(s["multipivot_pruned"])
    qps = float(p["qps"])
    rows.append({
        "pivot_count": active,
        "pca_dimensions": max(0, active - 1),
        "nprobe": int(s["nprobe"]),
        "recall": float(p["recall"]),
        "latency_ms": 1000.0 / qps,
        "qps": qps,
        "triangle_pruned": int(tri),
        "multipivot_checks": int(checks),
        "multipivot_pruned": int(mp),
        "multipivot_prune_pct": 100.0 * mp / checks if checks else 0.0,
        "overall_prune_pct": 100.0 * (1.0 - exact / total) if total else 0.0,
        "candidate_distance_computations": int(exact),
        "query_signature_us_per_query": s["query_signature_us_per_query"],
        "candidate_decision_us_per_query": s["candidate_decision_us_per_query"],
        "other_us_per_query": s["other_us_per_query"],
    })
rows.sort(key=lambda r: r["pivot_count"])
with (root / "summary.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print(root / "summary.csv")
PY

python3 "${ROOT}/scripts/plot_sift1m_fine_pivot.py" \
  --summary "${OUT_ROOT}/summary.csv" --output-dir "${OUT_ROOT}/figures"
echo "[$(date -Is)] FINISHED ${OUT_ROOT}" | tee -a "${MASTER_LOG}"
