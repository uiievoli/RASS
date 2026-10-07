#!/usr/bin/env bash
# Compare the legacy full-signature LB with one prefix-residual checkpoint.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_ROOT="${BENCH_ROOT:-/mnt/nvme/wxy/benchmarks}"
PERF_BIN="${PERF_BIN:-${ROOT}/build-perf/bin/query}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/logs/staged-lowerbound-${RUN_ID}}"
THREADS="${THREADS:-32}"
OMP_BIND="${OMP_BIND:-close}"
OMP_PLACES_VALUE="${OMP_PLACES_VALUE:-cores}"
REPEATS="${REPEATS:-3}"
STAGE_STRIDES="${STAGE_STRIDES:-0 1 2 4 8 16}"

DATASETS=(nuswide fasion_mnist_784 msong_holdout sift1m glove25 HandOutlines StarLightCurves)
if [[ -n "${ONLY_DATASETS:-}" ]]; then
  read -r -a DATASETS <<<"${ONLY_DATASETS}"
fi

declare -A NLIST=(
  [nuswide]=512 [fasion_mnist_784]=256 [msong_holdout]=1000
  [sift1m]=1000 [glove25]=1024 [HandOutlines]=32 [StarLightCurves]=128
)
declare -A NPROBE=(
  [nuswide]=3 [fasion_mnist_784]=7 [msong_holdout]=30
  [sift1m]=50 [glove25]=50 [HandOutlines]=5 [StarLightCurves]=5
)
declare -A PIVOTS=(
  [nuswide]=2 [fasion_mnist_784]=48 [msong_holdout]=34
  [sift1m]=23 [glove25]=14 [HandOutlines]=7 [StarLightCurves]=18
)
declare -A SCOPE=(
  [nuswide]=per_list [fasion_mnist_784]=per_list [msong_holdout]=per_list
  [sift1m]=per_list [glove25]=global [HandOutlines]=global [StarLightCurves]=global
)
declare -A LOOPS=(
  [nuswide]=100 [fasion_mnist_784]=4 [msong_holdout]=4
  [sift1m]=2 [glove25]=3 [HandOutlines]=200 [StarLightCurves]=100
)

mkdir -p "${OUT_ROOT}"/{csv,logs,manifests}
echo "[$(date -Is)] output=${OUT_ROOT}" | tee "${OUT_ROOT}/run.log"

for dataset in "${DATASETS[@]}"; do
  nlist="${NLIST[${dataset}]}"
  nprobe="${NPROBE[${dataset}]}"
  pivots="${PIVOTS[${dataset}]}"
  scope="${SCOPE[${dataset}]}"
  loops="${PERF_LOOPS:-${LOOPS[${dataset}]}}"
  for ((repeat=1; repeat<=REPEATS; ++repeat)); do
    # Rotate/reverse order between repeats to reduce temperature/order bias.
    read -r -a strides <<<"${STAGE_STRIDES}"
    if (( repeat % 2 == 0 )); then
      reversed=()
      for ((i=${#strides[@]}-1; i>=0; --i)); do reversed+=("${strides[i]}"); done
      strides=("${reversed[@]}")
    elif (( repeat > 2 )); then
      first="${strides[0]}"
      strides=("${strides[@]:1}" "${first}")
    fi
    for stride in "${strides[@]}"; do
      tag="${dataset}_P${pivots}_${scope}_stage${stride}_r${repeat}"
      csv="${OUT_ROOT}/csv/${tag}.csv"
      log="${OUT_ROOT}/logs/${tag}.log"
      echo "[$(date -Is)] START ${tag} loops=${loops}" | tee -a "${OUT_ROOT}/run.log"
      env -u TRIBASE_TRACE -u EDGE_DEVICE_ENABLED OMP_NUM_THREADS="${THREADS}" \
        OMP_PROC_BIND="${OMP_BIND}" OMP_PLACES="${OMP_PLACES_VALUE}" \
        "${PERF_BIN}" \
        --benchmarks_path "${BENCH_ROOT}" --dataset "${dataset}" \
        --input_format fvecs --output_format bin --metric l2 --k 1 \
        --nlist "${nlist}" --nprobes "${nprobe}" --cache \
        --opt_levels OPT_TRIANGLE --multipivot_modes projection \
        --multipivot_scope "${scope}" --multipivot_method pca \
        --pivot_counts "${pivots}" --pivot_seed 0 --signature_precision float32 \
        --projection_prefix_length "${stride}" \
        --pivot_manifest "${OUT_ROOT}/manifests/${dataset}_P${pivots}_${scope}.csv" \
        --loop "${loops}" --csv "${csv}" >"${log}" 2>&1
      echo "[$(date -Is)] DONE  ${tag}" | tee -a "${OUT_ROOT}/run.log"
    done
  done
done

python3 - "${OUT_ROOT}" <<'PY'
import csv
import re
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
pattern = re.compile(r"(.+)_P(\d+)_(global|per_list)_stage(\d+)_r(\d+)\.csv")
for path in sorted((root / "csv").glob("*.csv")):
    match = pattern.fullmatch(path.name)
    if not match:
        continue
    with path.open(newline="") as stream:
        record = list(csv.DictReader(stream))[-1]
    rows.append({
        "dataset": match.group(1), "pivot_count": int(match.group(2)),
        "scope": match.group(3), "prefix_length": int(match.group(4)),
        "repeat": int(match.group(5)), "latency_ms": 1000 / float(record["qps"]),
        "qps": float(record["qps"]), "recall": float(record["recall"]),
    })

with (root / "raw_results.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)

summary = []
keys = sorted({(r["dataset"], r["pivot_count"], r["scope"], r["prefix_length"]) for r in rows})
for dataset, pivots, scope, stride in keys:
    group = [r for r in rows if (r["dataset"], r["pivot_count"], r["scope"], r["prefix_length"]) == (dataset, pivots, scope, stride)]
    baseline = [r for r in rows if r["dataset"] == dataset and r["prefix_length"] == 0]
    latency = statistics.median(r["latency_ms"] for r in group)
    baseline_latency = statistics.median(r["latency_ms"] for r in baseline)
    summary.append({
        "dataset": dataset, "pivot_count": pivots, "scope": scope,
        "prefix_length": stride, "median_latency_ms": latency,
        "median_qps": statistics.median(r["qps"] for r in group),
        "min_recall": min(r["recall"] for r in group),
        "max_recall": max(r["recall"] for r in group),
        "speedup_vs_legacy": baseline_latency / latency,
    })
with (root / "summary.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
    writer.writeheader(); writer.writerows(summary)
print(root / "summary.csv")
PY

echo "[$(date -Is)] FINISHED" | tee -a "${OUT_ROOT}/run.log"
