#!/usr/bin/env bash
# Pivot-count vs. latency experiment at one fixed nprobe per dataset.
# Edit the configuration block below or override common options with environment variables.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_ROOT="${BENCH_ROOT:-/mnt/nvme/wxy/benchmarks}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/logs/pivot-latency-${RUN_ID}}"

# ------------------------- Experiment configuration -------------------------
DATASETS=(
  nuswide
  fasion_mnist_784
  msong_holdout
  sift1m
  glove25
  HandOutlines
  StarLightCurves
)
# Optional shell override, for example: ONLY_DATASETS="sift1m glove25".
if [[ -n "${ONLY_DATASETS:-}" ]]; then
  read -r -a DATASETS <<<"${ONLY_DATASETS}"
fi

declare -A DIMENSIONS=(
  [nuswide]=500
  [fasion_mnist_784]=784
  [msong_holdout]=420
  [sift1m]=128
  [glove25]=25
  [HandOutlines]=2710
  [StarLightCurves]=1025
)

declare -A NLISTS=(
  [nuswide]=512
  [fasion_mnist_784]=256
  [msong_holdout]=1000
  [sift1m]=1000
  [glove25]=1024
  [HandOutlines]=32
  [StarLightCurves]=128
)

# First nprobe reaching recall@1 >= 0.99 in the 2026-09-11 full sweep.
declare -A NPROBES=(
  [nuswide]=3
  [fasion_mnist_784]=7
  [msong_holdout]=30
  [sift1m]=50
  [glove25]=50
  [HandOutlines]=5
  [StarLightCurves]=5
)

declare -A SCOPES=(
  [nuswide]=per_list
  [fasion_mnist_784]=per_list
  [msong_holdout]=per_list
  [sift1m]=per_list
  [glove25]=per_list
  [HandOutlines]=global
  [StarLightCurves]=global
)

# Each sequence includes the first value strictly greater than 10% of D.
# P=1 is the Triangle-only baseline; P>=2 enables PCA projection pruning.
declare -A PIVOTS=(
  [nuswide]="1 4 16 64"
  [fasion_mnist_784]="1 4 16 64 128"
  [msong_holdout]="1 4 16 64"
  [sift1m]="1 4 16"
  [glove25]="1 4"
  [HandOutlines]="1 4 16 64 128 256 512"
  [StarLightCurves]="1 4 16 64 128"
)

# Short-running datasets need more repetitions to suppress OpenMP timing noise.
declare -A PERF_REPEATS=(
  [nuswide]=100
  [fasion_mnist_784]=20
  [msong_holdout]=20
  [sift1m]=10
  [glove25]=20
  [HandOutlines]=100
  [StarLightCurves]=100
)
# ---------------------------------------------------------------------------

STATS_BIN="${STATS_BIN:-${ROOT}/build-stats/bin/query}"
PERF_BIN="${PERF_BIN:-${ROOT}/build-perf/bin/query}"
THREADS="${THREADS:-32}"
BUILD_THREADS="${BUILD_THREADS:-${THREADS}}"
STATS_LOOPS="${STATS_LOOPS:-1}"
K="${K:-1}"
NQ="${NQ:-0}"
PIVOT_SEED="${PIVOT_SEED:-0}"
SIGNATURE_PRECISION="${SIGNATURE_PRECISION:-float32}"
RUN_STATS="${RUN_STATS:-1}"
RUN_PERF="${RUN_PERF:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
BUILD="${BUILD:-0}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${OUT_ROOT}"
MASTER_LOG="${OUT_ROOT}/run.log"

log() {
  echo "[$(date -Is)] $*" | tee -a "${MASTER_LOG}"
}

if [[ "${BUILD}" == "1" ]]; then
  cmake --build "${ROOT}/build-stats" --target query -j"${BUILD_THREADS}"
  cmake --build "${ROOT}/build-perf" --target query -j"${BUILD_THREADS}"
fi

if [[ "${RUN_STATS}" == "1" && ! -x "${STATS_BIN}" ]]; then
  echo "ERROR: stats binary is missing: ${STATS_BIN}" >&2
  exit 1
fi
if [[ "${RUN_PERF}" == "1" && ! -x "${PERF_BIN}" ]]; then
  echo "ERROR: perf binary is missing: ${PERF_BIN}" >&2
  exit 1
fi

csv_complete() {
  local path="$1"
  [[ -f "${path}" && "$(wc -l < "${path}")" -eq 2 ]]
}

run_phase() {
  local phase="$1"
  local binary="$2"
  local loops="$3"
  local threads="$4"
  local output_dir="$5"
  local tag="$6"
  shift 6

  local csv_path="${output_dir}/${phase}/${tag}.csv"
  local log_path="${output_dir}/logs/${tag}__${phase}.log"
  if [[ "${SKIP_EXISTING}" == "1" ]] && csv_complete "${csv_path}"; then
    log "SKIP ${phase}/${tag}"
    return
  fi
  rm -f "${csv_path}"
  log "START ${phase}/${tag} loops=${loops} threads=${threads}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%q ' env -u TRIBASE_TRACE -u EDGE_DEVICE_ENABLED OMP_NUM_THREADS="${threads}" \
      "${binary}" "$@" --loop "${loops}" --csv "${csv_path}" | tee -a "${MASTER_LOG}"
    echo | tee -a "${MASTER_LOG}"
    return
  fi
  if env -u TRIBASE_TRACE -u EDGE_DEVICE_ENABLED OMP_NUM_THREADS="${threads}" \
      "${binary}" "$@" --loop "${loops}" --csv "${csv_path}" >"${log_path}" 2>&1; then
    if ! csv_complete "${csv_path}"; then
      log "FAIL ${phase}/${tag}: expected one CSV row"
      return 1
    fi
    log "DONE ${phase}/${tag}"
  else
    local rc=$?
    log "FAIL ${phase}/${tag}: rc=${rc}, log=${log_path}"
    return "${rc}"
  fi
}

run_configuration() {
  local dataset="$1"
  local output_dir="$2"
  local perf_loops="$3"
  local tag="$4"
  shift 4

  if [[ "${RUN_STATS}" == "1" ]]; then
    run_phase stats "${STATS_BIN}" "${STATS_LOOPS}" "${THREADS}" \
      "${output_dir}" "${tag}" "$@"
  fi
  if [[ "${RUN_PERF}" == "1" ]]; then
    run_phase perf "${PERF_BIN}" "${perf_loops}" "${THREADS}" \
      "${output_dir}" "${tag}" "$@"
  fi
}

for dataset in "${DATASETS[@]}"; do
  dimension="${DIMENSIONS[${dataset}]}"
  nlist="${NLISTS[${dataset}]}"
  nprobe="${FIXED_NPROBE:-${NPROBES[${dataset}]}}"
  scope="${SCOPE:-${SCOPES[${dataset}]}}"
  pivot_values="${PIVOT_COUNTS:-${PIVOTS[${dataset}]}}"
  perf_loops="${PERF_LOOPS:-${PERF_REPEATS[${dataset}]}}"
  output_dir="${OUT_ROOT}/${dataset}"
  mkdir -p "${output_dir}"/{stats,perf,logs,manifests}

  base="${BENCH_ROOT}/${dataset}/origin/${dataset}_base.fvecs"
  query="${BENCH_ROOT}/${dataset}/origin/${dataset}_query.fvecs"
  ivfflat_index="${BENCH_ROOT}/${dataset}/index/v9_nlist_${nlist}_metric_l2_opt_0_subk_15_subNprobeRatio_1_mp_global_affine_fps_P0_seed0.index"
  triangle_index="${BENCH_ROOT}/${dataset}/index/v9_nlist_${nlist}_metric_l2_opt_1_subk_15_subNprobeRatio_1_mp_per_list_pca_P0_seed0.index"

  if [[ "${DRY_RUN}" != "1" ]]; then
    for required in "${base}" "${query}" "${ivfflat_index}"; do
      if [[ ! -f "${required}" ]]; then
        echo "ERROR: required input is missing: ${required}" >&2
        exit 1
      fi
    done
  fi

  log "DATASET ${dataset}: D=${dimension} nlist=${nlist} nprobe=${nprobe} scope=${scope} P=${pivot_values} perf_loops=${perf_loops}"
  common=(
    --benchmarks_path "${BENCH_ROOT}"
    --dataset "${dataset}"
    --input_format fvecs
    --output_format bin
    --metric l2
    --k "${K}"
    --nq "${NQ}"
    --nlist "${nlist}"
    --nprobes "${nprobe}"
    --cache
    --signature_precision "${SIGNATURE_PRECISION}"
    --opt_levels OPT_TRIANGLE
  )

  for pivot_count in ${pivot_values}; do
    if [[ ! "${pivot_count}" =~ ^[0-9]+$ ]] || ((pivot_count < 1 || pivot_count > 512)); then
      echo "ERROR: ${dataset} has invalid pivot count: ${pivot_count}" >&2
      exit 1
    fi

    if ((pivot_count == 1)); then
      tag="triangle_P1"
      args=(
        "${common[@]}"
        --multipivot_modes none
        --multipivot_scope per_list
        --multipivot_method pca
        --pivot_counts 1
        --from_index "${ivfflat_index}"
      )
    else
      if [[ "${DRY_RUN}" != "1" && ! -f "${triangle_index}" ]]; then
        echo "ERROR: Triangle index was not produced by P=1: ${triangle_index}" >&2
        exit 1
      fi
      tag="pca_${scope}_P${pivot_count}"
      args=(
        "${common[@]}"
        --multipivot_modes projection
        --multipivot_scope "${scope}"
        --multipivot_method pca
        --pivot_counts "${pivot_count}"
        --pivot_seed "${PIVOT_SEED}"
        --pivot_manifest "${output_dir}/manifests/${tag}.csv"
        --from_index "${triangle_index}"
      )
    fi
    run_configuration "${dataset}" "${output_dir}" "${perf_loops}" "${tag}" "${args[@]}"
  done
done

if [[ "${DRY_RUN}" != "1" ]]; then
  python3 - "${OUT_ROOT}" "${RUN_STATS}" "${RUN_PERF}" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
have_stats = sys.argv[2] == "1"
have_perf = sys.argv[3] == "1"
key_fields = (
    "dataset", "nlist", "nprobe", "opt_level", "multipivot_mode",
    "multipivot_scope", "multipivot_method", "pivot_count", "pivot_seed",
    "signature_precision", "simi_ratio",
)

def read_phase(directory):
    rows = {}
    if not directory.exists():
        return rows
    for path in sorted(directory.glob("*.csv")):
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                key = tuple(row.get(field, "") for field in key_fields)
                if key in rows:
                    raise SystemExit(f"duplicate configuration in {directory}: {key}")
                rows[key] = row
    return rows

def number(row, field):
    try:
        return float(row.get(field) or 0)
    except ValueError:
        return 0.0

stats = {}
perf = {}
for dataset_dir in sorted(path for path in root.iterdir() if path.is_dir()):
    stats.update(read_phase(dataset_dir / "stats"))
    perf.update(read_phase(dataset_dir / "perf"))

keys = stats.keys() & perf.keys() if have_stats and have_perf else stats.keys() if have_stats else perf.keys()
fields = [
    "dataset", "nlist", "nprobe", "pivot_count", "multipivot_mode",
    "multipivot_scope", "multipivot_method", "signature_precision", "recall",
    "latency_ms", "qps", "triangle_prune_pct", "multipivot_prune_pct",
    "overall_prune_pct", "query_signature_us_per_query",
    "candidate_decision_us_per_query", "other_us_per_query",
    "query_signature_share", "candidate_decision_share", "other_share",
]
with (root / "summary.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for key in sorted(keys):
        stat = stats.get(key, {})
        timing = perf.get(key, stat)
        latency_ms = 1000.0 / float(timing["qps"]) if float(timing.get("qps") or 0) else 0.0
        candidate = number(stat, "candidate_distance_computations")
        total = candidate * number(stat, "pruning_speedup")
        triangle_pruned = number(stat, "tri") + number(stat, "tri_large")
        multipivot_checks = number(stat, "multipivot_checks")
        multipivot_pruned = number(stat, "multipivot_pruned")
        writer.writerow({
            "dataset": timing.get("dataset", stat.get("dataset", "")),
            "nlist": timing.get("nlist", stat.get("nlist", "")),
            "nprobe": timing.get("nprobe", stat.get("nprobe", "")),
            "pivot_count": timing.get("pivot_count", stat.get("pivot_count", "")),
            "multipivot_mode": timing.get("multipivot_mode", stat.get("multipivot_mode", "")),
            "multipivot_scope": timing.get("multipivot_scope", stat.get("multipivot_scope", "")),
            "multipivot_method": timing.get("multipivot_method", stat.get("multipivot_method", "")),
            "signature_precision": timing.get("signature_precision", stat.get("signature_precision", "")),
            "recall": timing.get("recall", stat.get("recall", "")),
            "latency_ms": latency_ms,
            "qps": timing.get("qps", ""),
            "triangle_prune_pct": 100 * triangle_pruned / total if total else 0,
            "multipivot_prune_pct": (
                100 * multipivot_pruned / multipivot_checks
                if multipivot_checks else 0
            ),
            "overall_prune_pct": 100 * (1 - candidate / total) if total else 0,
            "query_signature_us_per_query": stat.get("query_signature_us_per_query", ""),
            "candidate_decision_us_per_query": stat.get("candidate_decision_us_per_query", ""),
            "other_us_per_query": stat.get("other_us_per_query", ""),
            "query_signature_share": stat.get("query_signature_share", ""),
            "candidate_decision_share": stat.get("candidate_decision_share", ""),
            "other_share": stat.get("other_share", ""),
        })
print(f"merged {len(keys)} configurations -> {root / 'summary.csv'}")
PY
fi

log "FINISHED results=${OUT_ROOT}"
