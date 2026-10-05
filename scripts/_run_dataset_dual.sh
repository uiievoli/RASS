#!/usr/bin/env bash
# Internal runner. Dataset-specific scripts set DATASET and defaults before sourcing this file.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_ROOT="${BENCH_ROOT:-/mnt/nvme/wxy/benchmarks}"
STATS_BIN="${STATS_BIN:-${ROOT}/build-stats/bin/query}"
PERF_BIN="${PERF_BIN:-${ROOT}/build-perf/bin/query}"
PIVOT_COUNTS="${PIVOT_COUNTS:-${DEFAULT_PIVOT_COUNTS:-16 64 128}}"
METHODS="${METHODS:-pca}"
SCOPE="${SCOPE:-${DEFAULT_SCOPE:-per_list}}"
K="${K:-10}"
NQ="${NQ:-0}"
NLIST="${NLIST:-${DEFAULT_NLIST}}"
NPROBES="${NPROBES:-${DEFAULT_NPROBES}}"
STATS_LOOPS="${STATS_LOOPS:-1}"
PERF_LOOPS="${PERF_LOOPS:-5}"
PIVOT_SEED="${PIVOT_SEED:-0}"
SIGNATURE_PRECISION="${SIGNATURE_PRECISION:-float32}"
IRLS_MAX_ITER="${IRLS_MAX_ITER:-5}"
IRLS_RESIDUAL_FLOOR="${IRLS_RESIDUAL_FLOOR:-1e-6}"
BUILD_THREADS="${BUILD_THREADS:-${OMP_NUM_THREADS:-$(nproc)}}"
STATS_THREADS="${STATS_THREADS:-${OMP_NUM_THREADS:-$(nproc)}}"
PERF_THREADS="${PERF_THREADS:-${OMP_NUM_THREADS:-$(nproc)}}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"
BUILD="${BUILD:-0}"

TS="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="${OUT_DIR:-${ROOT}/logs/${DATASET}-dual-${TS}}"
mkdir -p "${OUT_DIR}"/{stats,perf,logs}
MASTER_LOG="${OUT_DIR}/run.log"

log() {
  echo "[$(date -Is)] $*" | tee -a "${MASTER_LOG}"
}

if [[ -n "${DATA_WARNING:-}" ]]; then
  log "WARNING: ${DATA_WARNING}"
fi

if [[ "${BUILD}" == "1" ]]; then
  cmake --fresh -S "${ROOT}" -B "${ROOT}/build-stats" \
    -DCMAKE_BUILD_TYPE=Release -DENABLE_STATS=ON -DCMAKE_CXX_COMPILER=g++-14
  cmake --build "${ROOT}/build-stats" --target query -j"${BUILD_THREADS}"
  cmake --fresh -S "${ROOT}" -B "${ROOT}/build-perf" \
    -DCMAKE_BUILD_TYPE=Release -DENABLE_STATS=OFF -DCMAKE_CXX_COMPILER=g++-14
  cmake --build "${ROOT}/build-perf" --target query -j"${BUILD_THREADS}"
fi

for binary in "${STATS_BIN}" "${PERF_BIN}"; do
  if [[ ! -x "${binary}" ]]; then
    echo "ERROR: missing query binary: ${binary}; rerun with BUILD=1" >&2
    exit 1
  fi
done

base="${BENCH_ROOT}/${DATASET}/origin/${DATASET}_base.fvecs"
query="${BENCH_ROOT}/${DATASET}/origin/${DATASET}_query.fvecs"
if [[ "${DRY_RUN}" != "1" ]]; then
  for input in "${base}" "${query}"; do
    if [[ ! -f "${input}" ]]; then
      echo "ERROR: missing ${input}" >&2
      exit 1
    fi
  done
fi

for P in ${PIVOT_COUNTS}; do
  if [[ ! "${P}" =~ ^[0-9]+$ ]] || ((P < 2 || P > 512)); then
    echo "ERROR: invalid MP pivot count '${P}', expected 2..512" >&2
    exit 1
  fi
done

csv_has_rows() {
  local csv="$1"
  [[ -f "${csv}" ]] || return 1

  local expected_rows
  expected_rows=$(( ${#nprobe_array[@]} + 1 ))
  [[ "$(wc -l < "${csv}")" -eq "${expected_rows}" ]]
}

run_phase() {
  local phase="$1"
  local binary="$2"
  local loops="$3"
  local threads="$4"
  local tag="$5"
  shift 5
  local csv="${OUT_DIR}/${phase}/${tag}.csv"
  local logfile="${OUT_DIR}/logs/${tag}__${phase}.log"

  if [[ "${SKIP_EXISTING}" == "1" ]] && csv_has_rows "${csv}"; then
    log "SKIP ${phase}/${tag}"
    return
  fi
  [[ "${SKIP_EXISTING}" == "1" ]] || rm -f "${csv}"
  log "START ${phase}/${tag}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%q ' env -u TRIBASE_TRACE -u EDGE_DEVICE_ENABLED OMP_NUM_THREADS="${threads}" \
      "${binary}" "$@" --loop "${loops}" --csv "${csv}" | tee -a "${MASTER_LOG}"
    echo | tee -a "${MASTER_LOG}"
    return
  fi
  if env -u TRIBASE_TRACE -u EDGE_DEVICE_ENABLED OMP_NUM_THREADS="${threads}" \
      "${binary}" "$@" --loop "${loops}" --csv "${csv}" >"${logfile}" 2>&1; then
    if ! csv_has_rows "${csv}"; then
      log "FAIL ${phase}/${tag}: CSV has no data"
      return 1
    fi
    log "DONE ${phase}/${tag}"
  else
    local rc=$?
    log "FAIL ${phase}/${tag}: rc=${rc}, log=${logfile}"
    return "${rc}"
  fi
}

run_pair() {
  local tag="$1"
  shift
  run_phase stats "${STATS_BIN}" "${STATS_LOOPS}" "${STATS_THREADS}" "${tag}" "$@"
  run_phase perf "${PERF_BIN}" "${PERF_LOOPS}" "${PERF_THREADS}" "${tag}" "$@"
}

read -r -a nprobe_array <<<"${NPROBES}"
common=(
  --benchmarks_path "${BENCH_ROOT}"
  --dataset "${DATASET}"
  --input_format fvecs
  --output_format bin
  --metric l2
  --k "${K}"
  --nq "${NQ}"
  --nlist "${NLIST}"
  --nprobes
  "${nprobe_array[@]}"
  --cache
  --signature_precision "${SIGNATURE_PRECISION}"
)

log "dataset=${DATASET} nlist=${NLIST} nprobes=${NPROBES} nq=${NQ} k=${K}"
log "methods=${METHODS} P=${PIVOT_COUNTS} scope=${SCOPE}"
log "stats_loops=${STATS_LOOPS} perf_loops=${PERF_LOOPS} stats_threads=${STATS_THREADS} perf_threads=${PERF_THREADS}"

run_pair ivfflat \
  "${common[@]}" \
  --opt_levels OPT_NONE \
  --multipivot_modes none \
  --multipivot_scope global \
  --multipivot_method affine_fps \
  --pivot_counts 0

ivfflat_index="${BENCH_ROOT}/${DATASET}/index/v9_nlist_${NLIST}_metric_l2_opt_0_subk_15_subNprobeRatio_1_mp_global_affine_fps_P0_seed0.index"
if [[ "${DRY_RUN}" != "1" && ! -f "${ivfflat_index}" ]]; then
  echo "ERROR: expected IVF-Flat index was not created: ${ivfflat_index}" >&2
  exit 1
fi

run_pair tribase \
  "${common[@]}" \
  --opt_levels OPT_TRIANGLE \
  --multipivot_modes none \
  --multipivot_scope per_list \
  --multipivot_method pca \
  --pivot_counts 1 \
  --from_index "${ivfflat_index}"

tribase_index="${BENCH_ROOT}/${DATASET}/index/v9_nlist_${NLIST}_metric_l2_opt_1_subk_15_subNprobeRatio_1_mp_per_list_pca_P0_seed0.index"
if [[ "${DRY_RUN}" != "1" && ! -f "${tribase_index}" ]]; then
  echo "ERROR: expected Tribase index was not created: ${tribase_index}" >&2
  exit 1
fi

for method in ${METHODS}; do
  for P in ${PIVOT_COUNTS}; do
    method_args=()
    if [[ "${method}" == "irls_pca" ]]; then
      method_args+=(--irls_max_iter "${IRLS_MAX_ITER}")
      method_args+=(--irls_residual_floor "${IRLS_RESIDUAL_FLOOR}")
    fi
    run_pair "${method}_${SCOPE}_P${P}" \
      "${common[@]}" \
      --opt_levels OPT_TRIANGLE \
      --multipivot_modes projection \
      --multipivot_scope "${SCOPE}" \
      --multipivot_method "${method}" \
      --pivot_counts "${P}" \
      --pivot_seed "${PIVOT_SEED}" \
      --from_index "${tribase_index}" \
      "${method_args[@]}"
  done
done

if [[ "${DRY_RUN}" != "1" ]]; then
  python3 - "${OUT_DIR}" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
keys = (
    "dataset", "nlist", "nprobe", "opt_level", "multipivot_mode",
    "multipivot_scope", "multipivot_method", "pivot_count", "pivot_seed",
    "signature_precision", "simi_ratio",
)

def read(directory):
    result = {}
    for path in sorted(directory.glob("*.csv")):
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                key = tuple(row.get(field, "") for field in keys)
                if key in result:
                    raise SystemExit(f"duplicate configuration: {key}")
                result[key] = row
    return result

def number(row, field):
    try:
        return float(row.get(field) or 0)
    except ValueError:
        return 0.0

stats = read(root / "stats")
perf = read(root / "perf")
if stats.keys() != perf.keys():
    raise SystemExit(
        f"stats/perf mismatch: missing perf={len(stats.keys()-perf.keys())}, "
        f"missing stats={len(perf.keys()-stats.keys())}"
    )

fields = [
    *keys, "triangle_prune_pct", "multipivot_prune_pct", "overall_prune_pct",
    "multipivot_checks", "multipivot_pruned", "candidate_distances",
    "pivot_distances",
    "query_signature_seconds", "candidate_decision_seconds", "other_seconds",
    "search_worker_seconds", "query_signature_us_per_query",
    "candidate_decision_us_per_query", "other_us_per_query",
    "query_signature_share", "candidate_decision_share", "other_share",
    "qps", "latency_ms", "query_time", "recall", "r2",
]
rows = []
for key in sorted(stats):
    stat, timing = stats[key], perf[key]
    candidate = number(stat, "candidate_distance_computations")
    total = candidate * number(stat, "pruning_speedup")
    tri = number(stat, "tri") + number(stat, "tri_large")
    checks = number(stat, "multipivot_checks")
    pruned = number(stat, "multipivot_pruned")
    qps = number(timing, "qps")
    rows.append({
        **dict(zip(keys, key)),
        "triangle_prune_pct": 100 * tri / total if total else 0,
        "multipivot_prune_pct": 100 * pruned / checks if checks else 0,
        "overall_prune_pct": 100 * (1 - candidate / total) if total else 0,
        "multipivot_checks": int(checks),
        "multipivot_pruned": int(pruned),
        "candidate_distances": int(candidate),
        "pivot_distances": int(number(stat, "pivot_distance_computations")),
        "query_signature_seconds": number(stat, "query_signature_seconds"),
        "candidate_decision_seconds": number(stat, "candidate_decision_seconds"),
        "other_seconds": number(stat, "other_seconds"),
        "search_worker_seconds": number(stat, "search_worker_seconds"),
        "query_signature_us_per_query": number(stat, "query_signature_us_per_query"),
        "candidate_decision_us_per_query": number(stat, "candidate_decision_us_per_query"),
        "other_us_per_query": number(stat, "other_us_per_query"),
        "query_signature_share": number(stat, "query_signature_share"),
        "candidate_decision_share": number(stat, "candidate_decision_share"),
        "other_share": number(stat, "other_share"),
        "qps": qps,
        "latency_ms": 1000 / qps if qps else 0,
        "query_time": number(timing, "query_time"),
        "recall": number(timing, "recall"),
        "r2": number(timing, "r2"),
    })

output = root / "summary.csv"
with output.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(f"merged {len(rows)} configurations -> {output}")
PY
fi

log "finished; summary=${OUT_DIR}/summary.csv"
