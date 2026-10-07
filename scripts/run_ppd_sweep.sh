#!/usr/bin/env bash
set -euo pipefail

# Required: INDEX, QUERY, SIDECAR, OUT_DIR.
# Optional: GROUNDTRUTH, NPROBES, BLOCKS, K, NQ, TRAIN_SAMPLES, LOOP,
#           THREADS, CPU_SET, PPD_BIN, TRIANGLE=1, REBUILD_SIDECAR=1.

: "${INDEX:?set INDEX to an existing Tribase IVF index}"
: "${QUERY:?set QUERY to an fvecs query file}"
: "${SIDECAR:?set SIDECAR to the PPD sidecar path}"
: "${OUT_DIR:?set OUT_DIR to the result directory}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PPD_BIN="${PPD_BIN:-${ROOT}/build-perf/bin/ppd_query}"
GROUNDTRUTH="${GROUNDTRUTH:-}"
NPROBES="${NPROBES:-4 8 16 32 64 128}"
BLOCKS="${BLOCKS:-1 2 4 8 16 32 64}"
K="${K:-10}"
NQ="${NQ:-1000}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-65536}"
LOOP="${LOOP:-5}"
THREADS="${THREADS:-32}"
CPU_SET="${CPU_SET:-}"

[[ -x "${PPD_BIN}" ]] || {
  echo "Missing ppd_query binary: ${PPD_BIN}" >&2
  exit 1
}
mkdir -p "${OUT_DIR}"

read -r -a nprobe_args <<<"${NPROBES}"
read -r -a block_args <<<"${BLOCKS}"

prefix=(env OMP_NUM_THREADS="${THREADS}" OMP_PROC_BIND=close OMP_PLACES=cores)
[[ -n "${CPU_SET}" ]] && prefix+=(taskset -c "${CPU_SET}")

first=1
for block in "${block_args[@]}"; do
  output="${OUT_DIR}/ppd_B${block}.csv"
  args=(
    "${PPD_BIN}"
    --index "${INDEX}"
    --query "${QUERY}"
    --sidecar "${SIDECAR}"
    --nprobes "${nprobe_args[@]}"
    --k "${K}"
    --nq "${NQ}"
    --block-size "${block}"
    --train-samples "${TRAIN_SAMPLES}"
    --loop "${LOOP}"
    --csv "${output}"
  )
  [[ -n "${GROUNDTRUTH}" ]] && args+=(--groundtruth "${GROUNDTRUTH}")
  [[ "${TRIANGLE:-0}" == 1 ]] && args+=(--triangle)
  if ((first)) && [[ "${REBUILD_SIDECAR:-0}" == 1 ]]; then
    args+=(--rebuild-sidecar)
  fi
  echo "[$(date -Is)] PPD B=${block} -> ${output}"
  "${prefix[@]}" "${args[@]}"
  first=0
done
