#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_ROOT="${BENCH_ROOT:-/mnt/nvme/wxy/benchmarks}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/logs/small-global-fine-confirm-$(date +%Y%m%d-%H%M%S)}"
BIN="${PERF_BIN:-${ROOT}/build-perf/bin/query}"
THREADS="${THREADS:-32}"
mkdir -p "${OUT_ROOT}"

run_range() {
  local dataset="$1" nlist="$2" nprobe="$3" loops="$4"; shift 4
  env -u TRIBASE_TRACE -u EDGE_DEVICE_ENABLED OMP_NUM_THREADS="${THREADS}" \
    "${BIN}" --benchmarks_path "${BENCH_ROOT}" --dataset "${dataset}" \
      --input_format fvecs --output_format bin --metric l2 --k 1 --nq 0 \
      --nlist "${nlist}" --nprobes "${nprobe}" --cache --loop "${loops}" \
      --opt_levels OPT_TRIANGLE --multipivot_scope global --multipivot_method pca \
      --multipivot_modes projection --pivot_counts 512 --active_pivot_counts "$@" \
      --signature_precision float32 --csv "${OUT_ROOT}/${dataset}.csv" \
      >"${OUT_ROOT}/${dataset}.log" 2>&1
}

fashion=(); for ((p=128; p<=168; p+=2)); do fashion+=("${p}"); done
starlight=(); for ((p=14; p<=32; ++p)); do starlight+=("${p}"); done
run_range fasion_mnist_784 256 3 20 "${fashion[@]}"
run_range StarLightCurves 128 3 100 "${starlight[@]}"

python3 - "${OUT_ROOT}" <<'PY'
import csv, sys
from pathlib import Path
root=Path(sys.argv[1])
with (root/'confirmed_best.csv').open('w', newline='') as out:
    w=csv.writer(out); w.writerow(['dataset','pivot_count','recall','latency_ms','qps'])
    for path in sorted(root.glob('*.csv')):
        if path.name == 'confirmed_best.csv': continue
        rows=list(csv.DictReader(path.open()))
        best=max(rows, key=lambda r: float(r['qps']))
        w.writerow([best['dataset'],best['pivot_count'],best['recall'],
                    1000/float(best['qps']),best['qps']])
PY
