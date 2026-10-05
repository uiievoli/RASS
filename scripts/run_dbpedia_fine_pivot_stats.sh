#!/usr/bin/env bash
# Complete the DBpedia fine-pivot stats sweep from an existing Pmax PCA index.
# Each nprobe/pivot chunk is committed independently, so rerunning resumes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-}"
LOG_DIR="${LOG_DIR:-}"
STATS_BIN="${STATS_BIN:-${ROOT}/build-stats/bin/query}"
INDEX_PATH="${INDEX_PATH:-}"
CPU_SET="${CPU_SET:-0-31}"
THREADS="${THREADS:-}"
NPROBES="${NPROBES:-150 160 175 200}"
PIVOT_COUNTS="${PIVOT_COUNTS:-4 8 16 32 $(seq -s ' ' 48 4 256)}"
CHUNK_SIZE="${CHUNK_SIZE:-16}"
PIVOT_SEED="${PIVOT_SEED:-20261005}"
PMAX="${PMAX:-256}"
TARGET_ONLY=0
DRY_RUN=0
VERBOSE=0

usage() {
  cat <<'EOF'
Usage:
  scripts/run_dbpedia_fine_pivot_stats.sh \
    --data-root DIR --log-dir DIR [options]

Required:
  --data-root DIR       Root containing dbpedia1536m_holdout/.
  --log-dir DIR         Output directory for CSV, logs and manifests.

Options:
  --stats-bin PATH      ENABLE_STATS=ON query binary.
  --index PATH          Existing per-list PCA Pmax index. By default:
                        DATA_ROOT/dbpedia1536m_holdout/index/
                        v10_nlist_1000_metric_l2_opt_1_subk_15_
                        subNprobeRatio_1_mp_per_list_pca_P256_
                        seed20261005.index
  --cpus LIST           taskset CPU list (default 0-31).
  --threads N           OpenMP threads; defaults to CPU binding size.
  --nprobes LIST        Quoted/CSV list (default "150 160 175 200").
  --pivot-counts LIST   Quoted/CSV list; default is the 57-value fine sweep.
  --chunk-size N        Pivot configurations per process (default 16).
  --pmax N              Pmax stored in the loaded index (default 256).
  --pivot-seed N        Index seed namespace (default 20261005).
  --target-only         Only run nprobe=160, active P=128.
  --verbose             Pass --verbose to query.
  --dry-run             Validate and print commands without executing.
  -h, --help            Show this help.

This script never builds or changes an index. It uses --load_index and writes
the pivot manifest under LOG_DIR, avoiding writes to the dataset result path.
EOF
}

while (($#)); do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --stats-bin) STATS_BIN="$2"; shift 2 ;;
    --index) INDEX_PATH="$2"; shift 2 ;;
    --cpus) CPU_SET="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --nprobes) NPROBES="$2"; shift 2 ;;
    --pivot-counts) PIVOT_COUNTS="$2"; shift 2 ;;
    --chunk-size) CHUNK_SIZE="$2"; shift 2 ;;
    --pmax) PMAX="$2"; shift 2 ;;
    --pivot-seed) PIVOT_SEED="$2"; shift 2 ;;
    --target-only) TARGET_ONLY=1; shift ;;
    --verbose) VERBOSE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$DATA_ROOT" ]] || { echo "--data-root is required" >&2; exit 2; }
[[ -n "$LOG_DIR" ]] || { echo "--log-dir is required" >&2; exit 2; }
[[ "$CHUNK_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo "invalid --chunk-size" >&2; exit 2; }
[[ "$PMAX" =~ ^[1-9][0-9]*$ ]] && ((PMAX >= 2)) || {
  echo "invalid --pmax" >&2; exit 2;
}
[[ "$PIVOT_SEED" =~ ^[0-9]+$ ]] || { echo "invalid --pivot-seed" >&2; exit 2; }
command -v taskset >/dev/null || { echo "taskset is required" >&2; exit 1; }

if [[ -z "$THREADS" ]]; then
  THREADS="$(taskset -c "$CPU_SET" nproc)" || {
    echo "invalid CPU set: $CPU_SET" >&2; exit 2;
  }
fi
[[ "$THREADS" =~ ^[1-9][0-9]*$ ]] || { echo "invalid --threads" >&2; exit 2; }

DATA_ROOT="${DATA_ROOT%/}"
LOG_DIR="${LOG_DIR%/}"
if [[ -z "$INDEX_PATH" ]]; then
  INDEX_PATH="${DATA_ROOT}/dbpedia1536m_holdout/index/"
  INDEX_PATH+="v10_nlist_1000_metric_l2_opt_1_subk_15_subNprobeRatio_1_"
  INDEX_PATH+="mp_per_list_pca_P${PMAX}_seed${PIVOT_SEED}.index"
fi

if ((TARGET_ONLY)); then
  NPROBES="160"
  PIVOT_COUNTS="128"
fi
NPROBES="${NPROBES//,/ }"
PIVOT_COUNTS="${PIVOT_COUNTS//,/ }"
read -r -a PROBES <<<"$NPROBES"
read -r -a PIVOTS <<<"$PIVOT_COUNTS"

for value in "${PROBES[@]}" "${PIVOTS[@]}"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "invalid numeric value: $value" >&2; exit 2; }
done
for value in "${PIVOTS[@]}"; do
  ((value <= PMAX)) || { echo "active P=$value exceeds index Pmax=$PMAX" >&2; exit 2; }
done

if ((!DRY_RUN)); then
  [[ -x "$STATS_BIN" ]] || { echo "Missing stats binary: $STATS_BIN" >&2; exit 1; }
  [[ -s "$INDEX_PATH" ]] || { echo "Missing Pmax index: $INDEX_PATH" >&2; exit 1; }
  [[ -f "$DATA_ROOT/dbpedia1536m_holdout/origin/dbpedia1536m_holdout_query.fvecs" ]] || {
    echo "Missing DBpedia query vectors under $DATA_ROOT" >&2; exit 1;
  }
fi

mkdir -p "$LOG_DIR/chunks" "$LOG_DIR/manifests"

log() {
  local line="[$(date -Is)] $*"
  echo "$line"
  ((DRY_RUN)) || echo "$line" >>"$LOG_DIR/run.log"
}

run_bound() {
  if ((DRY_RUN)); then
    printf '%q ' env -u TRIBASE_TRACE -u EDGE_DEVICE_ENABLED \
      OMP_NUM_THREADS="$THREADS" OMP_PROC_BIND=close OMP_PLACES=cores \
      taskset -c "$CPU_SET" "$@"
    printf '\n'
  else
    env -u TRIBASE_TRACE -u EDGE_DEVICE_ENABLED \
      OMP_NUM_THREADS="$THREADS" OMP_PROC_BIND=close OMP_PLACES=cores \
      taskset -c "$CPU_SET" "$@"
  fi
}

csv_complete() {
  local path="$1" nprobe="$2"; shift 2
  [[ -s "$path" ]] || return 1
  python3 - "$path" "$nprobe" "$@" <<'PY'
import csv, sys
path, nprobe, *pivots = sys.argv[1:]
want={(int(nprobe),int(p)) for p in pivots}
try:
    with open(path,newline="") as f:
        got={(int(float(r["nprobe"])),int(float(r["pivot_count"])))
             for r in csv.DictReader(f)}
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if got == want else 1)
PY
}

log "CONFIG data=$DATA_ROOT index=$INDEX_PATH cpus=$CPU_SET threads=$THREADS nprobes='$NPROBES' pmax=$PMAX chunks=$CHUNK_SIZE"

for nprobe in "${PROBES[@]}"; do
  for ((begin=0, chunk_id=0; begin<${#PIVOTS[@]}; begin+=CHUNK_SIZE, ++chunk_id)); do
    chunk=("${PIVOTS[@]:begin:CHUNK_SIZE}")
    first="${chunk[0]}"
    last="${chunk[$((${#chunk[@]} - 1))]}"
    tag="nprobe${nprobe}_chunk$(printf '%02d' "$chunk_id")_P${first}-${last}"
    csv_path="$LOG_DIR/chunks/${tag}.csv"
    partial="$csv_path.partial"
    log_path="$LOG_DIR/chunks/${tag}.log"
    manifest="$LOG_DIR/manifests/${tag}.csv"

    if csv_complete "$csv_path" "$nprobe" "${chunk[@]}"; then
      log "SKIP complete $tag"
      continue
    fi

    cmd=(
      "$STATS_BIN"
      --benchmarks_path "$DATA_ROOT"
      --dataset dbpedia1536m_holdout
      --input_format fvecs --output_format bin --metric l2 --k 1
      --nlist 1000 --nprobes "$nprobe" --loop 1
      --signature_precision float32
      --opt_levels OPT_TRIANGLE
      --multipivot_modes projection
      --multipivot_scope per_list
      --multipivot_method pca
      --pivot_counts "$PMAX"
      --active_pivot_counts "${chunk[@]}"
      --projection_block_size 0
      --pivot_seed "$PIVOT_SEED"
      --load_index "$INDEX_PATH"
      --pivot_manifest "$manifest"
      --csv "$partial"
    )
    ((VERBOSE)) && cmd+=(--verbose)

    log "START $tag"
    if ((DRY_RUN)); then
      run_bound "${cmd[@]}"
      continue
    fi
    rm -f "$partial" "$log_path.partial"
    if run_bound "${cmd[@]}" >"$log_path.partial" 2>&1; then
      csv_complete "$partial" "$nprobe" "${chunk[@]}" || {
        log "FAILED incomplete CSV $partial"; exit 1;
      }
      mv "$partial" "$csv_path"
      mv "$log_path.partial" "$log_path"
      log "DONE $tag"
    else
      rc=$?
      log "FAILED $tag rc=$rc; inspect $log_path.partial"
      exit "$rc"
    fi
  done
done

# Rebuild a deterministic combined CSV from completed chunks.
if ((DRY_RUN)); then
  log "DRY-RUN complete"
  exit 0
fi
python3 - "$LOG_DIR" "$NPROBES" "$PIVOT_COUNTS" <<'PY'
import csv, glob, os, sys
root, probes, pivots = sys.argv[1:]
want={(int(n),int(p)) for n in probes.split() for p in pivots.split()}
by_key={}
for path in sorted(glob.glob(os.path.join(root,"chunks","*.csv"))):
    with open(path,newline="") as f:
        for row in csv.DictReader(f):
            key=(int(float(row["nprobe"])),int(float(row["pivot_count"])))
            if key in want:
                by_key[key]=row
seen=set(by_key)
if seen != want:
    missing=sorted(want-seen)
    raise SystemExit(f"combined result incomplete; missing {missing[:10]}")
rows=[by_key[key] for key in sorted(by_key)]
target=os.path.join(root,"dbpedia_fine_pivot_stats.csv")
with open(target,"w",newline="") as f:
    writer=csv.DictWriter(f,fieldnames=rows[0].keys())
    writer.writeheader(); writer.writerows(rows)
print(f"wrote {target} rows={len(rows)}")
PY

log "COMPLETE combined=$LOG_DIR/dbpedia_fine_pivot_stats.csv"
