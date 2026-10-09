#!/usr/bin/env bash
# Shared-IVF cosine/Angular experiments using unit-normalized L2 bounds.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$ROOT/datasets/cosine_supplement}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/cosine_supplement_$(date +%Y%m%d-%H%M%S)}"
SCOPE=global
NLIST=1024
SEARCH_K=1
PROBES='1 2 4 8 16 32 64 128 256 512'
DRY_RUN=0
FORWARD=()
while (($#)); do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --scope) SCOPE="$2"; shift 2 ;;
    --nlist) NLIST="$2"; shift 2 ;;
    --k) SEARCH_K="$2"; shift 2 ;;
    --nprobes) PROBES="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; FORWARD+=("$1"); shift ;;
    -h|--help)
      echo 'Usage: run_cosine_supplement.sh [--data-root DIR] [--log-dir DIR] [--scope global|per_list|both] [--nlist N] [--k N] [--nprobes "LIST"]'
      echo 'Other options (cpus, threads, repeats, warmup, phase, skip-build, skip-plots, etc.) are forwarded to build_and_run_all_10_datasets.sh.'
      exit ;;
    *) FORWARD+=("$1"); shift ;;
  esac
done
[[ "$SCOPE" =~ ^(global|per_list|both)$ ]] || { echo 'Invalid --scope' >&2; exit 2; }
[[ "$NLIST" =~ ^[1-9][0-9]*$ && "$SEARCH_K" =~ ^[1-9][0-9]*$ ]] || { echo 'nlist/k must be positive' >&2; exit 2; }
scopes=("$SCOPE"); [[ "$SCOPE" != both ]] || scopes=(global per_list)
for scope in "${scopes[@]}"; do
  output="$LOG_DIR/$scope"
  # This is an experiment profile, not an estimate of the unknown best static P.
  # Both static labels reuse the 10%-dimension reference measurement.
  if ((DRY_RUN)); then config="/tmp/tribase-cosine-profile-${scope}-$$.json"; else mkdir -p "$output"; config="$output/experiment_config.json"; fi
  python3 - "$config" "$scope" "$NLIST" "$SEARCH_K" "$PROBES" <<'PY'
import json,sys
from pathlib import Path
path,scope,nlist,k,probes=sys.argv[1:]
data={}
for dataset,reference,maximum,metric in [('glove-200-angular',21,64,'angular'),('landmark-dino-768-cosine',78,160,'cosine')]:
 data[dataset]=dict(NLIST=int(nlist),SEARCH_K=int(k),QUERY_COUNT=0,NPROBES=probes,
  PCA_P=reference,PCA_SCOPE=scope,BEST_P=reference,BEST_SCOPE=scope,
  DYNAMIC_PMAX=maximum,DYNAMIC_SCOPE=scope,LOOPS=3,NORMALIZE=1,
  EXTERNAL_GROUNDTRUTH='result/official_neighbors.i32bin',SOURCE_METRIC=metric)
Path(path).write_text(json.dumps(data,indent=2)+'\n')
PY
  "$ROOT/scripts/build_and_run_all_10_datasets.sh" --data-root "$DATA_ROOT" --log-dir "$output" --config "$config" \
    --datasets glove-200-angular,landmark-dino-768-cosine "${FORWARD[@]}"
done
