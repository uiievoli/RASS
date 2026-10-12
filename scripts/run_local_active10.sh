#!/usr/bin/env bash
# Serial local experiments: nine existing datasets plus OpenAI-3072, no SIFT1B.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${LOG_DIR:-$ROOT/logs/local_active10_$(date +%Y%m%d)}"
DATA="${DATA_ROOT:-/mnt/nvme/wxy/benchmarks}"
DB="${DBPEDIA_ROOT:-/mnt/nvme/wxy}"
COS="${COSINE_ROOT:-$ROOT/datasets/cosine_supplement}"
OPENAI="${OPENAI_ROOT:-$ROOT/datasets/openai_supplement}"
WAIT_PID="${WAIT_PID:-}"
THREADS="${THREADS:-32}"
CPUS="${CPU_SET:-0-31}"
mkdir -p "$OUT"
export MPLCONFIGDIR="$OUT/.matplotlib" XDG_CACHE_HOME="$OUT/.cache" MPLBACKEND=Agg
log() { echo "[$(date -Is)] $*"; }
status=0
# Input preparation can proceed while an existing benchmark finishes.
log "DOWNLOAD/PREPARE OpenAI-3072"
openai_ready=1
bash "$ROOT/scripts/download_current_datasets.sh" --data-root "$OPENAI" \
  --datasets dbpedia3072m_holdout > "$OUT/openai_download.log" 2>&1 || {
    log "OpenAI preparation failed; see $OUT/openai_download.log; existing datasets will still run"
    openai_ready=0; status=1
  }
if [[ -n "$WAIT_PID" ]]; then
  log "WAIT for existing experiment PID=$WAIT_PID (avoid competing CPU benchmarks)"
  python3 - "$WAIT_PID" <<'PY'
from pathlib import Path
import sys,time
path=Path('/proc')/sys.argv[1]/'stat'
def identity():
    try:
        fields=path.read_text().rsplit(')',1)[1].split()
        return None if fields[0]=='Z' else fields[19]
    except FileNotFoundError:
        return None
original=identity()
while original is not None and identity()==original:
    time.sleep(15)
PY
fi
log "START serial performance/stats tests; threads=$THREADS cpus=$CPUS"
common=(--threads "$THREADS" --cpus "$CPUS" --repeats 3 --warmups 1)
if ((openai_ready)); then
  log "OpenAI: recall calibration, fine static-P (step=4), peak refinement, all-method recall sweep"
  bash "$ROOT/scripts/run_openai3072_retrieval.sh" --data-root "$OPENAI" \
    --log-dir "$OUT/openai_fine" --stage all --scope both --pmax 512 --step 4 \
    --refine-radius 16 --nlist 1000 --loops 3 "${common[@]}" > "$OUT/openai_experiment.log" 2>&1 || status=1
fi
log "Nine existing datasets: baseline, triangle, best/max static, dynamic, frozen, updated, PPD"
python3 -u "$ROOT/scripts/run_progressive_10_datasets.py" --data-root "$DATA" \
  --dbpedia-root "$DB" --cosine-root "$COS" --log-dir "$OUT/main9" \
  --index-root "$OUT/indexes" --stage sweep --phase all --blocks 8 \
  "${common[@]}" > "$OUT/main9.log" 2>&1 || status=1
# PCA 10% is an independent reference, not an alias for the richer Pmax bank.
# Compute ceil(0.1*D) PCA axes + centroid and keep its outputs clearly named.
log "PREPARE true PCA-10% reference configurations"
python3 - "$ROOT" "$OUT" "$DATA" "$DB" "$COS" "$OPENAI" <<'PY'
import json,math,struct,sys
from pathlib import Path
repo,out,data,db,cos,op=map(Path,sys.argv[1:])
profiles=json.loads((repo/'scripts/progressive_10_profiles.json').read_text())
profiles.pop('sift1b',None)
for name,p in profiles.items():
    root=cos if p['normalize'] else db if name=='dbpedia1536m_holdout' else data
    directory=root if root.name==name else root/name
    with (directory/'origin'/f'{name}_base.fvecs').open('rb') as f:
        d=struct.unpack('<I',f.read(4))[0]
    count=max(2,math.ceil(d/10)+1)
    p['pmax']=max(p['pmax'],count)
    p['best_p']={'global':count,'per_list':count}
    p['best_status']='dimensional_PCA10_reference_not_tuned_best'
(out/'pca10_profiles.json').write_text(json.dumps(profiles,indent=2)+'\n')
source=out/'openai_fine/comparison_profile.json'
if source.exists():
    extra=json.loads(source.read_text())
    for p in extra.values():
        p['best_p']={'global':309,'per_list':309}
        p['best_status']='dimensional_PCA10_reference_not_tuned_best'
    (out/'openai_pca10_profile.json').write_text(json.dumps(extra,indent=2)+'\n')
PY
python3 -u "$ROOT/scripts/run_progressive_10_datasets.py" --data-root "$DATA" \
  --dbpedia-root "$DB" --cosine-root "$COS" --log-dir "$OUT/pca10_reference9" \
  --config "$OUT/pca10_profiles.json" --index-root "$OUT/indexes" \
  --stage sweep --phase all --methods static_best "${common[@]}" > "$OUT/pca10_reference9.log" 2>&1 || status=1
if [[ -f "$OUT/openai_pca10_profile.json" ]]; then
  python3 -u "$ROOT/scripts/run_progressive_10_datasets.py" --data-root "$OPENAI" \
    --log-dir "$OUT/openai_pca10_reference" --config "$OUT/openai_pca10_profile.json" \
    --index-root "$OUT/openai_fine/index_cache" --stage sweep --scope both --phase all \
    --methods static_best --loops 3 "${common[@]}" > "$OUT/openai_pca10_reference.log" 2>&1 || status=1
fi
log "FINISHED status=$status results=$OUT"
echo "$status" > "$OUT/pipeline.exitcode"
exit "$status"
