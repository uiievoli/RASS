# Rebuild PPD for the QPivot recall comparison

`run_ppd_qpivot_reference.py` runs PPD only. Its reference is the **raw** QPivot
experiment directory (for example `log/qpivot_dynamic`), not a figure directory.
The default dataset set is every dataset in that directory's `protocol.json`:
the current run contains all 11 non-1B datasets, including NUS-WIDE, HandOutlines,
GloVe-200, Landmark-DINO, DBpedia-1536 and DBpedia-3072. Neither 1B dataset is run.

The reference directory must retain:

- `protocol.json` and preferably `summary.csv` for recall cross-checks;
- `DATASET/queries.fvecs` and `DATASET/groundtruth.i32bin`;
- the exact Global IVF/PCA indexes used by that experiment.

The runner inherits nlist, query count, query prefix, groundtruth, normalization,
nprobe grid, timed loops and repeat count. Threads must match the reference
(56 for the current experiment). PPD uses B16, Triangle first, seed 0 and
`--train-samples 0` (full database training). Every nprobe receives two untimed
warmup batches per independent repetition by default. PPD uses a full-dimensional
PCA transform; the source index's QPivot Pmax does not limit PPD's dimension.

Fresh PPD sidecars are built inside the new output directory using
`--rebuild-sidecar`. Existing sidecars elsewhere are never reused or overwritten.
The source IVF/PCA index is reused without changing clustering or candidate
generation. Completed builds and CSVs can resume with identical arguments;
use a **new output directory** to force another complete rebuild.

```bash
REF_ROOT=/path/to/log/qpivot_dynamic
OUT_ROOT=/path/to/log/ppd_qpivot_rerun
mkdir -p "$OUT_ROOT"
nohup python3 -u scripts/run_ppd_qpivot_reference.py \
  --reference-root "$REF_ROOT" \
  --index-root "$REF_ROOT/indexes" \
  --log-dir "$OUT_ROOT" \
  --cpus 0-55 --threads 56 --build-jobs 16 \
  > "$OUT_ROOT/driver.log" 2>&1 < /dev/null &
echo "PPD driver PID=$!"
```

Compilation is automatic in `build-ppd-reference-perf`, Release with stats and
LTO disabled. Activate the working compiler/Conda environment first. To use an
existing compatible performance binary, add `--skip-build --build-dir DIR`
where `DIR/bin/ppd_query` is the executable. A relocated index root must contain
`DATASET/<the original index filename>`; filenames and headers are checked.

Use `--datasets sift1m dbpedia3072m_holdout` for a subset. `--dry-run` preflights
the real inputs and prints build/experiment commands without running them.

Outputs: `summary.csv` with latency, QPS, pruning, variability and work counts;
`protocol.json` with input/binary bindings; per-dataset `build.log`,
`full_training.ppd`, `perf_repN.csv` and `perf_repN.log`; `complete` on success.
The script checks candidate accounting and repeated recall/counters. When the
reference summary is available, recall disagreement larger than 1e-6 stops the
run for investigation. It does not smooth or delete measured nodes.

Running PPD later with matching settings removes query/thread configuration
mismatches, but does not make server load identical across separate runs.
