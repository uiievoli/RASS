# Build and run the complete ten-dataset recall experiment

```bash
scripts/build_and_run_all_10_datasets.sh \
  --data-root /path/to/data \
  --log-dir /path/to/log/all10_corrected \
  --cpus 0-55 --threads 56 \
  --build-jobs 16 --phase all \
  --perf-repeats 3 --warmup-loops 1 \
  --build-batch-vectors 262144 --coarse-hnsw-query-batch 8192
```

The entry point incrementally compiles stats-disabled and stats-enabled query
binaries, prepares/reuses indexes, runs the measurements, validates the CSVs,
and generates latency/QPS/overall-pruning vs recall PNGs (recall > 0.9).
Dataset files must already be downloaded. `--dry-run` prints all commands.
`--skip-build` uses existing binaries; `--skip-plots` needs no matplotlib.
All runner options are listed by `--help`.

## Input directories

For the eight fvecs datasets:

```text
DATA_ROOT/<dataset>/origin/<dataset>_base.fvecs
DATA_ROOT/<dataset>/origin/<dataset>_query.fvecs
```

The query program loads or generates the corresponding ground truth. Dataset
directory names are case sensitive; Fashion uses the existing spelling
`fasion_mnist_784`. `--data-root` is their common parent directory.

For the two large datasets, the runner reuses existing origin files, or creates
symlinks to the following raw files if origin is absent:

```text
DATA_ROOT/sift1b/raw/bigann_base.bvecs
DATA_ROOT/sift1b/raw/bigann_query.bvecs
DATA_ROOT/sift1b/raw/gnd/idx_1000M.ivecs
DATA_ROOT/spacev1b/raw/base.1B.i8bin
DATA_ROOT/spacev1b/raw/query.30K.i8bin
DATA_ROOT/spacev1b/raw/groundtruth.30K.i32bin
```

Override their raw directories with `--sift1b-dir DIR` and `--spacev-dir DIR`.
No full-size byte-to-float data files are generated. Query.cpp decodes bytes
in batches during construction and uses native float Index/IVF storage and
search, as in the other datasets. SpaceV values are signed int8.

Every selected input is checked before index construction. `input_manifest.csv`
records file-header vector counts/dimensions, query counts and configuration
provenance. SpaceV actually has 1,402,020,720 vectors and 29,316 queries.

HandOutlines D=2710 and StarLightCurves D=1025 are marked as having a suspected
label coordinate. The runner uses the input and ground truth as supplied;
it does not silently strip dimensions. To benchmark the standard label-free
datasets, prepare those files and regenerate ground truth first. Dimension
checks prevent reuse of an index made with a different dimensionality.

## Default comparison configurations

All methods use L2 and share IVF centroids/list membership/candidate order.
P includes the center; dynamic's maximum projected dimension is Pmax - 1.

| Dataset | nlist | k / nq | PCA10 scope/P | Static scope/P | Dynamic scope/Pmax |
|---|---:|---|---|---|---|
| nuswide | 512 | 1 / all | global/50 | per_list/2 | per_list/32 |
| fasion_mnist_784 | 256 | 1 / all | global/64 | per_list/48 | per_list/96 |
| msong_holdout | 1000 | 1 / all | per_list/42 | per_list/34 | per_list/96 |
| sift1m | 1000 | 1 / all | per_list/16 | per_list/23 | per_list/40 |
| glove25 | 1024 | 1 / all | per_list/8 | global/14 | global/26 |
| HandOutlines | 32 | 1 / all | global/271 | global/7 | global/64 |
| StarLightCurves | 128 | 1 / all | global/103 | global/18 | global/104 |
| dbpedia1536m_holdout | 1000 | 1 / all | global/128 | per_list/128 | per_list/256 |
| sift1b | 32768 | 10 / 10000 | global/16 | global/16 | global/16 |
| spacev1b | 32768 | 10 / 29316 | global/11 | global/11 (reference) | global/11 |

PCA10 retains the historical approximately-10% presets, not an exact universal
rounding formula. SpaceV P11 is a reference configuration, not a fine-sweep
optimum; it is identified in the manifest and plot title. Configuration arrays
are editable in `run_all_10_datasets.sh`.

SpaceV nprobe is 8/16/32/64/128/256/512/1024. SIFT1B nprobe is
32/64/128/256/512/1024. The other sweeps retain the corrected eight-dataset
settings, with HandOutlines using 1/2/3/4/5/7/10. Stats covers the same nprobes
as perf, providing complete pruning-vs-recall measurements.

## Index reuse

1. Regular datasets reuse the `recall_shared_v2_*` indexes when their provenance
   matches. Otherwise an existing compatible bare/Triangle native IVF supplies
   the clustering; missing PCA metadata is built from that same IVF.
2. SIFT1B and SpaceV each keep one native global PCA rich index. Baseline,
   Triangle, static prefixes and dynamic search load that one payload.
3. An existing compatible large bare/Triangle IVF is upgraded when the rich
   index is missing. If no IVF exists, the program reuses saved centroids when
   available, otherwise trains with IVF+HNSW and adds vectors in bounded batches.
4. `--sift1b-index FILE` or `--spacev-index FILE` explicitly selects a compatible
   native float v10 global PCA index, including one built at a larger P.
5. Dedicated old int8 indexes are incompatible. Auto-discovered incompatible
   caches are preserved and a separate `_float_ram.index` is built. An invalid
   explicitly requested index produces a clear error.

Header checks read only a few bytes. Full loading/validation happens in the
query program. Reuse assumes indexes belong to the supplied, unchanged dataset.
No index is deleted automatically. The binary format remains native v10.

## Timing and results

Only search is timed. Each nprobe/method gets an untimed warmup, and perf uses
three independent processes with rotated method order. The median batch
duration determines QPS and reciprocal-throughput latency. Recall/r2 evaluation
time is exported separately. Stats retains stage timings and all prune counters.
HandOutlines uses 200 perf loops per process; the other loop counts are described
in `README_recall_sweep.md` (SpaceV uses 1). Stages sum worker wall times and
cannot directly be added to recover parallel wall-clock latency.

Canonical results: `perf/<dataset>/<method>.csv`, `stats/<dataset>/<method>.csv`.
Raw repeats: `perf_repeats/`. Summaries: `perf_summary.csv`, `stats_summary.csv`,
`measurement_quality.csv`. Commands/resources: `build/`, per-run logs,
`.command`, `.resources`, `.time`, plus `environment.log`.

Identical static configurations reuse measured results (SIFT1B and SpaceV's
PCA10/reference-static presets), avoiding redundant runs of the same method.
Resume requires matching timing protocol, command, binary hash, index fingerprint
and CSV configuration. Use a new log directory to preserve earlier experiments.

Plots are written to `plots_recall_gt09/`:

- `latency_recall_gt09.png`
- `qps_recall_gt09.png`
- `overall_prune_rate_recall_gt09.png`

The existing `run_all_8_datasets.sh` and `plot_all8_recall_gt09.py` entry points
continue to default to the previous eight-dataset subset.
