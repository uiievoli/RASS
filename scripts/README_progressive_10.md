# RASS progressive experiments: ten datasets

Entry point: `scripts/run_progressive_10_datasets.py`. Default runs now exclude
SIFT1B, NUS-WIDE and SpaceV: nine datasets remain, including GloVe-200-angular
and Landmark-DINO-768-cosine. SIFT1B is retained in the profile registry for
explicit separate runs with `--datasets sift1b`. HandOutlines is included as a small-workload boundary
case. Profiles are editable in `scripts/progressive_10_profiles.json`.

## Build in separate directories

```bash
cmake -S . -B build-progressive-perf -DCMAKE_BUILD_TYPE=Release -DENABLE_STATS=OFF
cmake --build build-progressive-perf --target query ppd_query -j 8
cmake -S . -B build-progressive-stats -DCMAKE_BUILD_TYPE=Release -DENABLE_STATS=ON
cmake --build build-progressive-stats --target query -j 8
```

The existing Conda-aware CMake discovers compilers/dependencies from the active
environment. C++20 `std::format`, Eigen, MKL and GTest are required. Configure
fresh directories after changing compilers. These directories do not overwrite
the binaries used by the separate ongoing cosine PPD experiment. Schedule
experiments serially rather than running concurrent benchmarks on the same CPUs.

## Independent query sets

Prepare original full-query groundtruth first. The native datasets require
`result/groundtruth_1.bin`; cosine datasets require official IDs; SIFT1B uses
`raw/gnd/idx_1000M.ivecs`. Base/query files must follow the common origin layout.

```bash
python3 scripts/prepare_progressive_query_split.py \
  --data-root /path/to/data \
  --cosine-root /path/to/cosine-data \
  --dbpedia-root /path/to/dbpedia-parent \
  --output-root /path/to/log/query_views
```

This deterministically creates 20% tuning / 80% evaluation query views. Query
IDs and groundtruth rows are selected together. Base vectors and original
indexes are shared by symlink, not copied. No vector normalization or label
removal is performed by splitting. Omit optional roots if all ten datasets live
below the same data root. A dataset directory itself is also accepted for the
DBpedia override. Use a fresh view output directory.

## Stage 1: fixed-nprobe/block diagnosis

Run the nine non-billion-scale datasets first:

```bash
python3 scripts/run_progressive_10_datasets.py \
  --data-root /path/to/log/query_views/tune \
  --log-dir /path/to/log/progressive_fixed \
  --index-root /path/to/shared-index-cache \
  --datasets fasion_mnist_784 msong_holdout sift1m glove25 \
    HandOutlines StarLightCurves dbpedia1536m_holdout \
    glove-200-angular landmark-dino-768-cosine \
  --stage fixed --phase all --blocks 2 4 8 16 --greedy-ablation \
  --threads 56 --cpus 0-55 --repeats 3
```

Fixed nprobe references are 7/16/34/50/4/4/160/512/64 in that order. They are
historical reference points; calibration on the selected queries is still
needed. The config may replace each `fixed_nprobes` list. `--scope both` diagnoses
both bases; default scope follows the profile. `--ppd-blocks 16 32 64` adds a PPD
block sweep. Default PPD uses B=16 and full-database PCA. To reduce the first
diagnostic, select `--methods baseline static_max dynamic frozen updated`.

The following are distinct methods:

* `dynamic_B*`: original one-budget-per-list dynamic selection.
* `frozen_B*`: progressive with frozen per-list survival prior, **preserving the
  original cross-list exploration schedule**.
* `updated_B*`: same schedule, updating the prior from completed current stages.
* `*_greedy_B*`: opt-in ablation retaining only the initial valid observation;
  no continued original exploration. This is the variant tested in the earlier
  three-dataset progressive comparison.

Original exploration uses the original streaming scanner, keeping its heap
updates and observations. Once exploration finishes, new projection blocks,
scores and residuals are acquired incrementally. With very few probes, all
visited lists may still be exploratory; report the measured progressive stage
count rather than claiming that a progressive phase necessarily occurred.

Choose the block using tuning queries, freeze the setting, and evaluate the
disjoint query set. A single chosen block shared across datasets is the simplest
primary configuration; per-dataset tuning must be disclosed.

## Stage 2: recall curves on evaluation queries

Example with the chosen block fixed to 8 (replace if tuning selects another):

```bash
python3 scripts/run_progressive_10_datasets.py \
  --data-root /path/to/log/query_views/eval \
  --log-dir /path/to/log/progressive_recall \
  --index-root /path/to/shared-index-cache \
  --datasets fasion_mnist_784 msong_holdout sift1m glove25 \
    HandOutlines StarLightCurves dbpedia1536m_holdout \
    glove-200-angular landmark-dino-768-cosine \
  --stage sweep --phase all --blocks 8 \
  --threads 56 --cpus 0-55 --repeats 3
```

The configured grids start around the historical recall=.9 region and include
high-recall points. All methods use the same grid; plots retain actual recall>.9
points. Adjust the JSON if evaluation queries shift the recall boundary. Without
query splitting, `--data-root /path/to/data` runs all official queries, but using
those same queries for tuning and evaluation is not an independent evaluation.

## SIFT1B separately

```bash
python3 scripts/run_progressive_10_datasets.py \
  --data-root /path/to/log/query_views/eval \
  --log-dir /path/to/log/progressive_sift1b \
  --index-root /path/to/shared-index-cache \
  --datasets sift1b --stage sweep --blocks 8 \
  --threads 56 --cpus 0-55 --repeats 3
```

This uses k=1, nlist=32768, global Pmax=16 and the current native float IVF index.
Old k=10 curves are not reused for this comparison; nprobe must be recalibrated.
The input files are byte-coded but the current production pipeline stores float
vectors in RAM. Use the large-memory server. PPD is disabled for SIFT1B by
default; `--include-sift1b-ppd` explicitly enables its full-dimensional sidecar.
Omitting `--datasets` runs the nine non-SIFT1B profiles. SIFT1B must be selected
explicitly with `--datasets sift1b`; no existing index/data files are deleted.

## Controls, caches and outputs

Controls are IVF, Triangle, historical static reference, static Pmax, original
dynamic and PPD. MP/PPD include Triangle. All use k=1, identical partitions per
comparison, float32 signatures and official query/groundtruth pairing. The two
cosine datasets use normalized L2; their static P=21/78 are dimensional reference
presets, not proven optima. SIFT1B P16/per-list is also untuned. Existing UCR input
coordinates/groundtruth are preserved; inspect label-coordinate provenance
before claiming a clean UCR embedding benchmark.

The script selects compatible native v10 indexes from each dataset's `index/`.
Missing banks are built under `--index-root` with PCA only, preserving a reusable
IVF source when available. The same rich bank serves every prefix/method in its
scope. A larger cached bank is allowed; static prefix residuals are prepared
outside timing. Optional `indexes: {"global": "/path/index", "per_list": ...}`
in a profile pins a specific bank. `--require-index` refuses construction.
`--dry-run` prints every planned command without writing or starting experiments.

Perf and stats use separate binaries; one warm-up, dataset-specific measured
loop counts, three rotated process repetitions. `--loops`, `--warmups`, `--nq`,
`--phase`, binary paths and CPU binding are configurable. A low-nq preflight is
not a full evaluation. Do not rebuild binaries during an active run. Protocol,
binary hashes, source-index metadata, exact commands and raw CSVs are retained.
An OS lock rejects concurrent runners using the same output directory.
Completed runs resume only under the same protocol and index state. Identical
static-reference/Pmax configurations are explicitly aliased.

`summary.csv` reports median QPS, batch time/query, repeat variation, recall,
overall pruning, selected mean P/zero-P share and instrumented worker stage
times. For query.cpp, the denominator includes `total_candidates`, including
zero-P bypasses. PPD supplies its counters in perf mode. `--dump-list-stats` adds
large per-query/list files for distribution analysis. Worker timing is not a
parallel perf critical-path decomposition. Optional single-thread runs provide
a distinct latency measurement for the smallest datasets.

Fixed-point runs add QPS/block figures. Sweep runs generate QPS/latency/pruning
recall figures in PNG/PDF. To summarize or replot:

```bash
python3 scripts/summarize_progressive_10.py /path/to/log/progressive_recall
```
