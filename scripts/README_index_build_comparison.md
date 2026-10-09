# Index construction comparison

Run from the repository root:

```bash
python3 scripts/run_index_build_comparison.py \
  --data-root /mnt/nvme/wxy/benchmarks \
  --log-dir logs/index_build_eight \
  --pivot-count 16 \
  --cpus 0-31 --threads 32 --repeats 3
```

Without `--datasets`, the default eight datasets are Fashion-MNIST
(`fasion_mnist_784`), MillionSong (`msong_holdout`), SIFT1M, GloVe25, HandOutlines,
StarLightCurves, DBpedia1536 (`dbpedia1536m_holdout`), and SIFT1B. NUS-WIDE and
SpaceV are excluded. SIFT1B defaults to nlist=32768; the script checks
`sift1b/origin/sift1b_base.bvecs`, `sift1b/raw/bigann_base.bvecs`, then the
origin fvecs file. Use `--sift1b-dir /path/raw` to override that input directory.
SIFT1B still uses this benchmark's whole-matrix resident-float/native-clustering
path described below, not the streaming IVF+HNSW construction path.

To run only SIFT1M, add `--datasets sift1m`.

Multiple datasets: `--datasets "sift1m,msong_holdout,nuswide"`. Each dataset uses
its configured nlist (1000, 1000, 512 respectively). Use `--nlist` to override.
Global/per-list use the same P; P includes the center, so P=16 means 15 projection
coordinates. Use separate invocations/output roots to compare different P.

Default layout: `DATA_ROOT/DATASET/origin/DATASET_base.fvecs`. For another input
file use `--datasets custom --base-file /path/base.fvecs --nlist 1000`.
`--base-file` also accepts bvecs/i8bin, decoded into float in allocated RAM.
This benchmark loads the whole base matrix; it does not exercise the streaming
or IVF+HNSW builder used for billion-vector inputs. Its build timings apply to
the resident-memory native IVF construction path.

The script builds only the perf benchmark target. Use `--skip-build` with an
existing binary or `--bin /path/index_build_benchmark`. `--dry-run` prints planned
commands without compiling/building indexes. Every repeat needs a fresh/empty
directory; completed repeats can be resumed. Original dataset indexes are read
neither for clustering nor as a build cache. All new indexes are in the log root.

## Five methods and shared-IVF protocol

Every repeat freshly loads data, trains native IVF centroids with the existing
`Index::train`, and assigns/sorts vectors with `Index::add`. It saves a bare IVF
and independently reloads that same IVF for each method (reload is timed but
excluded from algorithm construction time):

- IVF: centroid training + assignment/list construction.
- Tribase: the OPT_TRIANGLE variant used in previous comparisons; adds centroid
  distance/radius arrays. This does not construct OPT_ALL/SubNN structures.
- PPD: full-dimensional PCA and all transformed candidate vectors; base IVF
  plus `ppd.sidecar` form its logical index. Default PCA sample count=0 means all
  database vectors. `--ppd-train-samples N` explicitly selects a sampled variant.
- Per-list PCA: Triangle plus per-list U/signatures/residual preparation.
- Global PCA: Triangle plus existing global U/signatures/residual preparation
  (basis learned from IVF centroids, per the production implementation).

`--pca-without-triangle` disables Triangle preparation for the two PCA methods.
No IRLS, query loading, ground-truth calculation or search is performed.

This controls clustering across methods. The total is the shared-IVF protocol
cost, not five independent native monolithic builds; Triangle radii are added
after assignment rather than fused into `Index::add`.

## Timing columns

- `input_load_seconds`: separate input decoding/read time, excluded from build.
- `ivf_train_seconds`, `ivf_add_seconds`: fresh clustering and assignment/list
  construction; `common_ivf_build_seconds` is their sum.
- `extra_build_seconds`: wall time spent preparing that method's structures.
- `total_build_seconds`: common IVF build + extra preparation.
- `pca_selection_seconds`, `signature_build_seconds`: existing wall-clock
  metadata counters, diagnostics inside extra build time (do not add again).
- `index_reload_seconds`: isolated convenience I/O; excluded from build total.
- `total_save_seconds`: saving the final logical index. For PCA/Triangle this is
  the rich index save; for PPD it is IVF save + sidecar save; for IVF the bare
  index save. Internal temporary-IVF saves are not charged to rich-index totals.
- `build_plus_save_seconds`: total build + final logical-index save, excluding
  input load and convenience reload.
- `logical_index_bytes`: size of the produced logical index, including both
  files for PPD. The sidecar is this reproduction's storage format.

Save timers measure serialization and ordinary buffered writes/close; they do
not force fsync or cold-cache disk I/O. Use algorithm build time for the primary
construction-cost comparison and report save time separately.

Each repeat's `build_times.csv` has all stage times. `all_runs.csv` combines
repeats; `summary.csv` gives medians and min/max. Medians of separate fields need
not add up, so use the directly computed total column for total-build plots.
