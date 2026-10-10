# PPD on angular/cosine datasets

This experiment compares GloVe-200-angular and Landmark-DINO-768-cosine with
PPD using the existing normalized-L2 IVF partitions. Unit-vector L2 ranking is
equivalent to cosine/angular ranking: `||q-x||^2 = 2 - 2*cos(q,x)`. This does not
extend PPD to arbitrary unnormalized inner product.

`ppd_query` now accepts `--normalize` and official `.i32bin`/`.ivecs` ground truth.
Normalization is outside search timing and applies to queries only; source IVF
vectors must already be normalized. A sampled norm check rejects obvious raw
base indexes. Query and PPD share the same official ID-recall loader. Legacy
ground-truth `.bin` remains supported. `--train-samples 0` requests full-database
PCA and rejects sampled sidecars. One warm-up is explicit even when loop=1.
The search timer covers the whole search call, excluding loading, sidecar
construction, normalization and recall calculation.

## Configurations

Both datasets: nlist=1024, k=1, float32 resident IVF, official IDs, 32 threads on
CPU 0-31, one warm-up, three measured loops, three independent repetitions with
rotated method order. All official queries are used by default (10000 and 1000).

* GloVe: nprobe=80,96,128,192,256,384,512; global reference P=21; Pmax=64.
* Landmark: nprobe=6,7,8,12,16,24,32,64,128,256,512; global reference P=78; Pmax=160.
* Static Pmax and original dynamic (block=2) run in global and per-list scope.
* IVF and Triangle are controls. PPD uses Triangle, all D PCA directions learned
  from all indexed base vectors, with block sizes 16/32/64.

The static reference uses the existing dimensional reference configuration,
not a claim that it is the optimum P on these new datasets. Progressive frozen/
updated strategies are not included in this comparison.

## Run

```bash
cmake --build build-perf --target query ppd_query -j 8
cmake --build build-stats --target query -j 8
python3 scripts/run_cosine_ppd_comparison.py \
  --log-dir logs/cosine_ppd_full \
  --data-root datasets/cosine_supplement --threads 32 --cpus 0-31
```

Existing normalized rich global and per-list PCA banks must be present.
No clustering or original index rewriting occurs. Full PPD sidecars are built
once and reused across all blocks and repetitions, with source-index SHA-256
bindings. To reuse a pilot's sidecars, add `--sidecar-root logs/cosine_ppd_pilot`.
Sidecars require about 0.95 GB for GloVe and 2.34 GB for Landmark.

A quick real-data preflight:

```bash
python3 scripts/run_cosine_ppd_comparison.py \
  --log-dir logs/cosine_ppd_pilot --nq 64 --nprobes 128 512 \
  --repeats 1 --loops 1 --blocks 16 32
```

Completed method runs have `.done` markers and can resume with identical options
and binary hashes. Changes to protocol or binaries require a fresh output root.
Commands and a driver snapshot are saved. Recompiling while running is rejected.
Completed experiments automatically generate `summary.csv`, `all_runs.csv` and
three PNG/PDF recall curves (`recall > .9`): QPS, batch time/query and overall
pruning. Use `--analyze-only` with the same dataset/block/repeat settings to replot.

## Accounting and interpretation

PPD candidates = Triangle pruned + PPD early exits + full transformed distances.
Overall prune rate includes Triangle in both numerator and denominator. For
MP/IVF, a separate stats run supplies `total_candidates`, so zero-P dynamic
bypasses are counted. Optional `--dump-list-stats` can produce very large files;
they are unnecessary for the normal comparison.

For PPD, reaching full D is a full *rotated* distance, reused as the final distance;
an early exit may already have consumed many dimensions. Therefore pruning rate
alone is not equivalent to work saved. `dimension_work_fraction` additionally
reports accumulated PPD coordinate work divided by `candidates * D`. Projection
and stage timers are worker-time diagnostics, not parallel batch latency.

The driver checks deterministic counters and matching recall at each nprobe.
Floating-point PCA ties must be investigated if official ID recall differs; such
runs are retained and flagged rather than silently treated as equivalent.

Small-fixture preflight passed for both datasets with k=1/10, block=8/16/32/64
and nprobe=1/4/32: IDs identical to ordinary IVF search, transformed distances
within float tolerance, and candidate accounting consistent.
