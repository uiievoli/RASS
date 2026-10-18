# Fixed-budget PCA prefix versus online MM performance

`run_qpivot_order_performance.py` supplements the frozen-threshold selection
experiment with native online query timings. Only PCA prefix and MM are tested.

Reuse the original `qpivot_global_pmax_order_all11_20261017` directory. Its
`protocol.json` supplies the exact Global indexes, selected query samples,
direction pools, budgets and nprobe. No new index is built or saved. An optional
`--index-root` remaps the same original filenames after moving servers.

MM projects the configured pool, scores directions using the size-weighted
means of the probed lists, selects L directions and performs native filtering.
PCA prefix projects and uses the first L directions. Query projection, online
MM scoring/sorting, lower bounds, distance verification and heap operations
are included in end-to-end time. List moment construction is offline and is
excluded. Dynamic pivot quantity selection is disabled for both methods.
The optional selection-pool field is runtime-only; index serialization is unchanged.

Native search updates heap thresholds normally. Therefore these performance
runs are distinct from the frozen-candidate/frozen-threshold pruning-quality
experiment. Every timed result must match the native Triangle result IDs and
distances on the same query samples; this is a result-agreement test, not an
official groundtruth recall measurement.

```bash
python3 scripts/run_qpivot_order_performance.py \
  --reference-root logs/qpivot_global_pmax_order_all11_20261017 \
  --log-dir logs/qpivot_fixed_order_perf_20261018 \
  --threads 32 --cpus 0-31 --repeats 5 --loops 5 --warmups 2
```

The runner builds `qpivot_fixed_order_perf` in Release with stats/LTO disabled.
Use `--skip-build` for an already compiled target. Query parallelism is bounded
by the sample count: this reference contains only 32 queries per dataset, so
more than 32 requested threads will not increase effective query concurrency.
Batch startup and scheduling overhead remain part of these small-batch timings.

Methods/budgets are shuffled independently each repetition, and every case
gets untimed warmup batches immediately before measurement. Outputs include
each raw repeat (`DATASET/perf.csv`), a median/min/max/CV `summary.csv`, and
individual/combined latency and QPS versus direction-budget PNG/PDF figures.
No smoothing or removal of measurements is performed. Latency is batch wall
time divided by the number of queries, the reciprocal of aggregate throughput.

Rerunning identical arguments resumes completed datasets; changing input,
binary or timing settings requires a fresh output directory. A failed dataset
is rerun in full. Use `--dry-run` to preflight and print commands, or
`--summarize-only` to regenerate available summaries and figures.
