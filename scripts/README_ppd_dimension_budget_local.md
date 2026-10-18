# Local PPD-informed fixed-budget experiment

The remote PPD dimension counts supply an exploratory fixed budget:

    mean = dimensions / (candidates - triangle_pruned)
    target L = ceil(mean), native pivot count P = L + 1

This uses candidates that actually enter PPD as the denominator. It is not
the dimension count averaged over candidates already removed by Triangle.
The remote mean is an experimentally informed budget, not a guarantee of
optimality or a deployable budget predictor validated on held-out queries.

`run_ppd_dimension_budget_local.py` compares the original static Global budget,
the target, and budgets near 0.75x/1.25x the target. It tests PCA prefix and
online MM, and reruns PPD-B16 locally. All use the exact same resident IVF and
query samples per dataset; only stored PCA metadata is extended if necessary.
Extending a bank preserves candidate generation but can relearn its PCA basis;
all controls then use that same extended bank. No K-means is retrained.

The MM direction pool is fixed to the largest tested budget for every MM
configuration. Thus changes in MM pool size versus historical experiments
must not be attributed solely to changes in its selected budget. PCA prefix
only computes its requested prefix. MM projection, scoring and sorting are
included in timing; offline list-moment preparation and PPD construction are
excluded. PPD is trained on the complete resident database with seed 0 and
kept in RAM, without creating another large sidecar on disk.

```bash
python3 -u scripts/run_ppd_dimension_budget_local.py \
  --log-dir logs/ppd_dimension_budget_local_20261018 \
  --threads 32 --cpus 0-31 --repeats 5 --loops 5 --warmups 2 \
  --wait-for-root logs/qpivot_fixed_order_perf_20261018
```

Default scope: all 11 non-1B datasets from the existing order-comparison
protocol. The 32 sampled queries and fixed nprobe come from that protocol;
do not compare these absolute timings against the remote larger query batches.
Every method/budget is warmed up immediately before timing. Methods and budgets
are shuffled across repetitions. Native query results are checked against
Triangle; PPD must return the same IDs and distances within relative 2e-5
roundoff tolerance. These checks do not establish recall against official GT.

The runner can wait on a previous experiment's OS-backed lock to avoid CPU
contention. Separate default build directory: `build-ppd-budget-perf`.
Outputs include `budget_plan.json`, `protocol.json`, per-dataset build/run logs,
all raw repetitions, `summary.csv`, `budget_comparison.csv` and individual
QPS/latency-versus-budget PNG/PDF figures. All reported speedups compare local
measurements, not timings transferred from another server. No smoothing is used.
