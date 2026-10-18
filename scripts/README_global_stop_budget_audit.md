# Global stopping-budget audit

This experiment reuses the existing SIFT1M Global PCA Pmax=40 index. It compares
the PCA-order Dynamic policy with MM-order QPivot. It does not rebuild an index
or replace its persisted projection basis or candidate signatures.

```bash
python3 scripts/run_global_stop_budget_audit.py \
  --index logs/sift1m_same_index_local_20261017/indexes/sift1m/global_pca_nlist1000_P40_seed0.index \
  --log-dir logs/sift1m_global_stop_budget_new \
  --nq 32 --nprobes 34 --capture-threads 32 --replay-threads 1
```

For a larger random sample and four concurrent timing workers bound to separate
physical cores on this machine:

```bash
python3 scripts/run_global_stop_budget_audit.py \
  --index logs/sift1m_same_index_local_20261017/indexes/sift1m/global_pca_nlist1000_P40_seed0.index \
  --log-dir logs/sift1m_global_stop_budget_128_new \
  --nq 128 --seed 20261018 --nprobes 34 \
  --capture-threads 32 --replay-threads 4 --replay-cpus 0,2,4,6
```

CPU/core mappings must be checked on another machine before reusing this binding.
Concurrent timing workers measure under a different load from a one-worker run;
do not pool the two runs as if their timing environments were identical.

The existing `build-predictor-stats` and `build-predictor-perf` CMake directories
must be configured with a C++20 compiler supporting `<format>`; the script builds
the required targets unless `--skip-build` is supplied.

The stats binary captures each query's actual list-entry threshold, heap ID,
projection-cache state, chosen budget and MM permutation after two warmups.
The perf binary evaluates **every integer P=0..40**, then independently retimes
every budget in seven batches. All tested budgets must agree on the exact
replayed nearest-neighbor ID. P includes the anchor; P=0 disables MP filtering.
Triangle filtering determines the captured candidate interval. Empty intervals
are excluded from stopping-budget timing because they have no filtering work.

For each captured visit, define delta = selected_P - best_measured_P. Report
mean delta, empirical variance (divisor N), sample variance (divisor N-1), MAE,
RMSE, the proportion within 5% of minimum time, and replay time ratios. Report
per-query aggregates separately; queries are the resampling unit for confidence
intervals. A mean near zero alone does not imply accurate choices. No smoothing
or interpolation is applied to measurements.

Outputs:

- `query_list_results.csv`: raw conditional errors, timing CV and correctness.
- `conditional_gap_summary.csv`: all visits and CV<=0.1 subsets.
- `query_aggregates.csv`: per-query conditional budget-error aggregates.
- `nprobe*/capture.csv.*.visits.csv`: actual entry states.
- `nprobe*/capture.csv.orders.csv`: exact captured MM row permutations.
- `nprobe*/*.replay.csv.costs.csv`: all 41 measured budgets per nonempty visit.
- `protocol.json`: index/binary hashes and sampling/timing semantics.
- `figures/`: five PNG/PDF figures, query-level raw points, population estimates
  and confidence intervals. They are generated automatically after the audit.

To regenerate only the figures:

```bash
python3 scripts/plot_global_stop_budget_audit.py \
  --root logs/sift1m_global_stop_budget_128_new
```

Population estimates refer to the 10,000 official queries, sampled uniformly
without replacement. Queries, not query-list rows, are the bootstrap units.
The variance of query-mean errors and the variance across query-list errors are
reported separately. Figures do not synthesize individual optima for untested
queries. Query-sampling intervals do not cover all uncertainty from noisy
timing-based winner selection.

These are **conditional list-entry optima within the stored bank**, not a
jointly optimal whole-query budget sequence. Replaying independent alternatives
cannot propagate counterfactual cache and pruning-prior changes into later lists.
Global query projections already cached at entry are not charged again. QPivot's
one-time full projection and MM ordering are not part of the per-list optimum;
their total cost remains in the capture's separate stage statistics.
