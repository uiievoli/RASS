# Global PCA / MM / Greedy fixed-budget comparison

```bash
python3 -u scripts/run_qpivot_order_comparison.py \
  --data-root /mnt/nvme/wxy/benchmarks \
  --dbpedia-root /mnt/nvme/wxy \
  --cosine-root datasets/cosine_supplement \
  --openai-root /mnt/nvme/wxy/dbpedia3072_supplement \
  --log-dir logs/qpivot_order_comparison \
  --nq 8 --threads 32 --cpus 0-31
```

The default set includes 11 datasets: NUS-WIDE, Fashion-MNIST, MillionSong,
SIFT1M, GloVe25, HandOutlines, StarLightCurves, DBpedia-1536, GloVe-200-angular,
Landmark-DINO-768, and DBpedia OpenAI-3072. Both 1B datasets are excluded.
Use `--datasets NAME ...` to select a subset. `--nq` controls a deterministic,
uniform query sample; each query uses complete candidate lists without
candidate subsampling. The default eight queries are a screening experiment;
increase to 32 or more for a stronger query sample.

Every dataset now rebuilds a fresh Global PCA index from raw base fvecs, using
`scripts/qpivot_performance_profiles.json` (SIFT1M Pmax=40, for example).
The index includes IVF training, Global U, candidate projection coordinates,
residuals and center norms. Outputs are isolated in `LOG_DIR/indexes/DATASET`.
Historical indexes, old centroids and previous result reuse are disabled;
`--index-root` and `--reuse-results` are rejected. Existing old files are preserved.
Use a new output directory for each experiment.

All selectors use this one bank and may only select from its M=Pmax-1 rows.
QPivot changes query-time ordering/budget, never the index basis. Both dataset
normalization and candidate signatures use the native Global builder. Before
measurement the runner reloads and validates all candidate signature tables.
No external dictionary is installed during evaluation.

For the previous Global Pmax and per-dataset checkpoint grids:

```bash
python3 scripts/run_qpivot_order_comparison.py \
  --log-dir logs/qpivot_global_pmax_fresh \
  --global-pool-config scripts/qpivot_performance_profiles.json \
  --nq 32 --threads 32 --cpus 0-31
```

At n=M all methods must return the same pruning count. This is checked in the
experiment analysis. Two PNG/PDF figures per dataset (selection and gain), plus
one combined selection overview, appear in `figures_fixed_pool/` when using
`--global-pool-config`. This is an offline selection-quality test, not latency.

Budget `L` counts PCA directions; the repository's pivot count is `P=L+1`,
including the center. Default checkpoints are `L=4,8,16,32,64,128`, clipped to
the available bank, with the smaller full-bank checkpoint retained. Each
checkpoint is a separate fixed-budget comparison, without dynamic quantities.

- **PCA**: original PCA direction prefix.
- **MM**: one order per query, scored over all probed lists with full list-size
  weights, matching the native MM policy. Uses stored list means of projections.
- **Greedy**: one order per query, forward selection maximizing actual cumulative
  pruning on frozen Triangle survivors. Ties use the sum of residual-aware
  squared lower bounds, then the smallest direction index. Greedy scans complete
  candidates and is an expensive offline empirical reference, not an optimum
  oracle or an online latency competitor.

For a fixed selected set, its internal order does not change the final lower
bound. These experiments compare the first L-element sets induced by each order.
All methods use the same residual-aware bound and strict `LB² > entry_tau²` test.

One native Triangle-only top-1 run records each list-entry threshold and
survivor interval. These remain frozen for every selector. The first list's
infinite threshold is retained. Thresholds are not replaced by the final
nearest-neighbor distance. Therefore these are controlled selection-quality
measurements, not end-to-end online QPS estimates.

Outputs:

- `protocol.json`: bank, M, nprobe, query IDs and budget conventions.
- `DATASET/visits.csv`: actual native list-entry thresholds and survivor ranges.
- `DATASET/orders.csv`: chosen direction prefixes for every query/method/budget.
- `DATASET/results.csv`: counts per query; exact-distance safety checks.
- `summary.csv`: candidate-count-weighted MP-only and overall prune rates.
- `figures/`: overview and separate dataset figures, PNG/PDF.
- `status.json` and `DATASET/run.log`: progress and failures.

MP-only pruning uses frozen Triangle survivors as the denominator. Overall
pruning includes Triangle and uses all visited candidates. The float32 stored
projections are accumulated in float64. The safety check flags material false
pruning beyond `2e-5 * max(1, exact_distance_squared)`; this is a floating-point
diagnostic tolerance rather than a proof of exact floating-point bounds.

`selection_seconds` is cumulative offline construction/evaluation time and
includes checkpoint reporting; do not use it as online selector latency.

Re-running the same command resumes completed datasets and rejects changes to
the query sampling/budget protocol. Use a new log directory to change the query
sample size. To regenerate figures:

```bash
python3 scripts/run_qpivot_order_comparison.py \
  --log-dir logs/qpivot_order_comparison --summarize-only
```
