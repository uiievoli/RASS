# Global PCA stage-cost ablation

Run five variants on the **same resident float32 Global PCA index**, query set,
coarse centroids, nprobe, k=1 and Triangle filtering:

| Method | Query projection | Bound | Direction order | Quantity |
|---|---|---|---|---|
| `no_signature` | None | Triangle only | None | 0 |
| `pd_only` | Fixed prefix | Squared projected distance only | PCA prefix | Fixed cap |
| `residual` | Fixed prefix | Projected distance + residual gap | PCA prefix | Fixed cap |
| `selection` | Entire configured pool | Same residual-aware LB | Query-global MM | Fixed cap |
| `adaptive` | Entire configured pool | Same residual-aware LB | Same query-global MM | Existing Global list-level cost/exploration policy, 0…cap |

`pd_only` does not read candidate residuals/radii or calculate query/candidate
residual energy or sqrt. It uses a candidate-parallel AVX2 kernel, with a scalar
tail/fallback. Its query path computes centered projections without accumulating
center distance or projected energy. Original vectors still supply exact distances
for survivors. No persisted index format changes; the existing default search
behavior is unchanged unless the new query-only flags are explicitly set.

## Meaning of 64 and stopping

Defaults: **64 projected directions**, plus the anchor (P=65), block size 8.
GloVe25 has only 25 directions, so both its pool and cap are clipped to 25 (P=26).
This clipping is printed and written into `protocol.json`.

The no-signature baseline uses zero directions, although it shares the same
resident index. The other three nonadaptive variants retain a fixed cap of 64;
there is no cost-aware budget reduction. All filtering variants use the same
block checkpoints: a scalar candidate can stop once pruned; an eight-candidate
SIMD batch stops when all lanes are pruned. Otherwise it continues to its cap.
They do not maintain/compact a multi-stage active list.

Adaptive additionally chooses a smaller budget per visited list, using the
existing Global MM quantity predictor (including first-list exploration). It
is **not** a new per-stage marginal-gain oracle. All full-pool query coordinates
are currently computed before MM ordering, including adaptive. Thus its
potential savings concern filtering/verification; this benchmark does not claim
lazy query-signature savings for adaptive MM.

With pool=64 and cap=64, selection permutes the same complete subspace. Its
final LB and pruning capability should agree with the canonical prefix, up to
floating-point and live-heap effects. Only early block exits, locality and
selection overhead can change execution cost. No monotonic five-step speedup
is guaranteed. To test actual *subset selection*, run an additional controlled
experiment with `--pool-directions 128 --fixed-directions 64`. Every method
still shares one bank/IVF partition, but selection/adaptive must project the
larger pool and pay that online cost.

## Run

The reference directory supplies prepared `DATASET/queries.fvecs`,
`DATASET/groundtruth.i32bin`, and `protocol.json` with `profiles`.
The default is `logs/qpivot_dynamic`, covering eleven non-1B datasets.
Default nprobe is each profile's `fixed_nprobe`; query counts, normalization,
and perf loop counts also follow that profile.

```bash
python3 scripts/run_qpivot_stage_ablation.py \
  --reference-root logs/qpivot_dynamic \
  --log-dir logs/qpivot_stage_ablation64 \
  --pool-directions 64 --fixed-directions 64 --block-size 8 \
  --cpus 0-31 --threads 32
```

For another server with prepared reference files and relocated indexes:

```bash
python3 scripts/run_qpivot_stage_ablation.py \
  --reference-root /path/to/qpivot_dynamic \
  --index-root /path/to/qpivot_dynamic/indexes \
  --index-root /path/to/data \
  --log-dir /path/to/log/qpivot_stage_ablation64 \
  --cpus 0-55 --threads 56
```

`--index-root` is repeatable. Compatible sources are found in root/DATASET,
root/DATASET/index, root/DATASET/index_cache, root/indexes/DATASET, and root.
One existing Global PCA+Triangle source is required per dataset; this script
never silently reclusters. If its bank is too small, `build_global_pca_index`
constructs a wider PCA bank into the new log directory, retaining the source's
IVF vectors, centroids and list assignments. Adequate larger banks are reused
without copying/rebuilding the IVF structure. Normalization is matched using
the reference profile and the existing `unit` filename convention. Existing
indexes are not modified or deleted.

Useful options:

```bash
# One dataset / alternative operating point / shorter pilot
python3 scripts/run_qpivot_stage_ablation.py \
  --log-dir logs/qpivot_stage_ablation64_sift \
  --datasets sift1m --nprobes 34 --nq 1000 --repeats 3 --stats-repeats 3

# Print the commands without writing results or building/running anything
python3 scripts/run_qpivot_stage_ablation.py \
  --log-dir logs/qpivot_stage_ablation64 --dry-run

# Rebuild plots from completed CSVs only
python3 scripts/run_qpivot_stage_ablation.py \
  --log-dir logs/qpivot_stage_ablation64 --summarize-only
```

Two Release binaries are built: `build-predictor-perf` (stats OFF) and
`build-predictor-stats` (stats ON), LTO OFF, by default. Override using
`--build-perf`, `--build-stats`; `--skip-build` requires matching binaries.
Default: 3 perf repeats and 3 stats repeats; stats uses one loop per repetition.
Method order is shuffled within each repeat, warmups default to two.
`--loops` overrides perf loop count, `--warmups` controls warmups.

## Results and interpretation

Per dataset:

- `perf.csv`: raw batch wall latency/QPS and measured Recall@1.
- `stats.csv`: raw pruning counts and nonoverlapping worker-stage timings.
- `stats.csv.budgets.csv`: actual list-level direction budgets, including zero.
- phase `.done` files: executable/CSV/histogram hashes for safe resume.
- logs and exact executed command JSON files.

`summary.csv` includes median/min/max perf latency, speedup relative to
no-signature, pruning, full distances/query, mean selected directions,
zero-budget list share, timing CV, and each stage's raw worker time, share,
and wall-latency allocation. `figures/` contains per-dataset/nprobe figures
and a five-method overview with stage stacks, inverted pruning curves and
min–max whiskers. These use fresh raw data, without fitted/smoothed curves.

Stage definitions:

1. **Verification**: exact original-vector distances and heap updates. This uses
   the exact timer in both the generic Triangle scan and the projection scan.
   Remaining generic scan/loop overhead belongs to Other.
2. **Bound**: lower-bound scan, including mask/loop/survivor dispatch overhead.
3. **Projection**: query signature work excluding MM selection, plus lazy
   candidate-path query projection bookkeeping where applicable.
4. **Selection**: MM scoring/order selection; it is subtracted from its enclosing
   query-signature timer to avoid double counting.
5. **Other**: remaining worker time, including coarse/triangle setup and quantity
   prediction/control not covered by the above timers.

Worker stages sum to worker time. With parallel queries, these are not wall
latencies. Like `plot_stacked_scope_cost.py`, bar height is measured median perf
latency; segments allocate that height by the median worker-stage shares.
Both raw values and allocated values are retained. Fine-grained instrumentation
can perturb stage shares; perf results remain independently measured.

Every search variant's Top-1 ID and exact distance are checked against a native
Triangle search with the same index/nprobe; actual recall is computed against
the prepared GT. Stats candidate conservation and the fixed-budget histograms
are checked before publishing a completed result. Inputs and saved geometry
are shared, and no index-format change is required.

If by “64 pivots” you mean **P=64 including the anchor**, use
`--pool-directions 63 --fixed-directions 63` instead.

## Validation performed

- Both perf and stats executables compiled successfully.
- Two new StageAblation tests passed: scalar/AVX2 PD masks, and native search
  agreement for Top-1/Top-10, nprobe=1/5/16, blocks=1/8/64, with a 64-direction bank.
- Five existing query-selection correctness tests passed.
- Synthetic CLI/runner tests passed for all five methods, separate perf/stats,
  three nprobe values, two repeats, histogram validation, summary/plots, safe
  resume and summarize-only. A separate run verified dimension clipping and
  construction of a wider bank while keeping the IVF partition.
- An extended existing test,
  `QueryEnergy.OriginalQuantityPredictorAndStagedMMRemainExact`, failed its
  `progressive_stages > 0` coverage assertion: the current policy executed zero
  such stages. Its result-agreement checks passed. This benchmark uses
  `projection_progressive=0` and does not exercise that stage-major path.
- No full real-dataset experiment was launched as part of this implementation.
