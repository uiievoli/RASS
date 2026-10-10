# Original-predictor progressive comparison

Two opt-in modes have been added. Existing `--projection_dynamic` behavior is
unchanged when `--projection_progressive` is not specified. Index format stays v10;
PCA geometry and signatures are reused without reconstruction.

```
--projection_dynamic --projection_block_size 8 --projection_progressive frozen
--projection_dynamic --projection_block_size 8 --projection_progressive updated
```

Supported experiment path: L2, ordinary float32 PCA signatures, no microblocks,
no separate `projection_prefix_length`. For cosine inputs, use existing normalized
L2 banks with `--normalize`. The block size counts projected directions; reported
P includes the center, except P=0 means no projection filtering.

## Shared predictor and execution

The original policy preserves the strongest observed log-survival per dimension
from preceding lists:

```
lambda = min(previous observed log(1 - MP_prune_rate) / projected_dims)
rho_hat(delta) = clamp(1 - exp(lambda * delta), 0, .9999)
```

`frozen` holds lambda fixed inside a list. `updated` replaces it after each
completed stage using dimension-weighted smoothing:

```
observed = log(clamp(remaining / before, .0001, 1)) / delta
lambda = (lambda * (rank + block_size) + observed * delta)
         / (rank + block_size + delta)
```

Here rank is the number of directions processed *before* that stage. This
smoothing rule is a declared engineering choice, not the paper's unspecified
estimator. Both modes feed overall list pruning back into the same preceding-list
prior used by the original implementation.

At each stage, proceed only if predicted exact savings exceed incremental cost:

```
saved = R * rho_hat(delta) * D
cost = newly_materialized_query_dims * D + R * (delta + 6 + 3)
```

These are the original scalar-coordinate cost units, with an additional three
state-field maintenance units per survivor. They are **not** calibrated cycles or
nanoseconds. Six is the existing residual-check cost assumption. The model's
accuracy and actual state-maintenance cost are part of the experiment.

Without any preceding-list observation, both modes run the original dynamic
scanner for calibration, including its exploratory prefix and streaming heap
updates. This prefix may be full Pmax for per-list. A one-shot stage at an initial
loose threshold would underestimate pruning and unfairly suppress later stages.
After a valid original observation, stages use the configured block size.

Scores, residual energies and candidate indexes are packed and compacted in
place. Each newly acquired direction is added once; pruned candidates leave the
state permanently. AVX2 evaluates eight packed survivors using SoA signature
gathers. Query residual is computed once per stage. Global projection readiness
is shared across all lists of one query; local readiness resets for each list.
Survivors receive exact distance once when filtering stops. No UB early-exact
policy or global/local switching is introduced in this comparison.

`multipivot_checks` counts unique candidates that enter projection filtering;
`progressive_candidate_updates` counts candidate-direction work across stages.
They have different meanings. Exact-seeded and bypassed candidates are included
in exact distance counts. Overall pruning is computed from
`(Triangle + MP pruned)/(Triangle + MP pruned + exact)`.

## Run

```bash
cmake --build build-perf --target query -j 8
cmake --build build-stats --target query -j 8
python3 scripts/run_progressive_comparison.py \
  --log-dir logs/progressive_compare_new \
  --nq 128 --sift-nq 2000 --block-size 8 --threads 32 --cpus 0-31
```

Defaults require existing rich PCA indexes. Override `--cosine-root` and
`--sift-root` for other machines. The script never builds or overwrites indexes;
all manifests, per-list logs and outputs go inside `--log-dir`. Use a fresh output
directory for each run. `--nq 0 --sift-nq 0` uses all queries for confirmation.

Configurations: SIFT1M per-list Pmax=40, nlist=1000, nprobe=34;
GloVe-200 global/per-list Pmax=64, nlist=1024, nprobe=512;
Landmark-DINO global/per-list Pmax=160, nlist=1024, nprobe=64 and 512.
All compare static full Pmax, original dynamic, frozen and updated with the
same rich index in each case. Static Pmax is not necessarily the best static P.
Each perf run has one warm-up and two measured loops; three independent repetitions
rotate method order. One separate stats run supplies worker timings and per-list
actual prefix depths. Stats fine timers include per-candidate instrumentation;
they do not describe parallel perf critical-path time.

Results: `summary.csv` contains median QPS/latency, timing variation, actual
pruning, average selected P, planner/compaction timings and newly materialized
query dimensions. The compaction counter covers state allocation/initialization,
packed-state compaction and the updated mode's posterior update; it is not a pure
memory-copy timer. The planner counter covers next-stage cost decisions.
`new_projection_dims_per_query` counts progressive stages only;
`actual_projection_dims_per_query` also includes calibration, reconstructed from
list depths (sum for local, maximum per query for global). Every measured
configuration must retain the static reference recall, identical per-query IDs
and distances, and satisfy per-list `scanned = MP pruned + exact`. Binary hashes are frozen
in `protocol.json`; rebuilding during the run is rejected. Small query-prefix runs
are preliminary evidence, not final full-benchmark results.

```bash
python3 scripts/verify_progressive_search.py --output /tmp/progressive_verify_new
python3 scripts/analyze_progressive_comparison.py logs/progressive_compare_new
python3 scripts/plot_progressive_comparison.py logs/progressive_compare_new
```

The small-fixture verification command requires the existing 4096-vector cosine
fixtures and nlist=32 banks under `/tmp/cosine-smoke-data`; override `--data-root`
to an equivalent fixture directory. It checks k=1/10, nprobe=1/4/32,
global/per-list and block=2/8 (64 processes, 192 search configurations).
