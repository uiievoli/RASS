# Section 4 predictor validation

Source inspected: `/home/iclab/wxy/test/vectorfilter.pdf`, printed page 7,
Section 4.2, equations (21)--(26). Equation (24) names a pruning-ratio estimator
but does not define its function, training procedure or cost calibration.
The stage-specific ridge regression in this experiment is an **engineering
instance**, not a predictor claimed to be specified by the paper.

## What is implemented

`query --progressive_oracle ... --oracle_predictor_features` writes a second
CSV, `<oracle_path>.predictor.csv`. Each row represents a query/list/transition
`r -> next_r`. Pivot count includes the center; projected rank is `P - 1`.

* `survivors` is the current population after IVF Triangle and `LB_r` filtering.
* `eta_x_mean` is the mean candidate residual-energy ratio in that population.
* `eta_q` is the query residual-energy ratio, an additional implementation input.
* `margin_tau_mean` averages `(tau2 - LB_r^2) / tau2` over those survivors.
* Seven bins describe relative margin `(tau2 - LB_r^2) / (4*hq_r*hx_r)`.
  Bin edges: .05, .1, .25, .5, .75, 1. A relative margin at least 1 cannot be
  pruned by a deeper prefix under the same threshold, since `UB^2-LB^2=4*hq*hx`.
* `true_next_pruned` supplies the next-stage pruning label. It is never an input
  feature. Coordinates are accumulated incrementally across checkpoints.
* `oracle_final_tau=1` explicitly identifies the fixed final kth-distance
  threshold among the probed lists. This is **not** the online heap threshold.

The old global oracle erroneously required each list's metadata to contain U.
It now validates global U in `global_pivots`, preserving the shared-geometry layout.

`test_paper_predictor.py` fits a separate small regression for each transition
and scope. Candidate-count weighting, ridge regularization and an upper-bound
eligibility cap are explicit implementation choices. Every fifth query is held
out together with all its lists/stages. A stage-average pruning predictor fitted
only on training queries is the comparison baseline. No cross-scope model or
scope-switching policy is inferred.

`refinement_value()` implements the marginal decision with **externally
calibrated time coefficients**:

```
gain_ns = survivors * predicted_rate * exact_ns
        - new_query_dims * projection_ns_per_dim
        - survivors * (new_bound_dims * bound_ns_per_dim + stage_ns_per_candidate)
        - predictor_ns
```

Global query coordinates already acquired in earlier lists have zero acquisition
cost. Local coordinates must reset for each list. A positive gain is permission
to compute the next bound; it never authorizes pruning by prediction alone.

## Run

Existing normalized L2 PCA banks from the cosine supplement are reused. Defaults:
GloVe-200 Pmax=64, nprobe=512; Landmark-DINO Pmax=160, nprobe=64; nlist=1024;
global and per-list; 64 queries; 32 workers. No index is rebuilt.

```bash
cmake --build build-perf --target query -j 8
python3 scripts/test_paper_predictor.py \
  --output logs/paper_predictor_validation_20261010 --nq 64 --threads 32
python3 -m unittest discover -s scripts/tests -p test_paper_predictor.py -v
```

Use `--analyze-only` to rerun analysis without rescanning candidates. Per-case
outputs include command/log, feature CSV, held-out predictions, report and model.

## Completed validation

64 queries per case, 51 training and 13 held out. Survivor-weighted mean absolute
error of predicted **additional pruning fraction**, in percentage points:

| Dataset | Scope | Current-state model | Stage-average baseline |
| --- | --- | ---: | ---: |
| GloVe-200 | global | 0.72 | 4.32 |
| GloVe-200 | per-list | 0.85 | 5.29 |
| Landmark-DINO | global | 3.74 | 11.97 |
| Landmark-DINO | per-list | 4.16 | 13.07 |

This is preliminary validation on a small query sample at one nprobe per dataset.
Errors pool the configured transitions, so the same candidate may contribute at
several stages. These are not QPS results, nor a comparison against the existing
list-level dynamic policy. Full threshold evolution, predictor/compaction time,
scope choice and online decisions remain untested.

**The learned estimator and refinement controller are not wired into production
scan.** Existing `--projection_dynamic` still executes the original list-level
policy. Production replacement requires an agreed estimator and calibrated cost
coefficients, then an independent stage-major search correctness/performance test.

## Why the predictor cannot be derived uniquely from eta and margin

At rank zero, take q=(1,0), c=(0,0), tau^2=1. Candidates x=(1,0) and x=(-1,0)
both have residual ratio 1 and margin 1. Refining direction (1,0) gives squared
bounds 0 and 4 respectively. Identical current residual ratios and margins can
therefore yield different future pruning. Their conditional pruning probability
needs distributional assumptions or empirical calibration; the current PDF
provides neither a closed-form rule nor its parameters.
