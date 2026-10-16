# Query-arrival adaptive pivot selection on SIFT1M

Run `python scripts/run_sift1m_adaptive_query_plan.py`; use `--out NEW_DIRECTORY` for reruns.
The default output is `logs/sift1m_adaptive_query_plan_20261016_interleaved`.
Regenerate reports with `--summarize-only`.

This extends Query-Energy / per-list-mean scoring with an offline-trained runtime model.
Each query computes its shared global-PCA coordinates, ranks directions, extracts 20
features from query coordinates, coarse distances and selected list sizes, and predicts
the cost of each budget. It selects among 0, 8, 16, 24, 32, 48, 64, 96 and 128 directions.
Partial budgets use the first P directions in that query's ranking. The 128 action uses
the canonical PCA early-exit kernel, and 0 skips multi-pivot pruning. All decisions are
query-level, shared across probed lists; there is no within-query feedback/replanning.
Directions are projection axes; the native pivot parameter includes a center (P+1).

128 sampled calibration queries exclude all 32 held-out query IDs. Calibration shuffles
query/action pairs independently each round, avoiding back-to-back hot-query calibration.
Each family fits per-action log-runtime ridge regressions. The regularization coefficient
is selected using 4-fold query-level cross-validation on calibration queries only, minimizing
actual calibration runtime of the predicted action. There is no test-time query-ID lookup,
candidate inspection for selection, or use of held-out labels for fitting.

Performance and stats builds must be built separately (`build-predictor-perf`,
`build-predictor-stats`). The benchmark installs the same frozen M128 dictionary as the
preceding SIFT1M experiment and loads its offline moment sidecar. Models bind to the
dictionary/index checksum and nprobe. Their binary format is QPLAN001, followed by
nine uint64 header fields, two 20-double standardization arrays, (P,canonical) uint64
action pairs and per-action 21-double log-runtime coefficients.

All baselines are rerun, including original global dynamic, original per-list PCA40
dynamic, fixed ranked budgets 16/24/32, canonical full128 and Triangle-only. The original
per-list index is required to have identical centroids, partition sizes and candidate order.
All online projection, statistics aggregation, ranking, features, prediction, scanning and
heap updates are included in timing. Offline calibration, metadata creation/loading are not.
The 0-action training target is Triangle-only and omits selector overhead; real adaptive
testing includes that overhead. Kernel-specific costs are learned, not assumed proportional
to P. In particular, canonical full128 does not necessarily process all 128 axes per candidate.

CSV outputs include timings, pruning counts and exact-result checks; stats choices CSV
records each held-out query's selected count and predicted cost. `summary.csv`,
`budget_distribution.json`, `models/diagnostics.json` and `report.md` expose final results,
calibration CV results and uncertainty. Nine-round paired timing bootstrap intervals are
conditional on these fixed 32 test queries. Matching Triangle top-1 is not full-library recall.
