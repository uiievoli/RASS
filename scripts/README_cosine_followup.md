# Focused cosine/angular follow-up

```bash
python3 scripts/run_cosine_followup.py \
  --data-root datasets/cosine_supplement \
  --log-dir logs/cosine_followup_20261009 \
  --cpus 0-31 --threads 32 --perf-repeats 3 --warmup-loops 1
```

The prepared official data and existing global normalized rich indexes are
required. No re-download or reclustering occurs. Per-list PCA metadata is
created from the global index's IVF when missing. Existing per-list banks and
matching repeated CSVs are reused. This driver is Python, so its executable code
is compiled at startup; its source snapshot and binary hashes are recorded.
Changing a binary during the run aborts subsequent measurements.

Cases:

- GloVe-200-angular: nprobe=512, reference P21, max P64.
- Landmark-DINO-768-cosine: nprobe=64 and512, reference P78, max P160.

Every case compares seven fresh measurements: IVF baseline; global fixed
reference/max/dynamic; per-list fixed reference/max/dynamic. Fixed max versus
dynamic has the same available basis/Pmax. The baseline is repeated as a
same-period anchor because earlier baseline QPS CV was large. All methods use
the same IVF centroids, memberships and candidate order; k=1, nlist=1024,
unit-normalized L2, official neighbor-ID recall, and no IRLS.

Perf: three independent processes, method order rotated, three measured passes
and one untimed warmup by default. Stats: one measured pass plus warmup; original
pruning counters and stage timings are retained. Generic P=0 exact work remains
outside the fine candidate_exact_seconds timer: use total decision time and
exact-distance counts for cross-method conclusions.

Output: `perf_repeats/`, `perf/`, `stats/`, `build/`, per-run GNU time resource
records, `protocol.json`, `driver_snapshot.py`, `followup_summary.csv`,
`pivot_histograms.csv`, and three fixed-recall comparison figures in `figures/`.
Per-list PCA does not yet have a validated static optimum; the reference and
max settings are controlled comparisons, not a claim of optimality.
