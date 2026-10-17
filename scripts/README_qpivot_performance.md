# Global QPivot performance rerun

The recall runner compares five methods using one freshly built Global index:
IVF (no pruning), Tribase (Triangle only), Global PCA (Triangle plus all M
directions in original PCA order, no dynamic selection), Dynamic (original
quantity selector, PCA prefix order, block size 8, no MM), and QPivot (Triangle
plus MM order and dynamic prefix selection). There is no per-method index build.
The Dynamic control is the original implementation executed on this new
Global index; it does not recreate historical per-list indexes or code versions.
Query execution defaults to 56 parallel workers (`--query-threads 56`, alias
`--threads 56`), with independent queries distributed across workers.
Use `--cpus 0-55` for binding. Offline construction uses `--offline-threads`.

QPS is aggregate throughput. The latency plot reports amortized wall time per
query (batch wall time divided by total queries); it is not the response
latency of an individual concurrent request. Stage statistics remain summed
worker time per query. Allocation diagnostics are NaN for parallel queries
because the diagnostic allocator only observes the calling thread.

The shared configuration is `scripts/qpivot_performance_profiles.json`.
All profiles use Global PCA; Pmax includes the center and the actual projected
dictionary has M=Pmax-1 rows. These are historical reference configurations for
a new performance test, not claims of optimality for the MM policy.

| Dataset | nlist | Pmax | M | Reference |
|---|---:|---:|---:|---|
| SIFT1M | 1000 | 40 | 39 | Original dynamic maximum; Global static fine optimum near P36 |
| Fashion-MNIST | 256 | 64 | 63 | Historical Global static reference |
| MillionSong | 1000 | 64 | 63 | Historical Global static reference |
| GloVe25 | 1024 | 14 | 13 | Fine Global static scan peak |
| StarLightCurves | 128 | 18 | 17 | Fine Global static scan peak |
| DBpedia OpenAI-1536 | 1000 | 128 | 127 | Historical Global static reference |
| DBpedia OpenAI-3072 | 1000 | 256 | 255 | Untuned reference; validate in this rerun |

The first seven are the default active set. Optional profiles also cover
NUS-WIDE (P8), HandOutlines (P8), GloVe-200-angular (P21), and Landmark-DINO
(P78). Select `--datasets all-small` for all eleven non-1B datasets. Neither
SIFT1B nor SpaceV1B is included. Optional cosine profiles and OpenAI-3072
normalize queries and use unit-normalized indexes and groundtruth.

```bash
bash scripts/run_qpivot_recall_sweep.sh \
  --data-root /path/to/data \
  --log-dir /path/to/logs/qpivot_global_bounded \
  --datasets all \
  --cpus 0-55 --query-threads 56 --offline-threads 56 --build-jobs 16 --repeats 3
```

Use a new result directory. The runner rebuilds each Global PCA index from
base fvecs, including IVF training and candidate signatures. It never loads
historical indexes, upgrades old Triangle banks or uses cached centroids.
New indexes live in `LOG_DIR/indexes/DATASET/`; old files are preserved.
Missing default groundtruth is generated automatically.
Query projection and MM scoring both use exactly the configured M dimensions.

`--index` and `--index-root` are rejected to prevent accidental old-bank reuse.
Both perf and stats binaries fully validate the new index before measurements.

To keep exactly the profiles/Pmax from the previous parallel run, use its
protocol. This copies configurations only; all indexes are still rebuilt:

```bash
bash scripts/run_qpivot_recall_sweep.sh \
  --data-root /path/to/data \
  --reference-protocol /path/to/qpivot_global_para/protocol.json \
  --log-dir /path/to/logs/qpivot_dynamic_same_pmax \
  --datasets all-small --cpus 0-55 --query-threads 56 \
  --offline-threads 56 --build-jobs 16 --repeats 5
```

Do not combine this with `--pmax` or `--nlist`: the reference freezes both.
Every method in every nprobe case must use that dataset's same persisted bank.
The summary records Dynamic and QPivot mean pivot counts and validates both
histograms against Pmax. Existing four-method runs can still be summarized
using their recorded method list, including `--summarize-only --allow-partial`.
All datasets must be under the supplied data root with the normal
`DATASET/origin/DATASET_{base,query}.fvecs` layout.

The data root, output root, dataset subset and all numeric overrides retain
their existing options. To inspect resolved settings without running:

```bash
python3 scripts/run_qpivot_recall_sweep.py \
  --data-root /path/to/data --log-dir /tmp/qpivot_config_preview \
  --datasets all --show-config
```

Customize the JSON directly or use `--config CUSTOM.json`. The runner records
the resolved profiles, M, config version and exact index paths in
`protocol.json`. nprobe grids, query counts, repetition/warmup protocol and
the online selection algorithm are retained; SIFT1M additionally includes
nprobe=100 for comparison with its historical dynamic experiments.
The query count must be at least the query thread count. The budget-cap sweep
now defaults to 64 sampled queries to support all 56 workers.

The budget-cap sweep also reads the shared profiles:

```bash
bash scripts/run_qpivot_pmax_sweep.sh \
  --data-root /path/to/data --log-dir /path/to/logs/sift1m_budget_caps \
  --dataset sift1m --cpus 0-55 --offline-threads 56 --build-jobs 16
```

It uses one fixed dictionary (SIFT1M: M=39) and varies only the maximum
candidate filtering budget. Every budget uses the same query projection and
scoring bank, so this sweep does not measure changes in dictionary size.
`--pmax ...` explicitly overrides its default grid, including larger diagnostic
banks if requested. The order-comparison runner now also rebuilds Global PCA
using the same bounded dataset profiles. It never replaces the stored basis
with an external dictionary: QPivot is solely a query-time optimization over
the Global index's persisted projection directions. Selecting L directions
always satisfies L <= M = Pmax-1.
