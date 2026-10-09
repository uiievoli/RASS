# Eight-dataset recall sweep (search_only_v2)

For the restored full suite, see `README_all10_recall.md` and use
`build_and_run_all_10_datasets.sh`. The eight-dataset wrapper keeps its previous
default subset; `run_all_10_datasets.sh` now defaults to all ten.

Build both query binaries from this checkout before running:

```bash
cmake --build build-perf --target query -j 16
cmake --build build-stats --target query -j 16
build-perf/bin/query --benchmark_info
build-stats/bin/query --benchmark_info
```

The outputs must be `benchmark_protocol=search_only_v2 stats_enabled=0`
and `benchmark_protocol=search_only_v2 stats_enabled=1`, respectively.
Configure the build directories first if they do not exist.

```bash
scripts/run_all_8_datasets.sh \
  --data-root /path/to/data \
  --log-dir /path/to/log/all8_search_only_v2 \
  --cpus 0-55 --threads 56 --phase all \
  --perf-repeats 3 --warmup-loops 1
```

`--datasets` accepts a comma/space-separated subset. HandOutlines and SpaceV
are excluded. The download/build/run entry point accepts the same repetition,
warmup, seed and wait-policy options.

## Dynamic configuration

P includes the center, so the maximum projected dimension is Pmax - 1.
These limits are independent of the historical PCA10% comparison.

| Dataset | nlist | Best static scope/P | Dynamic scope/Pmax |
|---|---:|---|---|
| nuswide | 512 | per_list/2 | per_list/32 |
| fasion_mnist_784 | 256 | per_list/48 | per_list/96 |
| msong_holdout | 1000 | per_list/34 | per_list/96 |
| sift1m | 1000 | per_list/23 | per_list/40 |
| glove25 | 1024 | global/14 | global/26 |
| StarLightCurves | 128 | global/18 | global/104 |
| dbpedia1536m_holdout | 1000 | per_list/128 | per_list/256 |
| sift1b | 32768 | global/16 | global/16 |

The default seed is 0, allowing reuse of the existing SIFT1B PCA16 index.
The scope and limits match the fine-sweep supplement; the original supplement
used seed 20261005, so its absolute measurements are not identical controls.

## Index sharing

For the first seven datasets, a Triangle IVF is prepared once. An existing bare
IVF supplies its clustering when available; otherwise it is trained once.
Rich PCA indexes for the required scopes are then upgraded from that same IVF.
All searches share centroids, list memberships and candidate order. Static
prefix residuals are prepared before timing. These experiment indexes use
`recall_shared_v2_*` filenames and do not replace the old independent PCA caches.

SIFT1B retains one existing global PCA16 index for all methods. No second
billion-vector payload is constructed. Its HNSW settings remain M=32,
efConstruction=200 and efSearch=128 by default, with exact query coarse search
unchanged. The serialization format remains v10.
SIFT1B PCA10% and best-static are both global/P16, so the best-static curve
reuses the validated PCA10% measurements instead of repeating an identical run.

## Timing and outputs

Every nprobe/method performs one untimed warmup by default, even with loop=1.
Only search calls contribute to `query_time`. Recall/r2 evaluation and warmup
durations are exported separately. Reported latency is reciprocal throughput
in ms/query; it is not a single-query response-time percentile.

Perf runs in three separate processes, rotating method order. Loops per process
are 50/10/5/2/10/50/1/1 in the dataset order shown above. The canonical perf CSV
uses median batch duration and exports QPS range, CV and repeat count.
Stats uses one measured pass at every perf nprobe, with all original counters
and stage durations retained. Stage totals are sums of worker wall time and
can include descheduling; they are not process CPU time.

- `perf_repeats/<dataset>/<method>_repN.csv`: individual perf measurements.
- `perf/<dataset>/<method>.csv`: median performance, used by plotting.
- `stats/<dataset>/<method>.csv`: pruning counters at all sweep nprobes.
- `measurement_quality.csv`: repeated measurements with CV > 10% marked noisy.
- `environment.log`: binary/source hashes, CPU topology, affinity and settings.
- `<run>.log.command`, `.resources`, `.time`: command, load/memory/cgroup snapshots,
  and GNU time CPU/RSS/page-fault/context-switch counters, when available.

Resumption requires both a matching command/binary/index fingerprint and CSV
protocol/configuration validation. Old timing-protocol CSVs cannot be reused.
Different recalls across repetitions or across methods on the shared IVF cause
an explicit validation failure. Choose a new log directory to preserve old runs.

```bash
MPLCONFIGDIR=/tmp/tribase-matplotlib python3 scripts/plot_all8_recall_gt09.py \
  --log-root /path/to/log/all8_search_only_v2
```
