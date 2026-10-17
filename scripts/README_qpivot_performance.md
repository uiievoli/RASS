# Global QPivot performance rerun

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
  --cpus 0-55 --offline-threads 56 --build-jobs 16 --repeats 3
```

Use a new result directory. The runner builds the required binaries, reuses
an exact-size rich Global bank when available, otherwise upgrades a compatible
Triangle IVF, otherwise builds from base/query fvecs. Missing default
groundtruth is generated automatically. Existing wider banks are preserved.
They are not silently used with a smaller active cap: query projection and
MM scoring must both use exactly the configured M dimensions.

Extra cache directories can be supplied with repeated `--index-root DIR`.
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
banks if requested. The fixed-pool order-comparison experiment remains separate
and may deliberately use a larger dictionary to study selection quality.
