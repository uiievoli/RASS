# GloVe-200 angular and Landmark-DINO-768 cosine

```bash
scripts/download_and_run_cosine_supplement.sh \
  --data-root /path/to/data/cosine_supplement \
  --log-dir /path/to/log/cosine_supplement \
  --cpus 0-31 --threads 32 --scope global \
  --perf-repeats 3 --warmup-loops 1
```

Use `--scope per_list` or `--scope both` for local PCA or both configurations.
Use `--download-only` to prepare files without experiments, or run
`download_cosine_datasets.sh` and `run_cosine_supplement.sh` separately.
The run script accepts `--skip-build`, `--skip-plots`, `--nlist`, `--k`,
`--nprobes`, CPU binding and the corrected benchmark options.

Dependencies: curl, flock, Python numpy/h5py/matplotlib, and the query build
dependencies. No Google Drive/gdown or Hugging Face cache locks are needed.

Sources:

- GloVe: https://ann-benchmarks.com/glove-200-angular.hdf5 (962,819,488 bytes).
- Landmark: the VIBE release at Hugging Face revision
  `07b387891a221b7b073b83d2f752b76462e5fa03`, file
  `landmark-dino-768-cosine.hdf5` (2,341,733,696 bytes). Its SHA256 is checked.

Interrupted HTTP transfers resume from `.partial`. Complete files are renamed
only after expected-size and HDF5 schema checks. A per-raw-directory flock
prevents duplicate writers. Preparation records a SHA256 and provenance marker,
retains official train/test ordering, checks finite/nonzero vectors, and exports
neighbors as zero-based i32bin IDs. Existing verified exports are reused.

Prepared fvecs preserve source values (with float32 storage). Unit normalization
occurs in query.cpp before index training/add and when queries are loaded, once
before search. Cached indexes already contain unit vectors. The identity
`||q_hat-x_hat||^2 = 2(1-cos(q,x))` gives the same cosine/angular ranking, enabling
the existing L2 Triangle/PCA bounds. This is cosine/angular retrieval through
normalized L2, not support for arbitrary unnormalized maximum inner product.

Default experiment: nlist=1024, k=1, all official queries, nprobe
1/2/4/8/16/32/64/128/256/512. Each scope includes baseline, Triangle, reference
static PCA and dynamic PCA. GloVe reference P21/Pmax64; Landmark P78/Pmax160.
P includes the center. The two static labels share the reference measurement;
no fine scan has yet established the optimal static P. All settings are recorded
in `<log-dir>/<scope>/experiment_config.json`.

All variants share a native IVF with isolated `_unit` index names. They reuse
indexes and compatible repeated CSVs. Search-only timings, three independent
perf processes, warmup, phase counters and measurement-quality diagnostics are
retained. Official neighbor IDs evaluate recall independently of distance scale.
Per-scope results and three recall > 0.9 figures are in `<log-dir>/<scope>/`.
