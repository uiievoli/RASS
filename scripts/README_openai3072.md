# OpenAI 3072-dimensional supplement

Selected dataset: **simplewiki-openai-3072-normalized**, published by VIBE.
It contains 260,372 corpus vectors, 3072 dimensions, official test queries and
100 neighbor IDs per query. The embedding-generation code uses OpenAI
`text-embedding-3-large`. Query count is read from the HDF5 metadata rather than
inventing an independent split. The source is an in-distribution text benchmark.
VIBE marks it archived/deprecated: the data remains available, but upstream
benchmark results will no longer be updated. This is not a model deprecation.

## Download sources and alternatives

Primary data/card and file details:

* https://huggingface.co/datasets/vector-index-bench/vibe
* https://huggingface.co/datasets/vector-index-bench/vibe/blob/main/simplewiki-openai-3072-normalized.hdf5
* https://github.com/vector-index-bench/vibe/blob/main/vibe/datasets.py

The artifact is about 3.21 GB. The script pins the snapshot:

```
revision: b1e7761d0873d92cc1c7721f565ba76cb44b9c63
SHA256:   5b012e832cc6ffd72c248ce5e8ac12d34441c7692a4546e2a5672ebe2d4d4aa3
```

Qdrant also publishes a 3072-dimensional DBpedia 1M corpus:
https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M
Its card declares only a train split and a roughly 24.8 GB Parquet download,
also including another 1536-dimensional embedding column. It would require
separate query/groundtruth preparation. The smaller VIBE snapshot was chosen
because it already supplies a canonical ANN query/neighbor protocol.

Precomputed vectors require no OpenAI API key, text embedding generation, or
embedding API calls. The VIBE dataset card declares CC-BY-4.0; retain the source
attribution when distributing derived artifacts. Upstream library code and
dataset licensing are separate.

## Independent download script

```bash
python3 -m pip install numpy h5py pandas matplotlib
bash scripts/download_openai3072.sh \
  --data-root /path/to/data --raw-dir /path/to/download-cache
```

Default transport is direct HTTPS through curl, with partial-file resume,
retries, SHA-256 verification and an OS-backed single-writer lock. It avoids
Hugging Face local-download locks. Completed partial files are recognized before
an EOF Range request. Incomplete/corrupt files are preserved on errors.
`--dry-run` prints commands. A mirror can be supplied with `--url` while retaining
the pinned checksum. For a manually downloaded file, use `--hdf5 FILE`.

Alternative manual Hugging Face CLI command:

```bash
hf download vector-index-bench/vibe \
  simplewiki-openai-3072-normalized.hdf5 --repo-type dataset \
  --revision b1e7761d0873d92cc1c7721f565ba76cb44b9c63 \
  --local-dir /path/to/download-cache
bash scripts/download_openai3072.sh --data-root /path/to/data \
  --hdf5 /path/to/download-cache/simplewiki-openai-3072-normalized.hdf5
```

Export layout:

```
DATA_ROOT/simplewiki-openai-3072-normalized/
  origin/simplewiki-openai-3072-normalized_base.fvecs
  origin/simplewiki-openai-3072-normalized_query.fvecs
  origin/cosine_source.json
  result/official_neighbors.i32bin
  download_manifest.json
```

Rows and neighbor IDs retain official order. The exporter verifies finite,
nonzero, unit-norm vectors for the source's `normalized` metric; it does not
normalize or truncate exported vectors. The production query path applies
`--normalize` before training/search as in the existing cosine supplements.
Unit-L2 ranking is cosine-equivalent. No mmap/quantized index path is introduced.

## Retrieval and fine static-pivot exploration

Use the independent retrieval entry point, backed by the existing native query
binaries:

```bash
cmake -S . -B build-progressive-perf -DCMAKE_BUILD_TYPE=Release -DENABLE_STATS=OFF
cmake --build build-progressive-perf --target query ppd_query -j 8
cmake -S . -B build-progressive-stats -DCMAKE_BUILD_TYPE=Release -DENABLE_STATS=ON
cmake --build build-progressive-stats --target query -j 8
```

Default nlist=512 (approximately sqrt(N)), k=1, Pmax=512, PCA only,
global/per-list, float32 signatures, shared IVF partitions. P includes centroid;
Pmax=512 means 511 projection coordinates. Maximum P is bounded by the current
implementation, not by the dataset dimension. In sparse lists, PCA pads missing
positive-eigenvalue directions with an orthogonal completion; higher P therefore
does not guarantee additional informative principal directions.

First calibrate nprobe using IVF against official IDs:

```bash
bash scripts/run_openai3072_retrieval.sh \
  --data-root /path/to/data --log-dir /path/to/log/openai3072 \
  --stage calibrate --threads 56 --cpus 0-55
```

The grid includes 1,2,4,8,16,24,32,48,64,96,128,192,256,384,512. It selects the
first available point reaching recall>=.99. No nprobe is asserted in advance.
Use `--target-recall .9` for a .9 fixed-point study, or `--nprobe N` to explicitly
override selection (reported as manual). Indexes are reused or built once under
`LOG_DIR/index_cache`. Corpus reading/PCA construction is outside query timing.

Fine screen, then unit-step refinement around each scope's QPS winner:

```bash
bash scripts/run_openai3072_retrieval.sh \
  --data-root /path/to/data --log-dir /path/to/log/openai3072 \
  --stage fine --threads 56 --cpus 0-55
bash scripts/run_openai3072_retrieval.sh \
  --data-root /path/to/data --log-dir /path/to/log/openai3072 \
  --stage refine --threads 56 --cpus 0-55
```

Default grid: P=2,4,8,12,...,512 (129 values). Every run evaluates prefixes of
one Pmax bank, preserving a nested basis; it does not reconstruct an index per P.
Refinement covers each scope's peak +/-16 with step=1. `--step 1` instead scans
all P=2..512 directly. `--pmax`, `--nlist`, `--scope`, `--nq`, repeats, loops and
binary paths are adjustable. Identical options must be used for successive stages;
`--stage` alone is a scheduling option. Changing protocol requires a fresh log root.
Three independent perf repetitions rotate pivot/scope order; stats use a separate
build/pass. High-D per-list PCA and storing U can be costly: plan roughly 20 GB of
disk as a lower-order estimate; reserve at least 30 GB for raw/exported files,
both banks and comparison sidecars, with RAM headroom for
parallel PCA workspace. Actual resource usage depends on implementation/config.

Outputs:

* `calibration.json`: chosen nprobe and target.
* `fine_pivot_summary.csv`, `fine_pivot_all_runs.csv`: costs, pruning and variation.
* `fine_pivot_winners.json`: best **tested** P per scope.
* `figures/qps_pivot.*`, `latency_ms_pivot.*`, `overall_prune_rate_pivot.*`.

The refined reruns replace coarse repetitions at overlapping points when
selecting winners; all raw measurements remain available. Construction/save are
excluded. Worker stage times are diagnostics, not parallel latency components.

## Baseline/PPD/dynamic recall curves

After refinement, the selected static P values replace the initial untuned
reference in `comparison_profile.json`:

```bash
bash scripts/run_openai3072_retrieval.sh \
  --data-root /path/to/data --log-dir /path/to/log/openai3072 \
  --stage compare --threads 56 --cpus 0-55
```

This runs IVF, Triangle, selected static P, static Pmax, original dynamic,
frozen/updated with retained exploration, and PPD on the recall grid. PPD uses
full-database PCA + Triangle. Curve plots retain recall>.9. `--stage all` performs
all four stages and may take substantial time. It is not merely a fine scan.

For an independent tuning/evaluation protocol, prepare query views with
`prepare_progressive_query_split.py --config scripts/openai3072_profile.json`.
Run fine/refinement on the tuning view. Then call the generic ten-dataset runner
with the generated `comparison_profile.json`, the evaluation data root and
`--index-root /path/to/log/openai3072/index_cache`. Changing the standalone
runner's data root inside an existing log root is rejected.
