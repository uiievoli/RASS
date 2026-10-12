# DBpedia OpenAI-3072 supplement

The current dataset is **DBpedia OpenAI-3072**, published by Qdrant:
https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M

Use `text-embedding-3-large-3072-embedding`; the accompanying 1536-dimensional
ada-002 column is not used. The source contains 1,000,000 rows in 63 Parquet
shards (24,796,927,580 downloaded bytes) and only a train split.
The revision and every shard's SHA256/size are pinned in
`dbpedia3072_download_manifest.json` (revision
`b2f301648eda4c5dbe659235589c86ffec702581`).

## Download and prepare

```bash
python3 -m pip install -r scripts/requirements-data.txt
bash scripts/download_openai3072.sh \
  --data-root /path/to/data --raw-dir /path/to/download-cache
```

`download_openai3072.sh` now forwards to `download_dbpedia3072.sh`.
Downloads resume `.partial` files, verify size and SHA256, then stream only the
3072-dimensional column into float32 fvecs. A completed partial is verified
before sending a Range request, avoiding an EOF resume failure.
The exporter uses bounded batches and does not materialize the full dataset.
Reserve at least 40 GB for archives and exported vectors; indexes/PPD sidecars
require additional space. Dataset preparation is protected by a single-writer
OS lock. `--dry-run` prints commands without downloading.

With seed 42, 1,000 uniformly sampled rows become queries, removed from the
base. Output is `dbpedia3072m_holdout`, containing 999,000 base rows and 1,000
queries. The metadata retains selected source row IDs, revision and checksums.
Queries retain source order; base IDs refer to the remaining rows in order.
There is no official query/groundtruth split in this source.
`query.cpp` normalizes base/query vectors for cosine-equivalent squared L2 and
computes exact groundtruth against the disjoint base. No mmap/quantized index
path is introduced. Precomputed embeddings need no OpenAI API key.

## Fine pivot exploration and comparison

Build the perf/stats query binaries and perf `ppd_query` target using the usual
`build-progressive-perf` and `build-progressive-stats` build directories.

```bash
bash scripts/run_openai3072_retrieval.sh \
  --data-root /path/to/data \
  --log-dir /path/to/results/dbpedia3072 \
  --stage all --scope both --nlist 1000 \
  --pmax 512 --step 4 --refine-radius 16 \
  --threads 32 --cpus 0-31 --repeats 3 --loops 3
```

`openai3072_profile.json` selects the DBpedia holdout. The workflow calibrates
nprobe using the first tested recall >= 0.99, builds reusable global/per-list
PCA banks, scans pivots, refines observed peaks with step 1, and compares the
best tested static configuration with dynamic/progressive/PPD methods across
nprobe values. P includes the centroid; Pmax=512 supplies 511 PCA axes.
Best P is measured on the supplied queries, not claimed to be an independently
validated universal optimum. Generated figures and raw results stay in the
specified log directory.

The current ten-dataset local queue is `run_local_active10.sh`: nine existing
non-SIFT1B profiles plus this DBpedia supplement. It includes an independent
PCA-10% reference (P=309 for D=3072), rather than aliasing it to Pmax.
Roots, logs, threads and CPU binding can be overridden through environment
variables documented at the top of that script.

## Legacy SimpleWiki source

SimpleWiki is excluded from the current experiment suite. Its separate
`download_simplewiki3072.sh` and `simplewiki3072_profile.json` are retained for
explicit legacy reproduction. Existing downloaded files/results are preserved;
they must not be relabeled or combined with DBpedia results.
