# Download the current experiment suite

Run from the repository root in the Python/conda environment used for data preparation:

```bash
python3 -m pip install -r scripts/requirements-data.txt
bash scripts/download_current_datasets.sh \
  --data-root /path/to/data \
  --raw-dir /path/to/download-cache
```

The default suite contains ten active datasets (SIFT1B is temporarily isolated): Fashion-MNIST (`fasion_mnist_784`),
MillionSong (`msong_holdout`), SIFT1M, GloVe25, HandOutlines, StarLightCurves,
DBpedia1536 holdout, GloVe-200-angular, Landmark-DINO-768-cosine and
DBpedia-OpenAI-3072 holdout. NUS-WIDE and SpaceV are excluded.
HandOutlines is included in the current ten-dataset comparison plus the new
OpenAI supplement. SIFT1B remains available only through explicit
`--datasets sift1b`; its data and indexes are preserved.

Use `--datasets "glove-200-angular,landmark-dino-768-cosine,dbpedia3072m_holdout"`
to download a subset. Add `--dry-run` to preview download/preparation commands.

The entry point checks vector headers, dimensions and file sizes before deciding
whether an export is ready. Cosine exports additionally require their source
marker, pinned checksum where available, and official groundtruth. Prepared
datasets are reused even if their downloaded archives have been removed.
Checks do not read all vector values or checksum the entire exported corpus.
New HDF5 downloads use the existing source validation/checksum adapters.

Rerunning the same command skips prepared datasets and resumes `.partial`
downloads. One failed source does not prevent the other datasets from being
prepared; the final exit code is nonzero and names all incomplete datasets.
One unified downloader per cache directory is enforced with an OS lock.

System commands: bash, curl (or optional aria2c for legacy HTTP sources), unzip,
gzip, tar and flock; gdown is needed only when the benchmark bundle is missing.
This command downloads/prepares inputs, without building indexes or starting
experiments. For native datasets it preserves bundled groundtruth when present;
missing generated groundtruth remains the retrieval workflow's responsibility.
The bundle transports several small datasets together even when only one is
requested. SIFT1B retains bvecs on disk, following the existing experiment path.
