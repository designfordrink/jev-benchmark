# HARTH dataset for J02

J02 uses the public HARTH activity-recognition dataset as an external dataset artifact. The benchmark repository does **not** commit the ~296 MB archive.

## Source

- Original: UCI Machine Learning Repository, HARTH dataset 779.
- DOI: `10.24432/C5NC90`.
- Benchmark mirror: Hugging Face `High-Light/jev-harth`.
- Archive: `harth.zip`.

Direct archive URL:

`https://huggingface.co/datasets/High-Light/jev-harth/resolve/main/harth.zip`

The mirror is a convenience for reproducible downloading; it does not replace UCI attribution.

## Download from Hugging Face subject CSVs

The benchmark also supports the exact workflow where the subject CSV files have already been uploaded to `High-Light/jev-harth`. It queries the Hub file tree, selects `Sxxx.csv` files, downloads the requested subjects, and writes a local provenance manifest. No ZIP archive is required.

```bash
jev-bench harth-hf-download \
  --repo-id High-Light/jev-harth \
  --output-dir data/harth
```

For a small data inspection:

```bash
jev-bench harth-hf-download \
  --repo-id High-Light/jev-harth \
  --subjects S001,S002,S003 \
  --output-dir data/harth-smoke
```

Then use the existing J02 commands unchanged against the materialized directory. The generated `harth-hf-manifest.json` records the Hub repository/revision, source file paths and object IDs, selected subjects, and local SHA-256 hashes.

## Download archive

After installing the benchmark:

```bash
jev-bench harth-download --output-dir data/harth
```

This downloads the archive, validates the ZIP, computes SHA-256, extracts it, and writes `harth-manifest.json`.

For a fresh download:

```bash
jev-bench harth-download --output-dir data/harth --force
```

The implementation uses the direct Hugging Face `resolve/main` endpoint and does not require the `datasets` package.

## Verify

The extracted archive should contain subject CSV files such as:

```text
data/harth/extracted/
  S001.csv
  S002.csv
  ...
```

Run:

```bash
jev-bench harth-manifest --dataset-root data/harth/extracted
jev-bench harth-inspect --dataset-root data/harth/extracted --subject S015
```

Then use a small smoke run before the complete matrix:

```bash
jev-bench harth-loso \
  --dataset-root data/harth/extracted \
  --model tiny_mlp \
  --hidden-units 8 \
  --subjects S001,S002,S003 \
  --max-train-windows-per-subject 100 \
  --max-test-windows 300 \
  --epochs 3 \
  --output results/j02-smoke.json
```

## Provenance

Every J02 result should retain the UCI identifier/DOI, Hugging Face repository and revision, archive SHA-256, subject list, window/stride, seed list, training hyperparameters, and benchmark software revision.

The archive itself stays outside Git history.
