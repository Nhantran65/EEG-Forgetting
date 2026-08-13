# EEG Forgetting

Reproducible data and experiment infrastructure for diagnosing catastrophic
forgetting in CBraMod across heterogeneous EEG tasks.

The locked main datasets are BCI Competition IV-2a, PhysioNet-MI, and
Sleep-EDF Expanded (Sleep Cassette). TUEV is the conditional fourth task once
access and preprocessing are validated. The previous VEP candidate is
quarantined from the main continual-learning matrix because its raw snapshot
contains no stimulus markers and fails phase-crossover validation.

## Data contract

Repository-owned protocol files live in `configs/datasets/`. All loaders emit
EEG in CBraMod units and shape:

```text
(channels, one-second patches, 200 samples)
```

The shared amplitude convention is microvolts divided by 100. Dataset payloads,
checkpoints, and results are intentionally ignored by Git. Paper-facing split
manifests are versioned under `manifests/` and are create-only.

## Development

```bash
uv sync
uv run pytest
```

Use `docs/plans/active/eeg-forgetting-icassp-2027.md` as the current execution
plan and `docs/product/dataset-protocol.md` as the authoritative data contract.

## Local raw data

The public data downloader materializes only the source files needed by the
locked protocol:

```bash
uv run python scripts/download_public_datasets.py --workers 16
uv run python scripts/audit_downloaded_data.py
uv run python scripts/audit_manifests.py --verify-sources
uv run python scripts/smoke_test_real_preprocessing.py
```

Raw files live under `data/raw/` and are ignored by Git. The current audited
snapshot contains 18 BCI IV-2a GDF files plus labels, 654 PhysioNet imagery EDF
files, and 153 paired Sleep Cassette recordings.

The active frozen split manifests are in `manifests/v3/`. They also freeze the
channel registry and shared preprocessing config used by the manifest-backed
loader. To materialize a later version from a new audited snapshot, pass an
explicit new output directory to `scripts/build_manifests.py`; the builder
refuses to overwrite an existing set.
