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

The training cache is derived only through that manifest-backed boundary and
is resumable per recording/session:

```bash
uv run python scripts/build_processed_cache.py \
  --datasets bciciv2a physionet_mi sleep_edf_sc \
  --splits train validation --workers 8
```

The audited v3 cache contains 2,678/982 BCI trials, 6,300/1,620 PhysioNet
trials, and 118,662/39,580 Sleep epochs for train/validation respectively.

## CBraMod integration

The model adapter is pinned to official CBraMod code commit
`b9e961003214326972c567eff390e75b0287e32a`. Download and verify the official
checkpoint, then run the real-batch GPU proof:

```bash
uv run python scripts/download_cbramod_checkpoint.py
CUDA_VISIBLE_DEVICES=0 uv run python scripts/smoke_test_cbramod_gpu.py
```

The checkpoint lives under ignored `checkpoints/`; its revision, byte count and
SHA-256 are locked in `configs/models/cbramod.yaml`. The current project lock
uses the PyTorch CUDA 13.0 wheel. The smoke script requires an environment that
exposes NVIDIA devices; unit tests remain CPU-compatible.

Pilot outputs are create-only under ignored `results/pilots/`. The fair frozen
probe uses every channel-patch feature followed by one linear layer. The depth
sweep uses the locked all-patch MLP head and keeps frozen CBraMod blocks in eval
mode so their dropout cannot confound the number of plastic blocks:

```bash
uv run python scripts/run_cbramod_depth_pilot.py \
  --dataset bciciv2a --depth 0 \
  --config configs/pilots/cbramod_linear_probe.yaml \
  --output-root results/pilots/cbramod_linear_probe_v1
uv run python scripts/run_cbramod_depth_matrix.py --devices 0 1 2 3
uv run python scripts/summarize_cbramod_depth_pilot.py
```

Depth v3 selected the final four encoder blocks by the predeclared 95% rule;
the evidence and rejected pilot history are recorded in
`docs/decisions/0001-lock-cbramod-head-and-depth.md`.

Budget-matched single-task runs save two explicitly different artifacts:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_single_task_baseline.py \
  --dataset bciciv2a --device cuda:0
```

`best.pt` is a validation diagnostic; `final.pt` is the exact step-2,500
checkpoint used for Fisher and budget-matched comparisons. The one-off
PhysioNet external reproduction is separate again: it uses the upstream
64-channel preprocessing and 50-epoch full-backbone protocol through
`scripts/run_physionet_cbramod_reproduction.py`.
