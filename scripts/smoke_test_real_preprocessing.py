#!/usr/bin/env python3
"""Run one frozen real-data preprocessing unit for each locked main dataset."""

from __future__ import annotations

import argparse
import gc
import json
from collections import Counter
from pathlib import Path

import numpy as np

from eeg_forgetting.data.channels import ChannelRegistry
from eeg_forgetting.data.frozen import BCISessionUnit, FrozenManifestSet, ManifestEEGLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-set",
        type=Path,
        default=PROJECT_ROOT / "manifests" / "v3" / "manifest-set.json",
    )
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    return parser.parse_args()


def summarize(loader: ManifestEEGLoader, unit) -> dict[str, object]:
    samples = loader.load_unit(unit)
    values = np.concatenate([sample.signal.reshape(-1) for sample in samples])
    summary = {
        "unit_id": unit.unit_id,
        "split": unit.split,
        "samples": len(samples),
        "shape": list(samples[0].signal.shape),
        "dtype": str(samples[0].signal.dtype),
        "label_counts": dict(sorted(Counter(sample.label for sample in samples).items())),
        "scaled_signal": {
            "finite": bool(np.isfinite(values).all()),
            "min": float(values.min()),
            "max": float(values.max()),
            "mean_abs": float(np.abs(values).mean()),
        },
    }
    del samples, values
    gc.collect()
    return summary


def main() -> None:
    args = parse_args()
    manifests = FrozenManifestSet(args.manifest_set, project_root=PROJECT_ROOT)
    registry = ChannelRegistry.from_yaml(PROJECT_ROOT / "configs" / "channels.yaml")
    loader = ManifestEEGLoader(manifests, registry)

    bci_units = loader.units("bciciv2a", args.split)
    # Exercise the competition evaluation-label boundary, not only the easier T session.
    bci = next(unit for unit in bci_units if isinstance(unit, BCISessionUnit) and unit.session == "E")
    selected = {
        "bciciv2a": bci,
        "physionet_mi": loader.units("physionet_mi", args.split)[0],
        "sleep_edf_sc": loader.units("sleep_edf_sc", args.split)[0],
    }
    result = {
        "manifest_version": manifests.version,
        "split": args.split,
        "datasets": {
            dataset: summarize(loader, unit) for dataset, unit in selected.items()
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
