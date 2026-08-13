#!/usr/bin/env python3
"""Run the one-off upstream-protocol PhysioNet CBraMod reproduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eeg_forgetting.data.contracts import load_yaml
from eeg_forgetting.training.reproduction import run_physionet_reproduction


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            PROJECT_ROOT
            / "configs"
            / "training"
            / "physionet_cbramod_reproduction.yaml"
        ),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "processed"
            / "v3-physionet-cbramod-reproduction"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results"
            / "reproductions"
            / "physionet_cbramod_v1"
            / "seed-3407"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_yaml(PROJECT_ROOT / "configs" / "models" / "cbramod.yaml")
    checkpoint = model["checkpoint"]
    result = run_physionet_reproduction(
        cache_root=args.cache_root,
        checkpoint_path=PROJECT_ROOT / str(checkpoint["local_path"]),
        checkpoint_sha256=str(checkpoint["sha256"]),
        config_path=args.config,
        output_dir=args.output_dir,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "run": result["run"],
                "best_epoch": result["best_epoch"],
                "best_validation_kappa": result["best_validation_kappa"],
                "test": result["test_at_best_validation"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
