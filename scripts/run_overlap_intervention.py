#!/usr/bin/env python3
"""Run one locked BCI-to-Sleep Fisher-overlap freeze intervention."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eeg_forgetting.data.contracts import load_yaml
from eeg_forgetting.training.intervention import run_overlap_freeze_intervention


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "configs"
        / "training"
        / "bci_sleep_overlap_intervention_v1.yaml",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "v3",
    )
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    output_root = args.output_root or (
        PROJECT_ROOT / "results" / "interventions" / str(config["id"])
    )
    model = load_yaml(PROJECT_ROOT / "configs" / "models" / "cbramod.yaml")
    checkpoint = model["checkpoint"]
    result = run_overlap_freeze_intervention(
        config_path=args.config,
        seed=args.seed,
        ratio=args.ratio,
        condition=args.condition,
        cache_root=args.cache_root,
        checkpoint_path=PROJECT_ROOT / str(checkpoint["local_path"]),
        checkpoint_sha256=str(checkpoint["sha256"]),
        output_dir=(
            output_root
            / f"ratio-{args.ratio:g}"
            / args.condition
            / f"seed-{args.seed}"
        ),
        device=args.device,
    )
    print(
        json.dumps(
            {
                "seed": result["seed"],
                "ratio": result["ratio"],
                "condition": result["condition"],
                "test_forgetting": result["forgetting"]["test"],
                "test_sleep": result["evaluations"]["test"]["sleep_edf_sc"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
