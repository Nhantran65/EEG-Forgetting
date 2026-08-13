#!/usr/bin/env python3
"""Run one locked three-task sequential fine-tuning smoke order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eeg_forgetting.data.contracts import load_yaml
from eeg_forgetting.training.continual import run_sequential_finetuning


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", required=True, choices=("forward", "reverse"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "pilots" / "sequential_ft_smoke_v1.yaml",
    )
    parser.add_argument(
        "--cache-root", type=Path, default=PROJECT_ROOT / "data" / "processed" / "v3"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "continual" / "sequential_ft_smoke_v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_yaml(PROJECT_ROOT / "configs" / "models" / "cbramod.yaml")
    checkpoint = model["checkpoint"]
    result = run_sequential_finetuning(
        config_path=args.config,
        order_name=args.order,
        cache_root=args.cache_root,
        checkpoint_path=PROJECT_ROOT / str(checkpoint["local_path"]),
        checkpoint_sha256=str(checkpoint["sha256"]),
        output_dir=args.output_root / args.order / "seed-3407",
        device=args.device,
    )
    print(
        json.dumps(
            {
                "run": result["run"],
                "order": result["order"],
                "pairwise_forgetting": result["pairwise_forgetting"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
