#!/usr/bin/env python3
"""Run one immutable EWC or DER++ three-task order/seed matrix cell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eeg_forgetting.data.contracts import load_yaml
from eeg_forgetting.training.continual_methods import (
    run_derpp_continual_matrix_run,
    run_ewc_continual_matrix_run,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("ewc", "derpp"), required=True)
    parser.add_argument("--order", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "v3",
    )
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config or (
        PROJECT_ROOT / "configs" / "training" / f"{args.method}_v1.yaml"
    )
    protocol = load_yaml(config_path)
    output_root = args.output_root or (
        PROJECT_ROOT / "results" / "continual" / str(protocol["id"])
    )
    model = load_yaml(PROJECT_ROOT / "configs" / "models" / "cbramod.yaml")
    checkpoint = model["checkpoint"]
    common = {
        "config_path": config_path,
        "order_name": args.order,
        "seed": args.seed,
        "cache_root": args.cache_root,
        "checkpoint_path": PROJECT_ROOT / str(checkpoint["local_path"]),
        "checkpoint_sha256": str(checkpoint["sha256"]),
        "output_dir": output_root / args.order / f"seed-{args.seed}",
        "device": args.device,
    }
    if args.method == "ewc":
        result = run_ewc_continual_matrix_run(**common)
    else:
        result = run_derpp_continual_matrix_run(**common)
    print(
        json.dumps(
            {
                "run": result["run"],
                "method": result["method"],
                "order": result["order"],
                "seed": result["seed"],
                "pairwise_forgetting": result["pairwise_forgetting"],
                "memory": result["memory"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
