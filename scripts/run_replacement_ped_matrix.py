#!/usr/bin/env python3
"""Run all pending order/seed cells for one replacement PED method."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from eeg_forgetting.data.contracts import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "xai" / "high_gamma_replacement_ped_v1.yaml",
    )
    parser.add_argument(
        "--method", choices=("sequential_finetuning", "ewc", "derpp"), required=True
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    method = config["methods"][args.method]
    training = load_yaml(PROJECT_ROOT / str(method["training_config"]))
    output_root = PROJECT_ROOT / str(config["output"]["root"])
    script = PROJECT_ROOT / "scripts" / "run_replacement_ped.py"
    for order in training["orders"]:
        for seed in training["seeds"]:
            result = output_root / args.method / order / f"seed-{seed}" / "result.json"
            if result.exists():
                print(f"skip complete {args.method} {order} seed={seed}", flush=True)
                continue
            print(f"launch {args.method} {order} seed={seed}", flush=True)
            subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--config",
                    str(args.config),
                    "--method",
                    args.method,
                    "--order",
                    str(order),
                    "--seed",
                    str(seed),
                    "--device",
                    args.device,
                ],
                cwd=PROJECT_ROOT,
                check=True,
            )


if __name__ == "__main__":
    main()
