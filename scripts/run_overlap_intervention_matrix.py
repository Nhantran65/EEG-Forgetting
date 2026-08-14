#!/usr/bin/env python3
"""Run every declared overlap intervention, resuming at run boundaries."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from eeg_forgetting.data.contracts import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "configs"
        / "training"
        / "bci_sleep_overlap_intervention_v1.yaml",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        help="Optional declared ratio subset for splitting the matrix across devices.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        help="Optional declared condition subset for splitting the matrix.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    root = args.output_root or (
        PROJECT_ROOT / "results" / "interventions" / str(config["id"])
    )
    default_conditions = ["high_overlap"] + [
        f"random_{index}"
        for index in range(int(config["mask"]["random_controls_per_ratio"]))
    ]
    declared_conditions = tuple(
        str(value) for value in config["mask"].get("conditions", default_conditions)
    )
    conditions = (
        declared_conditions if args.conditions is None else tuple(args.conditions)
    )
    if not conditions or any(value not in declared_conditions for value in conditions):
        raise ValueError(
            "--conditions must be a non-empty subset declared by the config"
        )
    declared_ratios = tuple(float(value) for value in config["mask"]["freeze_ratios"])
    ratios = declared_ratios if args.ratios is None else tuple(args.ratios)
    if not ratios or any(float(value) not in declared_ratios for value in ratios):
        raise ValueError("--ratios must be a non-empty subset declared by the config")
    for ratio in ratios:
        for condition in conditions:
            for seed in config["seeds"]:
                result = (
                    root
                    / f"ratio-{float(ratio):g}"
                    / condition
                    / f"seed-{seed}"
                    / "result.json"
                )
                if result.exists():
                    print(f"skip completed intervention: {result}", flush=True)
                    continue
                command = [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "run_overlap_intervention.py"),
                    "--config",
                    str(args.config),
                    "--seed",
                    str(seed),
                    "--ratio",
                    str(ratio),
                    "--condition",
                    condition,
                    "--device",
                    args.device,
                    "--output-root",
                    str(root),
                ]
                subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
