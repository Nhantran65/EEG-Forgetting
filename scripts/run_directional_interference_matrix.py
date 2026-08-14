#!/usr/bin/env python3
"""Run a resumable subset of the directional-interference matrix."""

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
        / "analysis"
        / "directional_interference_v1.yaml",
    )
    parser.add_argument("--directions", nargs="+")
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    root = args.output_root or (
        PROJECT_ROOT / "results" / "analysis" / str(config["id"])
    )
    declared = tuple(config["directions"])
    directions = declared if args.directions is None else tuple(args.directions)
    if not directions or any(direction not in declared for direction in directions):
        raise ValueError("--directions must be a non-empty declared subset")
    for direction in directions:
        for seed in config["seeds"]:
            result = root / direction / f"seed-{seed}" / "result.json"
            if result.exists():
                print(f"skip completed directional diagnostic: {result}", flush=True)
                continue
            subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "run_directional_interference.py"),
                    "--config",
                    str(args.config),
                    "--direction",
                    direction,
                    "--seed",
                    str(seed),
                    "--device",
                    args.device,
                    "--output-root",
                    str(root),
                ],
                cwd=PROJECT_ROOT,
                check=True,
            )


if __name__ == "__main__":
    main()
