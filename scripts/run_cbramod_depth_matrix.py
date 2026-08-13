#!/usr/bin/env python3
"""Distribute a locked CBraMod depth matrix across one worker per GPU."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path

from eeg_forgetting.data.contracts import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("bciciv2a", "physionet_mi", "sleep_edf_sc")
DEPTHS = (0, 1, 2, 4, 8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs="+", type=int, default=(0, 1, 2, 3))
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=DATASETS
    )
    parser.add_argument("--depths", nargs="+", type=int, choices=DEPTHS, default=DEPTHS)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "pilots" / "cbramod_depth_v3.yaml",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "pilots" / "cbramod_depth_v3",
    )
    return parser.parse_args()


def run_queue(
    device: int,
    jobs: list[tuple[str, int]],
    output_root: Path,
    config_path: Path,
    seed: int,
) -> None:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(device)
    environment.setdefault("UV_CACHE_DIR", "/tmp/eeg-forgetting-uv-cache")
    for dataset, depth in jobs:
        result = output_root / dataset / f"depth-{depth}" / f"seed-{seed}" / "result.json"
        if result.is_file():
            print(f"GPU {device}: skip completed {dataset}/depth-{depth}", flush=True)
            continue
        print(f"GPU {device}: start {dataset}/depth-{depth}", flush=True)
        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_cbramod_depth_pilot.py"),
                "--dataset",
                dataset,
                "--depth",
                str(depth),
                "--device",
                "cuda:0",
                "--config",
                str(config_path),
                "--output-root",
                str(output_root),
                "--summary-only",
            ],
            cwd=PROJECT_ROOT,
            env=environment,
            check=True,
        )


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if len(args.devices) != len(set(args.devices)) or not args.devices:
        raise ValueError("--devices must contain unique GPU indices")
    if any(dataset not in config["datasets"] for dataset in args.datasets):
        raise ValueError("--datasets contains a task not declared by the pilot config")
    if any(depth not in config["depths"] for depth in args.depths):
        raise ValueError("--depths contains a value not declared by the pilot config")
    seed = int(config["seed"])
    queues = {device: [] for device in args.devices}
    jobs = list(product(args.datasets, args.depths))
    for index, job in enumerate(jobs):
        queues[args.devices[index % len(args.devices)]].append(job)
    with ThreadPoolExecutor(max_workers=len(args.devices)) as executor:
        futures = [
            executor.submit(
                run_queue, device, queues[device], args.output_root, args.config, seed
            )
            for device in args.devices
        ]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
