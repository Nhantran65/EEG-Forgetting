#!/usr/bin/env python3
"""Run one predeclared CBraMod linear-probe/depth-pilot job."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eeg_forgetting.data.contracts import load_yaml
from eeg_forgetting.training.pilot import PilotSettings, run_pilot


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("bciciv2a", "physionet_mi", "sleep_edf_sc"),
    )
    parser.add_argument("--depth", required=True, type=int, choices=(0, 1, 2, 4, 8))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "pilots" / "cbramod_depth_v3.yaml",
    )
    parser.add_argument(
        "--cache-root", type=Path, default=PROJECT_ROOT / "data" / "processed" / "v3"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "pilots" / "cbramod_depth_v3",
    )
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if args.dataset not in config["datasets"]:
        raise ValueError(f"{args.dataset} is not declared by {args.config}")
    if args.depth not in config["depths"]:
        raise ValueError(f"depth {args.depth} is not declared by {args.config}")
    optimizer = config["optimizer"]
    schedule = config["schedule"]
    settings = PilotSettings(
        dataset=args.dataset,
        depth=args.depth,
        head=str(config["head"]),
        pilot_id=str(config["id"]),
        seed=int(config["seed"]),
        batch_size=int(config["batch_size"]),
        optimizer_steps=int(schedule["optimizer_steps"]),
        validation_interval_steps=int(schedule["validation_interval_steps"]),
        backbone_learning_rate=float(optimizer["backbone_learning_rate"]),
        head_learning_rate=float(optimizer["head_learning_rate"]),
        weight_decay=float(optimizer["weight_decay"]),
        gradient_clip_norm=float(optimizer["gradient_clip_norm"]),
        label_smoothing=float(config["loss"]["label_smoothing"]),
        minimum_learning_rate=float(schedule["minimum_learning_rate"]),
    )
    model = load_yaml(PROJECT_ROOT / "configs" / "models" / "cbramod.yaml")
    checkpoint = model["checkpoint"]
    output = args.output_root / args.dataset / f"depth-{args.depth}" / f"seed-{settings.seed}"
    result = run_pilot(
        settings,
        cache_root=args.cache_root,
        checkpoint_path=PROJECT_ROOT / str(checkpoint["local_path"]),
        checkpoint_sha256=str(checkpoint["sha256"]),
        output_dir=output,
        device=args.device,
    )
    output = (
        {
            "pilot": result["pilot"],
            "dataset": args.dataset,
            "depth": args.depth,
            "best_step": result["best_step"],
            "best_mean_subject_balanced_accuracy": result[
                "best_mean_subject_balanced_accuracy"
            ],
        }
        if args.summary_only
        else result
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
