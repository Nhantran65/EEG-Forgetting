#!/usr/bin/env python3
"""Run one locked exact-budget single-task CBraMod baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eeg_forgetting.data.contracts import load_yaml
from eeg_forgetting.training.pilot import PilotSettings, run_pilot


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("bciciv2a", "physionet_mi", "sleep_edf_sc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "training" / "single_task_budget.yaml",
    )
    parser.add_argument(
        "--cache-root", type=Path, default=PROJECT_ROOT / "data" / "processed" / "v3"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "single_task" / "budget_v1",
    )
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if config["mode"] != "budget_matched":
        raise ValueError("single-task baseline runner requires budget_matched mode")
    if args.dataset not in config["datasets"] or args.seed not in config["seeds"]:
        raise ValueError("dataset/seed is not declared by the locked baseline config")
    optimizer = config["optimizer"]
    schedule = config["schedule"]
    settings = PilotSettings(
        dataset=args.dataset,
        depth=int(config["plastic_final_encoder_blocks"]),
        head=str(config["head"]),
        pilot_id=str(config["id"]),
        seed=args.seed,
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
    model_config = load_yaml(PROJECT_ROOT / "configs" / "models" / "cbramod.yaml")
    checkpoint = model_config["checkpoint"]
    output_dir = args.output_root / args.dataset / f"seed-{args.seed}"
    result = run_pilot(
        settings,
        cache_root=args.cache_root,
        checkpoint_path=PROJECT_ROOT / str(checkpoint["local_path"]),
        checkpoint_sha256=str(checkpoint["sha256"]),
        output_dir=output_dir,
        device=args.device,
        save_final_checkpoint=True,
        protocol_config_path=args.config,
    )
    output = (
        {
            "run": result["pilot"],
            "dataset": args.dataset,
            "best_step": result["best_step"],
            "best_subject_ba": result["best_mean_subject_balanced_accuracy"],
            "final_step": result["final_step"],
            "final_subject_ba": result["final_validation"][
                "mean_subject_balanced_accuracy"
            ],
        }
        if args.summary_only
        else result
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
