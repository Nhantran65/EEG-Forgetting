#!/usr/bin/env python3
"""Run one locked early-stopped single-task CBraMod baseline."""

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
        default=PROJECT_ROOT / "configs" / "training" / "single_task_converged.yaml",
    )
    parser.add_argument(
        "--cache-root", type=Path, default=PROJECT_ROOT / "data" / "processed" / "v3"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "single_task" / "converged_v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if config["mode"] != "converged":
        raise ValueError("converged runner requires converged mode")
    if args.dataset not in config["datasets"] or args.seed not in config["seeds"]:
        raise ValueError("dataset/seed is not declared by the locked converged config")
    optimizer = config["optimizer"]
    schedule = config["schedule"]
    early_stopping = config["early_stopping"]
    settings = PilotSettings(
        dataset=args.dataset,
        depth=int(config["plastic_final_encoder_blocks"]),
        head=str(config["head"]),
        pilot_id=str(config["id"]),
        seed=args.seed,
        batch_size=int(config["batch_size"]),
        optimizer_steps=int(schedule["maximum_optimizer_steps"]),
        validation_interval_steps=int(schedule["validation_interval_steps"]),
        backbone_learning_rate=float(optimizer["backbone_learning_rate"]),
        head_learning_rate=float(optimizer["head_learning_rate"]),
        weight_decay=float(optimizer["weight_decay"]),
        gradient_clip_norm=float(optimizer["gradient_clip_norm"]),
        label_smoothing=float(config["loss"]["label_smoothing"]),
        minimum_learning_rate=float(schedule["minimum_learning_rate"]),
        early_stopping_patience_validations=int(
            early_stopping["patience_validations"]
        ),
        early_stopping_min_delta=float(early_stopping["min_delta"]),
    )
    model_config = load_yaml(PROJECT_ROOT / "configs" / "models" / "cbramod.yaml")
    checkpoint = model_config["checkpoint"]
    result = run_pilot(
        settings,
        cache_root=args.cache_root,
        checkpoint_path=PROJECT_ROOT / str(checkpoint["local_path"]),
        checkpoint_sha256=str(checkpoint["sha256"]),
        output_dir=args.output_root / args.dataset / f"seed-{args.seed}",
        device=args.device,
        protocol_config_path=args.config,
    )
    print(
        json.dumps(
            {
                "run": result["pilot"],
                "dataset": args.dataset,
                "completed_steps": result["completed_steps"],
                "stop_reason": result["stop_reason"],
                "best_step": result["best_step"],
                "best_subject_ba": result["best_mean_subject_balanced_accuracy"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
