#!/usr/bin/env python3
"""Train the one locked High-Gamma candidate seed without touching test data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eeg_forgetting.data.contracts import DatasetProtocolError, load_yaml
from eeg_forgetting.training.pilot import PilotSettings, run_pilot


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "pilots" / "high_gamma_candidate_v1.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if config.get("status") != "locked_replacement_candidate" or config.get("test_access") != "forbidden":
        raise DatasetProtocolError("High-Gamma training config is not locked/test-blind")
    training = config["training"]
    model_config = load_yaml(PROJECT_ROOT / "configs" / "models" / "cbramod.yaml")
    checkpoint = model_config["checkpoint"]
    settings = PilotSettings(
        dataset="high_gamma",
        depth=int(config["model"]["plastic_final_encoder_blocks"]),
        head=str(config["model"]["head"]),
        pilot_id=str(config["id"]),
        seed=int(config["seed"]),
        batch_size=int(training["batch_size"]),
        optimizer_steps=int(training["optimizer_steps"]),
        validation_interval_steps=int(training["validation_interval_steps"]),
        backbone_learning_rate=float(training["backbone_learning_rate"]),
        head_learning_rate=float(training["head_learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        gradient_clip_norm=float(training["gradient_clip_norm"]),
        label_smoothing=float(training["label_smoothing"]),
        minimum_learning_rate=float(training["minimum_learning_rate"]),
    )
    result = run_pilot(
        settings,
        cache_root=PROJECT_ROOT / str(config["cache_root"]),
        checkpoint_path=PROJECT_ROOT / str(checkpoint["local_path"]),
        checkpoint_sha256=str(checkpoint["sha256"]),
        output_dir=PROJECT_ROOT / "results" / "pilots" / str(config["id"]) / "seed-42",
        device=args.device,
        save_final_checkpoint=True,
        protocol_config_path=args.config,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
