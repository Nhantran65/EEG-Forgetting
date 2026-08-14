#!/usr/bin/env python3
"""Verify and aggregate the locked three-seed joint-training upper bound."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

from eeg_forgetting.data.contracts import DatasetProtocolError, load_yaml
from eeg_forgetting.data.manifests import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "training" / "joint_v1.yaml",
    )
    parser.add_argument("--result-root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    config_sha = sha256_file(args.config)
    root = args.result_root or PROJECT_ROOT / "results" / "joint" / str(config["id"])
    inputs = {}
    scores = {task: [] for task in config["canonical_tasks"]}
    for seed in config["seeds"]:
        path = root / f"seed-{seed}" / "result.json"
        with path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        if result["config_sha256"] != config_sha or int(result["seed"]) != int(seed):
            raise DatasetProtocolError(f"joint result identity mismatch: {path}")
        checkpoint = path.parent / result["checkpoint"]["file"]
        if sha256_file(checkpoint) != result["checkpoint"]["sha256"]:
            raise DatasetProtocolError(f"joint checkpoint digest mismatch: {checkpoint}")
        for task, prediction in result["predictions"].items():
            prediction_path = path.parent / prediction["file"]
            if sha256_file(prediction_path) != prediction["sha256"]:
                raise DatasetProtocolError(f"joint prediction digest mismatch: {prediction_path}")
            scores[task].append(
                float(result["evaluations"][task]["mean_subject_balanced_accuracy"])
            )
        inputs[f"seed-{seed}"] = sha256_file(path)
    summary = {
        "schema_version": 1,
        "run": config["id"],
        "config_sha256": config_sha,
        "input_result_sha256": inputs,
        "task_subject_balanced_accuracy": {
            task: {
                "n": len(values),
                "mean": statistics.mean(values),
                "sample_sd": statistics.stdev(values),
                "values": values,
            }
            for task, values in scores.items()
        },
        "memory": {"continual_state_bytes": 0},
        "role": "offline_joint_upper_bound_not_continual_method",
    }
    output = args.output or root / "summary.json"
    if output.exists():
        raise DatasetProtocolError(f"refusing to overwrite joint summary {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
