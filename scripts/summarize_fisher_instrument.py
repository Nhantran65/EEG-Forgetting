#!/usr/bin/env python3
"""Summarize split-half stability and cross-task Fisher overlap."""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

from eeg_forgetting.data.contracts import DatasetProtocolError, load_yaml
from eeg_forgetting.data.manifests import sha256_file
from eeg_forgetting.training.fisher import fisher_cosine, layer_l2_normalize, load_fisher


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "pilots" / "fisher_instrument_v1.yaml",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "fisher" / "instrument_v1",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    seed = int(config["model_seed"])
    full = {}
    within = {}
    input_results = {}
    for dataset in config["datasets"]:
        directory = args.result_root / dataset / f"seed-{seed}"
        result_path = directory / "result.json"
        with result_path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        if result["config_sha256"] != sha256_file(args.config):
            raise DatasetProtocolError(f"{dataset}: Fisher result config digest changed")
        full_path = directory / result["signatures"]["full.pt"]["file"]
        if sha256_file(full_path) != result["signatures"]["full.pt"]["sha256"]:
            raise DatasetProtocolError(f"{dataset}: Fisher signature digest changed")
        within[dataset] = float(result["split_half_cosine"])
        full[dataset] = layer_l2_normalize(load_fisher(full_path))
        input_results[dataset] = sha256_file(result_path)
    cross = {
        f"{left}__{right}": fisher_cosine(full[left], full[right])
        for left, right in itertools.combinations(config["datasets"], 2)
    }
    minimum_within = min(within.values())
    maximum_cross = max(cross.values())
    gate = config["instrument_gate"]
    passed = (
        minimum_within >= float(gate["minimum_split_half_cosine"])
        and minimum_within - maximum_cross
        >= float(gate["minimum_margin_above_max_cross_task_cosine"])
    )
    summary = {
        "run": config["id"],
        "config_sha256": sha256_file(args.config),
        "input_result_sha256": input_results,
        "split_half_cosine": within,
        "cross_task_cosine": cross,
        "minimum_split_half_cosine": minimum_within,
        "maximum_cross_task_cosine": maximum_cross,
        "margin": minimum_within - maximum_cross,
        "instrument_gate_passed": passed,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    output = args.output or args.result_root / "summary.json"
    if output.exists():
        raise DatasetProtocolError(f"refusing to overwrite Fisher summary {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(output)
    if not passed:
        raise DatasetProtocolError(
            "Fisher instrument gate failed; follow the configured escalation before CL"
        )


if __name__ == "__main__":
    main()
