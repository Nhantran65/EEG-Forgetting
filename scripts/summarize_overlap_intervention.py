#!/usr/bin/env python3
"""Verify and aggregate the locked Fisher-overlap intervention matrix."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

from eeg_forgetting.data.contracts import DatasetProtocolError, load_yaml
from eeg_forgetting.data.manifests import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "configs"
        / "training"
        / "bci_sleep_overlap_intervention_v1.yaml",
    )
    parser.add_argument("--result-root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _stats(values: list[float]) -> dict[str, object]:
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values),
        "values": values,
    }


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    config_sha = sha256_file(args.config)
    old_task = str(config["old_task"])
    new_task = str(config["new_task"])
    root = args.result_root or (
        PROJECT_ROOT / "results" / "interventions" / str(config["id"])
    )
    ratios = tuple(float(value) for value in config["mask"]["freeze_ratios"])
    random_count = int(config["mask"]["random_controls_per_ratio"])
    conditions = ("high_overlap",) + tuple(
        f"random_{index}" for index in range(random_count)
    )
    inputs: dict[str, str] = {}
    mask_hashes: dict[tuple[float, str], set[str]] = defaultdict(set)
    values: dict[tuple[float, str, str, str], list[float]] = defaultdict(list)
    for ratio in ratios:
        for condition in conditions:
            for seed in config["seeds"]:
                path = (
                    root
                    / f"ratio-{ratio:g}"
                    / condition
                    / f"seed-{seed}"
                    / "result.json"
                )
                with path.open(encoding="utf-8") as handle:
                    result = json.load(handle)
                if (
                    result["config_sha256"] != config_sha
                    or int(result["seed"]) != int(seed)
                    or float(result["ratio"]) != ratio
                    or result["condition"] != condition
                ):
                    raise DatasetProtocolError(
                        f"intervention result identity mismatch: {path}"
                    )
                checkpoint = path.parent / result["checkpoint"]["file"]
                if sha256_file(checkpoint) != result["checkpoint"]["sha256"]:
                    raise DatasetProtocolError(
                        f"intervention checkpoint digest mismatch: {checkpoint}"
                    )
                for prediction in result["predictions"].values():
                    prediction_path = path.parent / prediction["file"]
                    if sha256_file(prediction_path) != prediction["sha256"]:
                        raise DatasetProtocolError(
                            f"intervention prediction digest mismatch: {prediction_path}"
                        )
                mask_hashes[(ratio, condition)].add(result["mask"]["sha256"])
                for split in ("validation", "test"):
                    relative = result["forgetting"][split]["relative_forgetting"]
                    if relative is None:
                        raise DatasetProtocolError(
                            f"invalid {old_task} forgetting headroom in {path}"
                        )
                    values[(ratio, condition, split, "old_relative_forgetting")].append(
                        float(relative)
                    )
                    values[(ratio, condition, split, "new_subject_ba")].append(
                        float(
                            result["evaluations"][split][new_task][
                                "mean_subject_balanced_accuracy"
                            ]
                        )
                    )
                inputs[str(path.relative_to(root))] = sha256_file(path)
    if any(len(hashes) != 1 for hashes in mask_hashes.values()):
        raise DatasetProtocolError("an intervention mask changed across training seeds")
    for ratio in ratios:
        ratio_hashes = [next(iter(mask_hashes[(ratio, condition)])) for condition in conditions]
        if len(set(ratio_hashes)) != len(ratio_hashes):
            raise DatasetProtocolError(f"duplicate intervention controls at ratio {ratio:g}")

    tolerance = float(config["evaluation"]["matched_new_task_tolerance_absolute"])
    ratio_results = {}
    passing = 0
    for ratio in ratios:
        split_results = {}
        for split in ("validation", "test"):
            high_new = values[(ratio, "high_overlap", split, "new_subject_ba")]
            random_new = sum(
                (
                    values[(ratio, condition, split, "new_subject_ba")]
                    for condition in conditions[1:]
                ),
                [],
            )
            high_forgetting = values[
                (ratio, "high_overlap", split, "old_relative_forgetting")
            ]
            random_forgetting = sum(
                (
                    values[
                        (ratio, condition, split, "old_relative_forgetting")
                    ]
                    for condition in conditions[1:]
                ),
                [],
            )
            new_difference = statistics.mean(high_new) - statistics.mean(random_new)
            forgetting_difference = statistics.mean(high_forgetting) - statistics.mean(
                random_forgetting
            )
            split_results[split] = {
                "high_overlap": {
                    "new_task_subject_ba": _stats(high_new),
                    "old_task_relative_forgetting": _stats(high_forgetting),
                },
                "random_controls_pooled": {
                    "new_task_subject_ba": _stats(random_new),
                    "old_task_relative_forgetting": _stats(random_forgetting),
                },
                "high_minus_random_new_task_subject_ba": new_difference,
                "high_minus_random_old_task_relative_forgetting": forgetting_difference,
                "new_task_performance_matched": abs(new_difference) <= tolerance,
                "old_task_better_protected": forgetting_difference < 0.0,
            }
        test_gate = split_results["test"]
        passed = bool(
            test_gate["new_task_performance_matched"]
            and test_gate["old_task_better_protected"]
        )
        passing += int(passed)
        ratio_results[f"{ratio:g}"] = {
            "mask_sha256_by_condition": {
                condition: next(iter(mask_hashes[(ratio, condition)]))
                for condition in conditions
            },
            "splits": split_results,
            "test_gate_passed": passed,
        }
    required = int(config["claim_gate"]["minimum_matched_and_protected_ratios"])
    summary = {
        "schema_version": 1,
        "run": config["id"],
        "old_task": old_task,
        "new_task": new_task,
        "config_sha256": config_sha,
        "input_result_sha256": inputs,
        "ratios": ratio_results,
        "claim_gate": {
            "passing_ratios": passing,
            "required_ratios": required,
            "passed": passing >= required,
            "interpretation": (
                "high-overlap localization causally supported"
                if passing >= required
                else "high-overlap localization not causally supported"
            ),
        },
    }
    output = args.output or root / "summary.json"
    if output.exists():
        raise DatasetProtocolError(f"refusing to overwrite intervention summary {output}")
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
