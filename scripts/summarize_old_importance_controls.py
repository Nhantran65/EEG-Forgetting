#!/usr/bin/env python3
"""Compare high-overlap freezing with matched old-task-only importance controls."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

from eeg_forgetting.data.contracts import DatasetProtocolError, load_yaml
from eeg_forgetting.data.manifests import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = (
    PROJECT_ROOT
    / "configs"
    / "training"
    / "bci_sleep_old_importance_control_v1.yaml",
    PROJECT_ROOT
    / "configs"
    / "training"
    / "physionet_bci_old_importance_control_v1.yaml",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "results"
        / "interventions"
        / "old_importance_controls_v1"
        / "summary.json",
    )
    return parser.parse_args()


def _stats(values: list[float]) -> dict[str, object]:
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values),
        "values": values,
    }


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _load_verified_result(
    path: Path,
    *,
    config_sha256: str,
    seed: int,
    ratio: float,
    condition: str,
    old_task: str,
    new_task: str,
) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    if (
        result["config_sha256"] != config_sha256
        or int(result["seed"]) != seed
        or float(result["ratio"]) != ratio
        or result["condition"] != condition
        or result["old_task"] != old_task
        or result["new_task"] != new_task
    ):
        raise DatasetProtocolError(f"intervention result identity mismatch: {path}")
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
    return result


def main() -> None:
    args = parse_args()
    pair_results = {}
    input_hashes = {}
    test_gates = []
    for config_path in args.configs:
        config = load_yaml(config_path)
        config_sha = sha256_file(config_path)
        old_task = str(config["old_task"])
        new_task = str(config["new_task"])
        pair_key = f"{old_task}_to_{new_task}"
        if pair_key in pair_results:
            raise DatasetProtocolError(f"duplicate old-importance pair {pair_key}")
        if tuple(config["mask"].get("conditions", ())) != ("old_only",):
            raise DatasetProtocolError(f"control config must declare only old_only: {config_path}")

        reference = config["reference_intervention"]
        reference_config = _resolve(str(reference["config_path"]))
        reference_summary = _resolve(str(reference["summary_path"]))
        reference_root = _resolve(str(reference["result_root"]))
        if sha256_file(reference_config) != str(reference["config_sha256"]):
            raise DatasetProtocolError(f"reference config digest mismatch: {reference_config}")
        if sha256_file(reference_summary) != str(reference["summary_sha256"]):
            raise DatasetProtocolError(f"reference summary digest mismatch: {reference_summary}")
        control_root = PROJECT_ROOT / "results" / "interventions" / str(config["id"])
        tolerance = float(config["evaluation"]["matched_new_task_tolerance_absolute"])
        ratios = {}
        for ratio_value in config["mask"]["freeze_ratios"]:
            ratio = float(ratio_value)
            split_values = {
                split: {
                    "high_forgetting": [],
                    "old_only_forgetting": [],
                    "high_new_ba": [],
                    "old_only_new_ba": [],
                }
                for split in ("validation", "test")
            }
            mask_hashes = {"high_overlap": set(), "old_only": set()}
            for seed_value in config["seeds"]:
                seed = int(seed_value)
                reference_path = (
                    reference_root
                    / f"ratio-{ratio:g}"
                    / "high_overlap"
                    / f"seed-{seed}"
                    / "result.json"
                )
                control_path = (
                    control_root
                    / f"ratio-{ratio:g}"
                    / "old_only"
                    / f"seed-{seed}"
                    / "result.json"
                )
                high = _load_verified_result(
                    reference_path,
                    config_sha256=str(reference["config_sha256"]),
                    seed=seed,
                    ratio=ratio,
                    condition="high_overlap",
                    old_task=old_task,
                    new_task=new_task,
                )
                old_only = _load_verified_result(
                    control_path,
                    config_sha256=config_sha,
                    seed=seed,
                    ratio=ratio,
                    condition="old_only",
                    old_task=old_task,
                    new_task=new_task,
                )
                if (
                    high["mask"]["frozen_elements"]
                    != old_only["mask"]["frozen_elements"]
                    or high["mask"]["frozen_by_layer"]
                    != old_only["mask"]["frozen_by_layer"]
                ):
                    raise DatasetProtocolError(
                        f"high/old-only mask budget mismatch at {pair_key} ratio {ratio:g}"
                    )
                mask_hashes["high_overlap"].add(str(high["mask"]["sha256"]))
                mask_hashes["old_only"].add(str(old_only["mask"]["sha256"]))
                for split in ("validation", "test"):
                    high_forgetting = high["forgetting"][split]["relative_forgetting"]
                    old_forgetting = old_only["forgetting"][split]["relative_forgetting"]
                    if high_forgetting is None or old_forgetting is None:
                        raise DatasetProtocolError(
                            f"invalid forgetting headroom at {pair_key} ratio {ratio:g}"
                        )
                    split_values[split]["high_forgetting"].append(float(high_forgetting))
                    split_values[split]["old_only_forgetting"].append(float(old_forgetting))
                    split_values[split]["high_new_ba"].append(
                        float(
                            high["evaluations"][split][new_task][
                                "mean_subject_balanced_accuracy"
                            ]
                        )
                    )
                    split_values[split]["old_only_new_ba"].append(
                        float(
                            old_only["evaluations"][split][new_task][
                                "mean_subject_balanced_accuracy"
                            ]
                        )
                    )
                input_hashes[str(reference_path.relative_to(PROJECT_ROOT))] = sha256_file(
                    reference_path
                )
                input_hashes[str(control_path.relative_to(PROJECT_ROOT))] = sha256_file(
                    control_path
                )
            if any(len(values) != 1 for values in mask_hashes.values()):
                raise DatasetProtocolError(
                    f"mask changed across training seeds at {pair_key} ratio {ratio:g}"
                )
            if mask_hashes["high_overlap"] == mask_hashes["old_only"]:
                raise DatasetProtocolError(
                    f"high and old-only masks are identical at {pair_key} ratio {ratio:g}"
                )
            split_results = {}
            for split, values in split_values.items():
                forgetting_difference = [
                    high - old
                    for high, old in zip(
                        values["high_forgetting"],
                        values["old_only_forgetting"],
                        strict=True,
                    )
                ]
                new_difference = [
                    high - old
                    for high, old in zip(
                        values["high_new_ba"], values["old_only_new_ba"], strict=True
                    )
                ]
                matched = abs(statistics.mean(new_difference)) <= tolerance
                protected = statistics.mean(forgetting_difference) < 0.0
                split_results[split] = {
                    "high_overlap": {
                        "old_task_relative_forgetting": _stats(
                            values["high_forgetting"]
                        ),
                        "new_task_subject_ba": _stats(values["high_new_ba"]),
                    },
                    "old_only": {
                        "old_task_relative_forgetting": _stats(
                            values["old_only_forgetting"]
                        ),
                        "new_task_subject_ba": _stats(values["old_only_new_ba"]),
                    },
                    "paired_high_minus_old_only_old_task_relative_forgetting": _stats(
                        forgetting_difference
                    ),
                    "paired_high_minus_old_only_new_task_subject_ba": _stats(
                        new_difference
                    ),
                    "new_task_performance_matched": matched,
                    "high_overlap_better_protected": protected,
                }
            test_passed = bool(
                split_results["test"]["new_task_performance_matched"]
                and split_results["test"]["high_overlap_better_protected"]
            )
            test_gates.append(test_passed)
            ratios[f"{ratio:g}"] = {
                "mask_sha256": {
                    condition: next(iter(values))
                    for condition, values in mask_hashes.items()
                },
                "splits": split_results,
                "test_gate_passed": test_passed,
            }
        pair_results[pair_key] = {
            "control_config_sha256": config_sha,
            "reference_config_sha256": str(reference["config_sha256"]),
            "reference_summary_sha256": str(reference["summary_sha256"]),
            "ratios": ratios,
        }

    passed = bool(test_gates and all(test_gates))
    summary = {
        "schema_version": 1,
        "run": "old_importance_controls_v1",
        "comparison": "high_overlap_vs_old_task_fisher_only",
        "input_result_sha256": input_hashes,
        "pairs": pair_results,
        "claim_gate": {
            "passing_pair_ratios": sum(test_gates),
            "required_pair_ratios": len(test_gates),
            "passed": passed,
            "interpretation": (
                "shared_overlap_adds_beyond_old_task_importance"
                if passed
                else "no_consistent_evidence_shared_overlap_adds_beyond_old_task_importance"
            ),
            "scope": "descriptive_three_seed_final_control",
        },
    }
    output = args.output
    if output.exists():
        raise DatasetProtocolError(f"refusing to overwrite control summary {output}")
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
