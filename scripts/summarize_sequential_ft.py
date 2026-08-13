#!/usr/bin/env python3
"""Verify and summarize the locked sequential-FT seed/order matrix."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from scipy.stats import spearmanr

from eeg_forgetting.data.contracts import DatasetProtocolError, load_yaml
from eeg_forgetting.data.manifests import sha256_file
from eeg_forgetting.training.continual import aggregate_replicates


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "training" / "sequential_ft_v1.yaml",
    )
    parser.add_argument("--result-root", type=Path)
    parser.add_argument(
        "--fisher-summary",
        type=Path,
        default=PROJECT_ROOT / "results" / "fisher" / "instrument_v1" / "summary.json",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _pair_key(left: str, right: str, canonical: list[str]) -> str:
    ordered = sorted((left, right), key=canonical.index)
    return "__".join(ordered)


def _verify_run(path: Path, config_sha: str) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    if result["config_sha256"] != config_sha:
        raise DatasetProtocolError(f"continual config digest mismatch: {path}")
    for stage in result["stages"]:
        directory = path.parent / f"stage-{stage['stage']:02d}-{stage['learned_task']}"
        stage_path = directory / "stage.json"
        if sha256_file(stage_path) != stage["stage_result_sha256"]:
            raise DatasetProtocolError(f"continual stage digest mismatch: {stage_path}")
        with stage_path.open(encoding="utf-8") as handle:
            document = json.load(handle)
        checkpoint = directory / document["checkpoint"]["file"]
        if sha256_file(checkpoint) != document["checkpoint"]["sha256"]:
            raise DatasetProtocolError(f"continual checkpoint digest mismatch: {checkpoint}")
        for prediction in document["predictions"].values():
            prediction_path = directory / prediction["file"]
            if sha256_file(prediction_path) != prediction["sha256"]:
                raise DatasetProtocolError(
                    f"continual prediction digest mismatch: {prediction_path}"
                )
    return result


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    config_sha = sha256_file(args.config)
    result_root = args.result_root or (
        PROJECT_ROOT / "results" / "continual" / str(config["id"])
    )
    with args.fisher_summary.open(encoding="utf-8") as handle:
        fisher = json.load(handle)
    if not fisher["instrument_gate_passed"]:
        raise DatasetProtocolError("cannot summarize CL against a failed Fisher instrument")
    canonical = list(config["canonical_tasks"])
    directed: dict[str, list[float]] = defaultdict(list)
    pair_values: dict[str, list[float]] = defaultdict(list)
    runs = {}
    for order_name, expected_order in config["orders"].items():
        for seed in config["seeds"]:
            result_path = result_root / order_name / f"seed-{seed}" / "result.json"
            result = _verify_run(result_path, config_sha)
            if result["order"] != expected_order or int(result["seed"]) != int(seed):
                raise DatasetProtocolError(f"continual run identity mismatch: {result_path}")
            run_id = f"{order_name}:seed-{seed}"
            runs[run_id] = sha256_file(result_path)
            for row in result["pairwise_forgetting"]:
                if not row["relative_valid"]:
                    continue
                value = float(row["relative_forgetting"])
                direction = f"{row['old_task']}<-{row['learned_task']}"
                pair = _pair_key(row["old_task"], row["learned_task"], canonical)
                directed[direction].append(value)
                pair_values[pair].append(value)

    directed_summary = {
        key: aggregate_replicates(values) for key, values in sorted(directed.items())
    }
    pair_summary = {
        key: aggregate_replicates(values) for key, values in sorted(pair_values.items())
    }
    gate = config["analysis_gate"]
    minimum_n = int(gate["minimum_valid_replicates_per_direction"])
    minimum_sign = float(gate["minimum_same_sign_fraction_per_direction"])
    signal_directions = 0
    sign_consistent_directions = 0
    for values in directed_summary.values():
        if int(values["n"]) < minimum_n:
            continue
        sd = float(values["sample_sd"])
        if abs(float(values["mean"])) > sd:
            signal_directions += 1
        same_sign = max(float(values["positive_fraction"]), float(values["negative_fraction"]))
        if same_sign >= minimum_sign:
            sign_consistent_directions += 1
    required_directions = int(gate["minimum_signal_directions"])
    enough_replicates = len(directed_summary) == 6 and all(
        int(values["n"]) >= minimum_n for values in directed_summary.values()
    )
    stability_gate_passed = (
        enough_replicates
        and signal_directions >= required_directions
        and sign_consistent_directions >= required_directions
    )
    overlap = fisher["cross_task_cosine"]
    shared_pairs = sorted(pair_summary)
    rho, pvalue = spearmanr(
        [float(overlap[pair]) for pair in shared_pairs],
        [float(pair_summary[pair]["mean"]) for pair in shared_pairs],
    )
    summary = {
        "schema_version": 1,
        "run": config["id"],
        "config_sha256": config_sha,
        "fisher_summary_sha256": sha256_file(args.fisher_summary),
        "input_result_sha256": runs,
        "directed_relative_forgetting": directed_summary,
        "pair_relative_forgetting": pair_summary,
        "pair_fisher_overlap": {pair: overlap[pair] for pair in shared_pairs},
        "exploratory_pair_spearman": {
            "n_pairs": len(shared_pairs),
            "rho": float(rho),
            "pvalue": float(pvalue),
            "warning": "three_pairs_only_do_not_treat_as_confirmatory_inference",
        },
        "gate": {
            "enough_replicates": enough_replicates,
            "signal_directions": signal_directions,
            "sign_consistent_directions": sign_consistent_directions,
            "required_directions": required_directions,
            "passed": stability_gate_passed,
        },
    }
    output = args.output or result_root / "summary.json"
    if output.exists():
        raise DatasetProtocolError(f"refusing to overwrite sequential summary {output}")
    temporary = output.with_suffix(output.suffix + ".tmp")
    output.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
