#!/usr/bin/env python3
"""Verify and summarize the replacement noise-corrected PED scale."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
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
        default=PROJECT_ROOT / "configs" / "xai" / "high_gamma_replacement_ped_v1.yaml",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if config.get("status") != "locked_xai_scale":
        raise DatasetProtocolError("replacement PED summary config is not locked")
    config_sha = sha256_file(args.config)
    output_root = _resolve(str(config["output"]["root"]))
    input_sha = {}
    cells = []
    by_method_direction: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    by_key = {}
    for method, specification in config["methods"].items():
        training = load_yaml(_resolve(str(specification["training_config"])))
        for order in training["orders"]:
            for seed in training["seeds"]:
                result_path = output_root / method / order / f"seed-{seed}" / "result.json"
                if not result_path.is_file():
                    raise DatasetProtocolError(f"missing PED result {result_path}")
                with result_path.open(encoding="utf-8") as handle:
                    result = json.load(handle)
                if (
                    result["config_sha256"] != config_sha
                    or result["method"] != method
                    or result["order_name"] != order
                    or int(result["seed"]) != int(seed)
                    or result["test_was_loaded_for_xai"] is not False
                ):
                    raise DatasetProtocolError(f"PED result identity mismatch {result_path}")
                input_sha[f"{method}:{order}:seed-{seed}"] = sha256_file(result_path)
                for artifact in result["artifacts"].values():
                    artifact_path = result_path.parent / str(artifact["file"])
                    if sha256_file(artifact_path) != artifact["sha256"]:
                        raise DatasetProtocolError(f"PED map digest mismatch {artifact_path}")
                for transition in result["transitions"]:
                    direction = f"{transition['old_task']}<-{transition['learned_task']}"
                    subject_values = [
                        float(value["noise_corrected_ped"])
                        for value in transition["subjects"].values()
                    ]
                    performance = transition["performance"]
                    cell = {
                        "method": method,
                        "order": order,
                        "seed": int(seed),
                        "direction": direction,
                        "old_task": transition["old_task"],
                        "learned_task": transition["learned_task"],
                        "mean_subject_ped": float(np.mean(subject_values)),
                        "median_subject_ped": float(np.median(subject_values)),
                        "minimum_subject_ped": float(np.min(subject_values)),
                        "maximum_subject_ped": float(np.max(subject_values)),
                        "group_ped": float(transition["group"]["noise_corrected_ped"]),
                        "group_cross_jsd": float(transition["group"]["cross_checkpoint_jsd"]),
                        "group_within_noise": float(
                            transition["group"]["within_checkpoint_noise_jsd"]
                        ),
                        "raw_forgetting": float(performance["raw_forgetting"]),
                        "relative_forgetting": (
                            float(performance["relative_forgetting"])
                            if performance["relative_valid"]
                            else None
                        ),
                        "relative_valid": bool(performance["relative_valid"]),
                        "subjects": transition["subjects"],
                    }
                    cells.append(cell)
                    by_method_direction[(method, direction)].append(cell)
                    by_key[(method, order, int(seed), direction)] = cell
    if len(input_sha) != int(config["scale"]["expected_run_cells"]):
        raise DatasetProtocolError("PED run-cell count changed")
    if len(cells) != int(config["scale"]["expected_transition_cells"]):
        raise DatasetProtocolError("PED transition-cell count changed")
    aggregates = {}
    for (method, direction), rows in sorted(by_method_direction.items()):
        valid_forgetting = [
            float(row["relative_forgetting"])
            for row in rows
            if row["relative_forgetting"] is not None
        ]
        aggregates.setdefault(method, {})[direction] = {
            "transition_cells": len(rows),
            "mean_subject_ped": aggregate_replicates(
                [float(row["mean_subject_ped"]) for row in rows]
            ),
            "group_ped": aggregate_replicates([float(row["group_ped"]) for row in rows]),
            "relative_forgetting": (
                aggregate_replicates(valid_forgetting) if valid_forgetting else None
            ),
            "raw_forgetting": aggregate_replicates(
                [float(row["raw_forgetting"]) for row in rows]
            ),
        }
    associations = {}
    for method in config["methods"]:
        rows = [
            row
            for row in cells
            if row["method"] == method and row["relative_forgetting"] is not None
        ]
        rho, pvalue = spearmanr(
            [float(row["relative_forgetting"]) for row in rows],
            [float(row["mean_subject_ped"]) for row in rows],
        )
        associations[method] = {
            "n_transition_cells": len(rows),
            "spearman_rho": float(rho),
            "pvalue_exploratory": float(pvalue),
        }
    paired = {}
    for method in ("ewc", "derpp"):
        rows = []
        for key, sequential in by_key.items():
            if key[0] != "sequential_finetuning":
                continue
            comparison_key = (method, *key[1:])
            comparison = by_key[comparison_key]
            rows.append(
                {
                    "order": key[1],
                    "seed": key[2],
                    "direction": key[3],
                    "ped_delta_vs_sequential": float(comparison["mean_subject_ped"])
                    - float(sequential["mean_subject_ped"]),
                    "raw_forgetting_delta_vs_sequential": float(comparison["raw_forgetting"])
                    - float(sequential["raw_forgetting"]),
                }
            )
        paired[method] = {
            "n": len(rows),
            "ped_reduced_fraction": float(
                np.mean([row["ped_delta_vs_sequential"] < 0 for row in rows])
            ),
            "mean_ped_delta_vs_sequential": float(
                np.mean([row["ped_delta_vs_sequential"] for row in rows])
            ),
            "mean_raw_forgetting_delta_vs_sequential": float(
                np.mean([row["raw_forgetting_delta_vs_sequential"] for row in rows])
            ),
            "rows": rows,
        }
    summary = {
        "schema_version": 1,
        "run": config["id"],
        "config_sha256": config_sha,
        "input_result_sha256": input_sha,
        "transition_cells": len(cells),
        "aggregates": aggregates,
        "performance_ped_association": associations,
        "paired_method_comparison": paired,
        "cells": cells,
        "warning": "transition-cell associations are exploratory; subject rows are repeated measurements",
    }
    output = args.output or output_root / "summary.json"
    if output.exists():
        raise DatasetProtocolError(f"refusing to overwrite PED summary {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(output)
    print(json.dumps({key: summary[key] for key in ("transition_cells", "aggregates", "performance_ped_association", "paired_method_comparison")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
