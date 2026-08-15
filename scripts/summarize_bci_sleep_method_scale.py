#!/usr/bin/env python3
"""Summarize the locked BCI<-Sleep method explanation-drift scale matrix."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

from scipy.stats import spearmanr

from eeg_forgetting.data.contracts import DatasetProtocolError, load_yaml
from eeg_forgetting.data.manifests import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "xai" / "bci_sleep_method_scale_v1.yaml",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _stats(values: list[float]) -> dict[str, object]:
    if not values:
        raise DatasetProtocolError("cannot summarize an empty scale cell set")
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
        "values": values,
    }


def paired_method_differences(
    rows: dict[tuple[str, str, int], dict[str, object]], method: str
) -> dict[str, object]:
    forgetting, ped = [], []
    paired = []
    keys = sorted((order, seed) for name, order, seed in rows if name == "sequential")
    for order, seed in keys:
        baseline = rows[("sequential", order, seed)]
        candidate = rows[(method, order, seed)]
        delta_forgetting = float(candidate["relative_forgetting"]) - float(
            baseline["relative_forgetting"]
        )
        delta_ped = float(candidate["ped"]) - float(baseline["ped"])
        forgetting.append(delta_forgetting)
        ped.append(delta_ped)
        paired.append(
            {
                "order": order,
                "seed": seed,
                "candidate_minus_sequential_relative_forgetting": delta_forgetting,
                "candidate_minus_sequential_ped": delta_ped,
                "performance_and_explanation_both_better": bool(
                    delta_forgetting < 0 and delta_ped < 0
                ),
                "performance_better_explanation_worse": bool(
                    delta_forgetting < 0 and delta_ped > 0
                ),
            }
        )
    return {
        "relative_forgetting_difference": _stats(forgetting),
        "ped_difference": _stats(ped),
        "both_better_count": sum(
            bool(row["performance_and_explanation_both_better"]) for row in paired
        ),
        "performance_better_explanation_worse_count": sum(
            bool(row["performance_better_explanation_worse"]) for row in paired
        ),
        "paired_cells": paired,
    }


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if config.get("status") != "locked_xai_scale":
        raise DatasetProtocolError("scale summary config is not locked")
    config_sha = sha256_file(args.config)
    root = _resolve(str(config["output"]["root"]))
    output = args.output or root / "summary.json"
    if output.exists():
        raise DatasetProtocolError(f"refusing to overwrite scale summary {output}")
    rows: dict[tuple[str, str, int], dict[str, object]] = {}
    input_hashes = {}
    for method in config["methods"]:
        for order in config["orders"]:
            for seed_value in config["seeds"]:
                seed = int(seed_value)
                path = root / method / order / f"seed-{seed}" / "result.json"
                if not path.is_file():
                    raise DatasetProtocolError(f"missing scale result {path}")
                with path.open(encoding="utf-8") as handle:
                    result = json.load(handle)
                if (
                    result.get("config_sha256") != config_sha
                    or result.get("method") != method
                    or result.get("order") != order
                    or int(result.get("seed", -1)) != seed
                    or result.get("task") != config["task"]
                    or result.get("learned_task") != config["learned_task"]
                    or result.get("cache", {}).get("test_was_loaded") is not False
                ):
                    raise DatasetProtocolError(f"scale result identity mismatch: {path}")
                parent = _resolve(str(result["parent_result"]["path"]))
                if sha256_file(parent) != result["parent_result"]["sha256"]:
                    raise DatasetProtocolError(f"parent result digest mismatch: {parent}")
                for role in ("before", "after"):
                    for artifact in result["roles"][role]["artifacts"].values():
                        artifact_path = path.parent / str(artifact["file"])
                        if sha256_file(artifact_path) != artifact["sha256"]:
                            raise DatasetProtocolError(
                                f"scale prediction digest mismatch: {artifact_path}"
                            )
                null_artifact = result["ped"]["null_artifact"]
                null_path = path.parent / str(null_artifact["file"])
                if sha256_file(null_path) != null_artifact["sha256"]:
                    raise DatasetProtocolError(f"scale null digest mismatch: {null_path}")
                forgetting = result["performance_forgetting"]["relative_forgetting"]
                if forgetting is None:
                    raise DatasetProtocolError(f"invalid forgetting headroom: {path}")
                key = (method, order, seed)
                rows[key] = {
                    "relative_forgetting": float(forgetting),
                    "raw_forgetting": float(
                        result["performance_forgetting"]["raw_forgetting"]
                    ),
                    "ped": float(result["ped"]["mean_subject_jsd"]),
                    "ped_above_null": bool(result["ped"]["above_null"]),
                    "fidelity_before": bool(
                        result["roles"]["before"]["fidelity"]["passed"]
                    ),
                    "fidelity_after": bool(
                        result["roles"]["after"]["fidelity"]["passed"]
                    ),
                }
                input_hashes[path.relative_to(PROJECT_ROOT).as_posix()] = sha256_file(path)

    by_method = {}
    for method in config["methods"]:
        method_rows = [value for (name, _order, _seed), value in rows.items() if name == method]
        relative = [float(value["relative_forgetting"]) for value in method_rows]
        ped = [float(value["ped"]) for value in method_rows]
        correlation = float(spearmanr(relative, ped).statistic)
        by_method[method] = {
            "relative_forgetting": _stats(relative),
            "raw_forgetting": _stats([float(value["raw_forgetting"]) for value in method_rows]),
            "ped": _stats(ped),
            "descriptive_spearman_relative_forgetting_vs_ped": correlation,
            "ped_above_null_count": sum(bool(value["ped_above_null"]) for value in method_rows),
            "fidelity_both_roles_pass_count": sum(
                bool(value["fidelity_before"] and value["fidelity_after"])
                for value in method_rows
            ),
            "cells": len(method_rows),
        }
    summary = {
        "schema_version": 1,
        "run": config["id"],
        "config_sha256": config_sha,
        "scope": config["scope"],
        "input_result_sha256": input_hashes,
        "by_method": by_method,
        "paired_vs_sequential": {
            method: paired_method_differences(rows, method)
            for method in ("ewc", "derpp")
        },
        "interpretation_contract": {
            "negative_forgetting_difference": "candidate_retains_performance_better",
            "negative_ped_difference": "candidate_retains_explanation_better",
            "inference_warning": (
                "cells_are_descriptive_method_order_seed_pairs_not_independent_subjects"
            ),
        },
    }
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
