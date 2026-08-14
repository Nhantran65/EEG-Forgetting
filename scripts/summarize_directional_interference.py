#!/usr/bin/env python3
"""Verify and aggregate six directed interference diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

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
        / "analysis"
        / "directional_interference_v1.yaml",
    )
    parser.add_argument("--result-root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _correlation(left: list[float], right: list[float]) -> dict[str, float]:
    pearson = float(np.corrcoef(left, right)[0, 1])
    spearman = float(np.corrcoef(rankdata(left), rankdata(right))[0, 1])
    return {"pearson": pearson, "spearman": spearman}


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
    root = args.result_root or (
        PROJECT_ROOT / "results" / "analysis" / str(config["id"])
    )
    overlap_path = Path(str(config["fisher"]["overlap_summary_path"]))
    if sha256_file(overlap_path) != config["fisher"]["overlap_summary_sha256"]:
        raise DatasetProtocolError("directional overlap summary digest mismatch")
    with overlap_path.open(encoding="utf-8") as handle:
        overlap = json.load(handle)["cross_task_cosine"]
    inputs = {}
    directions = {}
    vectors = {
        "relative_forgetting": [],
        "symmetric_fisher_overlap": [],
        "gradient_conflict": [],
        "old_fisher_drift_concentration": [],
        "shared_fisher_drift_concentration": [],
    }
    for direction, specification in config["directions"].items():
        results = []
        for seed in config["seeds"]:
            path = root / direction / f"seed-{seed}" / "result.json"
            with path.open(encoding="utf-8") as handle:
                result = json.load(handle)
            if (
                result["config_sha256"] != config_sha
                or result["direction"] != direction
                or int(result["seed"]) != int(seed)
            ):
                raise DatasetProtocolError(f"directional result identity mismatch: {path}")
            results.append(result)
            inputs[str(path.relative_to(root))] = sha256_file(path)
        old_task = str(specification["old_task"])
        new_task = str(specification["new_task"])
        pair_key = "__".join(sorted((old_task, new_task)))
        metrics = {
            "relative_forgetting": [float(item["relative_forgetting"]) for item in results],
            "gradient_conflict": [
                float(item["gradient_geometry"]["conflict_score"]) for item in results
            ],
            "old_fisher_drift_concentration": [
                float(item["update_drift"]["old_fisher_concentration_ratio"])
                for item in results
            ],
            "shared_fisher_drift_concentration": [
                float(item["update_drift"]["shared_fisher_concentration_ratio"])
                for item in results
            ],
        }
        direction_summary = {
            "old_task": old_task,
            "new_task": new_task,
            "symmetric_fisher_overlap": float(overlap[pair_key]),
            **{name: _stats(values) for name, values in metrics.items()},
        }
        directions[direction] = direction_summary
        vectors["symmetric_fisher_overlap"].append(float(overlap[pair_key]))
        for name, values in metrics.items():
            vectors[name].append(statistics.mean(values))
    target = vectors["relative_forgetting"]
    correlations = {
        name: _correlation(values, target)
        for name, values in vectors.items()
        if name != "relative_forgetting"
    }
    summary = {
        "schema_version": 1,
        "run": config["id"],
        "config_sha256": config_sha,
        "input_result_sha256": inputs,
        "directions": directions,
        "correlation_with_direction_mean_relative_forgetting": correlations,
        "interpretation_guardrail": "descriptive_only_six_directed_transitions",
    }
    output = args.output or root / "summary.json"
    if output.exists():
        raise DatasetProtocolError(f"refusing to overwrite directional summary {output}")
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
