#!/usr/bin/env python3
"""Apply the predeclared depth-selection rule to completed pilot results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eeg_forgetting.data.contracts import DatasetProtocolError
from eeg_forgetting.training.pilot import TASK_CLASSES


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "pilots" / "cbramod_depth_v3",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.results_root
    depths = (0, 1, 2, 4, 8)
    rows = []
    for depth in depths:
        normalized = []
        task_metrics = {}
        for dataset, classes in TASK_CLASSES.items():
            paths = list((root / dataset / f"depth-{depth}").glob("seed-*/result.json"))
            if len(paths) != 1:
                raise DatasetProtocolError(
                    f"expected one result for {dataset}/depth-{depth}, got {paths}"
                )
            with paths[0].open(encoding="utf-8") as handle:
                result = json.load(handle)
            metric = float(result["best_mean_subject_balanced_accuracy"])
            chance = 1.0 / classes
            normalized_metric = (metric - chance) / (1.0 - chance)
            normalized.append(normalized_metric)
            task_metrics[dataset] = {
                "best_subject_ba": metric,
                "normalized": normalized_metric,
                "best_step": result["best_step"],
            }
        rows.append(
            {
                "depth": depth,
                "mean_normalized_validation_ba": sum(normalized) / len(normalized),
                "tasks": task_metrics,
            }
        )
    best = max(float(row["mean_normalized_validation_ba"]) for row in rows)
    candidates = {1, 2, 4, 8}
    candidate_best = max(
        float(row["mean_normalized_validation_ba"])
        for row in rows
        if int(row["depth"]) in candidates
    )
    threshold = 0.95 * candidate_best
    selected = min(
        int(row["depth"])
        for row in rows
        if int(row["depth"]) in candidates
        and float(row["mean_normalized_validation_ba"]) >= threshold
    )
    summary = {
        "rule": "smallest depth at least 95% of best mean normalized validation BA",
        "best_all_depths_including_frozen_control": best,
        "best_candidate_depth": candidate_best,
        "threshold": threshold,
        "selected_depth": selected,
        "rows": rows,
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
