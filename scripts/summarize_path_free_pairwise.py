#!/usr/bin/env python3
"""Build the six-direction path-free sequential forgetting matrix."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from scipy.stats import spearmanr

from eeg_forgetting.data.contracts import DatasetProtocolError, load_yaml
from eeg_forgetting.data.manifests import sha256_file
from eeg_forgetting.training.continual import CANONICAL_TASKS, aggregate_replicates


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "training" / "sequential_pairwise_clean_v1.yaml",
    )
    parser.add_argument(
        "--main-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "training" / "sequential_ft_v1.yaml",
    )
    parser.add_argument(
        "--fisher-summary",
        type=Path,
        default=PROJECT_ROOT / "results" / "fisher" / "instrument_v1" / "summary.json",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _load_result(path: Path, config_sha: str) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    if result["config_sha256"] != config_sha:
        raise DatasetProtocolError(f"pairwise source config mismatch: {path}")
    for stage in result["stages"]:
        stage_path = path.parent / f"stage-{stage['stage']:02d}-{stage['learned_task']}" / "stage.json"
        if sha256_file(stage_path) != stage["stage_result_sha256"]:
            raise DatasetProtocolError(f"pairwise stage digest mismatch: {stage_path}")
    return result


def _first_transition(result: dict[str, object]) -> dict[str, object]:
    rows = [row for row in result["pairwise_forgetting"] if int(row["after_stage"]) == 2]
    if len(rows) != 1:
        raise DatasetProtocolError("path-free source needs exactly one first transition")
    return rows[0]


def _pair_key(left: str, right: str) -> str:
    ordered = sorted((left, right), key=CANONICAL_TASKS.index)
    return "__".join(ordered)


def main() -> None:
    args = parse_args()
    pair_config = load_yaml(args.pair_config)
    main_config = load_yaml(args.main_config)
    pair_sha = sha256_file(args.pair_config)
    main_sha = sha256_file(args.main_config)
    pair_root = PROJECT_ROOT / "results" / "continual" / str(pair_config["id"])
    main_root = PROJECT_ROOT / "results" / "continual" / str(main_config["id"])
    sources = {
        "bciciv2a->physionet_mi": (main_root, "forward", main_sha),
        "sleep_edf_sc->physionet_mi": (main_root, "reverse", main_sha),
        "physionet_mi->bciciv2a": (main_root, "challenging", main_sha),
        "bciciv2a->sleep_edf_sc": (pair_root, "bci_to_sleep", pair_sha),
        "sleep_edf_sc->bciciv2a": (pair_root, "sleep_to_bci", pair_sha),
        "physionet_mi->sleep_edf_sc": (pair_root, "physionet_to_sleep", pair_sha),
    }
    directed: dict[str, list[float]] = defaultdict(list)
    inputs = {}
    for expected_direction, (root, order_name, config_sha) in sources.items():
        old, learned = expected_direction.split("->")
        for seed in pair_config["seeds"]:
            path = root / order_name / f"seed-{seed}" / "result.json"
            result = _load_result(path, config_sha)
            row = _first_transition(result)
            if row["old_task"] != old or row["learned_task"] != learned:
                raise DatasetProtocolError(f"unexpected path-free direction in {path}")
            if not row["relative_valid"]:
                raise DatasetProtocolError(f"invalid path-free relative forgetting in {path}")
            key = f"{old}<-{learned}"
            directed[key].append(float(row["relative_forgetting"]))
            inputs[f"{expected_direction}:seed-{seed}"] = sha256_file(path)

    directed_summary = {
        key: aggregate_replicates(values) for key, values in sorted(directed.items())
    }
    pairs: dict[str, list[float]] = defaultdict(list)
    for direction, values in directed.items():
        old, learned = direction.split("<-")
        pairs[_pair_key(old, learned)].extend(values)
    pair_summary = {
        key: aggregate_replicates(values) for key, values in sorted(pairs.items())
    }
    with args.fisher_summary.open(encoding="utf-8") as handle:
        fisher = json.load(handle)
    overlap = fisher["cross_task_cosine"]
    keys = sorted(pair_summary)
    rho, pvalue = spearmanr(
        [float(overlap[key]) for key in keys],
        [float(pair_summary[key]["mean"]) for key in keys],
    )
    summary = {
        "schema_version": 1,
        "run": pair_config["id"],
        "pair_config_sha256": pair_sha,
        "main_config_sha256": main_sha,
        "fisher_summary_sha256": sha256_file(args.fisher_summary),
        "input_result_sha256": inputs,
        "directed_relative_forgetting": directed_summary,
        "pair_relative_forgetting": pair_summary,
        "pair_fisher_overlap": {key: overlap[key] for key in keys},
        "exploratory_pair_spearman": {
            "n_independent_pairs": 3,
            "rho": float(rho),
            "pvalue": float(pvalue),
            "warning": "three_pairs_only_directions_do_not_increase_independent_n",
        },
        "design": "all_six_directions_start_from_pretrained_then_train_exactly_two_tasks",
    }
    output = args.output or pair_root / "summary.json"
    if output.exists():
        raise DatasetProtocolError(f"refusing to overwrite path-free summary {output}")
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
