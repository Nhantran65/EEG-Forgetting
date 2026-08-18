#!/usr/bin/env python3
"""Compute seed-matched offline-reference alignment change after strict joint gates."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from eeg_forgetting.data.contracts import DatasetProtocolError, load_yaml
from eeg_forgetting.data.manifests import sha256_file
from eeg_forgetting.xai.explanation_drift import noise_corrected_symmetric_jsd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs" / "xai" / "joint_high_gamma_alignment_v1.yaml",
    )
    return parser.parse_args()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_map(path: Path, expected_sha: str) -> dict[str, object]:
    if sha256_file(path) != expected_sha:
        raise DatasetProtocolError(f"alignment map digest mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {
            "subject_ids": [str(v) for v in archive["subject_ids"]],
            "fit_subject_maps": np.asarray(archive["fit_subject_maps"], dtype=np.float64),
            "gate_subject_maps": np.asarray(archive["gate_subject_maps"], dtype=np.float64),
            "fit_group_map": np.asarray(archive["fit_group_map"], dtype=np.float64),
            "gate_group_map": np.asarray(archive["gate_group_map"], dtype=np.float64),
        }


def _alignment(left: dict[str, object], joint: dict[str, object]) -> dict[str, object]:
    if left["subject_ids"] != joint["subject_ids"]:
        raise DatasetProtocolError("alignment subjects differ")
    by_subject = {}
    for index, subject in enumerate(left["subject_ids"]):
        by_subject[subject] = noise_corrected_symmetric_jsd(
            left["fit_subject_maps"][index],
            left["gate_subject_maps"][index],
            joint["fit_subject_maps"][index],
            joint["gate_subject_maps"][index],
        )
    group = noise_corrected_symmetric_jsd(
        left["fit_group_map"], left["gate_group_map"],
        joint["fit_group_map"], joint["gate_group_map"],
    )
    return {"subjects": by_subject, "group": group}


def _bootstrap_ci(values: np.ndarray, *, seed: int, replicates: int) -> list[float]:
    generator = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        means[index] = generator.choice(values, size=len(values), replace=True).mean()
    return [float(v) for v in np.percentile(means, [2.5, 97.5])]


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    config_sha = sha256_file(args.config)
    output_root = _resolve(str(config["output"]["root"]))
    gate_authority = config.get("gate_authority")
    gate_result_root = (
        _resolve(str(gate_authority["result_root"]))
        if gate_authority is not None
        else output_root
    )
    gate_summary_path = (
        _resolve(str(gate_authority["summary"]))
        if gate_authority is not None
        else output_root / "gate-summary.json"
    )
    with gate_summary_path.open(encoding="utf-8") as handle:
        gate_summary = json.load(handle)
    if (
        gate_summary["config_sha256"]
        != (
            str(gate_authority["config_sha256"])
            if gate_authority is not None
            else config_sha
        )
        or gate_summary["strict_all_six_passed"] is not True
    ):
        raise DatasetProtocolError("strict joint gate did not pass")
    ped = config["ped_authority"]
    ped_config_path = _resolve(str(ped["config"]))
    ped_summary_path = _resolve(str(ped["summary"]))
    if sha256_file(ped_config_path) != ped["config_sha256"]:
        raise DatasetProtocolError("PED authority digest mismatch")
    ped_config = load_yaml(ped_config_path)
    with ped_summary_path.open(encoding="utf-8") as handle:
        ped_summary = json.load(handle)
    if ped_summary.get("config_sha256") != ped["config_sha256"]:
        raise DatasetProtocolError("PED summary is not bound to its config")
    if ped.get("summary_sha256") is not None and sha256_file(ped_summary_path) != ped["summary_sha256"]:
        raise DatasetProtocolError("PED summary digest mismatch")
    joint_maps = {}
    for task in config["tasks"]:
        for seed in config["joint"]["seeds"]:
            result_path = gate_result_root / task / f"seed-{seed}" / "result.json"
            with result_path.open(encoding="utf-8") as handle:
                result = json.load(handle)
            artifact = result["artifacts"]["map"]
            joint_maps[(task, int(seed))] = _load_map(
                result_path.parent / artifact["file"], artifact["sha256"]
            )
    cells = []
    block_values: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for method, specification in ped_config["methods"].items():
        for order in load_yaml(_resolve(str(specification["training_config"]))) ["orders"]:
            for seed in config["joint"]["seeds"]:
                ped_result_path = (
                    _resolve(str(ped_config["output"]["root"]))
                    / method / order / f"seed-{seed}" / "result.json"
                )
                with ped_result_path.open(encoding="utf-8") as handle:
                    ped_result = json.load(handle)
                for transition in ped_result["transitions"]:
                    old_task = transition["old_task"]
                    if old_task not in config["tasks"]:
                        continue
                    before_key = f"stage-{int(transition['before_stage']):02d}-{old_task}"
                    after_key = f"stage-{int(transition['after_stage']):02d}-{old_task}"
                    before_artifact = ped_result["artifacts"][before_key]
                    after_artifact = ped_result["artifacts"][after_key]
                    before = _load_map(
                        ped_result_path.parent / before_artifact["file"], before_artifact["sha256"]
                    )
                    after = _load_map(
                        ped_result_path.parent / after_artifact["file"], after_artifact["sha256"]
                    )
                    joint = joint_maps[(old_task, int(seed))]
                    before_alignment = _alignment(before, joint)
                    after_alignment = _alignment(after, joint)
                    subject_delta = {
                        subject: float(
                            after_alignment["subjects"][subject]["noise_corrected_ped"]
                            - before_alignment["subjects"][subject]["noise_corrected_ped"]
                        )
                        for subject in before_alignment["subjects"]
                    }
                    mean_delta = float(np.mean(list(subject_delta.values())))
                    cell = {
                        "method": method,
                        "order": order,
                        "seed": int(seed),
                        "direction": f"{old_task}<-{transition['learned_task']}",
                        "old_task": old_task,
                        "learned_task": transition["learned_task"],
                        "mean_subject_delta_A": mean_delta,
                        "group_delta_A": float(
                            after_alignment["group"]["noise_corrected_ped"]
                            - before_alignment["group"]["noise_corrected_ped"]
                        ),
                        "subject_delta_A": subject_delta,
                    }
                    cells.append(cell)
                    block_values[(method, order, int(seed))].append(mean_delta)
    if len(cells) != int(config["alignment"]["expected_transition_cells"]):
        raise DatasetProtocolError("joint alignment transition count changed")
    block_means = {
        key: float(np.mean(values)) for key, values in block_values.items()
    }
    paired = {}
    success = False
    for method in ("ewc", "derpp"):
        differences = []
        rows = []
        for order in ("forward", "reverse", "challenging"):
            for seed in config["joint"]["seeds"]:
                candidate = block_means[(method, order, int(seed))]
                sequential = block_means[("sequential_finetuning", order, int(seed))]
                difference = candidate - sequential
                differences.append(difference)
                rows.append({"order": order, "seed": int(seed), "candidate_minus_sequential_delta_A": difference})
        values = np.asarray(differences, dtype=np.float64)
        ci = _bootstrap_ci(
            values,
            seed=int(config["alignment"]["bootstrap_seed"]),
            replicates=int(config["alignment"]["bootstrap_replicates"]),
        )
        wins = int(np.sum(values < 0))
        passed = bool(values.mean() < 0 and ci[1] < 0 and wins >= 8)
        success = success or passed
        paired[method] = {
            "mean_difference": float(values.mean()),
            "bootstrap_95_ci": ci,
            "wins_out_of_9": wins,
            "two_sided_sign_pvalue": float(binomtest(wins, 9, 0.5).pvalue),
            "passed": passed,
            "rows": rows,
        }
    summary = {
        "schema_version": 1,
        "run": config["id"],
        "config_sha256": config_sha,
        "gate_summary_sha256": sha256_file(gate_summary_path),
        "transition_cells": len(cells),
        "block_means": {f"{m}:{o}:seed-{s}": v for (m, o, s), v in block_means.items()},
        "paired_vs_sequential": paired,
        "hypothesis_passed": success,
        "action": "eligible_for_manuscript_decision" if success else "discard_joint_hypothesis",
        "cells": cells,
    }
    output = output_root / "alignment-summary.json"
    if output.exists():
        raise DatasetProtocolError(f"refusing to overwrite joint alignment {output}")
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(output)
    print(json.dumps({"paired_vs_sequential": paired, "hypothesis_passed": success, "action": summary["action"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
