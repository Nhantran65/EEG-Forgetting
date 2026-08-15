#!/usr/bin/env python3
"""Summarize one locked three-seed task attribution gate."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

from eeg_forgetting.data.contracts import DatasetProtocolError, load_yaml
from eeg_forgetting.data.manifests import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def task_gate_decision(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        raise DatasetProtocolError("task gate needs at least one seed")
    minimum = len(rows) // 2 + 1
    fidelity = sum(bool(row["fidelity_passed"]) for row in rows)
    drift = sum(bool(row["drift_above_null_passed"]) for row in rows)
    ig = sum(bool(row["integrated_gradients_all_positive"]) for row in rows)
    return {
        "seeds": len(rows),
        "minimum_hard_gate_passes": minimum,
        "fidelity_passes": fidelity,
        "drift_above_null_passes": drift,
        "ig_positive_seeds": ig,
        "require_ig_positive_all_seeds": True,
        "scale_allowed": bool(fidelity >= minimum and drift >= minimum and ig == len(rows)),
    }


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if config.get("status") != "locked_xai_replication":
        raise DatasetProtocolError("task attribution config is not locked")
    config_sha = sha256_file(args.config)
    root = _resolve(str(config["output"]["root"]))
    output = root / "summary.json"
    if output.exists():
        raise DatasetProtocolError(f"refusing to overwrite gate summary {output}")
    rows, input_hashes = [], {}
    for seed_value in config["seeds"]:
        seed = int(seed_value)
        path = root / f"seed-{seed}" / "result.json"
        if not path.is_file():
            raise DatasetProtocolError(f"missing task gate result {path}")
        with path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        if (
            result.get("config_sha256") != config_sha
            or result.get("task") != config["task"]
            or result.get("transition") != config["transition"]
            or int(result.get("seed", -1)) != seed
            or result.get("cache", {}).get("test_was_loaded") is not False
        ):
            raise DatasetProtocolError(f"task gate identity mismatch: {path}")
        for role in ("before", "after", "joint"):
            for field in (
                "cell_predictions",
                "fidelity_predictions",
                "integrated_gradients",
            ):
                artifact = result["roles"][role][field]
                artifact_path = path.parent / str(artifact["file"])
                if sha256_file(artifact_path) != artifact["sha256"]:
                    raise DatasetProtocolError(f"gate artifact digest mismatch: {artifact_path}")
        null = result["ped"]["null"]
        null_path = path.parent / str(null["file"])
        if sha256_file(null_path) != null["sha256"]:
            raise DatasetProtocolError(f"gate null digest mismatch: {null_path}")
        row = {
            "seed": seed,
            "fidelity_passed": bool(result["gates"]["fidelity_passed"]),
            "drift_above_null_passed": bool(
                result["gates"]["drift_above_null_passed"]
            ),
            "integrated_gradients_all_positive": bool(
                result["gates"]["integrated_gradients_all_positive"]
            ),
            "ped": float(result["ped"]["before_after_mean_subject_jsd"]),
            "null_threshold": float(result["ped"]["null"]["conservative_threshold"]),
            "ig_before": float(
                result["roles"]["before"]["integrated_gradients"][
                    "rank_agreement_spearman"
                ]
            ),
            "ig_after": float(
                result["roles"]["after"]["integrated_gradients"][
                    "rank_agreement_spearman"
                ]
            ),
        }
        rows.append(row)
        input_hashes[path.relative_to(PROJECT_ROOT).as_posix()] = sha256_file(path)
    decision = task_gate_decision(rows)
    summary = {
        "schema_version": 1,
        "run": config["id"],
        "config_sha256": config_sha,
        "task": config["task"],
        "transition": config["transition"],
        "input_result_sha256": input_hashes,
        "rows": rows,
        "aggregates": {
            "ped_mean": statistics.mean(float(row["ped"]) for row in rows),
            "ig_before_mean": statistics.mean(float(row["ig_before"]) for row in rows),
            "ig_after_mean": statistics.mean(float(row["ig_after"]) for row in rows),
        },
        "decision": decision,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
