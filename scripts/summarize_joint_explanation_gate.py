#!/usr/bin/env python3
"""Verify the strict all-six joint-reference explanation gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from eeg_forgetting.data.contracts import DatasetProtocolError, load_yaml
from eeg_forgetting.data.manifests import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs" / "xai" / "joint_high_gamma_alignment_v1.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    config_sha = sha256_file(args.config)
    root = PROJECT_ROOT / str(config["output"]["root"])
    rows, hashes = [], {}
    for task in config["tasks"]:
        for seed in config["joint"]["seeds"]:
            path = root / task / f"seed-{seed}" / "result.json"
            with path.open(encoding="utf-8") as handle:
                result = json.load(handle)
            if result["config_sha256"] != config_sha or result["task"] != task or int(result["seed"]) != int(seed):
                raise DatasetProtocolError(f"joint gate identity mismatch: {path}")
            for artifact in result["artifacts"].values():
                if artifact is None:
                    continue
                artifact_path = path.parent / artifact["file"]
                if sha256_file(artifact_path) != artifact["sha256"]:
                    raise DatasetProtocolError(f"joint gate artifact mismatch: {artifact_path}")
            rows.append({
                "task": task,
                "seed": int(seed),
                "reliability": result["reliability"],
                "fidelity": result["fidelity"],
                "passed": bool(result["gate_passed"]),
            })
            hashes[f"{task}:seed-{seed}"] = sha256_file(path)
    passed = len(rows) == 6 and all(row["passed"] for row in rows)
    summary = {
        "schema_version": 1,
        "run": config["id"],
        "config_sha256": config_sha,
        "input_result_sha256": hashes,
        "rows": rows,
        "strict_all_six_passed": passed,
        "action": "compute_delta_alignment" if passed else "discard_joint_hypothesis",
    }
    output = root / "gate-summary.json"
    if output.exists():
        raise DatasetProtocolError(f"refusing to overwrite joint gate summary {output}")
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
