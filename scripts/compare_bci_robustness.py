#!/usr/bin/env python3
"""Compare BCI-involving forgetting directions across the two subject folds."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from eeg_forgetting.data.contracts import DatasetProtocolError
from eeg_forgetting.data.manifests import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--main-summary",
        type=Path,
        default=PROJECT_ROOT / "results" / "continual" / "sequential_ft_v1" / "summary.json",
    )
    parser.add_argument(
        "--robustness-summary",
        type=Path,
        default=PROJECT_ROOT / "results" / "continual" / "sequential_ft_bci_robustness_v2" / "summary.json",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _sign(value: float) -> int:
    return (value > 0) - (value < 0)


def main() -> None:
    args = parse_args()
    with args.main_summary.open(encoding="utf-8") as handle:
        main = json.load(handle)
    with args.robustness_summary.open(encoding="utf-8") as handle:
        robustness = json.load(handle)
    directions = sorted(
        key
        for key in main["directed_relative_forgetting"]
        if "bciciv2a" in key
    )
    if set(directions) != {
        key
        for key in robustness["directed_relative_forgetting"]
        if "bciciv2a" in key
    }:
        raise DatasetProtocolError("BCI robustness directions do not align with main")
    rows = []
    for direction in directions:
        main_mean = float(main["directed_relative_forgetting"][direction]["mean"])
        robust_mean = float(
            robustness["directed_relative_forgetting"][direction]["mean"]
        )
        rows.append(
            {
                "direction": direction,
                "main_mean": main_mean,
                "robustness_mean": robust_mean,
                "same_sign": _sign(main_mean) == _sign(robust_mean),
            }
        )
    same = sum(bool(row["same_sign"]) for row in rows)
    document = {
        "schema_version": 1,
        "main_summary_sha256": sha256_file(args.main_summary),
        "robustness_summary_sha256": sha256_file(args.robustness_summary),
        "bci_directions": rows,
        "same_sign_directions": same,
        "total_bci_directions": len(rows),
        "all_directions_reversed": same == 0,
        "gate_passed": same > 0,
    }
    output = args.output or args.robustness_summary.parent / "comparison-to-main.json"
    if output.exists():
        raise DatasetProtocolError(f"refusing to overwrite robustness comparison {output}")
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(output)
    print(json.dumps(document, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
