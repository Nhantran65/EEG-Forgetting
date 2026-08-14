#!/usr/bin/env python3
"""Verify and select locked EWC/DER++ validation pilot candidates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from eeg_forgetting.data.contracts import DatasetProtocolError, load_yaml
from eeg_forgetting.data.manifests import sha256_file
from eeg_forgetting.training.continual_methods import select_method_candidate


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "pilots" / "continual_method_selection_v1.yaml",
    )
    parser.add_argument("--result-root", type=Path)
    return parser.parse_args()


def _path(method: str, candidate: float, root: Path) -> Path:
    name = f"lambda-{candidate:g}" if method == "ewc" else f"cap-{int(candidate)}mib"
    return root / method / name / "result.json"


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    root = args.result_root or (
        PROJECT_ROOT / "results" / "pilots" / str(config["id"])
    )
    config_sha = sha256_file(args.config)
    all_results = {}
    inputs = {}
    declarations = {
        "ewc": [float(value) for value in config["ewc"]["strength_candidates"]],
        "derpp": [int(value) for value in config["derpp"]["byte_cap_candidates_mib"]],
    }
    for method, candidates in declarations.items():
        documents = []
        for candidate in candidates:
            path = _path(method, candidate, root)
            if not path.is_file():
                raise DatasetProtocolError(f"missing {method} pilot result {path}")
            with path.open(encoding="utf-8") as handle:
                document = json.load(handle)
            if document["config_sha256"] != config_sha or document["method"] != method:
                raise DatasetProtocolError(f"pilot result metadata mismatch: {path}")
            documents.append(document)
            inputs[f"{method}:{candidate}"] = sha256_file(path)
        all_results[method] = select_method_candidate(
            documents,
            maximum_new_task_drop=float(
                config["selection"]["maximum_absolute_new_task_validation_drop"]
            ),
        )
    output = root / "summary.json"
    if output.exists():
        raise DatasetProtocolError(f"refusing to overwrite method pilot summary {output}")
    document = {
        "schema_version": 1,
        "pilot": config["id"],
        "config_sha256": config_sha,
        "input_result_sha256": inputs,
        "selection": all_results,
    }
    temporary = output.with_suffix(".json.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(output)
    print(json.dumps(document, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
