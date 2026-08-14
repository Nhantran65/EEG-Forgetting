#!/usr/bin/env python3
"""Run one locked EWC or DER++ BCI-to-Sleep validation pilot candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eeg_forgetting.data.contracts import load_yaml
from eeg_forgetting.training.continual_methods import (
    run_derpp_pair_pilot,
    run_ewc_pair_pilot,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("ewc", "derpp"), required=True)
    parser.add_argument("--candidate", type=float, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "pilots" / "continual_method_selection_v1.yaml",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "v3",
    )
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args()


def _candidate_name(method: str, candidate: float) -> str:
    if method == "ewc":
        return f"lambda-{candidate:g}"
    if not candidate.is_integer():
        raise ValueError("DER++ MiB candidate must be an integer")
    return f"cap-{int(candidate)}mib"


def main() -> None:
    args = parse_args()
    protocol = load_yaml(args.config)
    output_root = args.output_root or (
        PROJECT_ROOT / "results" / "pilots" / str(protocol["id"])
    )
    model = load_yaml(PROJECT_ROOT / "configs" / "models" / "cbramod.yaml")
    checkpoint = model["checkpoint"]
    common = {
        "config_path": args.config,
        "cache_root": args.cache_root,
        "checkpoint_path": PROJECT_ROOT / str(checkpoint["local_path"]),
        "checkpoint_sha256": str(checkpoint["sha256"]),
        "output_dir": output_root / args.method / _candidate_name(args.method, args.candidate),
        "device": args.device,
    }
    if args.method == "ewc":
        result = run_ewc_pair_pilot(strength=args.candidate, **common)
    else:
        result = run_derpp_pair_pilot(byte_cap_mib=int(args.candidate), **common)
    print(
        json.dumps(
            {
                "method": result["method"],
                "candidate": result["candidate"],
                "pairwise_forgetting": result["pairwise_forgetting"],
                "final_evaluations": result["final_evaluations"],
                "memory": result["memory"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
