#!/usr/bin/env python3
"""Build a concise human-readable report for the exploratory seed-matched study."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from eeg_forgetting.data.contracts import DatasetProtocolError
from eeg_forgetting.data.manifests import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "results"
        / "xai"
        / "joint_high_gamma_seedmatched_exploratory_v1"
        / "report.md",
    )
    return parser.parse_args()


def _load(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise DatasetProtocolError(f"missing report input {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise DatasetProtocolError(f"refusing to overwrite report {args.output}")
    performance_paths = {
        method: PROJECT_ROOT
        / "results"
        / "continual"
        / f"{method}_high_gamma_exploratory_v1"
        / "summary.json"
        for method in ("sequential_ft", "ewc", "derpp")
    }
    performance = {method: _load(path) for method, path in performance_paths.items()}
    ped_path = (
        PROJECT_ROOT
        / "results"
        / "xai"
        / "high_gamma_replacement_ped_exploratory_v1"
        / "summary.json"
    )
    gate_path = (
        PROJECT_ROOT
        / "results"
        / "xai"
        / "joint_high_gamma_exploratory_gate_v1"
        / "gate-summary.json"
    )
    alignment_path = args.output.parent / "alignment-summary.json"
    ped = _load(ped_path)
    gate = _load(gate_path)
    alignment = _load(alignment_path)
    lines = [
        "# Exploratory seed-matched continual/joint report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "> Post-gate exploratory sensitivity only. These seeds do not overwrite the official matrix or manuscript.",
        "",
        "## Execution and gates",
        "",
        "- Continual seeds: `7, 123, 999`; 27/27 runs complete.",
        f"- Joint BCI/High-Gamma maps: strict 6/6 pass = `{gate['strict_all_six_passed']}`.",
        f"- PED transition cells: `{ped['transition_cells']}`.",
        f"- Joint alignment hypothesis pass: `{alignment['hypothesis_passed']}`.",
        f"- Action: `{alignment['action']}`.",
        "",
        "## Performance forgetting",
        "",
        "| Direction | Sequential | EWC | DER++ |",
        "|---|---:|---:|---:|",
    ]
    directions = sorted(performance["sequential_ft"]["directed_relative_forgetting"])
    for direction in directions:
        values = [
            performance[method]["directed_relative_forgetting"][direction]["mean"]
            for method in ("sequential_ft", "ewc", "derpp")
        ]
        lines.append(f"| {direction} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} |")
    lines.extend(["", "## Noise-corrected PED", ""])
    for method in ("sequential_finetuning", "ewc", "derpp"):
        direction_values = ped["aggregates"][method]
        rendered = ", ".join(
            f"{direction}={row['mean_subject_ped']['mean']:.4f}"
            for direction, row in sorted(direction_values.items())
        )
        lines.append(f"- **{method}**: {rendered}")
    lines.extend(["", "## Seed-matched joint alignment", ""])
    for method, result in alignment["paired_vs_sequential"].items():
        lines.append(
            f"- **{method} − Sequential**: mean ΔA difference `{result['mean_difference']:.4f}`, "
            f"95% block-bootstrap CI `[{result['bootstrap_95_ci'][0]:.4f}, {result['bootstrap_95_ci'][1]:.4f}]`, "
            f"wins `{result['wins_out_of_9']}/9`, sign-test `p={result['two_sided_sign_pvalue']:.4f}`, "
            f"gate pass `{result['passed']}`."
        )
    lines.extend(["", "## Provenance", ""])
    for label, path in {
        **{f"performance-{k}": v for k, v in performance_paths.items()},
        "ped": ped_path,
        "joint-gate": gate_path,
        "alignment": alignment_path,
    }.items():
        lines.append(f"- `{label}`: `{path.relative_to(PROJECT_ROOT)}` — SHA-256 `{sha256_file(path)}`")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
