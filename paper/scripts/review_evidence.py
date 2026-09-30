#!/usr/bin/env python3
"""Export and independently check the small, shareable paper evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
BUNDLE = PAPER / "review_evidence.json"
SUMMARY = ROOT / "results/xai/high_gamma_replacement_ped_v1/summary.json"
GATES = PAPER / "tables/table1_reliability.sources.json"
REPORT = PAPER / "statistical_report.json"
METHODS = ("sequential_finetuning", "ewc", "derpp")
DIRECTIONS = (
    "bciciv2a<-high_gamma",
    "bciciv2a<-sleep_edf_sc",
    "high_gamma<-bciciv2a",
    "high_gamma<-sleep_edf_sc",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual: float, expected: float, *, tolerance: float = 1e-10) -> None:
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise ValueError(f"value mismatch: {actual} != {expected}")


def positive_map(values: np.ndarray) -> np.ndarray:
    positive = np.maximum(np.asarray(values, dtype=float), 0.0)
    if not np.isfinite(positive).all() or positive.sum() <= 0:
        raise ValueError("invalid positive reliance mass")
    return positive / positive.sum()


def export() -> None:
    summary = json.loads(SUMMARY.read_text())
    gates = json.loads(GATES.read_text())
    source_hashes = {
        str(SUMMARY.relative_to(ROOT)): digest(SUMMARY),
        str(GATES.relative_to(ROOT)): digest(GATES),
    }
    cells = []
    for row in summary["cells"]:
        if row["direction"] not in DIRECTIONS:
            continue
        subjects = {
            subject: {
                "cross_checkpoint_jsd": value["cross_checkpoint_jsd"],
                "within_checkpoint_noise_jsd": value["within_checkpoint_noise_jsd"],
            }
            for subject, value in row["subjects"].items()
        }
        cells.append({
            "method": row["method"],
            "order": row["order"],
            "seed": row["seed"],
            "direction": row["direction"],
            "relative_forgetting": row["relative_forgetting"],
            "mean_subject_ped": row["mean_subject_ped"],
            "subjects": subjects,
        })

    before, after = [], []
    for seed in (3407, 42, 2026):
        maps = SUMMARY.parent / "sequential_finetuning/forward" / f"seed-{seed}" / "maps"
        pair = []
        for stage in (1, 2):
            path = maps / f"stage-{stage:02d}-bciciv2a.npz"
            source_hashes[str(path.relative_to(ROOT))] = digest(path)
            with np.load(path, allow_pickle=False) as data:
                pair.append(positive_map(0.5 * (data["fit_group_map"] + data["gate_group_map"])))
        before.append(pair[0])
        after.append(pair[1])
    bundle = {
        "schema_version": 1,
        "description": "Derived review evidence; no EEG samples, predictions, or model weights.",
        "source_sha256": source_hashes,
        "gate_source_sha256": gates["source_sha256"],
        "gate_rows": gates["rows"],
        "cells": cells,
        "bci_reliance_shift": {
            "channels": 22,
            "bands": ["delta", "theta", "alpha", "beta", "gamma"],
            "seeds": [3407, 42, 2026],
            "before_positive_mass": np.mean(before, axis=0).tolist(),
            "after_positive_mass": np.mean(after, axis=0).tolist(),
        },
    }
    BUNDLE.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    print(f"exported {BUNDLE.relative_to(ROOT)} ({BUNDLE.stat().st_size} bytes)")


def verify() -> None:
    bundle = json.loads(BUNDLE.read_text())
    report = json.loads(REPORT.read_text())
    if bundle["schema_version"] != 1 or len(bundle["cells"]) != 63:
        raise ValueError("unexpected review bundle schema or cell count")
    if bundle["source_sha256"][str(SUMMARY.relative_to(ROOT))] != (
        "2005ef2c8b62207fed1215f896ce855a537ede128f39c12c2d1fe00b3027cc43"
    ):
        raise ValueError("unexpected source summary identity")

    # Check original source files when present; a clean clone needs only this bundle.
    for name, expected in {**bundle["source_sha256"], **bundle["gate_source_sha256"]}.items():
        path = ROOT / name
        if path.is_file() and digest(path) != expected:
            raise ValueError(f"source hash mismatch: {name}")

    gates = bundle["gate_rows"]
    if len(gates) != 3:
        raise ValueError("expected three dataset gates")
    for gate in gates:
        cosines = list(gate["subject_cosines"].values())
        close(statistics.median(cosines), gate["median_cosine"])
        if sum(value >= 0.5 for value in cosines) != gate["subjects_above_threshold"]:
            raise ValueError("reliability subject count mismatch")
        reliability = (
            len(cosines) == gate["subjects"]
            and all(math.isfinite(value) for value in cosines)
            and statistics.median(cosines) >= 0.70
            and sum(value >= 0.5 for value in cosines) / len(cosines) >= 2 / 3
        )
        fidelity = gate["fidelity"]
        faithful = (
            fidelity["top_margin_drop"] > fidelity["control_margin_drop_percentile_95"]
            and fidelity["top_ba_drop"] > fidelity["control_ba_drop_percentile_95"]
        )
        if reliability != gate["reliability_passed"] or faithful != fidelity["passed"]:
            raise ValueError(f"gate mismatch: {gate['dataset']}")
        if (reliability and faithful) != gate["ped_eligible"]:
            raise ValueError(f"PED eligibility mismatch: {gate['dataset']}")

    cells = bundle["cells"]
    keyed = {(c["method"], c["order"], c["seed"], c["direction"]): c for c in cells}
    if len(keyed) != 63:
        raise ValueError("duplicate transition cell")
    for cell in cells:
        if cell["method"] not in METHODS or cell["direction"] not in DIRECTIONS:
            raise ValueError("unexpected method or direction")
        if cell["relative_forgetting"] is None:
            raise ValueError("missing relative forgetting in reported PED cell")
        ped = statistics.mean(
            subject["cross_checkpoint_jsd"] - subject["within_checkpoint_noise_jsd"]
            for subject in cell["subjects"].values()
        )
        close(ped, cell["mean_subject_ped"])

    for method in METHODS:
        rows = [cell for cell in cells if cell["method"] == method]
        if len(rows) != 21:
            raise ValueError(f"unexpected cell count: {method}")
        rho = float(spearmanr(
            [row["relative_forgetting"] for row in rows],
            [row["mean_subject_ped"] for row in rows],
        ).statistic)
        close(rho, report["associations"][method]["spearman_rho"])
        for direction in DIRECTIONS:
            selected = [row for row in rows if row["direction"] == direction]
            expected_n = 3 if direction == DIRECTIONS[0] else 6
            if len(selected) != expected_n:
                raise ValueError(f"unexpected n: {method}, {direction}")
            for metric in ("relative_forgetting", "mean_subject_ped"):
                values = [row[metric] for row in selected]
                close(statistics.mean(values), _TABLE_VALUES[direction][method][metric][0])
                close(statistics.stdev(values), _TABLE_VALUES[direction][method][metric][1])

    for method in ("ewc", "derpp"):
        grouped = defaultdict(list)
        deltas = []
        for cell in cells:
            if cell["method"] != method:
                continue
            baseline = keyed[("sequential_finetuning", cell["order"], cell["seed"], cell["direction"])]
            delta = cell["mean_subject_ped"] - baseline["mean_subject_ped"]
            deltas.append(delta)
            grouped[(cell["order"], cell["seed"])].append(delta)
        block_means = np.asarray([statistics.mean(values) for values in grouped.values()])
        stats = report["methods"][method]
        if len(deltas) != 21 or len(grouped) != 9 or sum(x < 0 for x in deltas) != 21:
            raise ValueError(f"paired coverage mismatch: {method}")
        close(float(block_means.mean()), stats["mean_block_ped_delta"])
        if int(np.sum(block_means < 0)) != stats["negative_blocks"]:
            raise ValueError(f"block sign count mismatch: {method}")
        # An exact two-sided sign test for nine negative blocks.
        close(2 / 2**len(block_means), stats["two_sided_sign_test_p"])
        draws = np.random.default_rng(20260818).choice(
            block_means, size=(20000, len(block_means)), replace=True
        ).mean(axis=1)
        ci = np.quantile(draws, [0.025, 0.975])
        for actual, expected in zip(ci, stats["bootstrap_95_ci"], strict=True):
            close(float(actual), expected)

    shift = bundle["bci_reliance_shift"]
    if shift["channels"] != 22 or len(shift["bands"]) != 5:
        raise ValueError("wrong reliance map shape")
    for phase in ("before_positive_mass", "after_positive_mass"):
        values = shift[phase]
        if len(values) != 110 or min(values) < 0:
            raise ValueError(f"invalid reliance map: {phase}")
        close(sum(values), 1.0)
    print("review evidence verified: 3 gates, 63 PED cells, 21 pairs/method, 9 blocks/method")


# Independent expected Table II values from the supplied manuscript, in order:
# F_rel mean/SD, PED mean/SD. Precise values are sourced from the audited table.
_TABLE_VALUES = {
    "bciciv2a<-high_gamma": {
        "sequential_finetuning": {"relative_forgetting": (-0.25476682242646886, 0.10022172737448114), "mean_subject_ped": (0.24307010496093406, 0.06663055228732558)},
        "ewc": {"relative_forgetting": (-0.11264188199860259, 0.062097948617116566), "mean_subject_ped": (0.014618368238543324, 0.0043107039148492215)},
        "derpp": {"relative_forgetting": (-0.06950857706115647, 0.1013445904660807), "mean_subject_ped": (0.05160729385195118, 0.012336628932671517)},
    },
    "bciciv2a<-sleep_edf_sc": {
        "sequential_finetuning": {"relative_forgetting": (0.06702912413839066, 0.17113448341118806), "mean_subject_ped": (0.1400150273291981, 0.024118540399117285)},
        "ewc": {"relative_forgetting": (0.06422247411477189, 0.09310062186181668), "mean_subject_ped": (0.016723750195646633, 0.004995030659858678)},
        "derpp": {"relative_forgetting": (-0.05324476613902643, 0.10753755502052972), "mean_subject_ped": (0.06866293926685486, 0.017557837671653855)},
    },
    "high_gamma<-bciciv2a": {
        "sequential_finetuning": {"relative_forgetting": (0.15633882880006614, 0.03626906051490601), "mean_subject_ped": (0.08245406491457721, 0.020306873592049513)},
        "ewc": {"relative_forgetting": (-0.009100388400914995, 0.0063123860379999805), "mean_subject_ped": (0.0003225354681449559, 0.00022681035793395535)},
        "derpp": {"relative_forgetting": (0.035759076971673735, 0.051116290840260876), "mean_subject_ped": (0.03329401871857754, 0.005779447516938902)},
    },
    "high_gamma<-sleep_edf_sc": {
        "sequential_finetuning": {"relative_forgetting": (0.15277052605433533, 0.04915233307786831), "mean_subject_ped": (0.1691349607151305, 0.04200601186944576)},
        "ewc": {"relative_forgetting": (0.004430104416509104, 0.021353902698386378), "mean_subject_ped": (0.00965750511476274, 0.0038678433488778955)},
        "derpp": {"relative_forgetting": (0.0029457539450711834, 0.020676775749066345), "mean_subject_ped": (0.06270601996166889, 0.015968385336324154)},
    },
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", action="store_true", help="regenerate from local ignored results")
    args = parser.parse_args()
    if args.export:
        export()
    verify()
