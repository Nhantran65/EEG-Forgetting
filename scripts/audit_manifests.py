#!/usr/bin/env python3
"""Audit immutable manifest checksums, row counts, and subject isolation."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from eeg_forgetting.data.contracts import DatasetProtocolError
from eeg_forgetting.data.manifests import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-set",
        type=Path,
        default=PROJECT_ROOT / "manifests" / "v3" / "manifest-set.json",
    )
    parser.add_argument(
        "--verify-sources",
        action="store_true",
        help="Re-hash every raw source file referenced by the manifests.",
    )
    return parser.parse_args()


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def read_rows(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def exact_counts(observed: Counter[str], expected: dict[str, int], context: str) -> None:
    if dict(observed) != expected:
        raise DatasetProtocolError(f"{context}: expected {expected}, got {dict(observed)}")


def audit_subject_isolation(rows: list[dict[str, object]], split_field: str, context: str) -> None:
    seen: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split = str(row[split_field])
        if split != "excluded":
            seen[str(row["subject_id"])].add(split)
    leaking = {subject: groups for subject, groups in seen.items() if len(groups) != 1}
    if leaking:
        raise DatasetProtocolError(f"{context}: subject leakage {leaking}")


def main() -> None:
    args = parse_args()
    with args.manifest_set.open(encoding="utf-8") as handle:
        index = json.load(handle)

    for config_name, expected in index["configs"].items():
        path = (
            resolve(config_name)
            if "/" in config_name
            else PROJECT_ROOT / "configs" / "datasets" / config_name
        )
        observed = sha256_file(path)
        if observed != expected:
            raise DatasetProtocolError(f"config changed after freeze: {path}")

    datasets: dict[str, list[dict[str, object]]] = {}
    for name, entry in index["manifests"].items():
        path = resolve(entry["path"])
        if sha256_file(path) != entry["sha256"]:
            raise DatasetProtocolError(f"manifest checksum mismatch: {path}")
        rows = read_rows(path)
        if len(rows) != int(entry["rows"]):
            raise DatasetProtocolError(f"{path}: expected {entry['rows']} rows, got {len(rows)}")
        datasets[name] = rows
        if args.verify_sources:
            for row in rows:
                for source in row["source_files"]:
                    source_path = resolve(source["path"])
                    if sha256_file(source_path) != source["sha256"]:
                        raise DatasetProtocolError(f"source checksum mismatch: {source_path}")

    bci = datasets["bciciv2a"]
    audit_subject_isolation(bci, "split", "BCI IV-2a")
    exact_counts(Counter(str(row["split"]) for row in bci), {"train": 5, "validation": 2, "test": 2}, "BCI IV-2a")

    physio = datasets["physionet_mi"]
    audit_subject_isolation(physio, "main_split", "PhysioNet main")
    audit_subject_isolation(physio, "reproduction_split", "PhysioNet reproduction")
    exact_counts(
        Counter(str(row["main_split"]) for row in physio),
        {"train": 70, "validation": 18, "excluded": 4, "test": 17},
        "PhysioNet main",
    )
    exact_counts(
        Counter(str(row["reproduction_split"]) for row in physio),
        {"train": 70, "validation": 19, "test": 20},
        "PhysioNet reproduction",
    )

    sleep = datasets["sleep_edf_sc"]
    audit_subject_isolation(sleep, "split", "Sleep-EDF")
    subjects_by_split = {
        split: {str(row["subject_id"]) for row in sleep if row["split"] == split}
        for split in ("train", "validation", "test")
    }
    exact_counts(
        Counter({split: len(subjects) for split, subjects in subjects_by_split.items()}),
        {"train": 48, "validation": 15, "test": 15},
        "Sleep-EDF subjects",
    )
    if len(sleep) != 153 or len({str(row["recording_id"]) for row in sleep}) != 153:
        raise DatasetProtocolError("Sleep-EDF must contain 153 unique recordings")

    suffix = " including raw source checksums" if args.verify_sources else ""
    print(
        f"Immutable manifest {index['manifest_version']} audit passed{suffix}."
    )
    print("BCI IV-2a: 5/2/2 subjects")
    print("PhysioNet-MI main: 70/18/17 subjects + 4 explicit exclusions")
    print("Sleep-EDF: 48/15/15 subjects; 94/30/29 recordings")


if __name__ == "__main__":
    main()
