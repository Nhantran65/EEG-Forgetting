#!/usr/bin/env python3
"""Create the immutable v1 subject/split manifests from audited raw data."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import xlrd

from eeg_forgetting.data.contracts import DatasetProtocolError, SubjectSplit, load_yaml
from eeg_forgetting.data.datasets.physionet_mi import (
    EXCLUDED_SUBJECTS,
    IMAGERY_RUNS,
    cbramod_reproduction_split,
    clean_split,
)
from eeg_forgetting.data.datasets.sleep_edf import audit_sleep_cassette, discover_recordings
from eeg_forgetting.data.manifests import sha256_file, write_json_once, write_jsonl_once
from eeg_forgetting.data.splits import StratifiedSubject, stratified_subject_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New version directory, for example manifests/v4; existing files are never replaced.",
    )
    parser.add_argument(
        "--bci-split-config",
        type=Path,
        help="Optional locked BCI split config for a new robustness manifest set.",
    )
    return parser.parse_args()


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def read_sha256s(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            digest, name = line.rstrip().split(maxsplit=1)
            checksums[name.lstrip("*")] = digest
    return checksums


def checked_source(path: Path, expected_sha256: str | None = None) -> dict[str, object]:
    if not path.is_file():
        raise DatasetProtocolError(f"missing manifest source file {path}")
    observed = sha256_file(path)
    if expected_sha256 is not None and observed != expected_sha256:
        raise DatasetProtocolError(
            f"checksum mismatch for {path}: expected {expected_sha256}, got {observed}"
        )
    return {"path": relative(path), "sha256": observed, "bytes": path.stat().st_size}


def assignment(split: SubjectSplit) -> dict[str, str]:
    split.validate()
    return {
        subject: group
        for group, subjects in (
            ("train", split.train),
            ("validation", split.validation),
            ("test", split.test),
        )
        for subject in subjects
    }


def bci_rows(
    raw_root: Path, split_config_path: Path | None = None
) -> tuple[list[dict[str, object]], dict[str, object]]:
    root = raw_root / "bciciv2a"
    split_name = "fixed_subject_holdout_v1"
    if split_config_path is None:
        split = SubjectSplit(
            train=tuple(f"A{value:02d}" for value in range(1, 6)),
            validation=("A06", "A07"),
            test=("A08", "A09"),
        )
    else:
        document = load_yaml(split_config_path)
        if document.get("status") != "locked_robustness":
            raise DatasetProtocolError("BCI robustness split config is not locked")
        split_name = str(document["name"])
        split = SubjectSplit(
            train=tuple(str(value) for value in document["train"]),
            validation=tuple(str(value) for value in document["validation"]),
            test=tuple(str(value) for value in document["test"]),
        )
    split.validate(expected_total=9)
    groups = assignment(split)
    rows = []
    for number in range(1, 10):
        subject = f"A{number:02d}"
        files = [
            checked_source(root / f"{subject}{session}.{suffix}")
            for session, suffix in (("T", "gdf"), ("T", "mat"), ("E", "gdf"), ("E", "mat"))
        ]
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset": "bciciv2a",
                "subject_id": subject,
                "split": groups[subject],
                "sessions": ["T", "E"],
                "source_files": files,
            }
        )
    return rows, {
        "subjects": dict(Counter(groups.values())),
        "rows": len(rows),
        "split_name": split_name,
    }


def physionet_rows(raw_root: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    root = raw_root / "physionet_mi" / "eegmmidb" / "1.0.0"
    checksums = read_sha256s(root / "SHA256SUMS.txt")
    main = assignment(clean_split())
    reproduction = assignment(cbramod_reproduction_split())
    reasons = {
        88: "sampling_rate_128_and_nonstandard_event_count",
        92: "sampling_rate_128_and_nonstandard_event_count",
        100: "sampling_rate_128_and_missing_events",
        104: "truncated_run_08_and_missing_events",
    }
    rows = []
    for number in range(1, 110):
        subject = f"S{number:03d}"
        files = []
        for run in IMAGERY_RUNS:
            source_name = f"{subject}/{subject}R{run:02d}.edf"
            if source_name not in checksums:
                raise DatasetProtocolError(f"official checksum absent for {source_name}")
            files.append(checked_source(root / source_name, checksums[source_name]))
        excluded = number in EXCLUDED_SUBJECTS
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset": "physionet_mi",
                "subject_id": subject,
                "main_split": "excluded" if excluded else main[subject],
                "reproduction_split": reproduction[subject],
                "exclusion_reason": reasons.get(number),
                "imagery_runs": list(IMAGERY_RUNS),
                "source_files": files,
            }
        )
    return rows, {
        "subjects": dict(Counter(row["main_split"] for row in rows)),
        "reproduction_subjects": dict(Counter(reproduction.values())),
        "rows": len(rows),
    }


def age_band(age: int, bands: dict[str, list[int | None]]) -> str:
    matches = [
        name
        for name, (lower, upper) in bands.items()
        if age >= int(lower) and (upper is None or age <= int(upper))
    ]
    if len(matches) != 1:
        raise DatasetProtocolError(f"age {age} belongs to {len(matches)} configured bands")
    return matches[0]


def sleep_demographics(root: Path) -> dict[tuple[str, str], dict[str, object]]:
    sheet = xlrd.open_workbook(str(root / "SC-subjects.xls")).sheet_by_index(0)
    expected_header = ["subject", "night", "age", "sex (F=1)", "LightsOff"]
    if sheet.row_values(0) != expected_header:
        raise DatasetProtocolError(f"unexpected SC-subjects.xls header: {sheet.row_values(0)}")
    rows: dict[tuple[str, str], dict[str, object]] = {}
    for index in range(1, sheet.nrows):
        subject_number, night_number, age_value, sex_value, _ = sheet.row_values(index)
        subject = f"SC{int(subject_number):02d}"
        night = str(int(night_number))
        if int(sex_value) not in (1, 2):
            raise DatasetProtocolError(f"{subject} night {night}: invalid sex code {sex_value}")
        key = (subject, night)
        if key in rows:
            raise DatasetProtocolError(f"duplicate demographics row {key}")
        rows[key] = {
            "age": int(age_value),
            "sex": "female" if int(sex_value) == 1 else "male",
        }
    return rows


def sleep_rows(raw_root: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    root = raw_root / "sleep_edf" / "sleep-edfx" / "1.0.0"
    recordings = discover_recordings(root / "sleep-cassette")
    demographics = sleep_demographics(root)
    recording_keys = {(record.subject_id, record.night_id) for record in recordings}
    if recording_keys != set(demographics):
        raise DatasetProtocolError(
            "Sleep recording/demographics mismatch: "
            f"recording-only={sorted(recording_keys - set(demographics))}, "
            f"metadata-only={sorted(set(demographics) - recording_keys)}"
        )

    config = load_yaml(PROJECT_ROOT / "configs" / "datasets" / "sleep_edf.yaml")
    split_config = config["split"]
    bands = split_config["age_bands"]
    by_subject: dict[str, dict[str, object]] = {}
    for (subject, _night), values in demographics.items():
        existing = by_subject.setdefault(subject, values)
        if existing != values:
            raise DatasetProtocolError(f"inconsistent demographics across nights for {subject}")
    stratified = [
        StratifiedSubject(
            subject,
            (age_band(int(values["age"]), bands), values["sex"]),
        )
        for subject, values in sorted(by_subject.items())
    ]
    targets = split_config["target_subject_counts"]
    split = stratified_subject_split(
        stratified,
        train_count=int(targets["train"]),
        validation_count=int(targets["validation"]),
        test_count=int(targets["test"]),
        seed=int(split_config["seed"]),
    )
    audit_sleep_cassette(recordings, split)
    groups = assignment(split)
    checksums = read_sha256s(root / "SHA256SUMS.txt")
    rows = []
    for record in recordings:
        values = demographics[(record.subject_id, record.night_id)]
        source_files = []
        for path in (record.psg_path, record.hypnogram_path):
            source_name = path.relative_to(root).as_posix()
            if source_name not in checksums:
                raise DatasetProtocolError(f"official checksum absent for {source_name}")
            source_files.append(checked_source(path, checksums[source_name]))
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset": "sleep_edf_sc",
                "subject_id": record.subject_id,
                "night_id": record.night_id,
                "recording_id": record.psg_path.name[:6],
                "split": groups[record.subject_id],
                "age": values["age"],
                "sex": values["sex"],
                "age_band": age_band(int(values["age"]), bands),
                "source_files": source_files,
            }
        )

    subject_rows = [
        {
            "split": groups[subject],
            "sex": values["sex"],
            "age": values["age"],
            "age_band": age_band(int(values["age"]), bands),
        }
        for subject, values in sorted(by_subject.items())
    ]
    distributions: dict[str, Any] = {}
    for group in ("train", "validation", "test"):
        selected = [row for row in subject_rows if row["split"] == group]
        ages = [int(row["age"]) for row in selected]
        distributions[group] = {
            "subjects": len(selected),
            "recordings": sum(row["split"] == group for row in rows),
            "sex": dict(sorted(Counter(str(row["sex"]) for row in selected).items())),
            "age_bands": dict(
                sorted(Counter(str(row["age_band"]) for row in selected).items())
            ),
            "age_years": {
                "min": min(ages),
                "max": max(ages),
                "mean": round(sum(ages) / len(ages), 2),
            },
        }
    return rows, {"rows": len(rows), "split_distribution": distributions}


def main() -> None:
    args = parse_args()
    output_files = {
        "bciciv2a": args.output / "bciciv2a.jsonl",
        "physionet_mi": args.output / "physionet_mi.jsonl",
        "sleep_edf_sc": args.output / "sleep_edf_sc.jsonl",
    }
    index_path = args.output / "manifest-set.json"
    existing = [path for path in (*output_files.values(), index_path) if path.exists()]
    if existing:
        raise DatasetProtocolError(
            f"refusing to overwrite immutable manifest set; already exists: {existing}"
        )

    built = {
        "bciciv2a": bci_rows(args.raw_root, args.bci_split_config),
        "physionet_mi": physionet_rows(args.raw_root),
        "sleep_edf_sc": sleep_rows(args.raw_root),
    }
    manifest_index: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "manifest_version": args.output.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sleep_split_seed": 20260813,
        "bci_split_config": (
            relative(args.bci_split_config)
            if args.bci_split_config is not None
            else "configs/datasets/bciciv2a.yaml#split.main"
        ),
        "configs": {
            relative(path): sha256_file(path)
            for path in (
                PROJECT_ROOT / "configs" / "channels.yaml",
                PROJECT_ROOT / "configs" / "common.yaml",
                *sorted((PROJECT_ROOT / "configs" / "datasets").glob("*.yaml")),
            )
        },
        "manifests": {},
    }
    for name, path in output_files.items():
        rows, summary = built[name]
        digest = write_jsonl_once(path, rows)
        manifest_index["manifests"][name] = {
            "path": relative(path),
            "sha256": digest,
            "rows": len(rows),
            "summary": summary,
        }
    write_json_once(index_path, manifest_index)
    print(json.dumps(manifest_index, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
