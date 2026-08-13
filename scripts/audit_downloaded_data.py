#!/usr/bin/env python3
"""Audit real downloaded source files without performing full preprocessing."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import mne

from eeg_forgetting.data.datasets.bciciv2a import (
    _trials_from_annotations,
    load_evaluation_labels,
)
from eeg_forgetting.data.datasets.physionet_mi import (
    EXCLUDED_SUBJECTS,
    audit_run_inventory,
)
from eeg_forgetting.data.datasets.sleep_edf import discover_recordings


def audit_bciciv2a(root: Path) -> None:
    gdf_files = sorted(root.glob("A??[TE].gdf"))
    if len(gdf_files) != 18:
        raise RuntimeError(f"BCI IV-2a: expected 18 GDF files, found {len(gdf_files)}")
    for path in gdf_files:
        session = path.stem[-1]
        labels = load_evaluation_labels(root / f"{path.stem[:3]}E.mat") if session == "E" else None
        raw = mne.io.read_raw_gdf(path, preload=False, verbose="ERROR")
        try:
            _trials_from_annotations(raw, session=session, evaluation_labels=labels)
        finally:
            raw.close()
    print("BCI IV-2a: 18 GDF, 6 runs/session, 288 balanced trials/session")


def audit_physionet(root: Path) -> None:
    edf_files = sorted(root.glob("S???/S???R??.edf"))
    if len(edf_files) != 654:
        raise RuntimeError(f"PhysioNet-MI: expected 654 EDF files, found {len(edf_files)}")
    seen_subjects: set[int] = set()
    for path in edf_files:
        match = re.fullmatch(r"S(?P<subject>\d{3})R(?P<run>\d{2})", path.stem)
        if match is None:
            raise RuntimeError(f"unexpected PhysioNet filename: {path}")
        subject = int(match.group("subject"))
        seen_subjects.add(subject)
        raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
        try:
            audit_run_inventory(
                list(raw.annotations.description),
                sampling_rate_hz=float(raw.info["sfreq"]),
                strict_clean_protocol=subject not in EXCLUDED_SUBJECTS,
                context=str(path),
            )
        finally:
            raw.close()
    if seen_subjects != set(range(1, 110)):
        raise RuntimeError("PhysioNet-MI: subject inventory is not S001..S109")
    print("PhysioNet-MI: 109 subjects x 6 imagery runs; clean exclusions audited")


def audit_sleep_edf(root: Path) -> None:
    recordings = discover_recordings(root / "sleep-cassette")
    subjects = {recording.subject_id for recording in recordings}
    if len(recordings) != 153 or len(subjects) != 78:
        raise RuntimeError(
            f"Sleep-EDF: expected 153 recordings/78 subjects, "
            f"found {len(recordings)}/{len(subjects)}"
        )
    print("Sleep-EDF: 153 paired PSG/hypnogram recordings across 78 subjects")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    audit_bciciv2a(args.root / "bciciv2a")
    audit_physionet(args.root / "physionet_mi/eegmmidb/1.0.0")
    audit_sleep_edf(args.root / "sleep_edf/sleep-edfx/1.0.0")


if __name__ == "__main__":
    main()
