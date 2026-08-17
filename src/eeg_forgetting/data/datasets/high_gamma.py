from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import mne

from ..channels import ChannelRegistry
from ..contracts import DatasetProtocolError, EEGSample
from ..preprocessing import patchify, preprocess_mne_raw, scale_microvolts


CLASS_LABELS = {"left_hand": 0, "right_hand": 1, "feet": 2, "rest": 3}


@dataclass(frozen=True)
class HighGammaEvent:
    onset_seconds: float
    duration_seconds: float
    label: int
    trial_type: str


def read_high_gamma_events(path: str | Path) -> tuple[HighGammaEvent, ...]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    events = []
    for index, row in enumerate(rows):
        trial_type = str(row.get("trial_type", ""))
        if trial_type not in CLASS_LABELS:
            raise DatasetProtocolError(f"{path}: unknown trial type at row {index}: {trial_type!r}")
        duration = float(row["duration"])
        if duration != 4.0:
            raise DatasetProtocolError(f"{path}: event {index} duration changed to {duration}")
        events.append(
            HighGammaEvent(
                onset_seconds=float(row["onset"]),
                duration_seconds=duration,
                label=CLASS_LABELS[trial_type],
                trial_type=trial_type,
            )
        )
    if not events:
        raise DatasetProtocolError(f"{path}: no task events")
    return tuple(events)


class HighGammaLoader:
    def __init__(self, registry: ChannelRegistry):
        self.registry = registry

    def load_run(
        self,
        bdf_path: str | Path,
        events_path: str | Path,
        *,
        subject_id: str,
        acquisition: str,
    ) -> list[EEGSample]:
        if acquisition not in {"train", "test"}:
            raise DatasetProtocolError(f"unknown High-Gamma acquisition {acquisition!r}")
        events = read_high_gamma_events(events_path)
        raw = mne.io.read_raw_bdf(bdf_path, preload=False, verbose="ERROR")
        if float(raw.info["sfreq"]) != 500.0:
            raise DatasetProtocolError(f"{bdf_path}: expected 500 Hz")
        prepared = preprocess_mne_raw(
            raw,
            source_channels=raw.ch_names,
            registry=self.registry,
            montage="bciciv2a_22",
            common_average_reference=True,
        )
        samples = []
        for trial_index, event in enumerate(events):
            start = int(round(event.onset_seconds * prepared.info["sfreq"]))
            stop = start + 4 * 200
            signal_uv = prepared.get_data(start=start, stop=stop, units="uV")
            if signal_uv.shape != (22, 800):
                raise DatasetProtocolError(
                    f"{bdf_path}: event {trial_index} expected (22, 800), got {signal_uv.shape}"
                )
            samples.append(
                EEGSample(
                    signal=patchify(scale_microvolts(signal_uv)),
                    label=event.label,
                    subject_id=subject_id,
                    recording_id=f"{subject_id}-{acquisition}",
                    source_id=f"{Path(bdf_path).name}:trial-{trial_index:04d}",
                )
            )
        counts = Counter(sample.label for sample in samples)
        if set(counts) != {0, 1, 2, 3} or max(counts.values()) - min(counts.values()) > 1:
            raise DatasetProtocolError(f"{bdf_path}: unbalanced class counts {dict(counts)}")
        return samples
