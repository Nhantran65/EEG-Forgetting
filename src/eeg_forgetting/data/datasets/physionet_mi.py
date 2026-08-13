from __future__ import annotations

import warnings
from pathlib import Path

import mne
import numpy as np

from ..channels import ChannelRegistry
from ..contracts import DatasetProtocolError, EEGSample, SubjectSplit
from ..preprocessing import patchify, preprocess_mne_raw, scale_microvolts


IMAGERY_RUNS = (4, 6, 8, 10, 12, 14)
EXCLUDED_SUBJECTS = {88, 92, 100, 104}
UNILATERAL_RUNS = {4, 8, 12}
CBRAMOD_PHYSIONET_CHANNELS = (
    "Fc5.", "Fc3.", "Fc1.", "Fcz.", "Fc2.", "Fc4.", "Fc6.", "C5..",
    "C3..", "C1..", "Cz..", "C2..", "C4..", "C6..", "Cp5.", "Cp3.",
    "Cp1.", "Cpz.", "Cp2.", "Cp4.", "Cp6.", "Fp1.", "Fpz.", "Fp2.",
    "Af7.", "Af3.", "Afz.", "Af4.", "Af8.", "F7..", "F5..", "F3..",
    "F1..", "Fz..", "F2..", "F4..", "F6..", "F8..", "Ft7.", "Ft8.",
    "T7..", "T8..", "T9..", "T10.", "Tp7.", "Tp8.", "P7..", "P5..",
    "P3..", "P1..", "Pz..", "P2..", "P4..", "P6..", "P8..", "Po7.",
    "Po3.", "Poz.", "Po4.", "Po8.", "O1..", "Oz..", "O2..", "Iz..",
)


def event_label(run: int, description: str) -> int | None:
    event = description.upper()
    if event == "T0":
        return None
    if run in UNILATERAL_RUNS:
        mapping = {"T1": 0, "T2": 1}
    else:
        mapping = {"T1": 2, "T2": 3}
    if event not in mapping:
        raise DatasetProtocolError(f"run {run:02d}: unexpected event {description}")
    return mapping[event]


def audit_run_inventory(
    descriptions: list[str],
    *,
    sampling_rate_hz: float,
    strict_clean_protocol: bool,
    context: str,
) -> dict[str, int]:
    descriptions = [str(value).upper() for value in descriptions]
    unexpected = sorted(set(descriptions) - {"T0", "T1", "T2"})
    if unexpected:
        raise DatasetProtocolError(f"{context}: unexpected annotations {unexpected}")
    counts = {name: descriptions.count(name) for name in {"T0", "T1", "T2"}}
    movements = counts["T1"] + counts["T2"]
    if movements == 0:
        raise DatasetProtocolError(f"{context}: no movement events")
    if strict_clean_protocol:
        if sampling_rate_hz != 160.0:
            raise DatasetProtocolError(f"{context}: expected 160 Hz, got {sampling_rate_hz}")
        if len(descriptions) != 30 or counts["T0"] != 15 or movements != 15:
            raise DatasetProtocolError(f"{context}: nonstandard annotation inventory {counts}")
    return counts


def clean_split() -> SubjectSplit:
    split = SubjectSplit(
        train=tuple(f"S{value:03d}" for value in range(1, 71)),
        validation=tuple(f"S{value:03d}" for value in (*range(71, 88), 89)),
        test=tuple(
            f"S{value:03d}"
            for value in (90, 91, *range(93, 100), *range(101, 104), *range(105, 110))
        ),
    )
    split.validate(expected_total=105)
    return split


def cbramod_reproduction_split() -> SubjectSplit:
    split = SubjectSplit(
        train=tuple(f"S{value:03d}" for value in range(1, 71)),
        validation=tuple(f"S{value:03d}" for value in range(71, 90)),
        test=tuple(f"S{value:03d}" for value in range(90, 110)),
    )
    split.validate(expected_total=109)
    return split


class PhysioNetMILoader:
    def __init__(
        self,
        registry: ChannelRegistry,
        *,
        cbramod_reproduction: bool = False,
    ):
        self.registry = registry
        self.cbramod_reproduction = cbramod_reproduction

    def load_run(self, path: str | Path, *, subject_number: int, run: int) -> list[EEGSample]:
        if run not in IMAGERY_RUNS:
            raise DatasetProtocolError(f"run {run:02d} is not an imagery run")
        if subject_number in EXCLUDED_SUBJECTS and not self.cbramod_reproduction:
            raise DatasetProtocolError(f"S{subject_number:03d} is excluded by direct EDF audit")

        raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        descriptions = [str(value).upper() for value in raw.annotations.description]
        # The clean main experiment is intentionally strict. The one-off CBraMod
        # reproduction must retain the original 109-subject split, including the
        # four files whose rate/event inventories triggered the clean exclusions.
        counts = audit_run_inventory(
            descriptions,
            sampling_rate_hz=float(raw.info["sfreq"]),
            strict_clean_protocol=not self.cbramod_reproduction,
            context=str(path),
        )

        if self.cbramod_reproduction:
            missing = [name for name in CBRAMOD_PHYSIONET_CHANNELS if name not in raw.ch_names]
            if missing:
                raise DatasetProtocolError(f"{path}: missing upstream channels {missing}")
            prepared = raw.copy().pick(list(CBRAMOD_PHYSIONET_CHANNELS))
            prepared.set_eeg_reference("average", projection=False, verbose="ERROR")
            prepared.filter(0.3, None, verbose="ERROR")
            prepared.notch_filter(60.0, verbose="ERROR")
            prepared.resample(200.0, verbose="ERROR")
            expected_channels = 64
        else:
            prepared = preprocess_mne_raw(
                raw,
                source_channels=raw.ch_names,
                registry=self.registry,
                montage="bciciv2a_22",
                common_average_reference=True,
            )
            expected_channels = 22
        subject_id = f"S{subject_number:03d}"
        samples: list[EEGSample] = []
        movement_index = 0
        incomplete_events: list[int] = []
        for onset, description in zip(raw.annotations.onset, descriptions):
            label = event_label(run, description)
            if label is None:
                continue
            start = int(round(float(onset) * prepared.info["sfreq"]))
            signal_uv = prepared.get_data(start=start, stop=start + 800, units="uV")
            if signal_uv.shape != (expected_channels, 800):
                if self.cbramod_reproduction:
                    incomplete_events.append(movement_index)
                    movement_index += 1
                    continue
                raise DatasetProtocolError(f"{path}: incomplete event window {movement_index}")
            samples.append(
                EEGSample(
                    signal=patchify(scale_microvolts(signal_uv)),
                    label=label,
                    subject_id=subject_id,
                    recording_id=f"{subject_id}-R{run:02d}",
                    source_id=f"{Path(path).name}:event-{movement_index:02d}",
                )
            )
            movement_index += 1
        expected_movements = counts.get("T1", 0) + counts.get("T2", 0)
        if not samples:
            raise DatasetProtocolError(f"{path}: no complete movement events")
        if incomplete_events:
            warnings.warn(
                f"{path}: reproduction mode dropped incomplete movement events "
                f"{incomplete_events}",
                RuntimeWarning,
                stacklevel=2,
            )
        if not self.cbramod_reproduction and len(samples) != 15:
            raise DatasetProtocolError(f"{path}: expected 15 movement events, got {len(samples)}")
        if not self.cbramod_reproduction and len(samples) != expected_movements:
            raise DatasetProtocolError(
                f"{path}: extracted {len(samples)} of {expected_movements} movement events"
            )
        return samples
