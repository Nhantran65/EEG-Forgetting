from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mne
import numpy as np

from ..channels import ChannelRegistry
from ..contracts import DatasetProtocolError, EEGSample, SubjectSplit
from ..preprocessing import patchify, preprocess_mne_raw, scale_microvolts


STAGE_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}
DROP_STAGES = {"Sleep stage ?", "Movement time"}


@dataclass(frozen=True)
class SleepRecording:
    subject_id: str
    night_id: str
    psg_path: Path
    hypnogram_path: Path


def sleep_cassette_identity(filename: str) -> tuple[str, str]:
    match = re.match(r"SC4(?P<subject>\d{2})(?P<night>\d)", Path(filename).name)
    if not match:
        raise DatasetProtocolError(f"not a Sleep Cassette filename: {filename}")
    return f"SC{match.group('subject')}", match.group("night")


def discover_recordings(root: str | Path) -> list[SleepRecording]:
    root = Path(root)

    def index_unique(paths: Iterable[Path], *, kind: str) -> dict[str, Path]:
        indexed: dict[str, Path] = {}
        for path in paths:
            key = path.name[:6]
            if key in indexed:
                raise DatasetProtocolError(
                    f"duplicate {kind} recording key {key}: {indexed[key]} and {path}"
                )
            indexed[key] = path
        return indexed

    psg = index_unique(root.rglob("SC*-PSG.edf"), kind="PSG")
    hyp = index_unique(root.rglob("SC*-Hypnogram.edf"), kind="hypnogram")
    if set(psg) != set(hyp):
        raise DatasetProtocolError(
            f"unpaired Sleep-EDF files: PSG-only={sorted(set(psg)-set(hyp))}, "
            f"hypnogram-only={sorted(set(hyp)-set(psg))}"
        )
    recordings = []
    for key in sorted(psg):
        subject, night = sleep_cassette_identity(psg[key].name)
        recordings.append(SleepRecording(subject, night, psg[key], hyp[key]))
    return recordings


def audit_subject_split(recordings: Iterable[SleepRecording], split: SubjectSplit) -> None:
    split.validate()
    recordings = tuple(recordings)
    assignment = {
        subject: group
        for group, subjects in (
            ("train", split.train),
            ("validation", split.validation),
            ("test", split.test),
        )
        for subject in subjects
    }
    missing = sorted({record.subject_id for record in recordings} - set(assignment))
    if missing:
        raise DatasetProtocolError(f"recording subjects absent from split: {missing}")
    extra = sorted(set(assignment) - {record.subject_id for record in recordings})
    if extra:
        raise DatasetProtocolError(f"split subjects have no recording: {extra}")


def audit_sleep_cassette(
    recordings: Iterable[SleepRecording],
    split: SubjectSplit,
    *,
    expected_subjects: int = 78,
    expected_recordings: int = 153,
) -> None:
    recordings = tuple(recordings)
    audit_subject_split(recordings, split)
    subjects = {record.subject_id for record in recordings}
    recording_ids = {(record.subject_id, record.night_id) for record in recordings}
    if len(recordings) != expected_recordings or len(recording_ids) != expected_recordings:
        raise DatasetProtocolError(
            f"expected {expected_recordings} unique recordings, got "
            f"{len(recordings)} rows/{len(recording_ids)} unique IDs"
        )
    if len(subjects) != expected_subjects:
        raise DatasetProtocolError(f"expected {expected_subjects} subjects, got {len(subjects)}")


class SleepEDFLoader:
    def __init__(self, registry: ChannelRegistry, *, wake_crop_minutes: int = 30):
        self.registry = registry
        self.wake_crop_epochs = wake_crop_minutes * 2

    def load_recording(self, recording: SleepRecording) -> list[EEGSample]:
        raw = mne.io.read_raw_edf(recording.psg_path, preload=True, verbose="ERROR")
        annotations = mne.read_annotations(recording.hypnogram_path)
        raw.set_annotations(annotations, emit_warning=False)
        prepared = preprocess_mne_raw(
            raw,
            source_channels=raw.ch_names,
            registry=self.registry,
            montage="sleep_edf_bipolar",
            common_average_reference=False,
        )

        epochs: list[tuple[float, int]] = []
        for onset, duration, description in zip(
            raw.annotations.onset,
            raw.annotations.duration,
            raw.annotations.description,
        ):
            if description in DROP_STAGES:
                continue
            if description not in STAGE_MAP:
                raise DatasetProtocolError(
                    f"{recording.hypnogram_path}: unknown stage {description!r}"
                )
            for offset in range(int(float(duration) // 30)):
                epochs.append((float(onset) + offset * 30.0, STAGE_MAP[description]))
        nonwake = [onset for onset, label in epochs if label != 0]
        if not nonwake:
            raise DatasetProtocolError(f"{recording.psg_path}: no sleep epochs")
        keep_start_seconds = nonwake[0] - self.wake_crop_epochs * 30.0
        keep_stop_seconds = nonwake[-1] + (self.wake_crop_epochs + 1) * 30.0
        kept_epochs = [
            (index, onset, label)
            for index, (onset, label) in enumerate(epochs)
            if keep_start_seconds <= onset < keep_stop_seconds
        ]

        samples: list[EEGSample] = []
        for epoch_index, onset, label in kept_epochs:
            start = int(round(onset * prepared.info["sfreq"]))
            signal_uv = prepared.get_data(start=start, stop=start + 6000, units="uV")
            if signal_uv.shape != (2, 6000):
                raise DatasetProtocolError(
                    f"{recording.psg_path}: incomplete 30-second epoch {epoch_index}"
                )
            samples.append(
                EEGSample(
                    signal=patchify(scale_microvolts(signal_uv)),
                    label=label,
                    subject_id=recording.subject_id,
                    recording_id=f"{recording.subject_id}-night-{recording.night_id}",
                    source_id=f"{recording.psg_path.name}:epoch-{epoch_index:04d}",
                )
            )
        if not samples:
            raise DatasetProtocolError(f"{recording.psg_path}: no epochs after wake crop")
        return samples
