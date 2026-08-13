from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mne
import numpy as np
from scipy.io import loadmat

from ..channels import ChannelRegistry
from ..contracts import DatasetProtocolError, EEGSample, assert_class_counts
from ..preprocessing import patchify, preprocess_mne_raw, scale_microvolts


CLASS_EVENT_CODES = {769: 0, 770: 1, 771: 2, 772: 3}
TRIAL_START = 768
UNKNOWN_EVALUATION_CUE = 783
ARTIFACT = 1023
RUN_START = 32766


def _numeric_code(description: str) -> int | None:
    matches = re.findall(r"\d+", str(description))
    return int(matches[-1]) if matches else None


def load_evaluation_labels(path: str | Path) -> np.ndarray:
    payload = loadmat(path)
    if "classlabel" not in payload:
        raise DatasetProtocolError(f"{path}: missing classlabel field")
    labels = np.asarray(payload["classlabel"]).reshape(-1).astype(int) - 1
    assert_class_counts(
        labels,
        expected_classes=4,
        expected_per_class=72,
        context=f"{path}: evaluation labels",
    )
    return labels


@dataclass(frozen=True)
class _Trial:
    onset_seconds: float
    label: int
    artifact: bool


def _trials_from_annotations(raw, *, session: str, evaluation_labels: np.ndarray | None) -> list[_Trial]:
    annotations = sorted(
        (
            float(onset),
            _numeric_code(description),
        )
        for onset, description in zip(raw.annotations.onset, raw.annotations.description)
    )
    starts = [onset for onset, code in annotations if code == TRIAL_START]
    if len(starts) != 288:
        raise DatasetProtocolError(f"{session}: expected 288 trial starts, got {len(starts)}")
    run_starts = [onset for onset, code in annotations if code == RUN_START]
    trials_per_mi_run = []
    for run_index, run_start in enumerate(run_starts):
        run_stop = run_starts[run_index + 1] if run_index + 1 < len(run_starts) else float("inf")
        trial_count = sum(run_start <= trial_start < run_stop for trial_start in starts)
        if trial_count:
            trials_per_mi_run.append(trial_count)
    if trials_per_mi_run != [48] * 6:
        raise DatasetProtocolError(
            f"{session}: expected 6 MI runs x 48 trials, got {trials_per_mi_run}"
        )

    trials: list[_Trial] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else start + 7.5
        within = [(onset, code) for onset, code in annotations if start <= onset < end]
        cue_codes = [code for _, code in within if code in CLASS_EVENT_CODES or code == UNKNOWN_EVALUATION_CUE]
        if len(cue_codes) != 1:
            raise DatasetProtocolError(f"{session}: trial {index} has cue inventory {cue_codes}")
        cue = cue_codes[0]
        if session == "E":
            if evaluation_labels is None:
                raise DatasetProtocolError(
                    f"{session}: A0xE.mat classlabel is required; GDF cue {cue} is not a class label"
                )
            label = int(evaluation_labels[index])
        else:
            if cue not in CLASS_EVENT_CODES:
                raise DatasetProtocolError(f"{session}: training cue {cue} is not a known class")
            label = CLASS_EVENT_CODES[cue]
        trials.append(
            _Trial(
                onset_seconds=start,
                label=label,
                artifact=any(code == ARTIFACT for _, code in within),
            )
        )

    assert_class_counts(
        [trial.label for trial in trials],
        expected_classes=4,
        expected_per_class=72,
        context=f"{session}: pre-artifact trials",
    )
    return trials


class BCICIV2aLoader:
    """Load official GDF sessions with external evaluation labels and hard audits."""

    def __init__(self, registry: ChannelRegistry, *, exclude_artifacts: bool = True):
        self.registry = registry
        self.exclude_artifacts = exclude_artifacts

    def load_session(
        self,
        gdf_path: str | Path,
        *,
        subject_id: str,
        session: str,
        evaluation_label_path: str | Path | None = None,
    ) -> list[EEGSample]:
        session = session.upper()
        if session not in {"T", "E"}:
            raise DatasetProtocolError(f"unknown BCI IV-2a session {session}")
        external = load_evaluation_labels(evaluation_label_path) if evaluation_label_path else None
        if session == "E" and external is None:
            raise DatasetProtocolError("evaluation session requires A0xE.mat labels")

        raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose="ERROR")
        trials = _trials_from_annotations(raw, session=session, evaluation_labels=external)
        prepared = preprocess_mne_raw(
            raw,
            source_channels=raw.ch_names,
            registry=self.registry,
            montage="bciciv2a_22",
            common_average_reference=True,
        )
        samples: list[EEGSample] = []
        for trial_index, trial in enumerate(trials):
            if self.exclude_artifacts and trial.artifact:
                continue
            start = int(round((trial.onset_seconds + 2.0) * prepared.info["sfreq"]))
            stop = start + 4 * 200
            signal_uv = prepared.get_data(start=start, stop=stop, units="uV")
            if signal_uv.shape != (22, 800):
                raise DatasetProtocolError(
                    f"{gdf_path}: trial {trial_index} expected (22, 800), got {signal_uv.shape}"
                )
            samples.append(
                EEGSample(
                    signal=patchify(scale_microvolts(signal_uv)),
                    label=trial.label,
                    subject_id=subject_id,
                    recording_id=f"{subject_id}-{session}",
                    source_id=f"{Path(gdf_path).name}:trial-{trial_index:03d}",
                )
            )
        if set(sample.label for sample in samples) != {0, 1, 2, 3}:
            raise DatasetProtocolError(f"{gdf_path}: artifact exclusion removed a complete class")
        return samples

    def load_subject(
        self,
        *,
        subject_number: int,
        training_gdf: str | Path,
        evaluation_gdf: str | Path,
        evaluation_labels: str | Path,
    ) -> list[EEGSample]:
        subject_id = f"A{subject_number:02d}"
        training = self.load_session(training_gdf, subject_id=subject_id, session="T")
        evaluation = self.load_session(
            evaluation_gdf,
            subject_id=subject_id,
            session="E",
            evaluation_label_path=evaluation_labels,
        )
        return [*training, *evaluation]
