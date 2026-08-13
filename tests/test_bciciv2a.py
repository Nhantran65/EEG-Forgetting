from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.io import savemat

from eeg_forgetting.data.contracts import DatasetProtocolError
from eeg_forgetting.data.datasets.bciciv2a import _trials_from_annotations, load_evaluation_labels


def test_evaluation_labels_come_from_external_mat(tmp_path: Path) -> None:
    path = tmp_path / "A01E.mat"
    labels = np.repeat(np.arange(1, 5), 72).reshape(-1, 1)
    savemat(path, {"classlabel": labels})
    loaded = load_evaluation_labels(path)
    assert loaded.shape == (288,)
    assert set(loaded) == {0, 1, 2, 3}


def test_unbalanced_evaluation_labels_fail_loud(tmp_path: Path) -> None:
    path = tmp_path / "A01E.mat"
    savemat(path, {"classlabel": np.ones((288, 1), dtype=int)})
    with pytest.raises(DatasetProtocolError, match="expected 4 classes"):
        load_evaluation_labels(path)


def _synthetic_training_raw(*, runs: int = 6):
    onsets: list[float] = []
    descriptions: list[str] = []
    trial_counts = [48] * runs
    trial_counts[-1] += 288 - sum(trial_counts)
    for run in range(runs):
        run_start = run * 400.0
        onsets.append(run_start)
        descriptions.append("32766")
        for trial in range(trial_counts[run]):
            start = run_start + trial * 8.0
            onsets.extend([start, start + 2.0])
            descriptions.extend(["768", str(769 + trial % 4)])
    annotations = SimpleNamespace(onset=onsets, description=descriptions)
    return SimpleNamespace(annotations=annotations)


def test_session_requires_six_mi_runs_and_balanced_trials() -> None:
    trials = _trials_from_annotations(
        _synthetic_training_raw(), session="T", evaluation_labels=None
    )
    assert len(trials) == 288
    assert {trial.label for trial in trials} == {0, 1, 2, 3}


def test_eog_calibration_run_markers_are_not_counted_as_mi_runs() -> None:
    raw = _synthetic_training_raw()
    raw.annotations.onset.extend([-300.0, -200.0, -100.0])
    raw.annotations.description.extend(["32766", "32766", "32766"])
    trials = _trials_from_annotations(raw, session="T", evaluation_labels=None)
    assert len(trials) == 288


def test_short_mi_session_fails_loud() -> None:
    with pytest.raises(DatasetProtocolError, match="expected 6 MI runs x 48 trials"):
        _trials_from_annotations(
            _synthetic_training_raw(runs=5), session="T", evaluation_labels=None
        )
