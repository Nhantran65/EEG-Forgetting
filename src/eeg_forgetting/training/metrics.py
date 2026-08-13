from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from eeg_forgetting.data.contracts import DatasetProtocolError


def balanced_accuracy(
    truth: Sequence[int] | np.ndarray,
    prediction: Sequence[int] | np.ndarray,
) -> float:
    truth = np.asarray(truth, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    if truth.shape != prediction.shape or truth.ndim != 1 or truth.size == 0:
        raise DatasetProtocolError(
            f"balanced accuracy needs equal non-empty 1D arrays, got {truth.shape}/{prediction.shape}"
        )
    recalls = [
        float(np.mean(prediction[truth == label] == label))
        for label in np.unique(truth)
    ]
    return float(np.mean(recalls))


def subject_balanced_accuracy(
    truth: Sequence[int] | np.ndarray,
    prediction: Sequence[int] | np.ndarray,
    subjects: Sequence[str],
) -> tuple[float, dict[str, float]]:
    truth = np.asarray(truth, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    if len(subjects) != truth.size:
        raise DatasetProtocolError("subject IDs do not align with metric arrays")
    indices: dict[str, list[int]] = defaultdict(list)
    for index, subject in enumerate(subjects):
        indices[str(subject)].append(index)
    values = {
        subject: balanced_accuracy(truth[rows], prediction[rows])
        for subject, rows in sorted(indices.items())
    }
    return float(np.mean(list(values.values()))), values
