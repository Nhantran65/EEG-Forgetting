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


def multiclass_metrics(
    truth: Sequence[int] | np.ndarray,
    prediction: Sequence[int] | np.ndarray,
    *,
    classes: int,
) -> dict[str, float]:
    truth = np.asarray(truth, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    if truth.shape != prediction.shape or truth.ndim != 1 or truth.size == 0:
        raise DatasetProtocolError("multiclass metrics need equal non-empty 1D arrays")
    if classes <= 1:
        raise DatasetProtocolError("multiclass metrics need at least two classes")
    if np.any((truth < 0) | (truth >= classes)):
        raise DatasetProtocolError("truth contains labels outside the declared class range")
    if np.any((prediction < 0) | (prediction >= classes)):
        raise DatasetProtocolError(
            "prediction contains labels outside the declared class range"
        )
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (truth, prediction), 1)
    total = int(matrix.sum())
    observed = float(np.trace(matrix) / total)
    expected = float((matrix.sum(axis=1) @ matrix.sum(axis=0)) / (total * total))
    kappa = (observed - expected) / (1.0 - expected) if expected < 1.0 else 0.0
    f1_values = []
    supports = matrix.sum(axis=1)
    for label in range(classes):
        true_positive = int(matrix[label, label])
        false_positive = int(matrix[:, label].sum() - true_positive)
        false_negative = int(matrix[label, :].sum() - true_positive)
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(2 * true_positive / denominator if denominator else 0.0)
    return {
        "accuracy": observed,
        "balanced_accuracy": balanced_accuracy(truth, prediction),
        "macro_f1": float(np.mean(f1_values)),
        "weighted_f1": float(np.average(f1_values, weights=supports)),
        "cohen_kappa": float(kappa),
    }
