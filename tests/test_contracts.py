from __future__ import annotations

import numpy as np
import pytest

from eeg_forgetting.data.contracts import (
    DatasetProtocolError,
    SubjectSplit,
    assert_class_counts,
    fixed_step_batches,
)
from eeg_forgetting.data.preprocessing import patchify, scale_microvolts


def test_subject_split_rejects_leakage() -> None:
    with pytest.raises(DatasetProtocolError, match="subject leakage"):
        SubjectSplit(("S1",), ("S2",), ("S1",)).validate()


def test_class_inventory_fails_loud() -> None:
    with pytest.raises(DatasetProtocolError, match="72 trials/class"):
        assert_class_counts(
            [0] * 72 + [1] * 72 + [2] * 72 + [3] * 71,
            expected_classes=4,
            expected_per_class=72,
            context="synthetic session",
        )


def test_fixed_step_budget_is_exact_and_deterministic() -> None:
    first = list(fixed_step_batches(7, batch_size=4, optimizer_steps=5, seed=13))
    second = list(fixed_step_batches(7, batch_size=4, optimizer_steps=5, seed=13))
    assert len(first) == 5
    assert sum(len(batch) for batch in first) == 20
    assert all(np.array_equal(left, right) for left, right in zip(first, second))
    assert all(((0 <= batch) & (batch < 7)).all() for batch in first)


def test_cbramod_amplitude_and_patch_contract() -> None:
    signal_uv = np.full((2, 6000), 100.0, dtype=np.float32)
    patched = patchify(scale_microvolts(signal_uv))
    assert patched.shape == (2, 30, 200)
    assert np.all(patched == 1.0)
