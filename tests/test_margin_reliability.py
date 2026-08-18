from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_margin_reliability.py"
SPEC = spec_from_file_location("run_margin_reliability", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_reliability_decision_uses_subject_median_and_fraction() -> None:
    gate = {
        "subject_threshold": 0.5,
        "minimum_median_subject_cosine": 0.7,
        "minimum_subject_fraction_at_threshold": 2 / 3,
    }
    passed = MODULE.reliability_decision({"A": 0.9, "B": 0.8, "C": 0.4}, gate)
    assert passed["passed"] is True
    failed = MODULE.reliability_decision({"A": 0.9, "B": 0.4, "C": 0.3}, gate)
    assert failed["passed"] is False


def test_fidelity_random_masks_are_unique_and_exclude_top_mask() -> None:
    forbidden = (0, 2, 4)
    masks = MODULE._random_mask_indicators(
        cells=10,
        selected_cells=3,
        total=20,
        seed=17,
        forbidden=forbidden,
    )
    assert masks.shape == (20, 10)
    assert np.all(masks.sum(axis=1) == 3)
    selected = {tuple(np.flatnonzero(row)) for row in masks}
    assert len(selected) == 20
    assert forbidden not in selected
