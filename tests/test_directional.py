from collections import OrderedDict

import pytest
import torch

from eeg_forgetting.data.contracts import DatasetProtocolError
from eeg_forgetting.training.directional import gradient_geometry, weighted_update_energy


def test_gradient_geometry_preserves_direction() -> None:
    old = OrderedDict(weight=torch.tensor([1.0, 0.0]), bias=torch.tensor([1.0]))
    aligned = OrderedDict(weight=torch.tensor([2.0, 0.0]), bias=torch.tensor([2.0]))
    opposed = OrderedDict(weight=-aligned["weight"], bias=-aligned["bias"])
    assert gradient_geometry(old, aligned)["conflict_score"] == pytest.approx(-1.0)
    assert gradient_geometry(old, opposed)["conflict_score"] == pytest.approx(1.0)


def test_weighted_update_energy_emphasizes_selected_change() -> None:
    deltas = OrderedDict(weight=torch.tensor([1.0, 3.0]))
    first = OrderedDict(weight=torch.tensor([1.0, 0.0]))
    second = OrderedDict(weight=torch.tensor([0.0, 1.0]))
    assert weighted_update_energy(deltas, first) == pytest.approx(1.0)
    assert weighted_update_energy(deltas, second) == pytest.approx(9.0)


def test_directional_helpers_reject_misaligned_parameters() -> None:
    with pytest.raises(DatasetProtocolError, match="do not align"):
        gradient_geometry(
            OrderedDict(weight=torch.ones(1)), OrderedDict(bias=torch.ones(1))
        )
