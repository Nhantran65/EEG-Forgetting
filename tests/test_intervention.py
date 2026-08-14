from collections import OrderedDict

import pytest
import torch

from eeg_forgetting.data.contracts import DatasetProtocolError
from eeg_forgetting.training.intervention import (
    build_freeze_masks,
    freeze_mask_sha256,
    restore_frozen_parameters,
)


def _fishers():
    left = OrderedDict(
        {
            "backbone.encoder.layers.8.weight": torch.tensor([1.0, 2.0, 3.0, 4.0]),
            "backbone.encoder.layers.8.bias": torch.tensor([1.0, 2.0]),
            "backbone.encoder.layers.9.weight": torch.tensor([4.0, 3.0, 2.0, 1.0]),
        }
    )
    right = OrderedDict(
        {
            "backbone.encoder.layers.8.weight": torch.tensor([4.0, 3.0, 2.0, 1.0]),
            "backbone.encoder.layers.8.bias": torch.tensor([2.0, 1.0]),
            "backbone.encoder.layers.9.weight": torch.tensor([1.0, 2.0, 3.0, 4.0]),
        }
    )
    return left, right


def test_freeze_masks_match_counts_per_layer_and_are_deterministic() -> None:
    left, right = _fishers()
    high, high_counts = build_freeze_masks(
        left, right, ratio=0.25, condition="high_overlap", random_seed=17
    )
    random_a, random_counts = build_freeze_masks(
        left, right, ratio=0.25, condition="random_0", random_seed=17
    )
    random_b, _ = build_freeze_masks(
        left, right, ratio=0.25, condition="random_0", random_seed=17
    )
    assert high_counts == random_counts == {
        "backbone.encoder.layers.8": 2,
        "backbone.encoder.layers.9": 1,
    }
    assert sum(int(mask.sum()) for mask in high.values()) == 3
    assert freeze_mask_sha256(random_a) == freeze_mask_sha256(random_b)


def test_restore_frozen_parameters_undoes_adamw_decay_only_at_mask() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
    named = {"weight": parameter}
    masks = OrderedDict(weight=torch.tensor([True, False, True]))
    anchors = OrderedDict(weight=parameter.detach().clone())
    optimizer = torch.optim.AdamW([parameter], lr=0.1, weight_decay=0.1)
    parameter.grad = torch.tensor([0.0, 1.0, 0.0])
    optimizer.step()
    assert not torch.equal(parameter.detach(), anchors["weight"])
    restore_frozen_parameters(named, masks, anchors)
    assert torch.equal(parameter.detach()[masks["weight"]], anchors["weight"][masks["weight"]])
    assert parameter.detach()[1] != anchors["weight"][1]


def test_restore_frozen_parameters_rejects_misaligned_state() -> None:
    with pytest.raises(DatasetProtocolError, match="do not align"):
        restore_frozen_parameters(
            {"weight": torch.zeros(2)},
            OrderedDict(weight=torch.tensor([True, False])),
            OrderedDict(other=torch.zeros(2)),
        )
