from collections import OrderedDict

import numpy as np
import pytest
import torch
from torch import nn

from eeg_forgetting.training.fisher import (
    combine_fisher,
    deterministic_sample_indices,
    diagonal_empirical_fisher,
    fisher_cosine,
    layer_l2_normalize,
    selected_plastic_parameters,
)


class _ToyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(2, 2, bias=False), nn.Linear(2, 2, bias=False)]
        )


class _ToyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _ToyEncoder()


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _ToyBackbone()
        self.classifier = nn.Linear(2, 2, bias=False)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        value = signal
        for layer in self.backbone.encoder.layers:
            value = torch.tanh(layer(value))
        return self.classifier(value)


def test_fisher_sampling_is_deterministic_without_replacement() -> None:
    first = deterministic_sample_indices(100, sample_count=20, seed=7)
    second = deterministic_sample_indices(100, sample_count=20, seed=7)
    assert np.array_equal(first, second)
    assert len(set(first.tolist())) == 20


def test_plastic_parameter_selection_excludes_head_and_earlier_layers() -> None:
    selected = selected_plastic_parameters(_ToyModel(), final_blocks=1)
    assert list(selected) == ["backbone.encoder.layers.1.weight"]


def test_exact_per_example_fisher_is_finite_and_nonnegative() -> None:
    dataset = [
        (torch.tensor([1.0, 0.0]), torch.tensor(0), "A"),
        (torch.tensor([0.0, 1.0]), torch.tensor(1), "B"),
    ]
    signature = diagonal_empirical_fisher(
        _ToyModel(),
        dataset,
        [0, 1],
        final_blocks=1,
        microbatch_size=2,
        device="cpu",
    )
    assert list(signature) == ["backbone.encoder.layers.1.weight"]
    assert torch.isfinite(signature["backbone.encoder.layers.1.weight"]).all()
    assert torch.all(signature["backbone.encoder.layers.1.weight"] >= 0)


def test_layer_normalization_gives_each_layer_unit_norm() -> None:
    signature = OrderedDict(
        {
            "backbone.encoder.layers.8.weight": torch.tensor([3.0, 4.0]),
            "backbone.encoder.layers.9.weight": torch.tensor([0.0, 2.0]),
        }
    )
    normalized = layer_l2_normalize(signature)
    assert torch.linalg.vector_norm(normalized["backbone.encoder.layers.8.weight"]) == pytest.approx(1.0)
    assert torch.linalg.vector_norm(normalized["backbone.encoder.layers.9.weight"]) == pytest.approx(1.0)
    assert fisher_cosine(normalized, normalized) == pytest.approx(1.0)


def test_combined_fisher_is_sample_weighted() -> None:
    left = OrderedDict({"x": torch.tensor([1.0])})
    right = OrderedDict({"x": torch.tensor([3.0])})
    combined = combine_fisher(left, right, left_samples=1, right_samples=3)
    assert combined["x"].item() == pytest.approx(2.5)
