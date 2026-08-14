from collections import OrderedDict

import pytest
import torch

from eeg_forgetting.data.contracts import DatasetProtocolError
from eeg_forgetting.training.continual_methods import (
    ByteCappedReservoir,
    ewc_quadratic_penalty,
    select_method_candidate,
)


def test_ewc_quadratic_penalty_matches_definition() -> None:
    parameters = {"weight": torch.tensor([2.0, -1.0])}
    fisher = OrderedDict(weight=torch.tensor([0.5, 2.0]))
    anchors = OrderedDict(weight=torch.tensor([1.0, 1.0]))
    assert ewc_quadratic_penalty(parameters, fisher, anchors) == pytest.approx(8.5)


def test_ewc_quadratic_penalty_rejects_misaligned_state() -> None:
    with pytest.raises(DatasetProtocolError, match="aligned"):
        ewc_quadratic_penalty(
            {"weight": torch.zeros(1)},
            OrderedDict(weight=torch.ones(1)),
            OrderedDict(other=torch.zeros(1)),
        )


def test_byte_capped_reservoir_accounts_and_round_trips_tasks() -> None:
    probe = ByteCappedReservoir(0, seed=7)
    byte_cap = probe._COUNTER_BYTES + 2 * probe.slot_bytes
    reservoir = ByteCappedReservoir(byte_cap, seed=7)
    signals = torch.arange(2 * 22 * 4 * 200, dtype=torch.float32).reshape(2, 22, 4, 200)
    labels = torch.tensor([1, 3])
    logits = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
    reservoir.add_batch("bciciv2a", signals, labels, logits)
    assert reservoir.capacity == 2
    assert reservoir.allocated_bytes == byte_cap
    assert reservoir.inventory()["stored_eeg_seconds"] == 8
    sampled = reservoir.sample(2)["bciciv2a"]
    assert sampled["signals"].shape == (2, 22, 4, 200)
    assert sorted(sampled["labels"].tolist()) == [1, 3]

    restored = ByteCappedReservoir(byte_cap, seed=99)
    restored.load_state_dict(reservoir.state_dict())
    assert restored.inventory() == reservoir.inventory()


def test_zero_byte_reservoir_still_tracks_stream_without_storage() -> None:
    reservoir = ByteCappedReservoir(0, seed=7)
    reservoir.add_batch(
        "sleep_edf_sc",
        torch.zeros(3, 2, 30, 200),
        torch.tensor([0, 1, 2]),
        torch.zeros(3, 5),
    )
    assert reservoir.capacity == 0
    assert reservoir.size == 0
    assert reservoir.seen == 3
    assert reservoir.allocated_bytes == 0


def _selection_result(method: str, candidate: float, new: float, forgetting: float):
    return {
        "method": method,
        "candidate": candidate,
        "final_evaluations": {
            "sleep_edf_sc": {"mean_subject_balanced_accuracy": new}
        },
        "pairwise_forgetting": {"relative_forgetting": forgetting},
    }


def test_method_selection_enforces_new_task_plasticity_floor() -> None:
    selected = select_method_candidate(
        [
            _selection_result("ewc", 0, 0.70, 0.50),
            _selection_result("ewc", 1000, 0.69, 0.30),
            _selection_result("ewc", 10000, 0.60, 0.10),
        ],
        maximum_new_task_drop=0.02,
    )
    assert selected["minimum_allowed_new_task_validation"] == pytest.approx(0.68)
    assert selected["selected_candidate"] == 1000
