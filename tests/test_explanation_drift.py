import numpy as np
import pytest
import torch
from torch import nn

from eeg_forgetting.data.contracts import DatasetProtocolError
from eeg_forgetting.xai.explanation_drift import (
    apply_spectral_cell_weights,
    frozen_stratified_capped_halves,
    frozen_stratified_halves,
    jensen_shannon_divergence,
    noise_corrected_symmetric_jsd,
    reliance_maps,
    ridge_mask_coefficients,
    score_spectral_masks,
    spectral_mask_integrated_gradients,
    subject_class_balanced_margin_drop,
    subject_equal_margin_drop,
)


BANDS = (("low", 0.5, 4.0), ("high", 4.0, 40.0))


def test_frozen_halves_are_deterministic_stratified_and_digest_bound() -> None:
    labels = [0, 0, 1, 1] * 2
    subjects = ["A"] * 4 + ["B"] * 4
    first = frozen_stratified_halves(labels, subjects, seed=17)
    second = frozen_stratified_halves(labels, subjects, seed=17)
    assert first == second
    assert set(first["attribution_fit"]).isdisjoint(first["attribution_gate"])
    assert sorted(first["attribution_fit"] + first["attribution_gate"]) == list(range(8))
    for rows in (first["attribution_fit"], first["attribution_gate"]):
        assert {(subjects[row], labels[row]) for row in rows} == {
            ("A", 0), ("A", 1), ("B", 0), ("B", 1)
        }


def test_capped_halves_limit_each_subject_class_before_splitting() -> None:
    labels = [0] * 7 + [1] * 7
    subjects = ["A"] * 14
    split = frozen_stratified_capped_halves(
        labels, subjects, seed=19, maximum_rows_per_subject_class=4
    )
    assert len(split["attribution_fit"]) == 4
    assert len(split["attribution_gate"]) == 4
    assert split["strata"]["A:0"]["available"] == 7
    assert split["strata"]["A:0"]["selected"] == 4


def test_all_one_spectral_weights_reconstruct_and_one_cell_removes_band() -> None:
    time = torch.arange(800) / 200.0
    signal = (torch.sin(2 * torch.pi * 2 * time) + torch.sin(2 * torch.pi * 10 * time))
    signals = signal.reshape(1, 1, 4, 200)
    ones = torch.ones(1, 2)
    reconstructed = apply_spectral_cell_weights(
        signals, ones, sampling_rate_hz=200.0, bands=BANDS
    )
    assert torch.allclose(reconstructed, signals, atol=1e-5)
    low_removed = apply_spectral_cell_weights(
        signals, torch.tensor([[0.0, 1.0]]), sampling_rate_hz=200.0, bands=BANDS
    )
    spectrum = torch.fft.rfft(low_removed.reshape(-1))
    frequencies = torch.fft.rfftfreq(800, d=1 / 200.0)
    assert spectrum[(frequencies >= 0.5) & (frequencies < 4.0)].abs().max() < 1e-4
    assert spectrum[(frequencies >= 4.0) & (frequencies <= 40.0)].abs().max() > 100


def test_reliance_maps_average_subjects_not_trials() -> None:
    truth = np.array([0, 1, 0, 1, 0, 1])
    baseline = truth.copy()
    occluded = np.array([[1, 1, 0, 1, 0, 1]])
    aggregate, by_subject = reliance_maps(
        truth, baseline, occluded, ["A", "A", "B", "B", "B", "B"]
    )
    assert by_subject["A"][0] == pytest.approx(0.5)
    assert by_subject["B"][0] == pytest.approx(0.0)
    assert aggregate[0] == pytest.approx(0.25)


def test_jsd_is_zero_for_equal_and_one_for_disjoint_positive_maps() -> None:
    assert jensen_shannon_divergence([1, 2], [1, 2]) == pytest.approx(0.0)
    assert jensen_shannon_divergence([1, 0], [0, 1]) == pytest.approx(1.0)
    with pytest.raises(DatasetProtocolError, match="positive mass"):
        jensen_shannon_divergence([0, -1], [1, 0])


def test_noise_corrected_symmetric_jsd_subtracts_split_noise() -> None:
    stable = noise_corrected_symmetric_jsd(
        [4, 1], [3, 1], [4, 1], [3, 1]
    )
    assert stable["noise_corrected_ped"] == pytest.approx(0.0)
    changed = noise_corrected_symmetric_jsd(
        [4, 1], [3, 1], [1, 4], [1, 3]
    )
    assert changed["cross_checkpoint_jsd"] > changed["within_checkpoint_noise_jsd"]
    assert changed["noise_corrected_ped"] > 0


class _BandEnergyModel(nn.Module):
    def forward(self, task: str, signals: torch.Tensor) -> torch.Tensor:
        del task
        score = signals.square().mean(dim=(1, 2, 3))
        return torch.stack((score, -score), dim=1)


def test_spectral_mask_ig_has_finite_attribution_and_small_completeness_error() -> None:
    time = torch.arange(800) / 200.0
    signals = torch.sin(2 * torch.pi * 10 * time).reshape(1, 1, 4, 200)
    attribution, error = spectral_mask_integrated_gradients(
        _BandEnergyModel(),
        "task",
        signals,
        torch.tensor([0]),
        sampling_rate_hz=200.0,
        bands=BANDS,
        steps=32,
        alpha_chunk_size=8,
    )
    assert torch.isfinite(attribution).all()
    assert attribution[0, 0, 1] > attribution[0, 0, 0]
    assert error.abs().max() < 1e-4


def test_numpy_index_array_is_accepted_by_dataset_instruments() -> None:
    class _Dataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int):
            assert index == 0
            return torch.zeros(1, 4, 200), torch.tensor(0), "A"

    from eeg_forgetting.xai.explanation_drift import predict_spectral_masks

    result = predict_spectral_masks(
        _BandEnergyModel(),
        "task",
        _Dataset(),
        np.asarray([0]),
        torch.ones(1, 1, len(BANDS)),
        sampling_rate_hz=200.0,
        bands=BANDS,
        batch_size=1,
        mask_chunk_size=1,
        device=torch.device("cpu"),
    )
    assert result["predictions"].shape == (1, 1)


def test_margin_scoring_and_subject_equal_drop_are_continuous() -> None:
    class _Dataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: int):
            time = torch.arange(800) / 200.0
            signal = ((index + 1) * torch.sin(2 * torch.pi * 10 * time)).reshape(
                1, 4, 200
            )
            return signal, torch.tensor(0), "A" if index == 0 else "B"

    scored = score_spectral_masks(
        _BandEnergyModel(),
        "task",
        _Dataset(),
        [0, 1],
        torch.tensor([[[1.0, 1.0]], [[0.0, 0.0]]]),
        sampling_rate_hz=200.0,
        bands=BANDS,
        batch_size=2,
        mask_chunk_size=2,
        device=torch.device("cpu"),
    )
    aggregate, by_subject = subject_equal_margin_drop(
        scored["margins"][0], scored["margins"][1:], scored["subjects"]
    )
    assert aggregate.shape == (1,)
    assert aggregate[0] > 0
    assert set(by_subject) == {"A", "B"}


def test_ridge_mask_coefficients_recover_synthetic_additive_effects() -> None:
    generator = np.random.default_rng(11)
    indicators = generator.integers(0, 2, size=(200, 4)).astype(float)
    expected = np.array([0.4, -0.2, 0.1, 0.7])
    responses = 0.3 + indicators @ expected
    observed, intercept = ridge_mask_coefficients(indicators, responses, alpha=1e-8)
    assert observed == pytest.approx(expected, abs=1e-7)
    assert intercept == pytest.approx(0.3, abs=1e-7)


def test_class_balanced_margin_drop_does_not_weight_common_class_more() -> None:
    baseline = np.zeros(4)
    masked = np.array([[-1.0, -1.0, -1.0, -3.0]])
    aggregate, by_subject = subject_class_balanced_margin_drop(
        baseline, masked, [0, 0, 0, 1], ["A", "A", "A", "A"]
    )
    assert aggregate[0] == pytest.approx(2.0)
    assert by_subject["A"][0] == pytest.approx(2.0)
