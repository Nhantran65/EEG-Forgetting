import pytest
from pathlib import Path

from eeg_forgetting.models.cbramod import CBraMod, CBraModConfig, CBraModTaskModel
from eeg_forgetting.training.metrics import (
    balanced_accuracy,
    multiclass_metrics,
    subject_balanced_accuracy,
)
from eeg_forgetting.training.pilot import set_finetune_train_mode
from eeg_forgetting.training.pilot import PilotSettings, run_pilot
from eeg_forgetting.training.reproduction import run_physionet_reproduction
from eeg_forgetting.data.contracts import DatasetProtocolError


def test_balanced_accuracy_weights_classes_equally() -> None:
    assert balanced_accuracy([0, 0, 0, 1], [0, 0, 0, 0]) == pytest.approx(0.5)


def test_subject_balanced_accuracy_keeps_subjects_as_units() -> None:
    mean, values = subject_balanced_accuracy(
        [0, 1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0, 0],
        ["A", "A", "B", "B", "B", "B"],
    )
    assert values == {"A": 1.0, "B": 0.75}
    assert mean == pytest.approx(0.875)


def test_finetune_mode_keeps_frozen_dropout_deterministic() -> None:
    backbone = CBraMod(CBraModConfig(n_layer=3))
    backbone.set_trainable_depth(1)
    model = CBraModTaskModel(
        backbone, 4, head="flatten_linear", channels=22, patches=4
    )

    set_finetune_train_mode(model, depth=1)

    assert model.classifier.training
    assert not backbone.patch_embedding.training
    assert not backbone.encoder.layers[0].training
    assert not backbone.encoder.layers[1].training
    assert backbone.encoder.layers[2].training


def test_multiclass_metrics_match_perfect_predictions() -> None:
    metrics = multiclass_metrics([0, 1, 2, 3], [0, 1, 2, 3], classes=4)
    assert metrics == {
        "accuracy": 1.0,
        "balanced_accuracy": 1.0,
        "macro_f1": 1.0,
        "weighted_f1": 1.0,
        "cohen_kappa": 1.0,
    }


def test_multiclass_kappa_is_zero_at_independent_balanced_predictions() -> None:
    metrics = multiclass_metrics(
        [0, 0, 1, 1], [0, 1, 0, 1], classes=2
    )
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["cohen_kappa"] == pytest.approx(0.0)


def test_multiclass_metrics_reject_out_of_range_predictions() -> None:
    with pytest.raises(DatasetProtocolError, match="prediction contains labels"):
        multiclass_metrics([0, 1], [0, 2], classes=2)


def test_pilot_refuses_existing_output_before_loading_data(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    (output / "result.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(DatasetProtocolError, match="refusing to overwrite"):
        run_pilot(
            PilotSettings(dataset="bciciv2a", depth=4),
            cache_root=tmp_path / "missing-cache",
            checkpoint_path=tmp_path / "missing.pt",
            checkpoint_sha256="0" * 64,
            output_dir=output,
            device="cpu",
        )


def test_reproduction_refuses_existing_output_before_loading_data(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    output.mkdir()
    (output / "result.json").write_text("{}\n", encoding="utf-8")
    config = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "training"
        / "physionet_cbramod_reproduction.yaml"
    )

    with pytest.raises(DatasetProtocolError, match="refusing to overwrite"):
        run_physionet_reproduction(
            cache_root=tmp_path / "missing-cache",
            checkpoint_path=tmp_path / "missing.pt",
            checkpoint_sha256="0" * 64,
            config_path=config,
            output_dir=output,
            device="cpu",
        )


def test_pilot_rejects_invalid_early_stopping_settings() -> None:
    with pytest.raises(DatasetProtocolError, match="patience must be positive"):
        PilotSettings(
            dataset="bciciv2a", depth=4, early_stopping_patience_validations=0
        ).validate()
    with pytest.raises(DatasetProtocolError, match="min_delta cannot be negative"):
        PilotSettings(
            dataset="bciciv2a", depth=4, early_stopping_min_delta=-0.1
        ).validate()
