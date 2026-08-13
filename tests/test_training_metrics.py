import pytest

from eeg_forgetting.models.cbramod import CBraMod, CBraModConfig, CBraModTaskModel
from eeg_forgetting.training.metrics import balanced_accuracy, subject_balanced_accuracy
from eeg_forgetting.training.pilot import set_finetune_train_mode


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
