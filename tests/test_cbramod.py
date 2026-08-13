import pytest
import torch

from eeg_forgetting.data.contracts import DatasetProtocolError
from eeg_forgetting.models.cbramod import CBraMod, CBraModConfig, CBraModTaskModel


@pytest.mark.parametrize(
    ("channels", "patches", "classes"),
    [(22, 4, 4), (2, 30, 5), (16, 5, 6)],
)
def test_cbramod_supports_locked_variable_shapes(
    channels: int, patches: int, classes: int
) -> None:
    backbone = CBraMod(CBraModConfig(n_layer=1))
    model = CBraModTaskModel(backbone, classes).eval()
    with torch.no_grad():
        logits, features = model(
            torch.randn(1, channels, patches, 200), return_features=True
        )
    assert features.shape == (1, channels, patches, 200)
    assert logits.shape == (1, classes)


def test_trainable_depth_unfreezes_only_last_encoder_blocks() -> None:
    backbone = CBraMod(CBraModConfig(n_layer=3))
    backbone.set_trainable_depth(1)
    assert not any(parameter.requires_grad for parameter in backbone.patch_embedding.parameters())
    assert not any(parameter.requires_grad for parameter in backbone.encoder.layers[0].parameters())
    assert not any(parameter.requires_grad for parameter in backbone.encoder.layers[1].parameters())
    assert all(parameter.requires_grad for parameter in backbone.encoder.layers[2].parameters())
    assert not any(parameter.requires_grad for parameter in backbone.proj_out.parameters())


def test_flatten_mlp_head_preserves_patch_structure() -> None:
    backbone = CBraMod(CBraModConfig(n_layer=1))
    model = CBraModTaskModel(
        backbone, 4, head="flatten_mlp", channels=22, patches=4
    ).eval()
    with torch.no_grad():
        assert model(torch.randn(2, 22, 4, 200)).shape == (2, 4)


def test_flatten_linear_probe_preserves_patch_structure() -> None:
    backbone = CBraMod(CBraModConfig(n_layer=1))
    model = CBraModTaskModel(
        backbone, 4, head="flatten_linear", channels=22, patches=4
    ).eval()
    with torch.no_grad():
        assert model(torch.randn(2, 22, 4, 200)).shape == (2, 4)
    assert model.classifier[1].in_features == 22 * 4 * 200


def test_cbramod_rejects_wrong_patch_size() -> None:
    backbone = CBraMod(CBraModConfig(n_layer=1))
    with pytest.raises(DatasetProtocolError, match="200 points"):
        backbone(torch.randn(1, 2, 4, 199))
