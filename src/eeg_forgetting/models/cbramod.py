"""CBraMod backbone adapter pinned to the official ICLR 2025 implementation.

The architecture and state-dict names follow wjq-learning/CBraMod commit
b9e961003214326972c567eff390e75b0287e32a. The upstream code is MIT licensed;
see ``LICENSES/CBraMod-MIT.txt`` and ``docs/product/cbramod-integration.md``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from eeg_forgetting.data.contracts import DatasetProtocolError
from eeg_forgetting.data.manifests import sha256_file


CBRAMOD_UPSTREAM_COMMIT = "b9e961003214326972c567eff390e75b0287e32a"


@dataclass(frozen=True)
class CBraModConfig:
    in_dim: int = 200
    out_dim: int = 200
    d_model: int = 200
    dim_feedforward: int = 800
    n_layer: int = 12
    nhead: int = 8

    def validate(self) -> None:
        if self.in_dim != 200:
            raise DatasetProtocolError("CBraMod checkpoint requires 200-point patches")
        if self.d_model != 200:
            raise DatasetProtocolError("official patch embedding requires d_model=200")
        if self.n_layer <= 0 or self.nhead <= 0 or self.nhead % 2:
            raise DatasetProtocolError("CBraMod needs positive layers and an even head count")


class CrissCrossTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = None

    def forward(self, src: Tensor) -> Tensor:
        output = src
        for layer in self.layers:
            output = layer(output)
        return output


class CrissCrossTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn_s = nn.MultiheadAttention(
            d_model // 2, nhead // 2, dropout=dropout, batch_first=True
        )
        self.self_attn_t = nn.MultiheadAttention(
            d_model // 2, nhead // 2, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: Tensor) -> Tensor:
        x = src
        x = x + self._attention(self.norm1(x))
        x = x + self._feed_forward(self.norm2(x))
        return x

    def _attention(self, x: Tensor) -> Tensor:
        batch, channels, patches, features = x.shape
        spatial = x[..., : features // 2]
        temporal = x[..., features // 2 :]
        spatial = spatial.transpose(1, 2).contiguous().view(
            batch * patches, channels, features // 2
        )
        temporal = temporal.contiguous().view(
            batch * channels, patches, features // 2
        )
        spatial = self.self_attn_s(spatial, spatial, spatial, need_weights=False)[0]
        temporal = self.self_attn_t(temporal, temporal, temporal, need_weights=False)[0]
        spatial = spatial.view(batch, patches, channels, features // 2).transpose(1, 2)
        temporal = temporal.view(batch, channels, patches, features // 2)
        return self.dropout1(torch.cat((spatial, temporal), dim=-1))

    def _feed_forward(self, x: Tensor) -> Tensor:
        return self.dropout2(self.linear2(self.dropout(F.gelu(self.linear1(x)))))


class PatchEmbedding(nn.Module):
    def __init__(self, in_dim: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.positional_encoding = nn.Sequential(
            nn.Conv2d(
                d_model,
                d_model,
                kernel_size=(19, 7),
                stride=(1, 1),
                padding=(9, 3),
                groups=d_model,
            )
        )
        self.mask_encoding = nn.Parameter(torch.zeros(in_dim), requires_grad=False)
        self.proj_in = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
            nn.Conv2d(25, 25, kernel_size=(1, 3), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
            nn.Conv2d(25, 25, kernel_size=(1, 3), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
        )
        self.spectral_proj = nn.Sequential(
            nn.Linear(101, d_model),
            nn.Dropout(0.1),
        )

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        batch, channels, patches, patch_size = x.shape
        if patch_size != 200:
            raise DatasetProtocolError(
                f"CBraMod expected 200 points per patch, got {patch_size}"
            )
        masked = x if mask is None else x.clone()
        if mask is not None:
            masked[mask == 1] = self.mask_encoding
        flattened = masked.contiguous().view(batch, 1, channels * patches, patch_size)
        temporal = self.proj_in(flattened)
        temporal = temporal.permute(0, 2, 1, 3).contiguous().view(
            batch, channels, patches, self.d_model
        )
        spectra = torch.fft.rfft(
            flattened.view(batch * channels * patches, patch_size),
            dim=-1,
            norm="forward",
        ).abs()
        spectral = self.spectral_proj(
            spectra.view(batch, channels, patches, patch_size // 2 + 1)
        )
        embedded = temporal + spectral
        positional = self.positional_encoding(embedded.permute(0, 3, 1, 2))
        return embedded + positional.permute(0, 2, 3, 1)


class CBraMod(nn.Module):
    def __init__(self, config: CBraModConfig | None = None):
        super().__init__()
        self.config = config or CBraModConfig()
        self.config.validate()
        self.patch_embedding = PatchEmbedding(self.config.in_dim, self.config.d_model)
        layer = CrissCrossTransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            dim_feedforward=self.config.dim_feedforward,
        )
        self.encoder = CrissCrossTransformerEncoder(layer, self.config.n_layer)
        self.proj_out = nn.Sequential(nn.Linear(self.config.d_model, self.config.out_dim))
        self.apply(_weights_init)

    def forward_features(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        if x.ndim != 4:
            raise DatasetProtocolError(
                f"CBraMod expected (batch, channels, patches, points), got {tuple(x.shape)}"
            )
        return self.encoder(self.patch_embedding(x, mask))

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        return self.proj_out(self.forward_features(x, mask))

    def set_trainable_depth(self, encoder_layers: int) -> None:
        """Freeze the backbone except for the last N criss-cross blocks."""
        if not 0 <= encoder_layers <= len(self.encoder.layers):
            raise DatasetProtocolError(
                f"trainable depth must be within 0..{len(self.encoder.layers)}"
            )
        for parameter in self.parameters():
            parameter.requires_grad = False
        if encoder_layers:
            for layer in self.encoder.layers[-encoder_layers:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True


class CBraModTaskModel(nn.Module):
    """Diagnostic mean-pool head; the main experiment head is not locked yet."""

    def __init__(self, backbone: CBraMod, num_classes: int):
        super().__init__()
        if num_classes <= 1:
            raise DatasetProtocolError("classification head needs at least two classes")
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.config.d_model, num_classes)

    def forward(
        self, x: Tensor, *, return_features: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        features = self.backbone.forward_features(x)
        logits = self.classifier(features.mean(dim=(1, 2)))
        return (logits, features) if return_features else logits


def load_pretrained_backbone(
    checkpoint: str | Path,
    *,
    expected_sha256: str,
    config: CBraModConfig | None = None,
    map_location: str | torch.device = "cpu",
) -> CBraMod:
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise DatasetProtocolError(f"CBraMod checkpoint is missing: {checkpoint}")
    observed = sha256_file(checkpoint)
    if observed != expected_sha256:
        raise DatasetProtocolError(
            f"CBraMod checkpoint checksum mismatch: expected {expected_sha256}, got {observed}"
        )
    state = torch.load(checkpoint, map_location=map_location, weights_only=True)
    if not isinstance(state, dict):
        raise DatasetProtocolError("CBraMod checkpoint is not a state dictionary")
    model = CBraMod(config)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise DatasetProtocolError(f"CBraMod strict checkpoint load failed: {error}") from error
    return model


def _weights_init(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
