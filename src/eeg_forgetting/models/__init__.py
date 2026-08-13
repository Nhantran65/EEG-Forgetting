"""Model adapters used by the EEG forgetting experiments."""

from .cbramod import (
    CBRAMOD_UPSTREAM_COMMIT,
    CBraMod,
    CBraModConfig,
    CBraModTaskModel,
    load_pretrained_backbone,
)

__all__ = [
    "CBRAMOD_UPSTREAM_COMMIT",
    "CBraMod",
    "CBraModConfig",
    "CBraModTaskModel",
    "load_pretrained_backbone",
]
