from __future__ import annotations

from typing import Sequence

import numpy as np

from .channels import ChannelRegistry
from .contracts import DatasetProtocolError


TARGET_RATE_HZ = 200.0
PATCH_POINTS = 200
AMPLITUDE_DIVISOR_UV = 100.0


def scale_microvolts(signal_uv: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal_uv, dtype=np.float32) / AMPLITUDE_DIVISOR_UV
    if not np.isfinite(signal).all():
        raise DatasetProtocolError("scaled EEG contains non-finite values")
    return signal


def patchify(signal: np.ndarray, *, points: int = PATCH_POINTS) -> np.ndarray:
    signal = np.asarray(signal)
    if signal.ndim != 2:
        raise DatasetProtocolError(f"expected (channels, time), got {signal.shape}")
    if signal.shape[-1] % points:
        raise DatasetProtocolError(
            f"time dimension {signal.shape[-1]} is not divisible by patch size {points}"
        )
    return signal.reshape(signal.shape[0], signal.shape[-1] // points, points)


def preprocess_mne_raw(
    raw,
    *,
    source_channels: Sequence[str],
    registry: ChannelRegistry,
    montage: str,
    common_average_reference: bool,
    low_hz: float = 0.5,
    high_hz: float = 40.0,
):
    """Return a copied MNE Raw in shared channel/rate/filter convention."""
    selected = registry.indices(source_channels, montage)
    names = [source_channels[index] for index in selected]
    prepared = raw.copy().pick(names)
    if common_average_reference:
        prepared.set_eeg_reference("average", projection=False, verbose="ERROR")
    prepared.filter(low_hz, high_hz, verbose="ERROR")
    prepared.resample(TARGET_RATE_HZ, verbose="ERROR")
    return prepared
