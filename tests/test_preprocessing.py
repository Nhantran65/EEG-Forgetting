from pathlib import Path

import mne
import numpy as np

from eeg_forgetting.data.channels import ChannelRegistry
from eeg_forgetting.data.preprocessing import (
    patchify,
    preprocess_mne_raw,
    scale_microvolts,
)


ROOT = Path(__file__).parents[1]


def test_mne_pipeline_selects_filters_resamples_and_patches() -> None:
    source_rate = 100.0
    seconds = 20
    times = np.arange(int(source_rate * seconds)) / source_rate
    signal_volts = np.vstack(
        [
            50e-6 * np.sin(2 * np.pi * 10 * times),
            25e-6 * np.sin(2 * np.pi * 20 * times),
            10e-6 * np.sin(2 * np.pi * 45 * times),
        ]
    )
    raw = mne.io.RawArray(
        signal_volts,
        mne.create_info(
            ["EEG Fpz-Cz", "EEG Pz-Oz", "unused"],
            sfreq=source_rate,
            ch_types="eeg",
        ),
        verbose="ERROR",
    )
    registry = ChannelRegistry.from_yaml(ROOT / "configs/channels.yaml")

    prepared = preprocess_mne_raw(
        raw,
        source_channels=raw.ch_names,
        registry=registry,
        montage="sleep_edf_bipolar",
        common_average_reference=False,
    )

    assert prepared.info["sfreq"] == 200.0
    assert prepared.ch_names == ["EEG Fpz-Cz", "EEG Pz-Oz"]
    patched = patchify(scale_microvolts(prepared.get_data(units="uV")))
    assert patched.shape == (2, seconds, 200)
    assert np.isfinite(patched).all()
