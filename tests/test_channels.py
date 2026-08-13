from pathlib import Path

import pytest

from eeg_forgetting.data.channels import ChannelRegistry
from eeg_forgetting.data.contracts import DatasetProtocolError


ROOT = Path(__file__).parents[1]


def test_physionet_aliases_resolve_to_bciciv2a_order() -> None:
    registry = ChannelRegistry.from_yaml(ROOT / "configs/channels.yaml")
    source = [
        "Fz..", "Fc3.", "Fc1.", "Fcz.", "Fc2.", "Fc4.", "C5..", "C3..",
        "C1..", "Cz..", "C2..", "C4..", "C6..", "Cp3.", "Cp1.", "Cpz.",
        "Cp2.", "Cp4.", "P1..", "Pz..", "P2..", "Poz.",
    ]
    assert registry.indices(source, "bciciv2a_22") == tuple(range(22))


def test_missing_channel_fails_loud() -> None:
    registry = ChannelRegistry.from_yaml(ROOT / "configs/channels.yaml")
    with pytest.raises(DatasetProtocolError, match="missing channels"):
        registry.indices(["Fz", "Cz"], "bciciv2a_22")


def test_sleep_edf_prefixed_bipolar_names_resolve() -> None:
    registry = ChannelRegistry.from_yaml(ROOT / "configs/channels.yaml")
    assert registry.indices(
        ["EEG Fpz-Cz", "EEG Pz-Oz"], "sleep_edf_bipolar"
    ) == (0, 1)


def test_mne_bci_gdf_placeholder_names_resolve_to_official_order() -> None:
    registry = ChannelRegistry.from_yaml(ROOT / "configs/channels.yaml")
    source = [
        "EEG-Fz",
        *[f"EEG-{index}" for index in range(6)],
        "EEG-C3",
        "EEG-6",
        "EEG-Cz",
        "EEG-7",
        "EEG-C4",
        *[f"EEG-{index}" for index in range(8, 15)],
        "EEG-Pz",
        "EEG-15",
        "EEG-16",
    ]
    assert registry.indices(source, "bciciv2a_22") == tuple(range(22))
