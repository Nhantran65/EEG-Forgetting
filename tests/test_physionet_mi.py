from eeg_forgetting.data.datasets.physionet_mi import (
    CBRAMOD_PHYSIONET_CHANNELS,
    EXCLUDED_SUBJECTS,
    audit_run_inventory,
    cbramod_reproduction_split,
    clean_split,
    event_label,
)
from eeg_forgetting.data.contracts import DatasetProtocolError
import pytest


def test_direct_edf_exclusion_inventory() -> None:
    assert EXCLUDED_SUBJECTS == {88, 92, 100, 104}
    assert not ({89, 90, 106} & EXCLUDED_SUBJECTS)


def test_clean_split_preserves_original_ids() -> None:
    split = clean_split()
    assert (len(split.train), len(split.validation), len(split.test)) == (70, 18, 17)
    all_ids = set((*split.train, *split.validation, *split.test))
    assert "S089" in all_ids
    assert not {"S088", "S092", "S100", "S104"} & all_ids


def test_reproduction_split_keeps_all_109_subjects() -> None:
    split = cbramod_reproduction_split()
    assert (len(split.train), len(split.validation), len(split.test)) == (70, 19, 20)
    assert "S088" in split.validation and "S100" in split.test


def test_cbramod_reproduction_channel_order_is_exactly_64() -> None:
    assert len(CBRAMOD_PHYSIONET_CHANNELS) == 64
    assert CBRAMOD_PHYSIONET_CHANNELS[:4] == ("Fc5.", "Fc3.", "Fc1.", "Fcz.")
    assert CBRAMOD_PHYSIONET_CHANNELS[-4:] == ("O1..", "Oz..", "O2..", "Iz..")


def test_run_dependent_event_mapping() -> None:
    assert event_label(4, "T1") == 0
    assert event_label(4, "T2") == 1
    assert event_label(6, "T1") == 2
    assert event_label(6, "T2") == 3
    assert event_label(6, "T0") is None


def test_clean_inventory_rejects_audited_source_anomaly() -> None:
    descriptions = ["T0"] * 15 + ["T1"] * 8 + ["T2"] * 7
    with pytest.raises(DatasetProtocolError, match="expected 160 Hz"):
        audit_run_inventory(
            descriptions,
            sampling_rate_hz=128.0,
            strict_clean_protocol=True,
            context="S088R04",
        )


def test_reproduction_inventory_keeps_nonstandard_original_subject() -> None:
    counts = audit_run_inventory(
        ["T0"] * 12 + ["T1"] * 6 + ["T2"] * 6,
        sampling_rate_hz=128.0,
        strict_clean_protocol=False,
        context="S100R04",
    )
    assert counts == {"T0": 12, "T1": 6, "T2": 6}
