from pathlib import Path

import pytest

from eeg_forgetting.data.contracts import DatasetProtocolError, SubjectSplit
from eeg_forgetting.data.datasets.sleep_edf import (
    SleepRecording,
    audit_sleep_cassette,
    audit_subject_split,
    discover_recordings,
    sleep_cassette_identity,
)


def test_sleep_cassette_identity_groups_nights_by_subject() -> None:
    assert sleep_cassette_identity("SC4001E0-PSG.edf") == ("SC00", "1")
    assert sleep_cassette_identity("SC4002E0-PSG.edf") == ("SC00", "2")


def test_both_nights_use_one_subject_assignment() -> None:
    recordings = [
        SleepRecording("SC00", "1", Path("night1"), Path("hyp1")),
        SleepRecording("SC00", "2", Path("night2"), Path("hyp2")),
    ]
    audit_subject_split(recordings, SubjectSplit(("SC00",), (), ()))


def test_unassigned_recording_subject_fails() -> None:
    recordings = [SleepRecording("SC00", "1", Path("night1"), Path("hyp1"))]
    with pytest.raises(DatasetProtocolError, match="absent from split"):
        audit_subject_split(recordings, SubjectSplit((), (), ()))


def test_dataset_inventory_audit_checks_subject_and_recording_counts() -> None:
    recordings = [
        SleepRecording("SC00", "1", Path("night1"), Path("hyp1")),
        SleepRecording("SC00", "2", Path("night2"), Path("hyp2")),
    ]
    audit_sleep_cassette(
        recordings,
        SubjectSplit(("SC00",), (), ()),
        expected_subjects=1,
        expected_recordings=2,
    )


def test_duplicate_recording_key_fails_instead_of_overwriting(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "SC4001E0-PSG.edf").touch()
    (second / "SC4001E0-PSG.edf").touch()
    with pytest.raises(DatasetProtocolError, match="duplicate PSG recording key"):
        discover_recordings(tmp_path)
