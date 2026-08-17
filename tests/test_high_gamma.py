from pathlib import Path

import pytest

from eeg_forgetting.data.contracts import DatasetProtocolError
from eeg_forgetting.data.datasets.high_gamma import read_high_gamma_events
from eeg_forgetting.training.pilot import PilotSettings


def test_high_gamma_events_are_four_seconds_and_label_locked(tmp_path: Path) -> None:
    path = tmp_path / "events.tsv"
    path.write_text(
        "onset\tduration\ttrial_type\tvalue\tsample\n"
        "1.0\t4.0\tleft_hand\t2\t500\n"
        "8.0\t4.0\tright_hand\t4\t4000\n"
        "15.0\t4.0\tfeet\t1\t7500\n"
        "22.0\t4.0\trest\t3\t11000\n",
        encoding="utf-8",
    )
    events = read_high_gamma_events(path)
    assert [event.label for event in events] == [0, 1, 2, 3]
    assert all(event.duration_seconds == 4.0 for event in events)


def test_high_gamma_events_reject_unknown_class(tmp_path: Path) -> None:
    path = tmp_path / "events.tsv"
    path.write_text(
        "onset\tduration\ttrial_type\n1.0\t4.0\ttongue\n", encoding="utf-8"
    )
    with pytest.raises(DatasetProtocolError, match="unknown trial type"):
        read_high_gamma_events(path)


def test_high_gamma_is_a_declared_pilot_shape() -> None:
    PilotSettings(dataset="high_gamma", depth=4).validate()
