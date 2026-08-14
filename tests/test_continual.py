import pytest

from eeg_forgetting.training.continual import aggregate_replicates, pairwise_forgetting
from eeg_forgetting.training.joint import joint_task_schedule


def _stage(**task_values: float) -> dict[str, object]:
    return {
        "evaluations": {
            task: {"mean_subject_balanced_accuracy": value}
            for task, value in task_values.items()
        }
    }


def test_pairwise_forgetting_tracks_every_prior_task() -> None:
    order = ("bciciv2a", "physionet_mi", "sleep_edf_sc")
    stages = [
        _stage(bciciv2a=0.60),
        _stage(bciciv2a=0.50, physionet_mi=0.55),
        _stage(bciciv2a=0.45, physionet_mi=0.40, sleep_edf_sc=0.70),
    ]
    rows = pairwise_forgetting(stages, order, minimum_headroom=0.05)
    assert [(row["old_task"], row["learned_task"]) for row in rows] == [
        ("bciciv2a", "physionet_mi"),
        ("bciciv2a", "sleep_edf_sc"),
        ("physionet_mi", "sleep_edf_sc"),
    ]
    assert rows[0]["raw_forgetting"] == pytest.approx(0.10)
    assert rows[0]["relative_forgetting"] == pytest.approx(0.10 / 0.35)


def test_relative_forgetting_is_flagged_not_clipped_near_chance() -> None:
    rows = pairwise_forgetting(
        [
            _stage(bciciv2a=0.29),
            _stage(bciciv2a=0.20, physionet_mi=0.50),
        ],
        ("bciciv2a", "physionet_mi"),
        minimum_headroom=0.05,
    )
    assert rows[0]["raw_forgetting"] == pytest.approx(0.09)
    assert rows[0]["headroom"] == pytest.approx(0.04)
    assert rows[0]["relative_valid"] is False
    assert rows[0]["relative_forgetting"] is None


def test_sequential_summary_reports_seed_spread_and_signs() -> None:
    summary = aggregate_replicates([0.1, 0.2, -0.1])
    assert summary["n"] == 3
    assert summary["mean"] == pytest.approx(0.2 / 3)
    assert summary["positive_fraction"] == pytest.approx(2 / 3)
    assert summary["negative_fraction"] == pytest.approx(1 / 3)


def test_joint_schedule_is_task_balanced_and_canonical() -> None:
    schedule = joint_task_schedule(3)
    assert schedule == (
        "bciciv2a",
        "physionet_mi",
        "sleep_edf_sc",
    ) * 3
    assert {task: schedule.count(task) for task in set(schedule)} == {
        "bciciv2a": 3,
        "physionet_mi": 3,
        "sleep_edf_sc": 3,
    }
