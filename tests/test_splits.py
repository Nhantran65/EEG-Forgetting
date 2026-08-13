from collections import Counter

from eeg_forgetting.data.splits import StratifiedSubject, stratified_subject_split


def test_stratified_subject_split_is_exact_disjoint_and_deterministic() -> None:
    sizes = [10, 10, 10, 10, 10, 10, 9, 9]
    subjects = [
        StratifiedSubject(f"S{index:03d}", f"stratum-{stratum}")
        for stratum, size in enumerate(sizes)
        for index in range(sum(sizes[:stratum]), sum(sizes[:stratum]) + size)
    ]
    first = stratified_subject_split(
        subjects,
        train_count=48,
        validation_count=15,
        test_count=15,
        seed=20260813,
    )
    second = stratified_subject_split(
        subjects,
        train_count=48,
        validation_count=15,
        test_count=15,
        seed=20260813,
    )
    assert first == second
    assert (len(first.train), len(first.validation), len(first.test)) == (48, 15, 15)

    stratum_by_subject = {subject.subject_id: subject.stratum for subject in subjects}
    validation_strata = Counter(stratum_by_subject[value] for value in first.validation)
    test_strata = Counter(stratum_by_subject[value] for value in first.test)
    assert validation_strata == test_strata
