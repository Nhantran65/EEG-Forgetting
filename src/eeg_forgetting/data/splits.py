from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Iterable
from dataclasses import dataclass

import numpy as np

from .contracts import DatasetProtocolError, SubjectSplit


@dataclass(frozen=True)
class StratifiedSubject:
    subject_id: str
    stratum: Hashable


def _apportion(
    stratum_sizes: dict[Hashable, int], *, target: int, total: int
) -> dict[Hashable, int]:
    quotas = {key: size * target / total for key, size in stratum_sizes.items()}
    result = {key: int(np.floor(value)) for key, value in quotas.items()}
    remainder = target - sum(result.values())
    order = sorted(
        stratum_sizes,
        key=lambda key: (-(quotas[key] - result[key]), str(key)),
    )
    for key in order[:remainder]:
        result[key] += 1
    return result


def stratified_subject_split(
    subjects: Iterable[StratifiedSubject],
    *,
    train_count: int,
    validation_count: int,
    test_count: int,
    seed: int,
) -> SubjectSplit:
    subjects = tuple(subjects)
    ids = [subject.subject_id for subject in subjects]
    if len(ids) != len(set(ids)):
        raise DatasetProtocolError("stratified split received duplicate subject IDs")
    total = len(subjects)
    if train_count + validation_count + test_count != total:
        raise DatasetProtocolError(
            f"split targets do not sum to population: "
            f"{train_count}+{validation_count}+{test_count}!={total}"
        )

    groups: dict[Hashable, list[str]] = defaultdict(list)
    for subject in subjects:
        groups[subject.stratum].append(subject.subject_id)
    sizes = {key: len(value) for key, value in groups.items()}
    validation_by_stratum = _apportion(sizes, target=validation_count, total=total)
    test_by_stratum = _apportion(sizes, target=test_count, total=total)
    for key, size in sizes.items():
        allocated = validation_by_stratum[key] + test_by_stratum[key]
        if allocated > size:
            raise DatasetProtocolError(
                f"stratum {key!r} has {size} subjects but {allocated} val/test slots"
            )

    rng = np.random.default_rng(seed)
    train: list[str] = []
    validation: list[str] = []
    test: list[str] = []
    for key in sorted(groups, key=str):
        shuffled = [str(value) for value in rng.permutation(sorted(groups[key]))]
        validation_stop = validation_by_stratum[key]
        test_stop = validation_stop + test_by_stratum[key]
        validation.extend(shuffled[:validation_stop])
        test.extend(shuffled[validation_stop:test_stop])
        train.extend(shuffled[test_stop:])

    split = SubjectSplit(
        train=tuple(sorted(train)),
        validation=tuple(sorted(validation)),
        test=tuple(sorted(test)),
    )
    split.validate(expected_total=total)
    if (len(split.train), len(split.validation), len(split.test)) != (
        train_count,
        validation_count,
        test_count,
    ):
        raise DatasetProtocolError("apportioned split did not meet exact target counts")
    return split
