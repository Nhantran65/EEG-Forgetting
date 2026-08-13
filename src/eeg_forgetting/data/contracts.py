from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np
import yaml


class DatasetProtocolError(ValueError):
    """Raised when source data violates a predeclared dataset contract."""


@dataclass(frozen=True)
class EEGSample:
    signal: np.ndarray
    label: int
    subject_id: str
    recording_id: str
    source_id: str

    def validate(self, *, channels: int, patches: int, points: int = 200) -> None:
        expected = (channels, patches, points)
        if self.signal.shape != expected:
            raise DatasetProtocolError(
                f"{self.source_id}: expected signal shape {expected}, got {self.signal.shape}"
            )
        if not np.isfinite(self.signal).all():
            raise DatasetProtocolError(f"{self.source_id}: signal contains non-finite values")


@dataclass(frozen=True)
class SubjectSplit:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]

    def validate(self, expected_total: int | None = None) -> None:
        groups = {
            "train": set(self.train),
            "validation": set(self.validation),
            "test": set(self.test),
        }
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            overlap = groups[left] & groups[right]
            if overlap:
                raise DatasetProtocolError(
                    f"subject leakage between {left} and {right}: {sorted(overlap)}"
                )
        total = sum(len(group) for group in groups.values())
        if expected_total is not None and total != expected_total:
            raise DatasetProtocolError(f"expected {expected_total} subjects, got {total}")


def load_yaml(path: str | Path) -> Mapping[str, object]:
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise DatasetProtocolError(f"{path}: YAML root must be a mapping")
    return value


def assert_class_counts(
    labels: Sequence[int] | np.ndarray,
    *,
    expected_classes: int,
    expected_per_class: int | None = None,
    context: str,
) -> dict[int, int]:
    classes, counts = np.unique(np.asarray(labels, dtype=int), return_counts=True)
    observed = {int(label): int(count) for label, count in zip(classes, counts)}
    if len(observed) != expected_classes:
        raise DatasetProtocolError(
            f"{context}: expected {expected_classes} classes, got {observed}"
        )
    if expected_per_class is not None and set(observed.values()) != {expected_per_class}:
        raise DatasetProtocolError(
            f"{context}: expected {expected_per_class} trials/class, got {observed}"
        )
    return observed


def fixed_step_batches(
    population_size: int,
    *,
    batch_size: int,
    optimizer_steps: int,
    seed: int,
) -> Iterator[np.ndarray]:
    """Yield shuffled, cycling indices for an exact optimizer-step budget."""
    if population_size <= 0 or batch_size <= 0 or optimizer_steps <= 0:
        raise DatasetProtocolError("population_size, batch_size, and optimizer_steps must be positive")
    rng = np.random.default_rng(seed)
    order = rng.permutation(population_size)
    cursor = 0
    for _ in range(optimizer_steps):
        batch = np.empty(batch_size, dtype=np.int64)
        filled = 0
        while filled < batch_size:
            remaining = population_size - cursor
            take = min(batch_size - filled, remaining)
            batch[filled : filled + take] = order[cursor : cursor + take]
            cursor += take
            filled += take
            if cursor == population_size:
                order = rng.permutation(population_size)
                cursor = 0
        yield batch


def expand_ranges(ranges: Iterable[Sequence[int]]) -> tuple[int, ...]:
    values: list[int] = []
    for item in ranges:
        if len(item) != 2:
            raise DatasetProtocolError(f"invalid inclusive range: {item}")
        start, end = int(item[0]), int(item[1])
        if end < start:
            raise DatasetProtocolError(f"descending range: {item}")
        values.extend(range(start, end + 1))
    return tuple(values)
