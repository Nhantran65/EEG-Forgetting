from __future__ import annotations

import bisect
import json
import os
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .contracts import DatasetProtocolError, EEGSample
from .frozen import ManifestUnit
from .manifests import sha256_file


CACHE_SCHEMA_VERSION = 1


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _expected_sample_shape(dataset: str, protocol: str = "main") -> tuple[int, int, int]:
    if dataset == "physionet_mi" and protocol == "reproduction":
        return (64, 4, 200)
    shapes = {
        "bciciv2a": (22, 4, 200),
        "high_gamma": (22, 4, 200),
        "physionet_mi": (22, 4, 200),
        "sleep_edf_sc": (2, 30, 200),
    }
    if dataset not in shapes:
        raise DatasetProtocolError(f"unknown cache dataset {dataset!r}")
    return shapes[dataset]


def write_unit_cache(
    root: str | Path,
    *,
    dataset: str,
    split: str,
    unit: ManifestUnit,
    samples: Sequence[EEGSample],
    protocol: str = "main",
) -> dict[str, object]:
    root = Path(root)
    if not samples:
        raise DatasetProtocolError(f"{unit.unit_id}: cannot cache zero samples")
    expected_shape = _expected_sample_shape(dataset, protocol)
    if any(sample.signal.shape != expected_shape for sample in samples):
        raise DatasetProtocolError(f"{unit.unit_id}: inconsistent sample shape in cache input")
    if any(sample.subject_id != unit.subject_id for sample in samples):
        raise DatasetProtocolError(f"{unit.unit_id}: cache input crosses subjects")
    signals = np.stack([sample.signal for sample in samples]).astype(np.float32, copy=False)
    labels = np.asarray([sample.label for sample in samples], dtype=np.int64)
    if not np.isfinite(signals).all():
        raise DatasetProtocolError(f"{unit.unit_id}: non-finite signal in cache input")
    directory = root / dataset / split
    signal_path = directory / f"{unit.unit_id}.signals.npy"
    label_path = directory / f"{unit.unit_id}.labels.npy"
    if signal_path.exists() or label_path.exists():
        raise DatasetProtocolError(f"refusing to overwrite cache shard {unit.unit_id}")
    _atomic_save_npy(signal_path, signals)
    try:
        _atomic_save_npy(label_path, labels)
    except BaseException:
        signal_path.unlink(missing_ok=True)
        raise
    return {
        "unit_id": unit.unit_id,
        "subject_id": unit.subject_id,
        "samples": len(samples),
        "sample_shape": list(expected_shape),
        "signal_path": signal_path.relative_to(root).as_posix(),
        "label_path": label_path.relative_to(root).as_posix(),
        "signal_bytes": signal_path.stat().st_size,
        "label_bytes": label_path.stat().st_size,
        "label_counts": dict(sorted(Counter(int(value) for value in labels).items())),
    }


def inspect_unit_cache(
    root: str | Path,
    *,
    dataset: str,
    split: str,
    unit: ManifestUnit,
    protocol: str = "main",
) -> dict[str, object] | None:
    root = Path(root)
    directory = root / dataset / split
    signal_path = directory / f"{unit.unit_id}.signals.npy"
    label_path = directory / f"{unit.unit_id}.labels.npy"
    if not signal_path.exists() and not label_path.exists():
        return None
    if not signal_path.is_file() or not label_path.is_file():
        raise DatasetProtocolError(f"incomplete cache shard {unit.unit_id}")
    signals = np.load(signal_path, mmap_mode="r", allow_pickle=False)
    labels = np.load(label_path, mmap_mode="r", allow_pickle=False)
    expected_shape = _expected_sample_shape(dataset, protocol)
    if signals.dtype != np.float32 or signals.ndim != 4 or tuple(signals.shape[1:]) != expected_shape:
        raise DatasetProtocolError(f"invalid cached signal shard {signal_path}: {signals.shape}")
    if labels.dtype != np.int64 or labels.shape != (signals.shape[0],):
        raise DatasetProtocolError(f"invalid cached label shard {label_path}: {labels.shape}")
    if signals.shape[0] == 0:
        raise DatasetProtocolError(f"empty cache shard {unit.unit_id}")
    return {
        "unit_id": unit.unit_id,
        "subject_id": unit.subject_id,
        "samples": int(signals.shape[0]),
        "sample_shape": list(expected_shape),
        "signal_path": signal_path.relative_to(root).as_posix(),
        "label_path": label_path.relative_to(root).as_posix(),
        "signal_bytes": signal_path.stat().st_size,
        "label_bytes": label_path.stat().st_size,
        "label_counts": dict(
            sorted(Counter(int(value) for value in np.asarray(labels)).items())
        ),
    }


def write_cache_index(
    root: str | Path,
    *,
    dataset: str,
    split: str,
    manifest_set_path: str | Path,
    manifest_version: str,
    shards: Sequence[dict[str, object]],
    protocol: str = "main",
) -> Path:
    root = Path(root)
    path = root / dataset / split / "index.json"
    document = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "dataset": dataset,
        "split": split,
        "protocol": protocol,
        "manifest_version": manifest_version,
        "manifest_set_path": str(Path(manifest_set_path).resolve()),
        "manifest_set_sha256": sha256_file(manifest_set_path),
        "samples": sum(int(shard["samples"]) for shard in shards),
        "subjects": len({str(shard["subject_id"]) for shard in shards}),
        "units": len(shards),
        "shards": list(shards),
    }
    temporary = path.with_suffix(".json.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return path


class CachedEEGDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    def __init__(
        self,
        index_path: str | Path,
        *,
        verify_manifest: bool = True,
    ):
        self.index_path = Path(index_path).resolve()
        self.root = self.index_path.parents[2]
        with self.index_path.open(encoding="utf-8") as handle:
            self.index = json.load(handle)
        if self.index.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise DatasetProtocolError(f"unsupported cache schema in {self.index_path}")
        if verify_manifest:
            manifest_path = Path(str(self.index["manifest_set_path"]))
            if not manifest_path.is_file() or sha256_file(manifest_path) != self.index["manifest_set_sha256"]:
                raise DatasetProtocolError("processed cache does not match its frozen manifest")
        self.shards = tuple(self.index["shards"])
        self._ends: list[int] = []
        self._signals: list[np.ndarray] = []
        self._labels: list[np.ndarray] = []
        total = 0
        for shard in self.shards:
            signal_path = self.root / str(shard["signal_path"])
            label_path = self.root / str(shard["label_path"])
            signals = np.load(signal_path, mmap_mode="r", allow_pickle=False)
            labels = np.load(label_path, mmap_mode="r", allow_pickle=False)
            expected = (int(shard["samples"]), *map(int, shard["sample_shape"]))
            if signals.shape != expected or labels.shape != (expected[0],):
                raise DatasetProtocolError(f"cached shard shape changed: {shard['unit_id']}")
            self._signals.append(signals)
            self._labels.append(labels)
            total += expected[0]
            self._ends.append(total)
        if total != int(self.index["samples"]):
            raise DatasetProtocolError(f"cache index sample total mismatch: {self.index_path}")

    def __len__(self) -> int:
        return self._ends[-1] if self._ends else 0

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard_index = bisect.bisect_right(self._ends, index)
        start = 0 if shard_index == 0 else self._ends[shard_index - 1]
        local = index - start
        signal = torch.from_numpy(np.array(self._signals[shard_index][local], copy=True))
        label = torch.tensor(int(self._labels[shard_index][local]), dtype=torch.long)
        return signal, label, str(self.shards[shard_index]["subject_id"])
