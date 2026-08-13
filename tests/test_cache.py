from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from eeg_forgetting.data.cache import CachedEEGDataset, write_cache_index, write_unit_cache
from eeg_forgetting.data.contracts import EEGSample


def test_cached_dataset_preserves_signal_label_and_subject(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest-set.json"
    manifest.write_text('{"schema_version": 1}\n', encoding="utf-8")
    unit = SimpleNamespace(unit_id="A01T", subject_id="A01")
    samples = [
        EEGSample(
            signal=np.full((22, 4, 200), index, dtype=np.float32),
            label=index,
            subject_id="A01",
            recording_id="A01-T",
            source_id=str(index),
        )
        for index in range(2)
    ]
    shard = write_unit_cache(
        tmp_path / "cache",
        dataset="bciciv2a",
        split="train",
        unit=unit,
        samples=samples,
    )
    index = write_cache_index(
        tmp_path / "cache",
        dataset="bciciv2a",
        split="train",
        manifest_set_path=manifest,
        manifest_version="test",
        shards=[shard],
    )
    dataset = CachedEEGDataset(index)
    signal, label, subject = dataset[1]
    assert len(dataset) == 2
    assert signal.shape == (22, 4, 200)
    assert torch.all(signal == 1)
    assert label.item() == 1
    assert subject == "A01"
    assert dataset.index["protocol"] == "main"


def test_reproduction_cache_accepts_upstream_64_channel_shape(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest-set.json"
    manifest.write_text('{"schema_version": 1}\n', encoding="utf-8")
    unit = SimpleNamespace(unit_id="S001R04", subject_id="S001")
    sample = EEGSample(
        signal=np.zeros((64, 4, 200), dtype=np.float32),
        label=0,
        subject_id="S001",
        recording_id="S001-R04",
        source_id="S001R04:event-00",
    )

    shard = write_unit_cache(
        tmp_path / "cache",
        dataset="physionet_mi",
        split="train",
        unit=unit,
        samples=[sample],
        protocol="reproduction",
    )
    index = write_cache_index(
        tmp_path / "cache",
        dataset="physionet_mi",
        split="train",
        manifest_set_path=manifest,
        manifest_version="test",
        shards=[shard],
        protocol="reproduction",
    )

    signal, _label, _subject = CachedEEGDataset(index)[0]
    assert signal.shape == (64, 4, 200)
