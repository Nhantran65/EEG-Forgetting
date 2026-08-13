from __future__ import annotations

from pathlib import Path

import pytest

from eeg_forgetting.data.contracts import DatasetProtocolError
from eeg_forgetting.data.frozen import FrozenManifestSet
from eeg_forgetting.data.manifests import sha256_file, write_json_once, write_jsonl_once


def _source(root: Path, name: str) -> dict[str, object]:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(name.encode())
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _manifest_set(root: Path) -> Path:
    manifest_dir = root / "manifests" / "v1"
    config = root / "configs" / "channels.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("schema_version: 1\n", encoding="utf-8")
    bci_sources = [
        _source(root, f"raw/A01{session}.{suffix}")
        for session, suffix in (("T", "gdf"), ("T", "mat"), ("E", "gdf"), ("E", "mat"))
    ]
    bci = [
        {
            "dataset": "bciciv2a",
            "subject_id": "A01",
            "split": "train",
            "source_files": bci_sources,
        }
    ]
    physio_sources = [
        _source(root, f"raw/S001/S001R{run:02d}.edf")
        for run in (4, 6, 8, 10, 12, 14)
    ]
    physio = [
        {
            "dataset": "physionet_mi",
            "subject_id": "S001",
            "main_split": "train",
            "reproduction_split": "train",
            "source_files": physio_sources,
        }
    ]
    sleep = [
        {
            "dataset": "sleep_edf_sc",
            "subject_id": "SC00",
            "night_id": "1",
            "recording_id": "SC4001",
            "split": "train",
            "source_files": [
                _source(root, "raw/SC4001E0-PSG.edf"),
                _source(root, "raw/SC4001EC-Hypnogram.edf"),
            ],
        }
    ]
    documents = {"bciciv2a": bci, "physionet_mi": physio, "sleep_edf_sc": sleep}
    entries = {}
    for name, rows in documents.items():
        path = manifest_dir / f"{name}.jsonl"
        digest = write_jsonl_once(path, rows)
        entries[name] = {
            "path": path.relative_to(root).as_posix(),
            "sha256": digest,
            "rows": len(rows),
        }
    index = manifest_dir / "manifest-set.json"
    write_json_once(
        index,
        {
            "schema_version": 1,
            "manifest_version": "test-v1",
            "configs": {"configs/channels.yaml": sha256_file(config)},
            "manifests": entries,
        },
    )
    return index


def test_frozen_manifest_resolves_only_authorized_units(tmp_path: Path) -> None:
    manifests = FrozenManifestSet(_manifest_set(tmp_path), project_root=tmp_path)
    bci = manifests.bci_sessions("train")
    assert [(unit.unit_id, unit.evaluation_label_path is not None) for unit in bci] == [
        ("A01T", False),
        ("A01E", True),
    ]
    assert [unit.run for unit in manifests.physionet_runs("train")] == [4, 6, 8, 10, 12, 14]
    sleep = manifests.sleep_recordings("train")
    assert len(sleep) == 1 and sleep[0].unit_id == "SC00-night-1"
    assert manifests.bci_sessions("test") == ()


def test_frozen_manifest_detects_changed_source_size(tmp_path: Path) -> None:
    index = _manifest_set(tmp_path)
    (tmp_path / "raw" / "A01E.gdf").write_bytes(b"changed source")
    with pytest.raises(DatasetProtocolError, match="source size changed"):
        FrozenManifestSet(index, project_root=tmp_path)


def test_frozen_manifest_detects_manifest_tampering(tmp_path: Path) -> None:
    index = _manifest_set(tmp_path)
    path = tmp_path / "manifests" / "v1" / "bciciv2a.jsonl"
    path.write_text(path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(DatasetProtocolError, match="manifest checksum mismatch"):
        FrozenManifestSet(index, project_root=tmp_path)


def test_frozen_manifest_detects_config_tampering(tmp_path: Path) -> None:
    index = _manifest_set(tmp_path)
    (tmp_path / "configs" / "channels.yaml").write_text(
        "schema_version: changed\n", encoding="utf-8"
    )
    with pytest.raises(DatasetProtocolError, match="config does not match"):
        FrozenManifestSet(index, project_root=tmp_path)
