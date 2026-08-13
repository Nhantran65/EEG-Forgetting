from pathlib import Path

import pytest

from eeg_forgetting.data.contracts import DatasetProtocolError
from eeg_forgetting.data.manifests import write_json_once, write_jsonl_once


def test_manifest_is_create_only(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    digest = write_jsonl_once(path, [{"subject": "S001"}])
    assert len(digest) == 64
    with pytest.raises(DatasetProtocolError, match="refusing to overwrite"):
        write_jsonl_once(path, [{"subject": "S002"}])


def test_json_manifest_is_create_only(tmp_path: Path) -> None:
    path = tmp_path / "manifest-set.json"
    digest = write_json_once(path, {"schema_version": 1})
    assert len(digest) == 64
    with pytest.raises(DatasetProtocolError, match="refusing to overwrite"):
        write_json_once(path, {"schema_version": 2})
