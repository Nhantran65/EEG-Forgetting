from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Mapping

from .contracts import DatasetProtocolError


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl_once(path: str | Path, rows: Iterable[Mapping[str, object]]) -> str:
    """Atomically create, never overwrite, a paper-facing JSONL manifest."""
    path = Path(path)
    if path.exists():
        raise DatasetProtocolError(f"refusing to overwrite immutable manifest {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return sha256_file(path)


def write_json_once(path: str | Path, value: Mapping[str, object]) -> str:
    """Atomically create, never overwrite, a paper-facing JSON document."""
    path = Path(path)
    if path.exists():
        raise DatasetProtocolError(f"refusing to overwrite immutable manifest {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return sha256_file(path)
