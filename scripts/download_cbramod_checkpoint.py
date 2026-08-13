#!/usr/bin/env python3
"""Download the pinned official CBraMod checkpoint and verify its identity."""

from __future__ import annotations

import argparse
import os
import urllib.request
from pathlib import Path

from eeg_forgetting.data.contracts import DatasetProtocolError, load_yaml
from eeg_forgetting.data.manifests import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "models" / "cbramod.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    return parser.parse_args()


def main() -> None:
    config = load_yaml(parse_args().config)
    checkpoint = config["checkpoint"]
    destination = PROJECT_ROOT / str(checkpoint["local_path"])
    expected_digest = str(checkpoint["sha256"])
    expected_bytes = int(checkpoint["bytes"])
    if destination.is_file():
        if destination.stat().st_size != expected_bytes or sha256_file(destination) != expected_digest:
            raise DatasetProtocolError(
                f"existing checkpoint does not match pinned identity: {destination}"
            )
        print(f"CBraMod checkpoint already verified: {destination}")
        return

    repository = str(checkpoint["repository"]).rstrip("/")
    revision = str(checkpoint["revision"])
    filename = str(checkpoint["filename"])
    url = f"{repository}/resolve/{revision}/{filename}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    try:
        urllib.request.urlretrieve(url, temporary)
        if temporary.stat().st_size != expected_bytes:
            raise DatasetProtocolError(
                f"checkpoint byte count mismatch: expected {expected_bytes}, got {temporary.stat().st_size}"
            )
        observed = sha256_file(temporary)
        if observed != expected_digest:
            raise DatasetProtocolError(
                f"checkpoint checksum mismatch: expected {expected_digest}, got {observed}"
            )
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(f"Downloaded and verified CBraMod checkpoint: {destination}")


if __name__ == "__main__":
    main()
