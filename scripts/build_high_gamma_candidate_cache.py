#!/usr/bin/env python3
"""Build the locked High-Gamma replacement-candidate cache."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from eeg_forgetting.data.cache import (
    inspect_unit_cache,
    write_cache_index,
    write_unit_cache,
)
from eeg_forgetting.data.channels import ChannelRegistry
from eeg_forgetting.data.contracts import DatasetProtocolError, load_yaml
from eeg_forgetting.data.datasets.high_gamma import HighGammaLoader
from eeg_forgetting.data.manifests import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class HighGammaUnit:
    subject_id: str
    split: str
    acquisition: str
    bdf_path: Path
    events_path: Path
    protocol: str = "main"

    @property
    def unit_id(self) -> str:
        return f"{self.subject_id}-{self.acquisition}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "datasets" / "high_gamma.yaml",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "high_gamma" / "nm000172-v1.0.2",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "high_gamma_candidate_v1",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation", "test"),
        default=("train", "validation"),
    )
    parser.add_argument("--verify-source-checksums", action="store_true")
    return parser.parse_args()


def _units(config: dict[str, object], raw_root: Path, split: str) -> list[HighGammaUnit]:
    units = []
    for number in config["subject_split"][split]:
        subject_id = f"sub-{int(number)}"
        directory = raw_root / subject_id / "ses-0" / "eeg"
        for acquisition, run in (("train", 0), ("test", 1)):
            stem = f"{subject_id}_ses-0_task-imagery_acq-{acquisition}_run-{run}"
            units.append(
                HighGammaUnit(
                    subject_id=subject_id,
                    split=split,
                    acquisition=acquisition,
                    bdf_path=directory / f"{stem}_eeg.bdf",
                    events_path=directory / f"{stem}_events.tsv",
                )
            )
    return units


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if config.get("status") != "locked_replacement_candidate":
        raise DatasetProtocolError("High-Gamma candidate config is not locked")
    manifest_path = PROJECT_ROOT / str(config["source_pin"]["manifest"])
    if sha256_file(manifest_path) != config["source_pin"]["manifest_sha256"]:
        raise DatasetProtocolError("High-Gamma source manifest changed")
    with manifest_path.open(encoding="utf-8") as handle:
        source_manifest = {row["path"]: row for row in json.load(handle)}
    registry = ChannelRegistry.from_yaml(PROJECT_ROOT / "configs" / "channels.yaml")
    loader = HighGammaLoader(
        registry,
        maximum_class_count_difference=int(
            config["assertions"]["maximum_class_count_difference"]
        ),
    )
    summaries = []
    for split in args.splits:
        units = _units(config, args.raw_root, split)
        shards = []
        for number, unit in enumerate(units, start=1):
            relative = unit.bdf_path.relative_to(args.raw_root).as_posix()
            source = source_manifest.get(relative)
            if source is None or not unit.bdf_path.is_file() or not unit.events_path.is_file():
                raise DatasetProtocolError(f"missing High-Gamma source unit {unit.unit_id}")
            if unit.bdf_path.stat().st_size != int(source["size"]):
                raise DatasetProtocolError(f"High-Gamma source size mismatch {unit.bdf_path}")
            if args.verify_source_checksums and sha256_file(unit.bdf_path) != source["checksum"]:
                raise DatasetProtocolError(f"High-Gamma source digest mismatch {unit.bdf_path}")
            cached = inspect_unit_cache(
                args.cache_root,
                dataset="high_gamma",
                split=split,
                unit=unit,
            )
            if cached is None:
                samples = loader.load_run(
                    unit.bdf_path,
                    unit.events_path,
                    subject_id=unit.subject_id,
                    acquisition=unit.acquisition,
                )
                cached = write_unit_cache(
                    args.cache_root,
                    dataset="high_gamma",
                    split=split,
                    unit=unit,
                    samples=samples,
                )
                print(
                    f"[{split}] built {number}/{len(units)} {unit.unit_id}: "
                    f"{cached['samples']} samples",
                    flush=True,
                )
            shards.append(cached)
        index = write_cache_index(
            args.cache_root,
            dataset="high_gamma",
            split=split,
            manifest_set_path=args.config,
            manifest_version="high_gamma_candidate_v1",
            shards=shards,
        )
        summaries.append(
            {
                "split": split,
                "subjects": len(config["subject_split"][split]),
                "units": len(units),
                "samples": sum(int(shard["samples"]) for shard in shards),
                "index": str(index),
            }
        )
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
