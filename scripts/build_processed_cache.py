#!/usr/bin/env python3
"""Materialize resumable per-unit preprocessing caches from frozen manifests."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from eeg_forgetting.data.cache import inspect_unit_cache, write_cache_index, write_unit_cache
from eeg_forgetting.data.channels import ChannelRegistry
from eeg_forgetting.data.frozen import FrozenManifestSet, ManifestEEGLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_WORKER_LOADER: ManifestEEGLoader | None = None
_WORKER_CACHE_ROOT: Path | None = None


def _initialize_worker(manifest_set: str, cache_root: str) -> None:
    global _WORKER_LOADER, _WORKER_CACHE_ROOT
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    manifests = FrozenManifestSet(manifest_set, project_root=PROJECT_ROOT)
    registry = ChannelRegistry.from_yaml(PROJECT_ROOT / "configs" / "channels.yaml")
    _WORKER_LOADER = ManifestEEGLoader(manifests, registry)
    _WORKER_CACHE_ROOT = Path(cache_root)


def _build_unit(dataset: str, split: str, unit) -> dict[str, object]:
    if _WORKER_LOADER is None or _WORKER_CACHE_ROOT is None:
        raise RuntimeError("cache worker was not initialized")
    samples = _WORKER_LOADER.load_unit(unit)
    return write_unit_cache(
        _WORKER_CACHE_ROOT,
        dataset=dataset,
        split=split,
        unit=unit,
        samples=samples,
        protocol=str(getattr(unit, "protocol", "main")),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-set",
        type=Path,
        default=PROJECT_ROOT / "manifests" / "v3" / "manifest-set.json",
    )
    parser.add_argument(
        "--cache-root", type=Path
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("bciciv2a", "physionet_mi", "sleep_edf_sc"),
        default=("bciciv2a", "physionet_mi", "sleep_edf_sc"),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation", "test"),
        default=("train", "validation"),
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--physionet-protocol", choices=("main", "reproduction"), default="main"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.physionet_protocol == "reproduction" and args.datasets != ["physionet_mi"]:
        raise ValueError("reproduction cache must contain only physionet_mi")
    cache_root = args.cache_root or (
        PROJECT_ROOT / "data" / "processed" / "v3-physionet-cbramod-reproduction"
        if args.physionet_protocol == "reproduction"
        else PROJECT_ROOT / "data" / "processed" / "v3"
    )
    manifests = FrozenManifestSet(args.manifest_set, project_root=PROJECT_ROOT)
    registry = ChannelRegistry.from_yaml(PROJECT_ROOT / "configs" / "channels.yaml")
    loader = ManifestEEGLoader(manifests, registry)
    summaries = []
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    executor = (
        ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_initialize_worker,
            initargs=(str(args.manifest_set.resolve()), str(cache_root.resolve())),
        )
        if args.workers > 1
        else None
    )
    try:
        for dataset in args.datasets:
            for split in args.splits:
                units = loader.units(
                    dataset,
                    split,
                    physionet_protocol=args.physionet_protocol,
                )
                by_unit: dict[str, dict[str, object]] = {}
                missing = []
                for unit in units:
                    shard = inspect_unit_cache(
                        cache_root,
                        dataset=dataset,
                        split=split,
                        unit=unit,
                        protocol=str(getattr(unit, "protocol", "main")),
                    )
                    if shard is None:
                        missing.append(unit)
                    else:
                        by_unit[unit.unit_id] = shard
                print(
                    f"[{dataset}/{split}] {len(by_unit)} cached, "
                    f"{len(missing)} to build with {args.workers} worker(s)",
                    flush=True,
                )
                if executor is None:
                    _initialize_worker(str(args.manifest_set.resolve()), str(cache_root.resolve()))
                    for number, unit in enumerate(missing, start=1):
                        shard = _build_unit(dataset, split, unit)
                        by_unit[unit.unit_id] = shard
                        print(
                            f"[{dataset}/{split}] built {number}/{len(missing)} "
                            f"{unit.unit_id}: {shard['samples']} samples",
                            flush=True,
                        )
                else:
                    futures = {
                        executor.submit(_build_unit, dataset, split, unit): unit
                        for unit in missing
                    }
                    for number, future in enumerate(as_completed(futures), start=1):
                        unit = futures[future]
                        shard = future.result()
                        by_unit[unit.unit_id] = shard
                        print(
                            f"[{dataset}/{split}] built {number}/{len(missing)} "
                            f"{unit.unit_id}: {shard['samples']} samples",
                            flush=True,
                        )
                shards = [by_unit[unit.unit_id] for unit in units]
                index_path = write_cache_index(
                    cache_root,
                    dataset=dataset,
                    split=split,
                    manifest_set_path=args.manifest_set,
                    manifest_version=manifests.version,
                    shards=shards,
                    protocol=(
                        args.physionet_protocol
                        if dataset == "physionet_mi"
                        else "main"
                    ),
                )
                summaries.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "units": len(shards),
                        "samples": sum(int(shard["samples"]) for shard in shards),
                        "index": str(index_path),
                    }
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
