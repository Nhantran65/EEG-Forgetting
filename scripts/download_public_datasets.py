#!/usr/bin/env python3
"""Resume-safe downloader for the three public datasets used by this project."""

from __future__ import annotations

import argparse
import shutil
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


PHYSIONET_S3 = "https://physionet-open.s3.amazonaws.com"
BCI_BASE = "https://www.bbci.de"
IMAGERY_RUNS = (4, 6, 8, 10, 12, 14)


@dataclass(frozen=True)
class Download:
    url: str
    destination: Path
    expected_bytes: int | None = None


def download_one(item: Download) -> int:
    item.destination.parent.mkdir(parents=True, exist_ok=True)
    if item.destination.exists():
        size = item.destination.stat().st_size
        if item.expected_bytes is None or size == item.expected_bytes:
            return size
        raise RuntimeError(
            f"existing file has wrong size: {item.destination} "
            f"({size} != {item.expected_bytes})"
        )

    partial = item.destination.with_suffix(item.destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(item.url)
    if offset:
        request.add_header("Range", f"bytes={offset}-")

    with urllib.request.urlopen(request, timeout=180) as response:
        status = getattr(response, "status", 200)
        append = offset > 0 and status == 206
        mode = "ab" if append else "wb"
        if not append:
            offset = 0
        response_bytes = response.headers.get("Content-Length")
        expected_total = offset + int(response_bytes) if response_bytes else None
        with partial.open(mode) as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)

    size = partial.stat().st_size
    expected = item.expected_bytes if item.expected_bytes is not None else expected_total
    if expected is not None and size != expected:
        raise RuntimeError(f"incomplete download: {item.destination} ({size} != {expected})")
    partial.replace(item.destination)
    return size


def bci_downloads(root: Path) -> list[Download]:
    return [
        Download(
            f"{BCI_BASE}/competition/download/competition_iv/BCICIV_2a_gdf.zip",
            root / "downloads/BCICIV_2a_gdf.zip",
            439_968_864,
        ),
        Download(
            f"{BCI_BASE}/competition/iv/results/ds2a/true_labels.zip",
            root / "downloads/BCICIV_2a_true_labels.zip",
            7_197,
        ),
    ]


def physionet_downloads(root: Path) -> list[Download]:
    base = f"{PHYSIONET_S3}/eegmmidb/1.0.0"
    target = root / "physionet_mi/eegmmidb/1.0.0"
    downloads = [
        Download(f"{base}/RECORDS", target / "RECORDS"),
        Download(f"{base}/SHA256SUMS.txt", target / "SHA256SUMS.txt"),
    ]
    for subject in range(1, 110):
        subject_id = f"S{subject:03d}"
        for run in IMAGERY_RUNS:
            filename = f"{subject_id}R{run:02d}.edf"
            downloads.append(
                Download(f"{base}/{subject_id}/{filename}", target / subject_id / filename)
            )
    return downloads


def sleep_downloads(root: Path) -> list[Download]:
    base = f"{PHYSIONET_S3}/sleep-edfx/1.0.0"
    target = root / "sleep_edf/sleep-edfx/1.0.0"
    records = Download(f"{base}/RECORDS", target / "RECORDS")
    checksums = Download(f"{base}/SHA256SUMS.txt", target / "SHA256SUMS.txt")
    download_one(records)
    download_one(checksums)
    cassette_paths = [
        line.split(maxsplit=1)[1].strip()
        for line in checksums.destination.read_text(encoding="utf-8").splitlines()
        if len(line.split(maxsplit=1)) == 2
        and line.split(maxsplit=1)[1].strip().startswith("sleep-cassette/")
        and line.split(maxsplit=1)[1].strip().endswith(".edf")
    ]
    if len(cassette_paths) != 306:
        raise RuntimeError(f"expected 306 Sleep Cassette files, found {len(cassette_paths)}")
    downloads = [records]
    downloads.extend(
        [
            Download(f"{base}/SC-subjects.xls", target / "SC-subjects.xls"),
            checksums,
        ]
    )
    downloads.extend(Download(f"{base}/{path}", target / path) for path in cassette_paths)
    return downloads


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 32:
        parser.error("--workers must be between 1 and 32")

    downloads = [
        *bci_downloads(args.root),
        *physionet_downloads(args.root),
        *sleep_downloads(args.root),
    ]
    completed = 0
    downloaded_bytes = 0
    lock = threading.Lock()
    print(f"Materializing {len(downloads)} files with {args.workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_one, item): item for item in downloads}
        for future in as_completed(futures):
            item = futures[future]
            try:
                size = future.result()
            except Exception as error:
                raise RuntimeError(f"failed: {item.url}") from error
            with lock:
                completed += 1
                downloaded_bytes += size
                if completed % 25 == 0 or completed == len(downloads):
                    gib = downloaded_bytes / 1024**3
                    print(f"[{completed}/{len(downloads)}] materialized {gib:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
