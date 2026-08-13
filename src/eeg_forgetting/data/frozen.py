from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

from .channels import ChannelRegistry
from .contracts import DatasetProtocolError, EEGSample
from .datasets.bciciv2a import BCICIV2aLoader
from .datasets.physionet_mi import IMAGERY_RUNS, PhysioNetMILoader
from .datasets.sleep_edf import SleepEDFLoader, SleepRecording
from .manifests import sha256_file


SplitName = Literal["train", "validation", "test"]
PhysioNetProtocol = Literal["main", "reproduction"]


@dataclass(frozen=True)
class ManifestSource:
    path: Path
    sha256: str
    bytes: int


@dataclass(frozen=True)
class BCISessionUnit:
    subject_id: str
    subject_number: int
    split: SplitName
    session: Literal["T", "E"]
    gdf_path: Path
    evaluation_label_path: Path | None

    @property
    def unit_id(self) -> str:
        return f"{self.subject_id}{self.session}"


@dataclass(frozen=True)
class PhysioNetRunUnit:
    subject_id: str
    subject_number: int
    split: SplitName
    run: int
    edf_path: Path
    protocol: PhysioNetProtocol

    @property
    def unit_id(self) -> str:
        return f"{self.subject_id}R{self.run:02d}"


@dataclass(frozen=True)
class SleepRecordingUnit:
    subject_id: str
    split: SplitName
    recording: SleepRecording

    @property
    def unit_id(self) -> str:
        return f"{self.subject_id}-night-{self.recording.night_id}"


ManifestUnit = BCISessionUnit | PhysioNetRunUnit | SleepRecordingUnit


class FrozenManifestSet:
    """Read-only view of data units authorized by a frozen manifest set."""

    def __init__(
        self,
        manifest_set_path: str | Path,
        *,
        project_root: str | Path | None = None,
        verify_sources: bool = False,
    ):
        self.manifest_set_path = Path(manifest_set_path).resolve()
        self.project_root = (
            Path(project_root).resolve()
            if project_root is not None
            else self.manifest_set_path.parents[2]
        )
        with self.manifest_set_path.open(encoding="utf-8") as handle:
            index = json.load(handle)
        if index.get("schema_version") != 1:
            raise DatasetProtocolError(
                f"unsupported manifest-set schema {index.get('schema_version')}"
            )
        self.version = str(index.get("manifest_version"))
        self._verify_configs(index.get("configs", {}))
        self._rows = self._read_manifests(index.get("manifests", {}))
        self._validate_sources(verify_checksums=verify_sources)

    def _resolve(self, value: str | Path) -> Path:
        path = Path(value)
        return path.resolve() if path.is_absolute() else (self.project_root / path).resolve()

    def _verify_configs(self, configs: dict[str, str]) -> None:
        for name, expected_digest in configs.items():
            # v1/v2 used bare dataset-config names. v3+ stores project-relative
            # paths and also freezes channel and shared preprocessing configs.
            path = (
                self._resolve(name)
                if "/" in name
                else self.project_root / "configs" / "datasets" / name
            )
            if not path.is_file() or sha256_file(path) != expected_digest:
                raise DatasetProtocolError(f"config does not match frozen manifest: {path}")

    def _read_manifests(
        self, manifests: dict[str, dict[str, object]]
    ) -> dict[str, tuple[dict[str, object], ...]]:
        required = {"bciciv2a", "physionet_mi", "sleep_edf_sc"}
        if set(manifests) != required:
            raise DatasetProtocolError(
                f"manifest set must contain exactly {sorted(required)}, got {sorted(manifests)}"
            )
        result: dict[str, tuple[dict[str, object], ...]] = {}
        for dataset, entry in manifests.items():
            path = self._resolve(str(entry["path"]))
            expected_digest = str(entry["sha256"])
            if not path.is_file() or sha256_file(path) != expected_digest:
                raise DatasetProtocolError(f"manifest checksum mismatch: {path}")
            with path.open(encoding="utf-8") as handle:
                rows = tuple(json.loads(line) for line in handle if line.strip())
            if len(rows) != int(entry["rows"]):
                raise DatasetProtocolError(
                    f"{path}: expected {entry['rows']} rows, got {len(rows)}"
                )
            if any(row.get("dataset") != dataset for row in rows):
                raise DatasetProtocolError(f"{path}: row dataset does not match {dataset}")
            result[dataset] = rows
        return result

    def _sources(self, row: dict[str, object]) -> tuple[ManifestSource, ...]:
        sources = tuple(
            ManifestSource(
                path=self._resolve(str(source["path"])),
                sha256=str(source["sha256"]),
                bytes=int(source["bytes"]),
            )
            for source in row["source_files"]
        )
        if not sources:
            raise DatasetProtocolError(f"manifest row has no sources: {row}")
        return sources

    def _validate_sources(self, *, verify_checksums: bool) -> None:
        seen: dict[Path, ManifestSource] = {}
        for rows in self._rows.values():
            for row in rows:
                for source in self._sources(row):
                    previous = seen.get(source.path)
                    if previous is not None and previous != source:
                        raise DatasetProtocolError(
                            f"inconsistent manifest metadata for {source.path}"
                        )
                    seen[source.path] = source
        for source in seen.values():
            if not source.path.is_file():
                raise DatasetProtocolError(f"manifest source is missing: {source.path}")
            if source.path.stat().st_size != source.bytes:
                raise DatasetProtocolError(f"manifest source size changed: {source.path}")
            if verify_checksums and sha256_file(source.path) != source.sha256:
                raise DatasetProtocolError(f"manifest source checksum changed: {source.path}")

    @staticmethod
    def _validate_split(split: str) -> SplitName:
        if split not in {"train", "validation", "test"}:
            raise DatasetProtocolError(f"unknown split {split!r}")
        return split  # type: ignore[return-value]

    def bci_sessions(self, split: SplitName) -> tuple[BCISessionUnit, ...]:
        split = self._validate_split(split)
        units: list[BCISessionUnit] = []
        for row in self._rows["bciciv2a"]:
            if row["split"] != split:
                continue
            subject_id = str(row["subject_id"])
            sources = {source.path.name: source.path for source in self._sources(row)}
            for session in ("T", "E"):
                gdf_name = f"{subject_id}{session}.gdf"
                if gdf_name not in sources:
                    raise DatasetProtocolError(f"{subject_id}: missing {gdf_name} in manifest")
                labels = sources.get(f"{subject_id}E.mat") if session == "E" else None
                if session == "E" and labels is None:
                    raise DatasetProtocolError(f"{subject_id}: evaluation labels absent from manifest")
                units.append(
                    BCISessionUnit(
                        subject_id=subject_id,
                        subject_number=int(subject_id[1:]),
                        split=split,
                        session=session,
                        gdf_path=sources[gdf_name],
                        evaluation_label_path=labels,
                    )
                )
        return tuple(units)

    def physionet_runs(
        self,
        split: SplitName,
        *,
        protocol: PhysioNetProtocol = "main",
    ) -> tuple[PhysioNetRunUnit, ...]:
        split = self._validate_split(split)
        if protocol not in {"main", "reproduction"}:
            raise DatasetProtocolError(f"unknown PhysioNet protocol {protocol!r}")
        field = "main_split" if protocol == "main" else "reproduction_split"
        units: list[PhysioNetRunUnit] = []
        for row in self._rows["physionet_mi"]:
            if row[field] != split:
                continue
            subject_id = str(row["subject_id"])
            for source in self._sources(row):
                match = re.fullmatch(rf"{subject_id}R(?P<run>\d{{2}})\.edf", source.path.name)
                if not match:
                    raise DatasetProtocolError(
                        f"{subject_id}: unexpected PhysioNet source {source.path.name}"
                    )
                units.append(
                    PhysioNetRunUnit(
                        subject_id=subject_id,
                        subject_number=int(subject_id[1:]),
                        split=split,
                        run=int(match.group("run")),
                        edf_path=source.path,
                        protocol=protocol,
                    )
                )
        result = tuple(sorted(units, key=lambda unit: (unit.subject_id, unit.run)))
        by_subject: dict[str, set[int]] = {}
        for unit in result:
            by_subject.setdefault(unit.subject_id, set()).add(unit.run)
        invalid = {
            subject: sorted(runs)
            for subject, runs in by_subject.items()
            if runs != set(IMAGERY_RUNS)
        }
        if invalid:
            raise DatasetProtocolError(
                f"PhysioNet manifest does not contain the six imagery runs: {invalid}"
            )
        return result

    def sleep_recordings(self, split: SplitName) -> tuple[SleepRecordingUnit, ...]:
        split = self._validate_split(split)
        units: list[SleepRecordingUnit] = []
        for row in self._rows["sleep_edf_sc"]:
            if row["split"] != split:
                continue
            sources = self._sources(row)
            psg = [source.path for source in sources if source.path.name.endswith("-PSG.edf")]
            hyp = [
                source.path
                for source in sources
                if source.path.name.endswith("-Hypnogram.edf")
            ]
            if len(psg) != 1 or len(hyp) != 1:
                raise DatasetProtocolError(
                    f"{row['recording_id']}: expected one PSG and one hypnogram"
                )
            subject_id = str(row["subject_id"])
            recording = SleepRecording(
                subject_id=subject_id,
                night_id=str(row["night_id"]),
                psg_path=psg[0],
                hypnogram_path=hyp[0],
            )
            units.append(SleepRecordingUnit(subject_id, split, recording))
        return tuple(units)


class ManifestEEGLoader:
    """Dispatch frozen manifest units through the audited raw-data loaders."""

    def __init__(self, manifests: FrozenManifestSet, registry: ChannelRegistry):
        self.manifests = manifests
        self._bci = BCICIV2aLoader(registry)
        self._physio_main = PhysioNetMILoader(registry)
        self._physio_reproduction = PhysioNetMILoader(
            registry, allow_excluded_for_reproduction=True
        )
        self._sleep = SleepEDFLoader(registry)

    def units(
        self,
        dataset: str,
        split: SplitName,
        *,
        physionet_protocol: PhysioNetProtocol = "main",
    ) -> tuple[ManifestUnit, ...]:
        if dataset == "bciciv2a":
            return self.manifests.bci_sessions(split)
        if dataset == "physionet_mi":
            return self.manifests.physionet_runs(split, protocol=physionet_protocol)
        if dataset == "sleep_edf_sc":
            return self.manifests.sleep_recordings(split)
        raise DatasetProtocolError(f"unknown frozen dataset {dataset!r}")

    def load_unit(self, unit: ManifestUnit) -> list[EEGSample]:
        if isinstance(unit, BCISessionUnit):
            samples = self._bci.load_session(
                unit.gdf_path,
                subject_id=unit.subject_id,
                session=unit.session,
                evaluation_label_path=unit.evaluation_label_path,
            )
            expected_shape = (22, 4, 200)
        elif isinstance(unit, PhysioNetRunUnit):
            loader = (
                self._physio_reproduction
                if unit.protocol == "reproduction"
                else self._physio_main
            )
            samples = loader.load_run(
                unit.edf_path,
                subject_number=unit.subject_number,
                run=unit.run,
            )
            expected_shape = (22, 4, 200)
        elif isinstance(unit, SleepRecordingUnit):
            samples = self._sleep.load_recording(unit.recording)
            expected_shape = (2, 30, 200)
        else:
            raise TypeError(f"unsupported manifest unit {type(unit)!r}")
        for sample in samples:
            sample.validate(
                channels=expected_shape[0],
                patches=expected_shape[1],
                points=expected_shape[2],
            )
            if sample.subject_id != unit.subject_id:
                raise DatasetProtocolError(
                    f"{unit.unit_id}: emitted sample for {sample.subject_id}"
                )
        return samples

    def iter_samples(
        self,
        dataset: str,
        split: SplitName,
        *,
        physionet_protocol: PhysioNetProtocol = "main",
        max_units: int | None = None,
    ) -> Iterator[EEGSample]:
        units = self.units(dataset, split, physionet_protocol=physionet_protocol)
        selected = units if max_units is None else units[:max_units]
        for unit in selected:
            yield from self.load_unit(unit)
