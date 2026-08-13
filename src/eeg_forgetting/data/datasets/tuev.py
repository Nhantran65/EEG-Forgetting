from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

import numpy as np

from ..contracts import DatasetProtocolError, EEGSample
from ..preprocessing import scale_microvolts


class TUEVProcessedDataset:
    """Adapter for output of the pinned CBraMod TUEV preprocessing script."""

    def __init__(self, directory: str | Path, files: Sequence[str | Path] | None = None):
        self.directory = Path(directory)
        self.files = tuple(Path(path) for path in files) if files is not None else tuple(
            sorted(path.relative_to(self.directory) for path in self.directory.glob("*.pkl"))
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> EEGSample:
        relative = self.files[index]
        path = self.directory / relative
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        signal_uv = np.asarray(payload["signal"], dtype=np.float32)
        if signal_uv.shape != (16, 1000):
            raise DatasetProtocolError(f"{path}: expected upstream shape (16, 1000), got {signal_uv.shape}")
        label = int(np.asarray(payload["label"]).reshape(-1)[0]) - 1
        if label not in range(6):
            raise DatasetProtocolError(f"{path}: TUEV label outside 1..6")
        subject = path.name.split("_")[0]
        sample = EEGSample(
            signal=scale_microvolts(signal_uv).reshape(16, 5, 200),
            label=label,
            subject_id=subject,
            recording_id=path.stem.rsplit("-", 1)[0],
            source_id=path.name,
        )
        sample.validate(channels=16, patches=5)
        return sample
