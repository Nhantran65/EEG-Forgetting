from pathlib import Path
import pickle

import numpy as np
import pytest

from eeg_forgetting.data.contracts import DatasetProtocolError
from eeg_forgetting.data.datasets.tuev import TUEVProcessedDataset


def test_pinned_tuev_output_adapter(tmp_path: Path) -> None:
    path = tmp_path / "subject_record-0.pkl"
    with path.open("wb") as handle:
        pickle.dump({"signal": np.full((16, 1000), 100.0), "label": np.array([6])}, handle)
    sample = TUEVProcessedDataset(tmp_path)[0]
    assert sample.signal.shape == (16, 5, 200)
    assert np.all(sample.signal == 1.0)
    assert sample.label == 5


def test_tuev_wrong_shape_fails(tmp_path: Path) -> None:
    path = tmp_path / "subject_record-0.pkl"
    with path.open("wb") as handle:
        pickle.dump({"signal": np.zeros((23, 1000)), "label": np.array([1])}, handle)
    with pytest.raises(DatasetProtocolError, match="expected upstream shape"):
        TUEVProcessedDataset(tmp_path)[0]
