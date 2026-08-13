from pathlib import Path

from eeg_forgetting.data.contracts import load_yaml


ROOT = Path(__file__).parents[1]


def test_shared_preprocessing_and_training_budget_are_locked() -> None:
    config = load_yaml(ROOT / "configs/common.yaml")
    assert config["target_sampling_rate_hz"] == 200
    assert config["bandpass_hz"] == [0.5, 40.0]
    assert config["amplitude"]["divisor"] == 100.0
    assert config["amplitude"]["fit_statistics"] is False
    assert config["training"]["budget_matched"]["optimizer_steps"] == 2500
    assert config["training"]["converged"]["early_stopping"] is True


def test_dataset_inclusion_status_is_explicit() -> None:
    dataset_dir = ROOT / "configs/datasets"
    assert load_yaml(dataset_dir / "bciciv2a.yaml")["status"] == "main"
    assert load_yaml(dataset_dir / "physionet_mi.yaml")["status"] == "main"
    assert load_yaml(dataset_dir / "sleep_edf.yaml")["status"] == "main"
    assert load_yaml(dataset_dir / "tuev.yaml")["status"] == "access_pending_main_candidate"
    assert load_yaml(dataset_dir / "vep.yaml")["status"] == "quarantined_exploratory"


def test_frozen_subject_splits_are_locked_in_config() -> None:
    dataset_dir = ROOT / "configs/datasets"
    bci_split = load_yaml(dataset_dir / "bciciv2a.yaml")["split"]["main"]
    assert bci_split == {
        "name": "fixed_subject_holdout_v1",
        "train": ["A01", "A02", "A03", "A04", "A05"],
        "validation": ["A06", "A07"],
        "test": ["A08", "A09"],
    }

    sleep_split = load_yaml(dataset_dir / "sleep_edf.yaml")["split"]
    assert sleep_split["target_subject_counts"] == {
        "train": 48,
        "validation": 15,
        "test": 15,
    }
    assert sleep_split["stratify_fields"] == ["age", "sex"]
    assert sleep_split["seed"] == 20260813


def test_tuev_contract_uses_pinned_16_channel_output() -> None:
    config = load_yaml(ROOT / "configs/datasets/tuev.yaml")
    assert config["preprocessing_authority"]["commit"] == (
        "b9e961003214326972c567eff390e75b0287e32a"
    )
    assert config["output_bipolar_channels"] == 16
    assert config["assertions"]["sample_shape"] == [16, 5, 200]
    assert config["preprocessing_modes"]["reference_reproduction"]["raw_bandpass_hz"] == [
        0.3,
        75.0,
    ]
    assert config["preprocessing_modes"]["main_harmonized"]["raw_bandpass_hz"] == [
        0.5,
        40.0,
    ]
    assert config["normalization"]["loader_divisor"] == 100.0


def test_cbramod_code_and_checkpoint_identity_are_pinned() -> None:
    config = load_yaml(ROOT / "configs" / "models" / "cbramod.yaml")
    assert config["upstream"]["commit"] == (
        "b9e961003214326972c567eff390e75b0287e32a"
    )
    assert config["checkpoint"]["revision"] == (
        "500543c7e30bda1b22bfd51a49301b238dee21fd"
    )
    assert config["checkpoint"]["sha256"] == (
        "0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178"
    )
    assert config["checkpoint"]["load_mode"] == "torch_weights_only_strict"
    assert config["integration_probe"]["main_experiment_head_locked"] is False
