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
    assert config["integration_probe"]["main_experiment_head_locked"] is True
    assert config["main_task_head"]["name"] == "flatten_mlp"
    assert config["plasticity"]["selected_final_encoder_blocks"] == 4
    assert config["plasticity"]["frozen_backbone_mode"] == "eval"


def test_depth_v3_and_linear_probe_controls_are_explicit() -> None:
    pilots = ROOT / "configs" / "pilots"
    probe = load_yaml(pilots / "cbramod_linear_probe.yaml")
    sweep = load_yaml(pilots / "cbramod_depth_v3.yaml")
    invalid = load_yaml(pilots / "cbramod_depth_v2.yaml")

    assert probe["head"] == "flatten_linear"
    assert probe["depths"] == [0]
    assert sweep["frozen_backbone_mode"] == "eval"
    assert sweep["selection"]["candidate_depths"] == [1, 2, 4, 8]
    assert invalid["status"] == "invalid_frozen_backbone_dropout_confounded"


def test_budget_single_task_checkpoint_roles_are_locked() -> None:
    config = load_yaml(ROOT / "configs" / "training" / "single_task_budget.yaml")
    assert config["mode"] == "budget_matched"
    assert config["plastic_final_encoder_blocks"] == 4
    assert config["schedule"]["optimizer_steps"] == 2500
    assert config["checkpoints"]["best_validation"] == "diagnostic_only"
    assert config["checkpoints"]["exact_final_step"] == (
        "fisher_overlap_and_budget_matched_comparison"
    )


def test_explanation_drift_pilot_is_validation_only_and_digest_locked() -> None:
    config = load_yaml(
        ROOT / "configs" / "xai" / "bci_sleep_explanation_drift_pilot_v1.yaml"
    )
    assert config["status"] == "locked_xai_pilot"
    assert config["cache"]["test_access_during_pilot"] == "forbidden"
    assert config["attribution_split"]["assignment_sha256"] == (
        "db6875681e198d6374a9be404712cce80249eb783d6f1d0f1cb86c8d97244c48"
    )
    assert set(config["checkpoints"]) == {"before", "after", "joint"}
    assert config["fidelity"]["random_masks"] == 100
    assert config["integrated_gradients"]["minimum_rank_agreement"] == 0.30
    assert config["gate"]["required_checkpoint_roles"] == ["before", "after"]


def test_converged_single_task_early_stopping_is_locked() -> None:
    config = load_yaml(ROOT / "configs" / "training" / "single_task_converged.yaml")
    assert config["mode"] == "converged"
    assert config["schedule"]["maximum_optimizer_steps"] == 5000
    assert config["schedule"]["validation_interval_steps"] == 100
    assert config["early_stopping"] == {
        "monitor": "mean_subject_balanced_accuracy",
        "mode": "max",
        "patience_validations": 10,
        "min_delta": 0.0,
        "restore_best": True,
    }


def test_physionet_upstream_reproduction_contract_is_separate_from_manifest() -> None:
    config = load_yaml(
        ROOT / "configs" / "training" / "physionet_cbramod_reproduction.yaml"
    )
    assert config["input"]["channels"] == 64
    assert config["training"]["epochs"] == 50
    assert config["selection"]["metric"] == "validation_cohen_kappa"


def test_fisher_instrument_uses_exact_disjoint_split_halves() -> None:
    config = load_yaml(ROOT / "configs" / "pilots" / "fisher_instrument_v1.yaml")
    estimator = config["estimator"]
    assert config["parameters"]["scope"] == "shared_plastic_backbone_only"
    assert config["parameters"]["exclude_task_head"] is True
    assert estimator["gradient_unit"] == "individual_example"
    assert estimator["implementation"] == "standard_backward_per_example"
    assert estimator["model_mode"] == "eval"
    assert estimator["sample_count"] == 2 * estimator["split_half_samples"] == 1024
    assert estimator["gradient_microbatch_size"] == 1
    assert config["normalization"]["primary"] == "l2_within_encoder_layer"
    assert config["instrument_gate"]["if_failed"].startswith("double_sample_count")


def test_sequential_ft_smoke_covers_both_pair_directions() -> None:
    config = load_yaml(
        ROOT / "configs" / "pilots" / "sequential_ft_smoke_v1.yaml"
    )
    assert config["orders"]["reverse"] == list(reversed(config["orders"]["forward"]))
    assert config["training"]["optimizer_steps_per_task"] == 2500
    assert config["training"]["evaluation_split"] == "test"
    assert config["model"]["head_initialization"] == (
        "preinitialize_all_in_canonical_task_order"
    )
    assert config["forgetting"]["minimum_valid_headroom_above_chance"] == 0.05


def test_main_sequential_ft_matrix_locks_three_orders_and_seeds() -> None:
    config = load_yaml(ROOT / "configs" / "training" / "sequential_ft_v1.yaml")
    assert config["status"] == "locked_main"
    assert config["seeds"] == [3407, 42, 2026]
    assert set(config["orders"]) == {"forward", "reverse", "challenging"}
    assert config["orders"]["challenging"] == [
        "physionet_mi",
        "bciciv2a",
        "sleep_edf_sc",
    ]
    assert config["analysis_gate"]["minimum_valid_replicates_per_direction"] == 3
