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


def test_explanation_drift_replication_amendment_is_locked_before_results() -> None:
    config = load_yaml(
        ROOT / "configs" / "xai" / "bci_sleep_explanation_drift_replication_v2.yaml"
    )
    assert config["status"] == "locked_xai_replication"
    assert config["seeds"] == [42, 2026]
    assert set(config["checkpoints_by_seed"]) == {42, 2026}
    assert config["gate"]["hard_require"] == ["fidelity", "drift_above_null"]
    assert config["gate"]["cross_seed_decision"] == {
        "seeds_including_original": [3407, 42, 2026],
        "minimum_hard_gate_passes": 2,
        "require_ig_positive_for_all_seeds": True,
        "no_posthoc_ig_threshold": True,
    }
    assert "minimum_rank_agreement" not in config["integrated_gradients"]


def test_bci_sleep_method_scale_is_checkpoint_only_and_test_blind() -> None:
    config = load_yaml(
        ROOT / "configs" / "xai" / "bci_sleep_method_scale_v1.yaml"
    )
    assert config["status"] == "locked_xai_scale"
    assert set(config["methods"]) == {"sequential", "ewc", "derpp"}
    assert set(config["orders"]) == {"forward", "challenging"}
    assert config["seeds"] == [3407, 42, 2026]
    assert config["cache"]["test_access_during_scale"] == "forbidden"
    assert config["integrated_gradients"]["scale_policy"] == (
        "not_repeated_per_checkpoint"
    )


def test_physionet_and_sleep_attribution_sets_are_capped_and_test_blind() -> None:
    expected = {
        "physionet_sleep_explanation_replication_v1.yaml": (
            720,
            720,
            "419fafc23fec9f304086594531b868a9d2dba40f7c52fa186eb01aab39c81974",
        ),
        "sleep_physionet_explanation_replication_v1.yaml": (
            732,
            732,
            "e6be61c974a4d96b31514a583622cbb9a488b2c341c4f01e02c5fcdd6295e0de",
        ),
    }
    for filename, (fit, gate, digest) in expected.items():
        config = load_yaml(ROOT / "configs" / "xai" / filename)
        split = config["attribution_split"]
        assert config["status"] == "locked_xai_replication"
        assert config["cache"]["test_access_during_pilot"] == "forbidden"
        assert split["maximum_rows_per_subject_class"] == 20
        assert split["attribution_fit_samples"] == fit
        assert split["attribution_gate_samples"] == gate
        assert split["assignment_sha256"] == digest


def test_margin_mask_reliability_v2_is_single_checkpoint_go_no_go() -> None:
    config = load_yaml(
        ROOT / "configs" / "xai" / "bci_margin_mask_reliability_v2.yaml"
    )
    assert config["status"] == "locked_xai_reliability_pilot"
    assert config["seed"] == 42
    assert config["randomized_masks"] == {
        "total": 200,
        "ridge_train": 160,
        "held_out": 40,
        "occluded_cells_per_mask": 22,
        "unique_masks": True,
        "seed": 20260821,
    }
    assert config["ridge"]["alpha"] == 1.0
    assert config["gate"]["pass_at_or_above"] == 0.70
    assert config["cache"]["test_access"] == "forbidden"


def test_sleep_sample_size_amendment_is_one_bounded_final_pilot() -> None:
    config = load_yaml(
        ROOT / "configs" / "xai" / "sleep_sample_size_amendment_v3.yaml"
    )
    assert config["status"] == "locked_xai_reliability_pilot"
    assert list(config["sample_size_curve"]["caps"]) == [20, 50, 100, 200]
    assert config["sample_size_curve"]["aggregation"].startswith("mean_within_class")
    assert config["sleep_fidelity"]["combinations"] == 120
    assert config["sleep_fidelity"]["require_margin_and_ba"] is True
    assert config["decision"]["fail"] == "stop_xai"


def test_high_gamma_replacement_candidate_is_locked_before_download() -> None:
    dataset = load_yaml(ROOT / "configs" / "datasets" / "high_gamma.yaml")
    pilot = load_yaml(ROOT / "configs" / "pilots" / "high_gamma_candidate_v1.yaml")
    assert dataset["status"] == "locked_replacement_candidate"
    assert dataset["subject_split"] == {
        "train": [1, 2, 3, 4, 5, 6, 7],
        "validation": [8, 9, 10, 11],
        "test": [12, 13, 14],
    }
    assert dataset["channel_montage"] == "bciciv2a_22"
    assert dataset["assertions"]["maximum_class_count_difference"] == 2
    assert dataset["assertions"]["class_balance_policy"] == "preserve_all_published_events"
    assert pilot["training"]["optimizer_steps"] == 2500
    assert pilot["test_access"] == "forbidden"


def test_joint_high_gamma_is_budget_matched_and_uses_mixed_cache_roots() -> None:
    config = load_yaml(ROOT / "configs" / "training" / "joint_high_gamma_v1.yaml")
    assert config["status"] == "locked_main"
    assert config["canonical_tasks"] == ["bciciv2a", "high_gamma", "sleep_edf_sc"]
    assert set(config["cache_roots"]) == set(config["canonical_tasks"])
    assert config["training"]["optimizer_updates_per_task"] == 2500
    assert config["training"]["total_optimizer_updates"] == 7500
    assert config["training"]["task_order_within_round"] == config["canonical_tasks"]


def test_high_gamma_attribution_gate_is_locked_before_scoring() -> None:
    config = load_yaml(
        ROOT / "configs" / "xai" / "high_gamma_margin_reliability_v1.yaml"
    )
    assert config["status"] == "locked_xai_reliability_pilot"
    assert config["checkpoint"]["format"] == "single_task_state_dict"
    assert config["checkpoint"]["expected_step"] == 2500
    assert config["cache"]["test_access"] == "forbidden"
    assert config["cache"]["forbidden_test_subjects"] == [
        "sub-12",
        "sub-13",
        "sub-14",
    ]
    assert config["attribution_split"] == {
        "method": "sha256_rank_within_subject_class_v1",
        "seed": 20260818,
        "attribution_fit_samples": 1966,
        "attribution_gate_samples": 1968,
        "assignment_sha256": "52add2ae02588b2c68ca3e4b423099e379e9cb27ba762dbf38c11b447696449b",
        "aggregation": "mean_within_class_then_equal_classes_then_equal_subjects",
    }
    assert config["fidelity"] == {
        "run_only_if_reliability_passes": True,
        "selected_cells": 22,
        "random_masks": 100,
        "random_seed": 20260821,
        "percentile": 95,
        "require_margin_and_ba": True,
    }


def test_high_gamma_replacement_matrix_keeps_methods_and_budget_locked() -> None:
    expected_tasks = ["bciciv2a", "high_gamma", "sleep_edf_sc"]
    expected_orders = {
        "forward": ["bciciv2a", "high_gamma", "sleep_edf_sc"],
        "reverse": ["sleep_edf_sc", "high_gamma", "bciciv2a"],
        "challenging": ["high_gamma", "bciciv2a", "sleep_edf_sc"],
    }
    for name, method in (
        ("sequential_ft_high_gamma_v1.yaml", "sequential_finetuning"),
        ("ewc_high_gamma_v1.yaml", "ewc"),
        ("derpp_high_gamma_v1.yaml", "derpp"),
    ):
        config = load_yaml(ROOT / "configs" / "training" / name)
        assert config["status"] == "locked_main"
        assert config["method"] == method
        assert config["canonical_tasks"] == expected_tasks
        assert config["orders"] == expected_orders
        assert config["seeds"] == [3407, 42, 2026]
        assert config["training"]["optimizer_steps_per_task"] == 2500
        assert config["training"]["evaluation_split"] == "test"
        assert set(config["cache_roots"]) == set(expected_tasks)
    ewc = load_yaml(ROOT / "configs" / "training" / "ewc_high_gamma_v1.yaml")
    derpp = load_yaml(ROOT / "configs" / "training" / "derpp_high_gamma_v1.yaml")
    assert ewc["ewc"]["strength"] == 100000
    assert derpp["derpp"]["persistent_byte_cap"] == 8388608


def test_high_gamma_near_chance_amendment_does_not_lower_threshold() -> None:
    config = load_yaml(
        ROOT / "configs" / "analysis" / "high_gamma_sequential_stability_v2.yaml"
    )
    assert config["status"] == "locked_post_gate_amendment"
    assert config["fallback"]["directions"] == ["sleep_edf_sc<-bciciv2a"]
    assert config["fallback"]["metric"] == "raw_forgetting_for_all_replicates_in_direction"
    assert config["fallback"]["preserve_invalid_relative_as_null"] is True
    assert config["gate"]["minimum_replicates_per_direction"] == 3
    assert config["gate"]["minimum_signal_directions"] == 4


def test_replacement_ped_scale_excludes_sleep_and_locks_noise_correction() -> None:
    config = load_yaml(
        ROOT / "configs" / "xai" / "high_gamma_replacement_ped_v1.yaml"
    )
    assert config["status"] == "locked_xai_scale"
    assert config["scale"]["old_tasks"] == ["bciciv2a", "high_gamma"]
    assert config["scale"]["excluded_old_tasks"] == ["sleep_edf_sc"]
    assert config["scale"]["expected_run_cells"] == 27
    assert config["scale"]["expected_transition_cells"] == 63
    assert config["estimator"]["drift"] == (
        "symmetric_cross_checkpoint_jsd_minus_within_checkpoint_split_noise"
    )
    assert config["estimator"]["clip_negative_ped"] is False
    assert config["scale"]["test_access_for_xai"] == "forbidden"


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
