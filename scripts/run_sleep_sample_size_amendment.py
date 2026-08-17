#!/usr/bin/env python3
"""Run the final locked Sleep sample-size curve and margin fidelity gates."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from eeg_forgetting.data.cache import CachedEEGDataset
from eeg_forgetting.data.contracts import DatasetProtocolError, load_yaml
from eeg_forgetting.data.manifests import sha256_file
from eeg_forgetting.models.cbramod import CBraMod
from eeg_forgetting.training.continual import MultiHeadCBraMod, _atomic_json, _atomic_npz
from eeg_forgetting.training.pilot import set_determinism
from eeg_forgetting.xai.explanation_drift import (
    cosine_similarity,
    frozen_stratified_capped_halves,
    frozen_stratified_halves,
    score_spectral_masks,
    subject_balanced_accuracy_drop,
    subject_class_balanced_margin_drop,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "xai" / "sleep_sample_size_amendment_v3.yaml",
    )
    return parser.parse_args()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _verify(path: Path, expected: str, role: str) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise DatasetProtocolError(f"{role} digest mismatch: {path}")


def _load_model(
    checkpoint_path: Path,
    checkpoint_sha: str,
    parent_sha: str,
    expected_stage: int,
    expected_task: str,
    device: torch.device,
) -> MultiHeadCBraMod:
    _verify(checkpoint_path, checkpoint_sha, "checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("config_sha256") != parent_sha
        or int(checkpoint.get("stage", -1)) != expected_stage
        or checkpoint.get("task") != expected_task
    ):
        raise DatasetProtocolError("checkpoint identity mismatch")
    model = MultiHeadCBraMod(CBraMod())
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.to(device)


def _dataset_rows(dataset: CachedEEGDataset) -> tuple[list[int], list[str]]:
    labels, subjects = [], []
    for index in range(len(dataset)):
        _signal, label, subject = dataset[index]
        labels.append(int(label))
        subjects.append(subject)
    return labels, subjects


def _positions(global_indices: list[int] | np.ndarray, scored_indices: np.ndarray) -> np.ndarray:
    lookup = {int(value): index for index, value in enumerate(scored_indices)}
    try:
        return np.asarray([lookup[int(value)] for value in global_indices], dtype=np.int64)
    except KeyError as error:
        raise DatasetProtocolError("cap rows are missing from scored union") from error


def _subject_curve(
    scores: dict[str, np.ndarray],
    split: dict[str, object],
) -> dict[str, object]:
    fit = _positions(split["attribution_fit"], scores["indices"])
    gate = _positions(split["attribution_gate"], scores["indices"])
    margins = scores["margins"]
    truth = scores["truth"]
    subjects = scores["subjects"]
    fit_mean, fit_subject = subject_class_balanced_margin_drop(
        margins[0, fit], margins[1:, fit], truth[fit], subjects[fit]
    )
    gate_mean, gate_subject = subject_class_balanced_margin_drop(
        margins[0, gate], margins[1:, gate], truth[gate], subjects[gate]
    )
    values, positive = {}, {}
    for subject in sorted(fit_subject):
        try:
            values[subject] = cosine_similarity(fit_subject[subject], gate_subject[subject])
        except DatasetProtocolError:
            values[subject] = None
        positive[subject] = {
            "fit": int(np.sum(fit_subject[subject] > 0)),
            "gate": int(np.sum(gate_subject[subject] > 0)),
        }
    finite_values = np.asarray(
        [value for value in values.values() if value is not None and np.isfinite(value)],
        dtype=np.float64,
    )
    return {
        "subject_split_half_cosine": values,
        "subject_positive_cells": positive,
        "all_subject_cosines_finite": len(finite_values) == len(values),
        "median_subject_cosine": float(np.median(finite_values)) if len(finite_values) else None,
        "subject_fraction_at_or_above_0_50": float(np.mean(finite_values >= 0.5)) if len(finite_values) else 0.0,
        "group_mean_cosine": cosine_similarity(fit_mean, gate_mean),
        "fit_map": fit_mean,
        "gate_map": gate_mean,
        "fit_positions": fit,
        "gate_positions": gate,
    }


def _fidelity(
    baseline_margin: np.ndarray,
    baseline_prediction: np.ndarray,
    masked_margins: np.ndarray,
    masked_predictions: np.ndarray,
    truth: np.ndarray,
    subjects: np.ndarray,
    labels: np.ndarray,
    selected_index: int,
    percentile: float,
) -> dict[str, object]:
    margin_drop, _by_subject = subject_class_balanced_margin_drop(
        baseline_margin, masked_margins, labels, subjects
    )
    ba_drop = np.asarray(
        [
            subject_balanced_accuracy_drop(
                truth, baseline_prediction, prediction, subjects
            )[0]
            for prediction in masked_predictions
        ],
        dtype=np.float64,
    )
    controls = np.arange(len(margin_drop)) != selected_index
    margin_threshold = float(np.percentile(margin_drop[controls], percentile))
    ba_threshold = float(np.percentile(ba_drop[controls], percentile))
    return {
        "top_margin_drop": float(margin_drop[selected_index]),
        "control_margin_drop_percentile_95": margin_threshold,
        "margin_passed": bool(margin_drop[selected_index] > margin_threshold),
        "top_ba_drop": float(ba_drop[selected_index]),
        "control_ba_drop_percentile_95": ba_threshold,
        "ba_passed": bool(ba_drop[selected_index] > ba_threshold),
        "passed": bool(
            margin_drop[selected_index] > margin_threshold
            and ba_drop[selected_index] > ba_threshold
        ),
    }


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if config.get("status") != "locked_xai_reliability_pilot":
        raise DatasetProtocolError("Sleep amendment config is not locked")
    if config["cache"].get("test_access") != "forbidden":
        raise DatasetProtocolError("Sleep amendment must remain test blind")
    config_sha = sha256_file(args.config)
    output = _resolve(str(config["output"]["root"]))
    result_path = output / "result.json"
    if result_path.exists():
        raise DatasetProtocolError(f"refusing to overwrite Sleep amendment {result_path}")
    validation_index = _resolve(str(config["cache"]["validation_index"]))
    test_index = _resolve(str(config["cache"]["test_index"]))
    _verify(validation_index, str(config["cache"]["validation_index_sha256"]), "Sleep validation cache")
    _verify(test_index, str(config["cache"]["test_index_sha256"]), "untouched Sleep test cache")
    dataset = CachedEEGDataset(validation_index)
    labels, subjects = _dataset_rows(dataset)
    splits = {}
    for cap_value, expected in config["sample_size_curve"]["caps"].items():
        cap = int(cap_value)
        split = frozen_stratified_capped_halves(
            labels,
            subjects,
            seed=int(config["sample_size_curve"]["split_seed"]),
            maximum_rows_per_subject_class=cap,
        )
        if (
            len(split["attribution_fit"]) != int(expected["fit"])
            or len(split["attribution_gate"]) != int(expected["gate"])
            or split["assignment_sha256"] != expected["assignment_sha256"]
        ):
            raise DatasetProtocolError(f"Sleep cap-{cap} assignment changed")
        splits[cap] = split
    selected = np.asarray(
        sorted(
            set(splits[200]["attribution_fit"])
            | set(splits[200]["attribution_gate"])
        ),
        dtype=np.int64,
    )
    bands = tuple(
        (str(value["name"]), float(value["low_hz"]), float(value["high_hz"]))
        for value in config["cell_registry"]["bands"]
    )
    channels = tuple(config["cell_registry"]["channels"])
    cells = len(channels) * len(bands)
    weights = torch.ones(cells + 1, len(channels), len(bands))
    for cell in range(cells):
        weights[cell + 1].view(-1)[cell] = 0.0
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    set_determinism(int(config["seed"]))
    checkpoint = config["checkpoint"]
    sleep_model = _load_model(
        _resolve(str(checkpoint["path"])),
        str(checkpoint["sha256"]),
        str(checkpoint["parent_config_sha256"]),
        int(checkpoint["expected_stage"]),
        str(checkpoint["expected_task"]),
        device,
    )
    if args.validate_only:
        print(
            json.dumps(
                {
                    "config_sha256": config_sha,
                    "selected_rows": len(selected),
                    "caps": {cap: [len(value["attribution_fit"]), len(value["attribution_gate"])] for cap, value in splits.items()},
                    "sleep_cells": cells,
                    "sleep_fidelity_combinations": len(list(itertools.combinations(range(cells), 3))),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    output.mkdir(parents=True, exist_ok=True)
    print(f"Sleep individual masks: {len(selected)} rows x {cells + 1} masks", flush=True)
    sleep_scores = score_spectral_masks(
        sleep_model,
        str(config["task"]),
        dataset,
        selected,
        weights,
        sampling_rate_hz=float(config["cell_registry"]["sampling_rate_hz"]),
        bands=bands,
        batch_size=int(config["evaluation"]["batch_size"]),
        mask_chunk_size=int(config["evaluation"]["mask_chunk_size"]),
        device=device,
    )
    sleep_individual_path = output / "sleep-individual-margin-scores.npz"
    _atomic_npz(sleep_individual_path, {**sleep_scores, "config_sha256": np.asarray(config_sha)})
    sleep_scores = {key: np.asarray(value) for key, value in sleep_scores.items()}
    curve = {cap: _subject_curve(sleep_scores, split) for cap, split in splits.items()}
    cap200 = curve[200]
    gate_config = config["reliability_gate"]
    reliability_passed = bool(
        cap200["all_subject_cosines_finite"]
        and cap200["median_subject_cosine"] >= float(gate_config["minimum_median_subject_cosine"])
        and cap200["subject_fraction_at_or_above_0_50"]
        >= float(gate_config["minimum_subject_fraction_at_threshold"])
    )

    top_sleep = tuple(
        sorted(
            int(value)
            for value in np.argsort(cap200["fit_map"])[
                -int(config["sleep_fidelity"]["selected_cells"]):
            ]
        )
    )
    combinations = list(itertools.combinations(range(cells), 3))
    top_sleep_index = combinations.index(top_sleep)
    combination_weights = torch.ones(len(combinations), len(channels), len(bands))
    for index, combination in enumerate(combinations):
        combination_weights[index].view(-1)[list(combination)] = 0.0
    sleep_gate_global = np.asarray(splits[200]["attribution_gate"], dtype=np.int64)
    print(f"Sleep exhaustive fidelity: {len(sleep_gate_global)} rows x {len(combinations)} masks", flush=True)
    sleep_fidelity_scores = score_spectral_masks(
        sleep_model,
        str(config["task"]),
        dataset,
        sleep_gate_global,
        combination_weights,
        sampling_rate_hz=float(config["cell_registry"]["sampling_rate_hz"]),
        bands=bands,
        batch_size=int(config["evaluation"]["batch_size"]),
        mask_chunk_size=int(config["evaluation"]["mask_chunk_size"]),
        device=device,
    )
    sleep_fidelity_path = output / "sleep-exhaustive-fidelity-scores.npz"
    _atomic_npz(sleep_fidelity_path, {**sleep_fidelity_scores, "config_sha256": np.asarray(config_sha)})
    base_positions = _positions(sleep_gate_global, sleep_scores["indices"])
    sleep_fidelity = _fidelity(
        sleep_scores["margins"][0, base_positions],
        sleep_scores["predictions"][0, base_positions],
        sleep_fidelity_scores["margins"],
        sleep_fidelity_scores["predictions"],
        sleep_fidelity_scores["truth"],
        sleep_fidelity_scores["subjects"],
        sleep_fidelity_scores["truth"],
        top_sleep_index,
        float(config["sleep_fidelity"]["percentile"]),
    )
    sleep_fidelity["top_cell_indices"] = list(top_sleep)

    del sleep_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    bci = config["bci_fidelity"]
    _verify(_resolve(str(bci["result"])), str(bci["result_sha256"]), "BCI margin result")
    _verify(_resolve(str(bci["score_artifact"])), str(bci["score_artifact_sha256"]), "BCI margin score")
    bci_dataset = CachedEEGDataset(_resolve(str(bci["validation_index"])))
    bci_labels, bci_subjects = _dataset_rows(bci_dataset)
    bci_split = frozen_stratified_halves(
        bci_labels, bci_subjects, seed=int(bci["split_seed"])
    )
    if bci_split["assignment_sha256"] != bci["split_sha256"]:
        raise DatasetProtocolError("BCI fidelity split changed")
    with _resolve(str(bci["result"])).open(encoding="utf-8") as handle:
        bci_result = json.load(handle)
    with np.load(_resolve(str(bci["score_artifact"])), allow_pickle=False) as archive:
        random_indicators = np.asarray(archive["random_mask_indicators"], dtype=np.float32)
    bci_fit_map = np.asarray(bci_result["v2_single_cell_margin_drop"]["fit_map"])
    top_bci = np.sort(np.argsort(bci_fit_map)[-int(bci["selected_cells"]):])
    bci_indicators = np.zeros((2 + len(random_indicators), 110), dtype=np.float32)
    bci_indicators[1, top_bci] = 1.0
    bci_indicators[2:] = random_indicators
    bci_weights = torch.from_numpy(1.0 - bci_indicators).reshape(-1, 22, 5)
    bci_model = _load_model(
        _resolve(str(bci["checkpoint_path"])),
        str(bci["checkpoint_sha256"]),
        str(bci["parent_config_sha256"]),
        int(bci["expected_stage"]),
        str(bci["expected_task"]),
        device,
    )
    bci_gate = np.asarray(bci_split["attribution_gate"], dtype=np.int64)
    print(f"BCI margin fidelity: {len(bci_gate)} rows x {len(bci_weights)} masks", flush=True)
    bci_scores = score_spectral_masks(
        bci_model,
        "bciciv2a",
        bci_dataset,
        bci_gate,
        bci_weights,
        sampling_rate_hz=200.0,
        bands=bands,
        batch_size=16,
        mask_chunk_size=4,
        device=device,
    )
    bci_path = output / "bci-margin-fidelity-scores.npz"
    _atomic_npz(bci_path, {**bci_scores, "config_sha256": np.asarray(config_sha)})
    bci_fidelity = _fidelity(
        bci_scores["margins"][0],
        bci_scores["predictions"][0],
        bci_scores["margins"][1:],
        bci_scores["predictions"][1:],
        bci_scores["truth"],
        bci_scores["subjects"],
        bci_scores["truth"],
        0,
        float(bci["percentile"]),
    )
    bci_fidelity["top_cell_indices"] = top_bci.tolist()

    overall_passed = bool(
        reliability_passed and sleep_fidelity["passed"] and bci_fidelity["passed"]
    )
    curve_json = {}
    for cap, values in curve.items():
        curve_json[str(cap)] = {
            key: value
            for key, value in values.items()
            if key not in {"fit_map", "gate_map", "fit_positions", "gate_positions"}
        }
    result = {
        "schema_version": 1,
        "run": config["id"],
        "config_sha256": config_sha,
        "test_was_loaded": False,
        "sample_size_curve": curve_json,
        "reliability_gate_passed": reliability_passed,
        "sleep_fidelity": sleep_fidelity,
        "bci_fidelity": bci_fidelity,
        "decision": {
            "passed": overall_passed,
            "action": "allow_sleep_band_drift" if overall_passed else "stop_xai",
        },
        "artifacts": {
            "sleep_individual": {"file": sleep_individual_path.name, "sha256": sha256_file(sleep_individual_path)},
            "sleep_fidelity": {"file": sleep_fidelity_path.name, "sha256": sha256_file(sleep_fidelity_path)},
            "bci_fidelity": {"file": bci_path.name, "sha256": sha256_file(bci_path)},
        },
        "device": str(device),
        "torch": torch.__version__,
    }
    _atomic_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
