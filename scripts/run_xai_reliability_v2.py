#!/usr/bin/env python3
"""Run the locked margin/random-mask reliability pilot on one BCI checkpoint."""

from __future__ import annotations

import argparse
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
    frozen_stratified_halves,
    ridge_mask_coefficients,
    score_spectral_masks,
    subject_equal_margin_drop,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "xai" / "bci_margin_mask_reliability_v2.yaml",
    )
    return parser.parse_args()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _verify(path: Path, expected: str, role: str) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise DatasetProtocolError(f"{role} digest mismatch: {path}")


def _r2(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    denominator = float(np.sum((truth - truth.mean()) ** 2))
    if truth.shape != prediction.shape or denominator <= 0:
        raise DatasetProtocolError("held-out mask responses have no finite variance")
    return float(1.0 - np.sum((truth - prediction) ** 2) / denominator)


def _load_model(config: dict[str, object], device: torch.device) -> MultiHeadCBraMod:
    specification = config["checkpoint"]
    checkpoint_path = _resolve(str(specification["path"]))
    parent_path = _resolve(str(specification["parent_config"]))
    _verify(checkpoint_path, str(specification["sha256"]), "pilot checkpoint")
    _verify(parent_path, str(specification["parent_config_sha256"]), "parent config")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("config_sha256") != specification["parent_config_sha256"]
        or int(checkpoint.get("stage", -1)) != int(specification["expected_stage"])
        or checkpoint.get("task") != specification["expected_task"]
    ):
        raise DatasetProtocolError("reliability checkpoint identity mismatch")
    model = MultiHeadCBraMod(CBraMod())
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.to(device)


def _random_indicators(config: dict[str, object], cells: int) -> np.ndarray:
    specification = config["randomized_masks"]
    generator = np.random.default_rng(int(specification["seed"]))
    rows, seen = [], set()
    while len(rows) < int(specification["total"]):
        selected = tuple(
            sorted(
                int(value)
                for value in generator.choice(
                    cells,
                    int(specification["occluded_cells_per_mask"]),
                    replace=False,
                )
            )
        )
        if selected in seen:
            continue
        seen.add(selected)
        row = np.zeros(cells, dtype=np.float32)
        row[list(selected)] = 1.0
        rows.append(row)
    return np.stack(rows)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if config.get("status") != "locked_xai_reliability_pilot":
        raise DatasetProtocolError("reliability pilot config is not locked")
    if config["cache"].get("test_access") != "forbidden":
        raise DatasetProtocolError("reliability pilot must remain test blind")
    config_sha = sha256_file(args.config)
    output = _resolve(str(config["output"]["root"]))
    result_path = output / "result.json"
    if result_path.exists():
        raise DatasetProtocolError(f"refusing to overwrite reliability result {result_path}")

    validation_index = _resolve(str(config["cache"]["validation_index"]))
    test_index = _resolve(str(config["cache"]["test_index"]))
    _verify(validation_index, str(config["cache"]["validation_index_sha256"]), "validation cache")
    _verify(test_index, str(config["cache"]["test_index_sha256"]), "untouched test cache")
    reference_path = _resolve(str(config["v1_reference"]["result"]))
    _verify(reference_path, str(config["v1_reference"]["result_sha256"]), "v1 reference")
    with reference_path.open(encoding="utf-8") as handle:
        reference = json.load(handle)
    role = str(config["v1_reference"]["role"])
    if reference["roles"][role]["checkpoint_sha256"] != config["checkpoint"]["sha256"]:
        raise DatasetProtocolError("v1 reference and v2 checkpoint do not match")

    dataset = CachedEEGDataset(validation_index)
    labels, subjects = [], []
    for index in range(len(dataset)):
        _signal, label, subject = dataset[index]
        labels.append(int(label))
        subjects.append(subject)
    split = frozen_stratified_halves(
        labels, subjects, seed=int(config["attribution_split"]["seed"])
    )
    if split["assignment_sha256"] != config["attribution_split"]["assignment_sha256"]:
        raise DatasetProtocolError("reliability attribution split changed")
    fit = np.asarray(split["attribution_fit"], dtype=np.int64)
    gate = np.asarray(split["attribution_gate"], dtype=np.int64)

    channels = tuple(config["cell_registry"]["channels"])
    bands = tuple(
        (str(value["name"]), float(value["low_hz"]), float(value["high_hz"]))
        for value in config["cell_registry"]["bands"]
    )
    cells = len(channels) * len(bands)
    random_indicators = _random_indicators(config, cells)
    individual_weights = torch.ones(cells + 1, len(channels), len(bands))
    for cell in range(cells):
        individual_weights[cell + 1].view(-1)[cell] = 0.0
    random_weights = torch.from_numpy(1.0 - random_indicators).reshape(
        len(random_indicators), len(channels), len(bands)
    )
    all_weights = torch.cat((individual_weights, random_weights), dim=0)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    set_determinism(int(config["seed"]))
    model = _load_model(config, device)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "config_sha256": config_sha,
                    "checkpoint_sha256": config["checkpoint"]["sha256"],
                    "samples": len(dataset),
                    "fit": len(fit),
                    "gate": len(gate),
                    "cells": cells,
                    "random_masks": len(random_indicators),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    output.mkdir(parents=True, exist_ok=True)
    print(f"scoring {len(all_weights)} masks on {len(dataset)} validation rows", flush=True)
    scores = score_spectral_masks(
        model,
        str(config["task"]),
        dataset,
        range(len(dataset)),
        all_weights,
        sampling_rate_hz=float(config["cell_registry"]["sampling_rate_hz"]),
        bands=bands,
        batch_size=int(config["evaluation"]["batch_size"]),
        mask_chunk_size=int(config["evaluation"]["mask_chunk_size"]),
        device=device,
    )
    score_path = output / "margin-scores.npz"
    _atomic_npz(
        score_path,
        {
            **scores,
            "random_mask_indicators": random_indicators,
            "config_sha256": np.asarray(config_sha),
            "checkpoint_sha256": np.asarray(config["checkpoint"]["sha256"]),
        },
    )

    margins = np.asarray(scores["margins"], dtype=np.float64)
    subject_array = np.asarray(scores["subjects"], dtype=np.str_)
    baseline = margins[0]
    individual = margins[1 : cells + 1]
    randomized = margins[cells + 1 :]
    single_fit, single_fit_subject = subject_equal_margin_drop(
        baseline[fit], individual[:, fit], subject_array[fit]
    )
    single_gate, single_gate_subject = subject_equal_margin_drop(
        baseline[gate], individual[:, gate], subject_array[gate]
    )
    random_fit, random_fit_subject = subject_equal_margin_drop(
        baseline[fit], randomized[:, fit], subject_array[fit]
    )
    random_gate, random_gate_subject = subject_equal_margin_drop(
        baseline[gate], randomized[:, gate], subject_array[gate]
    )
    train_count = int(config["randomized_masks"]["ridge_train"])
    train_x = random_indicators[:train_count]
    heldout_x = random_indicators[train_count:]
    alpha = float(config["ridge"]["alpha"])
    fit_coefficients, fit_intercept = ridge_mask_coefficients(
        train_x, random_fit[:train_count], alpha=alpha
    )
    gate_coefficients, gate_intercept = ridge_mask_coefficients(
        train_x, random_gate[:train_count], alpha=alpha
    )
    fit_r2 = _r2(
        random_fit[train_count:], heldout_x @ fit_coefficients + fit_intercept
    )
    gate_r2 = _r2(
        random_gate[train_count:], heldout_x @ gate_coefficients + gate_intercept
    )
    subject_reliability = {}
    for subject in sorted(single_fit_subject):
        left, _left_intercept = ridge_mask_coefficients(
            train_x, random_fit_subject[subject][:train_count], alpha=alpha
        )
        right, _right_intercept = ridge_mask_coefficients(
            train_x, random_gate_subject[subject][:train_count], alpha=alpha
        )
        subject_reliability[subject] = cosine_similarity(left, right)
    v1_fit = np.asarray(reference["roles"][role]["reliance"]["attribution_fit"])
    v1_gate = np.asarray(reference["roles"][role]["reliance"]["attribution_gate"])
    v1_cosine = cosine_similarity(v1_fit, v1_gate)
    single_cosine = cosine_similarity(single_fit, single_gate)
    ridge_cosine = cosine_similarity(fit_coefficients, gate_coefficients)
    pass_threshold = float(config["gate"]["pass_at_or_above"])
    inconclusive_threshold = float(config["gate"]["inconclusive_at_or_above"])
    heldout_passed = bool(fit_r2 > 0 and gate_r2 > 0)
    status = (
        "pass"
        if ridge_cosine >= pass_threshold and heldout_passed
        else "inconclusive"
        if ridge_cosine >= inconclusive_threshold
        else "fail"
    )
    result = {
        "schema_version": 1,
        "run": config["id"],
        "config_sha256": config_sha,
        "task": config["task"],
        "seed": int(config["seed"]),
        "checkpoint_sha256": config["checkpoint"]["sha256"],
        "test_was_loaded": False,
        "split_sha256": split["assignment_sha256"],
        "v1_single_cell_ba_drop": {"split_half_cosine": v1_cosine},
        "v2_single_cell_margin_drop": {
            "split_half_cosine": single_cosine,
            "fit_map": single_fit.tolist(),
            "gate_map": single_gate.tolist(),
        },
        "v2_random_mask_ridge_margin_drop": {
            "split_half_cosine": ridge_cosine,
            "fit_coefficients": fit_coefficients.tolist(),
            "gate_coefficients": gate_coefficients.tolist(),
            "heldout_r2_fit_half": fit_r2,
            "heldout_r2_gate_half": gate_r2,
            "subject_split_half_cosine": subject_reliability,
            "mask_feature_count_minimum": int(random_indicators.sum(axis=0).min()),
            "mask_feature_count_maximum": int(random_indicators.sum(axis=0).max()),
        },
        "gate": {
            "status": status,
            "ridge_reliability_passed": bool(ridge_cosine >= pass_threshold),
            "heldout_prediction_passed": heldout_passed,
            "full_scale_allowed": status == "pass",
        },
        "artifact": {"file": score_path.name, "sha256": sha256_file(score_path)},
        "device": str(device),
        "torch": torch.__version__,
    }
    _atomic_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
