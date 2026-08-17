#!/usr/bin/env python3
"""Run one locked per-subject single-cell margin reliability pilot."""

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
    frozen_stratified_capped_halves,
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
        default=PROJECT_ROOT / "configs" / "xai" / "physionet_margin_reliability_v2.yaml",
    )
    return parser.parse_args()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _verify(path: Path, expected: str, role: str) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise DatasetProtocolError(f"{role} digest mismatch: {path}")


def reliability_decision(cosines: dict[str, float], gate: dict[str, object]) -> dict[str, object]:
    values = np.asarray(list(cosines.values()), dtype=np.float64)
    finite = bool(values.size and np.isfinite(values).all())
    median = float(np.median(values)) if finite else None
    threshold = float(gate["subject_threshold"])
    fraction = float(np.mean(values >= threshold)) if finite else 0.0
    passed = bool(
        finite
        and median >= float(gate["minimum_median_subject_cosine"])
        and fraction >= float(gate["minimum_subject_fraction_at_threshold"])
    )
    return {
        "all_subject_cosines_finite": finite,
        "median_subject_cosine": median,
        "subject_fraction_at_or_above_threshold": fraction,
        "passed": passed,
    }


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if config.get("status") != "locked_xai_reliability_pilot":
        raise DatasetProtocolError("margin reliability config is not locked")
    if config["cache"].get("test_access") != "forbidden":
        raise DatasetProtocolError("margin reliability pilot must remain test blind")
    config_sha = sha256_file(args.config)
    output = _resolve(str(config["output"]["root"]))
    result_path = output / "result.json"
    if result_path.exists():
        raise DatasetProtocolError(f"refusing to overwrite margin result {result_path}")
    validation_index = _resolve(str(config["cache"]["validation_index"]))
    test_index = _resolve(str(config["cache"]["test_index"]))
    _verify(validation_index, str(config["cache"]["validation_index_sha256"]), "validation cache")
    _verify(test_index, str(config["cache"]["test_index_sha256"]), "untouched test cache")
    specification = config["checkpoint"]
    checkpoint_path = _resolve(str(specification["path"]))
    parent_path = _resolve(str(specification["parent_config"]))
    _verify(checkpoint_path, str(specification["sha256"]), "checkpoint")
    _verify(parent_path, str(specification["parent_config_sha256"]), "parent config")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("config_sha256") != specification["parent_config_sha256"]
        or int(checkpoint.get("stage", -1)) != int(specification["expected_stage"])
        or checkpoint.get("task") != specification["expected_task"]
    ):
        raise DatasetProtocolError("margin reliability checkpoint identity mismatch")

    dataset = CachedEEGDataset(validation_index)
    labels, subjects = [], []
    for index in range(len(dataset)):
        _signal, label, subject = dataset[index]
        labels.append(int(label))
        subjects.append(subject)
    split_config = config["attribution_split"]
    split = frozen_stratified_capped_halves(
        labels,
        subjects,
        seed=int(split_config["seed"]),
        maximum_rows_per_subject_class=int(split_config["maximum_rows_per_subject_class"]),
    )
    if split["assignment_sha256"] != split_config["assignment_sha256"]:
        raise DatasetProtocolError("margin reliability split changed")
    fit = np.asarray(split["attribution_fit"], dtype=np.int64)
    gate = np.asarray(split["attribution_gate"], dtype=np.int64)
    channels = tuple(config["cell_registry"]["channels"])
    bands = tuple(
        (str(value["name"]), float(value["low_hz"]), float(value["high_hz"]))
        for value in config["cell_registry"]["bands"]
    )
    cells = len(channels) * len(bands)
    weights = torch.ones(cells + 1, len(channels), len(bands))
    for cell in range(cells):
        weights[cell + 1].view(-1)[cell] = 0.0
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    set_determinism(int(config["seed"]))
    model = MultiHeadCBraMod(CBraMod())
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device)
    if args.validate_only:
        print(json.dumps({"config_sha256": config_sha, "samples": len(dataset), "fit": len(fit), "gate": len(gate), "subjects": len(set(subjects)), "cells": cells}, indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    print(f"scoring {cells + 1} masks on {len(dataset)} rows", flush=True)
    scores = score_spectral_masks(
        model,
        str(config["task"]),
        dataset,
        range(len(dataset)),
        weights,
        sampling_rate_hz=float(config["cell_registry"]["sampling_rate_hz"]),
        bands=bands,
        batch_size=int(config["evaluation"]["batch_size"]),
        mask_chunk_size=int(config["evaluation"]["mask_chunk_size"]),
        device=device,
    )
    score_path = output / "margin-scores.npz"
    _atomic_npz(score_path, {**scores, "config_sha256": np.asarray(config_sha)})
    margins = np.asarray(scores["margins"], dtype=np.float64)
    subject_array = np.asarray(scores["subjects"], dtype=np.str_)
    fit_mean, fit_subject = subject_equal_margin_drop(
        margins[0, fit], margins[1:, fit], subject_array[fit]
    )
    gate_mean, gate_subject = subject_equal_margin_drop(
        margins[0, gate], margins[1:, gate], subject_array[gate]
    )
    cosines, positive = {}, {}
    for subject in sorted(fit_subject):
        try:
            cosines[subject] = cosine_similarity(fit_subject[subject], gate_subject[subject])
        except DatasetProtocolError:
            cosines[subject] = float("nan")
        positive[subject] = {
            "fit": int(np.sum(fit_subject[subject] > 0)),
            "gate": int(np.sum(gate_subject[subject] > 0)),
        }
    decision = reliability_decision(cosines, config["gate"])
    result = {
        "schema_version": 1,
        "run": config["id"],
        "config_sha256": config_sha,
        "task": config["task"],
        "seed": int(config["seed"]),
        "checkpoint_sha256": specification["sha256"],
        "test_was_loaded": False,
        "split_sha256": split["assignment_sha256"],
        "subject_split_half_cosine": cosines,
        "subject_positive_cells": positive,
        "group_mean_split_half_cosine": cosine_similarity(fit_mean, gate_mean),
        "gate": decision,
        "artifact": {"file": score_path.name, "sha256": sha256_file(score_path)},
        "device": str(device),
        "torch": torch.__version__,
    }
    _atomic_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
