#!/usr/bin/env python3
"""Run one locked per-subject single-cell margin reliability pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from eeg_forgetting.data.cache import CachedEEGDataset
from eeg_forgetting.data.contracts import DatasetProtocolError, load_yaml
from eeg_forgetting.data.manifests import sha256_file
from eeg_forgetting.models.cbramod import CBraMod, CBraModTaskModel
from eeg_forgetting.training.continual import MultiHeadCBraMod, _atomic_json, _atomic_npz
from eeg_forgetting.training.pilot import TASK_CLASSES, TASK_SHAPES, set_determinism
from eeg_forgetting.xai.explanation_drift import (
    cosine_similarity,
    frozen_stratified_capped_halves,
    frozen_stratified_halves,
    score_spectral_masks,
    subject_balanced_accuracy_drop,
    subject_class_balanced_margin_drop,
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


class _TaskBoundModel(nn.Module):
    """Expose a single-task classifier through the multi-head XAI call contract."""

    def __init__(self, task: str, model: CBraModTaskModel):
        super().__init__()
        self.task = task
        self.model = model

    def forward(self, task: str, signals: torch.Tensor) -> torch.Tensor:
        if task != self.task:
            raise DatasetProtocolError(
                f"single-task checkpoint is bound to {self.task!r}, got {task!r}"
            )
        return self.model(signals)


def _load_model(config: dict[str, object], device: torch.device) -> nn.Module:
    specification = config["checkpoint"]
    checkpoint_path = _resolve(str(specification["path"]))
    parent_path = _resolve(str(specification["parent_config"]))
    _verify(checkpoint_path, str(specification["sha256"]), "checkpoint")
    _verify(parent_path, str(specification["parent_config_sha256"]), "parent config")
    checkpoint_format = str(specification.get("format", "continual_checkpoint"))
    if checkpoint_format == "continual_checkpoint":
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (
            checkpoint.get("config_sha256") != specification["parent_config_sha256"]
            or int(checkpoint.get("stage", -1)) != int(specification["expected_stage"])
            or checkpoint.get("task") != specification["expected_task"]
        ):
            raise DatasetProtocolError("margin reliability checkpoint identity mismatch")
        model: nn.Module = MultiHeadCBraMod(CBraMod())
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    elif checkpoint_format == "single_task_state_dict":
        result_path = _resolve(str(specification["result"]))
        _verify(result_path, str(specification["result_sha256"]), "training result")
        with result_path.open(encoding="utf-8") as handle:
            training_result = json.load(handle)
        task = str(specification["expected_task"])
        if (
            training_result.get("pilot") != specification["expected_pilot"]
            or training_result.get("protocol_config_sha256")
            != specification["parent_config_sha256"]
            or training_result.get("final_checkpoint_sha256") != specification["sha256"]
            or int(training_result.get("final_step", -1)) != int(specification["expected_step"])
            or training_result.get("settings", {}).get("dataset") != task
        ):
            raise DatasetProtocolError("single-task checkpoint identity mismatch")
        channels, patches = TASK_SHAPES[task]
        task_model = CBraModTaskModel(
            CBraMod(),
            TASK_CLASSES[task],
            head=str(specification["head"]),
            channels=channels,
            patches=patches,
        )
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        task_model.load_state_dict(state, strict=True)
        model = _TaskBoundModel(task, task_model)
    else:
        raise DatasetProtocolError(f"unknown margin checkpoint format {checkpoint_format!r}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.to(device)


def _random_mask_indicators(
    *, cells: int, selected_cells: int, total: int, seed: int, forbidden: tuple[int, ...]
) -> np.ndarray:
    if not 0 < selected_cells < cells or total <= 0:
        raise DatasetProtocolError("invalid fidelity random-mask specification")
    generator = np.random.default_rng(seed)
    rows, seen = [], {forbidden}
    while len(rows) < total:
        selected = tuple(
            sorted(
                int(value)
                for value in generator.choice(cells, selected_cells, replace=False)
            )
        )
        if selected in seen:
            continue
        seen.add(selected)
        row = np.zeros(cells, dtype=np.float32)
        row[list(selected)] = 1.0
        rows.append(row)
    return np.stack(rows)


def _fidelity_result(
    scores: dict[str, object], *, percentile: float
) -> dict[str, object]:
    margins = np.asarray(scores["margins"], dtype=np.float64)
    predictions = np.asarray(scores["predictions"], dtype=np.int64)
    truth = np.asarray(scores["truth"], dtype=np.int64)
    subjects = np.asarray(scores["subjects"], dtype=np.str_)
    margin_drop, _ = subject_class_balanced_margin_drop(
        margins[0], margins[1:], truth, subjects
    )
    ba_drop = np.asarray(
        [
            subject_balanced_accuracy_drop(truth, predictions[0], prediction, subjects)[0]
            for prediction in predictions[1:]
        ],
        dtype=np.float64,
    )
    margin_threshold = float(np.percentile(margin_drop[1:], percentile))
    ba_threshold = float(np.percentile(ba_drop[1:], percentile))
    return {
        "ran": True,
        "top_margin_drop": float(margin_drop[0]),
        "control_margin_drop_percentile_95": margin_threshold,
        "margin_passed": bool(margin_drop[0] > margin_threshold),
        "top_ba_drop": float(ba_drop[0]),
        "control_ba_drop_percentile_95": ba_threshold,
        "ba_passed": bool(ba_drop[0] > ba_threshold),
        "passed": bool(
            margin_drop[0] > margin_threshold and ba_drop[0] > ba_threshold
        ),
    }


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
    _verify(validation_index, str(config["cache"]["validation_index_sha256"]), "validation cache")
    if config["cache"].get("test_index"):
        test_index = _resolve(str(config["cache"]["test_index"]))
        _verify(test_index, str(config["cache"]["test_index_sha256"]), "untouched test cache")
    elif not config["cache"].get("forbidden_test_subjects"):
        raise DatasetProtocolError("test-blind cache contract needs an index or forbidden subjects")
    specification = config["checkpoint"]

    dataset = CachedEEGDataset(validation_index)
    labels, subjects = [], []
    for index in range(len(dataset)):
        _signal, label, subject = dataset[index]
        labels.append(int(label))
        subjects.append(subject)
    observed_subjects = set(subjects)
    if observed_subjects != set(config["cache"]["validation_subjects"]):
        raise DatasetProtocolError("margin reliability validation subjects changed")
    if observed_subjects & set(config["cache"].get("forbidden_test_subjects", [])):
        raise DatasetProtocolError("margin reliability validation cache contains test subject")
    split_config = config["attribution_split"]
    if split_config["method"] == "sha256_rank_capped_within_subject_class_v1":
        split = frozen_stratified_capped_halves(
            labels,
            subjects,
            seed=int(split_config["seed"]),
            maximum_rows_per_subject_class=int(split_config["maximum_rows_per_subject_class"]),
        )
    elif split_config["method"] == "sha256_rank_within_subject_class_v1":
        split = frozen_stratified_halves(
            labels,
            subjects,
            seed=int(split_config["seed"]),
        )
    else:
        raise DatasetProtocolError("unknown margin reliability attribution split")
    if (
        split["assignment_sha256"] != split_config["assignment_sha256"]
        or len(split["attribution_fit"]) != int(split_config["attribution_fit_samples"])
        or len(split["attribution_gate"]) != int(split_config["attribution_gate_samples"])
    ):
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
    model = _load_model(config, device)
    if args.validate_only:
        print(json.dumps({"config_sha256": config_sha, "samples": len(dataset), "fit": len(fit), "gate": len(gate), "subjects": len(set(subjects)), "cells": cells, "fidelity_masks": 1 + int(config.get("fidelity", {}).get("random_masks", 0))}, indent=2))
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
    truth = np.asarray(scores["truth"], dtype=np.int64)
    subject_array = np.asarray(scores["subjects"], dtype=np.str_)
    class_balanced = (
        split_config.get("aggregation")
        == "mean_within_class_then_equal_classes_then_equal_subjects"
    )
    if class_balanced:
        fit_mean, fit_subject = subject_class_balanced_margin_drop(
            margins[0, fit], margins[1:, fit], truth[fit], subject_array[fit]
        )
        gate_mean, gate_subject = subject_class_balanced_margin_drop(
            margins[0, gate], margins[1:, gate], truth[gate], subject_array[gate]
        )
    else:
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
    fidelity = {"ran": False, "passed": False, "reason": "not_configured"}
    fidelity_artifact = None
    top_cells: list[int] | None = None
    if "fidelity" in config:
        if decision["passed"]:
            fidelity_config = config["fidelity"]
            selected_cells = int(fidelity_config["selected_cells"])
            top = tuple(sorted(int(value) for value in np.argsort(fit_mean)[-selected_cells:]))
            top_cells = list(top)
            random_indicators = _random_mask_indicators(
                cells=cells,
                selected_cells=selected_cells,
                total=int(fidelity_config["random_masks"]),
                seed=int(fidelity_config["random_seed"]),
                forbidden=top,
            )
            indicators = np.zeros((2 + len(random_indicators), cells), dtype=np.float32)
            indicators[1, list(top)] = 1.0
            indicators[2:] = random_indicators
            fidelity_weights = torch.from_numpy(1.0 - indicators).reshape(
                -1, len(channels), len(bands)
            )
            print(
                f"fidelity: scoring top + {len(random_indicators)} controls "
                f"on {len(gate)} gate rows",
                flush=True,
            )
            fidelity_scores = score_spectral_masks(
                model,
                str(config["task"]),
                dataset,
                gate,
                fidelity_weights,
                sampling_rate_hz=float(config["cell_registry"]["sampling_rate_hz"]),
                bands=bands,
                batch_size=int(config["evaluation"]["batch_size"]),
                mask_chunk_size=int(config["evaluation"]["mask_chunk_size"]),
                device=device,
            )
            fidelity_path = output / "fidelity-scores.npz"
            _atomic_npz(
                fidelity_path,
                {
                    **fidelity_scores,
                    "random_mask_indicators": random_indicators,
                    "config_sha256": np.asarray(config_sha),
                },
            )
            fidelity = _fidelity_result(
                fidelity_scores,
                percentile=float(fidelity_config["percentile"]),
            )
            fidelity["top_cell_indices"] = top_cells
            fidelity_artifact = {
                "file": fidelity_path.name,
                "sha256": sha256_file(fidelity_path),
            }
        else:
            fidelity = {
                "ran": False,
                "passed": False,
                "reason": "reliability_gate_failed",
            }
    overall_passed = bool(
        decision["passed"]
        and (fidelity["passed"] if "fidelity" in config else True)
    )
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
        "fidelity": fidelity,
        "decision": {
            "passed": overall_passed,
            "action": "allow_replacement_design" if overall_passed else "stop_replacement",
        },
        "artifact": {"file": score_path.name, "sha256": sha256_file(score_path)},
        "device": str(device),
        "torch": torch.__version__,
    }
    if fidelity_artifact is not None:
        result["fidelity_artifact"] = fidelity_artifact
    _atomic_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
