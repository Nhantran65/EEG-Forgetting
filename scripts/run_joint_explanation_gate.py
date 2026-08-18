#!/usr/bin/env python3
"""Score one seed/task joint-reference explanation reliability and fidelity gate."""

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
    score_spectral_masks,
    subject_balanced_accuracy_drop,
    subject_class_balanced_margin_drop,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("bciciv2a", "high_gamma"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "xai" / "joint_high_gamma_alignment_v1.yaml",
    )
    return parser.parse_args()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _verify(path: Path, expected: str, role: str) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise DatasetProtocolError(f"{role} digest mismatch: {path}")


def _random_masks(cells: int, selected: int, total: int, seed: int, forbidden: tuple[int, ...]) -> np.ndarray:
    generator = np.random.default_rng(seed)
    rows, seen = [], {forbidden}
    while len(rows) < total:
        choice = tuple(sorted(int(v) for v in generator.choice(cells, selected, replace=False)))
        if choice in seen:
            continue
        seen.add(choice)
        row = np.zeros(cells, dtype=np.float32)
        row[list(choice)] = 1.0
        rows.append(row)
    return np.stack(rows)


def _fidelity(scores: dict[str, object], percentile: float) -> dict[str, object]:
    margins = np.asarray(scores["margins"], dtype=np.float64)
    predictions = np.asarray(scores["predictions"], dtype=np.int64)
    truth = np.asarray(scores["truth"], dtype=np.int64)
    subjects = np.asarray(scores["subjects"], dtype=np.str_)
    margin_drop, _ = subject_class_balanced_margin_drop(
        margins[0], margins[1:], truth, subjects
    )
    ba_drop = np.asarray(
        [subject_balanced_accuracy_drop(truth, predictions[0], row, subjects)[0] for row in predictions[1:]],
        dtype=np.float64,
    )
    margin_p95 = float(np.percentile(margin_drop[1:], percentile))
    ba_p95 = float(np.percentile(ba_drop[1:], percentile))
    return {
        "top_margin_drop": float(margin_drop[0]),
        "control_margin_drop_percentile_95": margin_p95,
        "margin_passed": bool(margin_drop[0] > margin_p95),
        "top_ba_drop": float(ba_drop[0]),
        "control_ba_drop_percentile_95": ba_p95,
        "ba_passed": bool(ba_drop[0] > ba_p95),
        "passed": bool(margin_drop[0] > margin_p95 and ba_drop[0] > ba_p95),
    }


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if config.get("status") != "locked_joint_alignment_gate":
        raise DatasetProtocolError("joint explanation gate config is not locked")
    if int(args.seed) not in config["joint"]["seeds"]:
        raise DatasetProtocolError("joint explanation seed is not declared")
    if config["xai"].get("test_access") != "forbidden":
        raise DatasetProtocolError("joint explanation gate must remain validation-only")
    config_sha = sha256_file(args.config)
    joint = config["joint"]
    training_path = _resolve(str(joint["training_config"]))
    summary_path = _resolve(str(joint["summary"]))
    _verify(training_path, str(joint["training_config_sha256"]), "joint training config")
    _verify(summary_path, str(joint["summary_sha256"]), "joint summary")
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    result_path = _resolve(str(joint["result_root"])) / f"seed-{args.seed}" / "result.json"
    _verify(result_path, str(summary["input_result_sha256"][f"seed-{args.seed}"]), "joint result")
    with result_path.open(encoding="utf-8") as handle:
        joint_result = json.load(handle)
    checkpoint_path = result_path.parent / str(joint_result["checkpoint"]["file"])
    _verify(checkpoint_path, str(joint_result["checkpoint"]["sha256"]), "joint checkpoint")

    task_config = config["tasks"][args.task]
    validation_index = _resolve(str(task_config["validation_index"]))
    _verify(validation_index, str(task_config["validation_index_sha256"]), "validation cache")
    dataset = CachedEEGDataset(validation_index)
    labels, subjects = [], []
    for index in range(len(dataset)):
        _signal, label, subject = dataset[index]
        labels.append(int(label))
        subjects.append(str(subject))
    if set(subjects) != set(task_config["validation_subjects"]):
        raise DatasetProtocolError("joint gate validation subjects changed")
    split_config = task_config["split"]
    split = frozen_stratified_halves(labels, subjects, seed=int(split_config["seed"]))
    if (
        split["assignment_sha256"] != split_config["assignment_sha256"]
        or len(split["attribution_fit"]) != int(split_config["fit_samples"])
        or len(split["attribution_gate"]) != int(split_config["gate_samples"])
    ):
        raise DatasetProtocolError("joint gate attribution split changed")
    fit = np.asarray(split["attribution_fit"], dtype=np.int64)
    gate = np.asarray(split["attribution_gate"], dtype=np.int64)
    channels = tuple(config["cell_registry"]["channels"])
    bands = tuple(
        (str(v["name"]), float(v["low_hz"]), float(v["high_hz"]))
        for v in config["cell_registry"]["bands"]
    )
    cells = len(channels) * len(bands)
    individual_weights = torch.ones(cells + 1, len(channels), len(bands))
    for cell in range(cells):
        individual_weights[cell + 1].view(-1)[cell] = 0.0
    output = _resolve(str(config["output"]["root"])) / args.task / f"seed-{args.seed}"
    gate_result_path = output / "result.json"
    if gate_result_path.exists():
        raise DatasetProtocolError(f"refusing to overwrite joint gate {gate_result_path}")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = MultiHeadCBraMod(CBraMod(), tasks=tuple(config["canonical_tasks"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device)
    if args.validate_only:
        print(json.dumps({"config_sha256": config_sha, "task": args.task, "seed": args.seed, "samples": len(dataset), "checkpoint_sha256": joint_result["checkpoint"]["sha256"]}, indent=2))
        return
    set_determinism(int(config["xai"]["seed"]))
    output.mkdir(parents=True, exist_ok=True)
    scores = score_spectral_masks(
        model,
        args.task,
        dataset,
        range(len(dataset)),
        individual_weights,
        sampling_rate_hz=float(config["cell_registry"]["sampling_rate_hz"]),
        bands=bands,
        batch_size=int(config["xai"]["batch_size"]),
        mask_chunk_size=int(config["xai"]["mask_chunk_size"]),
        device=device,
    )
    margins = np.asarray(scores["margins"], dtype=np.float64)
    truth = np.asarray(scores["truth"], dtype=np.int64)
    subject_array = np.asarray(scores["subjects"], dtype=np.str_)
    fit_group, fit_subject = subject_class_balanced_margin_drop(
        margins[0, fit], margins[1:, fit], truth[fit], subject_array[fit]
    )
    gate_group, gate_subject = subject_class_balanced_margin_drop(
        margins[0, gate], margins[1:, gate], truth[gate], subject_array[gate]
    )
    subject_ids = sorted(fit_subject)
    cosines = {subject: cosine_similarity(fit_subject[subject], gate_subject[subject]) for subject in subject_ids}
    values = np.asarray(list(cosines.values()), dtype=np.float64)
    reliability = {
        "all_subject_cosines_finite": bool(np.isfinite(values).all()),
        "median_subject_cosine": float(np.median(values)),
        "subject_fraction_at_or_above_threshold": float(np.mean(values >= float(config["gate"]["subject_threshold"]))),
    }
    reliability["passed"] = bool(
        reliability["all_subject_cosines_finite"]
        and reliability["median_subject_cosine"] >= float(config["gate"]["minimum_median_subject_cosine"])
        and reliability["subject_fraction_at_or_above_threshold"] >= float(config["gate"]["minimum_subject_fraction_at_threshold"])
    )
    map_path = output / "joint-map.npz"
    _atomic_npz(
        map_path,
        {
            "subject_ids": np.asarray(subject_ids, dtype=np.str_),
            "fit_subject_maps": np.stack([fit_subject[s] for s in subject_ids]),
            "gate_subject_maps": np.stack([gate_subject[s] for s in subject_ids]),
            "fit_group_map": fit_group,
            "gate_group_map": gate_group,
            "config_sha256": np.asarray(config_sha),
            "checkpoint_sha256": np.asarray(joint_result["checkpoint"]["sha256"]),
            "task": np.asarray(args.task),
        },
    )
    fidelity = {"ran": False, "passed": False}
    fidelity_path = None
    top_cells = None
    if reliability["passed"]:
        fidelity_config = task_config["fidelity"]
        top = tuple(sorted(int(v) for v in np.argsort(fit_group)[-int(fidelity_config["selected_cells"]):]))
        top_cells = list(top)
        random = _random_masks(cells, len(top), int(fidelity_config["random_masks"]), int(fidelity_config["random_seed"]), top)
        indicators = np.zeros((2 + len(random), cells), dtype=np.float32)
        indicators[1, list(top)] = 1.0
        indicators[2:] = random
        fidelity_weights = torch.from_numpy(1.0 - indicators).reshape(-1, len(channels), len(bands))
        fidelity_scores = score_spectral_masks(
            model,
            args.task,
            dataset,
            gate,
            fidelity_weights,
            sampling_rate_hz=float(config["cell_registry"]["sampling_rate_hz"]),
            bands=bands,
            batch_size=int(config["xai"]["batch_size"]),
            mask_chunk_size=int(config["xai"]["mask_chunk_size"]),
            device=device,
        )
        fidelity = _fidelity(fidelity_scores, float(fidelity_config["percentile"]))
        fidelity["ran"] = True
        fidelity_path = output / "fidelity-scores.npz"
        _atomic_npz(fidelity_path, {**fidelity_scores, "config_sha256": np.asarray(config_sha)})
    result = {
        "schema_version": 1,
        "run": config["id"],
        "config_sha256": config_sha,
        "task": args.task,
        "seed": int(args.seed),
        "joint_checkpoint_sha256": joint_result["checkpoint"]["sha256"],
        "test_was_loaded": False,
        "reliability": reliability,
        "subject_split_half_cosine": cosines,
        "fidelity": fidelity,
        "top_cell_indices": top_cells,
        "gate_passed": bool(reliability["passed"] and fidelity["passed"]),
        "artifacts": {
            "map": {"file": map_path.name, "sha256": sha256_file(map_path)},
            "fidelity": (
                {"file": fidelity_path.name, "sha256": sha256_file(fidelity_path)}
                if fidelity_path is not None
                else None
            ),
        },
        "device": str(device),
        "torch": torch.__version__,
    }
    _atomic_json(gate_result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
