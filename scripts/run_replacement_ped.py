#!/usr/bin/env python3
"""Score one replacement matrix cell and compute noise-corrected PED."""

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
    frozen_stratified_halves,
    noise_corrected_symmetric_jsd,
    score_spectral_masks,
    subject_class_balanced_margin_drop,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "xai" / "high_gamma_replacement_ped_v1.yaml",
    )
    parser.add_argument(
        "--method", choices=("sequential_finetuning", "ewc", "derpp"), required=True
    )
    parser.add_argument("--order", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _verify(path: Path, expected: str, role: str) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise DatasetProtocolError(f"{role} digest mismatch: {path}")


def _load_authority(
    config: dict[str, object], method: str, order_name: str, seed: int
) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]], Path]:
    method_spec = config["methods"][method]
    training_path = _resolve(str(method_spec["training_config"]))
    summary_path = _resolve(str(method_spec["summary"]))
    _verify(training_path, str(method_spec["training_config_sha256"]), "training config")
    _verify(summary_path, str(method_spec["summary_sha256"]), "method summary")
    training = load_yaml(training_path)
    if tuple(training["canonical_tasks"]) != tuple(config["canonical_tasks"]):
        raise DatasetProtocolError("PED training task registry mismatch")
    if order_name not in training["orders"] or int(seed) not in training["seeds"]:
        raise DatasetProtocolError("PED run cell is not declared")
    order = list(training["orders"][order_name])
    result_path = (
        _resolve(str(method_spec["result_root"]))
        / order_name
        / f"seed-{seed}"
        / "result.json"
    )
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    run_key = f"{order_name}:seed-{seed}"
    _verify(result_path, str(summary["input_result_sha256"][run_key]), "matrix result")
    with result_path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    if (
        result["config_sha256"] != method_spec["training_config_sha256"]
        or result["order"] != order
        or int(result["seed"]) != int(seed)
    ):
        raise DatasetProtocolError("PED matrix result identity mismatch")
    stages = []
    for stage in result["stages"]:
        stage_dir = result_path.parent / f"stage-{stage['stage']:02d}-{stage['learned_task']}"
        stage_path = stage_dir / "stage.json"
        _verify(stage_path, str(stage["stage_result_sha256"]), "stage result")
        with stage_path.open(encoding="utf-8") as handle:
            document = json.load(handle)
        checkpoint_path = stage_dir / str(document["checkpoint"]["file"])
        _verify(checkpoint_path, str(document["checkpoint"]["sha256"]), "stage checkpoint")
        if (
            document["config_sha256"] != method_spec["training_config_sha256"]
            or int(document["stage"]) != int(stage["stage"])
            or document["learned_task"] != stage["learned_task"]
        ):
            raise DatasetProtocolError("PED stage identity mismatch")
        document["_checkpoint_path"] = checkpoint_path
        stages.append(document)
    return training, result, stages, result_path


def _task_data(
    config: dict[str, object], task: str
) -> tuple[CachedEEGDataset, dict[str, object]]:
    task_config = config["tasks"][task]
    if task_config["status"] not in {
        "enabled_passed_reliability_and_fidelity",
        "enabled_passed_exploratory_joint_gate",
    }:
        raise DatasetProtocolError(f"PED task {task!r} is not enabled")
    index_path = _resolve(str(task_config["validation_index"]))
    _verify(index_path, str(task_config["validation_index_sha256"]), "validation cache")
    dataset = CachedEEGDataset(index_path)
    labels, subjects = [], []
    for index in range(len(dataset)):
        _signal, label, subject = dataset[index]
        labels.append(int(label))
        subjects.append(str(subject))
    if set(subjects) != set(task_config["validation_subjects"]):
        raise DatasetProtocolError("PED validation subjects changed")
    split_config = task_config["split"]
    split = frozen_stratified_halves(
        labels, subjects, seed=int(split_config["seed"])
    )
    if (
        split["assignment_sha256"] != split_config["assignment_sha256"]
        or len(split["attribution_fit"]) != int(split_config["fit_samples"])
        or len(split["attribution_gate"]) != int(split_config["gate_samples"])
    ):
        raise DatasetProtocolError("PED attribution split changed")
    return dataset, split


def _load_map(
    path: Path, *, config_sha: str, checkpoint_sha: str, task: str
) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        if (
            str(archive["config_sha256"].item()) != config_sha
            or str(archive["checkpoint_sha256"].item()) != checkpoint_sha
            or str(archive["task"].item()) != task
        ):
            raise DatasetProtocolError(f"PED map identity mismatch: {path}")
        return {
            "subject_ids": np.asarray(archive["subject_ids"], dtype=np.str_),
            "fit_subject_maps": np.asarray(archive["fit_subject_maps"], dtype=np.float64),
            "gate_subject_maps": np.asarray(archive["gate_subject_maps"], dtype=np.float64),
            "fit_group_map": np.asarray(archive["fit_group_map"], dtype=np.float64),
            "gate_group_map": np.asarray(archive["gate_group_map"], dtype=np.float64),
        }


def _score_map(
    *,
    model: MultiHeadCBraMod,
    task: str,
    dataset: CachedEEGDataset,
    split: dict[str, object],
    weights: torch.Tensor,
    bands: tuple[tuple[str, float, float], ...],
    config: dict[str, object],
    config_sha: str,
    checkpoint_sha: str,
    output_path: Path,
    device: torch.device,
) -> dict[str, object]:
    if output_path.exists():
        return _load_map(
            output_path,
            config_sha=config_sha,
            checkpoint_sha=checkpoint_sha,
            task=task,
        )
    print(f"score task={task} rows={len(dataset)} masks={len(weights)}", flush=True)
    scores = score_spectral_masks(
        model,
        task,
        dataset,
        range(len(dataset)),
        weights,
        sampling_rate_hz=float(config["cell_registry"]["sampling_rate_hz"]),
        bands=bands,
        batch_size=int(config["evaluation"]["batch_size"]),
        mask_chunk_size=int(config["evaluation"]["mask_chunk_size"]),
        device=device,
    )
    margins = np.asarray(scores["margins"], dtype=np.float64)
    truth = np.asarray(scores["truth"], dtype=np.int64)
    subjects = np.asarray(scores["subjects"], dtype=np.str_)
    fit = np.asarray(split["attribution_fit"], dtype=np.int64)
    gate = np.asarray(split["attribution_gate"], dtype=np.int64)
    fit_group, fit_subject = subject_class_balanced_margin_drop(
        margins[0, fit], margins[1:, fit], truth[fit], subjects[fit]
    )
    gate_group, gate_subject = subject_class_balanced_margin_drop(
        margins[0, gate], margins[1:, gate], truth[gate], subjects[gate]
    )
    subject_ids = np.asarray(sorted(fit_subject), dtype=np.str_)
    if set(subject_ids) != set(gate_subject):
        raise DatasetProtocolError("PED fit/gate subjects differ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_npz(
        output_path,
        {
            "subject_ids": subject_ids,
            "fit_subject_maps": np.stack([fit_subject[str(value)] for value in subject_ids]),
            "gate_subject_maps": np.stack([gate_subject[str(value)] for value in subject_ids]),
            "fit_group_map": fit_group,
            "gate_group_map": gate_group,
            "config_sha256": np.asarray(config_sha),
            "checkpoint_sha256": np.asarray(checkpoint_sha),
            "task": np.asarray(task),
        },
    )
    return _load_map(
        output_path,
        config_sha=config_sha,
        checkpoint_sha=checkpoint_sha,
        task=task,
    )


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if config.get("status") not in {"locked_xai_scale", "exploratory_xai_scale"}:
        raise DatasetProtocolError("replacement PED config is not locked")
    if config["scale"].get("test_access_for_xai") != "forbidden":
        raise DatasetProtocolError("replacement PED must remain validation-only")
    config_sha = sha256_file(args.config)
    training, matrix_result, stages, _result_path = _load_authority(
        config, args.method, args.order, args.seed
    )
    order = list(training["orders"][args.order])
    old_tasks = set(config["scale"]["old_tasks"])
    transitions = []
    required_maps: set[tuple[int, str]] = set()
    for after_stage in range(2, len(order) + 1):
        learned_task = order[after_stage - 1]
        for old_task in order[: after_stage - 1]:
            if old_task not in old_tasks:
                continue
            transitions.append(
                {
                    "old_task": old_task,
                    "learned_task": learned_task,
                    "before_stage": after_stage - 1,
                    "after_stage": after_stage,
                }
            )
            required_maps.add((after_stage - 1, old_task))
            required_maps.add((after_stage, old_task))
    task_cache = {task: _task_data(config, task) for task in sorted(old_tasks)}
    if args.validate_only:
        print(
            json.dumps(
                {
                    "config_sha256": config_sha,
                    "method": args.method,
                    "order": args.order,
                    "seed": args.seed,
                    "required_maps": sorted([list(value) for value in required_maps]),
                    "transitions": transitions,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    set_determinism(int(config["seed"]))
    channels = tuple(config["cell_registry"]["channels"])
    bands = tuple(
        (str(value["name"]), float(value["low_hz"]), float(value["high_hz"]))
        for value in config["cell_registry"]["bands"]
    )
    cells = len(channels) * len(bands)
    weights = torch.ones(cells + 1, len(channels), len(bands))
    for cell in range(cells):
        weights[cell + 1].view(-1)[cell] = 0.0
    output_dir = (
        _resolve(str(config["output"]["root"]))
        / args.method
        / args.order
        / f"seed-{args.seed}"
    )
    result_path = output_dir / "result.json"
    if result_path.exists():
        raise DatasetProtocolError(f"refusing to overwrite PED result {result_path}")
    maps: dict[tuple[int, str], dict[str, object]] = {}
    artifacts = {}
    for stage_number in sorted({value[0] for value in required_maps}):
        stage = stages[stage_number - 1]
        checkpoint_path = Path(stage["_checkpoint_path"])
        checkpoint_sha = str(stage["checkpoint"]["sha256"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (
            checkpoint.get("config_sha256")
            != config["methods"][args.method]["training_config_sha256"]
            or int(checkpoint.get("stage", -1)) != stage_number
            or checkpoint.get("task") != stage["learned_task"]
        ):
            raise DatasetProtocolError("PED checkpoint identity mismatch")
        model = MultiHeadCBraMod(CBraMod(), tasks=tuple(config["canonical_tasks"]))
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.to(device)
        for task in sorted(value[1] for value in required_maps if value[0] == stage_number):
            map_path = output_dir / "maps" / f"stage-{stage_number:02d}-{task}.npz"
            dataset, split = task_cache[task]
            maps[(stage_number, task)] = _score_map(
                model=model,
                task=task,
                dataset=dataset,
                split=split,
                weights=weights,
                bands=bands,
                config=config,
                config_sha=config_sha,
                checkpoint_sha=checkpoint_sha,
                output_path=map_path,
                device=device,
            )
            artifacts[f"stage-{stage_number:02d}-{task}"] = {
                "file": str(map_path.relative_to(output_dir)),
                "sha256": sha256_file(map_path),
                "checkpoint_sha256": checkpoint_sha,
            }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    drift_rows = []
    for transition in transitions:
        before = maps[(transition["before_stage"], transition["old_task"])]
        after = maps[(transition["after_stage"], transition["old_task"])]
        before_subjects = [str(value) for value in before["subject_ids"]]
        after_subjects = [str(value) for value in after["subject_ids"]]
        if before_subjects != after_subjects:
            raise DatasetProtocolError("PED before/after subjects changed")
        subject_rows = {}
        for index, subject in enumerate(before_subjects):
            subject_rows[subject] = noise_corrected_symmetric_jsd(
                before["fit_subject_maps"][index],
                before["gate_subject_maps"][index],
                after["fit_subject_maps"][index],
                after["gate_subject_maps"][index],
            )
        group = noise_corrected_symmetric_jsd(
            before["fit_group_map"],
            before["gate_group_map"],
            after["fit_group_map"],
            after["gate_group_map"],
        )
        performance = next(
            row
            for row in matrix_result["pairwise_forgetting"]
            if row["old_task"] == transition["old_task"]
            and row["learned_task"] == transition["learned_task"]
            and int(row["before_stage"]) == int(transition["before_stage"])
            and int(row["after_stage"]) == int(transition["after_stage"])
        )
        drift_rows.append(
            {
                **transition,
                "performance": performance,
                "group": group,
                "subjects": subject_rows,
            }
        )
    result = {
        "schema_version": 1,
        "run": config["id"],
        "config_sha256": config_sha,
        "method": args.method,
        "order_name": args.order,
        "order": order,
        "seed": int(args.seed),
        "test_was_loaded_for_xai": False,
        "transitions": drift_rows,
        "artifacts": artifacts,
        "device": str(device),
        "torch": torch.__version__,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
