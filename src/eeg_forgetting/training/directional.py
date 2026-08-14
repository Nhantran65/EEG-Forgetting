from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from eeg_forgetting.data.cache import CachedEEGDataset
from eeg_forgetting.data.contracts import DatasetProtocolError, load_yaml
from eeg_forgetting.data.manifests import sha256_file
from eeg_forgetting.models.cbramod import load_pretrained_backbone

from .continual import MultiHeadCBraMod
from .continual_methods import _atomic_json
from .fisher import (
    deterministic_sample_indices,
    encoder_layer_key,
    indices_sha256,
    layer_l2_normalize,
    load_fisher,
    selected_plastic_parameters,
)
from .pilot import set_determinism


def gradient_geometry(
    left: Mapping[str, Tensor], right: Mapping[str, Tensor]
) -> dict[str, float]:
    if not left or left.keys() != right.keys():
        raise DatasetProtocolError("directional gradients do not align")
    dot = sum((left[name].double() * right[name].double()).sum() for name in left)
    left_norm = sum(left[name].double().square().sum() for name in left).sqrt()
    right_norm = sum(right[name].double().square().sum() for name in right).sqrt()
    if float(left_norm) == 0.0 or float(right_norm) == 0.0:
        raise DatasetProtocolError("directional gradient has zero norm")
    cosine = float(dot / (left_norm * right_norm))
    return {
        "cosine": cosine,
        "conflict_score": -cosine,
        "dot_product": float(dot),
        "old_gradient_l2": float(left_norm),
        "new_gradient_l2": float(right_norm),
        "old_loss_change_per_unit_new_step": float(-dot / right_norm),
    }


def weighted_update_energy(
    deltas: Mapping[str, Tensor], weights: Mapping[str, Tensor]
) -> float:
    if not deltas or deltas.keys() != weights.keys():
        raise DatasetProtocolError("directional drift and weights do not align")
    denominator = sum(weights[name].double().sum() for name in weights)
    if float(denominator) <= 0.0:
        raise DatasetProtocolError("directional drift weights must be positive")
    numerator = sum(
        (weights[name].double() * deltas[name].double().square()).sum()
        for name in weights
    )
    return float(numerator / denominator)


def _mean_task_gradient(
    model: MultiHeadCBraMod,
    task: str,
    dataset: CachedEEGDataset,
    indices: np.ndarray,
    *,
    final_blocks: int,
    batch_size: int,
    device: torch.device,
) -> OrderedDict[str, Tensor]:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    selected = selected_plastic_parameters(model, final_blocks=final_blocks)
    for parameter in selected.values():
        parameter.requires_grad = True
    accumulator = OrderedDict(
        (name, torch.zeros_like(parameter, device=device))
        for name, parameter in selected.items()
    )
    loader = DataLoader(
        Subset(dataset, [int(index) for index in indices]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    total = len(indices)
    for signals, labels, _subjects in loader:
        signals = signals.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(task, signals), labels, reduction="sum") / total
        loss.backward()
        for name, parameter in selected.items():
            if parameter.grad is None:
                raise DatasetProtocolError(f"no directional gradient for {name}")
            accumulator[name].add_(parameter.grad.detach())
    return OrderedDict((name, value.cpu()) for name, value in accumulator.items())


def _checkpoint_from_stage(
    result_path: Path,
    result: Mapping[str, object],
    *,
    stage: int,
    task: str,
) -> tuple[Path, str]:
    stage_entry = result["stages"][stage - 1]
    if int(stage_entry["stage"]) != stage or stage_entry["learned_task"] != task:
        raise DatasetProtocolError("directional source result stage mismatch")
    directory = result_path.parent / f"stage-{stage:02d}-{task}"
    stage_path = directory / "stage.json"
    if sha256_file(stage_path) != stage_entry["stage_result_sha256"]:
        raise DatasetProtocolError("directional stage document digest mismatch")
    with stage_path.open(encoding="utf-8") as handle:
        stage_document = json.load(handle)
    checkpoint = directory / stage_document["checkpoint"]["file"]
    expected = str(stage_document["checkpoint"]["sha256"])
    if sha256_file(checkpoint) != expected:
        raise DatasetProtocolError("directional checkpoint digest mismatch")
    return checkpoint, expected


def run_directional_interference(
    *,
    config_path: str | Path,
    direction: str,
    seed: int,
    cache_root: str | Path,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    output_dir: str | Path,
    device: str | torch.device,
) -> dict[str, object]:
    config_path = Path(config_path)
    config = load_yaml(config_path)
    if config.get("status") != "locked_analysis":
        raise DatasetProtocolError("directional analysis config is not locked")
    if direction not in config["directions"]:
        raise DatasetProtocolError(f"unknown directional analysis {direction!r}")
    if int(seed) not in tuple(int(value) for value in config["seeds"]):
        raise DatasetProtocolError(f"undeclared directional seed {seed}")
    output_dir = Path(output_dir)
    if (output_dir / "result.json").exists():
        raise DatasetProtocolError(f"refusing to overwrite directional result {output_dir}")
    specification = config["directions"][direction]
    old_task = str(specification["old_task"])
    new_task = str(specification["new_task"])
    result_path = Path(str(specification["result_pattern"]).format(seed=int(seed)))
    expected_result = str(specification["result_sha256_by_seed"][int(seed)])
    if sha256_file(result_path) != expected_result:
        raise DatasetProtocolError("directional continual result digest mismatch")
    with result_path.open(encoding="utf-8") as handle:
        continual_result = json.load(handle)
    source_path, source_sha = _checkpoint_from_stage(
        result_path, continual_result, stage=1, task=old_task
    )
    after_path, after_sha = _checkpoint_from_stage(
        result_path, continual_result, stage=2, task=new_task
    )
    forgetting_rows = [
        row
        for row in continual_result["pairwise_forgetting"]
        if row["old_task"] == old_task
        and row["learned_task"] == new_task
        and int(row["after_stage"]) == 2
    ]
    if len(forgetting_rows) != 1:
        raise DatasetProtocolError("directional forgetting row is not unique")

    fisher_config = config["fisher"]["tasks"]
    fishers = {}
    for task in (old_task, new_task):
        fisher_path = Path(str(fisher_config[task]["path"]))
        if sha256_file(fisher_path) != fisher_config[task]["sha256"]:
            raise DatasetProtocolError(f"directional Fisher digest mismatch for {task}")
        fishers[task] = load_fisher(fisher_path)
    old_fisher = layer_l2_normalize(fishers[old_task])
    new_fisher = layer_l2_normalize(fishers[new_task])
    shared_fisher = OrderedDict(
        (
            name,
            torch.sqrt(
                old_fisher[name].float().clamp_min(0)
                * new_fisher[name].float().clamp_min(0)
            ),
        )
        for name in old_fisher
    )
    source_state = torch.load(source_path, map_location="cpu", weights_only=True)[
        "model_state_dict"
    ]
    after_state = torch.load(after_path, map_location="cpu", weights_only=True)[
        "model_state_dict"
    ]
    deltas = OrderedDict(
        (name, after_state[name].float() - source_state[name].float())
        for name in old_fisher
    )
    uniform = OrderedDict((name, torch.ones_like(value)) for name, value in deltas.items())
    drift = {
        "unweighted_mean_squared": weighted_update_energy(deltas, uniform),
        "old_fisher_weighted_mean_squared": weighted_update_energy(deltas, old_fisher),
        "shared_fisher_weighted_mean_squared": weighted_update_energy(
            deltas, shared_fisher
        ),
    }
    drift["old_fisher_concentration_ratio"] = (
        drift["old_fisher_weighted_mean_squared"] / drift["unweighted_mean_squared"]
    )
    drift["shared_fisher_concentration_ratio"] = (
        drift["shared_fisher_weighted_mean_squared"] / drift["unweighted_mean_squared"]
    )

    set_determinism(int(seed))
    device = torch.device(device)
    torch.cuda.set_device(device)
    backbone = load_pretrained_backbone(
        checkpoint_path, expected_sha256=checkpoint_sha256, map_location="cpu"
    )
    model = MultiHeadCBraMod(backbone).to(device)
    model.load_state_dict(source_state)
    cache_root = Path(cache_root)
    datasets = {
        task: CachedEEGDataset(cache_root / task / "train" / "index.json")
        for task in (old_task, new_task)
    }
    sampling = config["sampling"]
    sample_count = int(sampling["samples_per_task"])
    indices = {
        task: deterministic_sample_indices(
            len(datasets[task]),
            sample_count=sample_count,
            seed=int(sampling["sample_seed_by_task"][task]),
        )
        for task in (old_task, new_task)
    }
    gradients = {
        task: _mean_task_gradient(
            model,
            task,
            datasets[task],
            indices[task],
            final_blocks=int(config["model"]["plastic_final_encoder_blocks"]),
            batch_size=int(sampling["batch_size"]),
            device=device,
        )
        for task in (old_task, new_task)
    }
    geometry = gradient_geometry(gradients[old_task], gradients[new_task])
    layers = sorted({encoder_layer_key(name) for name in gradients[old_task]})
    layer_geometry = {}
    for layer in layers:
        left = OrderedDict(
            (name, value)
            for name, value in gradients[old_task].items()
            if encoder_layer_key(name) == layer
        )
        right = OrderedDict(
            (name, gradients[new_task][name]) for name in left
        )
        layer_geometry[layer] = gradient_geometry(left, right)
    document = {
        "schema_version": 1,
        "run": config["id"],
        "direction": direction,
        "seed": int(seed),
        "old_task": old_task,
        "new_task": new_task,
        "config_sha256": sha256_file(config_path),
        "continual_result_sha256": expected_result,
        "source_checkpoint_sha256": source_sha,
        "after_checkpoint_sha256": after_sha,
        "sample_indices_sha256": {
            task: indices_sha256(values) for task, values in indices.items()
        },
        "samples_per_task": sample_count,
        "relative_forgetting": forgetting_rows[0]["relative_forgetting"],
        "raw_forgetting": forgetting_rows[0]["raw_forgetting"],
        "gradient_geometry": geometry,
        "gradient_geometry_by_layer": layer_geometry,
        "update_drift": drift,
        "device": str(device),
        "torch": torch.__version__,
    }
    _atomic_json(output_dir / "result.json", document)
    return document
