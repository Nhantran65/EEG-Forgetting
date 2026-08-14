from __future__ import annotations

import hashlib
from collections import OrderedDict, deque
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader

from eeg_forgetting.data.cache import CachedEEGDataset
from eeg_forgetting.data.contracts import DatasetProtocolError, load_yaml
from eeg_forgetting.data.manifests import sha256_file
from eeg_forgetting.models.cbramod import load_pretrained_backbone

from .continual import MultiHeadCBraMod, _atomic_npz, evaluate_task
from .continual_methods import _atomic_json, _atomic_torch, _optimizer_and_loader
from .fisher import encoder_layer_key, layer_l2_normalize, load_fisher
from .pilot import TASK_CLASSES, set_determinism


def _layer_parameter_names(values: Mapping[str, Tensor]) -> dict[str, list[str]]:
    layers: dict[str, list[str]] = {}
    for name in values:
        layers.setdefault(encoder_layer_key(name), []).append(name)
    return layers


def build_freeze_masks(
    left_fisher: Mapping[str, Tensor],
    right_fisher: Mapping[str, Tensor],
    *,
    ratio: float,
    condition: str,
    random_seed: int,
) -> tuple[OrderedDict[str, Tensor], dict[str, int]]:
    if left_fisher.keys() != right_fisher.keys() or not left_fisher:
        raise DatasetProtocolError("intervention Fisher signatures do not align")
    if not 0.0 < ratio < 1.0:
        raise DatasetProtocolError("intervention freeze ratio must be within (0,1)")
    if condition != "high_overlap" and not condition.startswith("random_"):
        raise DatasetProtocolError(f"unknown intervention condition {condition!r}")
    left = layer_l2_normalize(left_fisher)
    right = layer_l2_normalize(right_fisher)
    masks = OrderedDict(
        (name, torch.zeros_like(value, dtype=torch.bool)) for name, value in left.items()
    )
    counts = {}
    for layer_index, (layer, names) in enumerate(
        sorted(_layer_parameter_names(left).items()), start=1
    ):
        sizes = [left[name].numel() for name in names]
        total = sum(sizes)
        count = max(1, int(round(total * ratio)))
        if condition == "high_overlap":
            scores = torch.cat(
                [
                    torch.sqrt(
                        left[name].float().clamp_min(0)
                        * right[name].float().clamp_min(0)
                    ).reshape(-1)
                    for name in names
                ]
            )
            selected = torch.topk(scores, k=count, largest=True, sorted=False).indices
        else:
            layer_generator = torch.Generator(device="cpu").manual_seed(
                int(random_seed) + 1009 * layer_index
            )
            selected = torch.randperm(total, generator=layer_generator)[:count]
        offset = 0
        for name, size in zip(names, sizes, strict=True):
            local = selected[(selected >= offset) & (selected < offset + size)] - offset
            masks[name].view(-1)[local] = True
            offset += size
        counts[layer] = count
    return masks, counts


def freeze_mask_sha256(masks: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name, mask in masks.items():
        digest.update(name.encode("utf-8"))
        digest.update(np.asarray(mask.shape, dtype="<i8").tobytes())
        digest.update(mask.cpu().numpy().tobytes())
    return digest.hexdigest()


def restore_frozen_parameters(
    named_parameters: Mapping[str, Tensor],
    masks: Mapping[str, Tensor],
    anchors: Mapping[str, Tensor],
) -> None:
    """Undo optimizer-side changes (including AdamW decay) at frozen elements."""
    if masks.keys() != anchors.keys():
        raise DatasetProtocolError("intervention masks and anchors do not align")
    missing = [name for name in masks if name not in named_parameters]
    if missing:
        raise DatasetProtocolError(
            f"intervention model is missing masked parameter {missing[0]}"
        )
    with torch.no_grad():
        for name, mask in masks.items():
            parameter = named_parameters[name]
            anchor = anchors[name]
            if parameter.shape != mask.shape or parameter.shape != anchor.shape:
                raise DatasetProtocolError(
                    f"intervention state shape mismatch for {name}"
                )
            parameter[mask] = anchor[mask]


def _forgetting(
    before: float,
    after: float,
    *,
    classes: int,
    minimum_headroom: float,
) -> dict[str, object]:
    chance = 1.0 / classes
    headroom = before - chance
    valid = headroom >= minimum_headroom
    raw = before - after
    return {
        "before": before,
        "after": after,
        "chance": chance,
        "headroom": headroom,
        "raw_forgetting": raw,
        "relative_forgetting": raw / headroom if valid else None,
        "relative_valid": valid,
    }


def run_overlap_freeze_intervention(
    *,
    config_path: str | Path,
    seed: int,
    ratio: float,
    condition: str,
    cache_root: str | Path,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    output_dir: str | Path,
    device: str | torch.device,
) -> dict[str, object]:
    config_path = Path(config_path)
    config = load_yaml(config_path)
    if config.get("status") != "locked_intervention":
        raise DatasetProtocolError("intervention config is not locked")
    if int(seed) not in tuple(int(value) for value in config["seeds"]):
        raise DatasetProtocolError(f"intervention seed {seed} is not declared")
    ratios = tuple(float(value) for value in config["mask"]["freeze_ratios"])
    if float(ratio) not in ratios:
        raise DatasetProtocolError(f"freeze ratio {ratio} is not declared")
    random_controls = int(config["mask"]["random_controls_per_ratio"])
    if condition != "high_overlap":
        valid_random = {f"random_{index}" for index in range(random_controls)}
        if condition not in valid_random:
            raise DatasetProtocolError(f"intervention condition {condition!r} is not declared")
    output_dir = Path(output_dir)
    if (output_dir / "result.json").exists():
        raise DatasetProtocolError(f"refusing to overwrite intervention {output_dir}")

    source_pattern = str(config["source_stage"]["path_pattern"])
    source_path = Path(source_pattern.format(seed=int(seed)))
    expected_source = str(config["source_stage"]["sha256_by_seed"][int(seed)])
    if sha256_file(source_path) != expected_source:
        raise DatasetProtocolError("intervention source checkpoint digest mismatch")
    old_task = str(config["old_task"])
    new_task = str(config["new_task"])
    if old_task == new_task or old_task not in TASK_CLASSES or new_task not in TASK_CLASSES:
        raise DatasetProtocolError("intervention must declare two distinct known tasks")
    fisher_config = config["fisher"]
    if "tasks" in fisher_config:
        task_fishers = fisher_config["tasks"]
        left_path = Path(str(task_fishers[old_task]["path"]))
        right_path = Path(str(task_fishers[new_task]["path"]))
        expected_fisher = {
            old_task: str(task_fishers[old_task]["sha256"]),
            new_task: str(task_fishers[new_task]["sha256"]),
        }
    else:
        # Backward-compatible reader for the already completed BCI-to-Sleep config.
        left_path = Path(str(fisher_config[f"{old_task}_path"]))
        right_path = Path(str(fisher_config[f"{new_task}_path"]))
        expected_fisher = {
            old_task: str(fisher_config[f"{old_task}_sha256"]),
            new_task: str(fisher_config[f"{new_task}_sha256"]),
        }
    for task, path in ((old_task, left_path), (new_task, right_path)):
        if sha256_file(path) != expected_fisher[task]:
            raise DatasetProtocolError(f"{task} intervention Fisher digest mismatch")
    ratio_index = ratios.index(float(ratio))
    random_index = 0 if condition == "high_overlap" else int(condition.split("_")[1])
    mask_seed = int(config["mask"]["random_seed_base"]) + 100 * ratio_index + random_index
    masks, layer_counts = build_freeze_masks(
        load_fisher(left_path),
        load_fisher(right_path),
        ratio=float(ratio),
        condition=condition,
        random_seed=mask_seed,
    )

    set_determinism(int(seed))
    device = torch.device(device)
    torch.cuda.set_device(device)
    cache_root = Path(cache_root)
    train_data = CachedEEGDataset(cache_root / new_task / "train" / "index.json")
    evaluation_sets = {
        split: {
            task: CachedEEGDataset(cache_root / task / split / "index.json")
            for task in (old_task, new_task)
        }
        for split in ("validation", "test")
    }
    evaluation_loaders = {
        split: {
            task: DataLoader(
                dataset,
                batch_size=int(config["training"]["batch_size"]),
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )
            for task, dataset in datasets.items()
        }
        for split, datasets in evaluation_sets.items()
    }
    backbone = load_pretrained_backbone(
        checkpoint_path, expected_sha256=checkpoint_sha256, map_location="cpu"
    )
    model = MultiHeadCBraMod(backbone).to(device)
    source = torch.load(source_path, map_location="cpu", weights_only=True)
    model.load_state_dict(source["model_state_dict"])
    before = {}
    for split in ("validation", "test"):
        metrics, _arrays = evaluate_task(
            model, old_task, evaluation_loaders[split][old_task], device=device
        )
        before[split] = metrics

    trainable, optimizer, scheduler, loader = _optimizer_and_loader(
        model, train_data, new_task, config, seed=int(seed)
    )
    named_parameters = dict(model.named_parameters())
    missing = [name for name in masks if name not in named_parameters]
    if missing:
        raise DatasetProtocolError(f"intervention model is missing masked parameter {missing[0]}")
    device_masks = OrderedDict((name, mask.to(device)) for name, mask in masks.items())
    anchors = OrderedDict(
        (name, named_parameters[name].detach().clone()) for name in masks
    )
    handles = [
        named_parameters[name].register_hook(
            lambda gradient, frozen=device_masks[name]: gradient.masked_fill(frozen, 0)
        )
        for name in masks
    ]
    interval = int(config["training"]["progress_interval_steps"])
    recent: deque[float] = deque(maxlen=interval)
    curve = []
    try:
        for step, (signals, labels, _subjects) in enumerate(loader, start=1):
            signals = signals.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(
                model(new_task, signals),
                labels,
                label_smoothing=float(config["training"]["label_smoothing"]),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                trainable, float(config["training"]["gradient_clip_norm"])
            )
            optimizer.step()
            restore_frozen_parameters(named_parameters, device_masks, anchors)
            scheduler.step()
            recent.append(float(loss.detach()))
            if step % interval == 0:
                point = {"step": step, "train_loss": float(np.mean(recent))}
                curve.append(point)
                print(
                    f"intervention seed={seed} ratio={ratio:g} condition={condition} "
                    f"step={step}/{config['training']['optimizer_steps_per_task']} "
                    f"loss={point['train_loss']:.5f}",
                    flush=True,
                )
    finally:
        for handle in handles:
            handle.remove()

    evaluations = {}
    predictions = {}
    for split in ("validation", "test"):
        evaluations[split] = {}
        for task in (old_task, new_task):
            metrics, arrays = evaluate_task(
                model, task, evaluation_loaders[split][task], device=device
            )
            evaluations[split][task] = metrics
            if split == "test":
                output_dir.mkdir(parents=True, exist_ok=True)
                prediction_path = output_dir / f"predictions-{task}.npz"
                _atomic_npz(prediction_path, arrays)
                predictions[task] = {
                    "file": prediction_path.name,
                    "sha256": sha256_file(prediction_path),
                }
    forgetting = {
        split: _forgetting(
            float(before[split]["mean_subject_balanced_accuracy"]),
            float(evaluations[split][old_task]["mean_subject_balanced_accuracy"]),
            classes=TASK_CLASSES[old_task],
            minimum_headroom=float(
                config["evaluation"]["minimum_valid_headroom_above_chance"]
            ),
        )
        for split in ("validation", "test")
    }
    checkpoint_output = output_dir / "final.pt"
    config_sha = sha256_file(config_path)
    _atomic_torch(
        checkpoint_output,
        {
            "model_state_dict": model.state_dict(),
            "seed": int(seed),
            "ratio": float(ratio),
            "condition": condition,
            "mask_sha256": freeze_mask_sha256(masks),
            "config_sha256": config_sha,
        },
    )
    result = {
        "schema_version": 1,
        "run": config["id"],
        "method": config["method"],
        "seed": int(seed),
        "ratio": float(ratio),
        "condition": condition,
        "config_sha256": config_sha,
        "source_checkpoint_sha256": expected_source,
        "old_task": old_task,
        "new_task": new_task,
        "fisher_sha256": expected_fisher,
        "mask": {
            "sha256": freeze_mask_sha256(masks),
            "frozen_elements": sum(layer_counts.values()),
            "frozen_by_layer": layer_counts,
            "random_seed": mask_seed if condition.startswith("random_") else None,
        },
        "before_old_task": before,
        "evaluations": evaluations,
        "forgetting": forgetting,
        "curve": curve,
        "predictions": predictions,
        "checkpoint": {
            "file": checkpoint_output.name,
            "sha256": sha256_file(checkpoint_output),
        },
        "device": str(device),
        "torch": torch.__version__,
    }
    _atomic_json(output_dir / "result.json", result)
    return result
