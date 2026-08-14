from __future__ import annotations

import copy
import json
import os
from collections import Counter, OrderedDict, deque
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from eeg_forgetting.data.cache import CachedEEGDataset
from eeg_forgetting.data.contracts import DatasetProtocolError, load_yaml
from eeg_forgetting.data.manifests import sha256_file
from eeg_forgetting.models.cbramod import load_pretrained_backbone

from .continual import (
    CANONICAL_TASKS,
    MultiHeadCBraMod,
    _atomic_json as _atomic_continual_json,
    _atomic_npz,
    _atomic_torch_save,
    evaluate_task,
    pairwise_forgetting,
)
from .fisher import (
    deterministic_sample_indices,
    diagonal_empirical_fisher,
    indices_sha256,
    load_fisher,
)
from .pilot import (
    FixedStepBatchSampler,
    PilotSettings,
    TASK_CLASSES,
    TASK_SHAPES,
    set_determinism,
)


PAIR_TASKS = ("bciciv2a", "sleep_edf_sc")
_TASK_TO_INDEX = {task: index for index, task in enumerate(CANONICAL_TASKS)}


def ewc_quadratic_penalty(
    named_parameters: Mapping[str, Tensor],
    fisher: Mapping[str, Tensor],
    anchors: Mapping[str, Tensor],
) -> Tensor:
    """Return sum(F * (theta - theta*)^2) for aligned plastic parameters."""
    if not fisher or fisher.keys() != anchors.keys():
        raise DatasetProtocolError("EWC needs aligned non-empty Fisher and anchor states")
    missing = [name for name in fisher if name not in named_parameters]
    if missing:
        raise DatasetProtocolError(f"EWC model is missing protected parameter {missing[0]}")
    terms = []
    for name in fisher:
        parameter = named_parameters[name]
        importance = fisher[name]
        anchor = anchors[name]
        if parameter.shape != importance.shape or parameter.shape != anchor.shape:
            raise DatasetProtocolError(f"EWC state shape mismatch for {name}")
        terms.append((importance * (parameter - anchor).square()).sum())
    return torch.stack(terms).sum()


class ByteCappedReservoir:
    """Fixed-slot reservoir whose allocation never exceeds a declared byte cap.

    Slots pad signals and logits to the largest locked task. This makes every
    item the same size, so Algorithm R remains exactly uniform even though EEG
    tasks have different sample shapes.
    """

    _COUNTER_BYTES = 16  # signed 64-bit ``size`` and ``seen`` counters

    def __init__(self, byte_cap: int, *, seed: int):
        if byte_cap < 0:
            raise DatasetProtocolError("replay byte cap cannot be negative")
        self.byte_cap = int(byte_cap)
        self.max_signal_values = max(
            channels * patches * 200 for channels, patches in TASK_SHAPES.values()
        )
        self.max_classes = max(TASK_CLASSES.values())
        self.slot_bytes = (
            self.max_signal_values * torch.tensor([], dtype=torch.float32).element_size()
            + self.max_classes * torch.tensor([], dtype=torch.float32).element_size()
            + torch.tensor([], dtype=torch.int64).element_size()
            + torch.tensor([], dtype=torch.int8).element_size()
        )
        usable = max(0, self.byte_cap - self._COUNTER_BYTES)
        self.capacity = usable // self.slot_bytes
        self.signals = torch.empty(
            (self.capacity, self.max_signal_values), dtype=torch.float32
        )
        self.logits = torch.empty((self.capacity, self.max_classes), dtype=torch.float32)
        self.labels = torch.empty(self.capacity, dtype=torch.int64)
        self.tasks = torch.empty(self.capacity, dtype=torch.int8)
        self.size = 0
        self.seen = 0
        self._generator = np.random.default_rng(seed)

    @property
    def allocated_bytes(self) -> int:
        tensor_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.signals, self.logits, self.labels, self.tasks)
        )
        return self._COUNTER_BYTES + tensor_bytes if self.capacity else 0

    def _write(
        self,
        slot: int,
        *,
        task: str,
        signal: Tensor,
        label: Tensor | int,
        logits: Tensor,
    ) -> None:
        if task not in _TASK_TO_INDEX:
            raise DatasetProtocolError(f"cannot replay unknown task {task!r}")
        expected_values = int(np.prod((*TASK_SHAPES[task], 200)))
        flat_signal = signal.detach().float().cpu().reshape(-1)
        flat_logits = logits.detach().float().cpu().reshape(-1)
        if flat_signal.numel() != expected_values:
            raise DatasetProtocolError(f"replay signal shape mismatch for {task}")
        if flat_logits.numel() != TASK_CLASSES[task]:
            raise DatasetProtocolError(f"replay logit shape mismatch for {task}")
        self.signals[slot].zero_()
        self.logits[slot].zero_()
        self.signals[slot, :expected_values].copy_(flat_signal)
        self.logits[slot, : TASK_CLASSES[task]].copy_(flat_logits)
        self.labels[slot] = int(label)
        self.tasks[slot] = _TASK_TO_INDEX[task]

    def add_batch(
        self, task: str, signals: Tensor, labels: Tensor, logits: Tensor
    ) -> None:
        if signals.shape[0] != labels.shape[0] or labels.shape[0] != logits.shape[0]:
            raise DatasetProtocolError("replay batch fields have different lengths")
        for signal, label, response in zip(signals, labels, logits, strict=True):
            self.seen += 1
            if self.capacity == 0:
                continue
            if self.size < self.capacity:
                slot = self.size
                self.size += 1
            else:
                slot = int(self._generator.integers(0, self.seen))
                if slot >= self.capacity:
                    continue
            self._write(
                slot, task=task, signal=signal, label=label, logits=response
            )

    def sample(self, batch_size: int) -> dict[str, dict[str, Tensor]]:
        if batch_size <= 0:
            raise DatasetProtocolError("replay batch size must be positive")
        if self.size == 0:
            return {}
        count = min(int(batch_size), self.size)
        indices = self._generator.choice(self.size, size=count, replace=False)
        grouped: dict[str, list[int]] = {}
        for index in indices:
            task = CANONICAL_TASKS[int(self.tasks[int(index)])]
            grouped.setdefault(task, []).append(int(index))
        batches: dict[str, dict[str, Tensor]] = {}
        for task, slots in grouped.items():
            slot_tensor = torch.as_tensor(slots, dtype=torch.long)
            channels, patches = TASK_SHAPES[task]
            values = channels * patches * 200
            batches[task] = {
                "signals": self.signals[slot_tensor, :values].reshape(
                    len(slots), channels, patches, 200
                ),
                "labels": self.labels[slot_tensor],
                "logits": self.logits[slot_tensor, : TASK_CLASSES[task]],
            }
        return batches

    def inventory(self) -> dict[str, object]:
        counts = Counter(
            CANONICAL_TASKS[int(value)] for value in self.tasks[: self.size].tolist()
        )
        duration_seconds = sum(
            count * TASK_SHAPES[task][1] for task, count in counts.items()
        )
        return {
            "byte_cap": self.byte_cap,
            "allocated_bytes": self.allocated_bytes,
            "slot_bytes": self.slot_bytes,
            "capacity_items": self.capacity,
            "stored_items": self.size,
            "seen_items": self.seen,
            "items_by_task": dict(sorted(counts.items())),
            "stored_eeg_seconds": int(duration_seconds),
            "policy": "algorithm_r_uniform_fixed_padded_slots",
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "byte_cap": self.byte_cap,
            "signals": self.signals,
            "logits": self.logits,
            "labels": self.labels,
            "tasks": self.tasks,
            "size": self.size,
            "seen": self.seen,
            "generator_state": copy.deepcopy(self._generator.bit_generator.state),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if int(state["byte_cap"]) != self.byte_cap:
            raise DatasetProtocolError("replay checkpoint byte cap mismatch")
        for name in ("signals", "logits", "labels", "tasks"):
            source = state[name]
            target = getattr(self, name)
            if not isinstance(source, Tensor) or source.shape != target.shape:
                raise DatasetProtocolError(f"replay checkpoint shape mismatch for {name}")
            target.copy_(source)
        self.size = int(state["size"])
        self.seen = int(state["seen"])
        if not 0 <= self.size <= self.capacity or self.seen < self.size:
            raise DatasetProtocolError("invalid replay checkpoint counters")
        self._generator.bit_generator.state = copy.deepcopy(state["generator_state"])


def derpp_replay_loss(
    model: MultiHeadCBraMod,
    batches: Mapping[str, Mapping[str, Tensor]],
    *,
    device: torch.device,
    target: str,
    label_smoothing: float,
) -> Tensor:
    if target not in {"logits", "labels"}:
        raise DatasetProtocolError(f"unknown DER++ replay target {target!r}")
    losses = []
    weights = []
    for task, batch in batches.items():
        signals = batch["signals"].to(device, non_blocking=True)
        predicted = model(task, signals)
        if target == "logits":
            loss = F.mse_loss(
                predicted, batch["logits"].to(device, non_blocking=True)
            )
        else:
            loss = F.cross_entropy(
                predicted,
                batch["labels"].to(device, non_blocking=True),
                label_smoothing=label_smoothing,
            )
        losses.append(loss)
        weights.append(signals.shape[0])
    if not losses:
        return torch.zeros((), device=device)
    total = sum(weights)
    return sum(loss * (weight / total) for loss, weight in zip(losses, weights, strict=True))


def _atomic_json(path: Path, document: Mapping[str, object]) -> None:
    if path.exists():
        raise DatasetProtocolError(f"refusing to overwrite method pilot result {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_torch(path: Path, document: object) -> None:
    if path.exists():
        raise DatasetProtocolError(f"refusing to overwrite method pilot checkpoint {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(document, temporary)
    temporary.replace(path)


def _load_pair_data(cache_root: Path) -> tuple[dict[str, CachedEEGDataset], dict[str, DataLoader]]:
    train = {
        task: CachedEEGDataset(cache_root / task / "train" / "index.json")
        for task in PAIR_TASKS
    }
    validation_sets = {
        task: CachedEEGDataset(cache_root / task / "validation" / "index.json")
        for task in PAIR_TASKS
    }
    validation = {
        task: DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
        for task, dataset in validation_sets.items()
    }
    return train, validation


def _optimizer_and_loader(
    model: MultiHeadCBraMod,
    dataset: CachedEEGDataset,
    task: str,
    config: Mapping[str, object],
    *,
    seed: int,
) -> tuple[list[Tensor], torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, DataLoader]:
    training = config["training"]
    depth = int(config["model"]["plastic_final_encoder_blocks"])
    trainable = model.prepare_stage(task, depth=depth)
    backbone_parameters = [
        parameter for parameter in model.backbone.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": float(training["backbone_learning_rate"])},
            {"params": model.heads[task].parameters(), "lr": float(training["current_head_learning_rate"])},
        ],
        weight_decay=float(training["weight_decay"]),
    )
    steps = int(training["optimizer_steps_per_task"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=float(training["minimum_learning_rate"])
    )
    settings = PilotSettings(
        dataset=task,
        depth=depth,
        head="flatten_mlp",
        seed=seed,
        batch_size=int(training["batch_size"]),
        optimizer_steps=steps,
        validation_interval_steps=int(training["progress_interval_steps"]),
    )
    loader = DataLoader(
        dataset,
        batch_sampler=FixedStepBatchSampler(len(dataset), settings),
        num_workers=0,
        pin_memory=True,
    )
    return trainable, optimizer, scheduler, loader


def _pair_forgetting(before: float, after: float, *, minimum_headroom: float) -> dict[str, object]:
    chance = 1.0 / TASK_CLASSES[PAIR_TASKS[0]]
    headroom = before - chance
    valid = headroom >= minimum_headroom
    raw = before - after
    return {
        "old_task": PAIR_TASKS[0],
        "learned_task": PAIR_TASKS[1],
        "before": before,
        "after": after,
        "chance": chance,
        "headroom": headroom,
        "raw_forgetting": raw,
        "relative_forgetting": raw / headroom if valid else None,
        "relative_valid": valid,
    }


def _validate_config(config: Mapping[str, object]) -> None:
    if config.get("status") != "locked_pilot" or tuple(config.get("pair", ())) != PAIR_TASKS:
        raise DatasetProtocolError("method pilot config must lock BCI-to-Sleep")
    if int(config["training"]["optimizer_steps_per_task"]) != 2500:
        raise DatasetProtocolError("method pilot must preserve the 2500-step budget")
    if config["training"]["evaluation_split"] != "validation":
        raise DatasetProtocolError("method selection must use validation, not test")


def run_ewc_pair_pilot(
    *,
    config_path: str | Path,
    strength: float,
    cache_root: str | Path,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    output_dir: str | Path,
    device: str | torch.device,
) -> dict[str, object]:
    config_path = Path(config_path)
    config = load_yaml(config_path)
    _validate_config(config)
    declared = tuple(float(value) for value in config["ewc"]["strength_candidates"])
    if float(strength) not in declared:
        raise DatasetProtocolError(f"EWC strength {strength} is not predeclared")
    output_dir = Path(output_dir)
    result_path = output_dir / "result.json"
    if result_path.exists():
        raise DatasetProtocolError(f"refusing to overwrite EWC pilot {result_path}")

    seed = int(config["seed"])
    set_determinism(seed)
    device = torch.device(device)
    torch.cuda.set_device(device)
    train_sets, validation_loaders = _load_pair_data(Path(cache_root))
    backbone = load_pretrained_backbone(
        checkpoint_path, expected_sha256=checkpoint_sha256, map_location="cpu"
    )
    model = MultiHeadCBraMod(backbone).to(device)

    source_checkpoint = Path(str(config["ewc"]["stage_one_checkpoint"]))
    source_fisher = Path(str(config["ewc"]["fisher_path"]))
    if sha256_file(source_checkpoint) != config["ewc"]["stage_one_checkpoint_sha256"]:
        raise DatasetProtocolError("EWC stage-one checkpoint digest mismatch")
    if sha256_file(source_fisher) != config["ewc"]["fisher_sha256"]:
        raise DatasetProtocolError("EWC Fisher digest mismatch")
    state = torch.load(source_checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    fisher_cpu = load_fisher(source_fisher)
    parameter_map = dict(model.named_parameters())
    if any(name not in parameter_map for name in fisher_cpu):
        raise DatasetProtocolError("EWC Fisher does not align with the continual model")
    anchors_cpu = OrderedDict(
        (name, parameter_map[name].detach().cpu().clone()) for name in fisher_cpu
    )
    fisher = OrderedDict((name, value.to(device)) for name, value in fisher_cpu.items())
    anchors = OrderedDict((name, value.to(device)) for name, value in anchors_cpu.items())
    state_bytes = sum(
        fisher_cpu[name].numel()
        * (fisher_cpu[name].element_size() + anchors_cpu[name].element_size())
        for name in fisher_cpu
    )

    before_metrics, _ = evaluate_task(
        model, PAIR_TASKS[0], validation_loaders[PAIR_TASKS[0]], device=device
    )
    task = PAIR_TASKS[1]
    trainable, optimizer, scheduler, loader = _optimizer_and_loader(
        model, train_sets[task], task, config, seed=seed
    )
    training = config["training"]
    steps = int(training["optimizer_steps_per_task"])
    recent_total: deque[float] = deque(maxlen=int(training["progress_interval_steps"]))
    recent_current: deque[float] = deque(maxlen=int(training["progress_interval_steps"]))
    recent_penalty: deque[float] = deque(maxlen=int(training["progress_interval_steps"]))
    curve = []
    for step, (signals, labels, _subjects) in enumerate(loader, start=1):
        signals = signals.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        current_loss = F.cross_entropy(
            model(task, signals),
            labels,
            label_smoothing=float(training["label_smoothing"]),
        )
        quadratic = ewc_quadratic_penalty(
            dict(model.named_parameters()), fisher, anchors
        )
        penalty = 0.5 * float(strength) * quadratic
        loss = current_loss + penalty
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, float(training["gradient_clip_norm"]))
        optimizer.step()
        scheduler.step()
        recent_total.append(float(loss.detach()))
        recent_current.append(float(current_loss.detach()))
        recent_penalty.append(float(penalty.detach()))
        if step % int(training["progress_interval_steps"]) == 0:
            point = {
                "step": step,
                "total_loss": float(np.mean(recent_total)),
                "current_loss": float(np.mean(recent_current)),
                "ewc_penalty": float(np.mean(recent_penalty)),
            }
            curve.append(point)
            print(
                f"ewc lambda={strength:g} step={step}/{steps} "
                f"loss={point['total_loss']:.5f} penalty={point['ewc_penalty']:.5f}",
                flush=True,
            )

    evaluations = {}
    for evaluated_task in PAIR_TASKS:
        metrics, _arrays = evaluate_task(
            model, evaluated_task, validation_loaders[evaluated_task], device=device
        )
        evaluations[evaluated_task] = metrics
    forgetting = _pair_forgetting(
        float(before_metrics["mean_subject_balanced_accuracy"]),
        float(evaluations[PAIR_TASKS[0]]["mean_subject_balanced_accuracy"]),
        minimum_headroom=float(config["selection"]["minimum_valid_headroom_above_chance"]),
    )
    checkpoint_output = output_dir / "final.pt"
    _atomic_torch(
        checkpoint_output,
        {
            "model_state_dict": model.state_dict(),
            "method": "ewc",
            "strength": float(strength),
            "config_sha256": sha256_file(config_path),
        },
    )
    result = {
        "schema_version": 1,
        "pilot": config["id"],
        "method": "ewc",
        "candidate": float(strength),
        "seed": seed,
        "pair": list(PAIR_TASKS),
        "config_sha256": sha256_file(config_path),
        "cache_indices": {
            task: {
                split: sha256_file(
                    Path(cache_root) / task / split / "index.json"
                )
                for split in ("train", "validation")
            }
            for task in PAIR_TASKS
        },
        "stage_one": {
            "source_checkpoint": str(source_checkpoint),
            "source_checkpoint_sha256": sha256_file(source_checkpoint),
            "old_task_validation": before_metrics,
        },
        "final_evaluations": evaluations,
        "pairwise_forgetting": forgetting,
        "curve": curve,
        "memory": {
            "protected_parameters": sum(value.numel() for value in fisher_cpu.values()),
            "fisher_and_anchor_bytes": state_bytes,
            "states_retained": 1,
        },
        "checkpoint": {
            "file": checkpoint_output.name,
            "sha256": sha256_file(checkpoint_output),
        },
        "device": str(device),
        "torch": torch.__version__,
    }
    _atomic_json(result_path, result)
    return result


def _train_derpp_stage(
    model: MultiHeadCBraMod,
    buffer: ByteCappedReservoir,
    dataset: CachedEEGDataset,
    task: str,
    config: Mapping[str, object],
    *,
    seed: int,
    device: torch.device,
) -> list[dict[str, float | int]]:
    trainable, optimizer, scheduler, loader = _optimizer_and_loader(
        model, dataset, task, config, seed=seed
    )
    training = config["training"]
    method = config["derpp"]
    interval = int(training["progress_interval_steps"])
    recent: dict[str, deque[float]] = {
        name: deque(maxlen=interval)
        for name in ("total", "current", "logits", "labels")
    }
    curve = []
    for step, (signals_cpu, labels_cpu, _subjects) in enumerate(loader, start=1):
        signals = signals_cpu.to(device, non_blocking=True)
        labels = labels_cpu.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        current_logits = model(task, signals)
        current_loss = F.cross_entropy(
            current_logits,
            labels,
            label_smoothing=float(training["label_smoothing"]),
        )
        logits_loss = derpp_replay_loss(
            model,
            buffer.sample(int(method["replay_batch_size"])),
            device=device,
            target="logits",
            label_smoothing=float(training["label_smoothing"]),
        )
        label_loss = derpp_replay_loss(
            model,
            buffer.sample(int(method["replay_batch_size"])),
            device=device,
            target="labels",
            label_smoothing=float(training["label_smoothing"]),
        )
        loss = (
            current_loss
            + float(method["alpha"]) * logits_loss
            + float(method["beta"]) * label_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, float(training["gradient_clip_norm"]))
        optimizer.step()
        scheduler.step()
        buffer.add_batch(task, signals_cpu, labels_cpu, current_logits.detach().cpu())
        recent["total"].append(float(loss.detach()))
        recent["current"].append(float(current_loss.detach()))
        recent["logits"].append(float(logits_loss.detach()))
        recent["labels"].append(float(label_loss.detach()))
        if step % interval == 0:
            point: dict[str, float | int] = {
                "step": step,
                "total_loss": float(np.mean(recent["total"])),
                "current_loss": float(np.mean(recent["current"])),
                "logit_replay_loss": float(np.mean(recent["logits"])),
                "label_replay_loss": float(np.mean(recent["labels"])),
            }
            curve.append(point)
            print(
                f"derpp cap={buffer.byte_cap} task={task} step={step}/"
                f"{training['optimizer_steps_per_task']} loss={point['total_loss']:.5f} "
                f"stored={buffer.size}",
                flush=True,
            )
    return curve


def run_derpp_pair_pilot(
    *,
    config_path: str | Path,
    byte_cap_mib: int,
    cache_root: str | Path,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    output_dir: str | Path,
    device: str | torch.device,
) -> dict[str, object]:
    config_path = Path(config_path)
    config = load_yaml(config_path)
    _validate_config(config)
    declared = tuple(int(value) for value in config["derpp"]["byte_cap_candidates_mib"])
    if int(byte_cap_mib) not in declared:
        raise DatasetProtocolError(f"DER++ byte cap {byte_cap_mib} MiB is not predeclared")
    output_dir = Path(output_dir)
    result_path = output_dir / "result.json"
    if result_path.exists():
        raise DatasetProtocolError(f"refusing to overwrite DER++ pilot {result_path}")
    seed = int(config["seed"])
    set_determinism(seed)
    device = torch.device(device)
    torch.cuda.set_device(device)
    train_sets, validation_loaders = _load_pair_data(Path(cache_root))
    backbone = load_pretrained_backbone(
        checkpoint_path, expected_sha256=checkpoint_sha256, map_location="cpu"
    )
    model = MultiHeadCBraMod(backbone).to(device)
    byte_cap = int(byte_cap_mib) * 1024 * 1024
    buffer = ByteCappedReservoir(byte_cap, seed=seed + 9173)
    config_sha = sha256_file(config_path)

    stage_one_path = output_dir / "stage-01.pt"
    if stage_one_path.exists():
        stage_one = torch.load(stage_one_path, map_location="cpu", weights_only=True)
        if stage_one["config_sha256"] != config_sha or int(stage_one["byte_cap_mib"]) != int(byte_cap_mib):
            raise DatasetProtocolError("DER++ stage-one resume metadata mismatch")
        model.load_state_dict(stage_one["model_state_dict"])
        buffer.load_state_dict(stage_one["buffer_state_dict"])
        stage_one_metrics = stage_one["validation"]
        stage_one_curve = stage_one["curve"]
        print(f"resumed DER++ cap={byte_cap_mib} MiB after BCI stage", flush=True)
    else:
        stage_one_curve = _train_derpp_stage(
            model,
            buffer,
            train_sets[PAIR_TASKS[0]],
            PAIR_TASKS[0],
            config,
            seed=seed,
            device=device,
        )
        stage_one_metrics, _ = evaluate_task(
            model, PAIR_TASKS[0], validation_loaders[PAIR_TASKS[0]], device=device
        )
        _atomic_torch(
            stage_one_path,
            {
                "model_state_dict": model.state_dict(),
                "buffer_state_dict": buffer.state_dict(),
                "validation": stage_one_metrics,
                "curve": stage_one_curve,
                "config_sha256": config_sha,
                "byte_cap_mib": int(byte_cap_mib),
            },
        )

    stage_two_curve = _train_derpp_stage(
        model,
        buffer,
        train_sets[PAIR_TASKS[1]],
        PAIR_TASKS[1],
        config,
        seed=seed,
        device=device,
    )
    evaluations = {}
    for task in PAIR_TASKS:
        metrics, _arrays = evaluate_task(
            model, task, validation_loaders[task], device=device
        )
        evaluations[task] = metrics
    forgetting = _pair_forgetting(
        float(stage_one_metrics["mean_subject_balanced_accuracy"]),
        float(evaluations[PAIR_TASKS[0]]["mean_subject_balanced_accuracy"]),
        minimum_headroom=float(config["selection"]["minimum_valid_headroom_above_chance"]),
    )
    checkpoint_output = output_dir / "final.pt"
    _atomic_torch(
        checkpoint_output,
        {
            "model_state_dict": model.state_dict(),
            "buffer_state_dict": buffer.state_dict(),
            "method": "derpp",
            "byte_cap_mib": int(byte_cap_mib),
            "config_sha256": config_sha,
        },
    )
    result = {
        "schema_version": 1,
        "pilot": config["id"],
        "method": "derpp",
        "candidate": int(byte_cap_mib),
        "seed": seed,
        "pair": list(PAIR_TASKS),
        "config_sha256": config_sha,
        "stage_one": {
            "validation": stage_one_metrics,
            "curve": stage_one_curve,
            "checkpoint_sha256": sha256_file(stage_one_path),
        },
        "final_evaluations": evaluations,
        "pairwise_forgetting": forgetting,
        "stage_two_curve": stage_two_curve,
        "memory": buffer.inventory(),
        "checkpoint": {
            "file": checkpoint_output.name,
            "sha256": sha256_file(checkpoint_output),
        },
        "device": str(device),
        "torch": torch.__version__,
    }
    _atomic_json(result_path, result)
    return result


def select_method_candidate(
    results: Sequence[Mapping[str, object]], *, maximum_new_task_drop: float
) -> dict[str, object]:
    if not results:
        raise DatasetProtocolError("cannot select a method from zero pilot results")
    methods = {str(result["method"]) for result in results}
    if len(methods) != 1:
        raise DatasetProtocolError("method selection cannot mix EWC and DER++")
    baseline = [result for result in results if float(result["candidate"]) == 0.0]
    if len(baseline) != 1:
        raise DatasetProtocolError("method selection needs exactly one zero baseline")
    baseline_new = float(
        baseline[0]["final_evaluations"][PAIR_TASKS[1]][
            "mean_subject_balanced_accuracy"
        ]
    )
    floor = baseline_new - float(maximum_new_task_drop)
    rows = []
    for result in results:
        new_score = float(
            result["final_evaluations"][PAIR_TASKS[1]][
                "mean_subject_balanced_accuracy"
            ]
        )
        forgetting = result["pairwise_forgetting"]["relative_forgetting"]
        eligible = new_score >= floor and forgetting is not None
        rows.append(
            {
                "candidate": result["candidate"],
                "new_task_validation": new_score,
                "old_task_relative_forgetting": forgetting,
                "eligible": eligible,
            }
        )
    eligible_rows = [row for row in rows if row["eligible"]]
    if not eligible_rows:
        raise DatasetProtocolError("no method candidate meets the plasticity constraint")
    selected = min(
        eligible_rows,
        key=lambda row: (
            float(row["old_task_relative_forgetting"]),
            float(row["candidate"]),
        ),
    )
    return {
        "method": methods.pop(),
        "baseline_new_task_validation": baseline_new,
        "minimum_allowed_new_task_validation": floor,
        "maximum_absolute_new_task_drop": float(maximum_new_task_drop),
        "selected_candidate": selected["candidate"],
        "candidates": sorted(rows, key=lambda row: float(row["candidate"])),
    }


class _TaskFisherView(nn.Module):
    """Expose a multi-head task with Fisher-compatible backbone parameter names."""

    def __init__(self, model: MultiHeadCBraMod, task: str):
        super().__init__()
        self.backbone = model.backbone
        self.classifier = model.heads[task]

    def forward(self, signals: Tensor) -> Tensor:
        return self.classifier(self.backbone.forward_features(signals))


def _validate_main_method_config(
    config: Mapping[str, object], *, method: str, order_name: str, seed: int
) -> tuple[str, ...]:
    if config.get("status") != "locked_main" or config.get("method") != method:
        raise DatasetProtocolError(f"expected locked main {method} config")
    if tuple(config.get("canonical_tasks", ())) != CANONICAL_TASKS:
        raise DatasetProtocolError("main method config changed the canonical tasks")
    if order_name not in config["orders"]:
        raise DatasetProtocolError(f"unknown {method} order {order_name!r}")
    order = tuple(config["orders"][order_name])
    if set(order) != set(CANONICAL_TASKS) or len(order) != len(CANONICAL_TASKS):
        raise DatasetProtocolError("main method order must contain every task once")
    if int(seed) not in tuple(int(value) for value in config["seeds"]):
        raise DatasetProtocolError(f"seed {seed} is not declared by {method} config")
    if int(config["training"]["optimizer_steps_per_task"]) != 2500:
        raise DatasetProtocolError("main method changed the locked task step budget")
    if config["training"]["evaluation_split"] != "test":
        raise DatasetProtocolError("main method matrix must evaluate frozen test splits")
    derivation = config["derivation"]
    summary_path = Path(str(derivation["pilot_summary"]))
    if sha256_file(summary_path) != derivation["pilot_summary_sha256"]:
        raise DatasetProtocolError("method-selection summary digest mismatch")
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    selected = summary["selection"][method]["selected_candidate"]
    declared = (
        config["ewc"]["strength"]
        if method == "ewc"
        else config["derpp"]["selected_byte_cap_mib"]
    )
    if float(selected) != float(declared):
        raise DatasetProtocolError("main method setting differs from pilot selection")
    return order


def _load_main_data(
    cache_root: Path, *, batch_size: int
) -> tuple[
    dict[str, CachedEEGDataset],
    dict[str, CachedEEGDataset],
    dict[str, DataLoader],
]:
    train_sets = {
        task: CachedEEGDataset(cache_root / task / "train" / "index.json")
        for task in CANONICAL_TASKS
    }
    test_sets = {
        task: CachedEEGDataset(cache_root / task / "test" / "index.json")
        for task in CANONICAL_TASKS
    }
    test_loaders = {
        task: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        for task, dataset in test_sets.items()
    }
    return train_sets, test_sets, test_loaders


def _resume_main_stages(
    *,
    output_dir: Path,
    order: Sequence[str],
    config_sha: str,
) -> tuple[list[dict[str, object]], Path | None]:
    completed = []
    latest = None
    for index, task in enumerate(order, start=1):
        stage_dir = output_dir / f"stage-{index:02d}-{task}"
        stage_path = stage_dir / "stage.json"
        if not stage_path.exists():
            if stage_dir.exists() and any(stage_dir.iterdir()):
                raise DatasetProtocolError(
                    f"incomplete method stage requires recovery: {stage_dir}"
                )
            break
        with stage_path.open(encoding="utf-8") as handle:
            document = json.load(handle)
        checkpoint = stage_dir / str(document["checkpoint"]["file"])
        if (
            document["config_sha256"] != config_sha
            or sha256_file(checkpoint) != document["checkpoint"]["sha256"]
        ):
            raise DatasetProtocolError(f"method stage resume digest mismatch: {stage_dir}")
        completed.append(document)
        latest = checkpoint
    return completed, latest


def _execution_rng_state(device: torch.device) -> dict[str, Tensor]:
    state = {"torch_cpu": torch.get_rng_state()}
    if device.type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state(device)
    return state


def _restore_execution_rng_state(
    state: Mapping[str, Tensor], *, device: torch.device
) -> None:
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda":
        if "torch_cuda" not in state:
            raise DatasetProtocolError("method checkpoint lacks CUDA RNG state")
        torch.cuda.set_rng_state(state["torch_cuda"], device)


def _evaluate_stage(
    model: MultiHeadCBraMod,
    *,
    seen_tasks: Sequence[str],
    loaders: Mapping[str, DataLoader],
    stage_dir: Path,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, dict[str, str]]]:
    stage_dir.mkdir(parents=True, exist_ok=True)
    evaluations = {}
    predictions = {}
    for task in seen_tasks:
        metrics, arrays = evaluate_task(model, task, loaders[task], device=device)
        prediction_path = stage_dir / f"predictions-{task}.npz"
        _atomic_npz(prediction_path, arrays)
        evaluations[task] = metrics
        predictions[task] = {
            "file": prediction_path.name,
            "sha256": sha256_file(prediction_path),
        }
        print(
            f"eval={task} subject_BA={metrics['mean_subject_balanced_accuracy']:.5f}",
            flush=True,
        )
    return evaluations, predictions


def _ewc_state_bytes(states: Sequence[Mapping[str, object]]) -> int:
    total = 0
    for state in states:
        fisher = state["fisher"]
        anchors = state["anchors"]
        total += sum(
            fisher[name].numel()
            * (fisher[name].element_size() + anchors[name].element_size())
            for name in fisher
        )
    return total


def _train_ewc_main_stage(
    model: MultiHeadCBraMod,
    dataset: CachedEEGDataset,
    task: str,
    states: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
    *,
    seed: int,
    device: torch.device,
) -> list[dict[str, float | int]]:
    trainable, optimizer, scheduler, loader = _optimizer_and_loader(
        model, dataset, task, config, seed=seed
    )
    training = config["training"]
    strength = float(config["ewc"]["strength"])
    interval = int(training["progress_interval_steps"])
    device_states = [
        {
            "fisher": OrderedDict(
                (name, value.to(device)) for name, value in state["fisher"].items()
            ),
            "anchors": OrderedDict(
                (name, value.to(device)) for name, value in state["anchors"].items()
            ),
        }
        for state in states
    ]
    recent_total: deque[float] = deque(maxlen=interval)
    recent_current: deque[float] = deque(maxlen=interval)
    recent_penalty: deque[float] = deque(maxlen=interval)
    curve = []
    for step, (signals, labels, _subjects) in enumerate(loader, start=1):
        signals = signals.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        current_loss = F.cross_entropy(
            model(task, signals),
            labels,
            label_smoothing=float(training["label_smoothing"]),
        )
        named_parameters = dict(model.named_parameters())
        quadratics = [
            ewc_quadratic_penalty(
                named_parameters, state["fisher"], state["anchors"]
            )
            for state in device_states
        ]
        penalty = (
            0.5 * strength * torch.stack(quadratics).sum()
            if quadratics
            else torch.zeros((), device=device)
        )
        loss = current_loss + penalty
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, float(training["gradient_clip_norm"]))
        optimizer.step()
        scheduler.step()
        recent_total.append(float(loss.detach()))
        recent_current.append(float(current_loss.detach()))
        recent_penalty.append(float(penalty.detach()))
        if step % interval == 0:
            point: dict[str, float | int] = {
                "step": step,
                "total_loss": float(np.mean(recent_total)),
                "current_loss": float(np.mean(recent_current)),
                "ewc_penalty": float(np.mean(recent_penalty)),
                "protected_prior_tasks": len(states),
            }
            curve.append(point)
            print(
                f"ewc task={task} step={step}/{training['optimizer_steps_per_task']} "
                f"loss={point['total_loss']:.5f} penalty={point['ewc_penalty']:.5f}",
                flush=True,
            )
    return curve


def _extract_ewc_state(
    model: MultiHeadCBraMod,
    dataset: CachedEEGDataset,
    task: str,
    config: Mapping[str, object],
    *,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    settings = config["ewc"]
    canonical_index = CANONICAL_TASKS.index(task) + 1
    sampling_seed = int(seed) + 100003 * canonical_index
    indices = deterministic_sample_indices(
        len(dataset),
        sample_count=int(settings["sample_count_per_completed_nonfinal_task"]),
        seed=sampling_seed,
    )
    view = _TaskFisherView(model, task)
    last_reported = 0

    def report(completed: int, total: int) -> None:
        nonlocal last_reported
        interval = int(settings["progress_interval_examples"])
        if completed == total or completed - last_reported >= interval:
            print(f"ewc task={task} Fisher {completed}/{total}", flush=True)
            last_reported = completed

    fisher = diagonal_empirical_fisher(
        view,
        dataset,
        indices,
        final_blocks=int(config["model"]["plastic_final_encoder_blocks"]),
        microbatch_size=int(settings["gradient_microbatch_size"]),
        device=device,
        progress=report,
    )
    named = dict(model.named_parameters())
    anchors = OrderedDict(
        (name, named[name].detach().cpu().clone()) for name in fisher
    )
    return {
        "task": task,
        "fisher": fisher,
        "anchors": anchors,
        "samples": len(indices),
        "sampling_seed": sampling_seed,
        "sample_indices_sha256": indices_sha256(indices),
    }


def _main_result(
    *,
    config: Mapping[str, object],
    config_sha: str,
    order_name: str,
    order: Sequence[str],
    seed: int,
    checkpoint_sha256: str,
    cache_root: Path,
    train_sets: Mapping[str, CachedEEGDataset],
    test_sets: Mapping[str, CachedEEGDataset],
    completed_stages: Sequence[Mapping[str, object]],
    output_dir: Path,
    device: torch.device,
    memory: Mapping[str, object],
) -> dict[str, object]:
    result = {
        "schema_version": 1,
        "run": config["id"],
        "method": config["method"],
        "order_name": order_name,
        "order": list(order),
        "seed": int(seed),
        "config_sha256": config_sha,
        "pretrained_checkpoint_sha256": checkpoint_sha256,
        "cache_indices": {
            split: {
                task: sha256_file(dataset.index_path)
                for task, dataset in datasets.items()
            }
            for split, datasets in (("train", train_sets), ("test", test_sets))
        },
        "stages": [
            {
                "stage": stage["stage"],
                "learned_task": stage["learned_task"],
                "stage_result_sha256": sha256_file(
                    output_dir
                    / f"stage-{stage['stage']:02d}-{stage['learned_task']}"
                    / "stage.json"
                ),
                "evaluations": stage["evaluations"],
                "memory": stage["memory"],
            }
            for stage in completed_stages
        ],
        "pairwise_forgetting": pairwise_forgetting(
            completed_stages,
            order,
            minimum_headroom=float(
                config["forgetting"]["minimum_valid_headroom_above_chance"]
            ),
        ),
        "memory": dict(memory),
        "device": str(device),
        "torch": torch.__version__,
    }
    _atomic_continual_json(output_dir / "result.json", result)
    return result


def run_ewc_continual_matrix_run(
    *,
    config_path: str | Path,
    order_name: str,
    seed: int,
    cache_root: str | Path,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    output_dir: str | Path,
    device: str | torch.device,
) -> dict[str, object]:
    config_path = Path(config_path)
    config = load_yaml(config_path)
    order = _validate_main_method_config(
        config, method="ewc", order_name=order_name, seed=seed
    )
    output_dir = Path(output_dir)
    if (output_dir / "result.json").exists():
        raise DatasetProtocolError(f"refusing to overwrite EWC run {output_dir}")
    set_determinism(int(seed))
    device = torch.device(device)
    torch.cuda.set_device(device)
    cache_root = Path(cache_root)
    train_sets, test_sets, test_loaders = _load_main_data(
        cache_root, batch_size=int(config["training"]["batch_size"])
    )
    backbone = load_pretrained_backbone(
        checkpoint_path, expected_sha256=checkpoint_sha256, map_location="cpu"
    )
    model = MultiHeadCBraMod(backbone).to(device)
    config_sha = sha256_file(config_path)
    completed_stages, latest_checkpoint = _resume_main_stages(
        output_dir=output_dir, order=order, config_sha=config_sha
    )
    states: list[dict[str, object]] = []
    if latest_checkpoint is not None:
        checkpoint = torch.load(
            latest_checkpoint, map_location="cpu", weights_only=True
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        states = list(checkpoint["ewc_states"])
        _restore_execution_rng_state(checkpoint["execution_rng_state"], device=device)
        print(
            f"resumed ewc {order_name} seed={seed} after stage={len(completed_stages)}",
            flush=True,
        )

    for index in range(len(completed_stages), len(order)):
        task = order[index]
        stage_number = index + 1
        curve = _train_ewc_main_stage(
            model,
            train_sets[task],
            task,
            states,
            config,
            seed=int(seed),
            device=device,
        )
        stage_dir = output_dir / f"stage-{stage_number:02d}-{task}"
        evaluations, predictions = _evaluate_stage(
            model,
            seen_tasks=order[:stage_number],
            loaders=test_loaders,
            stage_dir=stage_dir,
            device=device,
        )
        fisher_metadata = None
        if stage_number < len(order):
            new_state = _extract_ewc_state(
                model,
                train_sets[task],
                task,
                config,
                seed=int(seed),
                device=device,
            )
            states.append(new_state)
            fisher_metadata = {
                key: value
                for key, value in new_state.items()
                if key not in {"fisher", "anchors"}
            }
        memory = {
            "retained_prior_task_states": len(states),
            "fisher_and_anchor_bytes": _ewc_state_bytes(states),
            "protected_parameters_per_state": (
                sum(value.numel() for value in states[0]["fisher"].values())
                if states
                else 0
            ),
        }
        checkpoint_output = stage_dir / "checkpoint.pt"
        _atomic_torch_save(
            checkpoint_output,
            {
                "model_state_dict": model.state_dict(),
                "ewc_states": states,
                "stage": stage_number,
                "task": task,
                "order": order,
                "seed": int(seed),
                "config_sha256": config_sha,
                "execution_rng_state": _execution_rng_state(device),
            },
        )
        stage_document = {
            "schema_version": 1,
            "run": config["id"],
            "method": "ewc",
            "order_name": order_name,
            "order": list(order),
            "seed": int(seed),
            "stage": stage_number,
            "learned_task": task,
            "optimizer_steps": int(config["training"]["optimizer_steps_per_task"]),
            "config_sha256": config_sha,
            "curve": curve,
            "evaluations": evaluations,
            "predictions": predictions,
            "fisher_estimation": fisher_metadata,
            "memory": memory,
            "checkpoint": {
                "file": checkpoint_output.name,
                "sha256": sha256_file(checkpoint_output),
            },
        }
        _atomic_continual_json(stage_dir / "stage.json", stage_document)
        completed_stages.append(stage_document)

    memory = {
        "peak_fisher_and_anchor_bytes": max(
            int(stage["memory"]["fisher_and_anchor_bytes"])
            for stage in completed_stages
        ),
        "final_retained_prior_task_states": int(
            completed_stages[-1]["memory"]["retained_prior_task_states"]
        ),
        "formulation": "offline_ewc",
    }
    return _main_result(
        config=config,
        config_sha=config_sha,
        order_name=order_name,
        order=order,
        seed=int(seed),
        checkpoint_sha256=checkpoint_sha256,
        cache_root=cache_root,
        train_sets=train_sets,
        test_sets=test_sets,
        completed_stages=completed_stages,
        output_dir=output_dir,
        device=device,
        memory=memory,
    )


def run_derpp_continual_matrix_run(
    *,
    config_path: str | Path,
    order_name: str,
    seed: int,
    cache_root: str | Path,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    output_dir: str | Path,
    device: str | torch.device,
) -> dict[str, object]:
    config_path = Path(config_path)
    config = load_yaml(config_path)
    order = _validate_main_method_config(
        config, method="derpp", order_name=order_name, seed=seed
    )
    output_dir = Path(output_dir)
    if (output_dir / "result.json").exists():
        raise DatasetProtocolError(f"refusing to overwrite DER++ run {output_dir}")
    set_determinism(int(seed))
    device = torch.device(device)
    torch.cuda.set_device(device)
    cache_root = Path(cache_root)
    train_sets, test_sets, test_loaders = _load_main_data(
        cache_root, batch_size=int(config["training"]["batch_size"])
    )
    backbone = load_pretrained_backbone(
        checkpoint_path, expected_sha256=checkpoint_sha256, map_location="cpu"
    )
    model = MultiHeadCBraMod(backbone).to(device)
    method = config["derpp"]
    buffer = ByteCappedReservoir(
        int(method["persistent_byte_cap"]), seed=int(seed) + 9173
    )
    config_sha = sha256_file(config_path)
    completed_stages, latest_checkpoint = _resume_main_stages(
        output_dir=output_dir, order=order, config_sha=config_sha
    )
    if latest_checkpoint is not None:
        checkpoint = torch.load(
            latest_checkpoint, map_location="cpu", weights_only=True
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        buffer.load_state_dict(checkpoint["buffer_state_dict"])
        _restore_execution_rng_state(checkpoint["execution_rng_state"], device=device)
        print(
            f"resumed derpp {order_name} seed={seed} after stage={len(completed_stages)}",
            flush=True,
        )

    for index in range(len(completed_stages), len(order)):
        task = order[index]
        stage_number = index + 1
        curve = _train_derpp_stage(
            model,
            buffer,
            train_sets[task],
            task,
            config,
            seed=int(seed),
            device=device,
        )
        stage_dir = output_dir / f"stage-{stage_number:02d}-{task}"
        evaluations, predictions = _evaluate_stage(
            model,
            seen_tasks=order[:stage_number],
            loaders=test_loaders,
            stage_dir=stage_dir,
            device=device,
        )
        memory = buffer.inventory()
        checkpoint_output = stage_dir / "checkpoint.pt"
        _atomic_torch_save(
            checkpoint_output,
            {
                "model_state_dict": model.state_dict(),
                "buffer_state_dict": buffer.state_dict(),
                "stage": stage_number,
                "task": task,
                "order": order,
                "seed": int(seed),
                "config_sha256": config_sha,
                "execution_rng_state": _execution_rng_state(device),
            },
        )
        stage_document = {
            "schema_version": 1,
            "run": config["id"],
            "method": "derpp",
            "order_name": order_name,
            "order": list(order),
            "seed": int(seed),
            "stage": stage_number,
            "learned_task": task,
            "optimizer_steps": int(config["training"]["optimizer_steps_per_task"]),
            "config_sha256": config_sha,
            "curve": curve,
            "evaluations": evaluations,
            "predictions": predictions,
            "memory": memory,
            "checkpoint": {
                "file": checkpoint_output.name,
                "sha256": sha256_file(checkpoint_output),
            },
        }
        _atomic_continual_json(stage_dir / "stage.json", stage_document)
        completed_stages.append(stage_document)

    inventories = [stage["memory"] for stage in completed_stages]
    memory = {
        "peak_allocated_bytes": max(int(item["allocated_bytes"]) for item in inventories),
        "final": inventories[-1],
        "persistent_byte_cap": int(method["persistent_byte_cap"]),
    }
    return _main_result(
        config=config,
        config_sha=config_sha,
        order_name=order_name,
        order=order,
        seed=int(seed),
        checkpoint_sha256=checkpoint_sha256,
        cache_root=cache_root,
        train_sets=train_sets,
        test_sets=test_sets,
        completed_stages=completed_stages,
        output_dir=output_dir,
        device=device,
        memory=memory,
    )
