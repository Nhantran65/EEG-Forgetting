from __future__ import annotations

import json
import os
import statistics
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from eeg_forgetting.data.cache import CachedEEGDataset
from eeg_forgetting.data.contracts import DatasetProtocolError
from eeg_forgetting.data.manifests import sha256_file
from eeg_forgetting.models.cbramod import CBraMod, CBraModTaskModel, load_pretrained_backbone

from .metrics import multiclass_metrics, subject_balanced_accuracy
from .pilot import (
    FixedStepBatchSampler,
    PilotSettings,
    TASK_CLASSES,
    TASK_SHAPES,
    set_determinism,
)


CANONICAL_TASKS = ("bciciv2a", "physionet_mi", "sleep_edf_sc")


def aggregate_replicates(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise DatasetProtocolError("cannot aggregate zero continual-learning replicates")
    values = [float(value) for value in values]
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
        "positive_fraction": sum(value > 0 for value in values) / len(values),
        "negative_fraction": sum(value < 0 for value in values) / len(values),
    }


class MultiHeadCBraMod(nn.Module):
    def __init__(self, backbone: CBraMod, *, tasks: Sequence[str] = CANONICAL_TASKS):
        super().__init__()
        tasks = tuple(tasks)
        if not tasks or len(set(tasks)) != len(tasks) or not set(tasks).issubset(TASK_CLASSES):
            raise DatasetProtocolError("task heads need distinct declared tasks")
        self.tasks = tasks
        self.backbone = backbone
        self.heads = nn.ModuleDict()
        for task in tasks:
            channels, patches = TASK_SHAPES[task]
            task_model = CBraModTaskModel(
                backbone,
                TASK_CLASSES[task],
                head="flatten_mlp",
                channels=channels,
                patches=patches,
            )
            self.heads[task] = task_model.classifier

    def forward(self, task: str, signals: Tensor) -> Tensor:
        if task not in self.heads:
            raise DatasetProtocolError(f"unknown continual-learning task {task!r}")
        return self.heads[task](self.backbone.forward_features(signals))

    def prepare_stage(self, task: str, *, depth: int) -> list[nn.Parameter]:
        self.eval()
        self.backbone.set_trainable_depth(depth)
        for head in self.heads.values():
            for parameter in head.parameters():
                parameter.requires_grad = False
        for parameter in self.heads[task].parameters():
            parameter.requires_grad = True
        self.heads[task].train()
        for layer in self.backbone.encoder.layers[-depth:]:
            layer.train()
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def prepare_joint(self, *, depth: int) -> list[nn.Parameter]:
        self.eval()
        self.backbone.set_trainable_depth(depth)
        for head in self.heads.values():
            for parameter in head.parameters():
                parameter.requires_grad = True
            head.train()
        for layer in self.backbone.encoder.layers[-depth:]:
            layer.train()
        return [parameter for parameter in self.parameters() if parameter.requires_grad]


@torch.no_grad()
def evaluate_task(
    model: MultiHeadCBraMod,
    task: str,
    loader: DataLoader,
    *,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    model.eval()
    truth: list[int] = []
    prediction: list[int] = []
    subjects: list[str] = []
    for signals, labels, batch_subjects in loader:
        signals = signals.to(device, non_blocking=True)
        logits = model(task, signals)
        truth.extend(int(value) for value in labels)
        prediction.extend(int(value) for value in logits.argmax(dim=1).cpu())
        subjects.extend(str(value) for value in batch_subjects)
    subject_mean, by_subject = subject_balanced_accuracy(truth, prediction, subjects)
    metrics: dict[str, object] = multiclass_metrics(
        truth, prediction, classes=TASK_CLASSES[task]
    )
    metrics["mean_subject_balanced_accuracy"] = subject_mean
    metrics["by_subject"] = by_subject
    metrics["samples"] = len(truth)
    arrays = {
        "truth": np.asarray(truth, dtype=np.int64),
        "prediction": np.asarray(prediction, dtype=np.int64),
        "subject": np.asarray(subjects, dtype=np.str_),
    }
    return metrics, arrays


def pairwise_forgetting(
    stages: Sequence[Mapping[str, object]],
    order: Sequence[str],
    *,
    minimum_headroom: float,
) -> list[dict[str, object]]:
    if len(stages) != len(order) or minimum_headroom < 0:
        raise DatasetProtocolError("invalid stages for pairwise forgetting")
    rows = []
    for after_index in range(1, len(stages)):
        learned_task = order[after_index]
        before_evaluations = stages[after_index - 1]["evaluations"]
        after_evaluations = stages[after_index]["evaluations"]
        for old_task in order[:after_index]:
            before = float(
                before_evaluations[old_task]["mean_subject_balanced_accuracy"]
            )
            after = float(
                after_evaluations[old_task]["mean_subject_balanced_accuracy"]
            )
            chance = 1.0 / TASK_CLASSES[old_task]
            headroom = before - chance
            valid = headroom >= minimum_headroom
            raw = before - after
            rows.append(
                {
                    "old_task": old_task,
                    "learned_task": learned_task,
                    "before_stage": after_index,
                    "after_stage": after_index + 1,
                    "before": before,
                    "after": after,
                    "chance": chance,
                    "headroom": headroom,
                    "raw_forgetting": raw,
                    "relative_forgetting": raw / headroom if valid else None,
                    "relative_valid": valid,
                }
            )
    return rows


def _atomic_torch_save(path: Path, document: object) -> None:
    if path.exists():
        raise DatasetProtocolError(f"refusing to overwrite continual artifact {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(document, temporary)
    temporary.replace(path)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    if path.exists():
        raise DatasetProtocolError(f"refusing to overwrite predictions {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_json(path: Path, document: Mapping[str, object]) -> None:
    if path.exists():
        raise DatasetProtocolError(f"refusing to overwrite continual result {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def run_sequential_finetuning(
    *,
    config_path: str | Path,
    order_name: str,
    seed: int | None = None,
    cache_root: str | Path,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    output_dir: str | Path,
    device: str | torch.device,
) -> dict[str, object]:
    from eeg_forgetting.data.contracts import load_yaml

    config_path = Path(config_path)
    config = load_yaml(config_path)
    status = config["status"]
    if status not in {"locked_smoke", "locked_main", "locked_pairwise", "exploratory_post_gate"} or order_name not in config["orders"]:
        raise DatasetProtocolError("unknown sequential FT smoke order")
    canonical_tasks = tuple(config.get("canonical_tasks", CANONICAL_TASKS))
    if (
        not canonical_tasks
        or len(set(canonical_tasks)) != len(canonical_tasks)
        or not set(canonical_tasks).issubset(TASK_CLASSES)
    ):
        raise DatasetProtocolError("sequential config has invalid canonical tasks")
    order = tuple(config["orders"][order_name])
    if status == "locked_pairwise":
        if len(order) != 2 or len(set(order)) != 2 or not set(order).issubset(canonical_tasks):
            raise DatasetProtocolError(
                "pairwise sequential order must contain two distinct locked tasks"
            )
    elif set(order) != set(canonical_tasks) or len(order) != len(canonical_tasks):
        raise DatasetProtocolError("sequential FT order must contain every locked task once")
    output_dir = Path(output_dir)
    result_path = output_dir / "result.json"
    if result_path.exists():
        raise DatasetProtocolError(f"refusing to overwrite continual result {result_path}")
    declared_seeds = tuple(
        int(value) for value in config.get("seeds", [config.get("seed")])
    )
    seed = declared_seeds[0] if seed is None else int(seed)
    if seed not in declared_seeds:
        raise DatasetProtocolError(f"seed {seed} is not declared by the sequential config")
    set_determinism(seed)
    device = torch.device(device)
    torch.cuda.set_device(device)
    cache_root = Path(cache_root)
    configured_roots = config.get("cache_roots")
    if configured_roots is None:
        task_cache_roots = {task: cache_root for task in canonical_tasks}
    else:
        project_root = config_path.resolve().parents[2]
        if set(configured_roots) != set(canonical_tasks):
            raise DatasetProtocolError("configured cache roots must cover every locked task")
        task_cache_roots = {
            task: (
                Path(str(value)).resolve()
                if Path(str(value)).is_absolute()
                else (project_root / str(value)).resolve()
            )
            for task, value in configured_roots.items()
        }
    train_sets = {
        task: CachedEEGDataset(task_cache_roots[task] / task / "train" / "index.json")
        for task in canonical_tasks
    }
    test_sets = {
        task: CachedEEGDataset(task_cache_roots[task] / task / "test" / "index.json")
        for task in canonical_tasks
    }
    test_loaders = {
        task: DataLoader(
            dataset,
            batch_size=int(config["training"]["batch_size"]),
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        for task, dataset in test_sets.items()
    }
    backbone = load_pretrained_backbone(
        checkpoint_path,
        expected_sha256=checkpoint_sha256,
        map_location="cpu",
    )
    model = MultiHeadCBraMod(backbone, tasks=canonical_tasks).to(device)
    config_sha = sha256_file(config_path)
    completed_stages = []
    resume_index = 0
    for index, task in enumerate(order, start=1):
        stage_dir = output_dir / f"stage-{index:02d}-{task}"
        stage_result = stage_dir / "stage.json"
        if not stage_result.exists():
            if stage_dir.exists() and any(stage_dir.iterdir()):
                raise DatasetProtocolError(f"incomplete continual stage requires recovery: {stage_dir}")
            break
        with stage_result.open(encoding="utf-8") as handle:
            document = json.load(handle)
        checkpoint = stage_dir / str(document["checkpoint"]["file"])
        if document["config_sha256"] != config_sha or sha256_file(checkpoint) != document[
            "checkpoint"
        ]["sha256"]:
            raise DatasetProtocolError(f"continual resume digest mismatch at {stage_dir}")
        completed_stages.append(document)
        resume_index = index
    if resume_index:
        latest = output_dir / f"stage-{resume_index:02d}-{order[resume_index - 1]}"
        with (latest / "stage.json").open(encoding="utf-8") as handle:
            latest_document = json.load(handle)
        state = torch.load(
            latest / latest_document["checkpoint"]["file"],
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state["model_state_dict"])
        print(f"resumed {order_name} after stage {resume_index}", flush=True)

    training = config["training"]
    depth = int(config["model"]["plastic_final_encoder_blocks"])
    for index in range(resume_index, len(order)):
        task = order[index]
        stage_number = index + 1
        trainable = model.prepare_stage(task, depth=depth)
        backbone_parameters = [
            parameter
            for parameter in model.backbone.parameters()
            if parameter.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": backbone_parameters,
                    "lr": float(training["backbone_learning_rate"]),
                },
                {
                    "params": model.heads[task].parameters(),
                    "lr": float(training["current_head_learning_rate"]),
                },
            ],
            weight_decay=float(training["weight_decay"]),
        )
        steps = int(training["optimizer_steps_per_task"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=steps,
            eta_min=float(training["minimum_learning_rate"]),
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
            train_sets[task],
            batch_sampler=FixedStepBatchSampler(len(train_sets[task]), settings),
            num_workers=0,
            pin_memory=True,
        )
        losses: deque[float] = deque(maxlen=int(training["progress_interval_steps"]))
        curve = []
        for step, (signals, labels, _subjects) in enumerate(loader, start=1):
            signals = signals.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(
                model(task, signals),
                labels,
                label_smoothing=float(training["label_smoothing"]),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                trainable, float(training["gradient_clip_norm"])
            )
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
            if step % int(training["progress_interval_steps"]) == 0:
                point = {"step": step, "train_loss": float(np.mean(losses))}
                curve.append(point)
                print(
                    f"{order_name} stage={stage_number} task={task} "
                    f"step={step}/{steps} loss={point['train_loss']:.5f}",
                    flush=True,
                )

        stage_dir = output_dir / f"stage-{stage_number:02d}-{task}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        evaluations = {}
        prediction_documents = {}
        for seen_task in order[:stage_number]:
            metrics, arrays = evaluate_task(
                model, seen_task, test_loaders[seen_task], device=device
            )
            prediction_path = stage_dir / f"predictions-{seen_task}.npz"
            _atomic_npz(prediction_path, arrays)
            evaluations[seen_task] = metrics
            prediction_documents[seen_task] = {
                "file": prediction_path.name,
                "sha256": sha256_file(prediction_path),
            }
            print(
                f"{order_name} after={task} eval={seen_task} "
                f"subject_BA={metrics['mean_subject_balanced_accuracy']:.5f}",
                flush=True,
            )
        checkpoint_output = stage_dir / "checkpoint.pt"
        _atomic_torch_save(
            checkpoint_output,
            {
                "model_state_dict": model.state_dict(),
                "stage": stage_number,
                "task": task,
                "order": order,
                "config_sha256": config_sha,
            },
        )
        stage_document = {
            "schema_version": 1,
            "run": config["id"],
            "order_name": order_name,
            "order": list(order),
            "stage": stage_number,
            "learned_task": task,
            "optimizer_steps": steps,
            "config_sha256": config_sha,
            "curve": curve,
            "evaluations": evaluations,
            "predictions": prediction_documents,
            "checkpoint": {
                "file": checkpoint_output.name,
                "sha256": sha256_file(checkpoint_output),
            },
        }
        _atomic_json(stage_dir / "stage.json", stage_document)
        completed_stages.append(stage_document)

    forgetting = pairwise_forgetting(
        completed_stages,
        order,
        minimum_headroom=float(
            config["forgetting"]["minimum_valid_headroom_above_chance"]
        ),
    )
    result = {
        "schema_version": 1,
        "run": config["id"],
        "method": config["method"],
        "order_name": order_name,
        "order": list(order),
        "seed": seed,
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
            }
            for stage in completed_stages
        ],
        "pairwise_forgetting": forgetting,
        "device": str(device),
        "torch": torch.__version__,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(result_path, result)
    return result
