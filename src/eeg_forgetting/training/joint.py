from __future__ import annotations

import json
from collections import deque
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from eeg_forgetting.data.cache import CachedEEGDataset
from eeg_forgetting.data.contracts import DatasetProtocolError, load_yaml
from eeg_forgetting.data.manifests import sha256_file
from eeg_forgetting.models.cbramod import load_pretrained_backbone

from .continual import (
    CANONICAL_TASKS,
    MultiHeadCBraMod,
    _atomic_json,
    _atomic_npz,
    _atomic_torch_save,
    evaluate_task,
)
from .pilot import FixedStepBatchSampler, PilotSettings, set_determinism


def joint_task_schedule(rounds: int) -> tuple[str, ...]:
    if rounds <= 0:
        raise DatasetProtocolError("joint training needs positive task rounds")
    return CANONICAL_TASKS * rounds


def run_joint_training(
    *,
    config_path: str | Path,
    seed: int,
    cache_root: str | Path,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    output_dir: str | Path,
    device: str | torch.device,
) -> dict[str, object]:
    config_path = Path(config_path)
    config = load_yaml(config_path)
    if config.get("status") != "locked_main" or config.get("method") != "joint_training_upper_bound":
        raise DatasetProtocolError("joint config is not locked main authority")
    if tuple(config["canonical_tasks"]) != CANONICAL_TASKS:
        raise DatasetProtocolError("joint config changed canonical task order")
    if int(seed) not in tuple(int(value) for value in config["seeds"]):
        raise DatasetProtocolError(f"joint seed {seed} is not declared")
    training = config["training"]
    rounds = int(training["optimizer_updates_per_task"])
    if int(training["total_optimizer_updates"]) != rounds * len(CANONICAL_TASKS):
        raise DatasetProtocolError("joint total updates do not match task-balanced rounds")
    if tuple(training["task_order_within_round"]) != CANONICAL_TASKS:
        raise DatasetProtocolError("joint task schedule is not canonical round-robin")
    output_dir = Path(output_dir)
    result_path = output_dir / "result.json"
    if result_path.exists():
        raise DatasetProtocolError(f"refusing to overwrite joint result {result_path}")

    set_determinism(int(seed))
    device = torch.device(device)
    torch.cuda.set_device(device)
    cache_root = Path(cache_root)
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
            batch_size=int(training["batch_size"]),
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        for task, dataset in test_sets.items()
    }
    backbone = load_pretrained_backbone(
        checkpoint_path, expected_sha256=checkpoint_sha256, map_location="cpu"
    )
    model = MultiHeadCBraMod(backbone).to(device)
    trainable = model.prepare_joint(
        depth=int(config["model"]["plastic_final_encoder_blocks"])
    )
    backbone_parameters = [
        parameter for parameter in model.backbone.parameters() if parameter.requires_grad
    ]
    head_parameters = [
        parameter for head in model.heads.values() for parameter in head.parameters()
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": float(training["backbone_learning_rate"])},
            {"params": head_parameters, "lr": float(training["task_head_learning_rate"])},
        ],
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(training["total_optimizer_updates"]),
        eta_min=float(training["minimum_learning_rate"]),
    )
    loaders = {}
    for task, dataset in train_sets.items():
        settings = PilotSettings(
            dataset=task,
            depth=int(config["model"]["plastic_final_encoder_blocks"]),
            head="flatten_mlp",
            seed=int(seed),
            batch_size=int(training["batch_size"]),
            optimizer_steps=rounds,
            validation_interval_steps=int(training["progress_interval_rounds"]),
        )
        loaders[task] = iter(
            DataLoader(
                dataset,
                batch_sampler=FixedStepBatchSampler(len(dataset), settings),
                num_workers=0,
                pin_memory=True,
            )
        )

    interval = int(training["progress_interval_rounds"])
    recent = {task: deque(maxlen=interval) for task in CANONICAL_TASKS}
    curve = []
    update = 0
    for round_number in range(1, rounds + 1):
        for task in CANONICAL_TASKS:
            signals, labels, _subjects = next(loaders[task])
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
            update += 1
            recent[task].append(float(loss.detach()))
        if round_number % interval == 0:
            point = {
                "round": round_number,
                "optimizer_update": update,
                "task_train_loss": {
                    task: float(np.mean(recent[task])) for task in CANONICAL_TASKS
                },
            }
            curve.append(point)
            print(
                f"joint seed={seed} round={round_number}/{rounds} "
                + " ".join(
                    f"{task}={point['task_train_loss'][task]:.5f}"
                    for task in CANONICAL_TASKS
                ),
                flush=True,
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    evaluations = {}
    predictions = {}
    for task in CANONICAL_TASKS:
        metrics, arrays = evaluate_task(model, task, test_loaders[task], device=device)
        prediction_path = output_dir / f"predictions-{task}.npz"
        _atomic_npz(prediction_path, arrays)
        evaluations[task] = metrics
        predictions[task] = {
            "file": prediction_path.name,
            "sha256": sha256_file(prediction_path),
        }
        print(
            f"joint seed={seed} eval={task} "
            f"subject_BA={metrics['mean_subject_balanced_accuracy']:.5f}",
            flush=True,
        )
    checkpoint_output = output_dir / "final.pt"
    config_sha = sha256_file(config_path)
    _atomic_torch_save(
        checkpoint_output,
        {
            "model_state_dict": model.state_dict(),
            "seed": int(seed),
            "optimizer_updates": update,
            "config_sha256": config_sha,
        },
    )
    result = {
        "schema_version": 1,
        "run": config["id"],
        "method": config["method"],
        "seed": int(seed),
        "config_sha256": config_sha,
        "pretrained_checkpoint_sha256": checkpoint_sha256,
        "optimizer_updates": update,
        "updates_per_task": rounds,
        "cache_indices": {
            split: {
                task: sha256_file(dataset.index_path)
                for task, dataset in datasets.items()
            }
            for split, datasets in (("train", train_sets), ("test", test_sets))
        },
        "curve": curve,
        "evaluations": evaluations,
        "predictions": predictions,
        "memory": {"continual_state_bytes": 0},
        "checkpoint": {
            "file": checkpoint_output.name,
            "sha256": sha256_file(checkpoint_output),
        },
        "device": str(device),
        "torch": torch.__version__,
    }
    _atomic_json(result_path, result)
    return result
