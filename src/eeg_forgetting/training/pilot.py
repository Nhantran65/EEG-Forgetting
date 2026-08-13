from __future__ import annotations

import copy
import json
import os
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler

from eeg_forgetting.data.cache import CachedEEGDataset
from eeg_forgetting.data.contracts import DatasetProtocolError, fixed_step_batches
from eeg_forgetting.data.manifests import sha256_file
from eeg_forgetting.models.cbramod import CBraModTaskModel, load_pretrained_backbone

from .metrics import balanced_accuracy, subject_balanced_accuracy


TASK_CLASSES = {"bciciv2a": 4, "physionet_mi": 4, "sleep_edf_sc": 5}
TASK_SHAPES = {
    "bciciv2a": (22, 4),
    "physionet_mi": (22, 4),
    "sleep_edf_sc": (2, 30),
}


@dataclass(frozen=True)
class PilotSettings:
    dataset: str
    depth: int
    head: str = "mean_pool_linear"
    pilot_id: str = "cbramod_depth_v1"
    seed: int = 3407
    batch_size: int = 64
    optimizer_steps: int = 500
    validation_interval_steps: int = 100
    backbone_learning_rate: float = 1e-4
    head_learning_rate: float = 5e-4
    weight_decay: float = 5e-2
    gradient_clip_norm: float = 1.0
    label_smoothing: float = 0.1
    minimum_learning_rate: float = 1e-6

    def validate(self) -> None:
        if self.dataset not in TASK_CLASSES:
            raise DatasetProtocolError(f"unknown pilot dataset {self.dataset!r}")
        if self.depth not in {0, 1, 2, 4, 8}:
            raise DatasetProtocolError(f"pilot depth is not predeclared: {self.depth}")
        if self.head not in {"mean_pool_linear", "flatten_linear", "flatten_mlp"}:
            raise DatasetProtocolError(f"unknown pilot head: {self.head}")
        if self.optimizer_steps <= 0 or self.validation_interval_steps <= 0:
            raise DatasetProtocolError("pilot step counts must be positive")


class FixedStepBatchSampler(Sampler[list[int]]):
    def __init__(self, population: int, settings: PilotSettings):
        self.population = population
        self.settings = settings

    def __iter__(self):
        for batch in fixed_step_batches(
            self.population,
            batch_size=self.settings.batch_size,
            optimizer_steps=self.settings.optimizer_steps,
            seed=self.settings.seed,
        ):
            yield [int(value) for value in batch]

    def __len__(self) -> int:
        return self.settings.optimizer_steps


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_finetune_train_mode(model: CBraModTaskModel, depth: int) -> None:
    """Train only the classifier and explicitly plastic encoder blocks."""
    model.eval()
    model.classifier.train()
    if depth:
        for layer in model.backbone.encoder.layers[-depth:]:
            layer.train()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    classes: int,
) -> dict[str, object]:
    model.eval()
    truth: list[int] = []
    prediction: list[int] = []
    subjects: list[str] = []
    losses = []
    for signals, labels, batch_subjects in loader:
        signals = signals.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(signals)
        losses.append(float(F.cross_entropy(logits, labels).detach()))
        truth.extend(int(value) for value in labels.cpu())
        prediction.extend(int(value) for value in logits.argmax(dim=1).cpu())
        subjects.extend(str(value) for value in batch_subjects)
    subject_mean, by_subject = subject_balanced_accuracy(truth, prediction, subjects)
    chance = 1.0 / classes
    return {
        "loss": float(np.mean(losses)),
        "balanced_accuracy": balanced_accuracy(truth, prediction),
        "mean_subject_balanced_accuracy": subject_mean,
        "normalized_mean_subject_balanced_accuracy": (subject_mean - chance) / (1.0 - chance),
        "by_subject": by_subject,
        "samples": len(truth),
    }


def run_pilot(
    settings: PilotSettings,
    *,
    cache_root: str | Path,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    output_dir: str | Path,
    device: str | torch.device,
) -> dict[str, object]:
    settings.validate()
    set_determinism(settings.seed)
    device = torch.device(device)
    torch.cuda.set_device(device)
    cache_root = Path(cache_root)
    train_data = CachedEEGDataset(
        cache_root / settings.dataset / "train" / "index.json"
    )
    validation_data = CachedEEGDataset(
        cache_root / settings.dataset / "validation" / "index.json"
    )
    train_loader = DataLoader(
        train_data,
        batch_sampler=FixedStepBatchSampler(len(train_data), settings),
        num_workers=0,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    backbone = load_pretrained_backbone(
        checkpoint_path,
        expected_sha256=checkpoint_sha256,
        map_location="cpu",
    )
    backbone.set_trainable_depth(settings.depth)
    channels, patches = TASK_SHAPES[settings.dataset]
    model = CBraModTaskModel(
        backbone,
        TASK_CLASSES[settings.dataset],
        head=settings.head,
        channels=channels,
        patches=patches,
    ).to(device)
    backbone_parameters = [
        parameter for parameter in backbone.parameters() if parameter.requires_grad
    ]
    groups = [{"params": model.classifier.parameters(), "lr": settings.head_learning_rate}]
    if backbone_parameters:
        groups.insert(
            0,
            {"params": backbone_parameters, "lr": settings.backbone_learning_rate},
        )
    optimizer = torch.optim.AdamW(groups, weight_decay=settings.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=settings.optimizer_steps,
        eta_min=settings.minimum_learning_rate,
    )

    curve = []
    recent_losses: deque[float] = deque(maxlen=settings.validation_interval_steps)
    best_metric = -float("inf")
    best_step = 0
    best_state = None
    for step, (signals, labels, _subjects) in enumerate(train_loader, start=1):
        set_finetune_train_mode(model, settings.depth)
        signals = signals.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(signals)
        loss = F.cross_entropy(
            logits, labels, label_smoothing=settings.label_smoothing
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            settings.gradient_clip_norm,
        )
        optimizer.step()
        scheduler.step()
        recent_losses.append(float(loss.detach()))
        if step % settings.validation_interval_steps == 0 or step == settings.optimizer_steps:
            metrics = evaluate(
                model,
                validation_loader,
                device=device,
                classes=TASK_CLASSES[settings.dataset],
            )
            point = {
                "step": step,
                "train_loss": float(np.mean(recent_losses)),
                "learning_rates": [group["lr"] for group in optimizer.param_groups],
                "validation": metrics,
            }
            curve.append(point)
            metric = float(metrics["mean_subject_balanced_accuracy"])
            print(
                f"{settings.dataset} depth={settings.depth} step={step}: "
                f"train_loss={point['train_loss']:.5f} subject_BA={metric:.5f}",
                flush=True,
            )
            if metric > best_metric:
                best_metric = metric
                best_step = step
                best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise DatasetProtocolError("pilot produced no validation checkpoint")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_output = output_dir / "best.pt"
    result_output = output_dir / "result.json"
    if checkpoint_output.exists() or result_output.exists():
        raise DatasetProtocolError(f"refusing to overwrite pilot output {output_dir}")
    torch.save(best_state, checkpoint_output)
    result = {
        "schema_version": 1,
        "pilot": settings.pilot_id,
        "settings": asdict(settings),
        "cache_indices": {
            "train": sha256_file(train_data.index_path),
            "validation": sha256_file(validation_data.index_path),
        },
        "pretrained_checkpoint_sha256": checkpoint_sha256,
        "device": str(device),
        "torch": torch.__version__,
        "train_samples": len(train_data),
        "validation_samples": len(validation_data),
        "trainable_backbone_parameters": sum(
            parameter.numel() for parameter in backbone_parameters
        ),
        "best_step": best_step,
        "best_mean_subject_balanced_accuracy": best_metric,
        "curve": curve,
        "best_checkpoint": checkpoint_output.name,
        "best_checkpoint_sha256": sha256_file(checkpoint_output),
    }
    temporary = result_output.with_suffix(".json.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(result_output)
    return result
