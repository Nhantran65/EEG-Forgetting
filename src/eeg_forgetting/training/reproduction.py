from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from eeg_forgetting.data.cache import CachedEEGDataset
from eeg_forgetting.data.contracts import DatasetProtocolError
from eeg_forgetting.data.manifests import sha256_file
from eeg_forgetting.models.cbramod import CBraModTaskModel, load_pretrained_backbone

from .metrics import multiclass_metrics


@torch.no_grad()
def evaluate_physionet(model, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    truth: list[int] = []
    prediction: list[int] = []
    for signals, labels, _subjects in loader:
        signals = signals.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(signals)
        truth.extend(int(value) for value in labels.cpu())
        prediction.extend(int(value) for value in logits.argmax(dim=1).cpu())
    return multiclass_metrics(truth, prediction, classes=4)


def run_physionet_reproduction(
    *,
    cache_root: str | Path,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    config_path: str | Path,
    output_dir: str | Path,
    device: str | torch.device,
) -> dict[str, object]:
    from eeg_forgetting.data.contracts import load_yaml

    config_path = Path(config_path)
    config = load_yaml(config_path)
    output_dir = Path(output_dir)
    best_path = output_dir / "best.pt"
    final_path = output_dir / "final.pt"
    result_path = output_dir / "result.json"
    if any(path.exists() for path in (best_path, final_path, result_path)):
        raise DatasetProtocolError(f"refusing to overwrite reproduction output {output_dir}")
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(device)
    torch.cuda.set_device(device)

    cache_root = Path(cache_root)
    datasets = {
        split: CachedEEGDataset(cache_root / "physionet_mi" / split / "index.json")
        for split in ("train", "validation", "test")
    }
    if any(dataset.index.get("protocol") != "reproduction" for dataset in datasets.values()):
        raise DatasetProtocolError("PhysioNet reproduction cache has the wrong protocol")
    batch_size = int(config["training"]["batch_size"])
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=0,
        ),
        "validation": DataLoader(
            datasets["validation"], batch_size=batch_size, shuffle=False, pin_memory=True
        ),
        "test": DataLoader(
            datasets["test"], batch_size=batch_size, shuffle=False, pin_memory=True
        ),
    }
    backbone = load_pretrained_backbone(
        checkpoint_path,
        expected_sha256=checkpoint_sha256,
        map_location="cpu",
    )
    for parameter in backbone.parameters():
        parameter.requires_grad = True
    model = CBraModTaskModel(
        backbone, 4, head="flatten_mlp", channels=64, patches=4
    ).to(device)
    training = config["training"]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone.parameters(), "lr": float(training["backbone_learning_rate"])},
            {"params": model.classifier.parameters(), "lr": float(training["head_learning_rate"])},
        ],
        weight_decay=float(training["weight_decay"]),
    )
    epochs = int(training["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs * len(loaders["train"]),
        eta_min=float(training["minimum_learning_rate"]),
    )

    curve = []
    best_kappa = -float("inf")
    best_epoch = 0
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for signals, labels, _subjects in loaders["train"]:
            signals = signals.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(
                model(signals),
                labels,
                label_smoothing=float(training["label_smoothing"]),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip_norm"])
            )
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
        validation = evaluate_physionet(model, loaders["validation"], device)
        point = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
            "validation": validation,
        }
        curve.append(point)
        print(
            f"physionet reproduction epoch={epoch}: loss={point['train_loss']:.5f} "
            f"BA={validation['balanced_accuracy']:.5f} "
            f"kappa={validation['cohen_kappa']:.5f} "
            f"weighted_F1={validation['weighted_f1']:.5f}",
            flush=True,
        )
        if validation["cohen_kappa"] > best_kappa:
            best_kappa = validation["cohen_kappa"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise DatasetProtocolError("reproduction produced no checkpoint")

    final_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    test_metrics = evaluate_physionet(model, loaders["test"], device)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, best_path)
    torch.save(final_state, final_path)
    result = {
        "schema_version": 1,
        "run": str(config["id"]),
        "config_sha256": sha256_file(config_path),
        "pretrained_checkpoint_sha256": checkpoint_sha256,
        "cache_indices": {
            split: sha256_file(dataset.index_path) for split, dataset in datasets.items()
        },
        "samples": {split: len(dataset) for split, dataset in datasets.items()},
        "best_epoch": best_epoch,
        "best_validation_kappa": best_kappa,
        "test_at_best_validation": test_metrics,
        "curve": curve,
        "best_checkpoint_sha256": sha256_file(best_path),
        "final_checkpoint_sha256": sha256_file(final_path),
        "torch": torch.__version__,
        "device": str(device),
    }
    temporary = result_path.with_suffix(".json.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(result_path)
    return result
