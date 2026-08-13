#!/usr/bin/env python3
"""Run pretrained CBraMod forward/loss/backward on frozen real EEG batches."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from eeg_forgetting.data.channels import ChannelRegistry
from eeg_forgetting.data.contracts import DatasetProtocolError, load_yaml
from eeg_forgetting.data.frozen import BCISessionUnit, FrozenManifestSet, ManifestEEGLoader
from eeg_forgetting.models.cbramod import CBraModTaskModel, load_pretrained_backbone


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-set",
        type=Path,
        default=PROJECT_ROOT / "manifests" / "v3" / "manifest-set.json",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--trainable-depth", type=int, default=1)
    return parser.parse_args()


def real_batch(loader: ManifestEEGLoader, dataset: str, batch_size: int):
    units = loader.units(dataset, "train")
    if dataset == "bciciv2a":
        unit = next(
            value
            for value in units
            if isinstance(value, BCISessionUnit) and value.session == "E"
        )
    else:
        unit = units[0]
    samples = loader.load_unit(unit)[:batch_size]
    if len(samples) != batch_size:
        raise DatasetProtocolError(
            f"{unit.unit_id}: requested batch {batch_size}, got {len(samples)}"
        )
    signal = torch.from_numpy(np.stack([sample.signal for sample in samples]))
    labels = torch.tensor([sample.label for sample in samples], dtype=torch.long)
    return unit, signal, labels


def run_task(
    *,
    model_config: dict[str, object],
    loader: ManifestEEGLoader,
    dataset: str,
    classes: int,
    batch_size: int,
    trainable_depth: int,
    device: torch.device,
) -> dict[str, object]:
    checkpoint = model_config["checkpoint"]
    backbone = load_pretrained_backbone(
        PROJECT_ROOT / str(checkpoint["local_path"]),
        expected_sha256=str(checkpoint["sha256"]),
        map_location="cpu",
    )
    backbone.set_trainable_depth(trainable_depth)
    model = CBraModTaskModel(backbone, classes).to(device)
    unit, signal, labels = real_batch(loader, dataset, batch_size)
    signal = signal.to(device)
    labels = labels.to(device)

    model.train()
    model.zero_grad(set_to_none=True)
    logits, features = model(signal, return_features=True)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    trainable_backbone = [
        parameter for parameter in backbone.parameters() if parameter.requires_grad
    ]
    if not trainable_backbone or not all(parameter.grad is not None for parameter in trainable_backbone):
        raise DatasetProtocolError(f"{dataset}: missing trainable-backbone gradients")
    gradients = torch.cat(
        [parameter.grad.detach().reshape(-1) for parameter in trainable_backbone]
    )
    if not torch.isfinite(gradients).all() or float(gradients.norm()) == 0.0:
        raise DatasetProtocolError(f"{dataset}: invalid or zero backbone gradients")
    result = {
        "unit_id": unit.unit_id,
        "input_shape": list(signal.shape),
        "feature_shape": list(features.shape),
        "logits_shape": list(logits.shape),
        "loss": float(loss.detach()),
        "backbone_gradient_norm": float(gradients.norm()),
        "trainable_backbone_parameters": sum(
            parameter.numel() for parameter in trainable_backbone
        ),
        "labels": labels.detach().cpu().tolist(),
    }
    del model, backbone, signal, labels, logits, features, loss, gradients
    torch.cuda.empty_cache()
    gc.collect()
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise DatasetProtocolError(
            "CUDA is unavailable; run this GPU proof in an environment exposing /dev/nvidia*"
        )
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    model_config = load_yaml(PROJECT_ROOT / "configs" / "models" / "cbramod.yaml")
    manifests = FrozenManifestSet(args.manifest_set, project_root=PROJECT_ROOT)
    registry = ChannelRegistry.from_yaml(PROJECT_ROOT / "configs" / "channels.yaml")
    loader = ManifestEEGLoader(manifests, registry)
    tasks = {"bciciv2a": 4, "physionet_mi": 4, "sleep_edf_sc": 5}
    results = {
        dataset: run_task(
            model_config=model_config,
            loader=loader,
            dataset=dataset,
            classes=classes,
            batch_size=args.batch_size,
            trainable_depth=args.trainable_depth,
            device=device,
        )
        for dataset, classes in tasks.items()
    }
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "device": torch.cuda.get_device_name(device),
                "manifest_version": manifests.version,
                "trainable_depth": args.trainable_depth,
                "tasks": results,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
