from __future__ import annotations

import hashlib
from collections import Counter, OrderedDict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from eeg_forgetting.data.contracts import DatasetProtocolError


FisherDict = OrderedDict[str, Tensor]


def deterministic_sample_indices(
    population: int, *, sample_count: int, seed: int
) -> np.ndarray:
    if population <= 0 or sample_count <= 0:
        raise DatasetProtocolError("Fisher sampling needs positive sizes")
    if sample_count > population:
        raise DatasetProtocolError(
            f"Fisher sample count {sample_count} exceeds population {population}"
        )
    generator = np.random.default_rng(seed)
    return generator.choice(population, size=sample_count, replace=False).astype(
        np.int64, copy=False
    )


def selected_plastic_parameters(model: nn.Module, *, final_blocks: int) -> FisherDict:
    if final_blocks <= 0:
        raise DatasetProtocolError("Fisher needs at least one plastic encoder block")
    layer_numbers = [
        int(name.split(".")[3])
        for name, _parameter in model.named_parameters()
        if name.startswith("backbone.encoder.layers.")
    ]
    if not layer_numbers:
        raise DatasetProtocolError("model exposes no shared CBraMod encoder parameters")
    if final_blocks > max(layer_numbers) + 1:
        raise DatasetProtocolError("Fisher plastic depth exceeds the encoder depth")
    first_layer = max(layer_numbers) + 1 - final_blocks
    selected = OrderedDict(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith("backbone.encoder.layers.")
        and int(name.split(".")[3]) >= first_layer
    )
    if not selected:
        raise DatasetProtocolError("no plastic backbone parameters selected for Fisher")
    return selected


def diagonal_empirical_fisher(
    model: nn.Module,
    dataset: Dataset,
    indices: Sequence[int] | np.ndarray,
    *,
    final_blocks: int,
    microbatch_size: int,
    device: str | torch.device,
    progress: Callable[[int, int], None] | None = None,
) -> FisherDict:
    if microbatch_size <= 0:
        raise DatasetProtocolError("Fisher microbatch size must be positive")
    indices = [int(index) for index in indices]
    if not indices:
        raise DatasetProtocolError("Fisher cannot use an empty sample")
    device = torch.device(device)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    parameters = selected_plastic_parameters(model, final_blocks=final_blocks)
    for parameter in parameters.values():
        parameter.requires_grad = True
    accumulator = OrderedDict(
        (name, torch.zeros_like(parameter, dtype=torch.float64, device=device))
        for name, parameter in parameters.items()
    )
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=microbatch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    completed = 0
    for signals, labels, _subjects in loader:
        signals = signals.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        for signal, label in zip(signals, labels, strict=True):
            model.zero_grad(set_to_none=True)
            logits = model(signal.unsqueeze(0))
            F.cross_entropy(logits, label.unsqueeze(0)).backward()
            for name, parameter in parameters.items():
                if parameter.grad is None:
                    raise DatasetProtocolError(
                        f"Fisher parameter received no gradient: {name}"
                    )
                accumulator[name].add_(parameter.grad.detach().double().square())
            completed += 1
            if progress is not None:
                progress(completed, len(indices))
    return OrderedDict(
        (name, (value / len(indices)).float().cpu())
        for name, value in accumulator.items()
    )


def combine_fisher(
    left: Mapping[str, Tensor],
    right: Mapping[str, Tensor],
    *,
    left_samples: int,
    right_samples: int,
) -> FisherDict:
    if left_samples <= 0 or right_samples <= 0 or left.keys() != right.keys():
        raise DatasetProtocolError("cannot combine incompatible Fisher signatures")
    total = left_samples + right_samples
    return OrderedDict(
        (
            name,
            (left[name] * left_samples + right[name] * right_samples) / total,
        )
        for name in left
    )


def layer_l2_normalize(importance: Mapping[str, Tensor]) -> FisherDict:
    if not importance:
        raise DatasetProtocolError("cannot normalize an empty Fisher signature")
    squared_norms: dict[str, Tensor] = {}
    for name, values in importance.items():
        layer = encoder_layer_key(name)
        squared_norms[layer] = squared_norms.get(
            layer, torch.zeros((), dtype=torch.float64)
        ) + values.double().square().sum()
    norms = {layer: value.sqrt() for layer, value in squared_norms.items()}
    if any(float(value) == 0.0 for value in norms.values()):
        raise DatasetProtocolError("Fisher signature contains a zero-importance layer")
    return OrderedDict(
        (name, values / norms[encoder_layer_key(name)].to(values.dtype))
        for name, values in importance.items()
    )


def fisher_cosine(left: Mapping[str, Tensor], right: Mapping[str, Tensor]) -> float:
    if not left or left.keys() != right.keys():
        raise DatasetProtocolError("Fisher cosine needs aligned non-empty signatures")
    numerator = sum(
        (left[name].double() * right[name].double()).sum() for name in left
    )
    left_norm = sum(left[name].double().square().sum() for name in left).sqrt()
    right_norm = sum(right[name].double().square().sum() for name in right).sqrt()
    if float(left_norm) == 0.0 or float(right_norm) == 0.0:
        raise DatasetProtocolError("Fisher cosine is undefined for a zero vector")
    return float(numerator / (left_norm * right_norm))


def encoder_layer_key(parameter_name: str) -> str:
    parts = parameter_name.split(".")
    if len(parts) < 5 or parts[:3] != ["backbone", "encoder", "layers"]:
        raise DatasetProtocolError(f"not a CBraMod encoder parameter: {parameter_name}")
    return ".".join(parts[:4])


def indices_sha256(indices: Sequence[int] | np.ndarray) -> str:
    values = np.asarray(indices, dtype="<i8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def sampled_inventory(dataset: Dataset, indices: Sequence[int]) -> dict[str, object]:
    labels: Counter[int] = Counter()
    subjects: Counter[str] = Counter()
    for index in indices:
        _signal, label, subject = dataset[int(index)]
        labels[int(label)] += 1
        subjects[str(subject)] += 1
    return {
        "labels": dict(sorted(labels.items())),
        "subjects": dict(sorted(subjects.items())),
    }


def save_fisher(path: str | Path, importance: Mapping[str, Tensor]) -> None:
    path = Path(path)
    if path.exists():
        raise DatasetProtocolError(f"refusing to overwrite Fisher signature {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(OrderedDict((name, value.cpu()) for name, value in importance.items()), temporary)
    temporary.replace(path)


def load_fisher(path: str | Path) -> FisherDict:
    document = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(document, Mapping) or not document:
        raise DatasetProtocolError(f"invalid Fisher signature {path}")
    return OrderedDict((str(name), value) for name, value in document.items())
