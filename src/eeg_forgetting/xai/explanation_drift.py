from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset

from eeg_forgetting.data.contracts import DatasetProtocolError
from eeg_forgetting.training.metrics import subject_balanced_accuracy


Band = tuple[str, float, float]


def frozen_stratified_halves(
    labels: Sequence[int], subjects: Sequence[str], *, seed: int
) -> dict[str, object]:
    """Freeze row IDs into deterministic halves within every subject/class stratum."""
    if len(labels) != len(subjects) or len(labels) == 0:
        raise DatasetProtocolError("attribution split needs aligned non-empty rows")
    strata: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, (label, subject) in enumerate(zip(labels, subjects, strict=True)):
        strata[(str(subject), int(label))].append(index)
    fit: list[int] = []
    gate: list[int] = []
    counts: dict[str, dict[str, int]] = {}
    for (subject, label), rows in sorted(strata.items()):
        if len(rows) < 2:
            raise DatasetProtocolError(
                f"subject/class stratum {subject}/{label} has fewer than two rows"
            )
        ranked = sorted(
            rows,
            key=lambda row: hashlib.sha256(
                f"{seed}|{subject}|{label}|{row}".encode()
            ).hexdigest(),
        )
        midpoint = len(ranked) // 2
        fit.extend(ranked[:midpoint])
        gate.extend(ranked[midpoint:])
        counts[f"{subject}:{label}"] = {
            "total": len(ranked),
            "attribution_fit": midpoint,
            "attribution_gate": len(ranked) - midpoint,
        }
    document: dict[str, object] = {
        "method": "sha256_rank_within_subject_class_v1",
        "seed": int(seed),
        "attribution_fit": sorted(fit),
        "attribution_gate": sorted(gate),
        "strata": counts,
    }
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    document["assignment_sha256"] = hashlib.sha256(canonical).hexdigest()
    return document


def frozen_stratified_capped_halves(
    labels: Sequence[int],
    subjects: Sequence[str],
    *,
    seed: int,
    maximum_rows_per_subject_class: int,
) -> dict[str, object]:
    """Freeze at most N hash-ranked rows per subject/class, then split in half."""
    if maximum_rows_per_subject_class < 2:
        raise DatasetProtocolError("capped attribution strata need at least two rows")
    if len(labels) != len(subjects) or len(labels) == 0:
        raise DatasetProtocolError("attribution split needs aligned non-empty rows")
    strata: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, (label, subject) in enumerate(zip(labels, subjects, strict=True)):
        strata[(str(subject), int(label))].append(index)
    fit: list[int] = []
    gate: list[int] = []
    counts: dict[str, dict[str, int]] = {}
    for (subject, label), rows in sorted(strata.items()):
        if len(rows) < 2:
            raise DatasetProtocolError(
                f"subject/class stratum {subject}/{label} has fewer than two rows"
            )
        ranked = sorted(
            rows,
            key=lambda row: hashlib.sha256(
                f"{seed}|{subject}|{label}|{row}".encode()
            ).hexdigest(),
        )
        selected = ranked[:maximum_rows_per_subject_class]
        midpoint = len(selected) // 2
        fit.extend(selected[:midpoint])
        gate.extend(selected[midpoint:])
        counts[f"{subject}:{label}"] = {
            "available": len(ranked),
            "selected": len(selected),
            "attribution_fit": midpoint,
            "attribution_gate": len(selected) - midpoint,
        }
    document: dict[str, object] = {
        "method": "sha256_rank_capped_within_subject_class_v1",
        "seed": int(seed),
        "maximum_rows_per_subject_class": int(maximum_rows_per_subject_class),
        "attribution_fit": sorted(fit),
        "attribution_gate": sorted(gate),
        "strata": counts,
    }
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    document["assignment_sha256"] = hashlib.sha256(canonical).hexdigest()
    return document


def _band_masks(
    samples: int, sampling_rate_hz: float, bands: Sequence[Band], *, device: torch.device
) -> Tensor:
    if samples <= 0 or sampling_rate_hz <= 0 or not bands:
        raise DatasetProtocolError("invalid spectral grid")
    frequencies = torch.fft.rfftfreq(
        samples, d=1.0 / float(sampling_rate_hz), device=device
    )
    masks = []
    for index, (name, low, high) in enumerate(bands):
        if not name or not 0 <= low < high <= sampling_rate_hz / 2:
            raise DatasetProtocolError(f"invalid frequency band {name!r}")
        mask = (frequencies >= low) & (
            frequencies <= high if index == len(bands) - 1 else frequencies < high
        )
        if not bool(mask.any()):
            raise DatasetProtocolError(f"frequency band {name!r} has no FFT bins")
        masks.append(mask)
    stacked = torch.stack(masks)
    if bool((stacked.sum(dim=0) > 1).any()):
        raise DatasetProtocolError("frequency bands overlap")
    return stacked


def apply_spectral_cell_weights(
    signals: Tensor,
    cell_weights: Tensor,
    *,
    sampling_rate_hz: float,
    bands: Sequence[Band],
) -> Tensor:
    """Scale continuous-trial rFFT cells, preserving frequencies outside the bands."""
    if signals.ndim != 4:
        raise DatasetProtocolError("spectral masking expects (batch, channel, patch, point)")
    batch, channels, patches, points = signals.shape
    if cell_weights.ndim == 2:
        cell_weights = cell_weights.unsqueeze(0).expand(batch, -1, -1)
    if cell_weights.shape != (batch, channels, len(bands)):
        raise DatasetProtocolError(
            f"cell weights do not match signals: {tuple(cell_weights.shape)}"
        )
    continuous = signals.reshape(batch, channels, patches * points)
    spectrum = torch.fft.rfft(continuous, dim=-1)
    band_masks = _band_masks(
        continuous.shape[-1], sampling_rate_hz, bands, device=signals.device
    ).to(signals.dtype)
    scale = 1.0 + torch.sum(
        (cell_weights - 1.0).unsqueeze(-1) * band_masks[None, None, :, :],
        dim=2,
    )
    masked = torch.fft.irfft(spectrum * scale, n=continuous.shape[-1], dim=-1)
    return masked.reshape_as(signals)


def reliance_maps(
    truth: Sequence[int] | np.ndarray,
    baseline_prediction: Sequence[int] | np.ndarray,
    occluded_predictions: np.ndarray,
    subjects: Sequence[str],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return subject-equal mean and per-subject BA drops for every cell."""
    truth_array = np.asarray(truth, dtype=np.int64)
    baseline = np.asarray(baseline_prediction, dtype=np.int64)
    occluded = np.asarray(occluded_predictions, dtype=np.int64)
    if occluded.ndim != 2 or occluded.shape[1] != truth_array.size:
        raise DatasetProtocolError("occluded predictions do not align with truth")
    if baseline.shape != truth_array.shape or len(subjects) != truth_array.size:
        raise DatasetProtocolError("baseline predictions do not align with truth")
    _mean, baseline_by_subject = subject_balanced_accuracy(
        truth_array, baseline, subjects
    )
    by_subject = {
        subject: np.empty(occluded.shape[0], dtype=np.float64)
        for subject in baseline_by_subject
    }
    for cell, predictions in enumerate(occluded):
        _mean, occluded_by_subject = subject_balanced_accuracy(
            truth_array, predictions, subjects
        )
        for subject, value in occluded_by_subject.items():
            by_subject[subject][cell] = baseline_by_subject[subject] - value
    aggregate = np.mean(np.stack(list(by_subject.values())), axis=0)
    return aggregate, by_subject


def subject_balanced_accuracy_drop(
    truth: Sequence[int] | np.ndarray,
    baseline_prediction: Sequence[int] | np.ndarray,
    masked_prediction: Sequence[int] | np.ndarray,
    subjects: Sequence[str],
) -> tuple[float, dict[str, float]]:
    baseline_mean, baseline_by_subject = subject_balanced_accuracy(
        truth, baseline_prediction, subjects
    )
    masked_mean, masked_by_subject = subject_balanced_accuracy(
        truth, masked_prediction, subjects
    )
    return baseline_mean - masked_mean, {
        subject: baseline_by_subject[subject] - masked_by_subject[subject]
        for subject in baseline_by_subject
    }


@torch.no_grad()
def predict_spectral_masks(
    model: nn.Module,
    task: str,
    dataset: Dataset,
    indices: Sequence[int],
    cell_weights: Tensor,
    *,
    sampling_rate_hz: float,
    bands: Sequence[Band],
    batch_size: int,
    mask_chunk_size: int,
    device: torch.device,
) -> dict[str, object]:
    """Predict a fixed row set under one or more channel-band masks."""
    if cell_weights.ndim != 3 or len(indices) == 0:
        raise DatasetProtocolError("mask prediction needs (mask, channel, band) weights")
    if batch_size <= 0 or mask_chunk_size <= 0:
        raise DatasetProtocolError("mask prediction batch sizes must be positive")
    loader = DataLoader(
        Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    weights = cell_weights.to(device=device, dtype=torch.float32)
    truth: list[np.ndarray] = []
    subjects: list[str] = []
    prediction_batches: list[np.ndarray] = []
    for signals, labels, batch_subjects in loader:
        signals = signals.to(device, non_blocking=True)
        batch = signals.shape[0]
        chunk_predictions = []
        for start in range(0, weights.shape[0], mask_chunk_size):
            chunk = weights[start : start + mask_chunk_size]
            count = chunk.shape[0]
            repeated_signals = signals.repeat(count, 1, 1, 1)
            repeated_weights = chunk.repeat_interleave(batch, dim=0)
            masked = apply_spectral_cell_weights(
                repeated_signals,
                repeated_weights,
                sampling_rate_hz=sampling_rate_hz,
                bands=bands,
            )
            predicted = model(task, masked).argmax(dim=1).reshape(count, batch)
            chunk_predictions.append(predicted.cpu().numpy())
        prediction_batches.append(np.concatenate(chunk_predictions, axis=0))
        truth.append(labels.numpy())
        subjects.extend(str(value) for value in batch_subjects)
    return {
        "truth": np.concatenate(truth),
        "subjects": np.asarray(subjects, dtype=np.str_),
        "predictions": np.concatenate(prediction_batches, axis=1),
        "indices": np.asarray(indices, dtype=np.int64),
    }


@torch.no_grad()
def score_spectral_masks(
    model: nn.Module,
    task: str,
    dataset: Dataset,
    indices: Sequence[int],
    cell_weights: Tensor,
    *,
    sampling_rate_hz: float,
    bands: Sequence[Band],
    batch_size: int,
    mask_chunk_size: int,
    device: torch.device,
) -> dict[str, object]:
    """Return true-vs-best-other classification margins under spectral masks."""
    if cell_weights.ndim != 3 or len(indices) == 0:
        raise DatasetProtocolError("mask scoring needs rows and 3D cell weights")
    loader = DataLoader(
        Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    weights = cell_weights.to(device=device, dtype=torch.float32)
    truth, subjects, margin_batches, prediction_batches = [], [], [], []
    for signals, labels, batch_subjects in loader:
        signals = signals.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch = signals.shape[0]
        chunks = []
        prediction_chunks = []
        for start in range(0, weights.shape[0], mask_chunk_size):
            chunk = weights[start : start + mask_chunk_size]
            count = chunk.shape[0]
            repeated_signals = signals.repeat(count, 1, 1, 1)
            repeated_weights = chunk.repeat_interleave(batch, dim=0)
            logits = model(
                task,
                apply_spectral_cell_weights(
                    repeated_signals,
                    repeated_weights,
                    sampling_rate_hz=sampling_rate_hz,
                    bands=bands,
                ),
            ).reshape(count, batch, -1)
            repeated_labels = labels.unsqueeze(0).expand(count, batch)
            true_logit = logits.gather(2, repeated_labels.unsqueeze(-1))[..., 0]
            competitors = logits.scatter(
                2,
                repeated_labels.unsqueeze(-1),
                torch.full_like(repeated_labels.unsqueeze(-1), -torch.inf, dtype=logits.dtype),
            ).max(dim=2).values
            chunks.append((true_logit - competitors).cpu().numpy())
            prediction_chunks.append(logits.argmax(dim=2).cpu().numpy())
        margin_batches.append(np.concatenate(chunks, axis=0))
        prediction_batches.append(np.concatenate(prediction_chunks, axis=0))
        truth.append(labels.cpu().numpy())
        subjects.extend(str(value) for value in batch_subjects)
    return {
        "truth": np.concatenate(truth),
        "subjects": np.asarray(subjects, dtype=np.str_),
        "margins": np.concatenate(margin_batches, axis=1),
        "predictions": np.concatenate(prediction_batches, axis=1),
        "indices": np.asarray(indices, dtype=np.int64),
    }


def subject_equal_margin_drop(
    baseline_margin: np.ndarray,
    masked_margins: np.ndarray,
    subjects: Sequence[str],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    baseline = np.asarray(baseline_margin, dtype=np.float64)
    masked = np.asarray(masked_margins, dtype=np.float64)
    subject_array = np.asarray(subjects, dtype=np.str_)
    if masked.ndim != 2 or masked.shape[1] != baseline.size or subject_array.size != baseline.size:
        raise DatasetProtocolError("margin arrays do not align")
    by_subject = {
        subject: (baseline[subject_array == subject][None, :] - masked[:, subject_array == subject]).mean(axis=1)
        for subject in sorted(set(subject_array))
    }
    return np.mean(np.stack(list(by_subject.values())), axis=0), by_subject


def subject_class_balanced_margin_drop(
    baseline_margin: np.ndarray,
    masked_margins: np.ndarray,
    labels: Sequence[int],
    subjects: Sequence[str],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Average margin drop within class, then equally across classes and subjects."""
    baseline = np.asarray(baseline_margin, dtype=np.float64)
    masked = np.asarray(masked_margins, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64)
    subject_array = np.asarray(subjects, dtype=np.str_)
    if (
        masked.ndim != 2
        or masked.shape[1] != baseline.size
        or label_array.size != baseline.size
        or subject_array.size != baseline.size
    ):
        raise DatasetProtocolError("class-balanced margin arrays do not align")
    by_subject = {}
    for subject in sorted(set(subject_array)):
        subject_rows = subject_array == subject
        class_effects = []
        for label in sorted(set(label_array[subject_rows])):
            rows = subject_rows & (label_array == label)
            class_effects.append(
                (baseline[rows][None, :] - masked[:, rows]).mean(axis=1)
            )
        by_subject[subject] = np.mean(np.stack(class_effects), axis=0)
    return np.mean(np.stack(list(by_subject.values())), axis=0), by_subject


def ridge_mask_coefficients(
    indicators: np.ndarray, responses: np.ndarray, *, alpha: float
) -> tuple[np.ndarray, float]:
    x = np.asarray(indicators, dtype=np.float64)
    y = np.asarray(responses, dtype=np.float64)
    if x.ndim != 2 or y.shape != (x.shape[0],) or alpha <= 0:
        raise DatasetProtocolError("invalid randomized-mask ridge inputs")
    x_mean = x.mean(axis=0)
    y_mean = float(y.mean())
    centered = x - x_mean
    coefficients = np.linalg.solve(
        centered.T @ centered + alpha * np.eye(x.shape[1]), centered.T @ (y - y_mean)
    )
    intercept = y_mean - float(x_mean @ coefficients)
    return coefficients, intercept


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if left.shape != right.shape or left.ndim != 1 or denominator <= 0:
        raise DatasetProtocolError("cosine similarity needs aligned nonzero vectors")
    return float(left @ right / denominator)


def positive_distribution(values: Sequence[float] | np.ndarray) -> np.ndarray:
    positive = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    mass = float(positive.sum())
    if not np.isfinite(mass) or mass <= 0:
        raise DatasetProtocolError("reliance map has no finite positive mass")
    return positive / mass


def jensen_shannon_divergence(
    left: Sequence[float] | np.ndarray, right: Sequence[float] | np.ndarray
) -> float:
    """Base-2 JSD of positive, L1-normalized reliance maps, in [0, 1]."""
    p = positive_distribution(left)
    q = positive_distribution(right)
    if p.shape != q.shape:
        raise DatasetProtocolError("JSD maps have different shapes")
    midpoint = 0.5 * (p + q)
    left_terms = np.zeros_like(p)
    right_terms = np.zeros_like(q)
    left_nonzero = p > 0
    right_nonzero = q > 0
    left_terms[left_nonzero] = p[left_nonzero] * np.log2(
        p[left_nonzero] / midpoint[left_nonzero]
    )
    right_terms[right_nonzero] = q[right_nonzero] * np.log2(
        q[right_nonzero] / midpoint[right_nonzero]
    )
    return float(0.5 * (left_terms.sum() + right_terms.sum()))


def noise_corrected_symmetric_jsd(
    before_fit: Sequence[float] | np.ndarray,
    before_gate: Sequence[float] | np.ndarray,
    after_fit: Sequence[float] | np.ndarray,
    after_gate: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """Cross-checkpoint JSD minus same-checkpoint split noise, symmetrically."""
    cross = 0.5 * (
        jensen_shannon_divergence(before_fit, after_gate)
        + jensen_shannon_divergence(before_gate, after_fit)
    )
    within = 0.5 * (
        jensen_shannon_divergence(before_fit, before_gate)
        + jensen_shannon_divergence(after_fit, after_gate)
    )
    return {
        "cross_checkpoint_jsd": float(cross),
        "within_checkpoint_noise_jsd": float(within),
        "noise_corrected_ped": float(cross - within),
    }


def spectral_mask_integrated_gradients(
    model: nn.Module,
    task: str,
    signals: Tensor,
    labels: Tensor,
    *,
    sampling_rate_hz: float,
    bands: Sequence[Band],
    steps: int,
    alpha_chunk_size: int,
) -> tuple[Tensor, Tensor]:
    """IG from zero to one spectral cell masks for each sample's true-class logit."""
    if steps <= 0 or alpha_chunk_size <= 0:
        raise DatasetProtocolError("IG steps and alpha chunk size must be positive")
    if signals.shape[0] != labels.shape[0]:
        raise DatasetProtocolError("IG labels do not align with signals")
    batch, channels = signals.shape[:2]
    total_gradient = torch.zeros(
        batch, channels, len(bands), device=signals.device, dtype=signals.dtype
    )
    alphas = (torch.arange(steps, device=signals.device, dtype=signals.dtype) + 0.5) / steps
    for start in range(0, steps, alpha_chunk_size):
        alpha = alphas[start : start + alpha_chunk_size]
        count = alpha.numel()
        weights = (
            alpha[:, None, None, None]
            .expand(count, batch, channels, len(bands))
            .reshape(count * batch, channels, len(bands))
            .clone()
            .requires_grad_(True)
        )
        repeated_signals = signals.repeat(count, 1, 1, 1)
        repeated_labels = labels.repeat(count)
        masked = apply_spectral_cell_weights(
            repeated_signals,
            weights,
            sampling_rate_hz=sampling_rate_hz,
            bands=bands,
        )
        selected = model(task, masked).gather(1, repeated_labels[:, None]).sum()
        gradient = torch.autograd.grad(selected, weights, create_graph=False)[0]
        total_gradient += gradient.reshape(count, batch, channels, len(bands)).sum(0)
    attribution = total_gradient / steps
    with torch.no_grad():
        zeros = torch.zeros(
            batch, channels, len(bands), device=signals.device, dtype=signals.dtype
        )
        ones = torch.ones_like(zeros)
        zero_logits = model(
            task,
            apply_spectral_cell_weights(
                signals, zeros, sampling_rate_hz=sampling_rate_hz, bands=bands
            ),
        ).gather(1, labels[:, None])[:, 0]
        one_logits = model(
            task,
            apply_spectral_cell_weights(
                signals, ones, sampling_rate_hz=sampling_rate_hz, bands=bands
            ),
        ).gather(1, labels[:, None])[:, 0]
        completeness_error = attribution.sum(dim=(1, 2)) - (one_logits - zero_logits)
    return attribution.detach(), completeness_error.detach()


def integrated_gradients_dataset(
    model: nn.Module,
    task: str,
    dataset: Dataset,
    indices: Sequence[int],
    *,
    sampling_rate_hz: float,
    bands: Sequence[Band],
    batch_size: int,
    steps: int,
    alpha_chunk_size: int,
    device: torch.device,
) -> dict[str, object]:
    if len(indices) == 0 or batch_size <= 0:
        raise DatasetProtocolError("IG dataset evaluation needs rows and a positive batch")
    loader = DataLoader(
        Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    attributions = []
    completeness = []
    truth = []
    subjects: list[str] = []
    for signals, labels, batch_subjects in loader:
        signals = signals.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        attribution, error = spectral_mask_integrated_gradients(
            model,
            task,
            signals,
            labels,
            sampling_rate_hz=sampling_rate_hz,
            bands=bands,
            steps=steps,
            alpha_chunk_size=alpha_chunk_size,
        )
        attributions.append(attribution.cpu().numpy())
        completeness.append(error.cpu().numpy())
        truth.append(labels.cpu().numpy())
        subjects.extend(str(value) for value in batch_subjects)
    return {
        "attribution": np.concatenate(attributions),
        "completeness_error": np.concatenate(completeness),
        "truth": np.concatenate(truth),
        "subjects": np.asarray(subjects, dtype=np.str_),
        "indices": np.asarray(indices, dtype=np.int64),
    }


def aggregate_positive_ig(
    attribution: np.ndarray, subjects: Sequence[str]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    values = np.maximum(np.asarray(attribution, dtype=np.float64), 0.0)
    if values.ndim != 3 or values.shape[0] != len(subjects):
        raise DatasetProtocolError("IG attribution does not align with subjects")
    by_subject = {
        subject: values[np.asarray(subjects) == subject].mean(axis=0).reshape(-1)
        for subject in sorted(set(str(value) for value in subjects))
    }
    return np.mean(np.stack(list(by_subject.values())), axis=0), by_subject


def mean_subject_jsd(
    left: Mapping[str, np.ndarray], right: Mapping[str, np.ndarray]
) -> tuple[float, dict[str, float]]:
    if set(left) != set(right) or not left:
        raise DatasetProtocolError("PED subject maps do not align")
    values = {
        subject: jensen_shannon_divergence(left[subject], right[subject])
        for subject in sorted(left)
    }
    return float(np.mean(list(values.values()))), values


def same_checkpoint_split_null(
    truth: np.ndarray,
    baseline_prediction: np.ndarray,
    occluded_predictions: np.ndarray,
    subjects: Sequence[str],
    *,
    replicates: int,
    seed: int,
) -> np.ndarray:
    """Repeated class-stratified half-split PED from one unchanged checkpoint."""
    if replicates <= 0:
        raise DatasetProtocolError("null needs positive replicates")
    truth = np.asarray(truth, dtype=np.int64)
    subject_array = np.asarray(subjects, dtype=np.str_)
    strata: dict[tuple[str, int], np.ndarray] = {}
    for subject in sorted(set(subject_array)):
        for label in sorted(set(truth[subject_array == subject])):
            rows = np.flatnonzero((subject_array == subject) & (truth == label))
            if rows.size < 2:
                raise DatasetProtocolError("null split has a stratum with fewer than two rows")
            strata[(str(subject), int(label))] = rows
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        left_rows: list[int] = []
        right_rows: list[int] = []
        for rows in strata.values():
            shuffled = rng.permutation(rows)
            midpoint = len(shuffled) // 2
            left_rows.extend(int(value) for value in shuffled[:midpoint])
            right_rows.extend(int(value) for value in shuffled[midpoint:])
        left_index = np.asarray(sorted(left_rows), dtype=np.int64)
        right_index = np.asarray(sorted(right_rows), dtype=np.int64)
        _left_mean, left_maps = reliance_maps(
            truth[left_index],
            baseline_prediction[left_index],
            occluded_predictions[:, left_index],
            subject_array[left_index],
        )
        _right_mean, right_maps = reliance_maps(
            truth[right_index],
            baseline_prediction[right_index],
            occluded_predictions[:, right_index],
            subject_array[right_index],
        )
        values[replicate] = mean_subject_jsd(left_maps, right_maps)[0]
    return values
