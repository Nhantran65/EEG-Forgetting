#!/usr/bin/env python3
"""Run the locked BCI-to-Sleep explanation-drift attribution gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from eeg_forgetting.data.cache import CachedEEGDataset
from eeg_forgetting.data.contracts import DatasetProtocolError, load_yaml
from eeg_forgetting.data.manifests import sha256_file
from eeg_forgetting.models.cbramod import CBraMod
from eeg_forgetting.training.continual import MultiHeadCBraMod, _atomic_json, _atomic_npz
from eeg_forgetting.training.metrics import subject_balanced_accuracy
from eeg_forgetting.training.pilot import set_determinism
from eeg_forgetting.xai.explanation_drift import (
    aggregate_positive_ig,
    frozen_stratified_halves,
    integrated_gradients_dataset,
    mean_subject_jsd,
    predict_spectral_masks,
    reliance_maps,
    same_checkpoint_split_null,
    subject_balanced_accuracy_drop,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "xai" / "bci_sleep_explanation_drift_pilot_v1.yaml",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _verify_file(path: Path, expected: str, *, role: str) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise DatasetProtocolError(f"{role} file digest mismatch: {path}")


def _bands(config: dict[str, object]) -> tuple[tuple[str, float, float], ...]:
    return tuple(
        (str(value["name"]), float(value["low_hz"]), float(value["high_hz"]))
        for value in config["cell_registry"]["bands"]
    )


def _dataset_rows(dataset: CachedEEGDataset) -> tuple[list[int], list[str]]:
    labels: list[int] = []
    subjects: list[str] = []
    for index in range(len(dataset)):
        _signal, label, subject = dataset[index]
        labels.append(int(label))
        subjects.append(subject)
    return labels, subjects


def _load_model(
    role: str, specification: dict[str, object], device: torch.device
) -> MultiHeadCBraMod:
    checkpoint_path = _resolve(str(specification["path"]))
    _verify_file(checkpoint_path, str(specification["sha256"]), role=f"{role} checkpoint")
    parent = _resolve(str(specification["parent_config"]))
    _verify_file(
        parent, str(specification["parent_config_sha256"]), role=f"{role} parent config"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("config_sha256") != specification["parent_config_sha256"]:
        raise DatasetProtocolError(f"{role} checkpoint is not bound to its parent config")
    if "expected_stage" in specification and (
        int(checkpoint.get("stage", -1)) != int(specification["expected_stage"])
        or checkpoint.get("task") != specification["expected_task"]
    ):
        raise DatasetProtocolError(f"{role} stage checkpoint identity mismatch")
    if "expected_seed" in specification and int(checkpoint.get("seed", -1)) != int(
        specification["expected_seed"]
    ):
        raise DatasetProtocolError(f"{role} joint checkpoint seed mismatch")
    model = MultiHeadCBraMod(CBraMod())
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except RuntimeError as error:
        raise DatasetProtocolError(f"{role} strict checkpoint load failed: {error}") from error
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.to(device)


def _artifact_metadata(config_sha: str, role: str, checkpoint_sha: str) -> dict[str, str]:
    return {
        "config_sha256": config_sha,
        "role": role,
        "checkpoint_sha256": checkpoint_sha,
    }


def _load_npz(path: Path, metadata: dict[str, str]) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as archive:
        document = {key: archive[key] for key in archive.files}
    for key, expected in metadata.items():
        if key not in document or str(document[key].item()) != expected:
            raise DatasetProtocolError(f"intermediate artifact identity mismatch: {path}")
    return document


def _save_bound_npz(
    path: Path, arrays: dict[str, object], metadata: dict[str, str]
) -> dict[str, np.ndarray]:
    document = {
        **{key: np.asarray(value) for key, value in arrays.items()},
        **{key: np.asarray(value) for key, value in metadata.items()},
    }
    _atomic_npz(path, document)
    return document


def _mask_digest(selected_cells: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(selected_cells, dtype=np.int16).tobytes()).hexdigest()


def _random_cell_sets(
    cells: int, selected: int, replicates: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    while len(rows) < replicates:
        row = np.sort(rng.choice(cells, size=selected, replace=False))
        key = tuple(int(value) for value in row)
        if key not in seen:
            seen.add(key)
            rows.append(row)
    return np.stack(rows)


def _rank_sensitivity(left: np.ndarray, right: np.ndarray) -> float:
    correlation = float(spearmanr(left, right).statistic)
    if not np.isfinite(correlation):
        raise DatasetProtocolError("rank drift is non-finite")
    return 1.0 - correlation


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if config.get("status") not in {"locked_xai_pilot", "locked_xai_replication"}:
        raise DatasetProtocolError("XAI config is not locked")
    if config["cache"].get("test_access_during_pilot") != "forbidden":
        raise DatasetProtocolError("pilot must forbid test access")
    config_sha = sha256_file(args.config)
    if "checkpoints_by_seed" in config:
        if args.seed is None or int(args.seed) not in tuple(
            int(value) for value in config["seeds"]
        ):
            raise DatasetProtocolError("replication seed is missing or not locked")
        run_seed = int(args.seed)
        checkpoint_specs = config["checkpoints_by_seed"].get(run_seed)
        if checkpoint_specs is None:
            checkpoint_specs = config["checkpoints_by_seed"].get(str(run_seed))
        if checkpoint_specs is None:
            raise DatasetProtocolError("replication checkpoint seed is not declared")
    else:
        run_seed = int(config["seed"])
        if args.seed is not None and int(args.seed) != run_seed:
            raise DatasetProtocolError("pilot seed override is forbidden")
        checkpoint_specs = config["checkpoints"]
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    set_determinism(run_seed)
    configured_output = _resolve(str(config["output"]["root"]))
    output = args.output_root or (
        configured_output / f"seed-{run_seed}"
        if "checkpoints_by_seed" in config
        else configured_output
    )
    result_path = output / "result.json"
    if result_path.exists():
        raise DatasetProtocolError(f"refusing to overwrite XAI result {result_path}")
    output.mkdir(parents=True, exist_ok=True)

    cache = config["cache"]
    validation_index = _resolve(str(cache["validation_index"]))
    test_index = _resolve(str(cache["test_index"]))
    _verify_file(validation_index, str(cache["validation_index_sha256"]), role="validation cache")
    _verify_file(test_index, str(cache["test_index_sha256"]), role="untouched test cache")
    dataset = CachedEEGDataset(validation_index)
    if len(dataset) != int(cache["validation_samples"]):
        raise DatasetProtocolError("validation cache sample count changed")
    labels, subjects = _dataset_rows(dataset)
    if sorted(set(subjects)) != sorted(str(value) for value in cache["validation_subjects"]):
        raise DatasetProtocolError("validation subject IDs changed")
    split_config = config["attribution_split"]
    split = frozen_stratified_halves(labels, subjects, seed=int(split_config["seed"]))
    if (
        split["assignment_sha256"] != split_config["assignment_sha256"]
        or len(split["attribution_fit"]) != int(split_config["attribution_fit_samples"])
        or len(split["attribution_gate"]) != int(split_config["attribution_gate_samples"])
    ):
        raise DatasetProtocolError("frozen attribution split identity changed")
    fit = np.asarray(split["attribution_fit"], dtype=np.int64)
    gate = np.asarray(split["attribution_gate"], dtype=np.int64)

    bands = _bands(config)
    channels = tuple(str(value) for value in config["cell_registry"]["channels"])
    cell_count = len(channels) * len(bands)
    cell_registry = [
        {
            "index": channel_index * len(bands) + band_index,
            "channel": channel,
            "band": band[0],
            "low_hz": band[1],
            "high_hz": band[2],
        }
        for channel_index, channel in enumerate(channels)
        for band_index, band in enumerate(bands)
    ]
    individual_masks = torch.ones(cell_count + 1, len(channels), len(bands))
    for cell in range(cell_count):
        individual_masks[cell + 1].view(-1)[cell] = 0.0

    if args.validate_only:
        for role, checkpoint_spec in checkpoint_specs.items():
            model = _load_model(role, checkpoint_spec, device)
            del model
        print(
            json.dumps(
                {
                    "config_sha256": config_sha,
                    "validation_samples": len(dataset),
                    "attribution_fit": len(fit),
                    "attribution_gate": len(gate),
                    "assignment_sha256": split["assignment_sha256"],
                    "cells": cell_count,
                    "checkpoints": {
                        role: value["sha256"]
                        for role, value in checkpoint_specs.items()
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    settings = config["evaluation"]
    fidelity_config = config["fidelity"]
    ig_config = config["integrated_gradients"]
    role_arrays: dict[str, dict[str, np.ndarray]] = {}
    role_results: dict[str, object] = {}
    for role, checkpoint_spec in checkpoint_specs.items():
        checkpoint_sha = str(checkpoint_spec["sha256"])
        metadata = _artifact_metadata(config_sha, role, checkpoint_sha)
        model = _load_model(role, checkpoint_spec, device)
        prediction_path = output / f"{role}-cell-predictions.npz"
        cell_data = _load_npz(prediction_path, metadata)
        if cell_data is None:
            print(f"[{role}] evaluating baseline + {cell_count} individual cells", flush=True)
            predicted = predict_spectral_masks(
                model,
                str(config["task"]),
                dataset,
                range(len(dataset)),
                individual_masks,
                sampling_rate_hz=float(config["sampling_rate_hz"]),
                bands=bands,
                batch_size=int(settings["batch_size"]),
                mask_chunk_size=int(settings["mask_chunk_size"]),
                device=device,
            )
            cell_data = _save_bound_npz(prediction_path, predicted, metadata)
        truth = np.asarray(cell_data["truth"], dtype=np.int64)
        all_subjects = np.asarray(cell_data["subjects"], dtype=np.str_)
        predictions = np.asarray(cell_data["predictions"], dtype=np.int64)
        if predictions.shape != (cell_count + 1, len(dataset)):
            raise DatasetProtocolError(f"{role} cell prediction shape changed")
        baseline = predictions[0]
        cells = predictions[1:]
        fit_map, fit_by_subject = reliance_maps(
            truth[fit], baseline[fit], cells[:, fit], all_subjects[fit]
        )
        gate_map, gate_by_subject = reliance_maps(
            truth[gate], baseline[gate], cells[:, gate], all_subjects[gate]
        )
        baseline_ba, baseline_by_subject = subject_balanced_accuracy(
            truth, baseline, all_subjects
        )

        selected_count = int(round(cell_count * float(fidelity_config["top_fraction"])))
        top_cells = np.sort(np.argsort(fit_map)[-selected_count:])
        random_cells = _random_cell_sets(
            cell_count,
            selected_count,
            int(fidelity_config["random_masks"]),
            int(fidelity_config["random_seed"]),
        )
        selected_sets = np.concatenate((top_cells[None, :], random_cells), axis=0)
        fidelity_masks = torch.ones(len(selected_sets), len(channels), len(bands))
        for mask_index, selected_cells in enumerate(selected_sets):
            fidelity_masks[mask_index].view(-1)[selected_cells] = 0.0
        fidelity_path = output / f"{role}-fidelity-predictions.npz"
        fidelity_data = _load_npz(fidelity_path, metadata)
        if fidelity_data is None:
            print(f"[{role}] evaluating top mask + {len(random_cells)} random controls", flush=True)
            predicted = predict_spectral_masks(
                model,
                str(config["task"]),
                dataset,
                gate,
                fidelity_masks,
                sampling_rate_hz=float(config["sampling_rate_hz"]),
                bands=bands,
                batch_size=int(settings["batch_size"]),
                mask_chunk_size=int(settings["mask_chunk_size"]),
                device=device,
            )
            predicted["selected_cells"] = selected_sets
            fidelity_data = _save_bound_npz(fidelity_path, predicted, metadata)
        if not np.array_equal(fidelity_data["selected_cells"], selected_sets):
            raise DatasetProtocolError(f"{role} fidelity masks changed")
        fidelity_predictions = np.asarray(fidelity_data["predictions"], dtype=np.int64)
        drops = np.asarray(
            [
                subject_balanced_accuracy_drop(
                    truth[gate], baseline[gate], prediction, all_subjects[gate]
                )[0]
                for prediction in fidelity_predictions
            ]
        )
        random_threshold = float(
            np.percentile(drops[1:], float(fidelity_config["threshold_percentile"]))
        )

        ig_path = output / f"{role}-integrated-gradients.npz"
        ig_data = _load_npz(ig_path, metadata)
        if ig_data is None:
            print(f"[{role}] evaluating spectral-mask IG on attribution_fit", flush=True)
            attributed = integrated_gradients_dataset(
                model,
                str(config["task"]),
                dataset,
                fit,
                sampling_rate_hz=float(config["sampling_rate_hz"]),
                bands=bands,
                batch_size=int(ig_config["batch_size"]),
                steps=int(ig_config["steps"]),
                alpha_chunk_size=int(ig_config["alpha_chunk_size"]),
                device=device,
            )
            ig_data = _save_bound_npz(ig_path, attributed, metadata)
        ig_map, ig_by_subject = aggregate_positive_ig(
            np.asarray(ig_data["attribution"]), np.asarray(ig_data["subjects"], dtype=np.str_)
        )
        rank_agreement = float(spearmanr(fit_map, ig_map).statistic)
        completeness = np.asarray(ig_data["completeness_error"], dtype=np.float64)
        if not np.isfinite(rank_agreement) or not np.isfinite(completeness).all():
            raise DatasetProtocolError(f"{role} IG validation is non-finite")

        role_arrays[role] = {
            "truth": truth,
            "subjects": all_subjects,
            "baseline": baseline,
            "cells": cells,
            "fit_map": fit_map,
            "gate_map": gate_map,
        }
        role_results[role] = {
            "checkpoint_sha256": checkpoint_sha,
            "unoccluded_validation_subject_ba": baseline_ba,
            "unoccluded_validation_ba_by_subject": baseline_by_subject,
            "cell_predictions": {
                "file": prediction_path.name,
                "sha256": sha256_file(prediction_path),
            },
            "fidelity_predictions": {
                "file": fidelity_path.name,
                "sha256": sha256_file(fidelity_path),
                "mask_sha256": _mask_digest(selected_sets),
            },
            "integrated_gradients": {
                "file": ig_path.name,
                "sha256": sha256_file(ig_path),
                "rank_agreement_spearman": rank_agreement,
                "completeness_error_mean_absolute": float(np.mean(np.abs(completeness))),
                "completeness_error_max_absolute": float(np.max(np.abs(completeness))),
                "aggregate_positive_map": ig_map.tolist(),
                "by_subject_positive_map": {
                    subject: values.tolist() for subject, values in ig_by_subject.items()
                },
            },
            "reliance": {
                "attribution_fit": fit_map.tolist(),
                "attribution_gate": gate_map.tolist(),
                "fit_by_subject": {
                    subject: values.tolist() for subject, values in fit_by_subject.items()
                },
                "gate_by_subject": {
                    subject: values.tolist() for subject, values in gate_by_subject.items()
                },
            },
            "fidelity": {
                "selected_cells": selected_count,
                "top_cell_indices": top_cells.tolist(),
                "top_mask_subject_ba_drop": float(drops[0]),
                "random_drop_percentile_95": random_threshold,
                "random_drop_mean": float(np.mean(drops[1:])),
                "passed": bool(drops[0] > random_threshold),
            },
        }
        print(
            f"[{role}] fidelity={drops[0]:.6f}>{random_threshold:.6f} "
            f"IG-rho={rank_agreement:.4f}",
            flush=True,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    before = role_arrays["before"]
    after = role_arrays["after"]
    joint = role_arrays["joint"]
    _before_mean, before_gate_subject = reliance_maps(
        before["truth"][gate], before["baseline"][gate], before["cells"][:, gate], before["subjects"][gate]
    )
    _after_mean, after_gate_subject = reliance_maps(
        after["truth"][gate], after["baseline"][gate], after["cells"][:, gate], after["subjects"][gate]
    )
    _joint_mean, joint_gate_subject = reliance_maps(
        joint["truth"][gate], joint["baseline"][gate], joint["cells"][:, gate], joint["subjects"][gate]
    )
    observed_ped, observed_by_subject = mean_subject_jsd(
        before_gate_subject, after_gate_subject
    )
    offline_alignment, offline_by_subject = mean_subject_jsd(
        after_gate_subject, joint_gate_subject
    )
    null_config = config["ped"]
    null_path = output / "same-checkpoint-null.npz"
    null_document = _load_npz(null_path, {"config_sha256": config_sha})
    if null_document is None:
        null_arrays = {}
        for offset, role in enumerate(("before", "after")):
            values = role_arrays[role]
            null_arrays[role] = same_checkpoint_split_null(
                values["truth"],
                values["baseline"],
                values["cells"],
                values["subjects"],
                replicates=int(null_config["null_replicates"]),
                seed=int(null_config["null_seed"]) + offset,
            )
        null_document = _save_bound_npz(
            null_path,
            {"before": null_arrays["before"], "after": null_arrays["after"]},
            {"config_sha256": config_sha},
        )
    null_arrays = {
        role: np.asarray(null_document[role], dtype=np.float64)
        for role in ("before", "after")
    }
    null_thresholds = {
        role: float(
            np.percentile(
                null_arrays[role], float(null_config["null_threshold_percentile"])
            )
        )
        for role in ("before", "after")
    }
    conservative_null = max(null_thresholds.values())
    required_roles = tuple(str(value) for value in config["gate"]["required_checkpoint_roles"])
    fidelity_pass = all(role_results[role]["fidelity"]["passed"] for role in required_roles)
    ig_threshold = ig_config.get("minimum_rank_agreement")
    ig_pass = (
        all(
            role_results[role]["integrated_gradients"]["rank_agreement_spearman"]
            >= float(ig_threshold)
            for role in required_roles
        )
        if ig_threshold is not None
        else None
    )
    ig_all_positive = all(
        role_results[role]["integrated_gradients"]["rank_agreement_spearman"] > 0
        for role in required_roles
    )
    drift_pass = observed_ped > conservative_null
    gate_config = config["gate"]
    hard_require_values = (
        gate_config["hard_require"]
        if "hard_require" in gate_config
        else gate_config["require_all"]
    )
    hard_require = tuple(str(value) for value in hard_require_values)
    hard_gate_values = {
        "fidelity": fidelity_pass,
        "integrated_gradients": bool(ig_pass),
        "drift_above_null": drift_pass,
    }
    if not set(hard_require).issubset(hard_gate_values):
        raise DatasetProtocolError("unknown hard XAI gate")
    result = {
        "schema_version": 1,
        "run": config["id"],
        "config_sha256": config_sha,
        "task": config["task"],
        "transition": config["transition"],
        "seed": run_seed,
        "cache": {
            "validation_index_sha256": sha256_file(validation_index),
            "untouched_test_index_sha256": sha256_file(test_index),
            "test_was_loaded": False,
        },
        "attribution_split": split,
        "cell_registry": cell_registry,
        "roles": role_results,
        "ped": {
            "before_after_mean_subject_jsd": observed_ped,
            "before_after_by_subject_jsd": observed_by_subject,
            "after_joint_mean_subject_jsd": offline_alignment,
            "after_joint_by_subject_jsd": offline_by_subject,
            "before_after_one_minus_spearman": _rank_sensitivity(
                before["gate_map"], after["gate_map"]
            ),
            "null": {
                "file": null_path.name,
                "sha256": sha256_file(null_path),
                "before_percentile_95": null_thresholds["before"],
                "after_percentile_95": null_thresholds["after"],
                "conservative_threshold": conservative_null,
            },
        },
        "gates": {
            "required_checkpoint_roles": list(required_roles),
            "fidelity_passed": fidelity_pass,
            "integrated_gradients_passed": ig_pass,
            "integrated_gradients_all_positive": ig_all_positive,
            "hard_require": list(hard_require),
            "drift_above_null_passed": drift_pass,
            "all_passed": all(hard_gate_values[name] for name in hard_require),
        },
        "device": str(device),
        "torch": torch.__version__,
    }
    _atomic_json(result_path, result)
    print(json.dumps({"gates": result["gates"], "ped": result["ped"]}, indent=2))


if __name__ == "__main__":
    main()
