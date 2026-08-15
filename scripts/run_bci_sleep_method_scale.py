#!/usr/bin/env python3
"""Run one locked method/order/seed BCI<-Sleep explanation-drift scale cell."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    frozen_stratified_halves,
    mean_subject_jsd,
    predict_spectral_masks,
    reliance_maps,
    same_checkpoint_split_null,
    subject_balanced_accuracy_drop,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    parser.add_argument("--order", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "xai" / "bci_sleep_method_scale_v1.yaml",
    )
    return parser.parse_args()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _verify(path: Path, expected: str, role: str) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise DatasetProtocolError(f"{role} digest mismatch: {path}")


def _bands(config: dict[str, object]) -> tuple[tuple[str, float, float], ...]:
    return tuple(
        (str(value["name"]), float(value["low_hz"]), float(value["high_hz"]))
        for value in config["cell_registry"]["bands"]
    )


def _rows(dataset: CachedEEGDataset) -> tuple[list[int], list[str]]:
    labels, subjects = [], []
    for index in range(len(dataset)):
        _signal, label, subject = dataset[index]
        labels.append(int(label))
        subjects.append(subject)
    return labels, subjects


def _random_sets(cells: int, selected: int, count: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    values, seen = [], set()
    while len(values) < count:
        row = np.sort(generator.choice(cells, selected, replace=False))
        key = tuple(int(value) for value in row)
        if key not in seen:
            seen.add(key)
            values.append(row)
    return np.stack(values)


def _metadata(
    config_sha: str, method: str, order: str, seed: int, role: str, checkpoint_sha: str
) -> dict[str, str]:
    return {
        "config_sha256": config_sha,
        "method": method,
        "order": order,
        "seed": str(seed),
        "role": role,
        "checkpoint_sha256": checkpoint_sha,
    }


def _load_npz(path: Path, metadata: dict[str, str]) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as archive:
        document = {key: archive[key] for key in archive.files}
    for key, value in metadata.items():
        if key not in document or str(document[key].item()) != value:
            raise DatasetProtocolError(f"scale artifact identity mismatch: {path}")
    return document


def _save_npz(
    path: Path, arrays: dict[str, object], metadata: dict[str, str]
) -> dict[str, np.ndarray]:
    document = {
        **{key: np.asarray(value) for key, value in arrays.items()},
        **{key: np.asarray(value) for key, value in metadata.items()},
    }
    _atomic_npz(path, document)
    return document


def _load_stage_model(
    stage_path: Path,
    *,
    expected_config_sha: str,
    expected_order: tuple[str, ...],
    expected_stage: int,
    expected_task: str,
    device: torch.device,
) -> tuple[MultiHeadCBraMod, dict[str, object]]:
    with stage_path.open(encoding="utf-8") as handle:
        stage = json.load(handle)
    if (
        stage.get("config_sha256") != expected_config_sha
        or tuple(stage.get("order", ())) != expected_order
        or int(stage.get("stage", -1)) != expected_stage
        or stage.get("learned_task") != expected_task
    ):
        raise DatasetProtocolError(f"stage identity mismatch: {stage_path}")
    checkpoint_path = stage_path.parent / str(stage["checkpoint"]["file"])
    _verify(checkpoint_path, str(stage["checkpoint"]["sha256"]), "stage checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("config_sha256") != expected_config_sha:
        raise DatasetProtocolError("checkpoint parent config mismatch")
    model = MultiHeadCBraMod(CBraMod())
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    stage["stage_json_sha256"] = sha256_file(stage_path)
    stage["checkpoint_path"] = checkpoint_path.relative_to(PROJECT_ROOT).as_posix()
    return model.to(device), stage


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if config.get("status") != "locked_xai_scale":
        raise DatasetProtocolError("XAI scale config is not locked")
    if args.method not in config["methods"] or args.order not in config["orders"]:
        raise DatasetProtocolError("method/order is not declared by scale config")
    if args.seed not in tuple(int(value) for value in config["seeds"]):
        raise DatasetProtocolError("seed is not declared by scale config")
    if config["cache"].get("test_access_during_scale") != "forbidden":
        raise DatasetProtocolError("scale must not read test subjects")
    config_sha = sha256_file(args.config)
    method = config["methods"][args.method]
    order = config["orders"][args.order]
    _verify(_resolve(str(method["config"])), str(method["config_sha256"]), "method config")
    summary_path = _resolve(str(method["summary"]))
    _verify(summary_path, str(method["summary_sha256"]), "method summary")
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    result_root = _resolve(str(method["result_root"]))
    parent_result_path = result_root / args.order / f"seed-{args.seed}" / "result.json"
    result_key = f"{args.order}:seed-{args.seed}"
    if sha256_file(parent_result_path) != summary["input_result_sha256"].get(result_key):
        raise DatasetProtocolError("parent continual result is not bound by its summary")
    with parent_result_path.open(encoding="utf-8") as handle:
        parent_result = json.load(handle)
    expected_order = tuple(order["full_order"])
    if (
        parent_result.get("config_sha256") != method["config_sha256"]
        or tuple(parent_result.get("order", ())) != expected_order
        or int(parent_result.get("seed", -1)) != args.seed
    ):
        raise DatasetProtocolError("parent continual result identity mismatch")

    cache = config["cache"]
    validation_index = _resolve(str(cache["validation_index"]))
    test_index = _resolve(str(cache["test_index"]))
    _verify(validation_index, str(cache["validation_index_sha256"]), "validation cache")
    _verify(test_index, str(cache["test_index_sha256"]), "untouched test cache")
    dataset = CachedEEGDataset(validation_index)
    labels, subjects = _rows(dataset)
    if len(dataset) != int(cache["validation_samples"]) or sorted(set(subjects)) != sorted(
        str(value) for value in cache["validation_subjects"]
    ):
        raise DatasetProtocolError("validation cache population changed")
    split_config = config["attribution_split"]
    split = frozen_stratified_halves(labels, subjects, seed=int(split_config["seed"]))
    if split["assignment_sha256"] != split_config["assignment_sha256"]:
        raise DatasetProtocolError("frozen attribution assignment changed")
    fit = np.asarray(split["attribution_fit"], dtype=np.int64)
    gate = np.asarray(split["attribution_gate"], dtype=np.int64)

    output = _resolve(str(config["output"]["root"])) / args.method / args.order / f"seed-{args.seed}"
    result_path = output / "result.json"
    if result_path.exists():
        raise DatasetProtocolError(f"refusing to overwrite scale result {result_path}")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    set_determinism(args.seed)
    roles = {
        "before": (int(order["before_stage"]), str(order["before_learned_task"])),
        "after": (int(order["after_stage"]), str(order["after_learned_task"])),
    }
    stage_documents, models = {}, {}
    for role, (stage_number, learned_task) in roles.items():
        stage_path = (
            parent_result_path.parent
            / f"stage-{stage_number:02d}-{learned_task}"
            / "stage.json"
        )
        model, stage = _load_stage_model(
            stage_path,
            expected_config_sha=str(method["config_sha256"]),
            expected_order=expected_order,
            expected_stage=stage_number,
            expected_task=learned_task,
            device=device,
        )
        models[role] = model
        stage_documents[role] = stage
    if args.validate_only:
        print(
            json.dumps(
                {
                    "method": args.method,
                    "order": args.order,
                    "seed": args.seed,
                    "parent_result_sha256": sha256_file(parent_result_path),
                    "checkpoints": {
                        role: stage["checkpoint"]["sha256"]
                        for role, stage in stage_documents.items()
                    },
                    "split_sha256": split["assignment_sha256"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    bands = _bands(config)
    channels = tuple(config["cell_registry"]["channels"])
    cells = len(channels) * len(bands)
    individual_masks = torch.ones(cells + 1, len(channels), len(bands))
    for cell in range(cells):
        individual_masks[cell + 1].view(-1)[cell] = 0
    role_arrays, role_results = {}, {}
    for role, model in models.items():
        checkpoint_sha = str(stage_documents[role]["checkpoint"]["sha256"])
        metadata = _metadata(
            config_sha, args.method, args.order, args.seed, role, checkpoint_sha
        )
        prediction_path = output / f"{role}-cell-predictions.npz"
        data = _load_npz(prediction_path, metadata)
        if data is None:
            print(f"[{args.method}/{args.order}/{args.seed}/{role}] 110 cells", flush=True)
            predicted = predict_spectral_masks(
                model,
                str(config["task"]),
                dataset,
                range(len(dataset)),
                individual_masks,
                sampling_rate_hz=float(config["cell_registry"]["sampling_rate_hz"]),
                bands=bands,
                batch_size=int(config["evaluation"]["batch_size"]),
                mask_chunk_size=int(config["evaluation"]["mask_chunk_size"]),
                device=device,
            )
            data = _save_npz(prediction_path, predicted, metadata)
        truth = np.asarray(data["truth"], dtype=np.int64)
        subject_array = np.asarray(data["subjects"], dtype=np.str_)
        predictions = np.asarray(data["predictions"], dtype=np.int64)
        baseline, occluded = predictions[0], predictions[1:]
        fit_map, fit_subject = reliance_maps(
            truth[fit], baseline[fit], occluded[:, fit], subject_array[fit]
        )
        gate_map, gate_subject = reliance_maps(
            truth[gate], baseline[gate], occluded[:, gate], subject_array[gate]
        )
        baseline_ba, baseline_subject_ba = subject_balanced_accuracy(
            truth, baseline, subject_array
        )
        fidelity = config["fidelity"]
        selected_count = int(round(cells * float(fidelity["top_fraction"])))
        top = np.sort(np.argsort(fit_map)[-selected_count:])
        random = _random_sets(
            cells,
            selected_count,
            int(fidelity["random_masks"]),
            int(fidelity["random_seed"]),
        )
        selected_sets = np.concatenate((top[None], random))
        masks = torch.ones(len(selected_sets), len(channels), len(bands))
        for index, selected in enumerate(selected_sets):
            masks[index].view(-1)[selected] = 0
        fidelity_path = output / f"{role}-fidelity-predictions.npz"
        fidelity_data = _load_npz(fidelity_path, metadata)
        if fidelity_data is None:
            predicted = predict_spectral_masks(
                model,
                str(config["task"]),
                dataset,
                gate,
                masks,
                sampling_rate_hz=float(config["cell_registry"]["sampling_rate_hz"]),
                bands=bands,
                batch_size=int(config["evaluation"]["batch_size"]),
                mask_chunk_size=int(config["evaluation"]["mask_chunk_size"]),
                device=device,
            )
            predicted["selected_cells"] = selected_sets
            fidelity_data = _save_npz(fidelity_path, predicted, metadata)
        if not np.array_equal(fidelity_data["selected_cells"], selected_sets):
            raise DatasetProtocolError("fidelity mask identity changed")
        drops = np.asarray(
            [
                subject_balanced_accuracy_drop(
                    truth[gate], baseline[gate], row, subject_array[gate]
                )[0]
                for row in fidelity_data["predictions"]
            ]
        )
        threshold = float(np.percentile(drops[1:], float(fidelity["threshold_percentile"])))
        role_arrays[role] = {
            "truth": truth,
            "subjects": subject_array,
            "baseline": baseline,
            "occluded": occluded,
            "gate_map": gate_map,
            "gate_subject": gate_subject,
        }
        role_results[role] = {
            "stage": stage_documents[role]["stage"],
            "learned_task": stage_documents[role]["learned_task"],
            "stage_json_sha256": stage_documents[role]["stage_json_sha256"],
            "checkpoint": {
                "path": stage_documents[role]["checkpoint_path"],
                "sha256": checkpoint_sha,
            },
            "unoccluded_validation_subject_ba": baseline_ba,
            "unoccluded_validation_ba_by_subject": baseline_subject_ba,
            "reliance": {
                "fit": fit_map.tolist(),
                "gate": gate_map.tolist(),
                "fit_by_subject": {key: value.tolist() for key, value in fit_subject.items()},
                "gate_by_subject": {key: value.tolist() for key, value in gate_subject.items()},
            },
            "fidelity": {
                "top_cell_indices": top.tolist(),
                "top_mask_subject_ba_drop": float(drops[0]),
                "random_drop_percentile_95": threshold,
                "passed": bool(drops[0] > threshold),
            },
            "artifacts": {
                "cell_predictions": {"file": prediction_path.name, "sha256": sha256_file(prediction_path)},
                "fidelity_predictions": {"file": fidelity_path.name, "sha256": sha256_file(fidelity_path)},
            },
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    observed_ped, observed_subject = mean_subject_jsd(
        role_arrays["before"]["gate_subject"], role_arrays["after"]["gate_subject"]
    )
    null_config = config["ped"]
    null_values, null_thresholds = {}, {}
    for offset, role in enumerate(("before", "after")):
        values = role_arrays[role]
        null_values[role] = same_checkpoint_split_null(
            values["truth"],
            values["baseline"],
            values["occluded"],
            values["subjects"],
            replicates=int(null_config["null_replicates"]),
            seed=int(null_config["null_seed"]) + offset,
        )
        null_thresholds[role] = float(
            np.percentile(null_values[role], float(null_config["null_threshold_percentile"]))
        )
    null_path = output / "same-checkpoint-null.npz"
    _atomic_npz(
        null_path,
        {
            "before": null_values["before"],
            "after": null_values["after"],
            "config_sha256": np.asarray(config_sha),
        },
    )
    conservative_null = max(null_thresholds.values())
    rank_correlation = float(
        spearmanr(role_arrays["before"]["gate_map"], role_arrays["after"]["gate_map"]).statistic
    )
    forgetting = [
        row
        for row in parent_result["pairwise_forgetting"]
        if row["old_task"] == config["task"] and row["learned_task"] == config["learned_task"]
    ]
    if len(forgetting) != 1:
        raise DatasetProtocolError("parent result lacks unique BCI<-Sleep forgetting row")
    result = {
        "schema_version": 1,
        "run": config["id"],
        "config_sha256": config_sha,
        "method": args.method,
        "order": args.order,
        "seed": args.seed,
        "task": config["task"],
        "learned_task": config["learned_task"],
        "parent_result": {
            "path": parent_result_path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(parent_result_path),
        },
        "cache": {
            "validation_index_sha256": sha256_file(validation_index),
            "untouched_test_index_sha256": sha256_file(test_index),
            "test_was_loaded": False,
        },
        "attribution_split_sha256": split["assignment_sha256"],
        "roles": role_results,
        "performance_forgetting": forgetting[0],
        "ped": {
            "mean_subject_jsd": observed_ped,
            "by_subject_jsd": observed_subject,
            "one_minus_spearman": 1.0 - rank_correlation,
            "null_before_percentile_95": null_thresholds["before"],
            "null_after_percentile_95": null_thresholds["after"],
            "conservative_null_percentile_95": conservative_null,
            "above_null": bool(observed_ped > conservative_null),
            "null_artifact": {"file": null_path.name, "sha256": sha256_file(null_path)},
        },
        "device": str(device),
        "torch": torch.__version__,
    }
    _atomic_json(result_path, result)
    print(
        json.dumps(
            {
                "method": args.method,
                "order": args.order,
                "seed": args.seed,
                "relative_forgetting": forgetting[0]["relative_forgetting"],
                "ped": observed_ped,
                "ped_above_null": result["ped"]["above_null"],
                "fidelity": {
                    role: role_results[role]["fidelity"]["passed"]
                    for role in ("before", "after")
                },
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
