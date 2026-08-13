#!/usr/bin/env python3
"""Extract one exact per-example empirical-Fisher instrument signature."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from eeg_forgetting.data.cache import CachedEEGDataset
from eeg_forgetting.data.contracts import DatasetProtocolError, load_yaml
from eeg_forgetting.data.manifests import sha256_file
from eeg_forgetting.models.cbramod import CBraModTaskModel, load_pretrained_backbone
from eeg_forgetting.training.fisher import (
    combine_fisher,
    deterministic_sample_indices,
    diagonal_empirical_fisher,
    fisher_cosine,
    indices_sha256,
    layer_l2_normalize,
    sampled_inventory,
    save_fisher,
)
from eeg_forgetting.training.pilot import TASK_CLASSES, TASK_SHAPES, set_determinism


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = tuple(TASK_CLASSES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "pilots" / "fisher_instrument_v1.yaml",
    )
    parser.add_argument(
        "--cache-root", type=Path, default=PROJECT_ROOT / "data" / "processed" / "v3"
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "single_task" / "budget_v1",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "fisher" / "instrument_v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if config["status"] != "instrument_pilot" or args.dataset not in config["datasets"]:
        raise DatasetProtocolError("dataset is not declared by the Fisher pilot")
    seed = int(config["model_seed"])
    output_dir = args.output_root / args.dataset / f"seed-{seed}"
    paths = {
        name: output_dir / name
        for name in ("full.pt", "split_a.pt", "split_b.pt", "result.json")
    }
    if any(path.exists() for path in paths.values()):
        raise DatasetProtocolError(f"refusing to overwrite Fisher output {output_dir}")
    set_determinism(seed)
    dataset = CachedEEGDataset(args.cache_root / args.dataset / "train" / "index.json")
    estimator = config["estimator"]
    indices = deterministic_sample_indices(
        len(dataset),
        sample_count=int(estimator["sample_count"]),
        seed=int(estimator["sampling_seed"]),
    )
    half = int(estimator["split_half_samples"])
    if len(indices) != 2 * half:
        raise DatasetProtocolError("instrument sample must consist of two equal split halves")

    model_config = load_yaml(PROJECT_ROOT / "configs" / "models" / "cbramod.yaml")
    pretrained = model_config["checkpoint"]
    backbone = load_pretrained_backbone(
        PROJECT_ROOT / str(pretrained["local_path"]),
        expected_sha256=str(pretrained["sha256"]),
        map_location="cpu",
    )
    channels, patches = TASK_SHAPES[args.dataset]
    model = CBraModTaskModel(
        backbone,
        TASK_CLASSES[args.dataset],
        head="flatten_mlp",
        channels=channels,
        patches=patches,
    )
    checkpoint_dir = args.checkpoint_root / args.dataset / f"seed-{seed}"
    checkpoint_path = checkpoint_dir / "final.pt"
    with (checkpoint_dir / "result.json").open(encoding="utf-8") as handle:
        checkpoint_result = json.load(handle)
    observed_checkpoint_sha = sha256_file(checkpoint_path)
    if checkpoint_result["final_step"] != 2500 or observed_checkpoint_sha != checkpoint_result[
        "final_checkpoint_sha256"
    ]:
        raise DatasetProtocolError("Fisher source is not the verified exact-step-2500 checkpoint")
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))

    def extract(name: str, selected) -> dict[str, torch.Tensor]:
        last_reported = 0

        def report(completed: int, total: int) -> None:
            nonlocal last_reported
            if completed == total or completed - last_reported >= 64:
                print(
                    f"{args.dataset} {name}: Fisher examples {completed}/{total}",
                    flush=True,
                )
                last_reported = completed

        return diagonal_empirical_fisher(
            model,
            dataset,
            selected,
            final_blocks=int(config["parameters"]["final_encoder_blocks"]),
            microbatch_size=int(estimator["gradient_microbatch_size"]),
            device=args.device,
            progress=report,
        )

    split_a = extract("split_a", indices[:half])
    split_b = extract("split_b", indices[half:])
    full = combine_fisher(split_a, split_b, left_samples=half, right_samples=half)
    within_task_cosine = fisher_cosine(
        layer_l2_normalize(split_a), layer_l2_normalize(split_b)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_fisher(paths["split_a.pt"], split_a)
    save_fisher(paths["split_b.pt"], split_b)
    save_fisher(paths["full.pt"], full)
    result = {
        "schema_version": 1,
        "run": config["id"],
        "dataset": args.dataset,
        "model_seed": seed,
        "config_sha256": sha256_file(args.config),
        "cache_index_sha256": sha256_file(dataset.index_path),
        "source_checkpoint_sha256": observed_checkpoint_sha,
        "samples": len(indices),
        "sample_indices_sha256": indices_sha256(indices),
        "sample_inventory": sampled_inventory(dataset, indices),
        "split_half_cosine": within_task_cosine,
        "signatures": {
            name: {"file": path.name, "sha256": sha256_file(path)}
            for name, path in paths.items()
            if name.endswith(".pt")
        },
        "torch": torch.__version__,
        "device": str(args.device),
    }
    temporary = paths["result.json"].with_suffix(".json.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(paths["result.json"])
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
