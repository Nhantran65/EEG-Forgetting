#!/usr/bin/env python3
"""Render Fig. 2 from official, digest-bound BCI reliance maps."""
from __future__ import annotations

import csv
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/xai/high_gamma_replacement_ped_v1"
OUT = ROOT / "paper/figures/figure2_reliance_shift"
SEEDS = (3407, 42, 2026)
CHANNELS = ("Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1",
            "Cz", "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4",
            "P1", "Pz", "P2", "POz")
BANDS = ("Delta", "Theta", "Alpha", "Beta", "Gamma")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg"), default=["png"],
                        help="PNG preview by default; export PDF only after layout approval.")
    args = parser.parse_args()
    summary_path = SOURCE / "summary.json"
    assert sha(summary_path) == "2005ef2c8b62207fed1215f896ce855a537ede128f39c12c2d1fe00b3027cc43"
    summary = json.loads(summary_path.read_text())
    hashes = {str(summary_path.relative_to(ROOT)): sha(summary_path)}
    maps = [[], []]
    for seed in SEEDS:
        run = SOURCE / "sequential_finetuning/forward" / f"seed-{seed}"
        result_path = run / "result.json"
        expected = summary["input_result_sha256"][f"sequential_finetuning:forward:seed-{seed}"]
        assert sha(result_path) == expected
        hashes[str(result_path.relative_to(ROOT))] = expected
        result = json.loads(result_path.read_text())
        assert result["test_was_loaded_for_xai"] is False
        for index, stage in enumerate(("stage-01-bciciv2a", "stage-02-bciciv2a")):
            entry = result["artifacts"][stage]
            path = run / entry["file"]
            assert sha(path) == entry["sha256"]
            hashes[str(path.relative_to(ROOT))] = entry["sha256"]
            with np.load(path, allow_pickle=False) as data:
                assert set(data["subject_ids"]) == {"A06", "A07"}
                assert str(data["checkpoint_sha256"]) == entry["checkpoint_sha256"]
                assert str(data["config_sha256"]) == result["config_sha256"]
                values = .5 * (data["fit_group_map"] + data["gate_group_map"])
            assert values.shape == (110,) and np.isfinite(values).all()
            positive = np.maximum(values, 0)
            assert positive.sum() > 0
            maps[index].append((positive / positive.sum()).reshape(22, 5))
    before, after = [100 * np.mean(values, axis=0) for values in maps]
    delta = after - before
    assert np.isclose(before.sum(), 100) and np.isclose(after.sum(), 100)
    assert np.isclose(delta.sum(), 0, atol=1e-10)
    cells = [c for c in summary["cells"] if c["method"] == "sequential_finetuning"
             and c["order"] == "forward" and c["old_task"] == "bciciv2a"
             and c["learned_task"] == "high_gamma"]
    assert {c["seed"] for c in cells} == set(SEEDS)
    f_rel = float(np.mean([c["relative_forgetting"] for c in cells]))
    ped = float(np.mean([c["mean_subject_ped"] for c in cells]))

    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix",
                         "font.size": 8, "font.weight": "bold", "axes.labelweight": "bold",
                         "axes.titleweight": "bold", "axes.labelsize": 8, "axes.titlesize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 7.5,
                         "text.color": "#202020", "axes.labelcolor": "#202020",
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none"})
    diverging = LinearSegmentedColormap.from_list("shift", ["#B65C13", "#FFFFFF", "#18568A"])
    vmax = float(np.ceil(max(before.max(), after.max())))
    dmax = float(np.ceil(np.abs(delta).max()))
    fig = plt.figure(figsize=(2.65, 3.05), facecolor="white")
    ax = fig.add_axes([.18, .20, .79, .73])
    ax.set_title("After − Before", loc="center", pad=5)
    image = ax.imshow(delta, interpolation="nearest", aspect="auto",
                      cmap=diverging, vmin=-dmax, vmax=dmax)
    ax.set_xticks(range(5), ["δ", "θ", "α", "β", "γ"])
    ax.set_yticks(range(22), CHANNELS)
    ax.tick_params(axis="both", length=0, pad=1.8)
    ax.set_xticks(np.arange(-.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 22, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=.35)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(image, cax=fig.add_axes([.18, .105, .79, .023]), orientation="horizontal")
    cb.set_label("Change (percentage points)", labelpad=1.8)
    ticks = [-dmax, -dmax/2, 0, dmax/2, dmax]
    cb.set_ticks(ticks, labels=[f"{v:+g}" if v else "0" for v in ticks])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, width=.5, labelsize=7.5, pad=1.5)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "figure2_reliance_shift"
    for suffix in args.formats:
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=600, facecolor="white")
    plt.close(fig)
    with stem.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["channel", "band", "before_percent", "after_percent", "change_pp"])
        for i, channel in enumerate(CHANNELS):
            for j, band in enumerate(BANDS):
                writer.writerow([channel, band, before[i,j], after[i,j], delta[i,j]])
    metadata = {"source_sha256": hashes, "seeds": SEEDS, "validation_subjects": ["A06", "A07"],
                "aggregation": "mean_fit_gate_then_positive_normalize_per_seed_then_mean_seeds",
                "before_after_units": "percent of positive reliance mass", "delta_units": "percentage points",
                "mean_relative_forgetting": f_rel, "mean_subject_ped": ped,
                "difference_scale": [-dmax, dmax], "displayed_panels": ["after_minus_before"],
                "exported_formats": args.formats, "figure_size_inches": [2.65, 3.05],
                "map_totals": [float(before.sum()), float(after.sum())], "delta_total": float(delta.sum())}
    stem.with_suffix(".sources.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output": str(stem), "relative_forgetting": f_rel, "ped": ped,
                      "shared_max": vmax, "delta_max": dmax}, indent=2))


if __name__ == "__main__":
    main()
