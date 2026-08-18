#!/usr/bin/env python3
"""Build paper figures, Table 1, and clustered statistical report."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "results" / "xai" / "high_gamma_replacement_ped_v1" / "summary.json"
PAPER = ROOT / "paper"
FIGURES = PAPER / "figures"
TABLES = PAPER / "tables"
SEEDS = (3407, 42, 2026)
BANDS = ("$\\delta$", "$\\theta$", "$\\alpha$", "$\\beta$", "$\\gamma$")
CHANNELS = (
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
    "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
)
METHOD_LABELS = {
    "sequential_finetuning": "Sequential FT",
    "ewc": "EWC",
    "derpp": "DER++",
}
METHOD_COLORS = {
    "sequential_finetuning": "#D97706",
    "ewc": "#2563EB",
    "derpp": "#657A22",
}
TASK_MARKERS = {"bciciv2a": "o", "high_gamma": "s"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Regenerate only figures; leave Table 1 and statistics unchanged.",
    )
    return parser.parse_args()


def _positive_distribution(values: np.ndarray) -> np.ndarray:
    positive = np.maximum(np.asarray(values, dtype=float), 0.0)
    if positive.sum() <= 0:
        raise ValueError("map has no positive mass")
    return positive / positive.sum()


def _save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(
            FIGURES / f"{stem}.{suffix}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 6.6,
            "legend.fontsize": 7.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#334155",
            "axes.labelcolor": "#0F172A",
            "text.color": "#0F172A",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def build_figure_1() -> None:
    before, after = [], []
    for seed in SEEDS:
        base = (
            ROOT
            / "results"
            / "xai"
            / "high_gamma_replacement_ped_v1"
            / "sequential_finetuning"
            / "forward"
            / f"seed-{seed}"
            / "maps"
        )
        with np.load(base / "stage-01-bciciv2a.npz", allow_pickle=False) as data:
            split_mean = 0.5 * (data["fit_group_map"] + data["gate_group_map"])
            before.append(_positive_distribution(split_mean).reshape(22, 5))
        with np.load(base / "stage-02-bciciv2a.npz", allow_pickle=False) as data:
            split_mean = 0.5 * (data["fit_group_map"] + data["gate_group_map"])
            after.append(_positive_distribution(split_mean).reshape(22, 5))
    before_mean = np.mean(before, axis=0)
    after_mean = np.mean(after, axis=0)
    delta = after_mean - before_mean

    fig = plt.figure(figsize=(7.15, 4.6), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(0.75, 3.0))
    protocol = fig.add_subplot(grid[0, :])
    protocol.axis("off")
    box_pad = 0.015
    box_gap = 0.026
    box_labels = (
        (0.134, "Paired old-task\ncheckpoints"),
        (0.176, "Same held-out\nsubjects / samples"),
        (0.190, "110 channel–band\nmargin occlusions"),
        (0.134, "Fit / gate\nreliance maps"),
        (0.111, "Cross JSD −\nwithin noise"),
    )
    span = sum(width + 2 * box_pad for width, _ in box_labels) + box_gap * (
        len(box_labels) - 1
    )
    cursor = (1.0 - span) / 2.0 + box_pad
    boxes = []
    for width, label in box_labels:
        boxes.append((cursor, width, label))
        cursor += width + 2 * box_pad + box_gap
    box_y, box_height = 0.28, 0.43
    box_middle = box_y + box_height / 2.0
    for x, width, label in boxes:
        protocol.add_patch(
            FancyBboxPatch(
                (x, box_y),
                width,
                box_height,
                boxstyle="round,pad=0.015",
                transform=protocol.transAxes,
                facecolor="#F8FAFC",
                edgecolor="#64748B",
                linewidth=1.0,
            )
        )
        protocol.text(
            x + width / 2,
            box_middle,
            label,
            ha="center",
            va="center",
            transform=protocol.transAxes,
            fontsize=7.2,
        )
    for left, right in zip(boxes[:-1], boxes[1:], strict=True):
        protocol.annotate(
            "",
            xy=(right[0] - box_pad, box_middle),
            xytext=(left[0] + left[1] + box_pad, box_middle),
            xycoords=protocol.transAxes,
            arrowprops={
                "arrowstyle": "-|>",
                "lw": 1.0,
                "color": "#334155",
                "mutation_scale": 9,
                "shrinkA": 0,
                "shrinkB": 0,
            },
        )
    protocol.text(0.5, 0.96, "(a) Reliability-aware PED protocol", ha="center", va="top", weight="bold")

    reliance_max = max(float(before_mean.max()), float(after_mean.max()))
    diff_max = float(np.abs(delta).max())
    diff_cmap = LinearSegmentedColormap.from_list(
        "orange_white_blue", ("#D97706", "#FFFFFF", "#2563EB")
    )
    panels = (
        (before_mean, "(b) Before High-Gamma", "Blues", 0.0, reliance_max),
        (after_mean, "(c) After High-Gamma", "Blues", 0.0, reliance_max),
        (delta, "(d) After − before", diff_cmap, -diff_max, diff_max),
    )
    for index, (values, title, cmap, vmin, vmax) in enumerate(panels):
        ax = fig.add_subplot(grid[1, index])
        image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, weight="bold", pad=5)
        ax.set_xticks(range(5), BANDS)
        if index == 0:
            ax.set_yticks(range(22), CHANNELS)
            ax.set_ylabel("BCI IV-2a channel")
        else:
            ax.set_yticks(range(22), [])
        ax.tick_params(length=2)
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
        cbar.ax.tick_params(labelsize=6)
        cbar.set_label("Reliance mass" if index < 2 else "Mass change", fontsize=7)
    fig.suptitle(
        "Channel–frequency reliance changes despite improved BCI accuracy",
        fontsize=10,
        weight="bold",
    )
    _save_figure(fig, "figure1_protocol_maps")


def _performance_scatter(ax: plt.Axes, summary: dict[str, object]) -> list[plt.Line2D]:
    cells = summary["cells"]
    for method in METHOD_LABELS:
        for old_task in TASK_MARKERS:
            rows = [
                row
                for row in cells
                if row["method"] == method
                and row["old_task"] == old_task
                and row["relative_forgetting"] is not None
            ]
            ax.scatter(
                [row["relative_forgetting"] for row in rows],
                [row["mean_subject_ped"] for row in rows],
                s=27,
                marker=TASK_MARKERS[old_task],
                facecolor=METHOD_COLORS[method],
                edgecolor="white",
                linewidth=0.45,
                alpha=0.86,
            )
    ax.axvline(0, color="#64748B", lw=0.8, ls="--")
    ax.axhline(0, color="#CBD5E1", lw=0.8)
    ax.set_xlabel("Relative forgetting $F_{rel}$")
    ax.set_ylabel("Noise-corrected PED")
    ax.set_xlim(-0.38, 0.32)
    ax.set_ylim(-0.01, 0.335)
    ax.grid(True, color="#E2E8F0", linewidth=0.5, zorder=0)
    association = summary["performance_ped_association"]
    rho_text = "   |   ".join(
        f"{METHOD_LABELS[m]} $\\rho$={association[m]['spearman_rho']:.2f}"
        for m in METHOD_LABELS
    )
    ax.set_title("Performance and explanation drift", weight="bold", pad=27)
    ax.text(
        0.5,
        1.015,
        rho_text,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.4,
    )
    return [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=METHOD_COLORS[method],
            markeredgecolor="white",
            markersize=6,
            label=METHOD_LABELS[method],
        )
        for method in METHOD_LABELS
    ] + [
        plt.Line2D(
            [0],
            [0],
            marker=TASK_MARKERS[task],
            color="#475569",
            linestyle="none",
            markersize=5,
            markerfacecolor="none",
            markeredgecolor="#475569",
            label="BCI old task" if task == "bciciv2a" else "High-Gamma old task",
        )
        for task in TASK_MARKERS
    ]


def build_figure_2a(summary: dict[str, object]) -> None:
    fig = plt.figure(figsize=(5.5, 4.15))
    grid = fig.add_gridspec(2, 1, height_ratios=(1.0, 0.10), hspace=0.58)
    ax = fig.add_subplot(grid[0])
    handles = _performance_scatter(ax, summary)
    legend_ax = fig.add_subplot(grid[1])
    legend_ax.axis("off")
    legend_ax.legend(
        handles=handles,
        loc="center",
        frameon=False,
        ncol=5,
        columnspacing=0.9,
        handletextpad=0.35,
        fontsize=6.3,
    )
    fig.subplots_adjust(left=0.13, right=0.98, top=0.88, bottom=0.03)
    _save_figure(fig, "figure2a_performance_ped")


def build_figure_2b(summary: dict[str, object]) -> None:
    fig, ax = plt.subplots(figsize=(3.8, 3.45))
    paired = summary["paired_method_comparison"]
    rng = np.random.default_rng(20260818)
    for x, method in enumerate(("ewc", "derpp"), start=1):
        values = np.asarray(
            [row["ped_delta_vs_sequential"] for row in paired[method]["rows"]],
            dtype=float,
        )
        jitter = rng.uniform(-0.11, 0.11, size=len(values))
        ax.scatter(
            x + jitter,
            values,
            s=21,
            color=METHOD_COLORS[method],
            alpha=0.72,
            edgecolor="white",
            linewidth=0.35,
        )
        ax.boxplot(
            values,
            positions=[x],
            widths=0.42,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "none", "edgecolor": "#334155", "linewidth": 0.9},
            medianprops={"color": "#0F172A", "linewidth": 1.2},
            whiskerprops={"color": "#64748B"},
            capprops={"color": "#64748B"},
        )
        ax.scatter([x], [values.mean()], marker="D", s=34, color="#0F172A", zorder=4)
    ax.axhline(0, color="#64748B", lw=0.9, ls="--")
    ax.set_xticks((1, 2), ("EWC", "DER++"))
    ax.set_ylabel("PED difference vs. Sequential FT")
    ax.set_ylim(-0.325, 0.005)
    ax.set_title("Paired explanation-retention gain", weight="bold", pad=27)
    ax.grid(True, axis="y", color="#E2E8F0", linewidth=0.5)
    ax.text(
        0.5,
        1.015,
        "21 matched cells per method; all differences are below zero",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.7,
    )
    fig.subplots_adjust(left=0.22, right=0.98, top=0.84, bottom=0.14)
    _save_figure(fig, "figure2b_paired_ped")


def _bootstrap_ci(values: np.ndarray, *, replicates: int = 20000) -> tuple[float, float]:
    rng = np.random.default_rng(20260818)
    draws = rng.choice(values, size=(replicates, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _clustered_statistics(summary: dict[str, object]) -> dict[str, object]:
    report = {"cluster_unit": "order_seed", "bootstrap_replicates": 20000, "methods": {}}
    paired = summary["paired_method_comparison"]
    for method in ("ewc", "derpp"):
        grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
        for row in paired[method]["rows"]:
            grouped[(row["order"], int(row["seed"]))].append(
                float(row["ped_delta_vs_sequential"])
            )
        run_means = np.asarray([np.mean(values) for values in grouped.values()])
        ci_low, ci_high = _bootstrap_ci(run_means)
        negatives = int(np.sum(run_means < 0))
        sign_p = min(1.0, 2.0 * sum(math.comb(len(run_means), k) for k in range(0, len(run_means) - negatives + 1)) / (2 ** len(run_means)))
        report["methods"][method] = {
            "n_order_seed_blocks": len(run_means),
            "mean_block_ped_delta": float(run_means.mean()),
            "median_block_ped_delta": float(np.median(run_means)),
            "bootstrap_95_ci": [ci_low, ci_high],
            "negative_blocks": negatives,
            "two_sided_sign_test_p": float(sign_p),
            "all_21_transition_cells_reduced": bool(
                all(float(row["ped_delta_vs_sequential"]) < 0 for row in paired[method]["rows"])
            ),
        }
    report["associations"] = summary["performance_ped_association"]
    report["association_note"] = (
        "Spearman values are descriptive because transition cells share order, seed, and subjects."
    )
    return report


def _fmt(value: dict[str, float], *, adaptive: bool = False) -> str:
    decimals = 4 if adaptive and abs(float(value["mean"])) < 0.001 else 3
    return (
        f"{value['mean']:.{decimals}f} $\\pm$ "
        f"{value['sample_sd']:.{decimals}f}"
    )


def build_table_1(summary: dict[str, object]) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    directions = (
        ("bciciv2a<-high_gamma", "BCI$\\leftarrow$High-$\\gamma$"),
        ("bciciv2a<-sleep_edf_sc", "BCI$\\leftarrow$Sleep"),
        ("high_gamma<-bciciv2a", "High-$\\gamma$$\\leftarrow$BCI"),
        ("high_gamma<-sleep_edf_sc", "High-$\\gamma$$\\leftarrow$Sleep"),
    )
    methods = ("sequential_finetuning", "ewc", "derpp")
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Performance forgetting and subject-level noise-corrected PED (mean $\\pm$ SD across matched order--seed transition cells). Lower is better; negative $F_{rel}$ denotes backward transfer.}",
        "\\label{tab:main}",
        "\\setlength{\\tabcolsep}{3.2pt}",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "& \\multicolumn{2}{c}{Sequential FT} & \\multicolumn{2}{c}{EWC} & \\multicolumn{2}{c}{DER++} " + r"\\",
        "Direction & $F_{rel}$ & PED & $F_{rel}$ & PED & $F_{rel}$ & PED " + r"\\",
        "\\midrule",
    ]
    for direction, label in directions:
        values = []
        for method in methods:
            row = summary["aggregates"][method][direction]
            values.extend(
                (
                    _fmt(row["relative_forgetting"]),
                    _fmt(row["mean_subject_ped"], adaptive=True),
                )
            )
        lines.append(label + " & " + " & ".join(values) + " " + r"\\")
    lines.extend(("\\bottomrule", "\\end{tabular}", "\\end{table*}", ""))
    (TABLES / "table1_main.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    _style()
    with SUMMARY.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    build_figure_1()
    build_figure_2a(summary)
    build_figure_2b(summary)
    if args.figures_only:
        return
    build_table_1(summary)
    statistics = _clustered_statistics(summary)
    (PAPER / "statistical_report.json").write_text(
        json.dumps(statistics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(statistics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
