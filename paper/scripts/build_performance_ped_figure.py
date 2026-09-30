#!/usr/bin/env python3
"""Fig. 3: performance/PED scatter and matched method-minus-Sequential PED.

Chart contract: two-column static Matplotlib scatter and box/strip plot;
63 order-seed-direction cells, 21 matched differences per method; no causal
or independent-observation inference. Orange/blue/olive identify methods,
marker shapes identify old tasks, and signed zero lines anchor comparisons.
Boxplots describe transition cells; diamonds are their unweighted means,
not the nine-block means used for inferential analysis in the manuscript.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
SOURCE = ROOT / "results/xai/high_gamma_replacement_ped_v1/summary.json"
COLORS = {"sequential_finetuning": "#D97706", "ewc": "#2563EB", "derpp": "#657A22"}
LABELS = {"sequential_finetuning": "Sequential FT", "ewc": "EWC", "derpp": "DER++"}
MARKERS = {"bciciv2a": "o", "high_gamma": "s"}


def main():
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert digest == "2005ef2c8b62207fed1215f896ce855a537ede128f39c12c2d1fe00b3027cc43"
    summary = json.loads(SOURCE.read_text())
    cells = summary["cells"]
    assert len(cells) == 63
    keyed = {(c["method"], c["order"], c["seed"], c["direction"]): c for c in cells}
    assert len(keyed) == 63
    differences, correlations = {}, {}
    for method in COLORS:
        rows = [c for c in cells if c["method"] == method]
        assert len(rows) == 21 and all(c["relative_valid"] for c in rows)
        rho = float(spearmanr([c["relative_forgetting"] for c in rows],
                              [c["mean_subject_ped"] for c in rows]).statistic)
        assert np.isclose(rho, summary["performance_ped_association"][method]["spearman_rho"])
        correlations[method] = rho
    for method in ("ewc", "derpp"):
        rows = []
        for row in summary["paired_method_comparison"][method]["rows"]:
            key = row["order"], row["seed"], row["direction"]
            delta = keyed[(method, *key)]["mean_subject_ped"] - keyed[("sequential_finetuning", *key)]["mean_subject_ped"]
            assert abs(delta - row["ped_delta_vs_sequential"]) < 1e-12
            rows.append(delta)
        assert len(rows) == 21 and np.isfinite(rows).all()
        differences[method] = np.asarray(rows)

    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix",
                         "font.size": 8, "font.weight": "bold", "axes.labelweight": "bold",
                         "axes.titleweight": "bold", "axes.labelsize": 8.5, "axes.titlesize": 9,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(7.16, 2.25), facecolor="white")
    ax = fig.add_axes([.07, .23, .33, .65])
    paired = fig.add_axes([.515, .23, .255, .65])
    for method in COLORS:
        for task, marker in MARKERS.items():
            rows = [c for c in cells if c["method"] == method and c["old_task"] == task]
            ax.scatter([c["relative_forgetting"] for c in rows],
                       [c["mean_subject_ped"] for c in rows], s=20, marker=marker,
                       color=COLORS[method], edgecolor="white", linewidth=.4, alpha=.95, zorder=3)
    ax.axvline(0, ls="--", lw=1.0, color="#555555")
    ax.axhline(0, lw=.65, color="#999999")
    ax.set(xlim=(-.39, .32), ylim=(-.012, .34),
           xlabel=r"Relative forgetting $\mathbf{F}_{\mathbf{rel}}$", ylabel="Noise-corrected PED")
    ax.set_xticks([-.3, -.2, -.1, 0, .1, .2, .3])
    ax.set_yticks([0, .1, .2, .3])
    ax.set_title("(a) Performance–PED relationship", loc="center", pad=5)
    rng = np.random.default_rng(20260818)
    for x, method in enumerate(("ewc", "derpp"), start=1):
        values = differences[method]
        paired.boxplot(values, positions=[x], widths=.43, showfliers=False,
                       patch_artist=True, boxprops={"facecolor": "none", "edgecolor": "#444444", "linewidth": .8},
                       medianprops={"color": "#444444", "linewidth": .9},
                       whiskerprops={"color": "#777777", "linewidth": .7},
                       capprops={"color": "#777777", "linewidth": .7})
        paired.scatter(x + rng.uniform(-.11, .11, len(values)), values, s=20,
                       color=COLORS[method], edgecolor="white", linewidth=.35, alpha=.9, zorder=3)
        paired.scatter([x], [values.mean()], marker="D", s=29, facecolor="#202020",
                       edgecolor="white", linewidth=.5, zorder=4)
    paired.axhline(0, color="#666666", lw=1.0, ls="--")
    paired.set(xlim=(.45, 2.55), ylim=(-.345, .022),
               ylabel="ΔPED vs. Sequential FT")
    paired.set_xticks([1, 2], ["EWC\n21/21 below 0", "DER++\n21/21 below 0"])
    paired.set_yticks([-.3, -.2, -.1, 0])
    paired.set_title("(b) Paired PED differences", loc="center", pad=5)
    for panel in (ax, paired):
        panel.set_axisbelow(True)
        panel.grid(axis="y", color="#E6E6E6", linewidth=.5)
        panel.spines[["top", "right"]].set_visible(False)
        for spine in panel.spines.values():
            spine.set_linewidth(.6)
            spine.set_color("#888888")
        panel.tick_params(length=3, width=.6)
    handles = [Line2D([], [], marker="o", linestyle="none", color=COLORS[m],
                      markersize=5, label=LABELS[m]) for m in COLORS]
    handles += [Line2D([], [], marker=MARKERS[t], linestyle="none", markerfacecolor="none",
                       markeredgecolor="#444444", markersize=5,
                       label="Old task: " + ("BCI" if t == "bciciv2a" else "High-Gamma")) for t in MARKERS]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(.79, .56),
               ncol=1, frameon=False, handletextpad=.45, labelspacing=1.0,
               borderaxespad=0, fontsize=8)
    stem = PAPER / "figure3_performance_ped"
    for ext in ("pdf", "svg", "png"):
        fig.savefig(stem.with_suffix("." + ext), dpi=600, facecolor="white")
    plt.close(fig)
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        fields = ["method", "order", "seed", "direction", "old_task", "relative_forgetting", "mean_subject_ped"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: c[k] for k in fields} for c in cells)
    stem.with_suffix(".sources.json").write_text(json.dumps({
        "source": str(SOURCE.relative_to(ROOT)), "sha256": digest,
        "n_cells": len(cells), "spearman_descriptive": correlations,
        "paired": {m: {"n": len(v), "below_zero": int(np.sum(v < 0)),
                        "cell_mean": float(v.mean()), "differences": v.tolist()} for m,v in differences.items()},
        "boxplot": "quartiles, median, and whiskers to furthest points within 1.5 IQR; all cells overlaid",
        "diamond": "unweighted mean of 21 paired cell differences; not nine-block inferential mean",
        "caption": "Performance and explanation retention across 63 stage-wise old-task transitions. "
                   "(a) Colors identify methods and marker shapes identify old tasks. "
                   "(b) Each point is a method-minus-Sequential PED difference matched on order, seed, and direction. "
                   "Negative values indicate less drift. Boxes show quartiles, lines show medians, and diamonds show cell means. "
                   "Both methods reduce PED in all 21 matched cells. Correlations and boxplots are descriptive; "
                   "inferential comparisons use nine order-seed blocks."}, indent=2) + "\n")
    path = PAPER / "figures-and-tables.md"
    section = """## Fig. 3 — Performance–PED Decoupling & Paired Reduction

![Fig. 3 — Performance–PED Decoupling & Paired Reduction](figure3_performance_ped.png)

[PDF](figure3_performance_ped.pdf) · [SVG](figure3_performance_ped.svg)

- **Chart (a) — Hiệu năng và cách đọc EEG:** Mỗi điểm ứng với một phương pháp–order–seed–bước chuyển task, bên trái đường dọc 0 là hiệu năng tăng, bên phải là giảm và càng cao thì explanation càng đổi nhiều; vì vậy các điểm cam phía trên bên trái cho thấy Sequential FT làm task cũ tốt hơn nhưng vẫn thay đổi cách sử dụng EEG, còn EWC xanh dương nằm gần đáy biểu thị drift thấp.
- **Chart (b) — Mức giảm drift so với Sequential:** Mỗi điểm là ΔPED = PED của EWC hoặc DER++ trừ PED của Sequential FT trong cùng điều kiện order–seed–hướng chuyển task, nên 0 là bằng nhau, giá trị âm là ít drift hơn và −0.10 nghĩa là PED thấp hơn 0.10 chứ không phải giảm 10%; cả hai phương pháp đều có 21/21 điểm dưới 0.
- **Hộp và hình thoi:** Hộp chứa 50% giá trị giữa, vạch trong hộp là median và hình thoi đen là trung bình 21 chênh lệch; EWC giảm drift nhiều hơn về trung bình nhưng hình chưa đánh giá cái giá về khả năng học task mới, còn các điểm có phụ thuộc nên suy luận thống kê của paper dùng 9 khối order–seed thay vì xem 21 điểm là độc lập.
"""
    text = path.read_text()
    if "## Fig. 3 —" in text:
        import re
        text = re.sub(r"## Fig\. 3 —.*?(?=\n## |\Z)", lambda _: section, text, flags=re.S)
    else:
        text = text.rstrip() + "\n\n" + section
    path.write_text(text)
    print("Verified 63 cells, 42 paired differences and 3 correlations; exported Fig. 3.")


if __name__ == "__main__":
    main()
