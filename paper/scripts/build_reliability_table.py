#!/usr/bin/env python3
"""Build Table I from verified gate artifacts, without running inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/tables"
SOURCES: dict[str, str] = {}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_result(run: str) -> tuple[dict, dict]:
    path = ROOT / "results/xai" / run / "result.json"
    result = json.loads(path.read_text())
    assert result["test_was_loaded"] is False
    SOURCES[str(path.relative_to(ROOT))] = digest(path)
    config = ROOT / "configs/xai" / f"{run}.yaml"
    assert digest(config) == result["config_sha256"]
    SOURCES[str(config.relative_to(ROOT))] = digest(config)
    artifacts = dict(result.get("artifacts", {}))
    for field in ("artifact", "fidelity_artifact"):
        if field in result:
            artifacts[field] = result[field]
    arrays = {}
    for name, entry in artifacts.items():
        source = path.parent / entry["file"]
        assert digest(source) == entry["sha256"], source
        SOURCES[str(source.relative_to(ROOT))] = entry["sha256"]
        with np.load(source, allow_pickle=False) as archive:
            arrays[name] = dict(archive)
    return result, arrays


def class_subject_mean(values, truth, subjects):
    """Independent equal-class, equal-subject aggregation of stored scores."""
    return {
        str(subject): np.mean([
            values[:, (subjects == subject) & (truth == label)].mean(axis=1)
            for label in np.unique(truth[subjects == subject])
        ], axis=0)
        for subject in np.unique(subjects)
    }


def bci_reliability(scores):
    # Reproduce frozen_stratified_halves(seed=20260815), retaining global row IDs.
    assert np.array_equal(scores["indices"], np.arange(len(scores["truth"])))
    truth, subjects = scores["truth"], scores["subjects"]
    halves = [[], []]
    for subject in np.unique(subjects):
        for label in np.unique(truth[subjects == subject]):
            rows = np.flatnonzero((subjects == subject) & (truth == label))
            ranked = sorted(rows, key=lambda row: hashlib.sha256(
                f"20260815|{subject}|{label}|{row}".encode()).hexdigest())
            mid = len(ranked) // 2
            halves[0].extend(ranked[:mid])
            halves[1].extend(ranked[mid:])
    maps = []
    margins = scores["margins"].astype(np.float64)
    for rows in halves:
        maps.append(class_subject_mean(
            margins[0, rows] - margins[1:111, rows], truth[rows], subjects[rows]))
    return {s: float(maps[0][s] @ maps[1][s] /
                     (np.linalg.norm(maps[0][s]) * np.linalg.norm(maps[1][s])))
            for s in maps[0]}


def verify_fidelity(scores, saved, baseline=None, selected=0):
    if baseline is None:
        base_margin, base_pred = scores["margins"][0], scores["predictions"][0]
        margins, predictions = scores["margins"][1:], scores["predictions"][1:]
    else:
        lookup = {int(row): i for i, row in enumerate(baseline["indices"])}
        positions = [lookup[int(row)] for row in scores["indices"]]
        base_margin = baseline["margins"][0, positions]
        base_pred = baseline["predictions"][0, positions]
        margins, predictions = scores["margins"], scores["predictions"]
    truth, subjects = scores["truth"], scores["subjects"]
    margin = np.mean(list(class_subject_mean(
        base_margin.astype(float) - margins.astype(float), truth, subjects).values()), axis=0)
    ba = np.mean(list(class_subject_mean(
        (base_pred == truth).astype(float) - (predictions == truth).astype(float),
        truth, subjects).values()), axis=0)
    controls = np.arange(len(margin)) != selected
    for metric, values in (("margin", margin), ("ba", ba)):
        top = float(values[selected])
        threshold = float(np.percentile(values[controls], 95))
        assert np.isclose(top, saved[f"top_{metric}_drop"], atol=1e-10, rtol=0)
        assert np.isclose(threshold, saved[f"control_{metric}_drop_percentile_95"], atol=1e-10, rtol=0)
        assert bool(top > threshold) == saved[f"{metric}_passed"]


def main():
    import itertools

    _, bci = load_result("bci_margin_mask_reliability_v2")
    hg, hg_arrays = load_result("high_gamma_margin_reliability_v1")
    sleep, sleep_arrays = load_result("sleep_sample_size_amendment_v3")
    assert np.all(bci["artifact"]["random_mask_indicators"].sum(axis=1) == 22)
    assert np.all(hg_arrays["fidelity_artifact"]["random_mask_indicators"].sum(axis=1) == 22)
    verify_fidelity(sleep_arrays["bci_fidelity"], sleep["bci_fidelity"])
    verify_fidelity(hg_arrays["fidelity_artifact"], hg["fidelity"])
    selected = list(itertools.combinations(range(10), 3)).index(
        tuple(sleep["sleep_fidelity"]["top_cell_indices"]))
    verify_fidelity(sleep_arrays["sleep_fidelity"], sleep["sleep_fidelity"],
                    baseline=sleep_arrays["sleep_individual"], selected=selected)
    inputs = [
        ("BCI IV-2a", bci_reliability(bci["artifact"]), sleep["bci_fidelity"]),
        ("High-Gamma", hg["subject_split_half_cosine"], hg["fidelity"]),
        ("Sleep-EDF", sleep["sample_size_curve"]["200"]["subject_split_half_cosine"], sleep["sleep_fidelity"]),
    ]
    rows = []
    lines = []
    for name, cosines, fidelity in inputs:
        values = np.array(list(cosines.values()))
        median = float(np.median(values))
        count = int(np.sum(values >= .5))
        reliable = bool(np.all(np.isfinite(values)) and median >= .7 and count / len(values) >= 2/3)
        eligible = reliable and fidelity["passed"]
        rows.append(dict(dataset=name, subject_cosines=cosines, median_cosine=median,
                         subjects_above_threshold=count, subjects=len(values),
                         reliability_passed=reliable, fidelity=fidelity, ped_eligible=eligible))
        # Display measured effects and control thresholds, not opaque gate labels.
        # BA is an absolute drop in percentage points, never a relative percent.
        margin = (f"{fidelity['top_margin_drop']:.3f} "
                  f"({fidelity['control_margin_drop_percentile_95']:.3f})")
        ba = (f"{100 * fidelity['top_ba_drop']:.2f} "
              f"({100 * fidelity['control_ba_drop_percentile_95']:.2f})")
        ped = "Included" if eligible else "Excluded"
        lines.append(f"{name} & {median:.3f} & {count}/{len(values)} & "
                     f"{margin} & {ba} & {ped} " + r"\\")
    table = r"""% Generated by paper/scripts/build_reliability_table.py; requires booktabs.
\begin{table}[t]
\centering
\caption{Quantitative validation of channel--frequency reliance.}
\label{tab:reliability}
\begingroup
\fontsize{8}{9.5}\selectfont
\setlength{\tabcolsep}{2.5pt}
\renewcommand{\arraystretch}{1.18}
\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lccccc@{}}
\toprule
 & \multicolumn{2}{c}{Reliability} & \multicolumn{2}{c}{Faithfulness} & PED \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Dataset & Median $c_s$ & $n/N$ & $\Delta m$ & $\Delta\mathrm{BA}$ (pp) & analysis \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular*}
\par\vspace{3pt}
\begin{minipage}{\columnwidth}
\fontsize{7}{8.3}\selectfont
Reliability: median subject split-half cosine; $n/N$ counts subjects
with cosine $\geq0.50$. Eligibility requires finite cosines, median
$\geq0.70$, and $n/N\geq2/3$.
Faithfulness: top-mask margin drop and subject-balanced accuracy drop
(percentage points); parentheses give the control 95th percentile.
\emph{Both drops must exceed their control thresholds.}
Masks: top 22/110 cells with 200/100 random controls for BCI/High-Gamma;
top 3/10 with all 119 alternatives for Sleep.
All results use seed-42 checkpoints, class-balanced aggregation and
held-out validation splits; Sleep uses cap 200 rows/subject/class.
Exclusion applies only to old-task PED, not training.
\end{minipage}
\endgroup
\end{table}
"""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "table1_reliability.tex").write_text(table)
    (OUT / "table1_reliability.sources.json").write_text(json.dumps(
        dict(rows=rows, source_sha256=SOURCES, fidelity_recomputed_from_scores=True,
             bci_reliability_recomputed_class_balanced=True, sleep_reliability_cap=200,
             display_units={"margin_drop": "logit margin", "ba_drop": "percentage points"},
             parenthesized_values="95th percentile of control-mask drops"),
        indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
