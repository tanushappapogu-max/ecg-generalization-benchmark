#!/usr/bin/env python3
"""Generate publication-ready ECG benchmark figures from completed Drive exports.

The script intentionally refuses to invent missing architectures. It discovers the
available 25-cell matrices in source_data/, validates them, and records coverage in
data_completeness.csv before producing figures.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "source_data"
OUT = ROOT / "output"
OUT.mkdir(parents=True, exist_ok=True)

# Exact colors extracted from the methods-pipeline PowerPoint.
PALETTE = {
    "navy": "#053351",
    "slate": "#4D6076",
    "slate_light": "#88919E",
    "mauve": "#9688A4",
    "purple": "#664E82",
    "deep_purple": "#341760",
    "rose": "#BB7E8C",
    "rose_light": "#D6B5C3",
    "dusty_rose": "#A08794",
    "paper": "#FFFFFF",
    "grid": "#C6C6C6",
    "wash": "#F3F2F4",
}

ARCH_ORDER = ["inception_time", "resnet1d", "transformer", "ecg_fm"]
ARCH_LABEL = {
    "inception_time": "InceptionTime",
    "resnet1d": "ResNet1D",
    "transformer": "Transformer",
    "ecg_fm": "ECG-FM",
}
ARCH_COLOR = {
    "inception_time": PALETTE["navy"],
    "resnet1d": PALETTE["slate"],
    # Rose is intentionally used for Transformer in analysis figures. The
    # pipeline palette contains several purples that collapse visually at
    # print size; this preserves the theme while making models distinguishable.
    "transformer": PALETTE["rose"],
    "ecg_fm": PALETTE["deep_purple"],
}
ARCH_MARKER = {"inception_time": "o", "resnet1d": "D", "transformer": "^", "ecg_fm": "s"}
ARCH_LINESTYLE = {"inception_time": "-", "resnet1d": ":", "transformer": "--", "ecg_fm": "-."}
DATASET_ORDER = ["ptbxl", "cpsc2018", "georgia", "mimic_iv", "code_ii"]
DATASET_LABEL = {
    "ptbxl": "PTB-XL",
    "cpsc2018": "CPSC2018",
    "georgia": "Georgia",
    "mimic_iv": "MIMIC-IV-ECG",
    "code_ii": "CODE-15%",
}
CLASS_COLUMNS = ["auroc_NSR", "auroc_AFIB_AFL", "auroc_IAVB", "auroc_LBBB", "auroc_RBBB"]
CLASS_LABEL = {
    "auroc_NSR": "NSR",
    "auroc_AFIB_AFL": "AF/AFL",
    "auroc_IAVB": "1° AVB",
    "auroc_LBBB": "LBBB",
    "auroc_RBBB": "RBBB",
}

mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "figure.titlesize": 12,
        "axes.edgecolor": PALETTE["slate"],
        "axes.labelcolor": PALETTE["navy"],
        "text.color": PALETTE["navy"],
        "xtick.color": PALETTE["slate"],
        "ytick.color": PALETTE["slate"],
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.55,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
)
sns.set_style("white", {"axes.edgecolor": PALETTE["slate"], "grid.color": PALETTE["grid"]})

AUROC_CMAP = LinearSegmentedColormap.from_list(
    "pipeline_auroc",
    [PALETTE["wash"], PALETTE["rose_light"], PALETTE["mauve"], PALETTE["purple"], PALETTE["deep_purple"]],
)
GAP_CMAP = LinearSegmentedColormap.from_list(
    "pipeline_gap",
    [PALETTE["navy"], PALETTE["paper"], PALETTE["rose"], PALETTE["deep_purple"]],
)


def sem(values: pd.Series | np.ndarray) -> float:
    x = pd.Series(values).dropna().astype(float)
    return float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float("nan")


def save_figure(fig: plt.Figure, stem: str) -> None:
    for ext in ("pdf", "svg", "png"):
        dpi = 350 if ext == "png" else None
        fig.savefig(OUT / f"{stem}.{ext}", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.10, 1.07, label, transform=ax.transAxes, fontsize=12, fontweight="bold", color=PALETTE["deep_purple"])


def clean_axes(ax: plt.Axes, *, grid_axis: str | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PALETTE["slate_light"])
    ax.spines["bottom"].set_color(PALETTE["slate_light"])
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=PALETTE["grid"], linewidth=0.6, alpha=0.65)
    else:
        ax.grid(False)


def model_title(ax: plt.Axes, architecture: str) -> None:
    ax.set_title(ARCH_LABEL[architecture], color=ARCH_COLOR[architecture], fontweight="bold", pad=9)
    ax.plot([0, 1], [1.02, 1.02], transform=ax.transAxes, color=ARCH_COLOR[architecture], linewidth=2.4, clip_on=False)


def load_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    ci_frames: list[pd.DataFrame] = []
    coverage: list[dict[str, object]] = []
    for architecture in ARCH_ORDER:
        matrix_path = DATA / f"{architecture}_five_label.csv"
        ci_path = DATA / f"{architecture}_bootstrap_ci.csv"
        if not matrix_path.exists():
            coverage.append(
                {
                    "architecture": architecture,
                    "five_label_cells": 0,
                    "expected_cells": 25,
                    "matrix_complete": False,
                    "bootstrap_ci_available": False,
                }
            )
            continue
        df = pd.read_csv(matrix_path).dropna(how="all")
        df = df[df["status"].eq("COMPLETE")].copy()
        observed = len(df)
        unique_cells = df[["source_dataset", "target_dataset"]].drop_duplicates().shape[0]
        if observed != unique_cells:
            raise ValueError(f"Duplicate source-target cells in {matrix_path}: {observed} rows, {unique_cells} unique")
        if set(df["source_dataset"]) - set(DATASET_ORDER) or set(df["target_dataset"]) - set(DATASET_ORDER):
            raise ValueError(f"Unexpected dataset name in {matrix_path}")
        coverage.append(
            {
                "architecture": architecture,
                "five_label_cells": observed,
                "expected_cells": 25,
                "matrix_complete": observed == 25,
                "bootstrap_ci_available": ci_path.exists(),
            }
        )
        if observed != 25:
            raise ValueError(f"{architecture} has {observed}/25 completed cells; refusing partial paper figure")
        frames.append(df)
        if ci_path.exists():
            ci_frames.append(pd.read_csv(ci_path).dropna(how="all"))

    coverage_df = pd.DataFrame(coverage)
    coverage_df.to_csv(OUT / "data_completeness.csv", index=False)
    if not frames:
        raise RuntimeError("No completed 25-cell matrices were found")

    results = pd.concat(frames, ignore_index=True)
    results["is_diagonal"] = results["source_dataset"].eq(results["target_dataset"])
    diagonals = (
        results.loc[results["is_diagonal"], ["architecture", "source_dataset", "macro_auroc"]]
        .rename(columns={"source_dataset": "source_dataset", "macro_auroc": "in_distribution_auroc"})
    )
    results = results.merge(diagonals, on=["architecture", "source_dataset"], how="left", validate="many_to_one")
    results["gap"] = results["in_distribution_auroc"] - results["macro_auroc"]
    ci = pd.concat(ci_frames, ignore_index=True) if ci_frames else pd.DataFrame()
    return results, ci, coverage_df


def load_shift_vectors() -> pd.DataFrame:
    path = ROOT.parent / "results" / "shift_metadata" / "shift_vectors.csv"
    shift = pd.read_csv(path)
    normalize = {
        "PTB-XL": "ptbxl",
        "MIMIC-IV-ECG": "mimic_iv",
        "CPSC2018": "cpsc2018",
        "Georgia 12-Lead": "georgia",
        "CODE-15%": "code_ii",
    }
    shift["source_dataset"] = shift["source_dataset"].map(normalize)
    shift["target_dataset"] = shift["target_dataset"].map(normalize)
    shift["shift_vector"] = shift.apply(lambda r: f"({int(r.PS)},{int(r.DS)},{int(r.LS)})", axis=1)
    return shift[["source_dataset", "target_dataset", "PS", "DS", "LS", "shift_vector", "weight_w_ij"]]


def make_summary_statistics(results: pd.DataFrame, shift: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for architecture, group in results.groupby("architecture"):
        for scope, values in (
            ("in_distribution_auroc", group.loc[group["is_diagonal"], "macro_auroc"]),
            ("cross_dataset_auroc", group.loc[~group["is_diagonal"], "macro_auroc"]),
            ("cross_dataset_gap", group.loc[~group["is_diagonal"], "gap"]),
        ):
            rows.append(
                {
                    "architecture": architecture,
                    "metric_scope": scope,
                    "n_cells": len(values),
                    "mean": values.mean(),
                    "standard_deviation": values.std(ddof=1),
                    "standard_error": sem(values),
                    "mean_minus_1.96_sem": values.mean() - 1.96 * sem(values),
                    "mean_plus_1.96_sem": values.mean() + 1.96 * sem(values),
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "summary_statistics_sd_sem.csv", index=False)

    off = results.loc[~results["is_diagonal"]].merge(
        shift, on=["source_dataset", "target_dataset"], how="left", validate="many_to_one"
    )
    if off[["PS", "DS", "LS"]].isna().any().any():
        raise ValueError("At least one result cell could not be matched to a shift vector")
    off.to_csv(OUT / "off_diagonal_cells_with_shift_vectors.csv", index=False)
    return summary, off


def figure_heatmaps(results: pd.DataFrame) -> None:
    available = [a for a in ARCH_ORDER if a in set(results["architecture"])]
    fig, axes = plt.subplots(1, len(available), figsize=(3.5 * len(available), 3.5), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for idx, (ax, architecture) in enumerate(zip(axes, available)):
        matrix = (
            results.loc[results["architecture"].eq(architecture)]
            .pivot(index="source_dataset", columns="target_dataset", values="macro_auroc")
            .reindex(index=DATASET_ORDER, columns=DATASET_ORDER)
        )
        color = ARCH_COLOR[architecture]
        for i in range(5):
            for j in range(5):
                val = matrix.iat[i, j]
                if i == j:
                    ax.add_patch(Rectangle((j, i), 0.90, 0.82, facecolor=color, edgecolor=color, linewidth=1.5))
                    ax.text(j + 0.45, i + 0.34, f"{val:.3f}", ha="center", va="center", color="white", fontsize=8.2, fontweight="bold")
                    ax.text(j + 0.45, i + 0.63, "ID", ha="center", va="center", color="white", fontsize=6.5, fontweight="bold", alpha=0.9)
                else:
                    ax.add_patch(Rectangle((j, i), 0.90, 0.82, facecolor=PALETTE["wash"], edgecolor="white", linewidth=1.2))
                    width = np.clip((val - 0.65) / 0.35, 0, 1) * 0.74
                    ax.add_patch(Rectangle((j + 0.08, i + 0.63), width, 0.08, facecolor=color, edgecolor="none", alpha=0.95))
                    ax.text(j + 0.45, i + 0.30, f"{val:.3f}", ha="center", va="center", color=PALETTE["navy"], fontsize=8.0)
        ax.set_xlim(-0.05, 4.95)
        ax.set_ylim(4.95, -0.08)
        ax.set_aspect("equal")
        ax.set_xticks(np.arange(5) + 0.45, [DATASET_LABEL[d] for d in DATASET_ORDER], rotation=34, ha="right")
        ax.set_yticks(np.arange(5) + 0.41, [DATASET_LABEL[d] for d in DATASET_ORDER])
        ax.set_xlabel("Evaluation target")
        if idx == 0:
            ax.set_ylabel("Training source")
        model_title(ax, architecture)
        clean_axes(ax, grid_axis=None)
        for spine in ax.spines.values():
            spine.set_visible(False)
        panel_label(ax, chr(ord("A") + idx))
    fig.suptitle("Five-class source-to-target AUROC", fontweight="bold", color=PALETTE["navy"])
    fig.text(0.5, -0.015, "Off-diagonal bar length uses the same 0.65–1.00 AUROC scale in every panel; filled cells are in-distribution.", ha="center", fontsize=8, color=PALETTE["slate"])
    save_figure(fig, "fig1_five_class_auroc_heatmaps")


def figure_generalization_summary(results: pd.DataFrame) -> None:
    available = [a for a in ARCH_ORDER if a in set(results["architecture"])]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.15), gridspec_kw={"width_ratios": [1.1, 0.9]}, constrained_layout=True)

    ax = axes[0]
    x = np.array([0, 1])
    for architecture in available:
        g = results[results["architecture"].eq(architecture)]
        vals = [g.loc[g["is_diagonal"], "macro_auroc"], g.loc[~g["is_diagonal"], "macro_auroc"]]
        means = [v.mean() for v in vals]
        errors = [sem(v) for v in vals]
        ax.errorbar(
            x,
            means,
            yerr=errors,
            marker=ARCH_MARKER[architecture],
            linestyle=ARCH_LINESTYLE[architecture],
            markersize=6.5,
            linewidth=2.0,
            capsize=3,
            color=ARCH_COLOR[architecture],
        )
        ax.text(1.06, means[1], f"{ARCH_LABEL[architecture]}  {means[1]:.3f}", ha="left", va="center", fontsize=8.2, color=ARCH_COLOR[architecture], fontweight="bold")
        ax.text(-0.05, means[0] + 0.004, f"{means[0]:.3f}", ha="right", va="bottom", fontsize=7.5, color=ARCH_COLOR[architecture])
    ax.set_xticks(x, ["Same-dataset\ntest", "Cross-dataset\ntest"])
    ax.set_ylabel("Macro-AUROC (mean ± SEM across cells)")
    ax.set_ylim(0.84, 1.0)
    ax.set_xlim(-0.15, 1.53)
    ax.set_title("Same-dataset performance does not predict transfer", fontweight="bold")
    clean_axes(ax, grid_axis="y")
    panel_label(ax, "A")

    ax = axes[1]
    for i, architecture in enumerate(available):
        values = results.loc[(results["architecture"].eq(architecture)) & (~results["is_diagonal"]), "gap"]
        mean, err = values.mean(), sem(values)
        ax.hlines(i, 0, mean, color=ARCH_COLOR[architecture], linewidth=3.0, alpha=0.82)
        ax.errorbar(mean, i, xerr=err, fmt=ARCH_MARKER[architecture], markersize=7, capsize=3, color=ARCH_COLOR[architecture], markerfacecolor="white", markeredgewidth=1.7)
        ax.text(mean + 0.006, i, f"{mean:.3f}", va="center", fontsize=8.2, color=ARCH_COLOR[architecture], fontweight="bold")
    ax.set_yticks(range(len(available)), [ARCH_LABEL[a] for a in available])
    for tick, architecture in zip(ax.get_yticklabels(), available):
        tick.set_color(ARCH_COLOR[architecture])
        tick.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 0.115)
    ax.set_xlabel("Mean gap ± SEM")
    ax.set_title("Average directional generalization gap", fontweight="bold")
    clean_axes(ax, grid_axis="x")
    panel_label(ax, "B")
    save_figure(fig, "fig2_generalization_summary")


def figure_target_difficulty(results: pd.DataFrame) -> None:
    available = [a for a in ARCH_ORDER if a in set(results["architecture"])]
    off = results.loc[~results["is_diagonal"]]
    fig, axes = plt.subplots(1, len(available), figsize=(3.2 * len(available), 3.15), sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    y = np.arange(len(DATASET_ORDER))
    for idx, (ax, architecture) in enumerate(zip(axes, available)):
        g = off[off["architecture"].eq(architecture)]
        means, errors = [], []
        for target in DATASET_ORDER:
            vals = g.loc[g["target_dataset"].eq(target), "macro_auroc"]
            means.append(vals.mean())
            errors.append(sem(vals))
        ax.axhspan(2.65, 3.35, color=PALETTE["rose_light"], alpha=0.22, linewidth=0)
        ax.errorbar(means, y, xerr=errors, fmt=ARCH_MARKER[architecture], markersize=6.5, capsize=3, color=ARCH_COLOR[architecture], markerfacecolor="white", markeredgewidth=1.6)
        for xx, yy in zip(means, y):
            ax.text(xx + 0.007, yy, f"{xx:.3f}", va="center", fontsize=7.5, color=ARCH_COLOR[architecture])
        ax.set_yticks(y, [DATASET_LABEL[d] for d in DATASET_ORDER])
        ax.invert_yaxis()
        ax.set_xlim(0.70, 0.97)
        ax.set_xlabel("External AUROC\n(mean ± SEM)")
        model_title(ax, architecture)
        clean_axes(ax, grid_axis="x")
        panel_label(ax, chr(ord("A") + idx))
    axes[0].set_ylabel("Evaluation target")
    fig.suptitle("MIMIC-IV-ECG is the hardest external target", fontweight="bold", color=PALETTE["navy"])
    save_figure(fig, "fig3_target_dataset_difficulty")


def figure_per_class(results: pd.DataFrame) -> None:
    available = [a for a in ARCH_ORDER if a in set(results["architecture"])]
    off = results.loc[~results["is_diagonal"]]
    long = off.melt(
        id_vars=["architecture", "source_dataset", "target_dataset"],
        value_vars=CLASS_COLUMNS,
        var_name="class",
        value_name="auroc",
    )
    stats = long.groupby(["architecture", "class"])["auroc"].agg(["mean", "std", "count"]).reset_index()
    stats["sem"] = stats["std"] / np.sqrt(stats["count"])
    stats.to_csv(OUT / "per_class_cross_dataset_sd_sem.csv", index=False)

    fig, axes = plt.subplots(1, len(available), figsize=(3.2 * len(available), 3.0), sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    y = np.arange(len(CLASS_COLUMNS))
    for idx, (ax, architecture) in enumerate(zip(axes, available)):
        g = stats[stats["architecture"].eq(architecture)].set_index("class").reindex(CLASS_COLUMNS)
        ax.errorbar(g["mean"], y, xerr=g["sem"], fmt=ARCH_MARKER[architecture], markersize=6.5, capsize=3, color=ARCH_COLOR[architecture], markerfacecolor="white", markeredgewidth=1.6)
        for xx, yy in zip(g["mean"], y):
            ax.text(xx + 0.008, yy, f"{xx:.3f}", va="center", fontsize=7.5, color=ARCH_COLOR[architecture])
        ax.set_yticks(y, [CLASS_LABEL[c] for c in CLASS_COLUMNS])
        ax.invert_yaxis()
        ax.set_xlim(0.68, 1.01)
        ax.set_xlabel("Cross-dataset AUROC\n(mean ± SEM)")
        model_title(ax, architecture)
        clean_axes(ax, grid_axis="x")
        panel_label(ax, chr(ord("A") + idx))
    axes[0].set_ylabel("Diagnostic class")
    fig.suptitle("Cross-dataset performance differs by diagnostic class", fontweight="bold", color=PALETTE["navy"])
    save_figure(fig, "fig4_per_class_cross_dataset_auroc")


def figure_gap_heatmap(results: pd.DataFrame) -> None:
    off = results.loc[~results["is_diagonal"]]
    mean_gap = off.groupby(["source_dataset", "target_dataset"])["gap"].mean().unstack().reindex(index=DATASET_ORDER, columns=DATASET_ORDER)
    matrix = mean_gap.values
    bounds = [-0.03, 0.0, 0.03, 0.07, 0.12, 0.17]
    colors = [PALETTE["slate_light"], PALETTE["wash"], PALETTE["rose_light"], "#CE98A6", PALETTE["rose"]]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)
    fig, ax = plt.subplots(figsize=(4.5, 3.8), constrained_layout=True)
    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="equal")
    for i in range(5):
        for j in range(5):
            if i == j or np.isnan(matrix[i, j]):
                ax.add_patch(Rectangle((j - 0.49, i - 0.49), 0.98, 0.98, facecolor=PALETTE["wash"], edgecolor="white", hatch="///", linewidth=0.4))
                continue
            val = matrix[i, j]
            ax.text(j, i, f"{val:+.3f}", ha="center", va="center", fontsize=8.2, fontweight="bold" if val >= 0.12 else "normal", color="white" if val < 0 or val >= 0.12 else PALETTE["navy"])
    ax.set_xticks(range(5), [DATASET_LABEL[d] for d in DATASET_ORDER], rotation=35, ha="right")
    ax.set_yticks(range(5), [DATASET_LABEL[d] for d in DATASET_ORDER])
    ax.set_xlabel("Evaluation target")
    ax.set_ylabel("Training source")
    ax.set_title("Mean directional generalization gap across completed models", fontweight="bold")
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04, boundaries=bounds, ticks=bounds)
    cbar.set_label("Gap (source diagonal − target AUROC)")
    save_figure(fig, "fig5_mean_directional_gap_heatmap")


def figure_shift_analysis(off: pd.DataFrame) -> None:
    vector_order = ["(0,0,1)", "(0,1,1)", "(1,1,1)"]
    vector_label = {
        "(0,0,1)": "LS only",
        "(0,1,1)": "DS + LS",
        "(1,1,1)": "PS + DS + LS",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), constrained_layout=True)
    ax = axes[0]
    rng = np.random.default_rng(2026)
    for i, vector in enumerate(vector_order):
        vals = off.loc[off["shift_vector"].eq(vector), "gap"]
        mean, err = vals.mean(), sem(vals)
        ax.bar(i, mean, color=[PALETTE["rose_light"], PALETTE["mauve"], PALETTE["deep_purple"]][i], width=0.62, edgecolor="white")
        ax.errorbar(i, mean, yerr=err, color=PALETTE["navy"], capsize=3, linewidth=1)
        jitter = rng.uniform(-0.18, 0.18, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=9, facecolor="white", edgecolor=PALETTE["slate"], linewidth=0.55, alpha=0.75)
        ax.text(i, mean + err + 0.012, f"{mean:.3f}", ha="center", fontsize=7)
    ax.axhline(0, color=PALETTE["slate"], linewidth=0.8)
    ax.set_xticks(range(3), [vector_label[v] for v in vector_order], rotation=15, ha="right")
    ax.set_ylabel("Gap (mean ± SEM across cells)")
    ax.set_title("Observed gaps by shift-vector group", fontweight="bold")
    panel_label(ax, "A")

    ax = axes[1]
    allocation_rows = []
    for architecture, group in off.groupby("architecture"):
        totals = {"PS": 0.0, "DS": 0.0, "LS": 0.0}
        weights = {"PS": 1.0, "DS": 2.0, "LS": 3.0}
        for _, row in group.iterrows():
            active = [s for s in ("PS", "DS", "LS") if int(row[s]) == 1]
            denom = sum(weights[s] for s in active)
            for shift_type in active:
                totals[shift_type] += float(row["gap"]) * weights[shift_type] / denom
        total = sum(totals.values())
        for shift_type in totals:
            allocation_rows.append({"architecture": architecture, "shift_type": shift_type, "allocated_gap_share": totals[shift_type] / total if total else np.nan})
    allocation = pd.DataFrame(allocation_rows)
    allocation.to_csv(OUT / "descriptive_shift_allocation.csv", index=False)
    bottoms = np.zeros(allocation["architecture"].nunique())
    available = [a for a in ARCH_ORDER if a in set(allocation["architecture"])]
    shift_colors = {"PS": PALETTE["rose"], "DS": PALETTE["rose_light"], "LS": PALETTE["dusty_rose"]}
    for shift_type in ("PS", "DS", "LS"):
        vals = [allocation.loc[(allocation["architecture"].eq(a)) & (allocation["shift_type"].eq(shift_type)), "allocated_gap_share"].iloc[0] for a in available]
        ax.bar(range(len(available)), vals, bottom=bottoms, color=shift_colors[shift_type], label=shift_type, width=0.65, edgecolor="white")
        bottoms += np.array(vals)
    ax.set_xticks(range(len(available)), [ARCH_LABEL[a] for a in available], rotation=17, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Allocated share of total gap")
    ax.set_title("Proposal-weighted attribution (descriptive, not causal)", fontweight="bold")
    ax.legend(frameon=False, ncol=3, loc="upper center")
    panel_label(ax, "B")
    save_figure(fig, "fig6_shift_vector_analysis")


def figure_binary_ablation() -> None:
    path = DATA / "two_class_gap_summary.csv"
    if not path.exists():
        return
    df = pd.read_csv(path).dropna(how="all")
    fig, ax = plt.subplots(figsize=(3.7, 3.0), constrained_layout=True)
    for _, row in df.iterrows():
        y = [row["five_label_generalization_gap"], row["binary_generalization_gap"]]
        color = ARCH_COLOR.get(row["architecture"], PALETTE["slate"])
        ax.plot([0, 1], y, marker="o", markersize=7, linewidth=2, color=color)
        ax.text(0, y[0] + 0.0022, f"{y[0]:.3f}", ha="center", fontsize=8, color=color)
        ax.text(1, y[1] + 0.0022, f"{y[1]:.3f}", ha="center", fontsize=8, color=color)
        ax.text(0.5, np.mean(y) - 0.001, f"−{row['percent_gap_disappearance']:.1f}%", ha="center", va="center", fontsize=9, fontweight="bold", color=PALETTE["rose"])
    ax.set_xticks([0, 1], ["Five-class task", "Normal vs abnormal"])
    ax.set_ylabel("Mean cross-dataset generalization gap")
    ax.set_ylim(0, max(df["five_label_generalization_gap"].max() * 1.35, 0.06))
    ax.set_title("Effect of label harmonization on generalization", fontweight="bold")
    if df["architecture"].nunique() < 4:
        ax.text(
            0.5,
            -0.22,
            f"{df['architecture'].nunique()} of 4 architectures currently available.",
            transform=ax.transAxes,
            ha="center",
            fontsize=6.8,
            color=PALETTE["slate"],
        )
    save_figure(fig, "fig7_two_class_ablation")


def figure_bootstrap_width(ci: pd.DataFrame) -> None:
    if ci.empty:
        return
    macro = ci[ci["metric"].eq("macro_auroc")].copy()
    macro["ci_width"] = macro["ci_95_upper"] - macro["ci_95_lower"]
    stats = macro.groupby(["architecture", "target_dataset"])["ci_width"].agg(["mean", "std", "count"]).reset_index()
    stats["sem"] = stats["std"] / np.sqrt(stats["count"])
    stats.to_csv(OUT / "bootstrap_ci_width_by_target.csv", index=False)
    available = [a for a in ARCH_ORDER if a in set(macro["architecture"])]
    fig, ax = plt.subplots(figsize=(6.9, 3.15), constrained_layout=True)
    x = np.arange(len(DATASET_ORDER))
    for architecture in available:
        g = stats[stats["architecture"].eq(architecture)].set_index("target_dataset").reindex(DATASET_ORDER)
        offset = (available.index(architecture) - (len(available) - 1) / 2) * 0.12
        ax.errorbar(x + offset, g["mean"], yerr=g["sem"], marker="o", markersize=5, linewidth=1.4, capsize=3, color=ARCH_COLOR[architecture], label=ARCH_LABEL[architecture])
    ax.set_xticks(x, [DATASET_LABEL[d] for d in DATASET_ORDER], rotation=20, ha="right")
    ax.set_ylabel("Bootstrap 95% CI width (mean ± SEM across sources)")
    ax.set_title("Uncertainty varies by target dataset", fontweight="bold")
    ax.legend(frameon=False, ncol=len(available), loc="upper left")
    save_figure(fig, "fig8_bootstrap_ci_widths")


def figure_directional_asymmetry(results: pd.DataFrame) -> None:
    off = results.loc[~results["is_diagonal"]]
    pairs = []
    for i, a in enumerate(DATASET_ORDER):
        for b in DATASET_ORDER[i + 1 :]:
            diffs = []
            for architecture in sorted(off["architecture"].unique()):
                ab = off.loc[(off["architecture"].eq(architecture)) & (off["source_dataset"].eq(a)) & (off["target_dataset"].eq(b)), "macro_auroc"]
                ba = off.loc[(off["architecture"].eq(architecture)) & (off["source_dataset"].eq(b)) & (off["target_dataset"].eq(a)), "macro_auroc"]
                if len(ab) == 1 and len(ba) == 1:
                    diffs.append(float(ab.iloc[0] - ba.iloc[0]))
            pairs.append({"pair": f"{DATASET_LABEL[a]} → {DATASET_LABEL[b]}", "mean_difference": np.mean(diffs), "sd": np.std(diffs, ddof=1), "sem": np.std(diffs, ddof=1) / np.sqrt(len(diffs)), "n_architectures": len(diffs)})
    df = pd.DataFrame(pairs).sort_values("mean_difference")
    df.to_csv(OUT / "directional_asymmetry_sd_sem.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.3, 4.1), constrained_layout=True)
    y = np.arange(len(df))
    colors = [PALETTE["rose"] if v < 0 else PALETTE["navy"] for v in df["mean_difference"]]
    ax.barh(y, df["mean_difference"], xerr=df["sem"], color=colors, alpha=0.9, capsize=2.5, edgecolor="white")
    ax.axvline(0, color=PALETTE["slate"], linewidth=0.9)
    ax.set_yticks(y, df["pair"])
    ax.set_xlabel("AUROC difference: A→B minus B→A (mean ± SEM across architectures)")
    ax.set_title("Cross-dataset transfer is directional", fontweight="bold")
    save_figure(fig, "fig9_directional_asymmetry")


def figure_model_tradeoff(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for architecture, g in results.groupby("architecture"):
        diag = g.loc[g["is_diagonal"], "macro_auroc"]
        cross = g.loc[~g["is_diagonal"], "macro_auroc"]
        gaps = g.loc[~g["is_diagonal"], "gap"]
        rows.append(
            {
                "architecture": architecture,
                "in_distribution_mean": diag.mean(),
                "in_distribution_sd": diag.std(ddof=1),
                "in_distribution_sem": sem(diag),
                "cross_dataset_mean": cross.mean(),
                "cross_dataset_sd": cross.std(ddof=1),
                "cross_dataset_sem": sem(cross),
                "gap_mean": gaps.mean(),
                "gap_sd": gaps.std(ddof=1),
                "gap_sem": sem(gaps),
            }
        )
    summary = pd.DataFrame(rows).sort_values("cross_dataset_mean", ascending=False)
    summary.to_csv(OUT / "model_summary_table.csv", index=False)

    fig, ax = plt.subplots(figsize=(4.4, 3.4), constrained_layout=True)
    for _, row in summary.iterrows():
        architecture = row["architecture"]
        ax.errorbar(
            row["in_distribution_mean"],
            row["cross_dataset_mean"],
            xerr=row["in_distribution_sem"],
            yerr=row["cross_dataset_sem"],
            fmt="o",
            markersize=8,
            capsize=3,
            color=ARCH_COLOR[architecture],
        )
        ax.annotate(ARCH_LABEL[architecture], (row["in_distribution_mean"], row["cross_dataset_mean"]), xytext=(5, 5), textcoords="offset points", fontsize=8, color=ARCH_COLOR[architecture], fontweight="bold")
    lo, hi = 0.86, 0.99
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=0.9, color=PALETTE["grid"])
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Mean in-distribution macro-AUROC")
    ax.set_ylabel("Mean cross-dataset macro-AUROC")
    ax.set_title("High in-distribution AUROC does not guarantee transfer", fontweight="bold")
    save_figure(fig, "fig10_model_tradeoff")
    return summary


def render_model_table(summary: pd.DataFrame) -> None:
    display = summary.copy()
    display["Model"] = display["architecture"].map(ARCH_LABEL)
    display["In-distribution"] = display.apply(lambda r: f"{r.in_distribution_mean:.3f} ± {r.in_distribution_sd:.3f}", axis=1)
    display["Cross-dataset"] = display.apply(lambda r: f"{r.cross_dataset_mean:.3f} ± {r.cross_dataset_sd:.3f}", axis=1)
    display["Gap"] = display.apply(lambda r: f"{r.gap_mean:.3f} ± {r.gap_sd:.3f}", axis=1)
    display = display[["Model", "In-distribution", "Cross-dataset", "Gap"]]
    fig, ax = plt.subplots(figsize=(6.2, 1.65), constrained_layout=True)
    ax.axis("off")
    table = ax.table(cellText=display.values, colLabels=display.columns, cellLoc="center", colLoc="center", loc="center", colWidths=[0.25, 0.25, 0.25, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        if row == 0:
            cell.set_facecolor(PALETTE["navy"])
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            architecture = summary.iloc[row - 1]["architecture"]
            cell.set_facecolor(PALETTE["wash"] if row % 2 else "white")
            if col == 0:
                cell.get_text().set_color(ARCH_COLOR[architecture])
                cell.get_text().set_fontweight("bold")
    ax.set_title("Model-level performance summary (mean ± SD across dataset cells)", fontweight="bold", pad=8)
    save_figure(fig, "table1_model_performance_summary")


def write_figure_manifest(coverage: pd.DataFrame) -> None:
    rows = [
        ("fig1", "five_class_auroc_heatmaps", "Main", "Shared-scale 5×5 macro-AUROC heatmaps"),
        ("fig2", "generalization_summary", "Main", "In-distribution versus cross-dataset performance and gap distributions"),
        ("fig3", "target_dataset_difficulty", "Main", "Target difficulty across architectures"),
        ("fig4", "per_class_cross_dataset_auroc", "Main/Supplement", "Per-class cross-dataset AUROC with SEM"),
        ("fig5", "mean_directional_gap_heatmap", "Main", "Average directional generalization-gap matrix"),
        ("fig6", "shift_vector_analysis", "Main", "Shift-vector gaps and descriptive weighted allocation"),
        ("fig7", "two_class_ablation", "Main after second architecture", "Five-class versus binary gap"),
        ("fig8", "bootstrap_ci_widths", "Supplement", "Bootstrap uncertainty by target dataset"),
        ("fig9", "directional_asymmetry", "Supplement", "A→B versus B→A transfer asymmetry"),
        ("fig10", "model_tradeoff", "Main", "In-distribution versus cross-dataset tradeoff"),
        ("table1", "model_performance_summary", "Main", "Model performance mean ± SD"),
    ]
    manifest = pd.DataFrame(rows, columns=["figure", "stem", "recommended_placement", "purpose"])
    manifest["formats"] = "PDF; SVG; PNG (350 dpi)"
    manifest.to_csv(OUT / "figure_manifest.csv", index=False)


def main() -> None:
    results, ci, coverage = load_results()
    shift = load_shift_vectors()
    _, off = make_summary_statistics(results, shift)
    figure_heatmaps(results)
    figure_generalization_summary(results)
    figure_target_difficulty(results)
    figure_per_class(results)
    figure_gap_heatmap(results)
    figure_shift_analysis(off)
    figure_binary_ablation()
    figure_bootstrap_width(ci)
    figure_directional_asymmetry(results)
    summary = figure_model_tradeoff(results)
    render_model_table(summary)
    write_figure_manifest(coverage)
    print(f"Generated figures in {OUT}")
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
