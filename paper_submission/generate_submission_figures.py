#!/usr/bin/env python3
"""Generate the four submission figures from audited result tables."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "navy": "#053351",
    "slate": "#4D6076",
    "slate_light": "#88919E",
    "mauve": "#9688A4",
    "purple": "#664E82",
    "deep_purple": "#341760",
    "rose": "#BB7E8C",
    "rose_light": "#D6B5C3",
    "paper": "#FFFFFF",
    "grid": "#C6C6C6",
    "wash": "#F3F2F4",
}

ARCH_ORDER = ["resnet1d", "inception_time", "ecg_fm", "transformer"]
ARCH_LABEL = {
    "resnet1d": "ResNet1D",
    "inception_time": "InceptionTime",
    "ecg_fm": "ECG-FM",
    "transformer": "Transformer",
}
ARCH_COLOR = {
    "resnet1d": PALETTE["slate"],
    "inception_time": PALETTE["navy"],
    "ecg_fm": PALETTE["deep_purple"],
    "transformer": PALETTE["rose"],
}
ARCH_MARKER = {"resnet1d": "D", "inception_time": "o", "ecg_fm": "s", "transformer": "^"}
DATASETS = ["ptbxl", "cpsc2018", "georgia", "mimic_iv", "code_ii"]
DATASET_LABEL = {
    "ptbxl": "PTB-XL",
    "cpsc2018": "CPSC2018",
    "georgia": "Georgia",
    "mimic_iv": "MIMIC-IV-ECG",
    "code_ii": "CODE-15%",
}

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "axes.edgecolor": PALETTE["slate"],
        "axes.labelcolor": PALETTE["navy"],
        "text.color": PALETTE["navy"],
        "xtick.color": PALETTE["slate"],
        "ytick.color": PALETTE["slate"],
        "grid.color": PALETTE["grid"],
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
sns.set_style("whitegrid", {"grid.color": PALETTE["grid"]})


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


matrix_long = pd.read_csv(
    ROOT / "analysis_outputs/reviewer_posthoc_binary/posthoc_binary_matrix_long.csv"
)
matrix_long = matrix_long.loc[matrix_long["aggregation"].eq("max_abnormal")].copy()


def heatmaps() -> None:
    colors = [PALETTE["rose_light"], PALETTE["mauve"], PALETTE["purple"], PALETTE["deep_purple"]]
    cmap = LinearSegmentedColormap.from_list("benchmark", colors)
    fig = plt.figure(figsize=(10.4, 7.9))
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=[1, 1, 0.045],
        left=0.085,
        right=0.925,
        bottom=0.095,
        top=0.90,
        wspace=0.34,
        hspace=0.43,
    )
    axes = np.array(
        [
            [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])],
            [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])],
        ]
    )
    colorbar_ax = fig.add_subplot(grid[:, 2])
    for panel, (ax, architecture) in enumerate(zip(axes.flat, ARCH_ORDER)):
        group = matrix_long.loc[matrix_long.architecture.eq(architecture)]
        pivot = group.pivot(
            index="source_dataset",
            columns="target_dataset",
            values="five_label_macro_auroc_all_records",
        ).reindex(index=DATASETS, columns=DATASETS)
        sns.heatmap(
            pivot,
            ax=ax,
            cmap=cmap,
            vmin=0.65,
            vmax=1.0,
            square=True,
            annot=True,
            fmt=".3f",
            annot_kws={"fontsize": 8},
            linewidths=0.8,
            linecolor="white",
            cbar=panel == 0,
            cbar_ax=colorbar_ax if panel == 0 else None,
            cbar_kws={"label": "Macro-AUROC"},
        )
        ax.set_title(ARCH_LABEL[architecture], color=ARCH_COLOR[architecture], fontweight="bold")
        ax.set_xlabel("Evaluation target")
        ax.set_ylabel("Training source")
        ax.set_xticklabels([DATASET_LABEL[x] for x in DATASETS], rotation=28, ha="right")
        ax.set_yticklabels([DATASET_LABEL[x] for x in DATASETS], rotation=0)
        ax.text(-0.13, 1.06, chr(65 + panel), transform=ax.transAxes, fontsize=12, fontweight="bold", color=PALETTE["deep_purple"])
        for i in range(len(DATASETS)):
            ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor=PALETTE["navy"], linewidth=2.0))
    colorbar_ax.set_ylabel("Macro-AUROC", color=PALETTE["navy"])
    fig.suptitle("Five-class performance across all source-to-target evaluations", fontsize=14, fontweight="bold", color=PALETTE["navy"], y=0.975)
    save(fig, "fig2_four_model_heatmaps")


def generalization_summary() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    fig.subplots_adjust(left=0.09, right=0.975, bottom=0.17, top=0.86, wspace=0.30)
    ax = axes[0]
    left_offsets = {"resnet1d": 0.0026, "inception_time": -0.0024, "ecg_fm": 0.0012, "transformer": 0.0}
    for architecture in ARCH_ORDER:
        group = matrix_long.loc[matrix_long.architecture.eq(architecture)]
        diagonal = group.loc[group.source_dataset.eq(group.target_dataset), "five_label_macro_auroc_all_records"].mean()
        cross = group.loc[~group.source_dataset.eq(group.target_dataset), "five_label_macro_auroc_all_records"].mean()
        ax.plot([0, 1], [diagonal, cross], marker=ARCH_MARKER[architecture], markersize=7, linewidth=2.2, color=ARCH_COLOR[architecture], label=ARCH_LABEL[architecture])
        ax.text(-0.025, diagonal + left_offsets[architecture], f"{diagonal:.3f}", ha="right", va="center", fontsize=8, color=ARCH_COLOR[architecture], clip_on=True)
        ax.text(1.03, cross, f"{cross:.3f}", ha="left", va="center", fontsize=8, color=ARCH_COLOR[architecture], clip_on=True)
    ax.set_xticks([0, 1], ["In-distribution", "Cross-dataset"])
    ax.set_xlim(-0.18, 1.22)
    ax.set_ylim(0.86, 0.985)
    ax.set_ylabel("Mean macro-AUROC")
    ax.set_title("A  High in-distribution AUROC does not guarantee transfer", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="lower left")

    ax = axes[1]
    external = matrix_long.loc[~matrix_long.source_dataset.eq(matrix_long.target_dataset)].copy()
    target = external.groupby("target_dataset")["five_label_macro_auroc_all_records"].agg(["mean", "sem"]).reindex(DATASETS).sort_values("mean")
    y = np.arange(len(target))
    colors = [PALETTE["rose"] if idx == "mimic_iv" else PALETTE["purple"] for idx in target.index]
    ax.errorbar(target["mean"], y, xerr=target["sem"], fmt="none", ecolor=PALETTE["slate"], capsize=3, linewidth=1.2)
    ax.scatter(target["mean"], y, s=65, c=colors, edgecolor="white", linewidth=0.8, zorder=3)
    for yy, value in enumerate(target["mean"]):
        ax.annotate(
            f"{value:.3f}",
            xy=(value, yy),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color=PALETTE["navy"],
            clip_on=False,
        )
    ax.set_yticks(y, [DATASET_LABEL[x] for x in target.index])
    ax.set_xlim(0.82, 0.94)
    ax.set_xlabel("Mean external-target macro-AUROC")
    ax.set_title("B  MIMIC-IV-ECG is the hardest external target", loc="left", fontweight="bold")
    save(fig, "fig3_generalization_summary")


def sensitivity_summary() -> None:
    binary = pd.read_csv(ROOT / "analysis_outputs/reviewer_posthoc_binary/posthoc_binary_summary.csv")
    binary = binary.loc[binary.aggregation.eq("max_abnormal")].set_index("architecture").reindex(ARCH_ORDER)
    mapping = pd.read_csv(ROOT / "analysis_outputs/reviewer_posthoc_mapping/posthoc_mapping_model_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    ax = axes[0]
    y = np.arange(len(ARCH_ORDER))
    for yy, architecture in enumerate(ARCH_ORDER):
        row = binary.loc[architecture]
        five = row["five_label_binary_eligible_generalization_gap"]
        two = row["binary_generalization_gap"]
        reduction = 100 * (five - two) / five
        ax.plot([five, two], [yy, yy], color=ARCH_COLOR[architecture], linewidth=2.2)
        ax.scatter([five, two], [yy, yy], s=55, marker=ARCH_MARKER[architecture], color=ARCH_COLOR[architecture], edgecolor="white", zorder=3)
        ax.text(max(five, two) + 0.002, yy, f"{reduction:.1f}%", va="center", fontsize=8, color=ARCH_COLOR[architecture], fontweight="bold")
    ax.set_yticks(y, [ARCH_LABEL[x] for x in ARCH_ORDER])
    ax.invert_yaxis()
    ax.set_xlabel("Matched-record generalization gap")
    ax.set_title("A  Normal-vs-abnormal reduces the gap by 22.5% overall", loc="left", fontweight="bold")

    ax = axes[1]
    variants = [
        "consensus_five_class",
        "exclusive_normal_five_class",
        "abnormal_only_four_class",
        "merged_bbb_four_class",
    ]
    labels = ["Baseline\n5-class", "Exclusive\nnormal", "Remove\nnormal", "Merge\nLBBB/RBBB"]
    for architecture in ARCH_ORDER:
        group = mapping.loc[mapping.architecture.eq(architecture)].set_index("mapping_variant").reindex(variants)
        ax.plot(range(len(variants)), group.mean_cross_dataset_macro_auroc, color=ARCH_COLOR[architecture], marker=ARCH_MARKER[architecture], linewidth=1.9, markersize=6, label=ARCH_LABEL[architecture])
    ax.set_xticks(range(len(variants)), labels)
    ax.set_ylim(0.84, 0.97)
    ax.set_ylabel("Mean cross-dataset macro-AUROC")
    ax.set_title("B  Endpoint mappings change AUROC, not model order", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower left")
    ax.axvline(1.5, color=PALETTE["grid"], linestyle="--", linewidth=0.9)
    ax.text(0.02, -0.26, "Five-class", transform=ax.transAxes, fontsize=7.5, color=PALETTE["slate"])
    ax.text(0.68, -0.26, "Four-class sensitivity", transform=ax.transAxes, fontsize=7.5, color=PALETTE["slate"])
    save(fig, "fig4_revision_sensitivity")


if __name__ == "__main__":
    heatmaps()
    generalization_summary()
    sensitivity_summary()
    print(f"Wrote submission figures to {OUT}")
