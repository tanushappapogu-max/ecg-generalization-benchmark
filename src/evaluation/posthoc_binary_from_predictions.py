#!/usr/bin/env python3
"""Compute a CPU-only normal-vs-abnormal analysis from saved five-label predictions.

This analysis does not retrain a binary classifier. It collapses the four
abnormal ground-truth labels with logical OR and combines their saved
probabilities using prespecified aggregation rules. The output labels the
result ``posthoc_label_collapse`` so it cannot be confused with the separately
trained binary ablation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data.week2_manifest import LABEL_COLUMNS
from src.evaluation.ecg_fm_matrix import DEFAULT_DATASETS, parse_named_paths
from src.training.binary_ablation_pipeline import ARCHITECTURES


ABNORMAL_LABELS = tuple(label for label in LABEL_COLUMNS if label != "normal")
AGGREGATIONS = ("max_abnormal", "noisy_or", "one_minus_normal")


def score_cell(frame: pd.DataFrame, aggregation: str) -> dict[str, float | int]:
    required = {
        *[f"target_{label}" for label in LABEL_COLUMNS],
        *[f"probability_{label}" for label in LABEL_COLUMNS],
    }
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"Prediction file is missing columns: {missing}")
    target_abnormal = frame[[f"target_{label}" for label in ABNORMAL_LABELS]].max(axis=1)
    has_benchmark_label = frame[[f"target_{label}" for label in LABEL_COLUMNS]].max(axis=1).astype(bool)
    kept = frame.loc[has_benchmark_label].copy()
    y_true = target_abnormal.loc[has_benchmark_label].to_numpy(int)
    abnormal_probabilities = kept[
        [f"probability_{label}" for label in ABNORMAL_LABELS]
    ].to_numpy(float)
    if aggregation == "max_abnormal":
        y_score = abnormal_probabilities.max(axis=1)
    elif aggregation == "noisy_or":
        y_score = 1.0 - np.prod(1.0 - abnormal_probabilities, axis=1)
    elif aggregation == "one_minus_normal":
        y_score = 1.0 - kept["probability_normal"].to_numpy(float)
    else:
        raise ValueError(f"Unknown aggregation {aggregation!r}")
    if np.unique(y_true).size != 2:
        raise ValueError("Collapsed target does not contain both classes")

    per_class_all_records = []
    per_class_binary_eligible = []
    for label in LABEL_COLUMNS:
        target_all = frame[f"target_{label}"].to_numpy(int)
        probability_all = frame[f"probability_{label}"].to_numpy(float)
        if np.unique(target_all).size == 2:
            per_class_all_records.append(
                float(roc_auc_score(target_all, probability_all))
            )
        target_eligible = kept[f"target_{label}"].to_numpy(int)
        probability_eligible = kept[f"probability_{label}"].to_numpy(float)
        if np.unique(target_eligible).size == 2:
            per_class_binary_eligible.append(
                float(roc_auc_score(target_eligible, probability_eligible))
            )
    return {
        "record_count_original": int(len(frame)),
        "record_count_scored": int(len(kept)),
        "excluded_all_zero_count": int((~has_benchmark_label).sum()),
        "abnormal_prevalence": float(y_true.mean()),
        "binary_auroc": float(roc_auc_score(y_true, y_score)),
        "five_label_macro_auroc_all_records": float(
            np.mean(per_class_all_records)
        ),
        "five_label_macro_auroc_binary_eligible_records": float(
            np.mean(per_class_binary_eligible)
        ),
    }


def _gap(frame: pd.DataFrame, metric: str) -> tuple[float, float, float]:
    diagonal = frame.loc[frame["source_dataset"].eq(frame["target_dataset"]), metric]
    cross = frame.loc[frame["source_dataset"].ne(frame["target_dataset"]), metric]
    return float(diagonal.mean()), float(cross.mean()), float(diagonal.mean() - cross.mean())


def _paired_cell_bootstrap(
    frame: pd.DataFrame, *, replicates: int, seed: int
) -> tuple[float, float]:
    is_diagonal = frame["source_dataset"].eq(frame["target_dataset"]).to_numpy()
    five = frame["five_label_macro_auroc_all_records"].to_numpy(float)
    binary = frame["binary_auroc"].to_numpy(float)
    diagonal_indices = np.flatnonzero(is_diagonal)
    cross_indices = np.flatnonzero(~is_diagonal)
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=float)
    for index in range(replicates):
        diagonal = rng.choice(diagonal_indices, size=len(diagonal_indices), replace=True)
        cross = rng.choice(cross_indices, size=len(cross_indices), replace=True)
        five_gap = five[diagonal].mean() - five[cross].mean()
        binary_gap = binary[diagonal].mean() - binary[cross].mean()
        values[index] = five_gap - binary_gap
    return tuple(map(float, np.quantile(values, [0.025, 0.975])))


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    datasets = tuple(args.datasets)
    roots = parse_named_paths(args.prediction_root)
    if missing := sorted(set(ARCHITECTURES) - set(roots)):
        raise ValueError(f"Missing prediction roots for: {missing}")
    rows: list[dict[str, object]] = []
    for architecture in ARCHITECTURES:
        for source in datasets:
            for target in datasets:
                path = roots[architecture] / f"{source}__to__{target}" / "test_predictions.csv"
                if not path.is_file():
                    raise FileNotFoundError(path)
                predictions = pd.read_csv(path, low_memory=False)
                for aggregation in AGGREGATIONS:
                    rows.append(
                        {
                            "analysis_type": "posthoc_label_collapse_no_retraining",
                            "architecture": architecture,
                            "source_dataset": source,
                            "target_dataset": target,
                            "aggregation": aggregation,
                            **score_cell(predictions, aggregation),
                        }
                    )
    long = pd.DataFrame(rows)
    summaries: list[dict[str, object]] = []
    for group_index, ((architecture, aggregation), frame) in enumerate(
        long.groupby(["architecture", "aggregation"], sort=False)
    ):
        five_id, five_cross, five_gap = _gap(frame, "five_label_macro_auroc_all_records")
        eligible_five_id, eligible_five_cross, eligible_five_gap = _gap(
            frame, "five_label_macro_auroc_binary_eligible_records"
        )
        binary_id, binary_cross, binary_gap = _gap(frame, "binary_auroc")
        reduction = five_gap - binary_gap
        ci_low, ci_high = _paired_cell_bootstrap(
            frame,
            replicates=args.bootstrap_replicates,
            seed=args.seed + group_index,
        )
        summaries.append(
            {
                "analysis_type": "posthoc_label_collapse_no_retraining",
                "architecture": architecture,
                "aggregation": aggregation,
                "five_label_mean_in_distribution_auroc": five_id,
                "five_label_mean_cross_dataset_auroc": five_cross,
                "five_label_generalization_gap": five_gap,
                "five_label_binary_eligible_mean_in_distribution_auroc": eligible_five_id,
                "five_label_binary_eligible_mean_cross_dataset_auroc": eligible_five_cross,
                "five_label_binary_eligible_generalization_gap": eligible_five_gap,
                "binary_mean_in_distribution_auroc": binary_id,
                "binary_mean_cross_dataset_auroc": binary_cross,
                "binary_generalization_gap": binary_gap,
                "absolute_gap_reduction": reduction,
                "percent_gap_reduction": 100.0 * reduction / five_gap if five_gap else math.nan,
                "gap_reduction_cell_bootstrap_ci_95_lower": ci_low,
                "gap_reduction_cell_bootstrap_ci_95_upper": ci_high,
            }
        )
    summary = pd.DataFrame(summaries)
    summary["model_rank_by_binary_cross_dataset_auroc"] = summary.groupby(
        "aggregation"
    )["binary_mean_cross_dataset_auroc"].rank(method="min", ascending=False).astype(int)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    long.to_csv(args.output_dir / "posthoc_binary_matrix_long.csv", index=False)
    summary.to_csv(args.output_dir / "posthoc_binary_summary.csv", index=False)
    audit = {
        "status": "COMPLETE",
        "analysis_type": "posthoc_label_collapse_no_retraining",
        "architectures": list(ARCHITECTURES),
        "datasets": list(datasets),
        "aggregations": list(AGGREGATIONS),
        "completed_cells_per_aggregation": len(ARCHITECTURES) * len(datasets) ** 2,
        "primary_aggregation": "max_abnormal",
        "warning": "This is not a retrained binary classifier experiment.",
        "bootstrap_unit": "paired source-target cells; descriptive, not independent training replicates",
    }
    (args.output_dir / "posthoc_binary_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    return long, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", action="append", default=[], metavar="ARCH=PATH")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.bootstrap_replicates <= 0:
        raise ValueError("--bootstrap-replicates must be positive")
    _, summary = run(args)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
