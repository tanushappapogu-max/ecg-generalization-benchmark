#!/usr/bin/env python3
"""Rescore saved five-label predictions under defensible mapping variants.

This is deliberately a *target-label rescoring* sensitivity analysis.  It does
not claim that the source models were retrained under the alternative mapping.
Each variant must provide a complete canonical manifest for every target while
preserving the baseline test record IDs and splits.  The command emits changes
in cross-dataset AUROC, generalization gap, per-class AUROC, and model rank.
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

from src.data.week2_manifest import LABEL_COLUMNS, validate_canonical_manifest
from src.evaluation.ecg_fm_matrix import DEFAULT_DATASETS, parse_named_paths
from src.training.binary_ablation_pipeline import ARCHITECTURES


def _load_test_manifest(path: Path, dataset: str) -> pd.DataFrame:
    manifest = validate_canonical_manifest(pd.read_csv(path, low_memory=False))
    found = manifest["dataset"].astype(str).unique().tolist()
    if found != [dataset]:
        raise ValueError(f"{path}: expected dataset {dataset!r}, found {found}")
    test = manifest.loc[manifest["split"].eq("test"), ["record_id", *LABEL_COLUMNS]].copy()
    test["record_id"] = test["record_id"].astype(str)
    if test["record_id"].duplicated().any():
        raise ValueError(f"{path}: duplicate test record IDs")
    return test


def _load_predictions(path: Path) -> pd.DataFrame:
    predictions = pd.read_csv(path, low_memory=False)
    required = {"record_id", *[f"probability_{label}" for label in LABEL_COLUMNS]}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"{path}: missing prediction columns {missing}")
    predictions["record_id"] = predictions["record_id"].astype(str)
    if predictions["record_id"].duplicated().any():
        raise ValueError(f"{path}: duplicate record IDs")
    return predictions


def _score(labels: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, float]:
    label_ids = set(labels["record_id"])
    prediction_ids = set(predictions["record_id"])
    if label_ids != prediction_ids:
        raise ValueError(
            "Variant labels and saved predictions must contain identical test record IDs; "
            f"missing_predictions={len(label_ids - prediction_ids)}, "
            f"extra_predictions={len(prediction_ids - label_ids)}"
        )
    joined = labels.merge(predictions, on="record_id", validate="one_to_one")
    per_class: dict[str, float] = {}
    for label in LABEL_COLUMNS:
        y_true = joined[label].to_numpy(int)
        y_score = joined[f"probability_{label}"].to_numpy(float)
        per_class[label] = (
            float(roc_auc_score(y_true, y_score))
            if np.unique(y_true).size == 2
            else math.nan
        )
    finite = [value for value in per_class.values() if math.isfinite(value)]
    return {
        "macro_auroc": float(np.mean(finite)) if finite else math.nan,
        **{f"auroc_{label}": value for label, value in per_class.items()},
    }


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    datasets = tuple(args.datasets)
    prediction_roots = parse_named_paths(args.prediction_root)
    manifest_roots = parse_named_paths(args.variant_manifest_root)
    if args.baseline_variant not in manifest_roots:
        raise ValueError(f"Baseline variant {args.baseline_variant!r} was not provided")
    missing_architectures = sorted(set(ARCHITECTURES) - set(prediction_roots))
    if missing_architectures:
        raise ValueError(f"Missing prediction roots for: {missing_architectures}")

    manifests: dict[tuple[str, str], pd.DataFrame] = {}
    for variant, root in manifest_roots.items():
        for dataset in datasets:
            path = root / f"{dataset}_week2.csv"
            if not path.is_file():
                raise FileNotFoundError(f"Missing {variant}/{dataset} manifest: {path}")
            manifests[(variant, dataset)] = _load_test_manifest(path, dataset)

    qc_rows: list[dict[str, object]] = []
    for dataset in datasets:
        baseline = manifests[(args.baseline_variant, dataset)].set_index("record_id")
        baseline_ids = set(baseline.index)
        for variant in manifest_roots:
            candidate = manifests[(variant, dataset)].set_index("record_id")
            if set(candidate.index) != baseline_ids:
                raise ValueError(
                    f"{variant}/{dataset} changes the baseline test record IDs; "
                    "that would confound mapping and sampling"
                )
            candidate = candidate.reindex(baseline.index)
            for label in LABEL_COLUMNS:
                qc_rows.append(
                    {
                        "mapping_variant": variant,
                        "dataset": dataset,
                        "label": label,
                        "test_records": len(candidate),
                        "positive_count": int(candidate[label].sum()),
                        "prevalence": float(candidate[label].mean()),
                        "records_changed_vs_baseline": int(
                            candidate[label].ne(baseline[label]).sum()
                        ),
                    }
                )

    rows: list[dict[str, object]] = []
    for architecture in ARCHITECTURES:
        root = prediction_roots[architecture]
        for source in datasets:
            for target in datasets:
                path = root / "predictions" / f"{source}__to__{target}" / "test_predictions.csv"
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Missing saved predictions for {architecture} {source}->{target}: {path}"
                    )
                predictions = _load_predictions(path)
                for variant in manifest_roots:
                    metrics = _score(manifests[(variant, target)], predictions)
                    rows.append(
                        {
                            "mapping_variant": variant,
                            "analysis_type": "target_rescore_no_retraining",
                            "architecture": architecture,
                            "source_dataset": source,
                            "target_dataset": target,
                            "is_diagonal": source == target,
                            "record_count": len(predictions),
                            **metrics,
                        }
                    )

    matrix = pd.DataFrame(rows)
    summaries: list[dict[str, object]] = []
    for (variant, architecture), group in matrix.groupby(
        ["mapping_variant", "architecture"], sort=False
    ):
        diagonal = group.loc[group["is_diagonal"], "macro_auroc"]
        cross = group.loc[~group["is_diagonal"], "macro_auroc"]
        summaries.append(
            {
                "mapping_variant": variant,
                "architecture": architecture,
                "mean_in_distribution_macro_auroc": float(diagonal.mean()),
                "mean_cross_dataset_macro_auroc": float(cross.mean()),
                "generalization_gap": float(diagonal.mean() - cross.mean()),
                **{
                    f"mean_cross_dataset_auroc_{label}": float(
                        group.loc[~group["is_diagonal"], f"auroc_{label}"].mean()
                    )
                    for label in LABEL_COLUMNS
                },
            }
        )
    summary = pd.DataFrame(summaries)
    summary["model_rank_by_cross_dataset_auroc"] = summary.groupby(
        "mapping_variant"
    )["mean_cross_dataset_macro_auroc"].rank(method="min", ascending=False).astype(int)
    baseline = summary.loc[
        summary["mapping_variant"].eq(args.baseline_variant),
        [
            "architecture",
            "mean_cross_dataset_macro_auroc",
            "generalization_gap",
            "model_rank_by_cross_dataset_auroc",
        ],
    ].rename(
        columns={
            "mean_cross_dataset_macro_auroc": "baseline_cross_dataset_macro_auroc",
            "generalization_gap": "baseline_generalization_gap",
            "model_rank_by_cross_dataset_auroc": "baseline_model_rank",
        }
    )
    summary = summary.merge(baseline, on="architecture", validate="many_to_one")
    summary["delta_cross_dataset_macro_auroc_vs_baseline"] = (
        summary["mean_cross_dataset_macro_auroc"]
        - summary["baseline_cross_dataset_macro_auroc"]
    )
    summary["delta_generalization_gap_vs_baseline"] = (
        summary["generalization_gap"] - summary["baseline_generalization_gap"]
    )
    summary["rank_change_vs_baseline"] = (
        summary["model_rank_by_cross_dataset_auroc"] - summary["baseline_model_rank"]
    )

    qc = pd.DataFrame(qc_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(args.output_dir / "mapping_sensitivity_matrix_long.csv", index=False)
    summary.to_csv(args.output_dir / "mapping_sensitivity_model_summary.csv", index=False)
    qc.to_csv(args.output_dir / "mapping_sensitivity_label_qc.csv", index=False)
    audit = {
        "status": "COMPLETE",
        "analysis_type": "target_rescore_no_retraining",
        "warning": (
            "This measures sensitivity to target-label definitions using fixed saved "
            "predictions. It is not equivalent to retraining source models under each mapping."
        ),
        "baseline_variant": args.baseline_variant,
        "mapping_variants": list(manifest_roots),
        "architectures": list(ARCHITECTURES),
        "datasets": list(datasets),
        "matrix_rows": len(matrix),
    }
    (args.output_dir / "mapping_sensitivity_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    return matrix, summary, qc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", action="append", default=[], metavar="ARCH=PATH")
    parser.add_argument(
        "--variant-manifest-root", action="append", default=[], metavar="VARIANT=PATH"
    )
    parser.add_argument("--baseline-variant", default="consensus_v1")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _, summary, _ = run(args)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
