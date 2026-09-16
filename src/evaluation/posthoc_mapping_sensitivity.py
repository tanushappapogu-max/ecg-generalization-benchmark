#!/usr/bin/env python3
"""Post-hoc mapping sensitivity analysis using saved five-class predictions.

This diagnostic does not retrain any model and cannot recover source-native
diagnoses excluded by the submitted mapping.  It tests only mapping changes
that are identifiable from the five saved target/probability pairs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ARCHITECTURES = ("ecg_fm", "inception_time", "resnet1d", "transformer")
DATASETS = ("ptbxl", "cpsc2018", "georgia", "mimic_iv", "code_ii")
BASE_LABELS = ("normal", "af_afl", "av_block_1", "lbbb", "rbbb")


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return (
        float(roc_auc_score(y_true, y_score))
        if np.unique(y_true).size == 2
        else math.nan
    )


def _base(frame: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        label: (
            frame[f"target_{label}"].to_numpy(int),
            frame[f"probability_{label}"].to_numpy(float),
        )
        for label in BASE_LABELS
    }


def _exclusive_normal(frame: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    values = _base(frame)
    abnormal = np.maximum.reduce([values[label][0] for label in BASE_LABELS[1:]])
    normal = values["normal"]
    values["normal"] = (normal[0] * (1 - abnormal), normal[1])
    return values


def _abnormal_only(frame: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    values = _base(frame)
    return {label: values[label] for label in BASE_LABELS[1:]}


def _merged_bbb(frame: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    values = _base(frame)
    lbbb_y, lbbb_p = values.pop("lbbb")
    rbbb_y, rbbb_p = values.pop("rbbb")
    values["bbb"] = (np.maximum(lbbb_y, rbbb_y), np.maximum(lbbb_p, rbbb_p))
    return values


VARIANTS: dict[str, Callable[[pd.DataFrame], dict[str, tuple[np.ndarray, np.ndarray]]]] = {
    "consensus_five_class": _base,
    "exclusive_normal_five_class": _exclusive_normal,
    "abnormal_only_four_class": _abnormal_only,
    "merged_bbb_four_class": _merged_bbb,
}


def _score(frame: pd.DataFrame, transform: Callable) -> dict[str, float]:
    pairs = transform(frame)
    scores = {label: _safe_auc(y, p) for label, (y, p) in pairs.items()}
    finite = [score for score in scores.values() if math.isfinite(score)]
    return {
        "macro_auroc": float(np.mean(finite)) if finite else math.nan,
        **{f"auroc_{label}": score for label, score in scores.items()},
    }


def run(prediction_root: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for architecture in ARCHITECTURES:
        for source in DATASETS:
            for target in DATASETS:
                path = prediction_root / architecture / f"{source}__to__{target}" / "test_predictions.csv"
                if not path.is_file():
                    raise FileNotFoundError(path)
                frame = pd.read_csv(path, low_memory=False)
                required = {
                    column
                    for label in BASE_LABELS
                    for column in (f"target_{label}", f"probability_{label}")
                }
                missing = sorted(required - set(frame.columns))
                if missing:
                    raise ValueError(f"{path}: missing columns {missing}")
                for variant, transform in VARIANTS.items():
                    rows.append(
                        {
                            "analysis_type": "posthoc_mapping_sensitivity_no_retraining",
                            "mapping_variant": variant,
                            "architecture": architecture,
                            "source_dataset": source,
                            "target_dataset": target,
                            "is_diagonal": source == target,
                            "record_count": len(frame),
                            **_score(frame, transform),
                        }
                    )

    matrix = pd.DataFrame(rows)
    summary_rows: list[dict[str, object]] = []
    for (variant, architecture), group in matrix.groupby(
        ["mapping_variant", "architecture"], sort=False
    ):
        diagonal = group.loc[group.is_diagonal, "macro_auroc"]
        cross = group.loc[~group.is_diagonal, "macro_auroc"]
        summary_rows.append(
            {
                "mapping_variant": variant,
                "architecture": architecture,
                "mean_in_distribution_macro_auroc": float(diagonal.mean()),
                "mean_cross_dataset_macro_auroc": float(cross.mean()),
                "generalization_gap": float(diagonal.mean() - cross.mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary["model_rank_by_cross_dataset_auroc"] = summary.groupby(
        "mapping_variant"
    )["mean_cross_dataset_macro_auroc"].rank(method="min", ascending=False).astype(int)

    baseline = summary.loc[
        summary.mapping_variant.eq("consensus_five_class"),
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
        summary.mean_cross_dataset_macro_auroc
        - summary.baseline_cross_dataset_macro_auroc
    )
    summary["delta_generalization_gap_vs_baseline"] = (
        summary.generalization_gap - summary.baseline_generalization_gap
    )
    summary["rank_change_vs_baseline"] = (
        summary.model_rank_by_cross_dataset_auroc - summary.baseline_model_rank
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(output_dir / "posthoc_mapping_matrix_long.csv", index=False)
    summary.to_csv(output_dir / "posthoc_mapping_model_summary.csv", index=False)
    (output_dir / "posthoc_mapping_audit.json").write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "analysis_type": "posthoc_mapping_sensitivity_no_retraining",
                "architectures": list(ARCHITECTURES),
                "datasets": list(DATASETS),
                "mapping_variants": list(VARIANTS),
                "completed_cells_per_variant": len(ARCHITECTURES) * len(DATASETS) ** 2,
                "warning": (
                    "Fixed saved predictions were rescored. No model was retrained, and "
                    "source-native diagnoses excluded by the original mapping cannot be recovered."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return matrix, summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    _, summary = run(args.prediction_root, args.output_dir)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
