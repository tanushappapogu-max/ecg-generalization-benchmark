#!/usr/bin/env python3
"""Build the pooled Week 3 ECG benchmark analysis from completed matrices.

The command consumes one five-label matrix per architecture plus the committed
20-row shift-vector table.  It emits the proposal gap/weight/Score components,
the shift-vector group table, a transparent descriptive attribution profile,
the lambda-weight sensitivity sweep, bootstrap confidence intervals from the
saved per-record predictions, and paired sign-flip tests across architectures.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score

try:
    from src.evaluation.composite_score import (
        PAIR_COLUMNS,
        SHIFT_COLUMNS,
        canonicalize_pair_columns,
        compute_composite_score,
    )
    from src.evaluation.ecg_fm_matrix import parse_named_paths
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.evaluation.composite_score import (
        PAIR_COLUMNS,
        SHIFT_COLUMNS,
        canonicalize_pair_columns,
        compute_composite_score,
    )
    from src.evaluation.ecg_fm_matrix import parse_named_paths


DEFAULT_LAMBDA_GRID = (0.5, 1.0, 2.0, 3.0)


def read_architecture_matrices(named_paths: Sequence[str]) -> tuple[pd.DataFrame, dict[str, Path]]:
    """Read and validate repeatable ``ARCHITECTURE=CSV`` arguments."""

    paths = parse_named_paths(named_paths)
    if not paths:
        raise ValueError("At least one --matrix ARCHITECTURE=CSV is required")
    frames: list[pd.DataFrame] = []
    for architecture, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = canonicalize_pair_columns(pd.read_csv(path))
        required = {*PAIR_COLUMNS, "macro_auroc"}
        if missing := sorted(required - set(frame.columns)):
            raise ValueError(f"{path} is missing columns: {missing}")
        if "status" in frame:
            blocked = frame.loc[~frame["status"].eq("COMPLETE")]
            if len(blocked):
                raise ValueError(f"{path} has {len(blocked)} incomplete matrix cells")
        if frame.duplicated(list(PAIR_COLUMNS)).any():
            raise ValueError(f"{path} contains duplicate source-target cells")
        frame["architecture"] = architecture
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), paths


def add_gaps_and_shifts(pooled: pd.DataFrame, shifts: pd.DataFrame) -> pd.DataFrame:
    """Attach each source diagonal, delta, and the binary shift vector."""

    shifts = canonicalize_pair_columns(shifts)
    required = {*PAIR_COLUMNS, *SHIFT_COLUMNS}
    if missing := sorted(required - set(shifts.columns)):
        raise ValueError(f"Shift table is missing columns: {missing}")
    if shifts.duplicated(list(PAIR_COLUMNS)).any():
        raise ValueError("Shift table contains duplicate ordered pairs")
    diagonal = pooled.loc[
        pooled["source_dataset"].eq(pooled["target_dataset"]),
        ["architecture", "source_dataset", "macro_auroc"],
    ].rename(columns={"macro_auroc": "in_distribution_auroc"})
    if diagonal.duplicated(["architecture", "source_dataset"]).any():
        raise ValueError("Pooled matrices contain duplicate diagonal cells")
    off = pooled.loc[pooled["source_dataset"].ne(pooled["target_dataset"])].copy()
    off = off.merge(
        diagonal,
        on=["architecture", "source_dataset"],
        how="left",
        validate="many_to_one",
    )
    off = off.merge(
        shifts[list(PAIR_COLUMNS) + list(SHIFT_COLUMNS)],
        on=list(PAIR_COLUMNS),
        how="left",
        validate="many_to_one",
    )
    if off[["in_distribution_auroc", *SHIFT_COLUMNS]].isna().any().any():
        raise ValueError("Every completed off-diagonal cell needs a diagonal and shift vector")
    for column in SHIFT_COLUMNS:
        off[column] = pd.to_numeric(off[column], errors="raise").astype(int)
        if not set(off[column].unique()).issubset({0, 1}):
            raise ValueError(f"{column} must be binary")
    off["delta_ij"] = off["in_distribution_auroc"] - off["macro_auroc"]
    off["w_ij"] = off["PS"] + 2.0 * off["DS"] + 3.0 * off["LS"]
    off["shift_vector"] = list(zip(off["PS"], off["DS"], off["LS"]))
    return off


def composite_summary(
    pooled: pd.DataFrame,
    shifts: pd.DataFrame,
    *,
    weights: tuple[float, float, float] = (1.0, 2.0, 3.0),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the proposal Score once per architecture."""

    summaries: list[dict[str, float | str]] = []
    components: list[pd.DataFrame] = []
    for architecture, frame in pooled.groupby("architecture", sort=True):
        score, part = compute_composite_score(
            frame,
            shifts,
            lambda_ps=weights[0],
            lambda_ds=weights[1],
            lambda_ls=weights[2],
        )
        if "architecture" in part:
            part["architecture"] = architecture
        else:
            part.insert(0, "architecture", architecture)
        components.append(part)
        summaries.append(
            {
                "architecture": architecture,
                "lambda_ps": weights[0],
                "lambda_ds": weights[1],
                "lambda_ls": weights[2],
                "mean_cross_dataset_auroc": float(part["macro_auroc"].mean()),
                "mean_cross_dataset_gap": float(part["delta_ij"].mean()),
                "composite_score": score,
            }
        )
    return pd.DataFrame(summaries), pd.concat(components, ignore_index=True)


def shift_group_table(off: pd.DataFrame) -> pd.DataFrame:
    """Average gap for every observed shift-vector group within each architecture."""

    return (
        off.groupby(["architecture", *SHIFT_COLUMNS], as_index=False)
        .agg(
            ordered_pair_count=("delta_ij", "size"),
            mean_gap=("delta_ij", "mean"),
            std_gap=("delta_ij", "std"),
            mean_cross_dataset_auroc=("macro_auroc", "mean"),
        )
        .sort_values(["architecture", *SHIFT_COLUMNS])
        .reset_index(drop=True)
    )


def descriptive_attribution(off: pd.DataFrame) -> pd.DataFrame:
    """Allocate observed gaps across active shifts using proposal weights.

    This is explicitly descriptive rather than causal.  A cell's gap is split
    among its active shift flags in proportion to the proposal weights 1/2/3.
    The resulting shares always sum to one when total allocated gap is nonzero.
    """

    rows: list[dict[str, float | int | str]] = []
    weight = {"PS": 1.0, "DS": 2.0, "LS": 3.0}
    method = "gap allocated across active flags proportional to proposal weights; descriptive, not causal"
    for architecture, frame in off.groupby("architecture", sort=True):
        allocated: dict[str, float] = {}
        for shift in SHIFT_COLUMNS:
            denominator = sum(weight[name] * frame[name] for name in SHIFT_COLUMNS)
            fractions = np.divide(
                weight[shift] * frame[shift],
                denominator,
                out=np.zeros(len(frame), dtype=float),
                where=np.asarray(denominator) != 0,
            )
            allocated[shift] = float(np.sum(frame["delta_ij"].to_numpy() * fractions))
        total = sum(allocated.values())
        for shift in SHIFT_COLUMNS:
            rows.append(
                {
                    "architecture": architecture,
                    "shift_type": shift,
                    "allocated_gap_total": allocated[shift],
                    "allocated_gap_share": allocated[shift] / total if total else math.nan,
                    "off_diagonal_cell_count": len(frame),
                    "method": method,
                }
            )
    return pd.DataFrame(rows)


def lambda_sensitivity(
    pooled: pd.DataFrame,
    shifts: pd.DataFrame,
    grid: Sequence[float] = DEFAULT_LAMBDA_GRID,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sweep shift weights and compare rankings with the proposal 1/2/3 ranking."""

    values = tuple(sorted(set(float(value) for value in grid)))
    if not values or min(values) < 0:
        raise ValueError("Lambda grid must contain non-negative values")

    def scores_for(weights: tuple[float, float, float]) -> dict[str, float]:
        output: dict[str, float] = {}
        for architecture, frame in pooled.groupby("architecture", sort=True):
            output[architecture] = compute_composite_score(
                frame,
                shifts,
                lambda_ps=weights[0],
                lambda_ds=weights[1],
                lambda_ls=weights[2],
            )[0]
        return output

    reference_scores = scores_for((1.0, 2.0, 3.0))
    reference_rank = pd.Series(reference_scores).rank(method="average", ascending=False)
    long_rows: list[dict[str, float | int | str]] = []
    summary_rows: list[dict[str, float | int]] = []
    for sweep_id, weights in enumerate(itertools.product(values, repeat=3), start=1):
        scores = scores_for(weights)
        ranks = pd.Series(scores).rank(method="average", ascending=False)
        tau = float(kendalltau(reference_rank, ranks).statistic)
        summary_rows.append(
            {
                "sweep_id": sweep_id,
                "lambda_ps": weights[0],
                "lambda_ds": weights[1],
                "lambda_ls": weights[2],
                "kendall_tau_vs_1_2_3": tau,
            }
        )
        for architecture in sorted(scores):
            long_rows.append(
                {
                    "sweep_id": sweep_id,
                    "lambda_ps": weights[0],
                    "lambda_ds": weights[1],
                    "lambda_ls": weights[2],
                    "architecture": architecture,
                    "composite_score": scores[architecture],
                    "rank": float(ranks[architecture]),
                    "kendall_tau_vs_1_2_3": tau,
                }
            )
    return pd.DataFrame(long_rows), pd.DataFrame(summary_rows)


def _bootstrap_metric(
    predictions: pd.DataFrame,
    *,
    replicates: int,
    rng: np.random.Generator,
) -> list[dict[str, float | int | str]]:
    target_columns = sorted(column for column in predictions if column.startswith("target_"))
    labels = [column.removeprefix("target_") for column in target_columns]
    if not labels:
        raise ValueError("Prediction CSV contains no target_* columns")
    probability_columns = [f"probability_{label}" for label in labels]
    missing = sorted(set(probability_columns) - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction CSV is missing columns: {missing}")
    y_true = predictions[target_columns].to_numpy(dtype=int)
    y_score = predictions[probability_columns].to_numpy(dtype=float)
    n = len(predictions)
    if n == 0:
        raise ValueError("Prediction CSV is empty")
    samples: dict[str, list[float]] = {label: [] for label in labels}
    samples["macro"] = []
    for _ in range(replicates):
        index = rng.integers(0, n, size=n)
        class_values: list[float] = []
        for label_index, label in enumerate(labels):
            truth = y_true[index, label_index]
            if np.unique(truth).size < 2:
                continue
            value = float(roc_auc_score(truth, y_score[index, label_index]))
            samples[label].append(value)
            class_values.append(value)
        if len(class_values) == len(labels):
            samples["macro"].append(float(np.mean(class_values)))
    rows: list[dict[str, float | int | str]] = []
    for metric, values in samples.items():
        array = np.asarray(values, dtype=float)
        rows.append(
            {
                "metric": "macro_auroc" if metric == "macro" else f"auroc_{metric}",
                "bootstrap_valid_replicates": len(array),
                "ci_95_lower": float(np.quantile(array, 0.025)) if len(array) else math.nan,
                "ci_95_upper": float(np.quantile(array, 0.975)) if len(array) else math.nan,
            }
        )
    return rows


def bootstrap_all_cells(
    matrix_paths: dict[str, Path],
    *,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    """Bootstrap every cell from the per-record predictions saved by evaluators."""

    if replicates <= 0:
        return pd.DataFrame()
    rows: list[dict[str, float | int | str]] = []
    for architecture_index, (architecture, matrix_path) in enumerate(sorted(matrix_paths.items())):
        matrix = canonicalize_pair_columns(pd.read_csv(matrix_path))
        for cell_index, cell in matrix.reset_index(drop=True).iterrows():
            source = str(cell["source_dataset"])
            target = str(cell["target_dataset"])
            prediction_path = matrix_path.parent / "predictions" / f"{source}__to__{target}" / "test_predictions.csv"
            if not prediction_path.is_file():
                raise FileNotFoundError(prediction_path)
            rng = np.random.default_rng(seed + architecture_index * 10_000 + cell_index)
            for result in _bootstrap_metric(
                pd.read_csv(prediction_path), replicates=replicates, rng=rng
            ):
                rows.append(
                    {
                        "architecture": architecture,
                        "source_dataset": source,
                        "target_dataset": target,
                        "bootstrap_requested_replicates": replicates,
                        **result,
                    }
                )
    return pd.DataFrame(rows)


def _exact_paired_sign_flip(differences: np.ndarray) -> float:
    differences = differences[np.isfinite(differences)]
    if not len(differences):
        return math.nan
    observed = abs(float(np.mean(differences)))
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(differences))))
    permuted = np.abs(np.mean(signs * differences, axis=1))
    return float(np.mean(permuted >= observed - 1e-15))


def paired_shift_vector_tests(groups: pd.DataFrame) -> pd.DataFrame:
    """Compare shift-vector mean gaps with architecture as the paired block."""

    vectors = sorted(
        {tuple(row) for row in groups[[*SHIFT_COLUMNS]].itertuples(index=False, name=None)}
    )
    rows: list[dict[str, float | int | str]] = []
    for left, right in itertools.combinations(vectors, 2):
        left_rows = groups.loc[
            (groups[list(SHIFT_COLUMNS)] == left).all(axis=1), ["architecture", "mean_gap"]
        ].rename(columns={"mean_gap": "left_mean_gap"})
        right_rows = groups.loc[
            (groups[list(SHIFT_COLUMNS)] == right).all(axis=1), ["architecture", "mean_gap"]
        ].rename(columns={"mean_gap": "right_mean_gap"})
        paired = left_rows.merge(right_rows, on="architecture", how="inner")
        differences = paired["left_mean_gap"].to_numpy() - paired["right_mean_gap"].to_numpy()
        rows.append(
            {
                "left_shift_vector": str(left),
                "right_shift_vector": str(right),
                "paired_architecture_count": len(paired),
                "mean_paired_gap_difference": float(np.mean(differences)) if len(differences) else math.nan,
                "exact_two_sided_sign_flip_p": _exact_paired_sign_flip(differences),
                "test_note": "architecture-level paired sign-flip test; exploratory with only four architectures",
            }
        )
    return pd.DataFrame(rows)


def run_analysis(args: argparse.Namespace) -> dict[str, object]:
    pooled, matrix_paths = read_architecture_matrices(args.matrix)
    shifts = canonicalize_pair_columns(pd.read_csv(args.shift_table))
    expected_architectures = set(args.expected_architectures)
    found_architectures = set(matrix_paths)
    if expected_architectures and found_architectures != expected_architectures:
        raise ValueError(
            f"Expected architectures {sorted(expected_architectures)}, found {sorted(found_architectures)}"
        )
    expected_cells = len(args.datasets) ** 2
    counts = pooled.groupby("architecture").size().to_dict()
    incorrect = {name: count for name, count in counts.items() if count != expected_cells}
    if incorrect:
        raise ValueError(f"Expected {expected_cells} cells per architecture, found {incorrect}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(args.output_dir / "pooled_five_label_results.csv", index=False)
    off = add_gaps_and_shifts(pooled, shifts)
    off.to_csv(args.output_dir / "all_architectures_gap_shift_components.csv", index=False)

    score, components = composite_summary(pooled, shifts)
    score.to_csv(args.output_dir / "composite_score_summary.csv", index=False)
    components.to_csv(args.output_dir / "composite_score_components.csv", index=False)

    groups = shift_group_table(off)
    groups.to_csv(args.output_dir / "per_shift_vector_mean_gap.csv", index=False)
    attribution = descriptive_attribution(off)
    attribution.to_csv(args.output_dir / "descriptive_shift_attribution_profile.csv", index=False)

    sweep, sweep_summary = lambda_sensitivity(pooled, shifts, args.lambda_grid)
    sweep.to_csv(args.output_dir / "lambda_weight_sweep.csv", index=False)
    sweep_summary.to_csv(args.output_dir / "lambda_weight_sweep_kendall_tau.csv", index=False)

    paired = paired_shift_vector_tests(groups)
    paired.to_csv(args.output_dir / "paired_shift_vector_tests.csv", index=False)

    bootstrap = bootstrap_all_cells(
        matrix_paths,
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    if len(bootstrap):
        bootstrap.to_csv(args.output_dir / "bootstrap_95ci_all_cells.csv", index=False)

    summary: dict[str, object] = {
        "status": "COMPLETE",
        "architectures": sorted(matrix_paths),
        "datasets": list(args.datasets),
        "matrix_cells_per_architecture": expected_cells,
        "off_diagonal_cells_per_architecture": len(args.datasets) * (len(args.datasets) - 1),
        "bootstrap_replicates": args.bootstrap_replicates,
        "outputs": sorted(path.name for path in args.output_dir.glob("*.csv")),
        "attribution_warning": (
            "The shift attribution profile is a descriptive proposal-weight allocation, not a causal estimate."
        ),
    }
    (args.output_dir / "week3_analysis_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", action="append", default=[], metavar="ARCHITECTURE=CSV")
    parser.add_argument("--shift-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ptbxl", "cpsc2018", "georgia", "mimic_iv", "code_ii"],
    )
    parser.add_argument(
        "--expected-architectures",
        nargs="+",
        default=["inception_time", "resnet1d", "transformer", "ecg_fm"],
    )
    parser.add_argument("--lambda-grid", nargs="+", type=float, default=list(DEFAULT_LAMBDA_GRID))
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_analysis(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
