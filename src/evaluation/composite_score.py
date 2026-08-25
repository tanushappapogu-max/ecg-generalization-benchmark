"""Composite cross-dataset score from the ECG benchmark proposal."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


PAIR_COLUMNS = ("source_dataset", "target_dataset")
SHIFT_COLUMNS = ("PS", "DS", "LS")


def compute_composite_score(
    matrix_rows: pd.DataFrame,
    shift_rows: pd.DataFrame,
    *,
    lambda_ps: float = 1.0,
    lambda_ds: float = 2.0,
    lambda_ls: float = 3.0,
) -> tuple[float, pd.DataFrame]:
    """Return proposal Score and the auditable off-diagonal component table.

    Proposal definitions:
    ``delta_ij = AUROC_ii - AUROC_ij``;
    ``w_ij = lambda_ps*PS + lambda_ds*DS + lambda_ls*LS``; and
    ``Score = CD * (1 - mean(delta_ij / (w_ij + 1)))``, clipped to [0, 1].
    """

    if min(lambda_ps, lambda_ds, lambda_ls) < 0:
        raise ValueError("Shift weights must be non-negative")
    required_matrix = {*PAIR_COLUMNS, "macro_auroc"}
    required_shifts = {*PAIR_COLUMNS, *SHIFT_COLUMNS}
    if missing := sorted(required_matrix - set(matrix_rows.columns)):
        raise ValueError(f"Matrix is missing columns: {missing}")
    if missing := sorted(required_shifts - set(shift_rows.columns)):
        raise ValueError(f"Shift table is missing columns: {missing}")
    matrix = matrix_rows.copy()
    if "status" in matrix:
        matrix = matrix.loc[matrix["status"].eq("COMPLETE")].copy()
    if matrix.duplicated(list(PAIR_COLUMNS)).any():
        raise ValueError("Matrix contains duplicate source-target cells")
    shifts = shift_rows.copy()
    if shifts.duplicated(list(PAIR_COLUMNS)).any():
        raise ValueError("Shift table contains duplicate source-target rows")
    for column in SHIFT_COLUMNS:
        values = set(pd.to_numeric(shifts[column], errors="raise").unique())
        if not values.issubset({0, 1}):
            raise ValueError(f"{column} must be binary")

    diagonal = matrix.loc[
        matrix["source_dataset"].eq(matrix["target_dataset"]),
        ["source_dataset", "macro_auroc"],
    ].rename(columns={"macro_auroc": "in_distribution_auroc"})
    if diagonal["source_dataset"].duplicated().any():
        raise ValueError("Matrix contains duplicate diagonal cells")
    off = matrix.loc[
        matrix["source_dataset"].ne(matrix["target_dataset"])
    ].copy()
    components = off.merge(diagonal, on="source_dataset", how="left", validate="many_to_one")
    components = components.merge(
        shifts[list(PAIR_COLUMNS) + list(SHIFT_COLUMNS)],
        on=list(PAIR_COLUMNS),
        how="left",
        validate="one_to_one",
    )
    if components.empty:
        raise ValueError("No completed off-diagonal cells are available")
    if components[["in_distribution_auroc", *SHIFT_COLUMNS]].isna().any().any():
        raise ValueError("Every off-diagonal cell needs a diagonal and shift vector")

    components["delta_ij"] = (
        components["in_distribution_auroc"] - components["macro_auroc"]
    )
    components["w_ij"] = (
        lambda_ps * components["PS"]
        + lambda_ds * components["DS"]
        + lambda_ls * components["LS"]
    )
    components["weighted_gap_term"] = components["delta_ij"] / (
        components["w_ij"] + 1.0
    )
    cross_dataset_mean = float(components["macro_auroc"].mean())
    penalty = float(components["weighted_gap_term"].mean())
    score = float(np.clip(cross_dataset_mean * (1.0 - penalty), 0.0, 1.0))
    components["cross_dataset_mean"] = cross_dataset_mean
    components["penalty_mean"] = penalty
    components["composite_score"] = score
    return score, components


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--shift-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lambda-ps", type=float, default=1.0)
    parser.add_argument("--lambda-ds", type=float, default=2.0)
    parser.add_argument("--lambda-ls", type=float, default=3.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    score, components = compute_composite_score(
        pd.read_csv(args.matrix),
        pd.read_csv(args.shift_table),
        lambda_ps=args.lambda_ps,
        lambda_ds=args.lambda_ds,
        lambda_ls=args.lambda_ls,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    components.to_csv(args.output, index=False)
    print({"score": score, "rows": len(components), "output": str(args.output)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

