#!/usr/bin/env python3
"""Merge the four independently produced binary-ablation matrices.

The four Colab worker notebooks intentionally write to separate directories so
they cannot overwrite one another.  This command is the final CPU-only merge:
it refuses partial or duplicate matrices, compares every binary result with the
matching five-label matrix, and emits the four-row result table requested by
the reviewer.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.evaluation.binary_matrix import compare_gaps, generalization_gap
from src.evaluation.ecg_fm_matrix import DEFAULT_DATASETS, parse_named_paths
from src.training.binary_ablation_pipeline import ARCHITECTURES


def _read_complete_matrix(path: Path, architecture: str, datasets: Sequence[str]) -> pd.DataFrame:
    rows = pd.read_csv(path)
    required = {"architecture", "source_dataset", "target_dataset", "status", "test_auroc"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    rows["architecture"] = rows["architecture"].astype(str).str.lower()
    rows = rows.loc[rows["architecture"].eq(architecture)].copy()
    expected = {(source, target) for source in datasets for target in datasets}
    observed = set(zip(rows["source_dataset"], rows["target_dataset"]))
    if rows.duplicated(["source_dataset", "target_dataset"]).any():
        raise ValueError(f"{path} contains duplicate source-target cells")
    if observed != expected:
        missing_cells = sorted(expected - observed)
        extra_cells = sorted(observed - expected)
        raise ValueError(
            f"{path} is not a complete {len(datasets)}x{len(datasets)} matrix; "
            f"missing={missing_cells}, extra={extra_cells}"
        )
    blocked = rows.loc[~rows["status"].eq("COMPLETE")]
    if len(blocked):
        raise ValueError(f"{path} contains {len(blocked)} incomplete cells")
    if pd.to_numeric(rows["test_auroc"], errors="coerce").isna().any():
        raise ValueError(f"{path} contains nonnumeric AUROC values")
    return rows


def _cell_bootstrap_gap_ci(
    rows: pd.DataFrame, *, metric: str, replicates: int, seed: int
) -> tuple[float, float]:
    diagonal = rows.loc[
        rows["source_dataset"].eq(rows["target_dataset"]), metric
    ].to_numpy(float)
    cross = rows.loc[
        rows["source_dataset"].ne(rows["target_dataset"]), metric
    ].to_numpy(float)
    rng = np.random.default_rng(seed)
    gaps = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled_diagonal = rng.choice(diagonal, size=len(diagonal), replace=True)
        sampled_cross = rng.choice(cross, size=len(cross), replace=True)
        gaps[index] = sampled_diagonal.mean() - sampled_cross.mean()
    return tuple(map(float, np.quantile(gaps, [0.025, 0.975])))


def merge(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    datasets = tuple(args.datasets)
    binary_paths = parse_named_paths(args.binary_matrix)
    five_paths = parse_named_paths(args.five_label_matrix)
    missing_architectures = sorted(set(ARCHITECTURES) - set(binary_paths))
    if missing_architectures:
        raise ValueError(f"Missing binary matrices for: {missing_architectures}")
    missing_five = sorted(set(ARCHITECTURES) - set(five_paths))
    if missing_five:
        raise ValueError(f"Missing five-label matrices for: {missing_five}")

    all_binary: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for architecture_index, architecture in enumerate(ARCHITECTURES):
        binary = _read_complete_matrix(binary_paths[architecture], architecture, datasets)
        five = pd.read_csv(five_paths[architecture])
        if "architecture" in five.columns:
            five["architecture"] = (
                five["architecture"]
                .astype(str)
                .str.lower()
                .str.replace(r"[^a-z0-9]+", "_", regex=True)
                .str.strip("_")
            )
            five = five.loc[five["architecture"].eq(architecture)]
        five_metric = "macro_auroc"
        five_required = {"source_dataset", "target_dataset", "status", five_metric}
        if five_required - set(five.columns):
            raise ValueError(f"{five_paths[architecture]} is not a long-form five-label matrix")
        if len(five) != len(datasets) ** 2 or not five["status"].eq("COMPLETE").all():
            raise ValueError(f"{five_paths[architecture]} is not a complete five-label matrix")

        binary_summary = generalization_gap(binary, "test_auroc")
        five_summary = generalization_gap(five, five_metric)
        binary_ci = _cell_bootstrap_gap_ci(
            binary,
            metric="test_auroc",
            replicates=args.bootstrap_replicates,
            seed=args.seed + architecture_index,
        )
        five_ci = _cell_bootstrap_gap_ci(
            five,
            metric=five_metric,
            replicates=args.bootstrap_replicates,
            seed=args.seed + 100 + architecture_index,
        )
        summaries.append(
            {
                "architecture": architecture,
                **{f"binary_{key}": value for key, value in binary_summary.items()},
                "binary_gap_cell_bootstrap_ci_95_lower": binary_ci[0],
                "binary_gap_cell_bootstrap_ci_95_upper": binary_ci[1],
                **{f"five_label_{key}": value for key, value in five_summary.items()},
                "five_label_gap_cell_bootstrap_ci_95_lower": five_ci[0],
                "five_label_gap_cell_bootstrap_ci_95_upper": five_ci[1],
                **compare_gaps(
                    float(five_summary["generalization_gap"]),
                    float(binary_summary["generalization_gap"]),
                ),
            }
        )
        all_binary.append(binary)

    combined = pd.concat(all_binary, ignore_index=True)
    summary = pd.DataFrame(summaries)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output_dir / "binary_matrix_long_all_architectures.csv", index=False)
    summary.to_csv(args.output_dir / "two_class_gap_summary_all_architectures.csv", index=False)
    audit = {
        "status": "COMPLETE",
        "architectures": list(ARCHITECTURES),
        "datasets": list(datasets),
        "completed_binary_cells": int(len(combined)),
        "expected_binary_cells": len(ARCHITECTURES) * len(datasets) ** 2,
        "bootstrap_unit": "source-target cells; descriptive, not independent training replicates",
        "bootstrap_replicates": args.bootstrap_replicates,
    }
    (args.output_dir / "binary_ablation_merge_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    return combined, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary-matrix", action="append", default=[], metavar="ARCH=PATH")
    parser.add_argument("--five-label-matrix", action="append", default=[], metavar="ARCH=PATH")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.bootstrap_replicates <= 0:
        raise ValueError("--bootstrap-replicates must be positive")
    _, summary = merge(args)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
