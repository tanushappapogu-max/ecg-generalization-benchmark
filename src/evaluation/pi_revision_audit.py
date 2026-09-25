#!/usr/bin/env python3
"""Build auditable tables for the PI-requested post-hoc analyses.

This script does not train or rescore models. It derives directional
normal-versus-abnormal gap changes from the fixed saved predictions, extracts
the ECG-FM source-to-MIMIC per-class results, and summarizes the exact training
label prevalence and available acquisition metadata for each benchmark source.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
LABEL_COLUMNS = ("normal", "af_afl", "av_block_1", "lbbb", "rbbb")
DATASETS = ("ptbxl", "cpsc2018", "georgia", "mimic_iv", "code_ii")
DATASET_DISPLAY = {
    "ptbxl": "PTB-XL",
    "cpsc2018": "CPSC2018",
    "georgia": "Georgia 12-Lead",
    "mimic_iv": "MIMIC-IV-ECG",
    "code_ii": "CODE-15%",
}
DISPLAY_TO_ID = {value: key for key, value in DATASET_DISPLAY.items()}


def directional_binary_gap_changes(binary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = binary.loc[binary["aggregation"].eq("max_abnormal")].copy()
    key = ["architecture", "source_dataset", "target_dataset"]
    if primary.duplicated(key).any():
        raise ValueError("Primary binary table contains duplicate architecture/source/target cells")
    if len(primary) != 100:
        raise ValueError(f"Expected 100 primary binary cells, found {len(primary)}")

    diagonal = primary.loc[
        primary["source_dataset"].eq(primary["target_dataset"])
    ].set_index(["architecture", "source_dataset"])
    if len(diagonal) != 20:
        raise ValueError(f"Expected 20 diagonal cells, found {len(diagonal)}")

    rows: list[dict[str, object]] = []
    external = primary.loc[~primary["source_dataset"].eq(primary["target_dataset"])]
    for row in external.itertuples(index=False):
        base = diagonal.loc[(row.architecture, row.source_dataset)]
        five_gap = float(
            base["five_label_macro_auroc_binary_eligible_records"]
            - row.five_label_macro_auroc_binary_eligible_records
        )
        binary_gap = float(base["binary_auroc"] - row.binary_auroc)
        reduction = five_gap - binary_gap
        rows.append(
            {
                "architecture": row.architecture,
                "source_dataset": row.source_dataset,
                "target_dataset": row.target_dataset,
                "matched_five_label_source_diagonal_auroc": float(
                    base["five_label_macro_auroc_binary_eligible_records"]
                ),
                "matched_five_label_external_auroc": float(
                    row.five_label_macro_auroc_binary_eligible_records
                ),
                "matched_five_label_gap": five_gap,
                "binary_source_diagonal_auroc": float(base["binary_auroc"]),
                "binary_external_auroc": float(row.binary_auroc),
                "binary_gap": binary_gap,
                "absolute_gap_reduction": reduction,
                "gap_direction": "reduced" if reduction > 0 else "increased" if reduction < 0 else "unchanged",
            }
        )
    cell = pd.DataFrame(rows).sort_values(key, kind="stable").reset_index(drop=True)
    if len(cell) != 80:
        raise ValueError(f"Expected 80 off-diagonal cells, found {len(cell)}")

    pair = (
        cell.groupby(["source_dataset", "target_dataset"], sort=True)
        .agg(
            architecture_count=("architecture", "size"),
            mean_matched_five_label_gap=("matched_five_label_gap", "mean"),
            mean_binary_gap=("binary_gap", "mean"),
            mean_absolute_gap_reduction=("absolute_gap_reduction", "mean"),
            sd_absolute_gap_reduction=("absolute_gap_reduction", "std"),
            min_absolute_gap_reduction=("absolute_gap_reduction", "min"),
            max_absolute_gap_reduction=("absolute_gap_reduction", "max"),
            architectures_with_reduced_gap=(
                "absolute_gap_reduction",
                lambda values: int((values > 0).sum()),
            ),
        )
        .reset_index()
    )
    if len(pair) != 20 or not pair["architecture_count"].eq(4).all():
        raise ValueError("Expected 20 directional pairs with four architectures each")
    pair["pair_direction"] = np.select(
        [pair["mean_absolute_gap_reduction"] > 0, pair["mean_absolute_gap_reduction"] < 0],
        ["reduced", "increased"],
        default="unchanged",
    )
    return cell, pair


def mimic_per_class(mapping: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "source_dataset",
        "record_count",
        "macro_auroc",
        "auroc_normal",
        "auroc_af_afl",
        "auroc_av_block_1",
        "auroc_lbbb",
        "auroc_rbbb",
    ]
    result = mapping.loc[
        mapping["mapping_variant"].eq("consensus_five_class")
        & mapping["architecture"].eq("ecg_fm")
        & mapping["target_dataset"].eq("mimic_iv"),
        columns,
    ].copy()
    result["target_dataset"] = "mimic_iv"
    result = result[["source_dataset", "target_dataset", *columns[1:]]]
    result = result.set_index("source_dataset").reindex(DATASETS).reset_index()
    if result["macro_auroc"].isna().any() or len(result) != 5:
        raise ValueError("ECG-FM source-to-MIMIC audit is incomplete")
    return result


def label_prevalence() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        path = ROOT / "data" / "week2" / f"{dataset}_week2.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing canonical manifest: {path}")
        frame = pd.read_csv(path, usecols=["split", *LABEL_COLUMNS])
        train = frame.loc[frame["split"].eq("train")]
        test = frame.loc[frame["split"].eq("test")]
        if train.empty or test.empty:
            raise ValueError(f"Manifest {path} has an empty train or test split")
        row: dict[str, object] = {
            "dataset": dataset,
            "display_name": DATASET_DISPLAY[dataset],
            "train_record_count": int(len(train)),
            "test_record_count": int(len(test)),
        }
        for label in LABEL_COLUMNS:
            row[f"train_prevalence_{label}"] = float(train[label].mean())
            row[f"test_prevalence_{label}"] = float(test[label].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def acquisition_metadata() -> pd.DataFrame:
    path = ROOT / "data" / "shift_metadata" / "demographic_counts.csv"
    frame = pd.read_csv(path)
    result = frame.loc[
        :,
        [
            "dataset",
            "sampling_rate_hz",
            "hardware_group",
            "hardware_documented",
            "label_schema",
            "metadata_basis",
            "metadata_source_url",
        ],
    ].copy()
    result["dataset_id"] = result["dataset"].map(DISPLAY_TO_ID)
    if result["dataset_id"].isna().any():
        raise ValueError("Acquisition metadata contains an unrecognized dataset name")
    return result.set_index("dataset_id").reindex(DATASETS).reset_index()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "analysis_outputs" / "pi_revision",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    binary = pd.read_csv(
        ROOT / "analysis_outputs" / "reviewer_posthoc_binary" / "posthoc_binary_matrix_long.csv"
    )
    mapping = pd.read_csv(
        ROOT / "analysis_outputs" / "reviewer_posthoc_mapping" / "posthoc_mapping_matrix_long.csv"
    )
    cells, pairs = directional_binary_gap_changes(binary)
    mimic = mimic_per_class(mapping)
    prevalence = label_prevalence()
    acquisition = acquisition_metadata()

    cells.to_csv(args.output_dir / "normal_abnormal_cell_gap_changes.csv", index=False)
    pairs.to_csv(args.output_dir / "normal_abnormal_pair_summary.csv", index=False)
    mimic.to_csv(args.output_dir / "mimic_ecg_fm_per_class.csv", index=False)
    prevalence.to_csv(args.output_dir / "label_prevalence_by_split.csv", index=False)
    acquisition.to_csv(args.output_dir / "acquisition_metadata.csv", index=False)

    audit = {
        "status": "PARTIAL_PI_REVISION",
        "completed_from_saved_artifacts": {
            "normal_abnormal_directional_cells": int(len(cells)),
            "normal_abnormal_directional_pairs": int(len(pairs)),
            "mimic_ecg_fm_source_cells": int(len(mimic)),
            "label_prevalence_datasets": int(len(prevalence)),
            "acquisition_metadata_datasets": int(len(acquisition)),
        },
        "directional_binary_finding": {
            "cells_with_reduced_gap": int(cells["absolute_gap_reduction"].gt(0).sum()),
            "cells_with_increased_gap": int(cells["absolute_gap_reduction"].lt(0).sum()),
            "pairs_with_reduced_mean_gap": int(pairs["mean_absolute_gap_reduction"].gt(0).sum()),
            "pairs_with_increased_mean_gap": int(pairs["mean_absolute_gap_reduction"].lt(0).sum()),
        },
        "not_completed_by_this_script": [
            "cross-dataset waveform duplicate and near-duplicate removal audit",
            "second non-overlapping foundation-model training and 5x5 evaluation",
            "native-label remapping experiment that recovers diagnoses excluded before training",
        ],
        "warnings": [
            "Normal-versus-abnormal results collapse fixed saved predictions; no model was retrained.",
            "Available hardware metadata are incomplete for MIMIC-IV-ECG, CPSC2018, and Georgia.",
            "The existing endpoint-mapping analysis cannot recover source-native diagnoses excluded before training.",
        ],
    }
    (args.output_dir / "audit_summary.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
