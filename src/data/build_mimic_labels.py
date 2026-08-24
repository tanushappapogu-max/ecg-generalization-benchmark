#!/usr/bin/env python3
"""Derive the frozen five MIMIC labels with ECG-FM's machine-report labeler.

This adapter intentionally keeps the source-label mapping in a CSV. The team
froze ``Sinus rhythm -> normal`` under mapping version
``ecg-fm-machine-report-v1``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd

try:
    from src.data.build_mimic_subset import DEFAULT_LABEL_COLUMNS
except ModuleNotFoundError:  # support ``python src/data/build_mimic_labels.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.data.build_mimic_subset import DEFAULT_LABEL_COLUMNS


def load_mapping(path: Path) -> tuple[dict[str, tuple[str, ...]], str]:
    mapping = pd.read_csv(path)
    required = {"source_label", "target_label", "mapping_version"}
    missing = sorted(required - set(mapping.columns))
    if missing:
        raise ValueError(f"MIMIC label mapping is missing columns: {missing}")
    invalid_targets = sorted(set(mapping["target_label"]) - set(DEFAULT_LABEL_COLUMNS))
    if invalid_targets:
        raise ValueError(f"Unknown target labels in mapping: {invalid_targets}")
    versions = mapping["mapping_version"].dropna().unique()
    if len(versions) != 1:
        raise ValueError("Exactly one mapping_version is required")
    grouped = (
        mapping.groupby("source_label", sort=False)["target_label"]
        .agg(lambda values: tuple(dict.fromkeys(values)))
        .to_dict()
    )
    return grouped, str(versions[0])


def load_ecg_fm_labeler(ecg_fm_root: Path) -> tuple[Any, Callable[[pd.Series], pd.Series]]:
    """Load the official ECG-FM pattern labeler from a local checkout."""

    python_dir = ecg_fm_root / "labeler"
    config_dir = ecg_fm_root / "data" / "mimic_iv_ecg" / "labeler"
    if not python_dir.is_dir() or not config_dir.is_dir():
        raise FileNotFoundError(
            f"Could not find ECG-FM labeler code/config under {ecg_fm_root}"
        )
    sys.path.insert(0, str(python_dir))
    try:
        from pattern_labeler import PatternLabeler, PatternLabelerConfig
        from preprocess import preprocess_texts
    except ImportError as exc:
        raise RuntimeError(
            "Could not import ECG-FM labeler. Its runtime requires pandas, numpy, "
            "matplotlib, networkx, and tqdm."
        ) from exc

    config = PatternLabelerConfig.from_json(str(config_dir), progress=False)
    return PatternLabeler(config), preprocess_texts


def combine_report_columns(chunk: pd.DataFrame, report_columns: Sequence[str]) -> pd.Series:
    """Combine MIMIC ``report_0...report_17`` into one interpretation per ECG."""

    report_frame = chunk[list(report_columns)].astype("string").fillna("")
    for column in report_columns:
        report_frame[column] = report_frame[column].str.strip()
    texts = report_frame.agg(
        lambda values: "; ".join(value for value in values if value), axis=1
    )
    return texts.astype("string").str.strip()


def label_texts(
    texts: pd.Series,
    *,
    labeler: Any,
    preprocess_texts: Callable[[pd.Series], pd.Series],
    source_to_targets: dict[str, tuple[str, ...]],
) -> pd.DataFrame:
    """Apply the official labeler and reduce source labels into five binary columns."""

    local_texts = texts.reset_index(drop=True)
    processed = preprocess_texts(local_texts.copy())
    result = labeler(texts=processed)
    labels_flat = result.labels_flat
    output = pd.DataFrame(0, index=local_texts.index, columns=DEFAULT_LABEL_COLUMNS)
    if labels_flat is not None and not labels_flat.empty:
        for source_label, target_labels in source_to_targets.items():
            positive_indices = labels_flat.index[
                labels_flat["name"].eq(source_label)
            ].unique()
            valid_indices = positive_indices.intersection(output.index)
            for target_label in target_labels:
                output.loc[valid_indices, target_label] = 1
    return output.astype("int8")


def build_labels(
    machine_measurements_path: Path,
    *,
    ecg_fm_root: Path,
    mapping_path: Path,
    chunk_size: int = 20_000,
    skip_rows: int = 0,
    limit: int | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Stream MIMIC reports and return one usable-label row per study."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if skip_rows < 0:
        raise ValueError("skip_rows cannot be negative")
    source_to_targets, mapping_version = load_mapping(mapping_path)
    labeler, preprocess_texts = load_ecg_fm_labeler(ecg_fm_root)

    columns = pd.read_csv(machine_measurements_path, nrows=0).columns.tolist()
    if "study_id" not in columns:
        raise ValueError("machine_measurements.csv is missing study_id")
    report_columns = [column for column in columns if column.startswith("report_")]
    if not report_columns:
        raise ValueError("No report_* columns found in machine_measurements.csv")

    frames: list[pd.DataFrame] = []
    source_rows = 0
    usable_rows = 0
    remaining = limit
    reader = pd.read_csv(
        machine_measurements_path,
        usecols=["study_id", *report_columns],
        chunksize=chunk_size,
        low_memory=False,
        skiprows=range(1, skip_rows + 1) if skip_rows else None,
    )
    for chunk in reader:
        if remaining is not None:
            if remaining <= 0:
                break
            chunk = chunk.iloc[:remaining].copy()
            remaining -= len(chunk)
        source_rows += len(chunk)
        texts = combine_report_columns(chunk, report_columns)
        usable_mask = texts.ne("")
        usable_chunk = chunk.loc[usable_mask, ["study_id"]].reset_index(drop=True)
        usable_texts = texts.loc[usable_mask].reset_index(drop=True)
        usable_rows += len(usable_chunk)
        if usable_chunk.empty:
            continue
        binary = label_texts(
            usable_texts,
            labeler=labeler,
            preprocess_texts=preprocess_texts,
            source_to_targets=source_to_targets,
        )
        frames.append(
            pd.concat([usable_chunk, binary], axis=1).assign(
                mapping_version=mapping_version
            )
        )

    if not frames:
        raise ValueError("No usable machine reports were found")
    labels = pd.concat(frames, ignore_index=True)
    if labels["study_id"].duplicated().any():
        raise ValueError("machine_measurements.csv contains duplicate study_id rows")
    summary: dict[str, object] = {
        "source_rows": source_rows,
        "skipped_source_rows": skip_rows,
        "usable_report_rows": usable_rows,
        "dropped_blank_reports": source_rows - usable_rows,
        "mapping_version": mapping_version,
        "prevalence": {
            label: float(labels[label].mean()) for label in DEFAULT_LABEL_COLUMNS
        },
    }
    return labels, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine-measurements", type=Path, required=True)
    parser.add_argument("--ecg-fm-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("data/label_mappings/ecg_fm_machine_report_five_labels.csv"),
    )
    parser.add_argument("--chunk-size", type=int, default=20_000)
    parser.add_argument(
        "--skip-rows",
        type=int,
        default=0,
        help="Skip this many data rows while retaining the CSV header.",
    )
    parser.add_argument("--limit", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    labels, summary = build_labels(
        args.machine_measurements,
        ecg_fm_root=args.ecg_fm_root,
        mapping_path=args.mapping,
        chunk_size=args.chunk_size,
        skip_rows=args.skip_rows,
        limit=args.limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(args.output, index=False)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
