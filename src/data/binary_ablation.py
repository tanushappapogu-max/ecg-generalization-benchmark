"""Frozen normal-versus-abnormal label rule for the Week 3 ablation."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.data.week2_manifest import LABEL_COLUMNS, validate_canonical_manifest


BINARY_LABEL_COLUMN = "abnormal"
BINARY_DEFINITION_VERSION = "normal-vs-any-benchmark-abnormal-v1"
ABNORMAL_COLUMNS = tuple(label for label in LABEL_COLUMNS if label != "normal")


def build_binary_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive the frozen binary target without changing source splits.

    ``abnormal=1`` means that at least one of AF/AFL, first-degree AV block,
    LBBB, or RBBB is positive.  ``abnormal=0`` requires the benchmark Normal
    label and no abnormal label.  Records with none of the five benchmark
    labels are excluded.  If Normal co-occurs with an abnormal label, the
    abnormal label wins; this keeps clinically abnormal ECGs in the positive
    class and makes the rule deterministic across datasets.
    """

    canonical = validate_canonical_manifest(frame)
    abnormal = canonical.loc[:, ABNORMAL_COLUMNS].max(axis=1).astype("int8")
    has_any_label = canonical.loc[:, LABEL_COLUMNS].max(axis=1).astype(bool)
    result = canonical.loc[has_any_label].copy()
    result[BINARY_LABEL_COLUMN] = abnormal.loc[has_any_label].to_numpy()
    result["binary_definition_version"] = BINARY_DEFINITION_VERSION
    for split in ("train", "validation", "test"):
        if result.loc[result["split"].eq(split), BINARY_LABEL_COLUMN].nunique() != 2:
            raise ValueError(f"Binary {split} split must contain normal and abnormal records")
    return result.reset_index(drop=True)


def binary_manifest_qc(frame: pd.DataFrame) -> dict[str, Any]:
    binary = build_binary_manifest(frame)
    return {
        "definition_version": BINARY_DEFINITION_VERSION,
        "normal_rule": "normal=1 and no abnormal benchmark label",
        "abnormal_rule": "OR(af_afl, av_block_1, lbbb, rbbb)",
        "normal_plus_abnormal_rule": "abnormal wins",
        "all_zero_rule": "excluded",
        "record_count": int(len(binary)),
        "split_counts": {
            split: int(binary["split"].eq(split).sum())
            for split in ("train", "validation", "test")
        },
        "abnormal_counts": {
            split: int(
                binary.loc[binary["split"].eq(split), BINARY_LABEL_COLUMN].sum()
            )
            for split in ("train", "validation", "test")
        },
    }

