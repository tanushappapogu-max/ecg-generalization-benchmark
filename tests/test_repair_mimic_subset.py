from __future__ import annotations

from src.data.build_mimic_subset import DEFAULT_LABEL_COLUMNS, ColumnSpec, run_pipeline
from src.data.repair_mimic_subset import repair_manifest
from tests.test_build_mimic_subset import make_metadata


def test_repair_refills_splits_without_patient_leakage() -> None:
    metadata = make_metadata(n_patients=500)
    original = run_pipeline(
        metadata,
        columns=ColumnSpec(),
        label_cols=DEFAULT_LABEL_COLUMNS,
        target_recordings=400,
        seed=42,
        mapping_version="test-v1",
        split_ratios={"train": 0.8, "validation": 0.1, "test": 0.1},
    ).manifest
    failures = set(
        original.groupby("split", sort=False).head(1)["study_id"].astype(str)
    )
    eligible = metadata.loc[~metadata["study_id"].astype(str).isin(failures)].copy()

    repaired, replacements = repair_manifest(
        original,
        eligible,
        failed_study_ids=failures,
    )

    assert len(repaired) == len(original)
    assert not set(repaired["study_id"].astype(str)) & failures
    assert repaired.groupby("subject_id")["split"].nunique().max() == 1
    assert repaired["split"].value_counts().to_dict() == original[
        "split"
    ].value_counts().to_dict()
    assert not replacements.empty
