from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.build_mimic_subset import (
    DEFAULT_LABEL_COLUMNS,
    ColumnSpec,
    derive_labels_from_long_table,
    main,
    run_pipeline,
    validate_metadata,
)


def make_metadata(n_patients: int = 240, seed: int = 7) -> pd.DataFrame:
    """Create correlated, multilabel records with unequal patient group sizes."""

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    study_id = 10_000
    probabilities = np.array([0.58, 0.16, 0.09, 0.055, 0.075])
    for subject_id in range(1_000, 1_000 + n_patients):
        n_recordings = int(rng.integers(1, 6))
        patient_risk = rng.normal(0.0, 0.035, len(probabilities))
        for recording_index in range(n_recordings):
            labels = rng.random(len(probabilities)) < np.clip(
                probabilities + patient_risk, 0.005, 0.95
            )
            # Introduce realistic multilabel correlation.
            if labels[1] and rng.random() < 0.2:
                labels[4] = True
            rows.append(
                {
                    "subject_id": subject_id,
                    "study_id": study_id,
                    "waveform_path": f"p{subject_id}/s{study_id}",
                    **{
                        label: int(value)
                        for label, value in zip(DEFAULT_LABEL_COLUMNS, labels, strict=True)
                    },
                }
            )
            study_id += 1
    return pd.DataFrame(rows)


def run_synthetic(metadata: pd.DataFrame, seed: int = 42):
    return run_pipeline(
        metadata,
        columns=ColumnSpec(),
        label_cols=DEFAULT_LABEL_COLUMNS,
        target_recordings=400,
        seed=seed,
        mapping_version="test-v1",
        split_ratios={"train": 0.8, "validation": 0.1, "test": 0.1},
    )


def test_selection_and_splits_are_deterministic() -> None:
    metadata = make_metadata()
    first = run_synthetic(metadata, seed=42).manifest
    second = run_synthetic(metadata, seed=42).manifest

    pd.testing.assert_frame_equal(first, second)


def test_patients_never_leak_across_splits() -> None:
    manifest = run_synthetic(make_metadata()).manifest

    assert manifest.groupby("subject_id")["split"].nunique().max() == 1
    assert set(manifest["split"]) == {"train", "validation", "test"}
    split_positive_counts = manifest.groupby("split")[list(DEFAULT_LABEL_COLUMNS)].sum()
    assert split_positive_counts.gt(0).all().all()


def test_split_record_ratios_are_as_close_as_patient_grouping_allows() -> None:
    metadata = make_metadata()
    manifest = run_synthetic(metadata).manifest
    split_counts = manifest["split"].value_counts()
    targets = {"train": 320, "validation": 40, "test": 40}
    largest_patient = metadata.groupby("subject_id").size().max()

    for split, target in targets.items():
        assert abs(int(split_counts[split]) - target) < largest_patient


def test_subset_is_close_and_preserves_multilabel_prevalence() -> None:
    metadata = make_metadata()
    result = run_synthetic(metadata)
    manifest = result.manifest

    largest_patient = metadata.groupby("subject_id").size().max()
    assert abs(len(manifest) - 400) < largest_patient

    full_prevalence = metadata[list(DEFAULT_LABEL_COLUMNS)].mean()
    subset_prevalence = manifest[list(DEFAULT_LABEL_COLUMNS)].mean()
    assert (subset_prevalence - full_prevalence).abs().max() < 0.04


def test_unusable_study_exclusions_happen_before_patient_sampling() -> None:
    metadata = make_metadata()
    excluded_studies = {
        str(value) for value in metadata.loc[metadata["subject_id"] == 1000, "study_id"]
    }

    result = run_pipeline(
        metadata,
        columns=ColumnSpec(),
        label_cols=DEFAULT_LABEL_COLUMNS,
        target_recordings=400,
        seed=42,
        mapping_version="test-v1",
        split_ratios={"train": 0.8, "validation": 0.1, "test": 0.1},
        exclude_study_ids=excluded_studies,
    )

    assert not set(result.manifest["study_id"].astype(str)) & excluded_studies
    assert result.cleaning_counts["excluded_unusable_studies"] == len(excluded_studies)


def test_validation_logs_filter_counts_and_keeps_all_zero_vectors() -> None:
    metadata = make_metadata(n_patients=8)
    all_zero = metadata.iloc[[0]].copy()
    all_zero["study_id"] = 999_001
    all_zero[list(DEFAULT_LABEL_COLUMNS)] = 0
    missing_path = metadata.iloc[[1]].copy()
    missing_path["study_id"] = 999_002
    missing_path["waveform_path"] = ""
    incomplete = metadata.iloc[[2]].copy()
    incomplete["study_id"] = 999_003
    incomplete[DEFAULT_LABEL_COLUMNS[0]] = pd.NA
    exact_duplicate = metadata.iloc[[3]].copy()
    frame = pd.concat(
        [metadata, all_zero, missing_path, incomplete, exact_duplicate], ignore_index=True
    )

    eligible, counts = validate_metadata(
        frame, columns=ColumnSpec(), label_cols=DEFAULT_LABEL_COLUMNS
    )

    assert counts["dropped_missing_waveform_paths"] == 1
    assert counts["dropped_incomplete_labels"] == 1
    assert counts["dropped_duplicate_study_rows"] == 1
    assert 999_001 in set(eligible["study_id"])


def test_conflicting_duplicate_study_raises() -> None:
    metadata = make_metadata(n_patients=3)
    conflict = metadata.iloc[[0]].copy()
    conflict["waveform_path"] = "different/path"
    frame = pd.concat([metadata, conflict], ignore_index=True)

    with pytest.raises(ValueError, match="conflicting"):
        validate_metadata(
            frame, columns=ColumnSpec(), label_cols=DEFAULT_LABEL_COLUMNS
        )


def test_long_diagnosis_mapping_is_configurable() -> None:
    diagnoses = pd.DataFrame(
        {
            "study_id": [1, 1, 2, 3],
            "code": ["N", "AF", "RBBB", "UNMAPPED"],
        }
    )
    mapping = pd.DataFrame(
        {
            "source_code": ["N", "AF", "RBBB"],
            "target_label": ["normal", "af_afl", "rbbb"],
        }
    )

    wide = derive_labels_from_long_table(
        diagnoses,
        mapping,
        study_id_col="study_id",
        diagnosis_code_col="code",
        map_code_col="source_code",
        map_label_col="target_label",
        label_cols=DEFAULT_LABEL_COLUMNS,
    ).set_index("study_id")

    assert wide.loc[1, "normal"] == 1
    assert wide.loc[1, "af_afl"] == 1
    assert wide.loc[2, "rbbb"] == 1
    assert wide.loc[3, list(DEFAULT_LABEL_COLUMNS)].sum() == 0


def test_cli_writes_outputs_and_dry_run_writes_nothing(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    metadata_path = tmp_path / "metadata.csv"
    output_path = tmp_path / "manifest.csv"
    qc_path = tmp_path / "qc.csv"
    make_metadata(n_patients=60).to_csv(metadata_path, index=False)

    exit_code = main(
        [
            "--metadata",
            str(metadata_path),
            "--output",
            str(output_path),
            "--qc-output",
            str(qc_path),
            "--target-recordings",
            "100",
            "--mapping-version",
            "test-cli-v1",
        ]
    )
    assert exit_code == 0
    assert output_path.is_file()
    assert qc_path.is_file()
    manifest = pd.read_csv(output_path)
    qc = pd.read_csv(qc_path)
    assert list(manifest.columns) == [
        "subject_id",
        "study_id",
        "waveform_path",
        *DEFAULT_LABEL_COLUMNS,
        "split",
        "seed",
        "mapping_version",
    ]
    assert set(qc["section"]) == {"summary", "label_prevalence", "split"}

    dry_manifest = tmp_path / "dry_manifest.csv"
    dry_qc = tmp_path / "dry_qc.csv"
    dry_exit_code = main(
        [
            "--metadata",
            str(metadata_path),
            "--output",
            str(dry_manifest),
            "--qc-output",
            str(dry_qc),
            "--target-recordings",
            "100",
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()
    assert dry_exit_code == 0
    assert "Eligible:" in captured.out
    assert not dry_manifest.exists()
    assert not dry_qc.exists()
