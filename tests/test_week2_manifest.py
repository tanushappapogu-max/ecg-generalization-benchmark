import pandas as pd

from src.data.week2_manifest import (
    LABEL_COLUMNS,
    build_cpsc_manifest,
    build_mimic_manifest,
    build_ptbxl_manifest,
    manifest_qc,
)


def _assert_manifest(frame, expected_size):
    assert len(frame) == expected_size
    assert set(frame["split"]) == {"train", "validation", "test"}
    assert frame.groupby("subject_id")["split"].nunique().max() == 1
    assert set(LABEL_COLUMNS).issubset(frame.columns)
    assert manifest_qc(frame)["status"] == "PASS"


def test_ptbxl_uses_official_folds_and_patient_groups():
    rows = []
    database = []
    target_codes = ["NORM", "AFIB", "1AVB", "CLBBB", "CRBBB"]
    for fold in range(1, 11):
        for variant in range(2):
            ecg_id = fold * 10 + variant
            codes = {code: 100.0 for i, code in enumerate(target_codes) if (i + variant) % 2 == 0}
            rows.append({"ecg_id": ecg_id, "raw_labels": repr(codes)})
            database.append(
                {"ecg_id": ecg_id, "patient_id": f"p{fold}-{variant}", "strat_fold": fold}
            )
    manifest = build_ptbxl_manifest(pd.DataFrame(rows), pd.DataFrame(database))
    _assert_manifest(manifest, 20)
    assert set(manifest.loc[manifest["split"].eq("validation"), "record_id"]) == {"90", "91"}
    assert set(manifest.loc[manifest["split"].eq("test"), "record_id"]) == {"100", "101"}


def test_cpsc_mapping_and_split_are_deterministic(tmp_path):
    mapping = pd.DataFrame(
        {
            "source_code": ["n", "af", "av", "l", "r"],
            "target_label": list(LABEL_COLUMNS),
            "mapping_version": ["test-v1"] * 5,
        }
    )
    mapping_path = tmp_path / "mapping.csv"
    mapping.to_csv(mapping_path, index=False)
    rows = []
    codes = ["n", "af", "av", "l", "r"]
    for index in range(120):
        selected = [code for offset, code in enumerate(codes) if (index + offset) % 3 == 0]
        rows.append({"ecg_id": f"A{index:04d}", "raw_labels": ",".join(selected)})
    source = pd.DataFrame(rows)
    first = build_cpsc_manifest(source, mapping_path=mapping_path, seed=42)
    second = build_cpsc_manifest(source, mapping_path=mapping_path, seed=42)
    _assert_manifest(first, 120)
    assert first[["record_id", "split"]].equals(second[["record_id", "split"]])


def test_mimic_preserves_existing_patient_split():
    rows = []
    for split_index, split in enumerate(("train", "validation", "test")):
        for variant in range(2):
            labels = {label: int((i + variant) % 2 == 0) for i, label in enumerate(LABEL_COLUMNS)}
            rows.append(
                {
                    "subject_id": f"p{split_index}-{variant}",
                    "study_id": f"s{split_index}-{variant}",
                    "waveform_path": f"files/p{split_index}/s{variant}/record",
                    "split": split,
                    **labels,
                }
            )
    manifest = build_mimic_manifest(pd.DataFrame(rows))
    _assert_manifest(manifest, 6)
    assert manifest.set_index("record_id").loc["s1-0", "split"] == "validation"
