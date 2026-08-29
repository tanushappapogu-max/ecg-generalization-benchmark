import numpy as np
import pandas as pd
import pytest

from src.data.prepare_code15 import build_code15_manifest
from src.data.week2_manifest import LABEL_COLUMNS
from src.training.ecg_fm_pipeline import ECGManifestDataset


def _candidate(rows: int = 600) -> pd.DataFrame:
    index = np.arange(rows)
    return pd.DataFrame(
        {
            "exam_id": 1_000_000 + index,
            "age": 18 + index % 70,
            "is_male": index % 2 == 0,
            "nn_predicted_age": 20 + index % 60,
            "1dAVb": index % 19 == 0,
            "RBBB": index % 17 == 0,
            "LBBB": index % 23 == 0,
            "SB": index % 29 == 0,
            "ST": index % 31 == 0,
            "AF": index % 13 == 0,
            "patient_id": index // 2,
            "death": False,
            "timey": 1.0,
            "normal_ecg": index % 3 == 0,
            "trace_file": [f"exams_part{value % 3}.hdf5" for value in index],
        }
    )


def test_code15_manifest_is_exact_patient_safe_and_reproducible():
    candidate = _candidate()
    first, qc = build_code15_manifest(candidate, sample_size=500, seed=42)
    second, _ = build_code15_manifest(candidate, sample_size=500, seed=42)

    assert len(first) == 500
    assert first["record_id"].tolist() == second["record_id"].tolist()
    assert set(first["split"]) == {"train", "validation", "test"}
    assert first.groupby("subject_id")["split"].nunique().max() == 1
    assert first["storage"].eq("hdf5").all()
    assert first["signal_path"].str.match(r"exams_part[0-2]\.hdf5::\d+").all()
    assert qc["selected_records"] == 500
    assert qc["patient_leakage_count"] == 0


def test_hdf5_manifest_dataset_reads_code15_exam_by_id(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "exams_part0.hdf5"
    rng = np.random.default_rng(42)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("exam_id", data=np.array([100, 200], dtype=np.int64))
        handle.create_dataset(
            "tracings", data=rng.normal(size=(2, 4096, 12)).astype(np.float32)
        )
    row = {
        "dataset": "code_ii",
        "record_id": "200",
        "subject_id": "20",
        "signal_path": "exams_part0.hdf5::200",
        "storage": "hdf5",
        "split": "train",
        **{label: int(label == "af_afl") for label in LABEL_COLUMNS},
        "valid_num_samples": 5000,
        "mapping_version": "test",
        "split_version": "test",
    }
    item = ECGManifestDataset(pd.DataFrame([row]), split="train", signal_root=tmp_path)[0]
    assert tuple(item["source"].shape) == (2, 12, 2500)
    assert np.isfinite(item["source"].numpy()).all()
