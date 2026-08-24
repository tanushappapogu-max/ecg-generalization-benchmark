from pathlib import Path

import pandas as pd
import pytest

from src.data.prepare_wfdb_manifest import prepare_manifest


def canonical_rows(dataset: str, record_ids: list[str]) -> pd.DataFrame:
    rows = []
    if len(record_ids) != 6:
        raise ValueError("Test fixture requires six record IDs")
    splits = ["train", "train", "validation", "validation", "test", "test"]
    for index, record_id in enumerate(record_ids):
        rows.append(
            {
                "dataset": dataset,
                "record_id": record_id,
                "subject_id": f"p{index}",
                "split": splits[index],
                "signal_path": f"signals/{record_id}.npy",
                "storage": "npy",
                "normal": index % 2,
                "af_afl": index % 2,
                "av_block_1": index % 2,
                "lbbb": index % 2,
                "rbbb": index % 2,
                "valid_num_samples": 5000,
                "mapping_version": "test-v1",
                "split_version": "test-split-v1",
                "seed": 42,
            }
        )
    return pd.DataFrame(rows)


def touch_pair(root: Path, relative: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".hea").touch()


def test_challenge_manifest_only_changes_storage_paths(tmp_path):
    ids = [f"E{index:05d}" for index in range(1, 7)]
    frame = canonical_rows("georgia", ids)
    for record_id in frame["record_id"]:
        touch_pair(tmp_path, f"nested/{record_id}")
    result = prepare_manifest(frame, wfdb_root=tmp_path)
    assert result["signal_path"].tolist() == [f"nested/{record_id}" for record_id in ids]
    assert set(result["storage"]) == {"wfdb"}
    assert result["split"].tolist() == frame["split"].tolist()


def test_ptbxl_uses_official_high_resolution_paths(tmp_path):
    frame = canonical_rows("ptbxl", [str(index) for index in range(1, 7)])
    metadata = tmp_path / "ptbxl_database.csv"
    pd.DataFrame(
        {
            "ecg_id": list(range(1, 7)),
            "filename_hr": [f"r/{index:05d}_hr" for index in range(1, 7)],
        }
    ).to_csv(metadata, index=False)
    for path in (f"r/{index:05d}_hr" for index in range(1, 7)):
        touch_pair(tmp_path, path)
    result = prepare_manifest(frame, wfdb_root=tmp_path, ptbxl_metadata=metadata)
    assert result["signal_path"].tolist() == [
        f"r/{index:05d}_hr" for index in range(1, 7)
    ]


def test_challenge_manifest_fails_if_a_header_is_missing(tmp_path):
    frame = canonical_rows("cpsc2018", [f"A{index:04d}" for index in range(1, 7)])
    touch_pair(tmp_path, "A0001")
    with pytest.raises(FileNotFoundError, match="Missing 5"):
        prepare_manifest(frame, wfdb_root=tmp_path)
