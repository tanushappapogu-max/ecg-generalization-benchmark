import pandas as pd

from src.data.binary_ablation import (
    BINARY_DEFINITION_VERSION,
    build_binary_manifest,
)


def _frame() -> pd.DataFrame:
    rows = []
    for split in ("train", "validation", "test"):
        rows.extend(
            [
                {
                    "dataset": "x",
                    "record_id": f"{split}-normal",
                    "subject_id": f"{split}-normal",
                    "signal_path": "signals/a.npy",
                    "storage": "npy",
                    "split": split,
                    "normal": 1,
                    "af_afl": 0,
                    "av_block_1": 0,
                    "lbbb": 0,
                    "rbbb": 0,
                    "valid_num_samples": 5000,
                    "mapping_version": "v1",
                    "split_version": "s1",
                },
                {
                    "dataset": "x",
                    "record_id": f"{split}-abnormal",
                    "subject_id": f"{split}-abnormal",
                    "signal_path": "signals/b.npy",
                    "storage": "npy",
                    "split": split,
                    "normal": 1,
                    "af_afl": 1,
                    "av_block_1": 1,
                    "lbbb": 1,
                    "rbbb": 1,
                    "valid_num_samples": 5000,
                    "mapping_version": "v1",
                    "split_version": "s1",
                },
                {
                    "dataset": "x",
                    "record_id": f"{split}-zero",
                    "subject_id": f"{split}-zero",
                    "signal_path": "signals/c.npy",
                    "storage": "npy",
                    "split": split,
                    "normal": 0,
                    "af_afl": 0,
                    "av_block_1": 0,
                    "lbbb": 0,
                    "rbbb": 0,
                    "valid_num_samples": 5000,
                    "mapping_version": "v1",
                    "split_version": "s1",
                },
            ]
        )
    return pd.DataFrame(rows)


def test_binary_rule_excludes_all_zero_and_abnormal_wins():
    result = build_binary_manifest(_frame())
    assert len(result) == 6
    assert not result["record_id"].str.endswith("zero").any()
    assert result.loc[result["record_id"].str.endswith("abnormal"), "abnormal"].eq(1).all()
    normal_rows = result["record_id"].str.match(r".*-normal$")
    assert result.loc[normal_rows, "abnormal"].eq(0).all()
    assert result["binary_definition_version"].eq(BINARY_DEFINITION_VERSION).all()
