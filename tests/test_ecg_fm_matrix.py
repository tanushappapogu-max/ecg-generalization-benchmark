import math

import pandas as pd
import pytest

from src.evaluation.ecg_fm_matrix import matrix_from_long, parse_named_paths


def test_parse_named_paths_requires_unique_dataset_names():
    assert parse_named_paths(["ptbxl=/tmp/ptb", "georgia=/tmp/ga"])["ptbxl"].name == "ptb"
    with pytest.raises(ValueError, match="Duplicate"):
        parse_named_paths(["ptbxl=/tmp/a", "ptbxl=/tmp/b"])


def test_matrix_preserves_missing_cells_as_nan():
    rows = pd.DataFrame(
        [
            {"source_dataset": "a", "target_dataset": "a", "macro_auroc": 0.8},
            {"source_dataset": "a", "target_dataset": "b", "macro_auroc": 0.7},
            {"source_dataset": "b", "target_dataset": "a", "macro_auroc": math.nan},
            {"source_dataset": "b", "target_dataset": "b", "macro_auroc": math.nan},
        ]
    )
    matrix = matrix_from_long(rows, ["a", "b"])
    assert matrix.loc["a", "b"] == 0.7
    assert math.isnan(matrix.loc["b", "a"])


def test_matrix_rejects_duplicate_cells():
    rows = pd.DataFrame(
        [
            {"source_dataset": "a", "target_dataset": "a", "macro_auroc": 0.8},
            {"source_dataset": "a", "target_dataset": "a", "macro_auroc": 0.9},
        ]
    )
    with pytest.raises(ValueError, match="duplicate"):
        matrix_from_long(rows, ["a"])

