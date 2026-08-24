import math

import pandas as pd

from src.evaluation.binary_matrix import compare_gaps, generalization_gap


def test_generalization_gap_uses_only_completed_cells():
    rows = pd.DataFrame(
        [
            {"source_dataset": "a", "target_dataset": "a", "status": "COMPLETE", "auroc": 0.9},
            {"source_dataset": "b", "target_dataset": "b", "status": "COMPLETE", "auroc": 0.8},
            {"source_dataset": "a", "target_dataset": "b", "status": "COMPLETE", "auroc": 0.6},
            {"source_dataset": "b", "target_dataset": "a", "status": "BLOCKED", "auroc": math.nan},
        ]
    )
    result = generalization_gap(rows, "auroc")
    assert result["completed_in_domain_cells"] == 2
    assert result["completed_cross_dataset_cells"] == 1
    assert math.isclose(result["generalization_gap"], 0.25)


def test_compare_gaps_reports_absolute_and_percent_disappearance():
    result = compare_gaps(0.20, 0.05)
    assert math.isclose(result["absolute_gap_disappearance"], 0.15)
    assert math.isclose(result["percent_gap_disappearance"], 75.0)
