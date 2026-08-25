import pandas as pd
import pytest

from src.evaluation.composite_score import compute_composite_score


def test_proposal_composite_score_matches_hand_calculation():
    matrix = pd.DataFrame(
        [
            {"source_dataset": "a", "target_dataset": "a", "macro_auroc": 0.9},
            {"source_dataset": "a", "target_dataset": "b", "macro_auroc": 0.7},
            {"source_dataset": "b", "target_dataset": "a", "macro_auroc": 0.6},
            {"source_dataset": "b", "target_dataset": "b", "macro_auroc": 0.8},
        ]
    )
    shifts = pd.DataFrame(
        [
            {"source_dataset": "a", "target_dataset": "b", "PS": 1, "DS": 0, "LS": 0},
            {"source_dataset": "b", "target_dataset": "a", "PS": 0, "DS": 1, "LS": 1},
        ]
    )
    score, components = compute_composite_score(matrix, shifts)
    # CD=.65; penalty=mean(.2/(1+1), .2/(5+1))=.066666...
    assert score == pytest.approx(0.65 * (1.0 - (0.1 + 1 / 30) / 2))
    assert components["delta_ij"].tolist() == pytest.approx([0.2, 0.2])
    assert components["w_ij"].tolist() == pytest.approx([1.0, 5.0])


def test_composite_score_requires_complete_shift_vectors():
    matrix = pd.DataFrame(
        [
            {"source_dataset": "a", "target_dataset": "a", "macro_auroc": 0.9},
            {"source_dataset": "a", "target_dataset": "b", "macro_auroc": 0.7},
        ]
    )
    shifts = pd.DataFrame(columns=["source_dataset", "target_dataset", "PS", "DS", "LS"])
    with pytest.raises(ValueError, match="shift vector"):
        compute_composite_score(matrix, shifts)

