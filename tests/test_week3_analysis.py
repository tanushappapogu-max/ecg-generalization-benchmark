import numpy as np
import pandas as pd
import pytest

from src.evaluation.week3_analysis import (
    _bootstrap_metric,
    add_gaps_and_shifts,
    descriptive_attribution,
    lambda_sensitivity,
    paired_shift_vector_tests,
    shift_group_table,
)


DATASETS = ["ptbxl", "mimic_iv", "cpsc2018"]
ARCHITECTURES = ["a", "b"]


def synthetic_matrix() -> pd.DataFrame:
    rows = []
    for architecture_index, architecture in enumerate(ARCHITECTURES):
        for source in DATASETS:
            for target_index, target in enumerate(DATASETS):
                rows.append(
                    {
                        "architecture": architecture,
                        "source_dataset": source,
                        "target_dataset": target,
                        "macro_auroc": 0.9
                        - architecture_index * 0.01
                        - (0.0 if source == target else 0.1 + target_index * 0.01),
                        "status": "COMPLETE",
                    }
                )
    return pd.DataFrame(rows)


def synthetic_shifts() -> pd.DataFrame:
    rows = []
    for source in DATASETS:
        for target in DATASETS:
            if source == target:
                continue
            rows.append(
                {
                    "source_dataset": source,
                    "target_dataset": target,
                    "PS": int(target == "cpsc2018"),
                    "DS": int(source == "mimic_iv"),
                    "LS": 1,
                }
            )
    return pd.DataFrame(rows)


def test_gap_shift_and_attribution_outputs_are_complete():
    off = add_gaps_and_shifts(synthetic_matrix(), synthetic_shifts())
    assert len(off) == len(ARCHITECTURES) * len(DATASETS) * (len(DATASETS) - 1)
    assert off["delta_ij"].min() == pytest.approx(0.1)
    groups = shift_group_table(off)
    assert groups["ordered_pair_count"].sum() == len(off)
    attribution = descriptive_attribution(off)
    assert len(attribution) == len(ARCHITECTURES) * 3
    shares = attribution.groupby("architecture")["allocated_gap_share"].sum()
    assert shares.tolist() == pytest.approx([1.0, 1.0])


def test_lambda_sweep_and_paired_tests_are_deterministic():
    pooled = synthetic_matrix()
    shifts = synthetic_shifts()
    sweep, summary = lambda_sensitivity(pooled, shifts, [1.0, 2.0])
    assert len(summary) == 8
    assert len(sweep) == 8 * len(ARCHITECTURES)
    tests = paired_shift_vector_tests(shift_group_table(add_gaps_and_shifts(pooled, shifts)))
    assert (tests["paired_architecture_count"] == len(ARCHITECTURES)).all()
    assert tests["exact_two_sided_sign_flip_p"].between(0, 1).all()


def test_bootstrap_metric_emits_macro_and_per_class_intervals():
    predictions = pd.DataFrame(
        {
            "target_normal": [0, 0, 1, 1] * 5,
            "probability_normal": [0.1, 0.2, 0.8, 0.9] * 5,
            "target_af_afl": [0, 1, 0, 1] * 5,
            "probability_af_afl": [0.1, 0.8, 0.2, 0.9] * 5,
        }
    )
    rows = _bootstrap_metric(
        predictions,
        replicates=50,
        rng=np.random.default_rng(42),
    )
    assert {row["metric"] for row in rows} == {
        "macro_auroc",
        "auroc_af_afl",
        "auroc_normal",
    }
    assert all(0 <= row["ci_95_lower"] <= row["ci_95_upper"] <= 1 for row in rows)
