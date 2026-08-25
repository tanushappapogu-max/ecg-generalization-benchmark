import csv
import math
from pathlib import Path

from src.evaluation.shift_vectors import (
    compute_shift_rows,
    directed_kl_nats,
    read_demographics,
)


ROOT = Path(__file__).resolve().parents[1]
DEMOGRAPHICS = ROOT / "data" / "shift_metadata" / "demographic_counts.csv"


def test_directed_kl_is_zero_for_equal_distributions() -> None:
    assert math.isclose(directed_kl_nats([10, 20], [10, 20]), 0.0, abs_tol=1e-12)


def test_directed_kl_can_be_asymmetric() -> None:
    forward = directed_kl_nats([90, 10], [50, 50])
    reverse = directed_kl_nats([50, 50], [90, 10])
    assert not math.isclose(forward, reverse)


def test_real_metadata_emits_all_twenty_ordered_pairs() -> None:
    rows = compute_shift_rows(read_demographics(DEMOGRAPHICS))
    pairs = {(row["source_dataset"], row["target_dataset"]) for row in rows}
    assert len(rows) == 20
    assert len(pairs) == 20
    assert all(source != target for source, target in pairs)
    assert all(row["PS"] in (0, 1) for row in rows)
    assert all(row["DS"] in (0, 1) for row in rows)
    assert all(row["LS"] in (0, 1) for row in rows)


def test_committed_csv_matches_recomputation() -> None:
    expected = compute_shift_rows(read_demographics(DEMOGRAPHICS))
    output = ROOT / "results" / "shift_metadata" / "shift_vectors.csv"
    with output.open(newline="", encoding="utf-8") as handle:
        actual = list(csv.DictReader(handle))
    assert actual == [{key: str(value) for key, value in row.items()} for row in expected]
