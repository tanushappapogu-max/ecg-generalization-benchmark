"""Compute ordered demographic KL divergences and PS/DS/LS shift vectors.

The demographic score is the KL divergence between product-of-marginals
age/sex distributions. This is equivalent to adding the directed age and sex
KL divergences and avoids inventing an age-by-sex joint table where a source
only publishes marginal counts.
"""

from __future__ import annotations

import argparse
import csv
import math
from itertools import permutations
from pathlib import Path
from typing import Iterable, Mapping, Sequence


DATASET_ORDER = (
    "PTB-XL",
    "MIMIC-IV-ECG",
    "CPSC2018",
    "Georgia 12-Lead",
    "CODE-15%",
)
AGE_COUNT_FIELDS = (
    "age_0_44_count",
    "age_45_74_count",
    "age_75_plus_count",
)
SEX_COUNT_FIELDS = ("sex_female_count", "sex_male_count")
OUTPUT_FIELDS = (
    "source_dataset",
    "target_dataset",
    "age_kl_nats",
    "sex_kl_nats",
    "demographic_kl_nats",
    "ps_threshold_nats",
    "PS",
    "DS",
    "LS",
    "shift_vector",
    "weight_w_ij",
    "source_age_n",
    "target_age_n",
    "source_sex_n",
    "target_sex_n",
    "source_metadata_basis",
    "target_metadata_basis",
    "ds_reason",
    "ls_reason",
)


def _counts(row: Mapping[str, str], fields: Sequence[str]) -> list[float]:
    values = [float(row[field]) for field in fields]
    if any(value < 0 for value in values):
        raise ValueError(f"Negative count in {row['dataset']}")
    if sum(values) <= 0:
        raise ValueError(f"No usable counts for {row['dataset']}")
    return values


def smoothed_probabilities(
    counts: Sequence[float], *, pseudocount: float = 0.5
) -> list[float]:
    """Return a normalized categorical distribution with Jeffreys smoothing."""

    if pseudocount <= 0:
        raise ValueError("pseudocount must be positive")
    denominator = sum(counts) + pseudocount * len(counts)
    return [(count + pseudocount) / denominator for count in counts]


def directed_kl_nats(
    source_counts: Sequence[float],
    target_counts: Sequence[float],
    *,
    pseudocount: float = 0.5,
) -> float:
    """Return D_KL(source || target) in natural-log units (nats)."""

    if len(source_counts) != len(target_counts):
        raise ValueError("source and target must have the same categories")
    source = smoothed_probabilities(source_counts, pseudocount=pseudocount)
    target = smoothed_probabilities(target_counts, pseudocount=pseudocount)
    return sum(p * math.log(p / q) for p, q in zip(source, target))


def read_demographics(path: str | Path) -> dict[str, dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {row["dataset"]: row for row in rows}
    missing = [dataset for dataset in DATASET_ORDER if dataset not in indexed]
    extras = [dataset for dataset in indexed if dataset not in DATASET_ORDER]
    if missing or extras or len(rows) != len(indexed):
        raise ValueError(
            f"Expected exactly {list(DATASET_ORDER)}; missing={missing}, extras={extras}, "
            f"duplicate_rows={len(rows) - len(indexed)}"
        )
    return indexed


def _domain_shift(source: Mapping[str, str], target: Mapping[str, str]) -> tuple[int, str]:
    source_rate = float(source["sampling_rate_hz"])
    target_rate = float(target["sampling_rate_hz"])
    if source_rate != target_rate:
        return 1, f"sampling rate differs ({source_rate:g} vs {target_rate:g} Hz)"

    both_hardware_documented = (
        source["hardware_documented"] == "1"
        and target["hardware_documented"] == "1"
    )
    if both_hardware_documented and source["hardware_group"] != target["hardware_group"]:
        return 1, "documented acquisition hardware differs"
    return 0, "same sampling rate; undocumented hardware is not treated as different"


def _label_shift(source: Mapping[str, str], target: Mapping[str, str]) -> tuple[int, str]:
    if source["label_schema"] != target["label_schema"]:
        return 1, f"{source['label_schema']} vs {target['label_schema']}"
    return 0, f"same raw coding convention ({source['label_schema']})"


def compute_shift_rows(
    demographics: Mapping[str, Mapping[str, str]],
    *,
    ps_threshold_nats: float = 0.10,
    pseudocount: float = 0.5,
) -> list[dict[str, str | int]]:
    """Compute all 5 x 4 ordered off-diagonal shift rows."""

    if ps_threshold_nats < 0:
        raise ValueError("PS threshold must be non-negative")

    rows: list[dict[str, str | int]] = []
    for source_name, target_name in permutations(DATASET_ORDER, 2):
        source = demographics[source_name]
        target = demographics[target_name]
        source_age = _counts(source, AGE_COUNT_FIELDS)
        target_age = _counts(target, AGE_COUNT_FIELDS)
        source_sex = _counts(source, SEX_COUNT_FIELDS)
        target_sex = _counts(target, SEX_COUNT_FIELDS)

        age_kl = directed_kl_nats(
            source_age, target_age, pseudocount=pseudocount
        )
        sex_kl = directed_kl_nats(
            source_sex, target_sex, pseudocount=pseudocount
        )
        demographic_kl = age_kl + sex_kl
        ps = int(demographic_kl > ps_threshold_nats)
        ds, ds_reason = _domain_shift(source, target)
        ls, ls_reason = _label_shift(source, target)
        weight = ps + 2 * ds + 3 * ls

        rows.append(
            {
                "source_dataset": source_name,
                "target_dataset": target_name,
                "age_kl_nats": f"{age_kl:.6f}",
                "sex_kl_nats": f"{sex_kl:.6f}",
                "demographic_kl_nats": f"{demographic_kl:.6f}",
                "ps_threshold_nats": f"{ps_threshold_nats:.2f}",
                "PS": ps,
                "DS": ds,
                "LS": ls,
                "shift_vector": f"({ps},{ds},{ls})",
                "weight_w_ij": weight,
                "source_age_n": int(sum(source_age)),
                "target_age_n": int(sum(target_age)),
                "source_sex_n": int(sum(source_sex)),
                "target_sex_n": int(sum(target_sex)),
                "source_metadata_basis": source["metadata_basis"],
                "target_metadata_basis": target["metadata_basis"],
                "ds_reason": ds_reason,
                "ls_reason": ls_reason,
            }
        )
    return rows


def write_shift_csv(rows: Iterable[Mapping[str, object]], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demographics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ps-threshold", type=float, default=0.10)
    parser.add_argument("--pseudocount", type=float, default=0.5)
    args = parser.parse_args()

    demographics = read_demographics(args.demographics)
    rows = compute_shift_rows(
        demographics,
        ps_threshold_nats=args.ps_threshold,
        pseudocount=args.pseudocount,
    )
    write_shift_csv(rows, args.output)
    print(f"Wrote {len(rows)} ordered off-diagonal rows to {args.output}")


if __name__ == "__main__":
    main()
