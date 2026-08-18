#!/usr/bin/env python3
"""Build a reproducible, patient-aware multilabel ECG subset manifest.

The script operates only on metadata and diagnosis labels. It deliberately does
not download or preprocess waveform files. Patients are the indivisible unit for
both subset selection and train/validation/test assignment, so a subject can
never leak across splits.

The grouped iterative stratification heuristic adapts multilabel iterative
stratification to variable-sized patient groups. Rare-label patients are handled
first. Each complete patient is assigned to the partition with the greatest
remaining need for its record count and multilabel-positive counts. For subset
selection, a final whole-patient refinement minimizes the absolute difference
from the requested ECG count; no individual recording is trimmed from a patient.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("build_mimic_subset")

DEFAULT_LABEL_COLUMNS = ("normal", "af_afl", "av_block_1", "lbbb", "rbbb")
DEFAULT_SPLIT_NAMES = ("train", "validation", "test")
DEFAULT_SPLIT_RATIOS = (0.8, 0.1, 0.1)


@dataclass(frozen=True)
class ColumnSpec:
    """Source column names needed to construct the canonical manifest."""

    subject_id: str = "subject_id"
    study_id: str = "study_id"
    waveform_path: str = "waveform_path"


@dataclass(frozen=True)
class PipelineResult:
    """In-memory products from a complete subset-building run."""

    manifest: pd.DataFrame
    qc: pd.DataFrame
    cleaning_counts: Mapping[str, int]


def load_table(path: Path) -> pd.DataFrame:
    """Load a CSV or Parquet table from ``path`` with a clear format error."""

    path = path.expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Input table does not exist: {path}")

    suffixes = [suffix.lower() for suffix in path.suffixes]
    try:
        if suffixes[-1:] == [".parquet"] or suffixes[-1:] == [".pq"]:
            return pd.read_parquet(path)
        if suffixes[-1:] == [".csv"] or suffixes[-2:] in (
            [".csv", ".gz"],
            [".csv", ".bz2"],
            [".csv", ".xz"],
        ):
            return pd.read_csv(path, low_memory=False)
    except ImportError as exc:
        raise RuntimeError(
            f"Could not read {path}. Parquet input requires a pandas-compatible "
            "Parquet engine such as pyarrow."
        ) from exc

    raise ValueError(
        f"Unsupported table format for {path}. Expected .csv, compressed .csv, "
        ".parquet, or .pq."
    )


def load_metadata(path: Path) -> pd.DataFrame:
    """Load the metadata table and report its raw size."""

    metadata = load_table(path)
    LOGGER.info("Loaded %s metadata rows from %s", f"{len(metadata):,}", path)
    return metadata


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], context: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}")


def _coerce_binary(series: pd.Series, column: str) -> pd.Series:
    """Convert common binary encodings to nullable Int8 without guessing."""

    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype("Int8")

    if pd.api.types.is_numeric_dtype(series.dtype):
        numeric = pd.to_numeric(series, errors="coerce")
        invalid = numeric.notna() & ~numeric.isin([0, 1])
        if invalid.any():
            examples = sorted(numeric.loc[invalid].astype(str).unique())[:5]
            raise ValueError(
                f"Label column {column!r} contains non-binary values: {examples}"
            )
        return numeric.astype("Int8")

    normalized = series.astype("string").str.strip().str.lower()
    value_map = {
        "0": 0,
        "0.0": 0,
        "false": 0,
        "f": 0,
        "no": 0,
        "n": 0,
        "1": 1,
        "1.0": 1,
        "true": 1,
        "t": 1,
        "yes": 1,
        "y": 1,
    }
    coerced = normalized.map(value_map)
    invalid = normalized.notna() & coerced.isna()
    if invalid.any():
        examples = sorted(normalized.loc[invalid].dropna().unique())[:5]
        raise ValueError(
            f"Label column {column!r} contains unrecognized binary values: {examples}"
        )
    return coerced.astype("Int8")


def derive_labels_from_long_table(
    diagnoses: pd.DataFrame,
    label_map: pd.DataFrame,
    *,
    study_id_col: str,
    diagnosis_code_col: str,
    map_code_col: str,
    map_label_col: str,
    label_cols: Sequence[str],
) -> pd.DataFrame:
    """Derive wide binary labels from a long diagnosis table and mapping CSV.

    Every study with at least one nonblank diagnosis code is retained. Codes not
    mapped to a benchmark label produce zeros, which is distinct from having no
    usable diagnosis information.
    """

    _require_columns(
        diagnoses, [study_id_col, diagnosis_code_col], "Long diagnosis table"
    )
    _require_columns(label_map, [map_code_col, map_label_col], "Label mapping table")

    diagnoses = diagnoses[[study_id_col, diagnosis_code_col]].copy()
    diagnoses["_normalized_code"] = (
        diagnoses[diagnosis_code_col].astype("string").str.strip()
    )
    usable = diagnoses[
        diagnoses[study_id_col].notna()
        & diagnoses["_normalized_code"].notna()
        & diagnoses["_normalized_code"].ne("")
    ].copy()

    mapping = label_map[[map_code_col, map_label_col]].copy()
    mapping["_normalized_code"] = mapping[map_code_col].astype("string").str.strip()
    mapping[map_label_col] = mapping[map_label_col].astype("string").str.strip()
    unknown_targets = sorted(
        set(mapping[map_label_col].dropna()) - set(label_cols)
    )
    if unknown_targets:
        LOGGER.warning(
            "Ignoring mapping rows for target labels not requested by --label-cols: %s",
            unknown_targets,
        )
    mapping = mapping[mapping[map_label_col].isin(label_cols)].drop_duplicates(
        ["_normalized_code", map_label_col]
    )

    matched = usable.merge(
        mapping[["_normalized_code", map_label_col]],
        on="_normalized_code",
        how="inner",
        validate="many_to_many",
    )
    matched["_positive"] = 1
    if matched.empty:
        pivoted = pd.DataFrame(index=pd.Index([], name=study_id_col))
    else:
        pivoted = matched.pivot_table(
            index=study_id_col,
            columns=map_label_col,
            values="_positive",
            aggfunc="max",
            fill_value=0,
        )

    all_studies = pd.DataFrame({study_id_col: usable[study_id_col].drop_duplicates()})
    wide = all_studies.merge(pivoted.reset_index(), on=study_id_col, how="left")
    for label in label_cols:
        if label not in wide:
            wide[label] = 0
    wide[list(label_cols)] = wide[list(label_cols)].fillna(0).astype("Int8")

    LOGGER.info(
        "Derived labels for %s studies from %s usable diagnosis rows; %s rows "
        "matched at least one requested mapping rule",
        f"{len(wide):,}",
        f"{len(usable):,}",
        f"{len(matched):,}",
    )
    return wide[[study_id_col, *label_cols]]


def prepare_labels(
    metadata: pd.DataFrame,
    *,
    label_cols: Sequence[str],
    study_id_col: str,
    labels: pd.DataFrame | None = None,
    label_map: pd.DataFrame | None = None,
    diagnosis_code_col: str = "snomed_code",
    map_code_col: str = "source_code",
    map_label_col: str = "target_label",
) -> pd.DataFrame:
    """Attach and validate configurable binary labels on metadata rows."""

    if not label_cols:
        raise ValueError("At least one label column is required")
    if len(set(label_cols)) != len(label_cols):
        raise ValueError(f"Label columns must be unique: {list(label_cols)}")

    if labels is None:
        if label_map is not None:
            raise ValueError("--label-map requires a long diagnosis table via --labels")
        combined = metadata.copy()
        _require_columns(combined, label_cols, "Metadata table")
    else:
        if label_map is not None:
            wide_labels = derive_labels_from_long_table(
                labels,
                label_map,
                study_id_col=study_id_col,
                diagnosis_code_col=diagnosis_code_col,
                map_code_col=map_code_col,
                map_label_col=map_label_col,
                label_cols=label_cols,
            )
        else:
            _require_columns(labels, [study_id_col, *label_cols], "Wide label table")
            wide_labels = labels[[study_id_col, *label_cols]].copy()
            for label in label_cols:
                wide_labels[label] = _coerce_binary(wide_labels[label], label)
            # Multiple source rows for one study are valid for a wide diagnosis
            # export; max implements multilabel OR while preserving all-missing.
            wide_labels = wide_labels.groupby(study_id_col, as_index=False)[
                list(label_cols)
            ].max()

        metadata_without_labels = metadata.drop(
            columns=[column for column in label_cols if column in metadata],
            errors="ignore",
        )
        combined = metadata_without_labels.merge(
            wide_labels,
            on=study_id_col,
            how="left",
            validate="many_to_one",
        )

    for label in label_cols:
        combined[label] = _coerce_binary(combined[label], label)
    return combined


def _blank_or_missing(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype("string").str.strip().eq("")


def validate_metadata(
    metadata: pd.DataFrame,
    *,
    columns: ColumnSpec,
    label_cols: Sequence[str],
    require_any_positive_label: bool = False,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Filter unusable rows with explicit counts and reject conflicting studies."""

    required = [columns.subject_id, columns.study_id, columns.waveform_path, *label_cols]
    _require_columns(metadata, required, "Prepared metadata")

    cleaned = metadata.copy()
    counts: dict[str, int] = {"input_records": len(cleaned)}

    missing_ids = _blank_or_missing(cleaned[columns.subject_id]) | _blank_or_missing(
        cleaned[columns.study_id]
    )
    counts["dropped_missing_ids"] = int(missing_ids.sum())
    cleaned = cleaned.loc[~missing_ids].copy()

    missing_paths = _blank_or_missing(cleaned[columns.waveform_path])
    counts["dropped_missing_waveform_paths"] = int(missing_paths.sum())
    cleaned = cleaned.loc[~missing_paths].copy()

    incomplete_labels = cleaned[list(label_cols)].isna().any(axis=1)
    counts["dropped_incomplete_labels"] = int(incomplete_labels.sum())
    cleaned = cleaned.loc[~incomplete_labels].copy()

    if require_any_positive_label:
        no_positive = cleaned[list(label_cols)].sum(axis=1).eq(0)
        counts["dropped_without_positive_target_label"] = int(no_positive.sum())
        cleaned = cleaned.loc[~no_positive].copy()
    else:
        counts["dropped_without_positive_target_label"] = 0

    duplicate_mask = cleaned.duplicated(columns.study_id, keep=False)
    duplicates = cleaned.loc[duplicate_mask]
    if not duplicates.empty:
        compare_columns = [columns.subject_id, columns.waveform_path, *label_cols]
        conflicts = duplicates.groupby(columns.study_id, dropna=False)[compare_columns].nunique(
            dropna=False
        )
        conflicting_studies = conflicts.index[conflicts.gt(1).any(axis=1)].tolist()
        if conflicting_studies:
            examples = conflicting_studies[:5]
            raise ValueError(
                "Duplicate study IDs have conflicting subject/path/label values. "
                f"Examples: {examples}"
            )

    before_deduplication = len(cleaned)
    cleaned = cleaned.drop_duplicates(columns.study_id, keep="first").copy()
    counts["dropped_duplicate_study_rows"] = before_deduplication - len(cleaned)
    counts["eligible_records"] = len(cleaned)
    counts["eligible_patients"] = cleaned[columns.subject_id].nunique()

    for reason, count in counts.items():
        if reason.startswith("dropped_"):
            LOGGER.info("%s: %s", reason, f"{count:,}")
    LOGGER.info(
        "Eligible population: %s ECGs from %s patients",
        f"{counts['eligible_records']:,}",
        f"{counts['eligible_patients']:,}",
    )

    if cleaned.empty:
        raise ValueError("No eligible ECG records remain after validation")
    return cleaned.reset_index(drop=True), counts


def compute_prevalence(frame: pd.DataFrame, label_cols: Sequence[str]) -> pd.Series:
    """Return record-level prevalence for each binary label."""

    if frame.empty:
        return pd.Series(np.nan, index=list(label_cols), dtype=float)
    return frame[list(label_cols)].astype(float).mean().rename("prevalence")


def build_patient_table(
    records: pd.DataFrame, *, subject_id_col: str, label_cols: Sequence[str]
) -> pd.DataFrame:
    """Aggregate records into indivisible patient groups.

    Label columns in the returned table are positive-record counts, rather than
    a single patient-level OR flag. This lets stratification preserve record-level
    prevalence while keeping each patient intact.
    """

    grouped = records.groupby(subject_id_col, sort=False, dropna=False)
    sizes = grouped.size().rename("n_recordings")
    positives = grouped[list(label_cols)].sum().astype("int64")
    patient_table = pd.concat([sizes, positives], axis=1).reset_index()
    return patient_table


def _validate_proportions(proportions: Sequence[float]) -> np.ndarray:
    values = np.asarray(proportions, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("At least two partition proportions are required")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Partition proportions must be finite and positive")
    total = float(values.sum())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(f"Partition proportions must sum to 1.0, got {total}")
    return values / total


def iterative_group_partition(
    patient_table: pd.DataFrame,
    *,
    label_cols: Sequence[str],
    proportions: Sequence[float],
    seed: int,
) -> np.ndarray:
    """Assign whole patients with a deterministic multilabel iterative heuristic.

    The currently rarest remaining label is handled first, as in multilabel
    iterative stratification. A patient carrying that label goes to the partition
    with the greatest remaining target count for the label. Needs for the
    patient's other labels and then total-record need break ties. Once only
    all-zero patients remain, their larger groups are assigned first by recording
    need. This adapts the standard row-level method to unequal patient groups.
    """

    ratios = _validate_proportions(proportions)
    if patient_table.empty:
        return np.array([], dtype=np.int16)

    _require_columns(patient_table, ["n_recordings", *label_cols], "Patient table")
    record_counts = patient_table["n_recordings"].to_numpy(dtype=np.int64)
    label_counts = patient_table[list(label_cols)].to_numpy(dtype=np.int64)
    if (record_counts <= 0).any() or (label_counts < 0).any():
        raise ValueError("Patient record and label-positive counts must be nonnegative")

    total_records = int(record_counts.sum())
    total_label_counts = label_counts.sum(axis=0).astype(float)
    target_records = ratios * total_records
    target_labels = ratios[:, None] * total_label_counts[None, :]
    remaining_records = target_records.copy()
    remaining_labels = target_labels.copy()

    rng = np.random.default_rng(seed)
    patient_presence = label_counts > 0
    remaining_label_patients = patient_presence.sum(axis=0).astype(np.int64)
    inverse_frequency = np.divide(
        1.0,
        remaining_label_patients,
        out=np.zeros_like(remaining_label_patients, dtype=float),
        where=remaining_label_patients > 0,
    )
    co_label_rarity = patient_presence @ inverse_frequency
    co_label_count = patient_presence.sum(axis=1)
    patient_ties = rng.random(len(patient_table))

    # Each label gets a deterministic queue. Pointers skip patients already
    # consumed through a different, rarer co-occurring label.
    label_queues: list[np.ndarray] = []
    label_pointers = np.zeros(len(label_cols), dtype=np.int64)
    for label_index in range(len(label_cols)):
        indices = np.flatnonzero(patient_presence[:, label_index])
        if len(indices):
            queue_frame = pd.DataFrame(
                {
                    "row": indices,
                    "co_label_rarity": co_label_rarity[indices],
                    "co_label_count": co_label_count[indices],
                    "positive_count": label_counts[indices, label_index],
                    "tie": patient_ties[indices],
                }
            )
            indices = queue_frame.sort_values(
                ["co_label_rarity", "co_label_count", "positive_count", "tie"],
                ascending=[False, False, False, True],
                kind="mergesort",
            )["row"].to_numpy(dtype=np.int64)
        label_queues.append(indices)

    assignments = np.full(len(patient_table), -1, dtype=np.int16)
    unassigned = np.ones(len(patient_table), dtype=bool)
    safe_label_totals = np.maximum(total_label_counts, 1.0)

    while (remaining_label_patients > 0).any():
        active_labels = np.flatnonzero(remaining_label_patients > 0)
        smallest_frequency = remaining_label_patients[active_labels].min()
        rarest_labels = active_labels[
            remaining_label_patients[active_labels] == smallest_frequency
        ]
        # A seeded choice prevents fixed label-column order from resolving exact
        # rarity ties on every run while remaining fully reproducible.
        chosen_label = int(rng.choice(rarest_labels))

        queue = label_queues[chosen_label]
        pointer = int(label_pointers[chosen_label])
        while pointer < len(queue) and not unassigned[queue[pointer]]:
            pointer += 1
        label_pointers[chosen_label] = pointer + 1
        if pointer >= len(queue):
            # Defensive recovery for an exhausted queue. The frequency counter
            # should normally have reached zero as co-labelled patients moved.
            remaining_label_patients[chosen_label] = 0
            continue

        row_index = int(queue[pointer])
        group_records = int(record_counts[row_index])
        group_labels = label_counts[row_index].astype(float)

        primary_need = remaining_labels[:, chosen_label]
        candidate_partitions = np.flatnonzero(
            np.isclose(primary_need, primary_need.max(), rtol=0.0, atol=1e-12)
        )
        if len(candidate_partitions) > 1:
            label_mix = group_labels / group_labels.sum()
            normalized_label_need = (
                np.clip(remaining_labels[candidate_partitions], 0.0, None)
                / safe_label_totals
            )
            secondary_need = normalized_label_need @ label_mix
            best_secondary = secondary_need.max()
            candidate_partitions = candidate_partitions[
                np.isclose(secondary_need, best_secondary, rtol=0.0, atol=1e-12)
            ]
        if len(candidate_partitions) > 1:
            record_need = remaining_records[candidate_partitions]
            best_record_need = record_need.max()
            candidate_partitions = candidate_partitions[
                np.isclose(record_need, best_record_need, rtol=0.0, atol=1e-12)
            ]
        choice = int(rng.choice(candidate_partitions))

        assignments[row_index] = choice
        unassigned[row_index] = False
        remaining_records[choice] -= group_records
        remaining_labels[choice] -= group_labels
        remaining_label_patients -= patient_presence[row_index].astype(np.int64)

    # All remaining patients have zero positive target labels. Largest groups go
    # first so recording-count ratios can be matched as closely as grouping allows.
    zero_label_indices = np.flatnonzero(unassigned)
    if len(zero_label_indices):
        zero_frame = pd.DataFrame(
            {
                "row": zero_label_indices,
                "n_recordings": record_counts[zero_label_indices],
                "tie": patient_ties[zero_label_indices],
            }
        )
        zero_order = zero_frame.sort_values(
            ["n_recordings", "tie"], ascending=[False, True], kind="mergesort"
        )["row"].to_numpy(dtype=np.int64)
        for row_index in zero_order:
            best_need = remaining_records.max()
            candidate_partitions = np.flatnonzero(
                np.isclose(remaining_records, best_need, rtol=0.0, atol=1e-12)
            )
            choice = int(rng.choice(candidate_partitions))
            assignments[row_index] = choice
            unassigned[row_index] = False
            remaining_records[choice] -= int(record_counts[row_index])

    if (assignments < 0).any():
        raise RuntimeError("Internal error: not every patient received a partition")
    return assignments


def _prevalence_distance(
    positive_counts: np.ndarray, record_count: int, reference: np.ndarray
) -> float:
    if record_count <= 0:
        return float("inf")
    observed = positive_counts / record_count
    scale = np.maximum(reference, 1.0 / record_count)
    return float(np.mean(np.abs(observed - reference) / scale))


def _refine_subset_size(
    patient_table: pd.DataFrame,
    selected: np.ndarray,
    *,
    label_cols: Sequence[str],
    target_recordings: int,
    reference_prevalence: np.ndarray,
) -> np.ndarray:
    """Improve target-size closeness using only whole-patient moves."""

    record_counts = patient_table["n_recordings"].to_numpy(dtype=np.int64)
    label_counts = patient_table[list(label_cols)].to_numpy(dtype=np.int64)
    selected = selected.copy()
    selected_records = int(record_counts[selected].sum())
    selected_labels = label_counts[selected].sum(axis=0).astype(float)

    while selected_records != target_recordings:
        current_difference = abs(selected_records - target_recordings)
        candidate_mask = ~selected if selected_records < target_recordings else selected
        candidate_indices = np.flatnonzero(candidate_mask)
        if not len(candidate_indices):
            break

        direction = 1 if selected_records < target_recordings else -1
        candidate_records = selected_records + direction * record_counts[candidate_indices]
        candidate_differences = np.abs(candidate_records - target_recordings)
        improving = candidate_differences < current_difference
        if not improving.any():
            break

        candidate_indices = candidate_indices[improving]
        candidate_records = candidate_records[improving]
        candidate_differences = candidate_differences[improving]
        best_size_difference = candidate_differences.min()
        finalists = np.flatnonzero(candidate_differences == best_size_difference)

        if len(finalists) == 1:
            finalist_position = int(finalists[0])
        else:
            prevalence_distances = []
            for position in finalists:
                candidate_index = candidate_indices[position]
                new_labels = (
                    selected_labels + direction * label_counts[candidate_index]
                )
                prevalence_distances.append(
                    _prevalence_distance(
                        new_labels,
                        int(candidate_records[position]),
                        reference_prevalence,
                    )
                )
            finalist_position = int(finalists[int(np.argmin(prevalence_distances))])

        patient_index = int(candidate_indices[finalist_position])
        selected[patient_index] = not selected[patient_index]
        selected_records += direction * int(record_counts[patient_index])
        selected_labels += direction * label_counts[patient_index]

    return selected


def _refine_subset_prevalence(
    patient_table: pd.DataFrame,
    selected: np.ndarray,
    *,
    label_cols: Sequence[str],
    reference_prevalence: np.ndarray,
    max_rounds: int = 4,
) -> np.ndarray:
    """Improve prevalence through equal-record-count whole-patient swaps.

    Swapping patients with the same number of ECGs leaves the selected recording
    count exactly unchanged. Candidate pairs are ranked by the current weighted
    label-count deficit and accepted only when they strictly reduce relative
    label-count error. This corrects dilution that can occur when the size pass
    adds low-label patients to reach the requested recording target.
    """

    if max_rounds <= 0 or selected.all() or (~selected).all():
        return selected

    record_counts = patient_table["n_recordings"].to_numpy(dtype=np.int64)
    label_counts = patient_table[list(label_cols)].to_numpy(dtype=np.int64)
    selected = selected.copy()
    selected_record_count = int(record_counts[selected].sum())
    target_label_counts = reference_prevalence * selected_record_count
    scale = np.maximum(target_label_counts, 1.0)
    current_label_counts = label_counts[selected].sum(axis=0).astype(float)

    def objective(counts: np.ndarray) -> float:
        return float(np.mean(np.abs(counts - target_label_counts) / scale))

    current_objective = objective(current_label_counts)
    unique_group_sizes = np.unique(record_counts)
    for _ in range(max_rounds):
        swaps_made = 0
        deficit = target_label_counts - current_label_counts
        direction = np.sign(deficit) / scale

        for group_size in unique_group_sizes:
            group_indices = np.flatnonzero(record_counts == group_size)
            selected_indices = group_indices[selected[group_indices]]
            remainder_indices = group_indices[~selected[group_indices]]
            if not len(selected_indices) or not len(remainder_indices):
                continue

            remove_scores = label_counts[selected_indices] @ direction
            add_scores = label_counts[remainder_indices] @ direction
            remove_order = np.argsort(remove_scores, kind="stable")
            add_order = np.argsort(-add_scores, kind="stable")
            pair_count = min(len(remove_order), len(add_order))

            for pair_position in range(pair_count):
                remove_index = int(selected_indices[remove_order[pair_position]])
                add_index = int(remainder_indices[add_order[pair_position]])
                proposed_counts = (
                    current_label_counts
                    - label_counts[remove_index]
                    + label_counts[add_index]
                )
                proposed_objective = objective(proposed_counts)
                if proposed_objective + 1e-12 < current_objective:
                    selected[remove_index] = False
                    selected[add_index] = True
                    current_label_counts = proposed_counts
                    current_objective = proposed_objective
                    swaps_made += 1

        if swaps_made == 0:
            break
    return selected


def _refine_subset_with_count_preserving_swaps(
    patient_table: pd.DataFrame,
    selected: np.ndarray,
    *,
    label_cols: Sequence[str],
    reference_prevalence: np.ndarray,
) -> np.ndarray:
    """Correct one label at a time without changing size or other label counts.

    For a target label, patients are paired only when they have the same number
    of recordings and identical positive-record counts for every other label.
    Swapping such a pair can improve the target label while mathematically
    preserving the subset size and all previously balanced label counts.
    """

    selected = selected.copy()
    record_counts = patient_table["n_recordings"].to_numpy(dtype=np.int64)
    label_counts = patient_table[list(label_cols)].to_numpy(dtype=np.int64)
    selected_record_count = int(record_counts[selected].sum())
    target_counts = reference_prevalence * selected_record_count

    for label_index in np.argsort(
        -np.abs(label_counts[selected].sum(axis=0) - target_counts)
    ):
        current_count = float(label_counts[selected, label_index].sum())
        target_count = float(target_counts[label_index])
        if abs(current_count - target_count) <= 0.5:
            continue
        desired_direction = 1 if current_count < target_count else -1
        other_indices = [
            index for index in range(len(label_cols)) if index != label_index
        ]
        grouping = pd.DataFrame(
            {
                "row": np.arange(len(patient_table), dtype=np.int64),
                "n_recordings": record_counts,
                **{
                    f"other_{position}": label_counts[:, other_index]
                    for position, other_index in enumerate(other_indices)
                },
            }
        )
        group_columns = [
            "n_recordings",
            *[f"other_{position}" for position in range(len(other_indices))],
        ]
        candidates: list[tuple[int, int, int]] = []
        for _, group in grouping.groupby(group_columns, sort=False, dropna=False):
            rows = group["row"].to_numpy(dtype=np.int64)
            chosen_rows = rows[selected[rows]]
            remainder_rows = rows[~selected[rows]]
            if not len(chosen_rows) or not len(remainder_rows):
                continue
            if desired_direction > 0:
                chosen_rows = chosen_rows[
                    np.argsort(label_counts[chosen_rows, label_index], kind="stable")
                ]
                remainder_rows = remainder_rows[
                    np.argsort(-label_counts[remainder_rows, label_index], kind="stable")
                ]
            else:
                chosen_rows = chosen_rows[
                    np.argsort(-label_counts[chosen_rows, label_index], kind="stable")
                ]
                remainder_rows = remainder_rows[
                    np.argsort(label_counts[remainder_rows, label_index], kind="stable")
                ]
            for remove_index, add_index in zip(
                chosen_rows, remainder_rows, strict=False
            ):
                delta = int(
                    label_counts[add_index, label_index]
                    - label_counts[remove_index, label_index]
                )
                if delta * desired_direction > 0:
                    candidates.append((delta, int(remove_index), int(add_index)))

        while candidates:
            current_error = abs(current_count - target_count)
            candidate_errors = np.asarray(
                [abs(current_count + delta - target_count) for delta, _, _ in candidates]
            )
            best_position = int(np.argmin(candidate_errors))
            if candidate_errors[best_position] >= current_error - 1e-12:
                break
            delta, remove_index, add_index = candidates.pop(best_position)
            selected[remove_index] = False
            selected[add_index] = True
            current_count += delta

    return selected


def select_patient_subset(
    patient_table: pd.DataFrame,
    *,
    label_cols: Sequence[str],
    target_recordings: int,
    seed: int,
    reference_prevalence: np.ndarray | None = None,
) -> pd.DataFrame:
    """Select whole patients near ``target_recordings`` with label preservation."""

    if target_recordings <= 0:
        raise ValueError("target_recordings must be positive")
    total_recordings = int(patient_table["n_recordings"].sum())
    if total_recordings <= target_recordings:
        LOGGER.warning(
            "Target (%s) is at least the eligible population (%s); selecting all patients",
            f"{target_recordings:,}",
            f"{total_recordings:,}",
        )
        return patient_table.copy().reset_index(drop=True)

    selected_fraction = target_recordings / total_recordings
    assignments = iterative_group_partition(
        patient_table,
        label_cols=label_cols,
        proportions=(selected_fraction, 1.0 - selected_fraction),
        seed=seed,
    )
    selected_mask = assignments == 0
    if reference_prevalence is None:
        reference_prevalence = (
            patient_table[list(label_cols)].sum().to_numpy(dtype=float)
            / total_recordings
        )
    else:
        reference_prevalence = np.asarray(reference_prevalence, dtype=float)
        if reference_prevalence.shape != (len(label_cols),):
            raise ValueError(
                "reference_prevalence must contain one value per label column"
            )
        if (
            not np.isfinite(reference_prevalence).all()
            or (reference_prevalence < 0).any()
            or (reference_prevalence > 1).any()
        ):
            raise ValueError("reference_prevalence values must be between 0 and 1")
    selected_mask = _refine_subset_size(
        patient_table,
        selected_mask,
        label_cols=label_cols,
        target_recordings=target_recordings,
        reference_prevalence=reference_prevalence,
    )
    selected_mask = _refine_subset_prevalence(
        patient_table,
        selected_mask,
        label_cols=label_cols,
        reference_prevalence=reference_prevalence,
    )
    selected_mask = _refine_subset_with_count_preserving_swaps(
        patient_table,
        selected_mask,
        label_cols=label_cols,
        reference_prevalence=reference_prevalence,
    )
    selected = patient_table.loc[selected_mask].copy().reset_index(drop=True)
    LOGGER.info(
        "Selected %s ECGs from %s patients (target %s; difference %s)",
        f"{int(selected['n_recordings'].sum()):,}",
        f"{len(selected):,}",
        f"{target_recordings:,}",
        f"{int(selected['n_recordings'].sum()) - target_recordings:+,}",
    )
    return selected


def _refine_partition_record_counts(
    patient_table: pd.DataFrame,
    assignments: np.ndarray,
    *,
    label_cols: Sequence[str],
    proportions: Sequence[float],
    seed: int,
) -> np.ndarray:
    """Move whole patients to improve split record ratios after stratification.

    The iterative stratifier prioritizes rare labels, so unequal patient group
    sizes can leave the record totals farther from their requested proportions
    than necessary. This deterministic local pass accepts only moves that
    strictly reduce total record-count error. Among equally useful moves, it
    chooses the one with the smallest normalized label-count error.
    """

    ratios = _validate_proportions(proportions)
    assignments = assignments.copy()
    record_counts = patient_table["n_recordings"].to_numpy(dtype=np.int64)
    label_counts = patient_table[list(label_cols)].to_numpy(dtype=np.int64)
    partition_count = len(ratios)
    target_records = ratios * record_counts.sum()
    total_labels = label_counts.sum(axis=0).astype(float)
    target_labels = ratios[:, None] * total_labels[None, :]
    label_scale = np.maximum(target_labels, 1.0)
    current_records = np.bincount(
        assignments, weights=record_counts, minlength=partition_count
    ).astype(float)
    current_labels = np.vstack(
        [label_counts[assignments == index].sum(axis=0) for index in range(partition_count)]
    ).astype(float)
    tie_scores = np.random.default_rng(seed).random(len(assignments))

    while True:
        current_size_error = float(np.abs(current_records - target_records).sum())
        best_move: tuple[float, float, float, int, int, int] | None = None

        for source in range(partition_count):
            source_indices = np.flatnonzero(assignments == source)
            if not len(source_indices):
                continue
            sizes = record_counts[source_indices].astype(float)
            for destination in range(partition_count):
                if source == destination:
                    continue
                proposed_source = current_records[source] - sizes
                proposed_destination = current_records[destination] + sizes
                pair_before = abs(current_records[source] - target_records[source]) + abs(
                    current_records[destination] - target_records[destination]
                )
                pair_after = np.abs(proposed_source - target_records[source]) + np.abs(
                    proposed_destination - target_records[destination]
                )
                improvements = pair_before - pair_after
                best_improvement = float(improvements.max(initial=0.0))
                if best_improvement <= 1e-12:
                    continue

                finalists = source_indices[
                    np.isclose(improvements, best_improvement, rtol=0.0, atol=1e-12)
                ]
                finalist_labels = label_counts[finalists].astype(float)
                source_errors = np.abs(
                    current_labels[source] - finalist_labels - target_labels[source]
                ) / label_scale[source]
                destination_errors = np.abs(
                    current_labels[destination]
                    + finalist_labels
                    - target_labels[destination]
                ) / label_scale[destination]
                label_errors = (source_errors + destination_errors).mean(axis=1)
                best_label_error = float(label_errors.min())
                label_finalists = finalists[
                    np.isclose(label_errors, best_label_error, rtol=0.0, atol=1e-12)
                ]
                chosen = int(label_finalists[np.argmin(tie_scores[label_finalists])])
                candidate = (
                    -best_improvement,
                    best_label_error,
                    float(tie_scores[chosen]),
                    source,
                    destination,
                    chosen,
                )
                if best_move is None or candidate < best_move:
                    best_move = candidate

        if best_move is None:
            break
        _, _, _, source, destination, patient_index = best_move
        size = float(record_counts[patient_index])
        labels = label_counts[patient_index].astype(float)
        assignments[patient_index] = destination
        current_records[source] -= size
        current_records[destination] += size
        current_labels[source] -= labels
        current_labels[destination] += labels
        new_size_error = float(np.abs(current_records - target_records).sum())
        if new_size_error >= current_size_error - 1e-12:
            raise RuntimeError("Internal error: split-size refinement did not improve")

    return assignments


def _refine_partition_with_count_preserving_swaps(
    patient_table: pd.DataFrame,
    assignments: np.ndarray,
    *,
    label_cols: Sequence[str],
    proportions: Sequence[float],
    max_rounds: int = 4,
) -> np.ndarray:
    """Balance split labels via swaps that preserve size and other labels."""

    ratios = _validate_proportions(proportions)
    assignments = assignments.copy()
    record_counts = patient_table["n_recordings"].to_numpy(dtype=np.int64)
    label_counts = patient_table[list(label_cols)].to_numpy(dtype=np.int64)
    partition_count = len(ratios)
    target_labels = ratios[:, None] * label_counts.sum(axis=0)[None, :]
    current_labels = np.vstack(
        [label_counts[assignments == index].sum(axis=0) for index in range(partition_count)]
    ).astype(float)

    label_order = np.argsort(
        -np.sum(np.abs(current_labels - target_labels), axis=0)
    )
    for label_index in label_order:
        other_indices = [
            index for index in range(len(label_cols)) if index != label_index
        ]
        for _ in range(max_rounds):
            swaps_made = 0
            errors = current_labels[:, label_index] - target_labels[:, label_index]
            sources = np.flatnonzero(errors > 0.5)
            destinations = np.flatnonzero(errors < -0.5)
            if not len(sources) or not len(destinations):
                break

            pair_order = sorted(
                (
                    (-(errors[source] - errors[destination]), int(source), int(destination))
                    for source in sources
                    for destination in destinations
                )
            )
            for _, source, destination in pair_order:
                rows = np.flatnonzero(
                    (assignments == source) | (assignments == destination)
                )
                grouping = pd.DataFrame(
                    {
                        "row": rows,
                        "partition": assignments[rows],
                        "n_recordings": record_counts[rows],
                        **{
                            f"other_{position}": label_counts[rows, other_index]
                            for position, other_index in enumerate(other_indices)
                        },
                    }
                )
                group_columns = [
                    "n_recordings",
                    *[
                        f"other_{position}"
                        for position in range(len(other_indices))
                    ],
                ]
                candidates: list[tuple[int, int, int]] = []
                for _, group in grouping.groupby(
                    group_columns, sort=False, dropna=False
                ):
                    source_rows = group.loc[
                        group["partition"] == source, "row"
                    ].to_numpy(dtype=np.int64)
                    destination_rows = group.loc[
                        group["partition"] == destination, "row"
                    ].to_numpy(dtype=np.int64)
                    if not len(source_rows) or not len(destination_rows):
                        continue
                    source_rows = source_rows[
                        np.argsort(
                            -label_counts[source_rows, label_index], kind="stable"
                        )
                    ]
                    destination_rows = destination_rows[
                        np.argsort(
                            label_counts[destination_rows, label_index], kind="stable"
                        )
                    ]
                    for source_row, destination_row in zip(
                        source_rows, destination_rows, strict=False
                    ):
                        source_delta = int(
                            label_counts[destination_row, label_index]
                            - label_counts[source_row, label_index]
                        )
                        if source_delta < 0:
                            candidates.append(
                                (
                                    source_delta,
                                    int(source_row),
                                    int(destination_row),
                                )
                            )

                while candidates:
                    pair_error = abs(
                        current_labels[source, label_index]
                        - target_labels[source, label_index]
                    ) + abs(
                        current_labels[destination, label_index]
                        - target_labels[destination, label_index]
                    )
                    proposed_errors = np.asarray(
                        [
                            abs(
                                current_labels[source, label_index]
                                + delta
                                - target_labels[source, label_index]
                            )
                            + abs(
                                current_labels[destination, label_index]
                                - delta
                                - target_labels[destination, label_index]
                            )
                            for delta, _, _ in candidates
                        ]
                    )
                    best_position = int(np.argmin(proposed_errors))
                    if proposed_errors[best_position] >= pair_error - 1e-12:
                        break
                    source_delta, source_row, destination_row = candidates.pop(
                        best_position
                    )
                    assignments[source_row] = destination
                    assignments[destination_row] = source
                    current_labels[source, label_index] += source_delta
                    current_labels[destination, label_index] -= source_delta
                    swaps_made += 1

            if swaps_made == 0:
                break

    return assignments


def assign_patient_splits(
    selected_patients: pd.DataFrame,
    *,
    subject_id_col: str,
    label_cols: Sequence[str],
    split_ratios: Mapping[str, float],
    seed: int,
) -> pd.DataFrame:
    """Assign selected patients to leakage-safe multilabel-stratified splits."""

    if len(split_ratios) < 2:
        raise ValueError("At least two splits are required")
    split_names = list(split_ratios)
    if len(set(split_names)) != len(split_names):
        raise ValueError("Split names must be unique")
    assignments = iterative_group_partition(
        selected_patients,
        label_cols=label_cols,
        proportions=list(split_ratios.values()),
        seed=seed,
    )
    assignments = _refine_partition_record_counts(
        selected_patients,
        assignments,
        label_cols=label_cols,
        proportions=list(split_ratios.values()),
        seed=seed,
    )
    assignments = _refine_partition_with_count_preserving_swaps(
        selected_patients,
        assignments,
        label_cols=label_cols,
        proportions=list(split_ratios.values()),
    )
    result = selected_patients[[subject_id_col]].copy()
    result["split"] = [split_names[index] for index in assignments]
    return result


def build_manifest(
    eligible_records: pd.DataFrame,
    selected_patients: pd.DataFrame,
    patient_splits: pd.DataFrame,
    *,
    columns: ColumnSpec,
    label_cols: Sequence[str],
    seed: int,
    mapping_version: str,
    split_order: Sequence[str],
) -> pd.DataFrame:
    """Build the canonical selected-record manifest."""

    selected_ids = selected_patients[[columns.subject_id]]
    manifest = eligible_records.merge(
        selected_ids, on=columns.subject_id, how="inner", validate="many_to_one"
    ).merge(
        patient_splits, on=columns.subject_id, how="left", validate="many_to_one"
    )
    if manifest["split"].isna().any():
        raise RuntimeError("Internal error: selected records are missing split assignments")
    leakage = manifest.groupby(columns.subject_id)["split"].nunique()
    if leakage.gt(1).any():
        raise RuntimeError("Internal error: at least one patient leaked across splits")

    manifest["seed"] = seed
    manifest["mapping_version"] = mapping_version
    manifest = manifest.rename(
        columns={
            columns.subject_id: "subject_id",
            columns.study_id: "study_id",
            columns.waveform_path: "waveform_path",
        }
    )
    output_columns = [
        "subject_id",
        "study_id",
        "waveform_path",
        *label_cols,
        "split",
        "seed",
        "mapping_version",
    ]
    split_rank = {name: index for index, name in enumerate(split_order)}
    manifest["_split_rank"] = manifest["split"].map(split_rank)
    manifest = manifest.sort_values(
        ["_split_rank", "subject_id", "study_id"], kind="mergesort"
    ).drop(columns="_split_rank")
    return manifest[output_columns].reset_index(drop=True)


def build_qc_report(
    eligible_records: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    eligible_subject_id_col: str,
    label_cols: Sequence[str],
    split_names: Sequence[str],
    target_recordings: int,
) -> pd.DataFrame:
    """Build a tidy QC table with population, prevalence, and split metrics."""

    rows: list[dict[str, object]] = []
    summary_values = {
        "total_eligible_ecgs": len(eligible_records),
        "total_eligible_patients": eligible_records[eligible_subject_id_col].nunique(),
        "target_recordings": target_recordings,
        "total_selected_ecgs": len(manifest),
        "selected_minus_target": len(manifest) - target_recordings,
        "total_selected_patients": manifest["subject_id"].nunique(),
    }
    for metric, value in summary_values.items():
        rows.append({"section": "summary", "metric": metric, "value": value})

    eligible_prevalence = compute_prevalence(eligible_records, label_cols)
    selected_prevalence = compute_prevalence(manifest, label_cols)
    for label in label_cols:
        full_value = float(eligible_prevalence[label])
        selected_value = float(selected_prevalence[label])
        absolute_difference = abs(selected_value - full_value)
        if full_value == 0:
            relative_difference = 0.0 if selected_value == 0 else np.nan
        else:
            relative_difference = absolute_difference / full_value
        rows.append(
            {
                "section": "label_prevalence",
                "metric": "record_level_prevalence",
                "label": label,
                "eligible_positive_count": int(eligible_records[label].sum()),
                "selected_positive_count": int(manifest[label].sum()),
                "eligible_prevalence": full_value,
                "selected_prevalence": selected_value,
                "absolute_prevalence_difference": absolute_difference,
                "relative_prevalence_difference": relative_difference,
            }
        )

    for split in split_names:
        split_frame = manifest.loc[manifest["split"] == split]
        rows.append(
            {
                "section": "split",
                "metric": "split_size",
                "split": split,
                "record_count": len(split_frame),
                "patient_count": split_frame["subject_id"].nunique(),
            }
        )
        for label in label_cols:
            split_prevalence = (
                float(split_frame[label].mean()) if len(split_frame) else np.nan
            )
            rows.append(
                {
                    "section": "split",
                    "metric": "split_label_prevalence",
                    "split": split,
                    "label": label,
                    "split_positive_count": int(split_frame[label].sum()),
                    "split_prevalence": split_prevalence,
                    "split_vs_selected_absolute_difference": abs(
                        split_prevalence - float(selected_prevalence[label])
                    ),
                }
            )

    columns = [
        "section",
        "metric",
        "label",
        "split",
        "value",
        "eligible_positive_count",
        "selected_positive_count",
        "eligible_prevalence",
        "selected_prevalence",
        "absolute_prevalence_difference",
        "relative_prevalence_difference",
        "record_count",
        "patient_count",
        "split_positive_count",
        "split_prevalence",
        "split_vs_selected_absolute_difference",
    ]
    return pd.DataFrame(rows).reindex(columns=columns)


def print_qc_summary(qc: pd.DataFrame) -> None:
    """Print a concise human-readable QC summary to stdout."""

    summary = qc.loc[qc["section"] == "summary", ["metric", "value"]]
    values = dict(zip(summary["metric"], summary["value"], strict=True))
    print(
        "Eligible: "
        f"{int(values['total_eligible_ecgs']):,} ECGs / "
        f"{int(values['total_eligible_patients']):,} patients"
    )
    print(
        "Selected: "
        f"{int(values['total_selected_ecgs']):,} ECGs / "
        f"{int(values['total_selected_patients']):,} patients "
        f"(target difference {int(values['selected_minus_target']):+,d})"
    )

    prevalence = qc.loc[
        qc["section"] == "label_prevalence",
        [
            "label",
            "eligible_prevalence",
            "selected_prevalence",
            "absolute_prevalence_difference",
        ],
    ].copy()
    prevalence.columns = ["label", "eligible", "selected", "abs_diff"]
    print("\nLabel prevalence:")
    print(prevalence.to_string(index=False, float_format=lambda value: f"{value:.5f}"))

    splits = qc.loc[
        (qc["section"] == "split") & (qc["metric"] == "split_size"),
        ["split", "record_count", "patient_count"],
    ].copy()
    splits[["record_count", "patient_count"]] = splits[
        ["record_count", "patient_count"]
    ].astype(int)
    print("\nSplits:")
    print(splits.to_string(index=False))


def _atomic_write_table(frame: pd.DataFrame, path: Path, overwrite: bool) -> None:
    """Write CSV/Parquet output atomically where the filesystem permits."""

    path = path.expanduser()
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists (pass --overwrite): {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    suffix = path.suffix.lower()
    if suffix not in {".csv", ".parquet", ".pq"}:
        raise ValueError(f"Output must end in .csv, .parquet, or .pq: {path}")

    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        if suffix == ".csv":
            frame.to_csv(temporary_path, index=False)
        else:
            frame.to_parquet(temporary_path, index=False)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def save_outputs(
    manifest: pd.DataFrame,
    qc: pd.DataFrame,
    *,
    output: Path,
    qc_output: Path,
    overwrite: bool = False,
) -> None:
    """Persist the manifest and QC table."""

    if output.expanduser().resolve() == qc_output.expanduser().resolve():
        raise ValueError("Manifest and QC output paths must be different")
    _atomic_write_table(manifest, output, overwrite)
    _atomic_write_table(qc, qc_output, overwrite)
    LOGGER.info("Wrote manifest to %s", output)
    LOGGER.info("Wrote QC report to %s", qc_output)


def run_pipeline(
    metadata: pd.DataFrame,
    *,
    columns: ColumnSpec,
    label_cols: Sequence[str],
    target_recordings: int,
    seed: int,
    mapping_version: str,
    split_ratios: Mapping[str, float],
    labels: pd.DataFrame | None = None,
    label_map: pd.DataFrame | None = None,
    diagnosis_code_col: str = "snomed_code",
    map_code_col: str = "source_code",
    map_label_col: str = "target_label",
    require_any_positive_label: bool = False,
    exclude_study_ids: set[str] | None = None,
) -> PipelineResult:
    """Run validation, grouped sampling, splitting, manifesting, and QC."""

    if not mapping_version.strip():
        raise ValueError("mapping_version must be nonblank")
    _validate_proportions(list(split_ratios.values()))

    prepared = prepare_labels(
        metadata,
        label_cols=label_cols,
        study_id_col=columns.study_id,
        labels=labels,
        label_map=label_map,
        diagnosis_code_col=diagnosis_code_col,
        map_code_col=map_code_col,
        map_label_col=map_label_col,
    )
    eligible, cleaning_counts = validate_metadata(
        prepared,
        columns=columns,
        label_cols=label_cols,
        require_any_positive_label=require_any_positive_label,
    )
    cleaning_counts["eligible_records_before_exclusions"] = len(eligible)
    if exclude_study_ids:
        normalized_exclusions = {
            str(value).strip() for value in exclude_study_ids if str(value).strip()
        }
        excluded = eligible[columns.study_id].astype("string").str.strip().isin(
            normalized_exclusions
        )
        cleaning_counts["excluded_unusable_studies"] = int(excluded.sum())
        eligible = eligible.loc[~excluded].reset_index(drop=True)
        if eligible.empty:
            raise ValueError("No eligible ECG records remain after study exclusions")
    else:
        cleaning_counts["excluded_unusable_studies"] = 0
    cleaning_counts["eligible_records"] = len(eligible)
    cleaning_counts["eligible_patients"] = eligible[columns.subject_id].nunique()
    patient_table = build_patient_table(
        eligible, subject_id_col=columns.subject_id, label_cols=label_cols
    )
    selected_patients = select_patient_subset(
        patient_table,
        label_cols=label_cols,
        target_recordings=target_recordings,
        seed=seed,
    )
    patient_splits = assign_patient_splits(
        selected_patients,
        subject_id_col=columns.subject_id,
        label_cols=label_cols,
        split_ratios=split_ratios,
        seed=seed,
    )
    manifest = build_manifest(
        eligible,
        selected_patients,
        patient_splits,
        columns=columns,
        label_cols=label_cols,
        seed=seed,
        mapping_version=mapping_version,
        split_order=list(split_ratios),
    )
    qc = build_qc_report(
        eligible,
        manifest,
        eligible_subject_id_col=columns.subject_id,
        label_cols=label_cols,
        split_names=list(split_ratios),
        target_recordings=target_recordings,
    )
    return PipelineResult(manifest=manifest, qc=qc, cleaning_counts=cleaning_counts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a reproducible patient-aware, multilabel-stratified ECG "
            "subset manifest without downloading waveforms."
        )
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument(
        "--labels",
        type=Path,
        help="Optional wide binary-label table or long diagnosis table keyed by study ID.",
    )
    parser.add_argument(
        "--label-map",
        type=Path,
        help=(
            "Optional CSV/Parquet mapping for a long --labels table. Expected "
            "columns default to source_code and target_label."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qc-output", type=Path, required=True)
    parser.add_argument("--target-recordings", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mapping-version", default="v1")
    parser.add_argument(
        "--label-cols", nargs="+", default=list(DEFAULT_LABEL_COLUMNS), metavar="LABEL"
    )
    parser.add_argument("--subject-id-col", default="subject_id")
    parser.add_argument("--study-id-col", default="study_id")
    parser.add_argument("--waveform-path-col", default="waveform_path")
    parser.add_argument("--diagnosis-code-col", default="snomed_code")
    parser.add_argument("--map-code-col", default="source_code")
    parser.add_argument("--map-label-col", default="target_label")
    parser.add_argument(
        "--split-names", nargs="+", default=list(DEFAULT_SPLIT_NAMES), metavar="NAME"
    )
    parser.add_argument(
        "--split-ratios",
        nargs="+",
        type=float,
        default=list(DEFAULT_SPLIT_RATIOS),
        metavar="RATIO",
    )
    parser.add_argument(
        "--require-any-positive-label",
        action="store_true",
        help="Drop ECGs whose requested label vector is all zero.",
    )
    parser.add_argument(
        "--exclude-studies",
        type=Path,
        help=(
            "Optional CSV/Parquet table of unusable study IDs to remove before "
            "patient-level sampling."
        ),
    )
    parser.add_argument(
        "--exclude-study-id-col",
        default="study_id",
        help="Study-ID column in --exclude-studies (default: study_id).",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Run fully and print QC without writing files."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing output files."
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )

    if len(args.split_names) != len(args.split_ratios):
        parser.error("--split-names and --split-ratios must have the same length")
    if len(set(args.split_names)) != len(args.split_names):
        parser.error("--split-names values must be unique")

    try:
        metadata = load_metadata(args.metadata)
        labels = load_table(args.labels) if args.labels else None
        label_map = load_table(args.label_map) if args.label_map else None
        excluded_studies = None
        if args.exclude_studies:
            exclusions = load_table(args.exclude_studies)
            _require_columns(
                exclusions,
                [args.exclude_study_id_col],
                "Study exclusion table",
            )
            excluded_studies = set(
                exclusions[args.exclude_study_id_col]
                .dropna()
                .astype("string")
                .str.strip()
            )
        result = run_pipeline(
            metadata,
            columns=ColumnSpec(
                subject_id=args.subject_id_col,
                study_id=args.study_id_col,
                waveform_path=args.waveform_path_col,
            ),
            label_cols=args.label_cols,
            target_recordings=args.target_recordings,
            seed=args.seed,
            mapping_version=args.mapping_version,
            split_ratios=dict(zip(args.split_names, args.split_ratios, strict=True)),
            labels=labels,
            label_map=label_map,
            diagnosis_code_col=args.diagnosis_code_col,
            map_code_col=args.map_code_col,
            map_label_col=args.map_label_col,
            require_any_positive_label=args.require_any_positive_label,
            exclude_study_ids=excluded_studies,
        )
        print_qc_summary(result.qc)
        if args.dry_run:
            LOGGER.info("Dry run complete; no files written")
        else:
            save_outputs(
                result.manifest,
                result.qc,
                output=args.output,
                qc_output=args.qc_output,
                overwrite=args.overwrite,
            )
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
