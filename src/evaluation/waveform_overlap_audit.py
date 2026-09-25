#!/usr/bin/env python3
"""Audit exact and near-duplicate ECGs across every source-target direction.

The audit compares each source development set (train plus validation) against
each different target test set.  Exact matches use a SHA-256 hash after
quantizing the canonical millivolt waveform to one microvolt.  Near matches use
an approximate-nearest-neighbor screen followed by a waveform-level check.  A
target record is removed when any of its five-second windows matches a source
development window.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.signal import resample

from src.data.manifest_signals import CanonicalSignalStore
from src.data.week2_manifest import validate_canonical_manifest
from src.evaluation.ecg_fm_matrix import DEFAULT_DATASETS, parse_named_paths


DEVELOPMENT_SPLITS = ("train", "validation")
TEST_SPLITS = ("test",)
WINDOW_SAMPLES = 2500
FEATURE_SAMPLES_PER_LEAD = 64


@dataclass(frozen=True)
class AuditThresholds:
    """Fixed thresholds used by the two-stage near-duplicate decision."""

    candidate_cosine: float = 0.97
    median_correlation: float = 0.995
    median_nrmse: float = 0.15
    verification_rate_hz: int = 100
    max_lag_ms: int = 100
    top_k: int = 5
    exact_quantization_microvolts: float = 1.0

    def validate(self) -> None:
        if not -1.0 <= self.candidate_cosine <= 1.0:
            raise ValueError("candidate_cosine must be in [-1, 1]")
        if not -1.0 <= self.median_correlation <= 1.0:
            raise ValueError("median_correlation must be in [-1, 1]")
        if self.median_nrmse < 0:
            raise ValueError("median_nrmse cannot be negative")
        if self.verification_rate_hz <= 0 or 500 % self.verification_rate_hz:
            raise ValueError("verification_rate_hz must be a positive divisor of 500")
        if self.max_lag_ms < 0 or self.top_k <= 0:
            raise ValueError("max_lag_ms must be nonnegative and top_k must be positive")
        if self.exact_quantization_microvolts <= 0:
            raise ValueError("exact quantization must be positive")


@dataclass(frozen=True)
class FingerprintCache:
    metadata: pd.DataFrame
    features: np.ndarray
    metadata_path: Path
    features_path: Path


def _valid_window_count(valid_num_samples: int) -> int:
    if valid_num_samples not in (2500, 5000):
        raise ValueError(
            f"valid_num_samples must be 2500 or 5000, got {valid_num_samples}"
        )
    return valid_num_samples // WINDOW_SAMPLES


def _window(signal: np.ndarray, window_index: int) -> np.ndarray:
    start = int(window_index) * WINDOW_SAMPLES
    stop = start + WINDOW_SAMPLES
    result = np.asarray(signal[:, start:stop], dtype=np.float32)
    if result.shape != (12, WINDOW_SAMPLES):
        raise ValueError(f"Window {window_index} has invalid shape {result.shape}")
    return result


def exact_window_hash(
    window: np.ndarray, *, quantization_microvolts: float = 1.0
) -> str:
    """Hash a canonical five-second waveform at a documented ADC resolution."""

    array = np.asarray(window, dtype=np.float64)
    if array.shape != (12, WINDOW_SAMPLES) or not np.isfinite(array).all():
        raise ValueError("Expected a finite (12, 2500) ECG window")
    scale = 1000.0 / float(quantization_microvolts)
    quantized = np.rint(array * scale).astype("<i4", copy=False)
    return hashlib.sha256(quantized.tobytes(order="C")).hexdigest()


def normalized_window(window: np.ndarray) -> np.ndarray:
    """Per-lead standardization used only for morphology comparison."""

    array = np.asarray(window, dtype=np.float32)
    means = array.mean(axis=1, keepdims=True)
    standard_deviations = array.std(axis=1, keepdims=True)
    return np.divide(
        array - means,
        standard_deviations + 1e-8,
        out=np.zeros_like(array),
        where=standard_deviations > 1e-8,
    )


def morphology_feature(window: np.ndarray) -> np.ndarray:
    """Make a compact, gain-invariant feature for candidate retrieval."""

    standardized = normalized_window(window)
    coarse = resample(standardized, FEATURE_SAMPLES_PER_LEAD, axis=1)
    feature = np.asarray(coarse.reshape(-1), dtype=np.float32)
    norm = float(np.linalg.norm(feature))
    if not math.isfinite(norm) or norm <= 1e-8:
        raise ValueError("Cannot fingerprint an entirely flat ECG window")
    return feature / norm


def verify_waveform_similarity(
    left: np.ndarray,
    right: np.ndarray,
    *,
    verification_rate_hz: int = 100,
    max_lag_ms: int = 100,
) -> dict[str, float | int]:
    """Return the best lag-aligned median correlation and normalized RMSE."""

    target_samples = 5 * int(verification_rate_hz)
    first = resample(normalized_window(left), target_samples, axis=1).astype(np.float32)
    second = resample(normalized_window(right), target_samples, axis=1).astype(np.float32)
    max_lag = int(round(max_lag_ms * verification_rate_hz / 1000.0))
    best: dict[str, float | int] | None = None
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a, b = first[:, -lag:], second[:, : target_samples + lag]
        elif lag > 0:
            a, b = first[:, : target_samples - lag], second[:, lag:]
        else:
            a, b = first, second
        numerator = np.sum(a * b, axis=1)
        denominator = np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b, axis=1))
        valid = denominator > 1e-8
        if not valid.any():
            continue
        correlations = numerator[valid] / denominator[valid]
        nrmse = np.sqrt(np.mean((a[valid] - b[valid]) ** 2, axis=1))
        candidate = {
            "median_correlation": float(np.median(correlations)),
            "median_nrmse": float(np.median(nrmse)),
            "lag_samples_at_verification_rate": int(lag),
        }
        if best is None or (
            candidate["median_correlation"], -candidate["median_nrmse"]
        ) > (best["median_correlation"], -best["median_nrmse"]):
            best = candidate
    if best is None:
        return {
            "median_correlation": float("nan"),
            "median_nrmse": float("inf"),
            "lag_samples_at_verification_rate": 0,
        }
    return best


def _select_rows(
    manifest: pd.DataFrame,
    splits: Sequence[str],
    max_records: int | None,
) -> pd.DataFrame:
    selected = manifest.loc[manifest["split"].isin(splits)].copy()
    if max_records is not None:
        if max_records <= 0:
            raise ValueError("max_records must be positive")
        selected = selected.head(max_records).copy()
    if selected.empty:
        raise ValueError(f"No records found for splits {list(splits)}")
    return selected.reset_index(drop=True)


def build_fingerprint_cache(
    manifest: pd.DataFrame,
    signal_root: Path,
    *,
    splits: Sequence[str],
    output_prefix: Path,
    thresholds: AuditThresholds,
    max_records: int | None = None,
    progress_every: int = 1000,
) -> FingerprintCache:
    """Create or reuse disk-backed window features and exact hashes."""

    thresholds.validate()
    metadata_path = output_prefix.with_suffix(".csv")
    features_path = output_prefix.with_suffix(".features.npy")
    config_path = output_prefix.with_suffix(".config.json")
    selected = _select_rows(manifest, splits, max_records)
    expected_config = {
        "dataset": str(selected["dataset"].iloc[0]),
        "splits": list(splits),
        "record_count": len(selected),
        "record_id_digest": hashlib.sha256(
            "\n".join(selected["record_id"].astype(str)).encode()
        ).hexdigest(),
        "feature_samples_per_lead": FEATURE_SAMPLES_PER_LEAD,
        "window_samples": WINDOW_SAMPLES,
        "exact_quantization_microvolts": thresholds.exact_quantization_microvolts,
    }
    if metadata_path.is_file() and features_path.is_file() and config_path.is_file():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing == expected_config:
            metadata = pd.read_csv(metadata_path, low_memory=False)
            features = np.load(features_path, mmap_mode="r")
            if len(metadata) == len(features):
                return FingerprintCache(metadata, features, metadata_path, features_path)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    store = CanonicalSignalStore(manifest, signal_root)
    rows: list[dict[str, Any]] = []
    features: list[np.ndarray] = []
    for position, (_, row) in enumerate(selected.iterrows(), 1):
        signal = store.load_row(row)
        for window_index in range(_valid_window_count(int(row["valid_num_samples"]))):
            segment = _window(signal, window_index)
            rows.append(
                {
                    "dataset": str(row["dataset"]),
                    "record_id": str(row["record_id"]),
                    "split": str(row["split"]),
                    "window_index": window_index,
                    "exact_hash": exact_window_hash(
                        segment,
                        quantization_microvolts=thresholds.exact_quantization_microvolts,
                    ),
                }
            )
            features.append(morphology_feature(segment))
        if progress_every > 0 and position % progress_every == 0:
            print(
                json.dumps(
                    {
                        "event": "FINGERPRINT_PROGRESS",
                        "dataset": store.dataset,
                        "splits": list(splits),
                        "records_complete": position,
                        "records_total": len(selected),
                    }
                ),
                flush=True,
            )
    store.close()
    metadata = pd.DataFrame(rows)
    feature_array = np.stack(features).astype(np.float32)
    metadata.to_csv(metadata_path, index=False)
    np.save(features_path, feature_array, allow_pickle=False)
    config_path.write_text(json.dumps(expected_config, indent=2) + "\n", encoding="utf-8")
    return FingerprintCache(
        metadata,
        np.load(features_path, mmap_mode="r"),
        metadata_path,
        features_path,
    )


def _nearest_neighbors(
    source: np.ndarray,
    target: np.ndarray,
    *,
    top_k: int,
    batch_size: int = 2048,
) -> Iterable[tuple[int, int, float]]:
    """Yield target index, source index, and cosine similarity."""

    source32 = np.ascontiguousarray(source, dtype=np.float32)
    target32 = np.ascontiguousarray(target, dtype=np.float32)
    k = min(top_k, len(source32))
    if k <= 0:
        return
    try:
        import faiss

        index = faiss.IndexHNSWFlat(source32.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 80
        index.hnsw.efSearch = max(64, k * 8)
        index.add(source32)
        for start in range(0, len(target32), batch_size):
            distances, indices = index.search(target32[start : start + batch_size], k)
            for local_target, (row_distances, row_indices) in enumerate(
                zip(distances, indices)
            ):
                for distance, source_index in zip(row_distances, row_indices):
                    if source_index >= 0:
                        yield start + local_target, int(source_index), float(distance)
    except ModuleNotFoundError:
        if len(source32) * len(target32) > 5_000_000:
            raise RuntimeError(
                "faiss-cpu is required for a full overlap audit; install it before rerunning"
            )
        from sklearn.neighbors import NearestNeighbors

        model = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
        model.fit(source32)
        distances, indices = model.kneighbors(target32)
        for target_index, (row_distances, row_indices) in enumerate(
            zip(distances, indices)
        ):
            for distance, source_index in zip(row_distances, row_indices):
                yield target_index, int(source_index), float(1.0 - distance)


def audit_direction(
    source_cache: FingerprintCache,
    target_cache: FingerprintCache,
    source_store: CanonicalSignalStore,
    target_store: CanonicalSignalStore,
    thresholds: AuditThresholds,
) -> pd.DataFrame:
    """Return one best exact/near match for every matched target record."""

    thresholds.validate()
    source_hashes: dict[str, list[int]] = {}
    for index, digest in enumerate(source_cache.metadata["exact_hash"].astype(str)):
        source_hashes.setdefault(digest, []).append(index)

    candidates: dict[int, dict[int, tuple[float, bool]]] = {}
    for target_index, digest in enumerate(target_cache.metadata["exact_hash"].astype(str)):
        for source_index in source_hashes.get(digest, []):
            candidates.setdefault(target_index, {})[source_index] = (1.0, True)
    for target_index, source_index, cosine in _nearest_neighbors(
        source_cache.features,
        target_cache.features,
        top_k=thresholds.top_k,
    ):
        if cosine < thresholds.candidate_cosine:
            continue
        previous = candidates.setdefault(target_index, {}).get(source_index)
        is_exact = bool(previous and previous[1])
        if previous is None or cosine > previous[0]:
            candidates[target_index][source_index] = (cosine, is_exact)

    @lru_cache(maxsize=256)
    def source_segment(record_id: str, window_index: int) -> np.ndarray:
        return _window(source_store.load(record_id), window_index)

    @lru_cache(maxsize=256)
    def target_segment(record_id: str, window_index: int) -> np.ndarray:
        return _window(target_store.load(record_id), window_index)

    accepted: list[dict[str, Any]] = []
    for target_index, source_options in candidates.items():
        target_row = target_cache.metadata.iloc[target_index]
        for source_index, (cosine, is_exact) in source_options.items():
            source_row = source_cache.metadata.iloc[source_index]
            similarity = verify_waveform_similarity(
                source_segment(
                    str(source_row["record_id"]), int(source_row["window_index"])
                ),
                target_segment(
                    str(target_row["record_id"]), int(target_row["window_index"])
                ),
                verification_rate_hz=thresholds.verification_rate_hz,
                max_lag_ms=thresholds.max_lag_ms,
            )
            near = bool(
                similarity["median_correlation"] >= thresholds.median_correlation
                and similarity["median_nrmse"] <= thresholds.median_nrmse
            )
            if not (is_exact or near):
                continue
            accepted.append(
                {
                    "source_dataset": str(source_row["dataset"]),
                    "source_record_id": str(source_row["record_id"]),
                    "source_split": str(source_row["split"]),
                    "source_window_index": int(source_row["window_index"]),
                    "target_dataset": str(target_row["dataset"]),
                    "target_record_id": str(target_row["record_id"]),
                    "target_split": str(target_row["split"]),
                    "target_window_index": int(target_row["window_index"]),
                    "match_type": "exact" if is_exact else "near",
                    "candidate_cosine": cosine,
                    **similarity,
                }
            )
    columns = [
        "source_dataset",
        "source_record_id",
        "source_split",
        "source_window_index",
        "target_dataset",
        "target_record_id",
        "target_split",
        "target_window_index",
        "match_type",
        "candidate_cosine",
        "median_correlation",
        "median_nrmse",
        "lag_samples_at_verification_rate",
    ]
    if not accepted:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(accepted)
    frame["_exact_rank"] = frame["match_type"].eq("exact").astype(int)
    frame = frame.sort_values(
        [
            "target_record_id",
            "_exact_rank",
            "median_correlation",
            "median_nrmse",
        ],
        ascending=[True, False, False, True],
    )
    return frame.drop_duplicates("target_record_id", keep="first").drop(
        columns="_exact_rank"
    )[columns]


def run_cross_dataset_audit(args: argparse.Namespace) -> pd.DataFrame:
    datasets = list(dict.fromkeys(args.datasets))
    if len(datasets) < 2:
        raise ValueError("At least two datasets are required")
    manifest_paths = parse_named_paths(args.manifest)
    signal_roots = parse_named_paths(args.signal_root)
    missing_manifests = [name for name in datasets if name not in manifest_paths]
    missing_roots = [name for name in datasets if name not in signal_roots]
    if missing_manifests or missing_roots:
        raise ValueError(
            f"Missing manifest paths={missing_manifests}; signal roots={missing_roots}"
        )
    thresholds = AuditThresholds(
        candidate_cosine=args.candidate_cosine,
        median_correlation=args.median_correlation,
        median_nrmse=args.median_nrmse,
        verification_rate_hz=args.verification_rate_hz,
        max_lag_ms=args.max_lag_ms,
        top_k=args.top_k,
        exact_quantization_microvolts=args.exact_quantization_microvolts,
    )
    thresholds.validate()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = args.output_dir / "fingerprint_cache"
    manifests: dict[str, pd.DataFrame] = {}
    stores: dict[str, CanonicalSignalStore] = {}
    development: dict[str, FingerprintCache] = {}
    test: dict[str, FingerprintCache] = {}
    for dataset in datasets:
        manifest = validate_canonical_manifest(
            pd.read_csv(manifest_paths[dataset], low_memory=False)
        )
        observed = manifest["dataset"].astype(str).unique().tolist()
        if observed != [dataset]:
            raise ValueError(
                f"Manifest for {dataset!r} contains dataset values {observed}"
            )
        manifests[dataset] = manifest
        stores[dataset] = CanonicalSignalStore(manifest, signal_roots[dataset])
        limit_tag = "all" if args.max_records_per_partition is None else str(
            args.max_records_per_partition
        )
        development[dataset] = build_fingerprint_cache(
            manifest,
            signal_roots[dataset],
            splits=DEVELOPMENT_SPLITS,
            output_prefix=cache_root / f"{dataset}__development__{limit_tag}",
            thresholds=thresholds,
            max_records=args.max_records_per_partition,
            progress_every=args.progress_every,
        )
        test[dataset] = build_fingerprint_cache(
            manifest,
            signal_roots[dataset],
            splits=TEST_SPLITS,
            output_prefix=cache_root / f"{dataset}__test__{limit_tag}",
            thresholds=thresholds,
            max_records=args.max_records_per_partition,
            progress_every=args.progress_every,
        )

    pair_root = args.output_dir / "pair_matches"
    filtered_root = args.output_dir / "deduplicated_manifests"
    pair_root.mkdir(parents=True, exist_ok=True)
    filtered_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    all_matches: list[pd.DataFrame] = []
    for source in datasets:
        for target in datasets:
            if source == target:
                continue
            print(
                json.dumps({"event": "AUDIT_DIRECTION", "source": source, "target": target}),
                flush=True,
            )
            matches = audit_direction(
                development[source],
                test[target],
                stores[source],
                stores[target],
                thresholds,
            )
            matches.to_csv(pair_root / f"{source}__to__{target}.csv", index=False)
            all_matches.append(matches)
            removed = set(matches["target_record_id"].astype(str))
            target_manifest = manifests[target]
            is_removed_test = target_manifest["split"].eq("test") & target_manifest[
                "record_id"
            ].astype(str).isin(removed)
            filtered = target_manifest.loc[~is_removed_test].copy()
            filtered.to_csv(
                filtered_root / f"{source}__to__{target}_week2.csv", index=False
            )
            test_records = int(target_manifest["split"].eq("test").sum())
            exact_records = int(
                matches.loc[matches["match_type"].eq("exact"), "target_record_id"].nunique()
            )
            near_records = int(
                matches.loc[matches["match_type"].eq("near"), "target_record_id"].nunique()
            )
            summaries.append(
                {
                    "source_dataset": source,
                    "target_dataset": target,
                    "source_development_records_audited": int(
                        development[source].metadata["record_id"].nunique()
                    ),
                    "target_test_records_audited": int(
                        test[target].metadata["record_id"].nunique()
                    ),
                    "target_test_records_in_full_manifest": test_records,
                    "exact_target_records_removed": exact_records,
                    "near_target_records_removed": near_records,
                    "total_target_records_removed": len(removed),
                    "fraction_of_audited_target_removed": (
                        len(removed) / max(test[target].metadata["record_id"].nunique(), 1)
                    ),
                    "deduplicated_target_manifest": str(
                        filtered_root / f"{source}__to__{target}_week2.csv"
                    ),
                }
            )

    summary = pd.DataFrame(summaries).sort_values(
        ["source_dataset", "target_dataset"]
    )
    expected_pairs = len(datasets) * (len(datasets) - 1)
    if len(summary) != expected_pairs:
        raise RuntimeError(f"Expected {expected_pairs} directed pairs, produced {len(summary)}")
    summary.to_csv(args.output_dir / "source_target_removal_summary.csv", index=False)
    pd.concat(all_matches, ignore_index=True).to_csv(
        args.output_dir / "all_cross_dataset_matches.csv", index=False
    )
    audit_metadata = {
        "status": "COMPLETE",
        "comparison": "source train+validation versus target test",
        "datasets": datasets,
        "directed_pair_count": expected_pairs,
        "thresholds": asdict(thresholds),
        "max_records_per_partition": args.max_records_per_partition,
        "decision_rule": (
            "Exact one-microvolt hash OR near match satisfying both median lead "
            "correlation and normalized-RMSE thresholds after lag alignment"
        ),
    }
    (args.output_dir / "audit_metadata.json").write_text(
        json.dumps(audit_metadata, indent=2) + "\n", encoding="utf-8"
    )
    for store in stores.values():
        store.close()
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        metavar="DATASET=PATH",
        help="Repeat once per dataset.",
    )
    parser.add_argument(
        "--signal-root",
        action="append",
        default=[],
        metavar="DATASET=PATH",
        help="Repeat once per dataset.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--max-records-per-partition", type=int)
    parser.add_argument("--candidate-cosine", type=float, default=0.97)
    parser.add_argument("--median-correlation", type=float, default=0.995)
    parser.add_argument("--median-nrmse", type=float, default=0.15)
    parser.add_argument("--verification-rate-hz", type=int, default=100)
    parser.add_argument("--max-lag-ms", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--exact-quantization-microvolts", type=float, default=1.0)
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = run_cross_dataset_audit(args)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
