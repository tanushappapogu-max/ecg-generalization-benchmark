from pathlib import Path

import numpy as np
import pandas as pd

from src.data.manifest_signals import CanonicalSignalStore
from src.data.week2_manifest import LABEL_COLUMNS, validate_canonical_manifest
from src.evaluation.waveform_overlap_audit import (
    AuditThresholds,
    audit_direction,
    build_fingerprint_cache,
    exact_window_hash,
    morphology_feature,
    verify_waveform_similarity,
)
from src.models.ecg_founder import preprocess_ecg_founder


def _manifest(dataset: str, root: Path, signals: list[np.ndarray]) -> pd.DataFrame:
    splits = ["train", "train", "validation", "validation", "test", "test"]
    rows = []
    (root / "signals").mkdir(parents=True)
    for index, (split, signal) in enumerate(zip(splits, signals)):
        np.save(root / "signals" / f"{index}.npy", signal, allow_pickle=False)
        value = index % 2
        rows.append(
            {
                "dataset": dataset,
                "record_id": str(index),
                "subject_id": f"{dataset}-{index}",
                "signal_path": f"signals/{index}.npy",
                "storage": "npy",
                "split": split,
                **{label: value for label in LABEL_COLUMNS},
                "valid_num_samples": 5000,
                "mapping_version": "test-v1",
                "split_version": "test-v1",
            }
        )
    return validate_canonical_manifest(pd.DataFrame(rows))


def test_exact_hash_and_morphology_feature_are_deterministic():
    rng = np.random.default_rng(7)
    window = rng.normal(size=(12, 2500)).astype(np.float32)
    assert exact_window_hash(window) == exact_window_hash(window.copy())
    feature = morphology_feature(window)
    assert feature.shape == (12 * 64,)
    assert np.isclose(np.linalg.norm(feature), 1.0)


def test_similarity_accepts_small_gain_and_offset_change():
    rng = np.random.default_rng(8)
    first = rng.normal(size=(12, 2500)).astype(np.float32)
    second = first * 1.02 + 0.03
    result = verify_waveform_similarity(first, second)
    assert result["median_correlation"] > 0.999
    assert result["median_nrmse"] < 0.01


def test_directional_audit_finds_exact_target_record(tmp_path):
    rng = np.random.default_rng(9)
    source_signals = [
        rng.normal(size=(12, 5000)).astype(np.float32) for _ in range(6)
    ]
    target_signals = [
        rng.normal(size=(12, 5000)).astype(np.float32) for _ in range(6)
    ]
    target_signals[4] = source_signals[0].copy()
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_manifest = _manifest("source", source_root, source_signals)
    target_manifest = _manifest("target", target_root, target_signals)
    thresholds = AuditThresholds()
    source_cache = build_fingerprint_cache(
        source_manifest,
        source_root,
        splits=("train", "validation"),
        output_prefix=tmp_path / "source_development",
        thresholds=thresholds,
    )
    target_cache = build_fingerprint_cache(
        target_manifest,
        target_root,
        splits=("test",),
        output_prefix=tmp_path / "target_test",
        thresholds=thresholds,
    )
    matches = audit_direction(
        source_cache,
        target_cache,
        CanonicalSignalStore(source_manifest, source_root),
        CanonicalSignalStore(target_manifest, target_root),
        thresholds,
    )
    match = matches.loc[matches["target_record_id"].eq("4")].iloc[0]
    assert match["source_record_id"] == "0"
    assert match["match_type"] == "exact"


def test_ecgfounder_preprocessing_is_finite_and_global_z_scored():
    time = np.arange(5000, dtype=np.float64) / 500.0
    signal = np.stack(
        [np.sin(2 * np.pi * (1.0 + lead / 10) * time) for lead in range(12)]
    ).astype(np.float32)
    processed = preprocess_ecg_founder(signal)
    assert processed.shape == (12, 5000)
    assert processed.dtype == np.float32
    assert np.isfinite(processed).all()
    assert abs(float(processed.mean())) < 1e-5
    assert np.isclose(float(processed.std()), 1.0, atol=1e-5)
