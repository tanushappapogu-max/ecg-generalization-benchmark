import numpy as np

from src.training.ecg_fm_pipeline import (
    compute_aurocs,
    shared_signal_to_ecg_fm_windows,
)


def test_adapter_creates_two_standardized_windows():
    rng = np.random.default_rng(42)
    signal = rng.normal(size=(12, 5000)).astype(np.float32)
    windows = shared_signal_to_ecg_fm_windows(signal)
    assert windows.shape == (2, 12, 2500)
    recombined = windows.transpose(1, 0, 2).reshape(12, 5000)
    assert np.allclose(recombined.mean(axis=1), 0.0, atol=1e-5)
    assert np.allclose(recombined.std(axis=1), 1.0, atol=1e-5)


def test_adapter_ignores_storage_padding_for_five_second_record():
    signal = np.zeros((12, 5000), dtype=np.float32)
    signal[:, :2500] = np.arange(2500, dtype=np.float32)
    signal[:, 2500:] = 99999.0
    windows = shared_signal_to_ecg_fm_windows(signal, valid_num_samples=2500)
    assert windows.shape == (1, 12, 2500)
    assert np.allclose(windows.mean(axis=2), 0.0, atol=1e-5)


def test_compute_aurocs_reports_undefined_class_as_nan():
    y_true = np.array(
        [
            [0, 0, 0, 0, 0],
            [1, 1, 0, 1, 1],
            [0, 0, 0, 0, 0],
            [1, 1, 0, 1, 1],
        ]
    )
    y_score = y_true * 0.8 + 0.1
    macro, per_class = compute_aurocs(y_true, y_score)
    assert macro == 1.0
    assert np.isnan(per_class["IAVB"])
