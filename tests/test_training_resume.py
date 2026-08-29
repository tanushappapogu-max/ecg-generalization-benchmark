from pathlib import Path

import numpy as np
import pandas as pd

from src.data.week2_manifest import LABEL_COLUMNS
from src.training.baseline_pipeline import build_parser, run_training


def _write_tiny_dataset(root: Path) -> Path:
    signal_dir = root / "signals"
    signal_dir.mkdir(parents=True)
    rows = []
    rng = np.random.default_rng(42)
    for split in ("train", "validation", "test"):
        for index in range(2):
            record_id = f"{split}-{index}"
            np.save(
                signal_dir / f"{record_id}.npy",
                rng.normal(size=(12, 5000)).astype(np.float32),
            )
            row = {
                "dataset": "tiny",
                "record_id": record_id,
                "subject_id": record_id,
                "signal_path": f"signals/{record_id}.npy",
                "storage": "npy",
                "split": split,
                "valid_num_samples": 5000,
                "mapping_version": "test-v1",
                "split_version": "test-v1",
            }
            row.update({label: index for label in LABEL_COLUMNS})
            rows.append(row)
    manifest = root / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


def _args(manifest: Path, output: Path, epochs: int):
    return build_parser().parse_args(
        [
            "--architecture",
            "inception_time",
            "--dataset",
            "tiny",
            "--manifest",
            str(manifest),
            "--signal-root",
            str(manifest.parent),
            "--output-dir",
            str(output),
            "--epochs",
            str(epochs),
            "--patience",
            "5",
            "--batch-size",
            "2",
            "--max-records-per-split",
            "2",
            "--inception-channels",
            "2",
            "--inception-depth",
            "3",
            "--smoke-test",
            "--resume",
            "--no-mixed-precision",
            "--progress-every",
            "0",
        ]
    )


def test_baseline_resume_continues_from_last_completed_epoch(tmp_path):
    manifest = _write_tiny_dataset(tmp_path / "data")
    output = tmp_path / "run"

    run_training(_args(manifest, output, epochs=1))
    assert (output / "last_checkpoint.pt").is_file()
    assert len(pd.read_csv(output / "training_history.csv")) == 1

    # Old runs only had best_checkpoint.pt. They must still resume instead of
    # throwing away the completed epoch.
    (output / "last_checkpoint.pt").unlink()
    run_training(_args(manifest, output, epochs=2))
    history = pd.read_csv(output / "training_history.csv")
    assert history["epoch"].tolist() == [0, 1]

    # New runs resume from last_checkpoint.pt, including optimizer/history.
    run_training(_args(manifest, output, epochs=3))
    history = pd.read_csv(output / "training_history.csv")
    assert history["epoch"].tolist() == [0, 1, 2]
    assert (output / "test_metrics.json").is_file()
