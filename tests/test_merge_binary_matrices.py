from argparse import Namespace

import pandas as pd

from src.evaluation.merge_binary_matrices import merge
from src.training.binary_ablation_pipeline import ARCHITECTURES


def test_merge_requires_and_summarizes_all_four_architectures(tmp_path):
    datasets = ["a", "b"]
    binary_arguments = []
    five_arguments = []
    for architecture_index, architecture in enumerate(ARCHITECTURES):
        binary_rows = []
        five_rows = []
        for source in datasets:
            for target in datasets:
                diagonal = source == target
                binary_rows.append(
                    {
                        "architecture": architecture,
                        "source_dataset": source,
                        "target_dataset": target,
                        "status": "COMPLETE",
                        "test_auroc": 0.90 if diagonal else 0.85,
                    }
                )
                five_rows.append(
                    {
                        "architecture": "ECG-FM" if architecture == "ecg_fm" else architecture,
                        "source_dataset": source,
                        "target_dataset": target,
                        "status": "COMPLETE",
                        "macro_auroc": 0.90 if diagonal else 0.80,
                    }
                )
        binary_path = tmp_path / f"{architecture}_binary.csv"
        five_path = tmp_path / f"{architecture}_five.csv"
        pd.DataFrame(binary_rows).to_csv(binary_path, index=False)
        pd.DataFrame(five_rows).to_csv(five_path, index=False)
        binary_arguments.append(f"{architecture}={binary_path}")
        five_arguments.append(f"{architecture}={five_path}")

    combined, summary = merge(
        Namespace(
            binary_matrix=binary_arguments,
            five_label_matrix=five_arguments,
            datasets=datasets,
            bootstrap_replicates=100,
            seed=42,
            output_dir=tmp_path / "output",
        )
    )
    assert len(combined) == len(ARCHITECTURES) * len(datasets) ** 2
    assert len(summary) == len(ARCHITECTURES)
    assert summary["absolute_gap_disappearance"].round(8).eq(0.05).all()
    assert summary["percent_gap_disappearance"].round(8).eq(50.0).all()
