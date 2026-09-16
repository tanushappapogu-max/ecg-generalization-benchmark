from argparse import Namespace

import pandas as pd

from src.data.week2_manifest import LABEL_COLUMNS
from src.evaluation.label_mapping_sensitivity import run
from src.training.binary_ablation_pipeline import ARCHITECTURES


def _manifest(dataset, variant):
    rows = []
    for split in ("train", "validation", "test"):
        for index in range(10):
            labels = {label: int((index + offset) % 2 == 0) for offset, label in enumerate(LABEL_COLUMNS)}
            if variant == "strict_normal_v1":
                labels["normal"] = 1 - labels["normal"]
            rows.append(
                {
                    "dataset": dataset,
                    "record_id": f"{dataset}-{split}-{index}",
                    "subject_id": f"{dataset}-{split}-{index}",
                    "signal_path": f"signals/{split}-{index}.npy",
                    "storage": "npy",
                    "split": split,
                    **labels,
                    "valid_num_samples": 5000,
                    "mapping_version": variant,
                    "split_version": "fixed-splits",
                }
            )
    return pd.DataFrame(rows)


def test_rescore_emits_all_variants_architectures_and_cells(tmp_path):
    datasets = ["a", "b"]
    variants = ["consensus_v1", "strict_normal_v1"]
    variant_arguments = []
    for variant in variants:
        root = tmp_path / variant
        root.mkdir()
        for dataset in datasets:
            _manifest(dataset, variant).to_csv(root / f"{dataset}_week2.csv", index=False)
        variant_arguments.append(f"{variant}={root}")

    prediction_arguments = []
    for architecture in ARCHITECTURES:
        root = tmp_path / architecture
        for source in datasets:
            for target in datasets:
                labels = _manifest(target, "consensus_v1").query("split == 'test'")
                predictions = {"record_id": labels["record_id"]}
                for label in LABEL_COLUMNS:
                    predictions[f"probability_{label}"] = labels[label] * 0.8 + 0.1
                path = root / "predictions" / f"{source}__to__{target}"
                path.mkdir(parents=True)
                pd.DataFrame(predictions).to_csv(path / "test_predictions.csv", index=False)
        prediction_arguments.append(f"{architecture}={root}")

    matrix, summary, qc = run(
        Namespace(
            prediction_root=prediction_arguments,
            variant_manifest_root=variant_arguments,
            baseline_variant="consensus_v1",
            datasets=datasets,
            output_dir=tmp_path / "output",
        )
    )
    assert len(matrix) == len(variants) * len(ARCHITECTURES) * len(datasets) ** 2
    assert len(summary) == len(variants) * len(ARCHITECTURES)
    assert len(qc) == len(variants) * len(datasets) * len(LABEL_COLUMNS)
    assert summary["rank_change_vs_baseline"].notna().all()
