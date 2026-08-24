#!/usr/bin/env python3
"""Evaluate the frozen two-class ablation without inventing missing cells.

The runner supports both ECG-FM and InceptionTime checkpoints. It evaluates
every available source checkpoint on every available target test set, emits a
complete source-by-target audit, and measures how much of the five-class
in-domain versus cross-dataset AUROC gap disappears after collapsing the task
to normal versus abnormal.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import torch
from torch import nn

try:
    from src.data.binary_ablation import BINARY_DEFINITION_VERSION, build_binary_manifest
    from src.evaluation.ecg_fm_matrix import DEFAULT_DATASETS, parse_named_paths
    from src.training.binary_ablation_pipeline import build_model, evaluate_binary
    from src.training.ecg_fm_pipeline import ECGManifestDataset, _loader, seed_everything
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.data.binary_ablation import BINARY_DEFINITION_VERSION, build_binary_manifest
    from src.evaluation.ecg_fm_matrix import DEFAULT_DATASETS, parse_named_paths
    from src.training.binary_ablation_pipeline import build_model, evaluate_binary
    from src.training.ecg_fm_pipeline import ECGManifestDataset, _loader, seed_everything


DEFAULT_ARCHITECTURES = ("ecg_fm", "inception_time")


def generalization_gap(rows: pd.DataFrame, metric: str) -> dict[str, float | int]:
    """Summarize diagonal and off-diagonal AUROC from completed cells only."""

    required = {"source_dataset", "target_dataset", "status", metric}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"Matrix rows are missing columns: {missing}")
    complete = rows.loc[rows["status"].eq("COMPLETE")].copy()
    complete[metric] = pd.to_numeric(complete[metric], errors="coerce")
    complete = complete.loc[complete[metric].notna()]
    diagonal = complete.loc[
        complete["source_dataset"].eq(complete["target_dataset"]), metric
    ]
    cross = complete.loc[
        complete["source_dataset"].ne(complete["target_dataset"]), metric
    ]
    diagonal_mean = float(diagonal.mean()) if len(diagonal) else math.nan
    cross_mean = float(cross.mean()) if len(cross) else math.nan
    gap = (
        diagonal_mean - cross_mean
        if math.isfinite(diagonal_mean) and math.isfinite(cross_mean)
        else math.nan
    )
    return {
        "completed_in_domain_cells": int(len(diagonal)),
        "completed_cross_dataset_cells": int(len(cross)),
        "mean_in_domain_auroc": diagonal_mean,
        "mean_cross_dataset_auroc": cross_mean,
        "generalization_gap": gap,
    }


def compare_gaps(five_label_gap: float, binary_gap: float) -> dict[str, float]:
    absolute = five_label_gap - binary_gap
    percentage = (
        100.0 * absolute / five_label_gap
        if math.isfinite(five_label_gap) and five_label_gap != 0
        else math.nan
    )
    return {
        "absolute_gap_disappearance": absolute,
        "percent_gap_disappearance": percentage,
    }


def _write_predictions(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "record_id": result["record_ids"],
            "target_abnormal": result["y_true"].astype(int),
            "probability_abnormal": result["y_score"],
        }
    ).to_csv(path, index=False)


def run_matrix(args: argparse.Namespace) -> pd.DataFrame:
    seed_everything(args.seed)
    datasets = tuple(args.datasets)
    architectures = tuple(args.architectures)
    if len(set(datasets)) != len(datasets):
        raise ValueError("--datasets contains duplicates")
    if len(set(architectures)) != len(architectures):
        raise ValueError("--architectures contains duplicates")
    signal_roots = parse_named_paths(args.signal_root)
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    use_amp = bool(args.mixed_precision and device.type == "cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    targets: dict[str, ECGManifestDataset] = {}
    target_errors: dict[str, str] = {}
    for target in datasets:
        manifest_path = args.manifest_root / f"{target}_week2.csv"
        signal_root = signal_roots.get(target)
        if not manifest_path.is_file():
            target_errors[target] = f"missing manifest: {manifest_path}"
            continue
        if signal_root is None or not signal_root.exists():
            target_errors[target] = f"missing signal root for {target}"
            continue
        try:
            binary = build_binary_manifest(pd.read_csv(manifest_path, low_memory=False))
            targets[target] = ECGManifestDataset(
                binary,
                split="test",
                signal_root=signal_root,
                max_records=args.max_records_per_target,
            )
        except Exception as exc:
            target_errors[target] = f"invalid target data: {exc}"

    rows: list[dict[str, Any]] = []
    for architecture in architectures:
        for source in datasets:
            checkpoint_path = (
                args.source_runs_root / architecture / source / "best_checkpoint.pt"
            )
            if not checkpoint_path.is_file():
                for target in datasets:
                    rows.append(
                        {
                            "architecture": architecture,
                            "task": "normal_vs_abnormal",
                            "binary_definition_version": BINARY_DEFINITION_VERSION,
                            "source_dataset": source,
                            "target_dataset": target,
                            "status": "BLOCKED_MISSING_SOURCE_CHECKPOINT",
                            "detail": str(checkpoint_path),
                            "record_count": 0,
                            "test_auroc": math.nan,
                        }
                    )
                continue

            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            checkpoint_architecture = str(checkpoint.get("architecture", ""))
            checkpoint_source = str(checkpoint.get("source_dataset", ""))
            if checkpoint_architecture != architecture or checkpoint_source != source:
                raise ValueError(
                    f"Checkpoint metadata mismatch in {checkpoint_path}: "
                    f"{checkpoint_architecture}/{checkpoint_source}"
                )
            config = checkpoint.get("run_config", {})
            model, _ = build_model(
                architecture,
                pretrained_checkpoint=args.pretrained_checkpoint,
                device=device,
                dropout=float(config.get("dropout", 0.0)),
                inception_channels=int(config.get("inception_channels", 32)),
                inception_depth=int(config.get("inception_depth", 6)),
            )
            model.load_state_dict(checkpoint["model_state_dict"])

            for target in datasets:
                if target in target_errors:
                    rows.append(
                        {
                            "architecture": architecture,
                            "task": "normal_vs_abnormal",
                            "binary_definition_version": BINARY_DEFINITION_VERSION,
                            "source_dataset": source,
                            "target_dataset": target,
                            "status": "BLOCKED_MISSING_OR_INVALID_TARGET_DATA",
                            "detail": target_errors[target],
                            "record_count": 0,
                            "test_auroc": math.nan,
                        }
                    )
                    continue
                loader = _loader(
                    targets[target],
                    batch_size=args.batch_size,
                    shuffle=False,
                    workers=args.num_workers,
                    seed=args.seed,
                )
                result = evaluate_binary(
                    model,
                    loader,
                    device=device,
                    criterion=nn.BCEWithLogitsLoss(),
                    use_amp=use_amp,
                )
                cell_dir = (
                    args.output_dir
                    / "predictions"
                    / architecture
                    / f"{source}__to__{target}"
                )
                _write_predictions(cell_dir / "test_predictions.csv", result)
                metrics = {
                    "architecture": architecture,
                    "task": "normal_vs_abnormal",
                    "binary_definition_version": BINARY_DEFINITION_VERSION,
                    "source_dataset": source,
                    "target_dataset": target,
                    "status": "COMPLETE",
                    "detail": "",
                    "record_count": len(result["record_ids"]),
                    "test_auroc": result["auroc"],
                }
                (cell_dir / "metrics.json").write_text(
                    json.dumps(metrics, indent=2, allow_nan=True) + "\n",
                    encoding="utf-8",
                )
                rows.append(metrics)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    results = pd.DataFrame(rows)
    expected_cells = len(architectures) * len(datasets) ** 2
    if len(results) != expected_cells:
        raise RuntimeError(f"Expected {expected_cells} rows, produced {len(results)}")
    results.to_csv(args.output_dir / "binary_matrix_long.csv", index=False)

    gap_rows: list[dict[str, Any]] = []
    five_label_paths = parse_named_paths(args.five_label_matrix)
    for architecture in architectures:
        architecture_rows = results.loc[results["architecture"].eq(architecture)]
        binary_summary = generalization_gap(architecture_rows, "test_auroc")
        row: dict[str, Any] = {
            "architecture": architecture,
            "binary_definition_version": BINARY_DEFINITION_VERSION,
            **{f"binary_{key}": value for key, value in binary_summary.items()},
        }
        five_path = five_label_paths.get(architecture)
        if five_path and five_path.is_file():
            five_rows = pd.read_csv(five_path)
            if "architecture" in five_rows:
                five_rows = five_rows.loc[
                    five_rows["architecture"].astype(str).str.lower().eq(architecture)
                ]
            five_summary = generalization_gap(five_rows, "macro_auroc")
            row.update({f"five_label_{key}": value for key, value in five_summary.items()})
            row.update(
                compare_gaps(
                    float(five_summary["generalization_gap"]),
                    float(binary_summary["generalization_gap"]),
                )
            )
        else:
            row.update(
                {
                    "five_label_generalization_gap": math.nan,
                    "absolute_gap_disappearance": math.nan,
                    "percent_gap_disappearance": math.nan,
                }
            )
        gap_rows.append(row)

        architecture_rows.pivot(
            index="source_dataset", columns="target_dataset", values="test_auroc"
        ).reindex(index=datasets, columns=datasets).to_csv(
            args.output_dir / f"{architecture}_binary_auroc_matrix.csv"
        )

    pd.DataFrame(gap_rows).to_csv(
        args.output_dir / "two_class_gap_summary.csv", index=False
    )
    summary = {
        "expected_cells": expected_cells,
        "completed_cells": int(results["status"].eq("COMPLETE").sum()),
        "blocked_cells": int(results["status"].ne("COMPLETE").sum()),
        "architectures": list(architectures),
        "datasets": list(datasets),
        "binary_definition_version": BINARY_DEFINITION_VERSION,
    }
    (args.output_dir / "binary_matrix_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--source-runs-root", type=Path, required=True)
    parser.add_argument("--pretrained-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--signal-root", action="append", default=[], metavar="DATASET=PATH")
    parser.add_argument(
        "--five-label-matrix",
        action="append",
        default=[],
        metavar="ARCHITECTURE=PATH",
    )
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument(
        "--architectures", nargs="+", default=list(DEFAULT_ARCHITECTURES)
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-records-per-target", type=int)
    parser.add_argument(
        "--mixed-precision", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = run_matrix(args)
    print(results["status"].value_counts(dropna=False).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
