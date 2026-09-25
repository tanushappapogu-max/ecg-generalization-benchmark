#!/usr/bin/env python3
"""Evaluate every ECGFounder source checkpoint on every benchmark target."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from torch import nn

from src.data.week2_manifest import validate_canonical_manifest
from src.evaluation.ecg_fm_matrix import DEFAULT_DATASETS, parse_named_paths
from src.models.ecg_founder import build_ecg_founder
from src.training.ecg_founder_pipeline import (
    CLASS_NAMES,
    ECGFounderManifestDataset,
    PREPROCESSING_VERSION,
    _loader,
    evaluate,
    seed_everything,
    write_predictions,
)


def matrix_from_long(rows: pd.DataFrame, datasets: Sequence[str]) -> pd.DataFrame:
    return (
        rows.pivot(index="source_dataset", columns="target_dataset", values="macro_auroc")
        .reindex(index=datasets, columns=datasets)
        .rename_axis(index="source", columns="target")
    )


def run_matrix(args: argparse.Namespace) -> pd.DataFrame:
    seed_everything(args.seed)
    datasets = list(dict.fromkeys(args.datasets))
    if not datasets:
        raise ValueError("At least one dataset is required")
    signal_roots = parse_named_paths(args.signal_root)
    missing_roots = [dataset for dataset in datasets if dataset not in signal_roots]
    if missing_roots:
        raise ValueError(f"Missing signal roots for {missing_roots}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifests: dict[str, pd.DataFrame] = {}
    target_errors: dict[str, str] = {}
    for target in datasets:
        path = args.manifest_root / f"{target}_week2.csv"
        try:
            manifests[target] = validate_canonical_manifest(
                pd.read_csv(path, low_memory=False)
            )
        except Exception as exc:
            target_errors[target] = f"{type(exc).__name__}: {exc}"

    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    use_amp = bool(args.mixed_precision and device.type == "cuda")
    rows: list[dict[str, object]] = []
    for source in datasets:
        checkpoint_path = args.source_runs_root / source / "best_checkpoint.pt"
        if not checkpoint_path.is_file():
            for target in datasets:
                rows.append(
                    {
                        "architecture": "ECGFounder",
                        "task": "five_label",
                        "source_dataset": source,
                        "target_dataset": target,
                        "status": "BLOCKED_MISSING_SOURCE_CHECKPOINT",
                        "detail": str(checkpoint_path),
                        "record_count": 0,
                        "macro_auroc": math.nan,
                        **{f"auroc_{name}": math.nan for name in CLASS_NAMES},
                    }
                )
            continue
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        checkpoint_source = str(checkpoint.get("dataset", ""))
        if checkpoint_source and checkpoint_source != source:
            raise ValueError(
                f"Checkpoint {checkpoint_path} says dataset={checkpoint_source!r}, "
                f"not {source!r}"
            )
        model, _ = build_ecg_founder(
            repository_root=args.ecg_founder_repository,
            checkpoint_path=args.pretrained_checkpoint,
            device=device,
            num_labels=5,
            linear_probe=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        for target in datasets:
            if target in target_errors:
                rows.append(
                    {
                        "architecture": "ECGFounder",
                        "task": "five_label",
                        "source_dataset": source,
                        "target_dataset": target,
                        "status": "BLOCKED_MISSING_OR_INVALID_TARGET_DATA",
                        "detail": target_errors[target],
                        "record_count": 0,
                        "macro_auroc": math.nan,
                        **{f"auroc_{name}": math.nan for name in CLASS_NAMES},
                    }
                )
                continue
            target_dataset = ECGFounderManifestDataset(
                manifests[target],
                split="test",
                signal_root=signal_roots[target],
                preprocessing_cache=args.preprocessing_cache,
                max_records=args.max_records_per_target,
            )
            loader = _loader(
                target_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                workers=args.num_workers,
                seed=args.seed,
            )
            result = evaluate(
                model,
                loader,
                device=device,
                criterion=nn.BCEWithLogitsLoss(),
                use_amp=use_amp,
            )
            cell_dir = args.output_dir / "predictions" / f"{source}__to__{target}"
            write_predictions(cell_dir / "test_predictions.csv", result)
            metrics = {
                "architecture": "ECGFounder",
                "task": "five_label",
                "source_dataset": source,
                "target_dataset": target,
                "status": "COMPLETE",
                "detail": "",
                "record_count": len(result["record_ids"]),
                "macro_auroc": result["macro_auroc"],
                **{
                    f"auroc_{name}": result["per_class_auroc"][name]
                    for name in CLASS_NAMES
                },
            }
            cell_dir.mkdir(parents=True, exist_ok=True)
            (cell_dir / "metrics.json").write_text(
                json.dumps(metrics, indent=2, allow_nan=True) + "\n",
                encoding="utf-8",
            )
            rows.append(metrics)

    results = pd.DataFrame(rows)
    expected_cells = len(datasets) ** 2
    if len(results) != expected_cells:
        raise RuntimeError(f"Expected {expected_cells} matrix rows, produced {len(results)}")
    results.to_csv(
        args.output_dir / "ecg_founder_five_label_matrix_long.csv", index=False
    )
    matrix_from_long(results, datasets).to_csv(
        args.output_dir / "ecg_founder_five_label_macro_auroc_matrix.csv"
    )
    summary = {
        "architecture": "ECGFounder",
        "task": "five_label",
        "preprocessing": PREPROCESSING_VERSION,
        "expected_cells": expected_cells,
        "completed_cells": int(results["status"].eq("COMPLETE").sum()),
        "blocked_cells": int(results["status"].ne("COMPLETE").sum()),
        "datasets": datasets,
        "published_pretraining_overlap_with_benchmark": False,
    }
    (args.output_dir / "matrix_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if args.fail_on_missing and summary["blocked_cells"]:
        raise RuntimeError(f"Matrix has {summary['blocked_cells']} blocked cells")
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-runs-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument(
        "--signal-root",
        action="append",
        default=[],
        metavar="DATASET=PATH",
        help="Repeat once for every target dataset.",
    )
    parser.add_argument("--ecg-founder-repository", type=Path, required=True)
    parser.add_argument("--pretrained-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preprocessing-cache", type=Path)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-records-per-target", type=int)
    parser.add_argument(
        "--mixed-precision", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--fail-on-missing", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    results = run_matrix(args)
    print(results[["source_dataset", "target_dataset", "status", "macro_auroc"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
