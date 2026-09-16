# ECG benchmark paper figures

This folder generates a publication-ready figure set from the completed five-class result matrices currently available in Google Drive. The visual style uses the exact palette from `deliverables/ECG_Cross_Dataset_Generalization_Workflow_NeurIPS.pptx`:

- navy `#053351`
- slate `#4D6076`
- mauve `#9688A4`
- purple `#664E82`
- deep purple `#341760`
- rose `#BB7E8C`
- light rose `#D6B5C3`

## Generate

```bash
MPLCONFIGDIR=/tmp/ecg-mpl-cache python3 paper_figures/generate_neurips_figures.py
```

All figures are written to `paper_figures/output/` as editable SVG, vector PDF, and 350-dpi PNG. The source CSV exports are retained in `paper_figures/source_data/` for auditability.

## What the uncertainty means

- `SD` describes variation across dataset cells.
- `SEM = SD / sqrt(n)` is shown only where the plot explicitly says `mean ± SEM`.
- Bootstrap intervals come from 1,000 prediction-level bootstrap replicates for each AUROC cell.
- Dataset cells are not independent model-training replicates. Therefore, cell-level SEM must not be described as uncertainty across random seeds or retraining runs.
- The shift attribution is a proposal-weighted descriptive allocation. It is not a causal decomposition.

## Current completeness boundary

The generator validates each architecture before plotting it. The current exports contain complete 25-cell matrices for InceptionTime, Transformer, and ECG-FM. ResNet1D is recorded as incomplete and is not fabricated or silently included. The two-class ablation currently contains only InceptionTime; the proposal requires at least two architectures before that result should be presented as final.

See `output/data_completeness.csv` and `output/figure_manifest.csv` before inserting figures into the paper.
