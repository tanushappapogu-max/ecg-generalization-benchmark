# Suggested paper captions

## Figure 1 — Five-class AUROC matrices

Five-class macro-AUROC for every completed source-to-target evaluation. Rows indicate the training source and columns indicate the evaluation target. Outlined diagonal cells are in-distribution evaluations; off-diagonal cells measure external generalization. All panels use the same color scale.

## Figure 2 — Generalization summary

In-distribution and cross-dataset macro-AUROC for each completed architecture. Points and error bars in panel A show the mean and SEM across dataset cells. Panel B shows all 20 directional cross-dataset gaps per architecture; boxes indicate the interquartile range and white points show individual source-to-target cells.

## Figure 3 — Target-dataset difficulty

Mean cross-dataset macro-AUROC for each evaluation target, averaging over the four external training sources. Error bars show SEM across sources. MIMIC-IV-ECG had the lowest pooled external-target AUROC, driven most strongly by ECG-FM and Transformer.

## Figure 4 — Per-class transfer

Cross-dataset AUROC for the five harmonized diagnostic classes. Points show the mean across 20 directional transfer cells and error bars show SEM. The figure demonstrates that aggregate generalization gaps are not distributed uniformly across diagnoses.

## Figure 5 — Directional generalization gaps

Mean generalization gap for each ordered dataset pair, averaged across the completed architectures. The gap is defined relative to the corresponding source dataset's in-distribution AUROC. Positive values indicate degraded external performance, while negative values indicate a target AUROC above the source diagonal.

## Figure 6 — Dataset-shift analysis

Observed generalization gaps grouped by the binary population-, device-, and label-schema-shift vectors. Panel A shows individual cells, group means, and SEM. Panel B allocates each gap across active shifts in proportion to the prespecified proposal weights; this allocation is descriptive and should not be interpreted causally.

## Figure 7 — Two-class ablation

Change in mean cross-dataset generalization gap for each completed architecture when the five-class problem is collapsed to normal versus any benchmark abnormality. Each architecture uses the same datasets, frozen splits, preprocessing, and training policy as its five-class experiment; only the target definition, output dimension, and loss change. Percentages show the relative reduction from the corresponding five-class gap.

## Figure 8 — Bootstrap uncertainty

Mean width of the prediction-level bootstrap 95% confidence interval for macro-AUROC, grouped by target dataset and architecture. Each AUROC cell used 1,000 bootstrap replicates; error bars show SEM across the five source models evaluated on each target.

## Figure 9 — Directional asymmetry

Difference between A-to-B and B-to-A transfer AUROC for each unordered dataset pair, averaged across completed architectures. Error bars show SEM across architectures. Nonzero values demonstrate that cross-dataset transfer is directional.

## Figure 10 — Model tradeoff

Mean in-distribution macro-AUROC versus mean cross-dataset macro-AUROC. Error bars show SEM across five diagonal or 20 off-diagonal cells. ECG-FM has the highest in-distribution mean, whereas InceptionTime has the strongest cross-dataset mean among completed models.

## Table 1 — Model performance summary

Model-level in-distribution AUROC, cross-dataset AUROC, and generalization gap reported as mean ± SD across dataset cells.
