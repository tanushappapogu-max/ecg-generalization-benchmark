# Week 3 cross-dataset and two-class protocol

## Frozen inputs

All Week 3 evaluations reuse the Week 1 labels and splits and the Week 2 best
checkpoints. No evaluation command is allowed to recompute a split. The five
labels remain `normal`, `af_afl`, `av_block_1`, `lbbb`, and `rbbb`. In MIMIC,
the ECG-FM machine-report label `Sinus rhythm` is frozen as `normal` under
mapping version `ecg-fm-machine-report-v1`.

## ECG-FM 5 x 5 matrix

For each source checkpoint, evaluate the held-out test split from PTB-XL,
CPSC2018, Georgia, MIMIC-IV-ECG, and CODE-II. Each matrix cell contains one
recording-level macro-AUROC and five per-class AUROCs. This creates 25 planned
cells. Missing source checkpoints or target data are written as explicit
blocked rows with NaN metrics; they are never silently removed or replaced by
estimated results.

## Normal-versus-abnormal ablation

The two-class target is versioned as
`normal-vs-any-benchmark-abnormal-v1`:

- Abnormal is the logical OR of AF/AFL, first-degree AV block, LBBB, and RBBB.
- Normal requires the Normal label and none of those four abnormal labels.
- If Normal and an abnormal label co-occur, abnormal wins.
- Records with none of the five benchmark labels are excluded.

ECG-FM and InceptionTime are retrained from each available source using the
same frozen source splits, then evaluated on every available target test split.
ECG-FM uses the same frozen convolutional feature extractor policy as the
five-class experiment. InceptionTime is trained from scratch with all
parameters trainable. Both architectures score only valid five-second windows
and average their logits into one recording-level prediction, so Georgia's
storage padding never becomes model input. Both select the best checkpoint
using validation AUROC
and evaluate held-out test data only after checkpoint selection.

Final paper metrics should be produced on the declared Colab CUDA environment.
Apple MPS may warn that convolution backward is not bitwise deterministic, so
an MPS run is retained as a hardware smoke/execution check rather than the final
reproducibility run unless the team explicitly accepts that limitation.

## Gap calculation

For each architecture and task, the generalization gap is:

`mean in-domain AUROC - mean cross-dataset AUROC`

The amount that disappears in the two-class experiment is:

`five-class gap - two-class gap`

The percentage disappearance is that difference divided by the five-class gap.
Only completed cells with defined AUROC are used, and the number of completed
diagonal and off-diagonal cells is reported beside every summary. A comparison
is not declared complete while required data or checkpoints are missing.

## Required outputs

- `ecg_fm_five_label_matrix_long.csv`
- `ecg_fm_five_label_macro_auroc_matrix.csv`
- `binary_matrix_long.csv`
- one binary AUROC matrix per architecture
- `two_class_gap_summary.csv`
- record-level predictions and JSON metrics for every completed cell
- an audit JSON counting planned, completed, and blocked cells

CODE-II remains an external dependency until its final canonical manifest and
waveform files are placed in the shared benchmark folder. Its missing cells
must stay visibly blocked rather than being presented as finished.
