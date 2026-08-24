# Week 2 ECG-FM in-distribution protocol

## Purpose

Train one ECG-FM model per source dataset and measure the diagonal entries of
the later cross-dataset matrix. Each checkpoint is evaluated only on its own
source dataset's frozen test split in Week 2. Week 3 must reuse these exact
checkpoints for off-diagonal evaluation.

## Five labels

The fixed output order is `normal`, `af_afl`, `av_block_1`, `lbbb`, `rbbb`.
The paper-facing names are NSR, AFIB/AFL, IAVB, LBBB, and RBBB.

PTB-XL uses SCP codes `NORM`, `AFIB`/`AFLT`, `1AVB`, `CLBBB`, and `CRBBB`.
CPSC2018 and Georgia use the versioned SNOMED mapping in
`data/label_mappings/physionet_challenge_2020_five_labels.csv`. MIMIC uses the
frozen v3 manifest produced from the ECG-FM machine-report labeler. The MIMIC
Normal definition is frozen as ECG-FM's machine-report label “Sinus rhythm”
under mapping version `ecg-fm-machine-report-v1`.

## Frozen splits

- PTB-XL: official patient-stratified folds 1–8 train, 9 validation, 10 test.
- MIMIC-IV-ECG: the v3 patient-aware 40,000/5,000/5,000 split, seed 42.
- CPSC2018: deterministic multilabel 80/10/10 split, seed 42. The processed
  index contains no patient identifier, so record IDs are the grouping unit.
- Georgia: deterministic multilabel 80/10/10 split, seed 42. The processed
  index contains no patient identifier, so record IDs are the grouping unit.
- CODE-II: must use Leslie's final 50,000-record manifest and patient grouping.
  The processed CODE-II folder was empty when this protocol was prepared; no
  split or result may be fabricated in its absence.

All architectures must consume these same manifest files. Training code is not
allowed to recompute splits.

## Input adapter and recording aggregation

Stored signals follow the shared `(12, 5000)`, 500 Hz, 10-second, millivolt,
`float32` contract. The ECG-FM adapter standardizes each lead using only valid
samples, then creates non-overlapping `(12, 2500)` five-second windows. A true
five-second Georgia signal creates one valid window; its storage padding is
never treated as ECG. During both training and evaluation, valid window logits
are averaged to produce one prediction and one loss contribution per recording.

## Frozen versus trained parameters

The primary policy matches the official fairseq-signals ECG diagnosis config:

- Frozen: ECG-FM convolutional feature extractor (`feature_grad_mult=0.0`).
- Trained from update zero: post-extraction projection/layer normalization,
  positional convolution, and context Transformer (`freeze_finetune_updates=0`).
- Newly initialized and trained: five-output linear classification head.
- Removed: quantizer and other pretraining-only projection heads.

Every run writes `parameter_policy.json` with parameter names and counts. The
optimizer only receives parameters whose `requires_grad` flag is true.

## Optimization and checkpoint selection

- Positive-weighted `BCEWithLogitsLoss`, using weights computed from that
  source dataset's training split only.
- AdamW, learning rate `1e-6`, betas `(0.9, 0.98)`, weight decay `1e-4`.
- Mixed precision on CUDA, gradient accumulation 8, gradient clipping 1.0.
- Seed 42 and deterministic PyTorch settings.
- Early stopping patience 10 on validation macro-AUROC.
- The held-out test split is evaluated once using the best validation checkpoint.

Batch size 4 is the Colab-safe starting value. It may be reduced after an OOM
only if gradient accumulation is increased to preserve the effective batch.

## Required run artifacts

Each source run must contain:

- `run_config.json`
- `data_summary.json`
- `parameter_policy.json`
- `training_history.csv`
- `best_checkpoint.pt`
- `test_predictions.csv`
- `test_metrics.csv` and `test_metrics.json`

The final Week 2 table contains one row per source and the five per-class AUROCs
plus macro-AUROC. Undefined AUROC is reported as NaN rather than replaced with
a made-up value.

## Interpretation warning

ECG-FM was pretrained on MIMIC-IV-ECG and PhysioNet 2021. Its MIMIC diagonal,
and potentially related PhysioNet datasets, therefore have pretraining exposure
that the from-scratch baselines do not. This must be stated when comparing the
architectures and in the paper's limitations.
