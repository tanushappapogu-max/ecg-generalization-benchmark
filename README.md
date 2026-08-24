# ECG Generalization Benchmark

Utilities for building reproducible, leakage-safe ECG dataset manifests for the
cross-dataset generalization benchmark.

The team-wide Drive entry point and exact access instructions for all five
planned datasets are documented in
[`docs/shared_dataset_access.md`](docs/shared_dataset_access.md). Tanush's Week 1
completion record is in [`docs/week1_status.md`](docs/week1_status.md).

The frozen ECG-FM fine-tuning policy is in
[`docs/week2_ecg_fm_protocol.md`](docs/week2_ecg_fm_protocol.md). The 5 x 5
evaluation, two-class ablation, and gap calculation are specified in
[`docs/week3_protocol.md`](docs/week3_protocol.md). Generated checkpoints and
predictions are deliberately ignored by Git; commit the code, manifests,
configuration, notebooks, and documentation instead.
The honest deliverable-by-deliverable execution state, including completed
preliminary cells and external blockers, is in
[`docs/tanush_task_audit.md`](docs/tanush_task_audit.md).

## MIMIC-IV ECG subset manifest

`src/data/build_mimic_subset.py` selects whole patients until the selected ECG
count is as close as possible to the requested target. It uses a deterministic,
patient-level multilabel stratification heuristic for both subset selection and
train/validation/test assignment. It reads metadata and labels only; it does not
download waveforms.

When the metadata already contains the binary benchmark labels:

```bash
python src/data/build_mimic_subset.py \
  --metadata data/raw/mimic_metadata.csv \
  --output data/manifests/mimic_50k.csv \
  --qc-output data/manifests/mimic_50k_qc.csv \
  --target-recordings 50000 \
  --seed 42 \
  --mapping-version v1
```

Use `--labels` when the binary columns live in a separate wide table keyed by
`study_id`. Use `--labels`, `--label-map`, and `--diagnosis-code-col` together
when diagnoses are a long table. The mapping CSV must contain `source_code` and
`target_label`; `target_label` values must match `--label-cols`.

The default label columns are `normal`, `af_afl`, `av_block_1`, `lbbb`, and
`rbbb`. Column names, identifier names, waveform-path name, split names/ratios,
mapping version, target size, and seed are all configurable through the CLI.
Run with `--help` for the full interface, or `--dry-run` to execute selection and
print QC without writing files.

Rows with missing identifiers, blank waveform paths, incomplete binary label
vectors, and redundant duplicate studies are counted and logged. Conflicting
duplicate study rows fail with a validation error instead of being resolved
arbitrarily. All-zero label vectors remain eligible by default because they can
represent a valid "none of the mapped classes" result; pass
`--require-any-positive-label` if the benchmark mapping requires at least one
positive target class per ECG.

## Shared signal contract

`src/data/signal_contract.py` defines the dataset-independent storage format:

- one `.npy` file per ECG;
- lead order `I, II, III, aVR, aVL, aVF, V1-V6`;
- 500 Hz and 5,000 samples (10 seconds);
- physical values in millivolts;
- NumPy `float32` with shape `(12, 5000)`.

Storage is model-agnostic. Model-specific standardization and windowing happen
in the inference adapter, after loading the shared signal.

## Georgia preprocessing

Download the official PhysioNet Georgia mirror with KaggleHub, then run:

```bash
python src/data/preprocess_georgia.py \
  --input-root /path/to/Georgia \
  --output-root data/processed/georgia \
  --mapping data/label_mappings/physionet_challenge_2020_five_labels.csv

python src/data/sanity_check.py \
  --signal-dir data/processed/georgia/signals \
  --report data/processed/georgia/georgia_sanity.csv \
  --plot data/processed/georgia/georgia_five_sample_plot.png \
  --seed 42
```

The completed Georgia run contains 10,344/10,344 passing signals. Nine source
records have one flat precordial lead and are retained with a QC warning; 52
source records are five seconds long and are right-padded according to the
shared contract. `valid_num_samples` and `was_padded` preserve this provenance
for model adapters.

## MIMIC label derivation

`src/data/build_mimic_labels.py` applies ECG-FM's official machine-report
labeler before the subset builder runs. The source-to-benchmark mapping is
versioned in `data/label_mappings/ecg_fm_machine_report_five_labels.csv`.
The frozen MIMIC mapping counts the ECG-FM machine-report label
`Sinus rhythm` as the benchmark Normal class.

## MIMIC full-subset sanity check

`src/data/sanity_check_mimic.py` streams the official WFDB records named by the
manifest, converts each one to the shared contract in memory, and applies
Siddharth's shape, sampling-rate, NaN, and flatline checks without writing a
second 12 GB copy of the signals. It also emits the required label distribution
and deterministic five-record plot:

```bash
python src/data/sanity_check_mimic.py \
  --manifest data/manifests/mimic_50k_v3.csv \
  --waveform-root data/raw/mimic_50k \
  --report data/manifests/mimic_50k_v3_sanity.csv \
  --summary data/manifests/mimic_50k_v3_sanity_summary.json \
  --plot data/manifests/mimic_50k_v3_five_sample_plot.png \
  --failures data/manifests/mimic_50k_v3_failures.csv
```

## Colab checkpoint verification

`notebooks/tanush_week1_pipeline.ipynb` reproduces Georgia preprocessing in the
shared Drive and downloads/loads the pretrained ECG-FM checkpoint. Its adapter
matches ECG-FM's official order: per-lead standardization over the valid signal,
then non-overlapping five-second windows. Five-second Georgia recordings create
one model window rather than treating the storage padding as ECG data.

## Tests

```bash
pytest -q
```

## Week 2 ECG-FM fine-tuning

Week 2 uses frozen canonical manifests instead of the shared draft harness's
mock mapping and record-hash split. Build a manifest with
`src/data/week2_manifest.py`, then run
`src/training/ecg_fm_pipeline.py`. The ECG-FM adapter standardizes each lead
over valid samples, creates one or two five-second windows, and averages window
logits to produce one recording-level prediction.

The primary fine-tuning policy follows fairseq-signals' official ECG diagnosis
configuration: the convolutional feature extractor is frozen, the context
Transformer is fine-tuned from update zero, and a new five-label head is
trained. Every run writes the exact parameter inventory, configuration,
training history, best validation-macro-AUROC checkpoint, held-out predictions,
and per-class/macro test AUROC. See `docs/week2_ecg_fm_protocol.md` for the
commands and interpretation notes.

## Week 3 evaluation and two-class ablation

Evaluate every available ECG-FM source checkpoint while keeping unavailable
cells visible:

```bash
python -m src.evaluation.ecg_fm_matrix \
  --manifest-root data/week2 \
  --source-runs-root data/week2/runs/ecg_fm \
  --pretrained-checkpoint /path/to/mimic_iv_ecg_physionet_pretrained.pt \
  --signal-root ptbxl=/path/to/ptbxl \
  --signal-root cpsc2018=/path/to/cpsc2018 \
  --signal-root georgia=/path/to/georgia \
  --signal-root mimic_iv=/path/to/mimic_iv \
  --output-dir data/week3/results/ecg_fm_five_label
```

Train one frozen normal-versus-abnormal source model by choosing either
`ecg_fm` or `inception_time`:

```bash
python -m src.training.binary_ablation_pipeline \
  --architecture inception_time \
  --manifest data/week2/georgia_week2.csv \
  --signal-root /path/to/georgia \
  --output-dir data/week3/runs/inception_time/georgia
```

After the source checkpoints exist, build both binary matrices and quantify
gap disappearance:

```bash
python -m src.evaluation.binary_matrix \
  --manifest-root data/week2 \
  --source-runs-root data/week3/runs \
  --pretrained-checkpoint /path/to/mimic_iv_ecg_physionet_pretrained.pt \
  --signal-root ptbxl=/path/to/ptbxl \
  --signal-root cpsc2018=/path/to/cpsc2018 \
  --signal-root georgia=/path/to/georgia \
  --signal-root mimic_iv=/path/to/mimic_iv \
  --five-label-matrix ecg_fm=data/week3/results/ecg_fm_five_label/ecg_fm_five_label_matrix_long.csv \
  --output-dir data/week3/results/binary
```

The binary rule and gap formula are frozen in `docs/week3_protocol.md`.
