# Reusing the patient-aware stratified sampler

`src/data/build_mimic_subset.py` operates on metadata and binary labels; it does
not download or preprocess waveforms. Despite the historical filename, its
column names, labels, target size, split names, split ratios, seed, and mapping
version are configurable, so the same selection method can be used for another
ECG dataset.

## Required input

Provide one metadata row per ECG with:

- a patient identifier;
- a unique recording identifier;
- a waveform path;
- one binary column per target label.

Labels may instead be supplied in a separate wide table keyed by recording ID,
or as a long diagnosis table together with a code-to-label mapping table.

## Example for another dataset

```bash
python src/data/build_mimic_subset.py \
  --metadata data/raw/other_dataset_metadata.csv \
  --output data/manifests/other_dataset_subset.csv \
  --qc-output data/manifests/other_dataset_subset_qc.csv \
  --subject-id-col patient_id \
  --study-id-col recording_id \
  --waveform-path-col signal_path \
  --label-cols label_a label_b label_c \
  --target-recordings 50000 \
  --split-names train validation test \
  --split-ratios 0.8 0.1 0.1 \
  --mapping-version other-dataset-v1 \
  --seed 42
```

Start with `--dry-run` to inspect eligibility and prevalence without writing
files. The output keeps whole patients together, preserves multilabel
prevalence as closely as possible, records the seed and mapping version, and
includes a QC table comparing the full eligible population with the selected
subset and each split.

## Important boundary

Keep this metadata selection step separate from waveform preprocessing. First
freeze the exact recording manifest and patient-isolated splits; then use a
dataset-specific ingestion adapter to convert signals to the shared contract.
