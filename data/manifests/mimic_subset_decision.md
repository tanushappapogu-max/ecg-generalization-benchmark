# MIMIC-IV-ECG 50k subset decision record

Status: technically complete; benchmark-label definition awaits team approval.

## Recommended team decision

Use a deterministic, patient-aware, multilabel-stratified sample of exactly
50,000 ECGs from the official MIMIC-IV-ECG v1.0 population. Preserve the five
benchmark-label prevalences at the recording level, keep every patient in only
one split, use seed 42, and assign 80%/10%/10% train/validation/test splits.

This is preferable to selecting a fixed number per class because an ECG can
have multiple labels and equal class quotas would distort the real MIMIC
prevalence.

## Implemented quality-controlled sample

- Eligible population after waveform QC exclusions: 799,325 ECGs from 161,233
  patients.
- Selected population: 50,000 ECGs from 22,035 patients.
- Train: 40,000 ECGs from 17,459 patients.
- Validation: 5,000 ECGs from 2,249 patients.
- Test: 5,000 ECGs from 2,327 patients.
- Duplicate studies: 0.
- Patients appearing in multiple splits: 0.
- Random seed: 42.
- Final manifest SHA-256:
  `1f7fc98e55d62913cde62c1dc65ca84db802d3d30c3b86fdf8615f8fca6b26e2`.
- Siddharth-compatible sanity check: 50,000/50,000 passed.
- Source signals: 500 Hz, 10 seconds, 12 leads, millivolts.
- Shared-contract correction: reorder source `aVF, aVL` to canonical
  `aVL, aVF`; no unit conversion or resampling is required.

| Label | Full population | Selected sample | Absolute difference |
| --- | ---: | ---: | ---: |
| Normal proxy | 80.9653% | 80.9480% | 0.0173 percentage points |
| AF / AFL | 11.8972% | 11.8980% | 0.0008 percentage points |
| First-degree AV block | 7.3667% | 7.3680% | 0.0013 percentage points |
| LBBB | 3.7125% | 3.7120% | 0.0005 percentage points |
| RBBB | 8.2412% | 8.2400% | 0.0012 percentage points |

Percentages do not sum to 100% because this is a multilabel problem.

## Reproducibility artifacts

- `mimic_50k_v3.csv`: final selected IDs, labels, split, seed, mapping version,
  and official waveform path.
- `mimic_50k_v3_qc.csv`: post-QC eligible/selected prevalence and split audit.
- `mimic_50k_v3_sanity.csv` and `mimic_50k_v3_sanity_summary.json`: full
  50,000-record sanity results.
- `mimic_50k_v3_replacement_overlay.csv`: the 696 valid records replacing
  unusable records in the original 50-archive upload.
- `mimic_50k_waveform_shards.csv`: archive inventory for the downloaded raw
  WFDB records.

The original 50 archives remain usable. Extract them, then extract the one
replacement-overlay archive into the same root, and train strictly from
`mimic_50k_v3.csv`; the 696 rejected original records remain on disk but are
not referenced by the final manifest.

An end-to-end deterministic rebuild was verified from the official 800,035-row
record list and saved label table. `build_mimic_subset.py` reproduced the
initial manifest byte-for-byte; `repair_mimic_subset.py` then reproduced the
696-record first repair and 13-record second repair byte-for-byte. The rebuilt
final manifest matched the SHA-256 above.

## Frozen Normal definition

The team decision counts the ECG-FM report label `Sinus rhythm` as MIMIC's
benchmark Normal definition. The frozen mapping version is
`ecg-fm-machine-report-v1`. If a later mapping revision changes the label rule,
rerun `build_mimic_subset.py` with the new label columns and mapping version;
the pipeline is designed for that rebuild.
