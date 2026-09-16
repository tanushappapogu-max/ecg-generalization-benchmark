# Week 1 manifests

## MIMIC-IV-ECG 50k subset

`mimic_50k_v3.csv` is the final reproducible, patient-aware sample selected
after full waveform QC with seed 42. It contains 50,000 ECGs from 22,035
patients and has no patient leakage, duplicate studies, NaNs, all-zero signals,
shape errors, or sampling-rate errors.

The split is exactly:

- train: 40,000 ECGs;
- validation: 5,000 ECGs;
- test: 5,000 ECGs.

Post-QC eligible-population versus selected prevalence:

| Label | Full | Selected |
| --- | ---: | ---: |
| Normal proxy | 80.9653% | 80.9480% |
| AF / AFL | 11.8972% | 11.8980% |
| First-degree AV block | 7.3667% | 7.3680% |
| LBBB | 3.7125% | 3.7120% |
| RBBB | 8.2412% | 8.2400% |

These percentages do not add to 100% because this is a multilabel problem: one
ECG can have more than one positive class.

`mimic_50k_v3_qc.csv` contains the eligibility, prevalence, split-size, and
per-split prevalence audit. `mimic_50k_v3_sanity.csv` contains all 50,000
record-level results, and `mimic_50k_v3_sanity_summary.json` records the final
50,000/50,000 PASS. All source records are already 500 Hz, 10 seconds, and mV;
the ingestion path reorders source `aVF, aVL` into canonical `aVL, aVF`.

The Drive upload consists of the original 50 waveform archives plus a separate
696-record replacement overlay. Extract both into the same folder and use only
`mimic_50k_v3.csv` as the training index.

`mimic_labels_full.csv` contains the official ECG-FM machine-report labeler
output reduced to the five benchmark columns. The frozen mapping version is
`ecg-fm-machine-report-v1`; the team decision counts
`Sinus rhythm -> normal` for MIMIC.

## ECG-FM checkpoint

`ecg_fm_checkpoint_qc.json` records the verified checkpoint load and forward
smoke test used by `tanush_week1_pipeline.ipynb` in the LSTS Drive folder.
