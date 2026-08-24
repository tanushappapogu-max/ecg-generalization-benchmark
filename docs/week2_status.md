# Tanush Week 2 status

## Completed and verified

- Implemented ECG-FM five-label fine-tuning and recording-level evaluation.
- Implemented the official diagnosis fine-tuning freeze policy and a parameter audit.
- Replaced the shared harness's mock label mapping with versioned mappings.
- Replaced the shared harness's generated 70/10/20 split with frozen manifests.
- Verified all repository tests: 50 passed.
- Built and validated four real source manifests:
  - PTB-XL: 21,799 records; 17,418/2,183/2,198; official patient folds; zero leakage.
  - CPSC2018: 6,877 records; 5,501/688/688; zero detected leakage.
  - Georgia: 10,344 records; 8,275/1,035/1,034; zero detected leakage.
  - MIMIC-IV-ECG: 50,000 records; 40,000/5,000/5,000; 22,035 patients; zero leakage.
- Loaded real Georgia and MIMIC signals through the ECG-FM adapter from every split.

## Still running or externally blocked

- Full ECG-FM training requires a GPU and is run from `tanush_week2_ecg_fm.ipynb`.
- CODE-II is blocked because the assigned teammate's processed Drive folder is empty.
- The MIMIC mapping is frozen as `Sinus rhythm -> normal` under
  `ecg-fm-machine-report-v1`.
- Test AUROC cells must remain empty until their corresponding best-checkpoint runs finish.

## Definition of a completed source run

A source is complete only when its run folder contains `best_checkpoint.pt`,
`training_history.csv`, `test_predictions.csv`, `test_metrics.csv`,
`run_config.json`, `parameter_policy.json`, and `data_summary.json`. The status
table must never replace an unfinished result with an estimated value.
