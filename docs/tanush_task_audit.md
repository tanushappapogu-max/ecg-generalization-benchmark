# Tanush task audit

Updated: 2026-08-23

## Week 1

| Deliverable | State | Evidence |
| --- | --- | --- |
| Frozen MIMIC 50,000-record patient-aware 80/10/10 sample | Complete | 50,000 records; 40,000/5,000/5,000; 22,035 patients; zero split leakage |
| MIMIC waveform sanity check | Complete | 50,000/50,000 records passed |
| Georgia download, preprocessing, and sanity check | Complete | 10,344/10,344 records passed |
| ECG-FM checkpoint load and adapter forward pass in Colab | Complete | T4 checkpoint and forward smoke test passed |
| MIMIC Normal label decision | Complete | `Sinus rhythm -> normal`, mapping version `ecg-fm-machine-report-v1` |

## Week 2

| Deliverable | State | Evidence or blocker |
| --- | --- | --- |
| ECG-FM fine-tuning pipeline and frozen/trainable parameter audit | Complete | Repository implementation and automated tests |
| Frozen manifests for PTB-XL, CPSC2018, Georgia, and MIMIC | Complete | 21,799; 6,877; 10,344; and 50,000 records respectively |
| PTB-XL ECG-FM full fine-tuning | Running | 21,799/21,799 raw records downloaded; contract check passed; real T4 run launched |
| CPSC2018 ECG-FM full fine-tuning | Waiting on active download | Official 6,877-record waveform download is running |
| Georgia ECG-FM full fine-tuning | Ready, not complete | Manifest and all 10,344 standardized signals are available |
| MIMIC ECG-FM full fine-tuning | Ready, not complete | Exact 50,000 records are available locally; Colab transfer remains the bottleneck |
| CODE-II ECG-FM full fine-tuning | Blocked externally | No proposal-compatible 50,000-record CODE-II manifest or waveforms have been supplied; the public CODE-II-open set has only 15,000 unique-patient ECGs and cannot be substituted without a team decision |

## Week 3

| Deliverable | State | Evidence or blocker |
| --- | --- | --- |
| ECG-FM five-label 5 x 5 evaluator | Complete | Produces all 25 audit rows and preserves missing cells as blocked |
| Frozen two-class target | Complete | Abnormal is OR of the four abnormal benchmark labels; abnormal wins co-occurrence; all-zero rows excluded |
| ECG-FM and InceptionTime two-class training/evaluation | Complete as code | Both pipelines train source checkpoints and evaluate target test splits |
| Georgia InceptionTime two-class source run | Complete, preliminary hardware run | Early-stopped after epoch 26; Georgia test AUROC 0.983885 on 403 records; Apple MPS backward reported a nondeterministic kernel, so repeat on CUDA or CPU for final reproducibility |
| Two-architecture binary matrices and gap-disappearance calculation | Complete as code; partial numerically | Produces all 50 planned audit rows; 2 cells are complete (Georgia to Georgia 0.983885 and Georgia to MIMIC 0.858661) and 48 retain explicit blockers |
| Final Week 3 numerical results | Not complete | Requires finished Week 2 checkpoints, binary retraining runs, and CODE-II data |

## Verification and repository state

- Automated tests: 50 passed.
- No files are staged, committed, or pushed by Codex.
- Large waveform data, checkpoints, predictions, and generated runs are ignored by Git.
- The Drive code bundle contains code, frozen manifests, configs, notebooks,
  documentation, and tests only; it excludes raw waveforms and generated model
  checkpoints.
