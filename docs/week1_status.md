# Tanush Week 1 status

This checklist follows the Week 1 tasks in the team proposal.

## 1. Decide how to subset MIMIC-IV-ECG

Implemented: select exactly 50,000 recordings using deterministic,
patient-aware multilabel stratification, preserve the five-label recording
prevalence, keep patients in one split only, use seed 42, and split
80%/10%/10% into train/validation/test. The final sample contains 50,000 ECGs
from 22,035 patients with 40,000/5,000/5,000 recordings and zero patient
leakage.

Team decision frozen: the ECG-FM machine-report label `Sinus rhythm` counts as
the benchmark's MIMIC Normal class under mapping version
`ecg-fm-machine-report-v1`.

## 2. Download and preprocess Georgia 12-Lead

Complete. All 10,344 official recordings were converted to the shared contract:
canonical 12-lead order, 500 Hz, 5,000 samples, 10 seconds, millivolts, and
`float32` shape `(12, 5000)`. All 10,344 passed the full sanity check. Nine
records contain one flat source lead and remain documented warnings. Fifty-two
five-second source records are right-padded, with `valid_num_samples=2500` and
`was_padded=true` recorded so model adapters do not treat padding as signal.

## 3. Build the MIMIC-IV-ECG 50k subset

Complete with the official PhysioNet WFDB waveform records. The final manifest
is `mimic_50k_v3.csv`, with selected patient/study IDs, official waveform path,
five labels, split, seed, and mapping version. All 50,000 selected records are
present after applying the 696-record replacement overlay. The selected label
prevalences differ from the eligible population by at most 0.0173 percentage
points across the five current labels.

The exact rebuild was rerun from the 800,035-row official record list. The
initial deterministic sample matched byte-for-byte, the first QC repair replaced
696 unusable waveforms, and the second repair replaced 13 remaining NaN
waveforms. The rebuilt final manifest matched `mimic_50k_v3.csv` byte-for-byte
with SHA-256
`1f7fc98e55d62913cde62c1dc65ca84db802d3d30c3b86fdf8615f8fca6b26e2`.

## 4. Pass Siddharth's sanity check and document corrections

Complete on the full datasets, not only the five plotted samples:

- Georgia: 10,344/10,344 passed; no resampling or unit conversion; 9 flat-lead
  warnings; 52 documented padding operations.
- MIMIC: 50,000/50,000 passed; all sources are 500 Hz, 10 seconds, and mV; all
  records require the source `aVF, aVL` order to be changed to canonical
  `aVL, aVF`; no unit conversion, resampling, padding, or truncation.

The check covers tensor shape, sampling rate, NaN/infinity, flatline status,
label distribution, and a deterministic plot of five recordings.

## 5. Load ECG-FM in Colab and confirm its input

Complete. The official `mimic_iv_ecg_physionet_pretrained.pt` checkpoint loads
as `Wav2Vec2CMSCModel` with 90,883,072 parameters. A T4 GPU forward smoke test
passed with one `(1, 12, 2500)` five-second window and returned encoded features
with shape `(1, 156, 768)`.

The adapter first standardizes each lead over the valid samples and then creates
non-overlapping five-second, 2,500-sample windows. A 10-second recording yields
two windows; a genuine five-second Georgia recording yields one. The model
adapter is separate from the shared storage contract.

## Shared locations

- Drive root: [LSTS/ecg_benchmark](https://drive.google.com/drive/folders/1VylK35jF1Pu5XxnSQuy64qwjPfsOm1z8)
- Dataset map: `docs/shared_dataset_access.md`
- Georgia: [processed/georgia](https://drive.google.com/drive/folders/1-bvKx1-hqonIOnPcCWS2gfg65uS461sY)
- MIMIC: [processed/mimic_iv](https://drive.google.com/drive/folders/1pEEFIo1YxlbBPQE6lWM7gNfcNoXmiDSD)
- Colab: [tanush_week1_pipeline.ipynb](https://colab.research.google.com/drive/18AzgWTDQfmgbVQ0qUAu0M4R9f-OnrdFp)
