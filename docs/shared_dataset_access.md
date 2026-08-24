# Shared ECG dataset access

The single team entry point is the shared
[`LSTS/ecg_benchmark`](https://drive.google.com/drive/folders/1VylK35jF1Pu5XxnSQuy64qwjPfsOm1z8)
folder. Dataset artifacts live under
[`processed`](https://drive.google.com/drive/folders/1Y8kO_tsskPb8CUC-s_dofeqQ0pilwJWv).

| Dataset | Status | Direct Drive folder | Index / manifest |
| --- | --- | --- | --- |
| PTB-XL | Available | [processed/ptbxl](https://drive.google.com/drive/folders/1pF66WsuBgZmfB7G6tX1VX8hNAWTQRo-9) | [ptbxl_index.csv](https://drive.google.com/file/d/1kQc_LmGVvrqcw2QFpCrZ2l3GtCQsmOqx/view) |
| CPSC2018 | Available | [processed/cpsc2018](https://drive.google.com/drive/folders/11LDnGkbYq93jpuyTNH89LkGy2dsBaLC6) | [cpsc2018_index.csv](https://drive.google.com/file/d/16MdZe6eQa0uUKOxQKjiaKfAz2TSZn4g5/view) |
| Georgia 12-Lead | Available, packaged | [processed/georgia](https://drive.google.com/drive/folders/1-bvKx1-hqonIOnPcCWS2gfg65uS461sY) | [canonical index](https://drive.google.com/file/d/1EJVf1SsLYG-FVHlsWNghumROdCWaFbMK/view), [harness index](https://drive.google.com/file/d/1aDbdCDEYZhL2nmRIHjnXlGaSmLq_mOB2/view) |
| MIMIC-IV-ECG 50k | Available, packaged | [processed/mimic_iv](https://drive.google.com/drive/folders/1pEEFIo1YxlbBPQE6lWM7gNfcNoXmiDSD) | [root index](https://drive.google.com/file/d/1BQ9r25Q5Y6jiTEnUc40B9d_p53J9FTN6/view), [canonical manifest](https://drive.google.com/file/d/1K7HQewFEo6rziGerGvwqtpTQ_F3slcTi/view) |
| CODE-II 50k | Missing | Not present in the shared Drive as of 2026-08-23 | Leslie's task is still required |

Do not say all five datasets are ready. Four are locatable, and only PTB-XL,
CPSC2018, and Georgia currently have contract-format NumPy artifacts. CODE-II
has not been uploaded. MIMIC is intentionally stored as official WFDB
`.hea`/`.dat` records to avoid creating another approximately 12 GB copy.

The [peer-reviewed CODE-II paper](https://doi.org/10.1038/s41746-026-02704-4)
describes a public **CODE-II-open** subset of 15,000 unique-patient ECGs, but
that is not the proposal's requested 50,000-record CODE-II source. The full
CODE-II collection is restricted. The team must either provide the planned
50,000-record manifest and waveforms or explicitly approve replacing that
source with CODE-II-open; this substitution must not be made silently.

## Shared signal contract

Every model-facing recording must be `float32` with shape `(12, 5000)`, 500 Hz,
10 seconds, millivolts, and lead order
`I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`.

## Use Georgia in Colab

The Georgia folder contains one archive split into Drive-safe pieces. Rebuild it
on the Colab runtime disk rather than writing thousands of small files back to
Drive:

```bash
DRIVE_GEORGIA="/content/drive/MyDrive/LSTS/ecg_benchmark/processed/georgia"
RUNTIME_GEORGIA="/content/ecg_benchmark/processed/georgia"
mkdir -p "$RUNTIME_GEORGIA"
cat "$DRIVE_GEORGIA"/georgia_signals.tar.gz.part-00 \
    "$DRIVE_GEORGIA"/georgia_signals.tar.gz.part-01 \
    "$DRIVE_GEORGIA"/georgia_signals.tar.gz.part-02 \
    "$DRIVE_GEORGIA"/georgia_signals.tar.gz.part-03.sub-* \
    "$DRIVE_GEORGIA"/georgia_signals.tar.gz.part-04.sub-* \
    "$DRIVE_GEORGIA"/georgia_signals.tar.gz.part-05.sub-* \
    "$DRIVE_GEORGIA"/georgia_signals.tar.gz.part-06 \
    > /content/georgia_signals.tar.gz
echo "0138ebdb2bf3f26ad8590c6f9eb5b5c1b876ada476c66e041fa36b5cebc1583d  /content/georgia_signals.tar.gz" | shasum -a 256 -c -
tar -xzf /content/georgia_signals.tar.gz -C "$RUNTIME_GEORGIA"
```

This creates `/content/ecg_benchmark/processed/georgia/signals/*.npy` for the
current runtime. The Drive folder also contains the index, QC report, sanity
report, and deterministic five-record plot.

## Use MIMIC-IV-ECG 50k in Colab

Use `mimic_50k_v3.csv`, not an earlier manifest. Extract all 50 original
waveform shards, then extract the replacement overlay into the same directory:

```bash
DRIVE_MIMIC="/content/drive/MyDrive/LSTS/ecg_benchmark/processed/mimic_iv"
RUNTIME_MIMIC="/content/ecg_benchmark/processed/mimic_iv"
mkdir -p "$RUNTIME_MIMIC"
for shard in "$DRIVE_MIMIC"/mimic_50k_waveforms/mimic_50k_waveforms_*-of-050.tar.gz; do
  tar -xzf "$shard" -C "$RUNTIME_MIMIC"
done
tar -xzf "$DRIVE_MIMIC"/mimic_50k_waveforms/mimic_50k_v3_replacement_overlay/mimic_50k_v3_replacement_overlay_001-of-001.tar.gz \
  -C "$RUNTIME_MIMIC"
```

The extracted root is
`/content/ecg_benchmark/processed/mimic_iv/mimic_50k_waveforms`. The reusable
loader in `src/data/sanity_check_mimic.py` reads each WFDB record, reorders the
source `aVF, aVL` leads to canonical `aVL, aVF`, and creates the shared contract
in memory. Training code should reuse that ingestion boundary instead of
requiring 50,000 additional `.npy` files.

## Current compatibility warning

The draft shared `TrainingHarness.ipynb` expects every dataset to already have
`processed/<dataset>/signals/*.npy` inside Drive. It can read PTB-XL and
CPSC2018 directly. A compatible root-level `georgia_index.csv` is now present,
but the unchanged harness must either point at the runtime extraction path above
or Georgia must be extracted into its Drive folder. MIMIC requires the WFDB
ingestion adapter, and CODE-II cannot run until its subset is delivered. Moving
folders alone does not make those conditions disappear.
