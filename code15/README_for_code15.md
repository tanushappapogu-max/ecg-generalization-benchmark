# CODE-15% Download Instructions

This directory includes a Python script for downloading the first three
CODE-15% waveform archives from the official Zenodo record:

- `exams_part0.zip`
- `exams_part1.zip`
- `exams_part2.zip`

The archives contain the ECG waveform files corresponding to
`exams_part0.hdf5`, `exams_part1.hdf5`, and `exams_part2.hdf5`.

## Requirements

- Python 3.8 or newer
- `curl` available on `PATH`
- Approximately 9 GB for the three ZIP archives
- Additional space if the archives will also be extracted

No third-party Python packages are required by the download script.

## Download the archives

Run the following command from the repository root:

```bash
python3 code15/download_code15_zips.py
```

By default, the ZIP files are saved in the same directory as the script:

```text
code15/
├── exams_part0.zip
├── exams_part1.zip
└── exams_part2.zip
```

The downloader resumes interrupted files, retries transient download errors,
and verifies every archive using the MD5 checksum published by Zenodo. An
existing archive is skipped only when its checksum and ZIP structure are valid.

## Choose another download directory

Use `--output-dir` to save the ZIP files elsewhere:

```bash
python3 code15/download_code15_zips.py \
  --output-dir /path/to/code15-downloads
```

## Resume an interrupted download

Run the same command again. The script uses `curl` resume support and retains
an incomplete archive so the next run can continue it instead of restarting.

```bash
python3 code15/download_code15_zips.py
```

After all three files pass checksum and ZIP validation, the script prints:

```text
All requested archives are downloaded and verified.
```

Dataset source: [CODE-15% on Zenodo](https://zenodo.org/records/4916206)
