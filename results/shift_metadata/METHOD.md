# Dataset shift-vector method

The committed `shift_vectors.csv` contains the 20 ordered off-diagonal pairs among PTB-XL, MIMIC-IV-ECG, CPSC2018, Georgia 12-Lead, and CODE-15%.

## Population shift (PS)

Age is represented by the shared bins `0-44`, `45-74`, and `75+`; sex is represented as `female` and `male`. Unknown age/sex counts are retained in the source table for auditability but excluded from the demographic probability distributions. Each non-missing category receives a Jeffreys pseudocount of 0.5 before normalization.

For ordered pair `i -> j`, the demographic divergence is

`D_demo(i || j) = D_KL(age_i || age_j) + D_KL(sex_i || sex_j)`.

Adding the marginal KL values is the KL divergence of the age/sex product-of-marginals model. This is used because the published MIMIC aggregate provides marginal age and sex counts but not their joint cross-tabulation. Natural logarithms are used, so values are in nats.

`PS = 1` when `D_demo > 0.10 nats`. The 0.10-nat threshold was fixed without using model AUROC results. It is intentionally above tiny sampling fluctuations in these large cohorts while still identifying a material demographic distribution change. Because KL is directed, `PS(i -> j)` can differ from `PS(j -> i)`.

## Device shift (DS)

`DS = 1` when the distributed sampling rate differs or when both hardware families are documented and differ. Unknown hardware is not treated as proof of a difference. With the currently documented metadata, all CODE-15% comparisons have `DS = 1` because CODE-15% is distributed at 400 Hz while the other four datasets are at 500 Hz; other pairs have `DS = 0`.

## Label shift (LS)

`LS = 1` when the raw coding convention differs. The conventions used here are SCP-ECG (PTB-XL), machine-report text (MIMIC-IV-ECG), the custom CPSC 9-class schema, SNOMED-CT (Georgia), and CODE custom binary labels. Therefore every off-diagonal pair has `LS = 1`, even though preprocessing later maps all sources into the same five benchmark targets.

## Data provenance and limitation

PTB-XL, CPSC2018, Georgia, and CODE-15% counts were calculated from record-level public metadata. MIMIC counts come from the published full-cohort table covering 800,034 ECGs because the linked age/sex file is credentialed. They do **not** describe the current 50,000-record benchmark manifest exactly. Recompute the MIMIC row from the selected `study_id` values after credentialed MIMIC-IV-ECG-Ext-ICD metadata is available.

Rebuild the pair table with:

```bash
python3 -m src.evaluation.shift_vectors \
  --demographics data/shift_metadata/demographic_counts.csv \
  --output results/shift_metadata/shift_vectors.csv
```
