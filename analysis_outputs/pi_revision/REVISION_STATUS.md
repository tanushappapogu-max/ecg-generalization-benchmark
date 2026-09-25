# PI revision audit

This audit distinguishes completed analyses from experiments that still require new waveform access or model training. It should not be used to claim that every PI request is complete.

## Scope

The revision analyses reuse saved predictions and do not retrain or rerun the original InceptionTime, ResNet1D, Transformer, or ECG-FM experiments. Any future duplicate exclusion or native-label remapping should filter and rescore the saved outputs. The requested second foundation model is a new comparison rather than a repeat of the original benchmark.

## Original writing requests

- **Introduction references and benchmark-decomposition paragraph:** complete in `paper_submission/FINAL_COPY_PASTE.tex`.
- **Detailed Figure 1 caption and Results cross-references/conclusions:** complete in `paper_submission/FINAL_COPY_PASTE.tex`.
- **Discussion supported by specific examples:** complete in `paper_submission/FINAL_COPY_PASTE.tex`.

## PI experiment requests

1. **Normal-versus-abnormal change for every source-target pair — complete from saved predictions.**
   - `normal_abnormal_cell_gap_changes.csv` contains all 80 architecture-specific off-diagonal cells.
   - `normal_abnormal_pair_summary.csv` contains all 20 ordered dataset pairs averaged over four architectures.
   - The paper reports the complete 5-by-5 directional table.

2. **Detailed ECG-FM source-to-MIMIC analysis — complete as a descriptive audit.**
   - `mimic_ecg_fm_per_class.csv` reports five per-class AUROCs for every source-to-MIMIC cell.
   - `label_prevalence_by_split.csv` reports source training and target test prevalence.
   - `acquisition_metadata.csv` records the acquisition metadata available for each dataset.
   - Hardware models are unavailable for CPSC2018, Georgia, and MIMIC-IV-ECG, so the paper does not make a causal device claim.

3. **Cross-dataset duplicate/near-duplicate waveform audit — not complete.**
   - The current workspace does not contain the complete waveform collections for PTB-XL, CPSC2018, and CODE-15%, so all 20 ordered pairs cannot be audited here.
   - No record-removal count is reported in the paper.
   - Once waveform fingerprints are available, flagged evaluation rows can be removed from saved prediction tables and rescored without retraining the original models.

4. **Controlled label-mapping manipulation — partially complete.**
   - Saved predictions were rescored on the same eligible records under exclusive normal, removal of normal, and merged LBBB/RBBB endpoints.
   - Absolute AUROC changed, while the model order remained ResNet1D, InceptionTime, ECG-FM, Transformer.
   - This does not recover diagnoses excluded before training and is therefore an endpoint-sensitivity analysis, not fully causal native-label remapping.

5. **Second foundation model with disjoint pretraining data — not complete.**
   - No second foundation-model checkpoints or 5-by-5 predictions exist in the repository.
   - The paper explicitly retains this as a limitation instead of implying the confound has been resolved.

6. **Sharper main contribution — complete.**
   - The paper now frames source-target direction and label harmonization as the primary benchmark contribution.
   - Population and device indicators are treated as supporting descriptive analyses.
