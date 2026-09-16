# Week 2 writing draft

## Discussion notes to finalize after the five runs

Week 2 establishes the in-distribution reference performance for ECG-FM before
any cross-dataset evaluation is performed. The five diagonal scores should be
compared alongside class prevalence and sample size, because a higher source
AUROC does not by itself show better generalization. In Week 3, each transfer
score will be interpreted relative to the corresponding Week 2 diagonal rather
than as an isolated number. Any unusually strong MIMIC result also needs to be
read in light of ECG-FM's pretraining exposure to MIMIC-IV-ECG. Specific claims
about which dataset or diagnosis performs best must be added only after all
held-out results have been produced.

## Limitations

This benchmark only aligns five diagnoses, so it does not capture the full
range of abnormalities represented in each source dataset. Differences in
labeling practices, patient populations, and recording equipment remain partly
mixed together even though the analysis groups them into population, device,
and label-schema shifts. CPSC2018 and Georgia do not expose patient identifiers
in the processed indexes available to us, which limits our ability to prove
that their record-level splits contain no repeat-patient leakage. ECG-FM was
pretrained on MIMIC-IV-ECG and PhysioNet data, giving it source exposure that
the from-scratch baselines do not share. Finally, subsampling MIMIC-IV-ECG and
CODE-II makes the experiments practical but may leave out rare cases present in
the complete datasets.
