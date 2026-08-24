# Tanush Week 3 status

## Implemented and verified

- Frozen MIMIC `Sinus rhythm -> normal` as a final mapping decision.
- Implemented the complete ECG-FM five-label matrix evaluator.
- Implemented the frozen normal-versus-abnormal label transformation.
- Implemented binary training and evaluation for ECG-FM and InceptionTime.
- Implemented the two-architecture binary matrix evaluator.
- Implemented automatic five-class versus two-class gap-disappearance analysis.
- Preserved missing cells as blocked audit rows rather than fake scores.
- Verified the repository test suite after these additions.

## Execution state

The runnable source datasets are PTB-XL, CPSC2018, Georgia, and the exact
50,000-record MIMIC-IV-ECG subset. GPU execution artifacts belong under
`data/week2/runs` and `data/week3/runs`; those large generated artifacts are
ignored by Git. CODE-II cannot be executed until the teammate responsible for
it supplies both its canonical manifest and waveforms. Results are complete
only when the required non-smoke checkpoints and matrix-cell files exist; a
passing smoke test is not reported as a completed experiment.

One real, non-smoke InceptionTime source run is now complete on Georgia. It
early-stopped after epoch 26, selected epoch 16 by validation AUROC, and scored
0.983885 AUROC on 403 held-out Georgia records. The available-cell matrix also
scored that Georgia source checkpoint on 4,732 held-out MIMIC records at
0.858661 AUROC. These two values are preliminary because Apple MPS reported a
nondeterministic backward kernel during training; the final paper result should
repeat the source run on CUDA or CPU. The matrix contains all 50 planned rows:
2 complete and 48 explicitly blocked by missing checkpoints or target data.
