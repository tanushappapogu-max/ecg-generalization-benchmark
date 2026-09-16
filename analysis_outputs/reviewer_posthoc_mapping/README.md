# Post-hoc label-mapping sensitivity analysis

This analysis rescored the saved five-class predictions for all four
architectures and all 25 source-to-target cells. No model was retrained.

## Variants

- `consensus_five_class`: submitted five-class endpoint.
- `exclusive_normal_five_class`: a record counts as normal only when none of
  the four benchmark abnormalities is positive.
- `abnormal_only_four_class`: removes the ambiguous normal/sinus-rhythm
  endpoint and averages AUROC across the four abnormalities.
- `merged_bbb_four_class`: combines LBBB and RBBB into one bundle-branch-block
  endpoint using an OR label and maximum predicted probability.

## Results

| Variant | Mean cross-dataset macro-AUROC | Mean gap | Rank changes |
|---|---:|---:|---:|
| Consensus five-class | 0.9024 | 0.0576 | 0 |
| Exclusive normal | 0.9020 | 0.0510 | 0 |
| Abnormal-only | 0.9316 | 0.0400 | 0 |
| Merged BBB | 0.8866 | 0.0661 | 0 |

The cross-dataset ranking was unchanged under every variant: ResNet1D,
InceptionTime, ECG-FM, then Transformer. The exclusive-normal rule changed
mean cross-dataset AUROC by only -0.0005. Removing the ambiguous normal class
increased it by 0.0292, while merging LBBB and RBBB reduced it by 0.0159.
Because the latter two variants contain four rather than five endpoints, their
absolute macro-AUROC values are sensitivity diagnostics rather than directly
comparable improvements or degradations.

## Limitation

The saved predictions cannot recover source-native diagnoses excluded before
training, such as incomplete RBBB or isolated prolonged PR. Testing those
specific additions would require rebuilding the manifests and retraining the
affected source models. Therefore, these results should be described as a
post-hoc endpoint-mapping sensitivity analysis, not a full retrained mapping
ablation.
