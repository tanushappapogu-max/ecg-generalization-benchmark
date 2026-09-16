# Label-mapping sensitivity analysis

The reviewer-requested primary set contains three prespecified mapping
specifications: the submitted consensus mapping, an explicit-normal-ECG
mapping, and a broader conduction mapping. `mapping_variants.csv` records the
only decisions that change between specifications.

Do not manufacture the alternative labels from the five canonical binary
columns. The strict-normal specification requires the source-native normal ECG
field or a validated report-text rule. The broad-conduction specification
requires source-native prolonged-PR and incomplete-bundle-branch codes. Create
one canonical `<dataset>_week2.csv` per dataset under each variant directory,
while preserving every baseline record ID and split.

Once those manifests and the saved five-label predictions are available, run
`python -m src.evaluation.label_mapping_sensitivity`. The output is explicitly
identified as target-label rescoring without retraining. If this diagnostic
changes the headline model ranking, the definitive follow-up is to retrain the
source models under the affected mapping.
