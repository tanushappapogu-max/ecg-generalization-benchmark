# Overleaf submission package

Upload the contents of this directory to a new Overleaf project and set
`main.tex` as the main document. Compile with pdfLaTeX.

The source uses the NeurIPS 2026 double-blind workshop option and identifies
the workshop as World Models for High-Stakes Health (WMHS). Author names are
intentionally omitted.

## Intentional visual set

1. `fig1_methods_pipeline.png` explains the complete study design once, so the
   Methods section does not need multiple small diagrams.
2. `fig2_four_model_heatmaps.pdf` is the core evidence: all four architectures
   and all 100 directional source-to-target evaluations share one color scale.
3. `fig3_generalization_summary.pdf` makes the two primary findings easy to
   read: in-distribution rank does not determine external rank, and MIMIC-IV-ECG
   is the hardest external target.
4. `fig4_revision_sensitivity.pdf` directly answers the reviewer revisions:
   the four-model normal-vs-abnormal result and the endpoint-mapping analysis.

All figures use the navy, slate, mauve, purple, and dusty-rose palette extracted
from the existing methods pipeline. Distinct markers preserve readability when
printed without color.

## Required checks before submission

- Resolve every `% TODO` comment in `main.tex`.
- Add complete references for Berger et al. (2026) and Zhou et al. (2026).
- Recompute the composite score from the audited ResNet1D matrix before adding
  any numerical composite ranking back to the paper.
- Confirm the paper remains within the selected WMHS page limit after Overleaf
  compiles it.
- Confirm that the PDF contains no author names, affiliations, acknowledgments,
  identifying repository links, or identifying PDF metadata.
- Keep the responsible-use statement; WMHS lists it as required.

`checklist.tex` is included from the supplied NeurIPS template but is not
currently imported by `main.tex`, because the WMHS call does not explicitly
request the main-conference checklist.
