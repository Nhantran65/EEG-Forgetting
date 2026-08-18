# Manuscript Validation Report

## Overall assessment: Ready for technical review

The evidence, calculations, figures, table, and manuscript source are internally
consistent and ready for technical review. Tectonic compiled a four-page
US-Letter PDF with technical content on pages 1--3 and references only on page
4. The draft is not submission-ready until author metadata is supplied and
layout is rechecked with the official ICASSP 2027 author kit.

## Methodology review

- The manuscript answers the two stated questions: old-task channel–frequency
  drift and performance–explanation alignment.
- PED is computed at subject level from fixed fit/gate samples and subtracts
  same-checkpoint split noise. Negative values are not clipped.
- Old-task Sleep is excluded consistently because its fidelity gate failed.
- EWC/DER++ comparisons are paired on order, seed, and directed transition.
- Inferential reporting aggregates transitions within nine `(order, seed)`
  blocks before bootstrap/sign testing; the 63 transition cells and subject rows
  are not presented as independent samples.

## Calculation spot-checks

- EWC block-level PED reduction: `-0.1222`, bootstrap 95% CI
  `[-0.1481, -0.0955]`, 9/9 blocks, sign-test `p=0.00390625`.
- DER++ block-level PED reduction: `-0.0804`, bootstrap 95% CI
  `[-0.1032, -0.0581]`, 9/9 blocks, sign-test `p=0.00390625`.
- Direction-level means in Table 1 reproduce
  `results/xai/high_gamma_replacement_ped_v1/summary.json`.
- Spearman values are labeled descriptive because transition cells share
  subjects, seeds, and orders; no causal claim is made.

## Visualization review

- Figure 1 uses all three forward seeds rather than a selected run and keeps a
  common before/after color scale.
- Figure 2 uses one observation grain, explicit zero references, method colors,
  and old-task marker shapes. It avoids dual axes and encodes method/task without
  relying on color alone.
- Both PNG exports were inspected at final aspect ratio; labels, legends, and
  colorbars are readable with no clipping.

## Required caveats included in the manuscript

- One backbone and small BCI/High-Gamma test cohorts.
- High-Gamma may include movement/EMG confounding and is not pure MI.
- High-Gamma BA fidelity passes by only about `0.0010`.
- Sleep explanation drift is not reported after fidelity failure.
- Dataset replacement occurred after the PhysioNet attribution gate failed.
- PED is post-hoc attribution stability, not a biological engram or causal
  mechanism of forgetting.
- Replacement joint/offline explanation alignment was not run.

## Blocking items before submission

1. Replace author, affiliation, and contact placeholders.
2. Recompile with the official ICASSP 2027 author kit when available; current
   IEEEtran/Tectonic output already satisfies the page structure.
3. Run IEEE PDF eXpress/compliance checks when the conference portal provides
   them.
