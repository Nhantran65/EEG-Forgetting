# Paper source and chart notes

## Controlling quantitative source

- `results/xai/high_gamma_replacement_ped_v1/summary.json`
  - SHA-256: `2005ef2c8b62207fed1215f896ce855a537ede128f39c12c2d1fe00b3027cc43`
  - 27 XAI results, 63 immediate transition cells, 108 digest-bound map artifacts.
- `paper/statistical_report.json`
  - Deterministic 20,000-replicate bootstrap seed: `20260818`.
  - Cluster unit: `(order, seed)`; transitions are averaged within nine blocks.

## Chart map

1. **Figure 1 — protocol and BCI reliance maps**
   - Question: can accuracy improve while channel–frequency reliance changes?
   - Form: process diagram plus three heatmaps.
   - Data: Sequential forward BCI stage 1/2, positive-normalized and averaged
     over seeds `3407/42/2026`; fit/gate maps averaged within checkpoint.
   - Claim: BCI backward transfer does not imply explanation retention.
   - Palette: single blue root for reliance; orange/white/blue diverging delta.

2. **Figure 2 — performance/PED and paired reductions**
   - Question: are performance forgetting and PED aligned, and do CL methods
     reduce PED relative to Sequential FT?
   - Form: scatter at transition-cell grain; box/strip paired deltas.
   - Data: all 63 cells; 21 matched cells for each EWC/DER++ comparison.
   - Claim: within-method association is weak, while both methods reduce PED in
     every matched comparison.
   - Palette: orange Sequential, blue EWC, olive DER++; task also encoded by
     marker shape so color is not the only distinction.

## Omitted or pending items

- Offline joint-reference alignment is omitted from the current manuscript
  because no replacement joint checkpoint matrix was run.
- Old-task Sleep PED is omitted because the Sleep fidelity gate failed.
- The current PDF passes four-page US-Letter layout under Tectonic 0.16.9 and
  bundled `IEEEtran.cls`; final compliance should be repeated with the official
  ICASSP 2027 author kit.
