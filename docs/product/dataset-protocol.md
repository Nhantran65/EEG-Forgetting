# Dataset Protocol

This document is the authority for dataset inclusion, preprocessing, labels,
splits, and training exposure in the ICASSP 2027 forgetting study. Machine
readable values live in `configs/`.

## Shared CBraMod contract

- Resample EEG to 200 Hz and represent it as one-second, 200-point patches.
- Apply the declared dataset reference, then a 0.5–40 Hz band-pass.
- Convert to microvolts and divide by 100. Do not fit dataset or test-set
  normalization statistics.
- Use batch size 64 and exactly 2,500 optimizer steps per task for
  budget-matched single-task and continual-learning runs.
- Run a separate converged, early-stopped single-task baseline for Week-1
  validation and task performance ceilings. Fisher signatures use the
  budget-matched checkpoints.

## BCI Competition IV-2a

Use 22 EEG channels and the half-open interval `[2.0, 6.0)` seconds from trial
onset. Drop EOG and expert-marked artifact trials. Merge T and E sessions only
within a subject; splits remain subject-disjoint. Evaluation-session labels must
come from `A0xE.mat:classlabel`, never the unknown event code in the GDF.

Before artifact exclusion, every session must contain 288 trials, four classes,
and 72 trials per class. EOG calibration blocks are detected by absent trial
labels so the short A04T calibration does not cause two MI runs to be discarded.
Every session must expose six MI runs.

The frozen main subject holdout is A01–A05 for training, A06–A07 for
validation, and A08–A09 for testing. A separate robustness split remains a
predeclared Week-2 task and must be frozen as a new manifest version before use.

## PhysioNet-MI

Use imagery runs 04/06/08/10/12/14, discard T0, and map T1/T2 by run type to
left fist, right fist, both fists, and both feet. The main montage is the 22
channels shared with BCI IV-2a.

Direct EDF audit excludes S088, S092, S100, and S104. IDs are never renumbered.
The main clean split contains 70/18/17 train/validation/test subjects. One
separate run uses the original unfiltered CBraMod 70/19/20 split solely to
validate the external preprocessing-to-checkpoint pipeline.

## Sleep-EDF Expanded

Use only the 78-subject, 153-recording Sleep Cassette cohort. Both nights from
one subject must remain in the same split. Use Fpz-Cz and Pz-Oz as existing
bipolar channels without CAR. Merge stages 3 and 4 into N3, drop movement and
unknown, and retain at most 30 minutes of wake before the first and after the
last sleep epoch.

The frozen split contains 48/15/15 train/validation/test subjects. It uses seed
`20260813` and strata formed by sex crossed with four age bands: 25–39, 40–64,
65–79, and 80+. The resulting recording counts are 94/30/29 because three
subjects have only one available night. Exact assignments and source checksums
are in `manifests/v1/`.

## TUEV

TUEV replaces VEP as the fourth main task if access and a runnable v2.0.0
snapshot exist by 2026-08-19. Reuse the deterministic CBraMod preprocessing at
commit `b9e961003214326972c567eff390e75b0287e32a`. Its model input is 16 TCP
bipolar derivations, five one-second patches, and 200 points per patch.

The pinned script emits microvolts and the pinned CBraMod dataset loader divides
them by 100. For the one-off reference reproduction, retain the upstream
0.3–75 Hz plus 60 Hz notch filter. For the main harmonized experiment, keep the
pinned montage, event extraction, and deterministic split code but change only
the continuous-raw filter parameters to 0.5–40 Hz before event windows are
created. Do not refilter isolated five-second windows because boundary effects
would become another dataset-specific confound.

## VEP quarantine

The current VEP snapshot has 31 usable subjects and 3,612 one-second windows,
but zero stimulus markers across 64 JSON files and zero annotations in all 63
available EDF files. Raw phase-crossover classification is at four-class chance.
It remains available through a pluggable adapter but cannot contribute to the
main RQ2, continual-learning matrix, or physiological claims.
