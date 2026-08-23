f

# 0002 Accept the Sequential Fine-Tuning Stability Gate

Date: 2026-08-14

## Status

Accepted

## Context

Scaling EWC and DER++ was conditional on sequential fine-tuning producing a
stable, measurable forgetting signal. The predeclared matrix contains three
task orders and three seeds at the locked final-four-block depth. Its gate
requires at least three valid replicates in all six directed transitions, at
least four directions with absolute mean relative forgetting larger than their
sample standard deviation, and at least two thirds of replicates sharing the
same sign.

## Decision

Accept the sequential fine-tuning gate and proceed to validation-selected EWC
and DER++ pilots before scaling their full matrices.

All nine immutable runs completed. All six directed transitions had enough
replicates, passed the signal rule, and were sign-consistent. Mean relative
forgetting by unordered pair was:

| Pair                     | Fisher cosine | Mean relative forgetting | Sample SD | N |
| ------------------------ | ------------: | -----------------------: | --------: | -: |
| BCI IV-2a / PhysioNet-MI |        0.8277 |                   0.0193 |    0.2079 | 9 |
| BCI IV-2a / Sleep-EDF    |        0.5672 |                   0.3724 |    0.2800 | 9 |
| PhysioNet-MI / Sleep-EDF |        0.5871 |                   0.3262 |    0.1009 | 9 |

The only consistently negative direction was BCI IV-2a after PhysioNet-MI
(`mean F_rel=-0.2351`, three of three negative), indicating backward transfer
rather than forgetting. The other five directions were consistently positive.

The three unordered pairs have an exploratory Spearman correlation of `-1.0`
between Fisher cosine and mean relative forgetting. This is not confirmatory
evidence because there are only three independent pair-level points.

## Alternatives Considered

1. Stop after the low unordered BCI/PhysioNet mean: rejected because averaging
   its two directions hides stable backward transfer in one direction and
   stable forgetting in the other.
2. Treat the three-pair correlation as the paper result: rejected because its
   effective sample size is three and its sign opposes the motivating positive
   predictor hypothesis.
3. Add or substitute tasks after observing the result: rejected because that
   would make the benchmark outcome-dependent. TUEV remains governed only by
   its predeclared access deadline.

## Consequences

Positive:

- The continual-learning signal is large and stable enough to evaluate EWC,
  DER++, and the predeclared causal intervention.
- Directional reporting exposes backward transfer that an unordered average
  would conceal.

Tradeoffs:

- The current three-task benchmark does not support a positive association
  between overlap and forgetting; the final claim must remain null or cautious
  unless predeclared additional evidence changes it.
- Pair-level inference remains underpowered even though subject-level metrics
  and seed/order replicates are numerous.

## Follow-Up

- Select one EWC strength and one DER++ byte cap using equal validation budgets
  on one predeclared high-forgetting transition, then freeze them across all
  orders and seeds.
- Run the full EWC and DER++ matrices and BCI split robustness.
- Preserve all nine sequential result hashes and generate downstream tables
  only from the immutable summary.
