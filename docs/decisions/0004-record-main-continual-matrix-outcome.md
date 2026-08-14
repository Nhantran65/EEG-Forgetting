# 0004 Record the Main Continual-Learning Matrix Outcome

Date: 2026-08-14

## Status

Accepted

## Context

Sequential fine-tuning, EWC, and DER++ were each run for three predeclared task
orders and three seeds at the locked final-four-block depth. All 27 immutable
runs completed. The EWC and DER++ summaries bind all 18 new result, stage,
checkpoint, and prediction artifacts by digest and contain no execution error.

The predeclared stability gate was designed to decide whether sequential
fine-tuning exposed enough forgetting to justify scaling continual-learning
methods. Its signal-count field remains useful descriptively for EWC and DER++,
but it is not a method-success criterion: a successful anti-forgetting method
can intentionally push directions below seed noise.

## Decision

Accept the EWC and DER++ matrices as the main method results. Record pair-level
mean relative forgetting as follows:

| Unordered task pair | Sequential FT | EWC | DER++ |
|---|---:|---:|---:|
| BCI IV-2a / PhysioNet-MI | 0.0193 | -0.0776 | -0.0682 |
| BCI IV-2a / Sleep-EDF | 0.3724 | 0.0726 | 0.0065 |
| PhysioNet-MI / Sleep-EDF | 0.3262 | 0.0084 | 0.0662 |

EWC leaves only two of six directed transitions with mean signal larger than
seed spread; this makes the inherited signal gate report `false`, but reflects
forgetting suppression rather than a failed run. DER++ retains four signal
directions and passes that descriptive gate. Both methods have enough valid
replicates and six of six sign-consistent directions under the configured
summary rule.

Peak persistent method state is 25,766,400 bytes for offline EWC (two retained
Fisher/anchor states) and 8,381,067 allocated bytes for DER++ (119 fixed replay
slots under the 8 MiB cap).

The exploratory pair-level overlap association is negative for every completed
method: Spearman `rho=-1.0` for sequential FT and EWC, and `rho=-0.5` for DER++.
There are only three independent unordered task pairs, so neither the nominal
p-value nor the number of subject observations turns this into confirmatory
inference.

## Alternatives Considered

1. Treat EWC's false signal gate as an invalid method run: rejected because all
   artifacts and replicates are valid and the gate fails by reducing effect
   magnitude, which is EWC's intended outcome.
2. Claim that Fisher overlap positively predicts forgetting: rejected because
   the observed pair ordering is negative in all three methods.
3. Retune EWC or DER++ after the test matrix: rejected because their settings
   were validation-selected and locked before these runs.

## Consequences

Positive:

- The benchmark now has matched 9-run matrices for the unprotected baseline and
  two standard continual-learning methods.
- EWC and DER++ strongly reduce the large Sleep-related forgetting observed
  under sequential fine-tuning, with explicit persistent-memory accounting.

Tradeoffs:

- The current three-task result does not support the motivating positive
  overlap-risk hypothesis. The paper must present this as a bounded negative or
  inverse descriptive result unless predeclared additional evidence changes it.
- Three task pairs remain too few for a strong general association claim.

## Follow-Up

- Run the BCI robustness split and joint-training upper bound.
- Continue the predeclared matched-plasticity causal intervention; it can still
  establish whether high-overlap parameters are mechanistically important even
  when pair-level ranking is not positively predictive.
- Generate the method table from immutable summaries, reporting both forgetting
  and persistent memory.
