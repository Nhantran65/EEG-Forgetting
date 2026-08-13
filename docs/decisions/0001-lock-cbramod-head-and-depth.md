# 0001 Lock the CBraMod Task Head and Plasticity Depth

Date: 2026-08-13

## Status

Accepted

## Context

All continual-learning, Fisher and single-task runs need one task-head family
and one shared-backbone plasticity boundary. Selecting depth from forgetting
would leak the paper outcome into the protocol, so the predeclared criterion is
single-task validation only: choose the smallest of 1/2/4/8 final encoder
blocks whose mean chance-normalized subject balanced accuracy across the three
locked tasks is at least 95% of the best candidate.

The first mean-pool sweep was not representative because pooling removed
channel-patch structure. A second sweep was invalid because training mode
enabled dropout inside frozen blocks, making the nominally frozen upstream
representation stochastic. Both remain preserved as rejected pilot evidence.

## Decision

Use the CBraMod all-patch MLP head family and unfreeze the final four encoder
blocks. Patch embedding and earlier frozen blocks remain in eval mode; only the
classifier and four plastic blocks enter train mode.

Eligible depth-v3 results at seed 3407 were:

| Depth | BCI subject BA | PhysioNet subject BA | Sleep subject BA | Mean normalized BA |
|---:|---:|---:|---:|---:|
| 1 | 0.5092 | 0.4311 | 0.6587 | 0.3868 |
| 2 | 0.5178 | 0.4401 | 0.6653 | 0.3974 |
| 4 | 0.5559 | 0.4711 | 0.6784 | 0.4335 |
| 8 | 0.5639 | 0.4880 | 0.6848 | 0.4473 |

The 95% threshold is 0.4249. Depth 4 is the smallest eligible depth above it.
The all-patch frozen linear probes were 0.4902, 0.4319 and 0.5802 subject BA for
BCI, PhysioNet and Sleep respectively; depth 4 improves all three.

## Alternatives Considered

1. Depth 8: best mean validation result, but adds twice as many plastic blocks
   for only 0.0138 absolute normalized mean gain.
2. Depth 1 or 2: cheaper, but neither reaches the predeclared 95% threshold.
3. Mean-pool linear head: rejected because it discards the spatial/temporal
   patch layout and remained near chance on motor imagery.

## Consequences

Positive:

- The main depth is chosen without observing forgetting.
- Fine-tuning beats the fair frozen linear probe on every locked task.
- Four plastic blocks leave a substantial frozen representation for controlled
  interference analysis.

Tradeoffs:

- Task-specific all-patch heads have different parameter counts because input
  channel/patch shapes differ; memory reporting must separate heads from the
  shared backbone.
- BCI peaks around step 300 while Sleep peaks around step 2100 in this sweep,
  reinforcing the need for separate converged and exact-2500-step baselines.

## Follow-Up

- Run converged and budget-matched single-task baselines at depth 4.
- Run the one-off PhysioNet 70/19/20 external reproduction.
- Use exact step 2500, not the best pilot checkpoint, for Fisher and CL budget
  matching.
