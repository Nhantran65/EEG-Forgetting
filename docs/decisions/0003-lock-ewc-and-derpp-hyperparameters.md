# 0003 Lock EWC Strength and DER++ Replay Memory

Date: 2026-08-14

## Status

Accepted

## Context

The full EWC and DER++ matrices require one method-specific setting that must
remain fixed across task order and seed. A predeclared BCI IV-2a to Sleep-EDF
pilot compared four EWC strengths and four DER++ byte caps using validation
only. A candidate was eligible when its Sleep validation subject balanced
accuracy was no more than 0.02 below the zero-method baseline; among eligible
candidates the rule selected minimum BCI relative forgetting, breaking ties in
favor of lower strength or memory.

All eight immutable pilot results completed and passed digest verification.

## Decision

Use EWC strength `lambda=100000` and DER++ persistent replay cap `8 MiB` for
every main order and seed.

| Method | Candidate | Sleep validation BA | BCI relative forgetting | Persistent state |
|---|---:|---:|---:|---:|
| EWC control | 0 | 0.6765 | 0.2023 | — |
| EWC selected | 100000 | 0.6618 | 0.0096 | 12,883,200 bytes per retained Fisher/anchor state |
| DER++ control | 0 MiB | 0.6791 | 0.1915 | 0 bytes |
| DER++ selected | 8 MiB | 0.6774 | 0.0095 | 8,381,067 allocated bytes; 119 fixed slots |

EWC uses the offline formulation, retaining one diagonal Fisher and one
`theta*` anchor for every prior task. DER++ uses Algorithm R over the observed
training-draw stream, two independent replay draws, and `alpha=beta=0.5`.

## Alternatives Considered

1. EWC `lambda=10000`: preserved more Sleep performance but left BCI relative
   forgetting at 0.1309, well above the selected candidate.
2. DER++ 16 or 32 MiB: neither improved the selected stability-plasticity point;
   8 MiB had the lowest eligible forgetting and the smallest nonzero footprint.
3. Expand the grid after seeing that EWC selected its upper boundary: rejected
   because the selected candidate already nearly eliminated forgetting and
   remained inside the predeclared plasticity constraint; expanding only after
   observing results would weaken protocol integrity.

## Consequences

Positive:

- Both methods substantially reduce BCI forgetting on the selection transition.
- DER++ uses the smallest tested nonzero memory and EWC has explicit byte
  accounting that grows with the number of protected prior tasks.

Tradeoffs:

- EWC gives up 0.0147 absolute Sleep validation BA relative to its control.
- The settings were selected on one transition and may not be optimal for every
  other task direction; they cannot be retuned per order or seed.

## Follow-Up

- Run three orders by three seeds for each method at the locked settings.
- Report peak persistent method memory alongside final task performance.
- Keep pilot artifacts separate from test-based main results.
